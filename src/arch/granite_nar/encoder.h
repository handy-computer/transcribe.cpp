// arch/granite_nar/encoder.h - NLE Conformer encoder + BPE CTC head.
//
// Reference: NLENARConformerEncoder in the IBM Granite Speech NLE HF
// repo's modeling_nle.py. Structurally identical to the AR granite
// encoder (block-local Shaw self-attention, conv_expansion=2 GLU, mid-
// layer self-conditioned CTC bypass) plus two NLE-only additions:
//
//   1. A BPE CTC head (1024 → 100353) over a posterior-weighted
//      window=4 pool of valid frames. We expose the raw frame-level
//      logits (1024→348) as `enc.ctc_logits` for validate.py — the
//      pooled BPE head is computed host-side at run time.
//   2. All-hidden-states capture: the projector consumes 4 encoder
//      hidden states (post-LN, pre-bypass at the chosen layer
//      boundaries; indices [4, 8, 12, -1] 1-indexed → layer outputs
//      after blocks 3, 7, 11, 15 in 0-indexed). The encoder graph
//      emits these concatenated along the channel axis so the
//      projector consumes one [4096, T_enc] tensor.
//
// This header is INTERNAL to src/arch/granite_nar/.

#pragma once

#include "weights.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe { class MelFrontend; }

namespace transcribe::granite_nar {

// Host-side mel + 2-frame stack. Reuses the AR granite implementation.
transcribe_status compute_mel_encoder_input(
    const transcribe::MelFrontend & mel,
    const float *                   pcm,
    int                             n_samples,
    int                             n_threads,
    std::vector<float> &            out_mel,
    int &                           out_t_enc);

struct EncoderBuild {
    // Graph inputs (caller uploads at compute time).
    ggml_tensor * mel_in           = nullptr;  // [input_dim, T_enc]
    ggml_tensor * attention_dists  = nullptr;  // [ctx*ctx] i32
    ggml_tensor * last_block_mask  = nullptr;  // [ctx, ctx, n_blocks_local]
    ggml_tensor * zero_pad         = nullptr;  // optional [hidden, T_pad-T_enc]

    // Outputs.
    ggml_tensor * cat_out          = nullptr;  // [num_enc_layers * hidden, T_enc]
                                               //  the projector input
    ggml_tensor * ctc_logits       = nullptr;  // [enc_out_dim=348, T_enc]
    ggml_tensor * ctc_bpe_logits   = nullptr;  // [bpe_out_dim=100353, T_enc]
                                               //  raw frame-level (no pool)
    ggml_tensor * mid_blank_probs  = nullptr;  // [T_enc] — softmax(mid_ctc)[blank].
                                               //  Used host-side as the BPE pool's
                                               //  importance weight (importance =
                                               //  1 - blank_prob_mid).

    ggml_cgraph * graph            = nullptr;

    struct Dumps {
        ggml_tensor * input_linear_out = nullptr;
        ggml_tensor * block_0_out      = nullptr;
        ggml_tensor * block_mid_pre    = nullptr;  // block (N/2 - 1) post-LN
                                                   // (== `enc.block.7.out`)
        ggml_tensor * block_mid_post   = nullptr;  // block (N/2)   post-LN
                                                   // (== `enc.block.8.out`)
        ggml_tensor * block_last_out   = nullptr;  // block (N-1)
        ggml_tensor * ctc_logits       = nullptr;  // == ctc_logits, named
    } dumps;

    int n_blocks_local = 0;
    int last_block_rem = 0;
};

EncoderBuild build_encoder_graph(ggml_context *            ctx,
                                 const GraniteNarWeights & weights,
                                 const GraniteNarHParams & hp,
                                 int                       T_enc,
                                 bool                      use_flash);

// Shaw bookkeeping helpers (identical to AR granite).
std::vector<int32_t> precompute_attention_dists(int context_size, int max_pos_emb);
std::vector<float>   precompute_last_block_mask(int context_size, int t_enc_remainder);

// Host-side BPE CTC pool + greedy decode.
//
// Input:
//   importance_non_blank  [T_enc] host row of (1 - blank_prob) values used
//                         as per-frame pool weights. Must come from the
//                         *middle* (self-conditioning) CTC head's softmax,
//                         not the final head — see modeling_ctc.py.
//   ctc_bpe      [T_enc, 100353] host-row-major (frame-level BPE logits)
//   pool_window  4 in this family
//   blank_id     0
// Output:
//   token_ids    initial-hypothesis BPE token ids after greedy + collapse +
//                blank removal. Used as the text portion of the NLE LLM
//                forward (each token gets an eos slot inserted around it).
//
// Reference: NLE NARDecoder.compute_text_ctc_preds: argmax over the
// CTC head per frame, only frames where the argmax is non-blank are
// "valid" — those frames are then bucketed into windows of pool_window=4
// and the BPE logits over each window are weighted by the (softmaxed)
// CTC blank-vs-non-blank posterior. We approximate the same path:
//
//   per-frame: blank_prob = softmax(ctc_logits)[blank_id]
//              non_blank_prob = 1 - blank_prob
//   pool: for each window of pool_window consecutive frames whose
//         non_blank_prob > 0.5, sum non_blank_prob[k] * ctc_bpe[k]
//         normalised by sum non_blank_prob[k] -> [bpe_dim]
//   greedy: argmax per pooled window, then collapse repeats + drop blanks
void compute_bpe_ctc_initial_hypothesis(
    const std::vector<float> & importance_non_blank,
    const std::vector<float> & ctc_bpe_logits,
    int                        n_bpe_vocab,
    int                        T_enc,
    int                        pool_window,
    int                        blank_id,
    std::vector<int32_t> &     out_token_ids);

} // namespace transcribe::granite_nar
