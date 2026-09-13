// arch/granite_nar/encoder.h - NLE Conformer encoder + BPE CTC head.
//
// Reference: GraniteSpeechNarCTCEncoder in modeling_granite_speech_nar.py
// from the IBM Granite Speech NAR HF repo. Structurally identical to the
// AR granite encoder (block-local Shaw self-attention, conv_expansion=2
// GLU, mid-layer self-conditioned CTC bypass) plus two NAR-only additions:
//
//   1. A BPE CTC head (1024 -> 100352) over a posterior-weighted
//      window=4 pool of valid frames. We expose the bypass-step char-CTC
//      mid_logits (1024 -> 348) as `enc.ctc_logits` -- the exact tensor
//      the reference model computes at self_conditioning_layer for the
//      self-conditioning residual. The very wide BPE head runs in bounded
//      chunks after the encoder so frame-level vocabulary logits are never
//      retained for the full utterance.
//   2. All-hidden-states capture: the projector consumes 4 encoder
//      hidden states (post-LN, pre-bypass at the chosen layer
//      boundaries; indices [4, 8, 12, -1] 1-indexed → layer outputs
//      after blocks 3, 7, 11, 15 in 0-indexed). The encoder graph
//      emits these concatenated along the channel axis so the
//      projector consumes one [4096, T_enc] tensor.
//
// This header is INTERNAL to src/arch/granite_nar/.

#pragma once

#include "ggml.h"
#include "weights.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe {
class MelFrontend;
}

namespace transcribe::granite_nar {

// Host-side mel + 2-frame stack. Reuses the AR granite implementation.
transcribe_status compute_mel_encoder_input(const transcribe::MelFrontend & mel,
                                            const float *                   pcm,
                                            int                             n_samples,
                                            int                             n_threads,
                                            std::vector<float> &            out_mel,
                                            int &                           out_t_enc);

struct EncoderBuild {
    // Graph inputs (caller uploads at compute time).
    ggml_tensor * mel_in          = nullptr;  // [input_dim, T_enc]
    ggml_tensor * pos_rows        = nullptr;  // [2*ctx-1] i32
    ggml_tensor * last_block_mask = nullptr;  // [ctx, ctx, n_blocks_local]
    ggml_tensor * zero_pad        = nullptr;  // optional [hidden, T_pad-T_enc]

    // Outputs.
    ggml_tensor * cat_out         = nullptr;  // [num_enc_layers * hidden, T_enc]
                                              //  the projector input
    ggml_tensor * ctc_logits      = nullptr;  // [enc_out_dim=348, T_enc]
    ggml_tensor * mid_blank_probs = nullptr;  // [T_enc] -- softmax(mid_ctc)[blank].
                                              //  Used host-side as the BPE pool's
                                              //  importance weight (importance =
                                              //  1 - blank_prob_mid).

    ggml_cgraph * graph = nullptr;

    struct Dumps {
        ggml_tensor * input_linear_out  = nullptr;
        ggml_tensor * block_0_out       = nullptr;
        ggml_tensor * block_mid_pre     = nullptr;  // block (N/2 - 1) post-LN
                                                    // (== `enc.block.7.out`)
        ggml_tensor * block_mid_post    = nullptr;  // block (N/2)   post-LN
                                                    // (== `enc.block.8.out`)
        ggml_tensor * block_last_out    = nullptr;  // block (N-1)
        ggml_tensor * ctc_logits        = nullptr;  // == ctc_logits, named
        // Block-0 sub-step taps.
        ggml_tensor * block_0_post_ff1  = nullptr;
        ggml_tensor * block_0_post_attn = nullptr;
        ggml_tensor * block_0_post_conv = nullptr;
        ggml_tensor * block_0_post_ff2  = nullptr;
    } dumps;

    int     n_blocks_local       = 0;
    int     last_block_rem       = 0;
    int64_t final_capture_offset = -1;  // channel offset in cat_out
};

EncoderBuild build_encoder_graph(ggml_context *            ctx,
                                 const GraniteNarWeights & weights,
                                 const GraniteNarHParams & hp,
                                 int                       T_enc,
                                 bool                      use_flash);

// Shaw bookkeeping helpers (identical to AR granite).
std::vector<int32_t> precompute_pos_rows(int context_size, int max_pos_emb);
std::vector<float>   precompute_last_block_mask(int context_size, int t_enc_remainder);

// Bounded BPE-CTC projection. The caller supplies posterior-weighted encoder
// states, one per pooling window. Linearity makes projecting a weighted hidden
// state equivalent to weighting the projected frame logits. The graph returns
// one vocabulary argmax per window, avoiding a full-utterance [vocab, T_enc]
// tensor.
struct BpeCtcBuild {
    ggml_tensor * hidden_in = nullptr;  // [enc_hidden, n_windows]
    ggml_tensor * token_ids = nullptr;  // [n_windows] i32
    ggml_cgraph * graph     = nullptr;

    int n_windows = 0;
};

BpeCtcBuild build_bpe_ctc_graph(ggml_context *            ctx,
                                const GraniteNarWeights & weights,
                                const GraniteNarHParams & hp,
                                int                       n_windows);

}  // namespace transcribe::granite_nar
