// arch/granite5_ctc/encoder.h - Granite Speech 5.0 TurboCTC encoder.
//
// Reference: GraniteSpeech5Encoder / GraniteSpeech5EncoderBlock /
// GraniteSpeech5EncoderSubsamplingBlock in
// transformers/models/granite_speech5/modeling_granite_speech5.py, and
// GraniteSpeech5FeatureExtractor._extract_features in
// feature_extraction_granite_speech5.py.
//
// Structurally the granite 4.x Conformer (macaron FFN + block-local Shaw
// attention + GLU conv module with BatchNorm + post-block LN + a
// self-conditioned CTC injection at the halfway block) with two
// deviations that drive everything in this file:
//
//   1. In-block time subsampling. The blocks listed in
//      `subsample_layers` ([0, 1] here) run their depthwise conv at
//      stride 2 and mean-pool the residual over frame pairs, so the
//      sequence length falls 4x across the first two blocks:
//      50 Hz -> 25 Hz -> 12.5 Hz. Attention inside a subsampling block
//      still runs at the INPUT rate, so the block-local attention
//      geometry (number of blocks, pad tile, pad mask) differs between
//      stages and is built once per distinct length.
//
//   2. A richer frontend. (80 log-mel + 80 delta) x 2 stacked frames =
//      320 features, and the trailing stacking group is FILLED by
//      right-padding the waveform rather than dropped. granite 4.x
//      drops it.

#pragma once

#include "transcribe.h"
#include "weights.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe {
class MelFrontend;
}

namespace transcribe::granite5_ctc {

// Host-side frontend.

// Number of stacked encoder frames a clip of `n_samples` produces.
// Mirrors GraniteSpeech5FeatureExtractor:
//     mel_frames = n_samples / hop
//     num_frames = stack * ceil(mel_frames / stack)
//     T_enc      = num_frames / stack
// Returns 0 when the clip is too short to produce a frame.
int encoder_frames_for(const Granite5CtcHParams & hp, int64_t n_samples);

// Run the full reference frontend: torchaudio-equivalent log-mel with
// the per-utterance floor, first-order deltas, and 2-frame stacking.
//
// Output `out_feats` is [T_enc, input_dim] row-major, where each row is
//     [logmel(2t) || delta(2t) || logmel(2t+1) || delta(2t+1)]
// matching the reference's channel-concat-then-reshape order.
transcribe_status compute_encoder_input(const transcribe::MelFrontend & mel,
                                        const Granite5CtcHParams &      hp,
                                        const float *                   pcm,
                                        int64_t                         n_samples,
                                        int                             n_threads,
                                        std::vector<float> &            out_feats,
                                        int &                           out_t_enc);

// Shaw lookup rows, one per relative offset, shared by every block.
//
// Reference `GraniteSpeech5Encoder.compute_attention_dists` builds a full
// [query, key] table:
//     seq         = arange(context_size)
//     relpos_dist = seq.view(-1, 1) - seq.view(1, -1)      # [query, key]
//     dists       = clamp(relpos_dist, +/- context_size) + max_position_embeddings
//
// so the value at (ne0 = key, ne1 = query) is `query - key`. This is the
// SAME convention granite 4.x uses: `granite::precompute_pos_rows` writes
// `dists[c * ctx + r] = c - r` with c the outer (query) index and r the
// inner (key) index, which is the identical array. An earlier note here
// claimed the signs were opposite; they are not, and the two families now
// share one `precompute_pos_rows` formula.
//
// Every entry of that table is a function of (query - key) alone, so only
// the 2*context_size - 1 distinct offsets are needed. shaw_block_attn's
// skew path indexes this vector by `d = key - query + context_size - 1`
// (the layout conformer::rel_shift rotates), which is why the stored
// offset is `context_size - 1 - d`. Collapsing the table this way is what
// lets the positional bias be one fat GEMM instead of a ctx*ctx lookup;
// see shaw_attn.h.
std::vector<int32_t> precompute_pos_rows(int context_size, int max_pos_emb);

// Additive block-local pad mask for one attention stage.
//
// Only pad KEY columns are masked; pad QUERY rows are left alone. Masking
// a whole query row would make softmax over it produce garbage that the
// depthwise conv could smear into valid frames — instead pad-query
// outputs stay bounded and are sliced off before `o_proj`.
//
// The masked value is a large FINITE negative, not -INF, mirroring the
// reference's `masked_fill(..., torch.finfo(dtype).min)`. It matters only
// in the batched path, where a short utterance can leave an entire block
// with no real keys: -INF there gives softmax(all -INF) = NaN, which then
// survives the multiplicative conv mask (0 * NaN = NaN) and poisons real
// frames. A finite floor degrades to a uniform row instead. Wherever -INF
// was well defined the two agree exactly, since exp(-1e30 - max)
// underflows to 0 in f32.
//
// `real_lens` holds each utterance's valid frame count at this stage;
// pass a single entry for the single-utterance path. Shape is
// [context_size, context_size, n_blocks * B] in ggml ne order (ne[0] =
// key), with the block index varying fastest — the layout
// shaw_block_attn folds the utterance batch into.
std::vector<float> precompute_pad_mask(int context_size, int t_len, const std::vector<int> & real_lens);

// Per-frame validity mask for one stage: [1, t_len, B] f32, 1.0 on real
// frames and 0.0 on padding. Mirrors the reference's
// `masked_fill(~attention_mask.unsqueeze(-1), 0.0)` after input_linear
// and after the conv module's GLU.
std::vector<float> precompute_frame_mask(int t_len, const std::vector<int> & real_lens);

// Graph.

// One attention geometry. There is one of these per distinct sequence
// length in the forward (3 on this variant: T0, T0/2, T0/4).
struct AttnStage {
    int           t_len      = 0;        // sequence length entering attention
    int           n_blocks   = 0;        // ceil(t_len / context_size)
    ggml_tensor * pad_mask   = nullptr;  // [ctx, ctx, n_blocks * B] f32, graph input
    ggml_tensor * zero_pad   = nullptr;  // [hidden, t_pad - t_len, B] f32 zeros, or null
    // Only built when the batch has uneven lengths: [1, t_len, B] f32.
    ggml_tensor * frame_mask = nullptr;
};

struct EncoderBuild {
    // Graph inputs (uploaded by the caller after allocation).
    ggml_tensor * feats_in = nullptr;    // [input_dim, T_enc]
    ggml_tensor * pos_rows = nullptr;    // [2*ctx-1] i32, shared

    std::vector<AttnStage> stages;       // one per distinct sequence length
    std::vector<int32_t>   block_stage;  // block index -> stages[] index

    // Graph outputs.
    ggml_tensor * out        = nullptr;  // [hidden, T_out]   ("enc.out")
    ggml_tensor * ctc_logits = nullptr;  // [vocab,  T_out]   ("enc.ctc_logits")

    ggml_cgraph * graph = nullptr;

    // Final sequence length after every subsampling block (the padded
    // length; per-utterance valid counts come from real_lens_out).
    int t_out = 0;

    // Utterance batch size (1 for single-shot) and, for each utterance,
    // the number of VALID frames left in the encoder output.
    int              n_batch = 1;
    std::vector<int> real_lens_out;

    // Validation dump points, in build order. Explicit (name, tensor)
    // pairs rather than a name scan over graph nodes: ggml auto-names
    // views as "<parent> (view)", which a prefix scan would pick up as
    // phantom tensors, and the last block's output legitimately carries
    // two names (`enc.block.N.out` and `enc.out`) that a per-tensor name
    // cannot both hold.
    std::vector<std::pair<std::string, ggml_tensor *>> dump_list;
};

// Build the encoder + tied CTC head graph. `T_enc` is the stacked
// frontend frame count (`encoder_frames_for`).
// `real_lens` gives each utterance's valid stacked-frame count; its size
// is the batch. Every entry equal to T_enc (the common case, and always
// true for a single utterance) builds the mask-free graph, which is
// bit-identical to the single-shot one. Any shorter entry turns on the
// per-utterance masking the reference applies for a padded batch.
EncoderBuild build_encoder_graph(ggml_context *             ctx,
                                 const Granite5CtcWeights & weights,
                                 const Granite5CtcHParams & hp,
                                 int                        T_enc,
                                 const std::vector<int> &   real_lens = {});

}  // namespace transcribe::granite5_ctc
