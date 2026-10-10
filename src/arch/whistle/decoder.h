// arch/whistle/decoder.h - Whistle decoder graphs: cross-K/V, one decode
// step over n_seq hypotheses, and beam reordering.

#pragma once

#include "ggml.h"
#include "weights.h"

#include <vector>

namespace transcribe::whistle {

struct WhistleAux;
struct WhistleDecCache;

// Cross-attention K/V for one utterance, written into cache slot `utt`.
struct CrossKvBuild {
    ggml_cgraph * graph  = nullptr;
    ggml_tensor * mem_in = nullptr;  // [d, T_utt]
};

CrossKvBuild build_cross_kv_graph(ggml_context *          ctx,
                                  const WhistleWeights &  w,
                                  const WhistleHParams &  hp,
                                  const WhistleDecCache & cache,
                                  int                     utt,
                                  int                     T_utt);

// One decode step: every hypothesis feeds one token at position `pos`
// (all hypotheses are always at the same position).
struct StepBuild {
    ggml_cgraph *              graph      = nullptr;
    ggml_tensor *              tok_in     = nullptr;  // [n_seq] I32
    ggml_tensor *              pos_in     = nullptr;  // [n_seq] I32, all == pos
    ggml_tensor *              eg_idx_in  = nullptr;  // [n_tables * n_seq] I32 row ids into the Engram tables
    // Paired tables (WhistleEngram::paired): eg_idx_in holds row / 2 and this
    // holds row % 2, ne [1, n_tables * n_seq] I32; null otherwise.
    ggml_tensor *              eg_lo_in   = nullptr;
    ggml_tensor *              eg_mask_in = nullptr;  // [1, n_tables, 1] F32 n-gram validity
    ggml_tensor *              xmask_in   = nullptr;  // [T_enc, 1, 1, n_seq] F32 cross padding mask (n_utt > 1 only)
    ggml_tensor *              logits     = nullptr;  // [vocab, n_seq]
    // Cross-attention probabilities per layer, [T_enc, 1, n_heads, n_seq]
    // (only when want_cross_attn).
    std::vector<ggml_tensor *> cross_attn;
};

StepBuild build_step_graph(ggml_context *          ctx,
                           const WhistleWeights &  w,
                           const WhistleHParams &  hp,
                           const WhistleAux &      aux,
                           const WhistleDecCache & cache,
                           int                     pos,
                           bool                    want_cross_attn);

// Reorder every per-hypothesis cache tensor: new slot s <- old slot src[s].
// Only the filled self-attention positions [0, n_pos) are moved.
struct ReorderBuild {
    ggml_cgraph * graph  = nullptr;
    ggml_tensor * src_in = nullptr;  // [n_seq] I32
    ggml_tensor * src_k  = nullptr;  // [n_seq, n_kv] I32, src replicated per KV head
    ggml_tensor * src_v  = nullptr;  // [n_seq, vd * n_kv] I32, src replicated per V row
};

ReorderBuild build_reorder_graph(ggml_context * ctx, const WhistleDecCache & cache, int n_pos);

// Host-side Engram hashing (architecture.py::engram_indices) for the token
// at position `pos` of `tokens` (tokens[0..pos]). Writes n_tables row ids
// (table * slots + slot) into out.
void engram_rows(const std::vector<int> & tokens, int pos, const WhistleHParams & hp, int32_t * out);

}  // namespace transcribe::whistle
