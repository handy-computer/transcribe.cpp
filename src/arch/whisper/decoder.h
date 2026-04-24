// arch/whisper/decoder.h - Whisper decoder graph builder.
//
// The decoder is an autoregressive Transformer with:
//   - Token embedding + learned positional embedding (NO post-embed LayerNorm)
//   - N decoder layers (pre-LN self-attn + pre-LN cross-attn + pre-LN FFN)
//   - Final LayerNorm
//   - Linear head (tied to token embedding, NO bias) + log-softmax
//
// Two graph families are exposed:
//
//   1. build_decoder_prefill_graph() — non-cached prefill. Processes
//      the full prompt sequence at once with a causal self-attn mask;
//      cross-attention K/V are recomputed from the encoder output on
//      every call. Retained as a correctness oracle; reachable with
//      TRANSCRIBE_WHISPER_NO_KV=1.
//
//   2. build_cross_kv_graph() + build_decoder_graph_kv() — KV-cached
//      autoregressive path. Cross-K/V are precomputed once per
//      utterance into the cross cache; self-K/V are written on each
//      step into the self cache. Same dec.* dump points as (1) so
//      validate.py tolerances apply unchanged when the cached path is
//      exercised for the prompt pass.
//
// Whisper attention quirk (matches encoder.cpp): q/v/out projections
// carry bias; k_proj has NO bias on both self- and cross-attention.

#pragma once

#include "ggml.h"

#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::whisper {

struct WhisperHParams;
struct WhisperWeights;
struct WhisperKvCache;

struct DecoderDumps {
    ggml_tensor * token_emb       = nullptr;
    ggml_tensor * pos_emb         = nullptr;
    ggml_tensor * embed_sum       = nullptr;
    std::vector<ggml_tensor *> block_outs;       // per-layer residual stream
    ggml_tensor * out_before_head = nullptr;     // post final LN
    ggml_tensor * logits_raw      = nullptr;     // pre log-softmax logits
    ggml_tensor * logits          = nullptr;     // log-softmax(logits_raw)
};

struct DecoderBuild {
    // Inputs (caller fills via ggml_backend_tensor_set after alloc).
    ggml_tensor * token_ids_in   = nullptr;  // [seq_len] i32
    ggml_tensor * encoder_out_in = nullptr;  // [d_model, T_enc] f32 (non-cached + cross_kv graph)
    ggml_tensor * causal_mask_in = nullptr;  // [n_kv, seq_len] f32 (cast to f16 inside)

    // Output: log-softmax logits [vocab_size, seq_len].
    ggml_tensor * out = nullptr;

    DecoderDumps  dumps {};
    ggml_cgraph * graph = nullptr;
};

// Build a decoder prefill graph in compute_ctx (no KV cache).
DecoderBuild build_decoder_prefill_graph(ggml_context *         compute_ctx,
                                         const WhisperWeights & weights,
                                         const WhisperHParams & hp,
                                         int                    seq_len,
                                         int                    T_enc,
                                         bool                   use_flash = true);

// Build a graph that computes cross-attention K/V for every decoder
// layer from the encoder output and writes them into the cross-attn
// KV cache. Run once per utterance.
//
// The returned DecoderBuild has only encoder_out_in and graph set.
// Caller uploads encoder output and calls ggml_backend_sched_graph_compute.
DecoderBuild build_cross_kv_graph(ggml_context *         compute_ctx,
                                  const WhisperWeights & weights,
                                  const WhisperHParams & hp,
                                  WhisperKvCache &       kv_cache,
                                  int                    T_enc);

// Build a KV-cached decoder graph. Works for both prompt pass
// (n_tokens > 1, n_past = 0) and step pass (n_tokens = 1, n_past =
// prev total). Self-K/V are written into the self cache at position
// n_past; the full [0..n_past+n_tokens) window is read back for
// attention. Cross-attention reads from the pre-populated cross cache.
//
// On the prompt pass dumps are wired for every dec.* tensor so the
// validate.py harness keeps working. Step passes produce the same
// graph shape minus the dumps.
//
// skip_log_softmax=true outputs pre-softmax logits (argmax-invariant,
// cheaper readback). skip_log_softmax=false matches the reference dump.
DecoderBuild build_decoder_graph_kv(ggml_context *         compute_ctx,
                                    const WhisperWeights & weights,
                                    const WhisperHParams & hp,
                                    WhisperKvCache &       kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    bool                   skip_log_softmax = false,
                                    bool                   use_flash        = true);

} // namespace transcribe::whisper
