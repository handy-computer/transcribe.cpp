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
//   1. build_decoder_prefill_graph() — non-cached prefill, used only
//      for language detection (a single SOT-only forward pass on the
//      first chunk before the cross-KV cache exists). Cross-K/V are
//      recomputed from the encoder output inside the graph.
//
//   2. build_cross_kv_graph() + build_decoder_graph_kv() — KV-cached
//      autoregressive path. The production decoder driver. Cross-K/V
//      are precomputed once per chunk into the cross cache; self-K/V
//      are written on each step into the self cache. The same graph
//      handles both the prompt pass (n_tokens > 1, n_past = 0) and
//      per-step generation (n_tokens = 1, n_past = current). All
//      dec.* dump points are wired on the prompt pass so validate.py
//      tolerances apply directly.
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
    ggml_tensor * encoder_out_in = nullptr;  // [d_model, T_enc] f32 (lang-det prefill + cross_kv graph)
    ggml_tensor * causal_mask_in = nullptr;  // [n_kv, seq_len] f32 (cast to f16 inside)

    // Cross-attention mask, only present when the cross K/V cache
    // is padded (T_enc_pad > T_enc). Shape [T_enc_pad, n_tokens] f32
    // (cast to f16 inside). Caller populates with 0 for k in [0,
    // T_enc) and -inf for k in [T_enc, T_enc_pad), so the unmasked
    // padded slots do not contribute to the cross-attn output.
    ggml_tensor * cross_mask_in  = nullptr;

    // Output: log-softmax logits [vocab_size, seq_len].
    ggml_tensor * out = nullptr;

    DecoderDumps  dumps {};
    ggml_cgraph * graph = nullptr;
};

// Build a decoder prefill graph in compute_ctx (no KV cache). Used
// only by the language-detection forward pass on the first chunk;
// every other decoder forward goes through build_decoder_graph_kv.
DecoderBuild build_decoder_prefill_graph(ggml_context *         compute_ctx,
                                         const WhisperWeights & weights,
                                         const WhisperHParams & hp,
                                         int                    seq_len,
                                         int                    T_enc,
                                         bool                   use_flash = true);

// Build a graph that computes cross-attention K/V for every decoder
// layer from the encoder output and writes them into the cross-attn
// KV cache. Run once per chunk (long-form audio runs the encoder per
// 30 s window and rebuilds the cross cache each time).
//
// encoder_out is the backend-resident persistent tensor populated by
// the encoder graph (shape [d_model, T_enc]). The cross-KV graph
// reads it via a view in compute_ctx — no host roundtrip required.
// Caller calls ggml_backend_sched_graph_compute after alloc_graph.
DecoderBuild build_cross_kv_graph(ggml_context *         compute_ctx,
                                  const WhisperWeights & weights,
                                  const WhisperHParams & hp,
                                  WhisperKvCache &       kv_cache,
                                  ggml_tensor *          encoder_out,
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
//
// kv_pad is the active-KV padding multiple for the FA kernel
// (whisper.cpp: 32 on Metal+FA, 1 elsewhere). When kv_pad > 1 the
// effective n_kv is rounded up to a multiple of kv_pad, and the
// caller-supplied causal_mask_in covers the trailing padded slots
// with -inf so they do not contribute to the FA output.
DecoderBuild build_decoder_graph_kv(ggml_context *         compute_ctx,
                                    const WhisperWeights & weights,
                                    const WhisperHParams & hp,
                                    WhisperKvCache &       kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    int                    kv_pad           = 1,
                                    bool                   skip_log_softmax = false,
                                    bool                   use_flash        = true);

} // namespace transcribe::whisper
