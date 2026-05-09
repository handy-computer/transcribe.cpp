// arch/cohere/decoder.h - Cohere ASR Transformer decoder graph builder.
//
// The decoder is an autoregressive Transformer with:
//   - Token embedding + learned positional embedding + LayerNorm
//   - N decoder layers (self-attn + cross-attn + FFN)
//   - Final LayerNorm
//   - Linear head (tied to token embedding) + log-softmax
//
// Two graph modes are supported:
//   1. Prompt pass: process N tokens, write KV cache for all layers.
//   2. Step pass: process 1 token, read+extend KV cache.
//
// Cross-attention K/V are computed in a separate graph
// (build_cross_kv_graph) and stored in the cross-attention cache.
// Both prompt and step passes read from this pre-computed cache.

#pragma once

#include "ggml.h"

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::cohere {

struct CohereHParams;
struct CohereWeights;
struct CohereKvCache;

struct DecoderDumps {
    ggml_tensor * token_emb       = nullptr;
    ggml_tensor * pos_emb         = nullptr;
    ggml_tensor * embed_norm      = nullptr;
    ggml_tensor * block_out[64]   = {};       // per-layer outputs (up to 64 layers)
    ggml_tensor * out_before_head = nullptr;
    ggml_tensor * logits_raw      = nullptr;  // pre-log-softmax logits
    ggml_tensor * logits          = nullptr;  // post-log-softmax logits
};

struct DecoderBuild {
    // Input handles.
    ggml_tensor * token_ids_in   = nullptr;  // [seq_len] i32
    ggml_tensor * pos_ids_in     = nullptr;  // [seq_len] i32
    ggml_tensor * encoder_out_in = nullptr;  // [dec_hidden, T_enc] f32

    // Output.
    ggml_tensor * out = nullptr;  // logits [vocab_size, seq_len]
    ggml_tensor * argmax_out = nullptr;  // [seq_len] i32, set when skip_log_softmax

    DecoderDumps dumps {};
    ggml_cgraph * graph = nullptr;
};

// Build a decoder prompt-pass graph. No KV cache -- processes the full
// prompt sequence at once with a causal self-attention mask.
//
// seq_len:   number of prompt tokens.
// T_enc:     number of encoder frames (after enc-dec projection).
// use_flash: true -> fused ggml_flash_attn_ext path;
//            false -> manual mul_mat + soft_max_ext + mul_mat path.
//            Defaults true because dk=128 has flash kernels on every
//            backend we ship.
DecoderBuild build_decoder_graph(ggml_context *         compute_ctx,
                                 const CohereWeights &  weights,
                                 const CohereHParams &  hp,
                                 int                    seq_len,
                                 int                    T_enc,
                                 bool                   use_flash = true);

// Build a graph that computes cross-attention K/V for all decoder
// layers from the encoder output, writing them into the cross-attn
// KV cache. This is run once per utterance.
//
// The graph reads from encoder_out (which must be set as an input)
// and writes into kv_cache.cross_k / kv_cache.cross_v via ggml_cpy.
//
// Returns: a DecoderBuild with only .encoder_out_in and .graph set.
DecoderBuild build_cross_kv_graph(ggml_context *         compute_ctx,
                                  const CohereWeights &  weights,
                                  const CohereHParams &  hp,
                                  CohereKvCache &        kv_cache,
                                  int                    T_enc);

// Build a decoder graph that uses the KV cache. Works for both
// prompt pass (n_tokens > 1) and step pass (n_tokens = 1).
//
// n_tokens: number of new tokens to process.
// n_past:   number of tokens already in the self-attention KV cache.
//           For prompt pass this is 0; for step passes it's the
//           running total of previously processed tokens.
// T_enc:    number of encoder frames (for cross-attention cache shape).
//
// The graph:
//   - Computes Q/K/V for self-attention from the new tokens only.
//   - Writes new K/V into the self-attention cache at position n_past
//     via ggml_cpy.
//   - Reads the full self-attention cache [0..n_past+n_tokens) for
//     attention computation.
//   - Reads cross-attention K/V from the pre-populated cache.
//   - Outputs logits for the new tokens only.
DecoderBuild build_decoder_graph_kv(ggml_context *         compute_ctx,
                                    const CohereWeights &  weights,
                                    const CohereHParams &  hp,
                                    CohereKvCache &        kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    bool                   skip_log_softmax = false,
                                    bool                   use_flash        = true);

// Static-topology single-token decoder graph. Built once per utterance
// after the prompt pass and reused for every step in the autoregressive
// loop. Compared to build_decoder_graph_kv with n_tokens=1, this graph
// is independent of n_past:
//   - KV cache writes go through ggml_set_rows with kv_idx_in as the
//     runtime row index (instead of ggml_cpy at a build-time-baked
//     view offset).
//   - Self-attn K/V reads span the full [0, max_n_kv) window; mask_in
//     gates which positions are valid each step.
// Caller pre-allocates the graph + sched once and only updates the
// runtime inputs (token_id_in, pos_id_in, kv_idx_in, mask_in) per step.
struct StepBuild {
    ggml_tensor * token_id_in = nullptr;  // i32 [1]
    ggml_tensor * pos_id_in   = nullptr;  // i32 [1]
    ggml_tensor * kv_idx_in   = nullptr;  // i64 [1]
    ggml_tensor * mask_in     = nullptr;  // f16 [max_n_kv, 1]
    ggml_tensor * argmax_out  = nullptr;  // i32 [1]
    int           max_n_kv    = 0;
    ggml_cgraph * graph       = nullptr;
};

StepBuild build_step_graph(ggml_context *         compute_ctx,
                           const CohereWeights &  weights,
                           const CohereHParams &  hp,
                           CohereKvCache &        kv_cache,
                           int                    max_n_kv,
                           int                    T_enc,
                           bool                   use_flash = true);

} // namespace transcribe::cohere
