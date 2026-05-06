// arch/moonshine/decoder.h - Moonshine decoder graph builders.
//
// Two graph families:
//
//   1. build_cross_kv_graph() - precomputes cross-attn K/V for every
//      decoder layer from the encoder output and writes them into the
//      cross cache. Run once per session (T_enc may vary across calls
//      because moonshine consumes variable-length raw audio).
//
//   2. build_decoder_graph_kv() - the autoregressive prompt/step path.
//      Handles both the prompt pass (n_tokens=1, n_past=0 — moonshine's
//      prompt is a single bos token) and steady-state step passes
//      (n_tokens=1, n_past=current). All dec.* dump points are wired
//      on the prompt pass; the validate.py harness compares them
//      against the reference dumps.
//
// All attention projections (q/k/v/o) are bias-less (attention_bias=False).
// Self-attn applies partial RoPE 0.9 to q/k; cross-attn does NOT rotate.
// MLP is SwiGLU: fc1 hidden→2·intermediate, split [x_proj, gate],
// silu(gate)⊙x_proj, fc2 intermediate→hidden.

#pragma once

#include "ggml.h"

#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::moonshine {

struct MoonshineHParams;
struct MoonshineWeights;
struct MoonshineKvCache;

struct DecoderDumps {
    ggml_tensor * token_emb       = nullptr;   // dec.token_emb
    ggml_tensor * embed_sum       = nullptr;   // dec.embed_sum (= token_emb)
    std::vector<ggml_tensor *> block_outs;     // dec.block.{i}.out
    ggml_tensor * out_before_head = nullptr;   // dec.out_before_head
    ggml_tensor * logits_raw      = nullptr;   // dec.logits_raw
    ggml_tensor * logits          = nullptr;   // dec.logits (log_softmax)
};

struct DecoderBuild {
    ggml_tensor * token_ids_in   = nullptr;  // [n_tokens] i32
    ggml_tensor * pos_ids_in     = nullptr;  // [n_tokens] i32  (absolute positions)
    ggml_tensor * encoder_out_in = nullptr;  // [d_model, T_enc] f32 (cross_kv graph only)
    ggml_tensor * causal_mask_in = nullptr;  // [n_kv, n_tokens] f32 (n_tokens>1 only)

    ggml_tensor * out         = nullptr;     // logits or log-softmax depending on flag
    ggml_tensor * argmax_out  = nullptr;     // [n_tokens] i32, set when skip_log_softmax

    DecoderDumps  dumps {};
    ggml_cgraph * graph = nullptr;
};

// Build a graph that computes cross-attention K/V for every decoder
// layer from the encoder output and writes them into the cross-attn
// KV cache. Reads encoder_out [d_model, T_enc] (must already be allocated
// in the same backend the cache lives on). Writes into kv_cache.cross_k /
// .cross_v.
DecoderBuild build_cross_kv_graph(ggml_context *           compute_ctx,
                                  const MoonshineWeights & weights,
                                  const MoonshineHParams & hp,
                                  MoonshineKvCache &       kv_cache,
                                  int                      T_enc);

// Build a KV-cached decoder graph covering both the prompt pass
// (n_past=0) and per-step generation (n_past>=1). For moonshine the
// prompt pass always has n_tokens=1 (just the bos token), but the
// builder accepts n_tokens > 1 for completeness.
//
// `skip_log_softmax=true` outputs raw pre-softmax logits (argmax-
// invariant; cheaper readback). `skip_log_softmax=false` matches the
// reference dump's `dec.logits` tensor (log_softmax of logits_raw).
DecoderBuild build_decoder_graph_kv(ggml_context *           compute_ctx,
                                    const MoonshineWeights & weights,
                                    const MoonshineHParams & hp,
                                    MoonshineKvCache &       kv_cache,
                                    int                      n_tokens,
                                    int                      n_past,
                                    int                      T_enc,
                                    bool                     skip_log_softmax = false,
                                    bool                     use_flash        = true);

} // namespace transcribe::moonshine
