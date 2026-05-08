// arch/canary/decoder.h - Canary autoregressive Transformer decoder graph builder.
//
// Modeled on src/arch/cohere/decoder.h. Differences:
//   - per-sublayer dumps (self_attn / cross_attn / ffn outputs) at
//     layers {0, n_layers/2, n_layers-1} so the C++ matches the
//     reference dumper's hooks
//   - LM head is UNTIED: head.weight is read as a separate tensor
//     and `mul_mat(head.weight, x) + head.bias` produces the logits
//   - tensor naming follows canary's GGUF (norm1/2/3, q/k/v/o, ffn.up/down)

#pragma once

#include "ggml.h"

#include <cstdint>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::canary {

struct CanaryHParams;
struct CanaryWeights;
struct CanaryKvCache;

struct DecoderDumps {
    ggml_tensor * token_emb       = nullptr;
    ggml_tensor * pos_emb         = nullptr;
    ggml_tensor * embed_norm      = nullptr;
    // For canary's per-sublayer dumps: indexed by layer, three slots each.
    // Slot 0 = post self_attn residual, slot 1 = post cross_attn residual,
    // slot 2 = post ffn residual (i.e. block out).
    ggml_tensor * sub_self [64] = {};
    ggml_tensor * sub_cross[64] = {};
    ggml_tensor * sub_ffn  [64] = {};
    ggml_tensor * out_before_head = nullptr;
    ggml_tensor * logits_raw      = nullptr;  // pre-softmax
    ggml_tensor * logits          = nullptr;  // post-softmax (when log_softmax fires)
};

struct DecoderBuild {
    ggml_tensor * token_ids_in   = nullptr;
    ggml_tensor * pos_ids_in     = nullptr;
    ggml_tensor * encoder_out_in = nullptr;

    ggml_tensor * out        = nullptr;  // logits [vocab, seq]
    ggml_tensor * argmax_out = nullptr;  // [seq] i32 when skip_log_softmax

    DecoderDumps  dumps {};
    ggml_cgraph * graph = nullptr;
};

// Build a graph that computes cross-attention K/V for all decoder
// layers from the encoder output, writing them into the cross-attn
// KV cache.
DecoderBuild build_cross_kv_graph(ggml_context *         ctx,
                                  const CanaryWeights &  w,
                                  const CanaryHParams &  hp,
                                  CanaryKvCache &        kv_cache,
                                  int                    T_enc);

// Build a decoder graph that uses the KV cache. Works for both prompt
// pass (n_tokens > 1, n_past == 0) and step pass (n_tokens == 1).
//
// skip_log_softmax: when true, skip the log-softmax tail and add a GPU
// argmax over the last new-token position (saves a vocab-wide softmax
// per step). The pre-softmax logits remain available via dumps.logits_raw.
DecoderBuild build_decoder_graph_kv(ggml_context *         ctx,
                                    const CanaryWeights &  w,
                                    const CanaryHParams &  hp,
                                    CanaryKvCache &        kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    bool                   skip_log_softmax = false,
                                    bool                   use_flash        = true);

} // namespace transcribe::canary
