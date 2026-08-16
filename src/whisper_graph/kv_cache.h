// src/whisper_graph/kv_cache.h - KV cache + persistent encoder output for the
// shared Whisper encoder-decoder graph.
//
// Family-agnostic backend-resident state: the self/cross attention KV cache
// (single-shot and batched layouts), the persistent F32 encoder-output tensor
// the cross-KV graph reads through a view, and the flash-attention padding
// policy. Shared by arch/whisper and arch/crisperwhisper; see
// whisper_graph.h for why this component exists.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"

struct ggml_context;
struct ggml_tensor;

namespace transcribe::whisper_graph {

// KV cache for the autoregressive decoder. Self-attention cache grows one
// entry per decode step until EOS; cross-attention cache is computed once from
// encoder output and reused across steps. One flat 1D tensor per role;
// per-layer slices are views built at graph time.
struct KvCache {
    // Self-attention cache.
    // Flat tensors of size [d_model * n_layer * n_ctx].
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;

    // Cross-attention cache.
    // Flat tensors of size [d_model * n_layer * T_enc].
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;

    // ggml context that owns the cache tensor metadata.
    ggml_context * ctx = nullptr;

    // Backend buffer backing all cache tensors.
    ggml_backend_buffer_t buffer = nullptr;

    // Maximum self-attention sequence length.
    int n_ctx = 0;

    // Batch dimension. 1 for the single-shot path (layout byte-identical
    // to the pre-batch cache); >1 for the offline batched decode, where
    // each role tensor carries a per-utterance slab. Layout (batched):
    //   self_k/self_v  : [d_model * n_ctx * n_batch * n_layer]
    //   cross_k/cross_v: [d_model * T_enc * n_batch * n_layer]
    // with slab(layer,b) at offset (b + n_batch*layer)*<rows>*d_model.
    int n_batch = 1;

    // Current number of filled positions in the self-attention cache.
    int n = 0;

    // Write head for the next self-attention step.
    int head = 0;

    // Number of encoder frames in the cross-attention cache.
    int T_enc = 0;

    // Padded encoder length (= GGML_PAD(T_enc, 256)). The cross K/V
    // tensors are allocated to T_enc_pad rows per layer so the FA op
    // sees a sequence dim that is a multiple of the Metal kernel's
    // block size; the trailing T_enc_pad - T_enc rows are left zero
    // by buffer_clear and never written by build_cross_kv_graph.
    // Layer offsets in the flat cache must use T_enc_pad (not T_enc)
    // so adjacent layers do not overlap.
    int T_enc_pad = 0;

    // Whether cross-attention cache has been populated this run.
    bool cross_populated = false;

    void free() {
        if (buffer != nullptr) {
            safe_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        self_k          = nullptr;
        self_v          = nullptr;
        cross_k         = nullptr;
        cross_v         = nullptr;
        n_batch         = 1;
        n               = 0;
        head            = 0;
        T_enc           = 0;
        T_enc_pad       = 0;
        cross_populated = false;
    }
};

// Persistent backend-resident F32 encoder output [d_model, T_enc], held across
// the encoder->cross-KV transition so cross-KV reads it via a view (no
// GPU->CPU->GPU roundtrip): the encoder graph ends with a ggml_cpy into it.
// Allocated on first use, reallocated only on shape change (T_enc is fixed at
// max_source_positions=1500 for stock variants, so effectively never).
struct EncOut {
    ggml_tensor *         tensor  = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    int                   d_model = 0;
    int                   T_enc   = 0;

    void free() {
        if (buffer != nullptr) {
            safe_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        tensor  = nullptr;
        d_model = 0;
        T_enc   = 0;
    }
};

bool enc_out_init(EncOut & enc_out, ggml_backend_t backend, int d_model, int T_enc);

// Active-KV padding for the self-attention step graph (whisper.cpp's
// whisper_kv_cache_get_padding): 32 on Metal+FA, 1 otherwise. Aligns the FA
// op's n_kv to a kernel-preferred multiple; trailing positions are masked to
// -inf. Returns 1 (no padding) when use_flash is false.
int kv_pad_self_attn(transcribe::BackendKind kind, bool use_flash);

// Cross-attention cache padding multiple. Universal in whisper.cpp (not gated
// on backend); cost is a few unused rows per layer plus small cross-attn
// dilution.
constexpr int k_cross_kv_pad = 256;

// Allocate cache tensors. n_ctx caps self-attention length; T_enc is
// fixed at 1500 for whisper (max_source_positions after the stride-2
// conv). d_model is the per-layer hidden dim. kv_type is the storage
// dtype for all four cache tensors.
bool kv_cache_init(KvCache &      cache,
                   ggml_backend_t backend,
                   int            n_ctx,
                   int            T_enc,
                   int            d_model,
                   int            n_layer,
                   ggml_type      kv_type);

// Allocate batched cache tensors for the offline batched decode. Same as
// kv_cache_init but each role tensor carries an n_batch dimension; the
// cross cache uses T_enc (== T_enc_max, no 256-pad — the batched cross
// graph + a per-utterance cross-pad mask handle the FA shape). n_batch==1
// is byte-identical to kv_cache_init.
bool kv_cache_init_batched(KvCache &      cache,
                           ggml_backend_t backend,
                           int            n_ctx,
                           int            T_enc,
                           int            d_model,
                           int            n_layer,
                           int            n_batch,
                           ggml_type      kv_type);

}  // namespace transcribe::whisper_graph
