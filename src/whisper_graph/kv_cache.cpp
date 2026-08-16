// src/whisper_graph/kv_cache.cpp - allocation for the shared Whisper
// encoder-decoder KV cache and persistent encoder output.

#include "kv_cache.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-log.h"

namespace transcribe::whisper_graph {

bool enc_out_init(EncOut & enc_out, ggml_backend_t backend, int d_model, int T_enc) {
    enc_out.free();

    const size_t     ctx_size = ggml_tensor_overhead() + 256;
    ggml_init_params params{};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    enc_out.ctx = ggml_init(params);
    if (enc_out.ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph enc_out: ggml_init failed");
        return false;
    }

    enc_out.tensor = ggml_new_tensor_2d(enc_out.ctx, GGML_TYPE_F32, d_model, T_enc);
    ggml_set_name(enc_out.tensor, "enc_out");

    enc_out.buffer = ggml_backend_alloc_ctx_tensors(enc_out.ctx, backend);
    if (enc_out.buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph enc_out: buffer alloc failed");
        ggml_free(enc_out.ctx);
        enc_out.ctx    = nullptr;
        enc_out.tensor = nullptr;
        return false;
    }

    enc_out.d_model = d_model;
    enc_out.T_enc   = T_enc;
    return true;
}

int kv_pad_self_attn(transcribe::BackendKind kind, bool use_flash) {
    if (!use_flash) {
        return 1;
    }
    switch (kind) {
        // Match whisper.cpp's whisper_kv_cache_get_padding (Metal+FA).
        case transcribe::BackendKind::Metal:
            return 32;
        // CUDA uses 256 in whisper.cpp; not exercised here yet.
        default:
            return 1;
    }
}

bool kv_cache_init(KvCache &      cache,
                   ggml_backend_t backend,
                   int            n_ctx,
                   int            T_enc,
                   int            d_model,
                   int            n_layer,
                   ggml_type      kv_type) {
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "whisper_graph kv_cache: unsupported kv_type=%d "
                "(only F16/F32)",
                static_cast<int>(kv_type));
        return false;
    }

    const size_t     ctx_size = 4 * ggml_tensor_overhead() + 256;
    ggml_init_params params{};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph kv_cache: ggml_init failed");
        return false;
    }

    // Cross K/V allocated at GGML_PAD(T_enc, 256) rows per layer so
    // the FA op consumes a sequence dim that is a multiple of the
    // Metal kernel's block size. Only the first T_enc rows are
    // written by build_cross_kv_graph; the trailing rows stay zero
    // (buffer_clear below). With K=V=0 in the padded slots, the
    // unmasked FA cross-attn output picks up a small dilution
    // factor — whisper.cpp ships with this trade-off.
    const int     T_enc_pad      = static_cast<int>(GGML_PAD(T_enc, k_cross_kv_pad));
    const int64_t self_elements  = static_cast<int64_t>(d_model) * n_layer * n_ctx;
    const int64_t cross_elements = static_cast<int64_t>(d_model) * n_layer * T_enc_pad;

    cache.self_k  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.self_v  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

    ggml_set_name(cache.self_k, "kv_self_k");
    ggml_set_name(cache.self_v, "kv_self_v");
    ggml_set_name(cache.cross_k, "kv_cross_k");
    ggml_set_name(cache.cross_v, "kv_cross_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph kv_cache: buffer alloc failed");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx           = n_ctx;
    cache.n_batch         = 1;
    cache.T_enc           = T_enc;
    cache.T_enc_pad       = T_enc_pad;
    cache.n               = 0;
    cache.head            = 0;
    cache.cross_populated = false;

    return true;
}

bool kv_cache_init_batched(KvCache &      cache,
                           ggml_backend_t backend,
                           int            n_ctx,
                           int            T_enc,
                           int            d_model,
                           int            n_layer,
                           int            n_batch,
                           ggml_type      kv_type) {
    if (n_batch <= 1) {
        // Degenerate batch — defer to the single-shot layout so callers
        // that accidentally pass n_batch==1 stay byte-identical.
        return kv_cache_init(cache, backend, n_ctx, T_enc, d_model, n_layer, kv_type);
    }
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph kv_cache(batched): unsupported kv_type=%d",
                static_cast<int>(kv_type));
        return false;
    }

    cache.free();

    const size_t     ctx_size = 4 * ggml_tensor_overhead() + 256;
    ggml_init_params params{};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph kv_cache(batched): ggml_init failed");
        return false;
    }

    // No 256-row cross pad in the batched layout: every short-form
    // utterance has T_enc == 1500 and the batched cross graph writes the
    // full slab; the per-utterance cross-pad mask gates invalid columns.
    const int64_t self_elements  = static_cast<int64_t>(d_model) * n_ctx * n_batch * n_layer;
    const int64_t cross_elements = static_cast<int64_t>(d_model) * T_enc * n_batch * n_layer;

    cache.self_k  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.self_v  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

    ggml_set_name(cache.self_k, "kv_self_k_b");
    ggml_set_name(cache.self_v, "kv_self_v_b");
    ggml_set_name(cache.cross_k, "kv_cross_k_b");
    ggml_set_name(cache.cross_v, "kv_cross_v_b");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whisper_graph kv_cache(batched): buffer alloc failed");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx           = n_ctx;
    cache.n_batch         = n_batch;
    cache.T_enc           = T_enc;
    cache.T_enc_pad       = T_enc;  // no extra pad in batched layout
    cache.n               = 0;
    cache.head            = 0;
    cache.cross_populated = false;

    return true;
}

}  // namespace transcribe::whisper_graph
