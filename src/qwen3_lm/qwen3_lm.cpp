// src/qwen3_lm/qwen3_lm.cpp - shared Qwen3 LM block math.
//
// See qwen3_lm.h for the API contract. The block forward functions are
// strict extractions of the per-family code that lived in
// arch/qwen3_asr/decoder.cpp and arch/funasr_nano/decoder.cpp before
// this module landed; tensor topology, math, and KV memory layout are
// preserved bit-for-bit.

#include "qwen3_lm.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace transcribe::qwen3_lm {

namespace {

// Qwen3 RMSNorm: weight * rsqrt(mean(x²) + eps) * x.
ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x,
                       ggml_tensor * weight, float eps)
{
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

} // namespace

// ---------------------------------------------------------------------------
// KvCache
// ---------------------------------------------------------------------------

void KvCache::free() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    self_k = nullptr;
    self_v = nullptr;
    n    = 0;
    head = 0;
}

bool kv_init(KvCache &      cache,
             ggml_backend_t backend,
             int            n_ctx,
             int            n_kv_heads,
             int            head_dim,
             int            n_layer,
             ggml_type      kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
                     "qwen3_lm kv_init: unsupported kv_type=%d\n",
                     static_cast<int>(kv_type));
        return false;
    }

    const size_t ctx_size = 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "qwen3_lm kv_init: ggml_init failed\n");
        return false;
    }

    const int64_t elems =
        static_cast<int64_t>(n_kv_heads) * head_dim * n_ctx * n_layer;
    cache.self_k = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
    cache.self_v = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
    ggml_set_name(cache.self_k, "kv_self_k");
    ggml_set_name(cache.self_v, "kv_self_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        std::fprintf(stderr, "qwen3_lm kv_init: buffer alloc failed\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx = n_ctx;
    cache.n     = 0;
    cache.head  = 0;
    return true;
}

// ---------------------------------------------------------------------------
// Block forward — prefill
// ---------------------------------------------------------------------------

ggml_tensor * block_prefill(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    ggml_tensor *       x,
    const BlockView &   view,
    const BlockParams & params,
    KvCache &           kv_cache,
    int                 layer_idx,
    int                 T_seq,
    ggml_tensor *       mask,
    ggml_tensor *       positions,
    BlockOpts           opts)
{
    const int64_t n_heads    = params.n_heads;
    const int64_t n_kv_heads = params.n_kv_heads;
    const int64_t n_groups   = n_heads / n_kv_heads;
    const int64_t head_dim   = params.head_dim;
    const int64_t q_dim      = n_heads    * head_dim;
    const int64_t kv_dim     = n_kv_heads * head_dim;
    const int     n_ctx      = kv_cache.n_ctx;
    const float   rms_eps    = params.rms_eps;
    const float   rope_theta = params.rope_theta;
    const float   scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    // ---- Attention sub-layer ----
    ggml_tensor * x_norm = rms_norm(ctx, x, view.norm_attn_w, rms_eps);

    // Q/K/V projections (bias-free on Qwen3). Packing into one mul_mat
    // consistently regresses on Metal; left separate.
    ggml_tensor * Q = ggml_mul_mat(ctx, view.attn_q_w, x_norm);
    ggml_tensor * K = ggml_mul_mat(ctx, view.attn_k_w, x_norm);
    ggml_tensor * V = ggml_mul_mat(ctx, view.attn_v_w, x_norm);

    Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads,    T_seq, 1);
    K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_seq, 1);
    V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_seq, 1);

    // Per-head Q/K RMSNorm on head_dim (Qwen3 innovation).
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), view.attn_q_norm);
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), view.attn_k_norm);

    // RoPE (NeoX rotate_half) on Q and K. MRoPE collapses to NeoX for
    // text-only ASR (every position has the same (T,H,W) coordinate).
    Q = ggml_rope_ext(ctx, Q, positions, /*c=*/nullptr,
                      static_cast<int>(head_dim),
                      GGML_ROPE_TYPE_NEOX,
                      params.max_position,
                      rope_theta,
                      1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    K = ggml_rope_ext(ctx, K, positions, nullptr,
                      static_cast<int>(head_dim),
                      GGML_ROPE_TYPE_NEOX,
                      params.max_position,
                      rope_theta,
                      1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // KV write: K, V have ne=[D, Hkv, T_seq, 1]; memory order
    // (fastest→slowest) is D, Hkv, T. Cache stores position-major
    // within each layer (per position, all D·Hkv values contiguous).
    // Same layout — a 1D cpy handles it.
    {
        const size_t layer_off = static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
        const size_t n_elem    = static_cast<size_t>(T_seq) * kv_dim;

        ggml_tensor * k_dst = ggml_view_1d(
            ctx, kv_cache.self_k, n_elem, k_elem * layer_off);
        ggml_tensor * v_dst = ggml_view_1d(
            ctx, kv_cache.self_v, n_elem, v_elem * layer_off);

        ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
    }

    // Read K, V back from the cache for attention as strided
    // [D, T_seq, Hkv] views (no permute + cont needed). mul_mat and
    // flash_attn_ext both accept strided inputs.
    const size_t layer_off_bytes =
        k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
    ggml_tensor * K_att = ggml_view_3d(
        ctx, kv_cache.self_k,
        head_dim, T_seq, n_kv_heads,
        /*nb1=*/k_elem * kv_dim,
        /*nb2=*/k_elem * head_dim,
        layer_off_bytes);
    ggml_tensor * V_att = ggml_view_3d(
        ctx, kv_cache.self_v,
        head_dim, T_seq, n_kv_heads,
        v_elem * kv_dim,
        v_elem * head_dim,
        v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim);

    // Permute Q for attention: [D, H, T_seq, 1] → [D, T_seq, H, 1].
    ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

    ggml_tensor * o;
    if (opts.use_flash) {
        // ggml_flash_attn_ext handles GQA natively when K/V have
        // n_kv_heads along axis 2 and Q has n_heads — no
        // repeat_interleave needed. Mask is F16 (host upload avoids
        // per-layer ggml_cast).
        o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask,
                                scale_attn, /*max_bias=*/0.0f,
                                /*logit_softcap=*/0.0f);
        o = ggml_reshape_2d(ctx, o, q_dim, T_seq);
    } else {
        // Manual GQA path: emulate repeat_interleave via reshape
        // -into-(1,Hkv) → repeat → collapse. Numerics match the
        // reference's explicit repeat_kv.
        ggml_tensor * K_att_c = ggml_cont(ctx, K_att);
        ggml_tensor * V_att_c = ggml_cont(ctx, V_att);
        ggml_tensor * K_4d = ggml_reshape_4d(
            ctx, K_att_c, head_dim, T_seq, 1, n_kv_heads);
        ggml_tensor * V_4d = ggml_reshape_4d(
            ctx, V_att_c, head_dim, T_seq, 1, n_kv_heads);
        ggml_tensor * K_rep_template = ggml_new_tensor_4d(
            ctx, K_att->type, head_dim, T_seq, n_groups, n_kv_heads);
        ggml_tensor * V_rep_template = ggml_new_tensor_4d(
            ctx, V_att->type, head_dim, T_seq, n_groups, n_kv_heads);
        ggml_tensor * K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
        ggml_tensor * V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
        ggml_tensor * K_full = ggml_reshape_3d(
            ctx, K_rep, head_dim, T_seq, n_heads);
        ggml_tensor * V_full = ggml_reshape_3d(
            ctx, V_rep, head_dim, T_seq, n_heads);

        ggml_tensor * kq = ggml_mul_mat(ctx, K_full, Q_att);
        ggml_tensor * kq_soft = ggml_soft_max_ext(
            ctx, kq, mask, scale_attn, /*max_bias=*/0.0f);
        ggml_tensor * V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, V_t, kq_soft);
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
        o = ggml_reshape_2d(ctx, o, q_dim, T_seq);
    }

    o = ggml_mul_mat(ctx, view.attn_o_w, o);
    x = ggml_add(ctx, x, o);

    // Optional last-position slice before FFN (caller uses on the LAST
    // block when only the final logits are consumed).
    if (opts.slice_last_before_ffn) {
        const int64_t hidden = x->ne[0];
        const size_t  elem   = ggml_element_size(x);
        x = ggml_view_2d(ctx, x, hidden, 1,
                         elem * hidden,
                         elem * hidden * static_cast<size_t>(T_seq - 1));
        x = ggml_cont(ctx, x);
    }

    // ---- MLP sub-layer (SwiGLU on packed gate_up) ----
    ggml_tensor * ff_norm = rms_norm(ctx, x, view.norm_ffn_w, rms_eps);
    ggml_tensor * gate_up = ggml_mul_mat(ctx, view.ffn_gate_up_w, ff_norm);
    ggml_tensor * ff      = ggml_swiglu(ctx, gate_up);
    ff = ggml_mul_mat(ctx, view.ffn_down_w, ff);

    x = ggml_add(ctx, x, ff);
    return x;
}

// ---------------------------------------------------------------------------
// Block forward — step
// ---------------------------------------------------------------------------

ggml_tensor * block_step(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    ggml_tensor *       x,
    const BlockView &   view,
    const BlockParams & params,
    KvCache &           kv_cache,
    int                 layer_idx,
    int                 max_n_kv,
    ggml_tensor *       mask,
    ggml_tensor *       position,
    ggml_tensor *       kv_idx,
    bool                use_flash)
{
    const int64_t n_heads    = params.n_heads;
    const int64_t n_kv_heads = params.n_kv_heads;
    const int64_t n_groups   = n_heads / n_kv_heads;
    const int64_t head_dim   = params.head_dim;
    const int64_t q_dim      = n_heads    * head_dim;
    const int64_t kv_dim     = n_kv_heads * head_dim;
    const int     n_ctx      = kv_cache.n_ctx;
    const float   rms_eps    = params.rms_eps;
    const float   rope_theta = params.rope_theta;
    const float   scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    // ---- Attention sub-layer ----
    ggml_tensor * x_norm = rms_norm(ctx, x, view.norm_attn_w, rms_eps);

    ggml_tensor * Q = ggml_mul_mat(ctx, view.attn_q_w, x_norm);
    ggml_tensor * K = ggml_mul_mat(ctx, view.attn_k_w, x_norm);
    ggml_tensor * V = ggml_mul_mat(ctx, view.attn_v_w, x_norm);

    Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads,    1, 1);
    K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, 1, 1);
    V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, 1, 1);

    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), view.attn_q_norm);
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), view.attn_k_norm);

    Q = ggml_rope_ext(ctx, Q, position, nullptr,
                      static_cast<int>(head_dim),
                      GGML_ROPE_TYPE_NEOX,
                      params.max_position,
                      rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    K = ggml_rope_ext(ctx, K, position, nullptr,
                      static_cast<int>(head_dim),
                      GGML_ROPE_TYPE_NEOX,
                      params.max_position,
                      rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // KV write via ggml_set_rows with a dynamic index. Static topology;
    // only `kv_idx`'s value changes per step.
    {
        const size_t layer_off_k =
            k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
        const size_t layer_off_v =
            v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;

        ggml_tensor * k_layer = ggml_view_2d(
            ctx, kv_cache.self_k,
            kv_dim, n_ctx,
            k_elem * kv_dim, layer_off_k);
        ggml_tensor * v_layer = ggml_view_2d(
            ctx, kv_cache.self_v,
            kv_dim, n_ctx,
            v_elem * kv_dim, layer_off_v);

        ggml_tensor * K_row = ggml_reshape_2d(ctx, K, kv_dim, 1);
        ggml_tensor * V_row = ggml_reshape_2d(ctx, V, kv_dim, 1);

        ggml_build_forward_expand(
            gf, ggml_set_rows(ctx, k_layer, K_row, kv_idx));
        ggml_build_forward_expand(
            gf, ggml_set_rows(ctx, v_layer, V_row, kv_idx));
    }

    // Attention reads the FULL [0, max_n_kv) window. Mask zeros valid
    // positions and -inf's the rest, so softmax ignores empty slots.
    // Full-window read costs extra bandwidth (~1.6× avg) but the graph
    // is static → zero per-step rebuild overhead.
    const size_t layer_off_bytes =
        k_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim;
    ggml_tensor * K_att = ggml_view_3d(
        ctx, kv_cache.self_k,
        head_dim, max_n_kv, n_kv_heads,
        k_elem * kv_dim, k_elem * head_dim,
        layer_off_bytes);
    ggml_tensor * V_att = ggml_view_3d(
        ctx, kv_cache.self_v,
        head_dim, max_n_kv, n_kv_heads,
        v_elem * kv_dim, v_elem * head_dim,
        v_elem * static_cast<size_t>(layer_idx) * n_ctx * kv_dim);

    ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask,
                                scale_attn, /*max_bias=*/0.0f,
                                /*logit_softcap=*/0.0f);
        o = ggml_reshape_2d(ctx, o, q_dim, 1);
    } else {
        ggml_tensor * K_att_c = ggml_cont(ctx, K_att);
        ggml_tensor * V_att_c = ggml_cont(ctx, V_att);
        ggml_tensor * K_4d = ggml_reshape_4d(
            ctx, K_att_c, head_dim, max_n_kv, 1, n_kv_heads);
        ggml_tensor * V_4d = ggml_reshape_4d(
            ctx, V_att_c, head_dim, max_n_kv, 1, n_kv_heads);
        ggml_tensor * K_rep_template = ggml_new_tensor_4d(
            ctx, K_att->type, head_dim, max_n_kv, n_groups, n_kv_heads);
        ggml_tensor * V_rep_template = ggml_new_tensor_4d(
            ctx, V_att->type, head_dim, max_n_kv, n_groups, n_kv_heads);
        ggml_tensor * K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
        ggml_tensor * V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
        ggml_tensor * K_full = ggml_reshape_3d(
            ctx, K_rep, head_dim, max_n_kv, n_heads);
        ggml_tensor * V_full = ggml_reshape_3d(
            ctx, V_rep, head_dim, max_n_kv, n_heads);

        ggml_tensor * kq = ggml_mul_mat(ctx, K_full, Q_att);
        ggml_tensor * kq_soft = ggml_soft_max_ext(
            ctx, kq, mask, scale_attn, /*max_bias=*/0.0f);
        ggml_tensor * V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, V_t, kq_soft);
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
        o = ggml_reshape_2d(ctx, o, q_dim, 1);
    }

    o = ggml_mul_mat(ctx, view.attn_o_w, o);
    x = ggml_add(ctx, x, o);

    // ---- MLP sub-layer ----
    ggml_tensor * ff_norm = rms_norm(ctx, x, view.norm_ffn_w, rms_eps);
    ggml_tensor * gate_up = ggml_mul_mat(ctx, view.ffn_gate_up_w, ff_norm);
    ggml_tensor * ff      = ggml_swiglu(ctx, gate_up);
    ff = ggml_mul_mat(ctx, view.ffn_down_w, ff);

    x = ggml_add(ctx, x, ff);
    return x;
}

// ---------------------------------------------------------------------------
// pack_gate_up
// ---------------------------------------------------------------------------

void PackedGateUpHandles::free() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}


bool pack_gate_up(ggml_backend_t                  backend,
                  int                             hidden,
                  int                             intermediate,
                  const std::vector<GateUpEntry> & entries,
                  PackedGateUpHandles &           out_handles,
                  const char *                    error_tag)
{
    if (backend == nullptr || hidden <= 0 || intermediate <= 0 ||
        entries.empty())
    {
        std::fprintf(stderr,
                     "%s: pack_gate_up invalid args (hidden=%d intermediate=%d n=%zu)\n",
                     error_tag, hidden, intermediate, entries.size());
        return false;
    }

    const size_t ctx_size =
        entries.size() * ggml_tensor_overhead() + 1024;
    ggml_init_params packed_params {};
    packed_params.mem_size   = ctx_size;
    packed_params.mem_buffer = nullptr;
    packed_params.no_alloc   = true;

    out_handles.ctx = ggml_init(packed_params);
    if (out_handles.ctx == nullptr) {
        std::fprintf(stderr,
                     "%s: pack_gate_up ggml_init failed\n", error_tag);
        return false;
    }

    // Allocate one packed tensor per block with the SAME dtype as
    // gate_w. Mixed-quant blocks (e.g. all Q4_K_M) are fine because
    // gate and up share a row-wise quant block size.
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto & e = entries[i];
        if (e.gate_w == nullptr || e.up_w == nullptr ||
            e.gate_up_w_out == nullptr)
        {
            std::fprintf(stderr,
                         "%s: pack_gate_up entry %zu has null member\n",
                         error_tag, i);
            return false;
        }
        if (e.gate_w->type != e.up_w->type) {
            std::fprintf(stderr,
                         "%s: pack_gate_up entry %zu gate/up type mismatch (%d vs %d)\n",
                         error_tag, i,
                         static_cast<int>(e.gate_w->type),
                         static_cast<int>(e.up_w->type));
            return false;
        }
        ggml_tensor * t = ggml_new_tensor_2d(
            out_handles.ctx, e.gate_w->type, hidden, 2 * intermediate);
        if (t == nullptr) {
            std::fprintf(stderr,
                         "%s: pack_gate_up new_tensor_2d failed at %zu\n",
                         error_tag, i);
            return false;
        }
        *e.gate_up_w_out = t;
    }

    out_handles.buffer = ggml_backend_alloc_ctx_tensors(
        out_handles.ctx, backend);
    if (out_handles.buffer == nullptr) {
        std::fprintf(stderr,
                     "%s: pack_gate_up backend buffer alloc failed\n",
                     error_tag);
        return false;
    }
    ggml_backend_buffer_set_usage(out_handles.buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Fill: for row-wise quants (and f32/f16) concat-along-dim-1 is
    // gate's bytes followed by up's bytes. One CPU round-trip per
    // block at load time; zero cost at inference.
    std::vector<uint8_t> buf;
    for (const auto & e : entries) {
        ggml_tensor * gate_up = *e.gate_up_w_out;
        const size_t  gate_bytes = ggml_nbytes(e.gate_w);
        const size_t  up_bytes   = ggml_nbytes(e.up_w);
        if (ggml_nbytes(gate_up) != gate_bytes + up_bytes) {
            std::fprintf(stderr,
                         "%s: pack_gate_up size mismatch (%zu vs %zu + %zu)\n",
                         error_tag, ggml_nbytes(gate_up), gate_bytes, up_bytes);
            return false;
        }
        buf.resize(std::max(gate_bytes, up_bytes));
        ggml_backend_tensor_get(e.gate_w, buf.data(), 0, gate_bytes);
        ggml_backend_tensor_set(gate_up,  buf.data(), 0, gate_bytes);
        ggml_backend_tensor_get(e.up_w,   buf.data(), 0, up_bytes);
        ggml_backend_tensor_set(gate_up,  buf.data(), gate_bytes, up_bytes);
    }
    return true;
}

} // namespace transcribe::qwen3_lm
