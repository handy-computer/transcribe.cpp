// arch/whistle/decoder.cpp - Whistle decoder graphs.
//
// Decoder block (architecture.py::Block + cross-attention, order probed on
// the engine, forward-map Decoder):
//   [Engram at sites]  u += sigmoid(<rms(u), rms(k_e)> / sqrt(d)) * v_e
//   u += sigmoid(attn_gate)  * ZCRMS(SelfAttn(ZCRMS(u)))     causal q/k/v taps, RoPE
//   u += sigmoid(cross_gate) * ZCRMS(CrossAttn(ZCRMS(u), mem))
//   u += HadamardMLP(ZCRMS(u))
// inside the 4-lane mHC stream; lane mean -> ZCRMS -> tied head.
//
// Every hypothesis advances one token per step at the same position, so a
// step graph is built per position (CPU-style dynamic graph, like cohere's
// CPU path). Tap and Engram histories are short ring buffers indexed by
// position; ring lengths keep the slot written this step distinct from
// every slot read.

#include "decoder.h"

#include "cpu_avx2.h"
#include "encoder.h"
#include "ggml.h"
#include "whistle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace transcribe::whistle {

namespace {

constexpr uint32_t kEngramSeed  = 0x9E3779B9u;
constexpr uint32_t kEngramPrime = 0x01000193u;

}  // namespace

void engram_rows(const std::vector<int> & tokens, int pos, const WhistleHParams & hp, int32_t * out) {
    const int      heads  = hp.engram_heads();
    const uint32_t stride = static_cast<uint32_t>(hp.engram_seed_heads > 0 ? hp.engram_seed_heads : heads);
    int            r      = 0;
    for (size_t oi = 0; oi < hp.engram_orders.size(); ++oi) {
        const int order = hp.engram_orders[oi];
        for (int h = 0; h < heads; ++h) {
            uint32_t acc = kEngramSeed * (static_cast<uint32_t>(oi) * stride + static_cast<uint32_t>(h) + 1u);
            for (int j = 0; j < order; ++j) {
                const int      p = pos - j;
                const uint32_t u = p >= 0 ? static_cast<uint32_t>(tokens[static_cast<size_t>(p)]) : 0u;
                acc              = (acc ^ u) * kEngramPrime;
            }
            acc ^= acc >> 15;
            out[r] = static_cast<int32_t>(r) * hp.engram_slots +
                     static_cast<int32_t>(acc % static_cast<uint32_t>(hp.engram_slots));
            ++r;
        }
    }
}

CrossKvBuild build_cross_kv_graph(ggml_context *          ctx,
                                  const WhistleWeights &  w,
                                  const WhistleHParams &  hp,
                                  const WhistleDecCache & cache,
                                  int                     utt,
                                  int                     T_utt) {
    CrossKvBuild  cb{};
    const int64_t qk = hp.qk_head_dim, vd = hp.v_head_dim, nh = hp.n_heads, d = hp.d_model;
    cb.graph  = ggml_new_graph_custom(ctx, 4096, false);
    cb.mem_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, T_utt);
    ggml_set_name(cb.mem_in, "dec.mem");
    ggml_set_input(cb.mem_in);
    ggml_tensor * mem = hada_in(ctx, cb.mem_in, w.hada);

    for (int l = 0; l < hp.dec_n_layers; ++l) {
        const WhistleAttn & c = w.dec_blocks[l].cross;
        ggml_tensor *       k = ggml_reshape_3d(ctx, linear(ctx, w, c.k, mem), qk, nh, T_utt);
        k                     = zcrms(ctx, k, c.k_norm);
        k                     = ggml_permute(ctx, k, 0, 2, 1, 3);  // [qk, T, nh]
        ggml_tensor * kc      = cache.k_cross[l];
        ggml_tensor * kv =
            ggml_view_3d(ctx, kc, qk, T_utt, nh, kc->nb[1], kc->nb[2], static_cast<size_t>(utt) * kc->nb[3]);
        ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, k, kv));

        ggml_tensor * v  = ggml_reshape_3d(ctx, linear(ctx, w, c.v, mem), vd, nh, T_utt);
        // Gather the transpose into a contiguous [T, vd, nh] first: the copy
        // into the cache is then row-contiguous instead of a strided scatter.
        v                = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [T, vd, nh]
        ggml_tensor * vc = cache.v_cross[l];
        ggml_tensor * vv =
            ggml_view_3d(ctx, vc, T_utt, vd, nh, vc->nb[1], vc->nb[2], static_cast<size_t>(utt) * vc->nb[3]);
        ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, v, vv));
    }
    return cb;
}

namespace {

// Slot `slot` of a ring tensor [dim, len, n_seq] -> [dim, n_seq] view.
ggml_tensor * ring_slot(ggml_context * ctx, ggml_tensor * ring, int slot) {
    return ggml_view_2d(ctx, ring, ring->ne[0], ring->ne[2], ring->nb[2], static_cast<size_t>(slot) * ring->nb[1]);
}

ggml_tensor * tap_row(ggml_context * ctx, ggml_tensor * taps, int j) {
    return ggml_view_1d(ctx, taps, taps->ne[0], static_cast<size_t>(j) * taps->nb[1]);
}

inline void split_rows(int64_t total, int ith, int nth, int64_t & b, int64_t & e) {
    const int64_t per = (total + nth - 1) / nth;
    b                 = std::min(total, per * ith);
    e                 = std::min(total, b + per);
}

inline int32_t input_pos(const ggml_tensor * pos_in) {
    return static_cast<const int32_t *>(pos_in->data)[0];
}

// One sequence's taps: out = x * t0 + sum_j ring(pos - j * dil) * tj, then x
// into the ring slot for pos. rb: the sequence's ring [dim, len] (row stride
// rs bytes); t: taps [dim, nt].
inline void taps_row(const float * x,
                     char *        rb,
                     size_t        rs,
                     int64_t       len,
                     const float * t,
                     int64_t       nt,
                     int64_t       dim,
                     int64_t       pos,
                     int64_t       dil,
                     float *       out) {
    for (int64_t i = 0; i < dim; ++i) {
        out[i] = x[i] * t[i];
    }
    for (int64_t j = 1; j < nt && pos - j * dil >= 0; ++j) {
        const float * r  = reinterpret_cast<const float *>(rb + ((pos - j * dil) % len) * rs);
        const float * tj = t + j * dim;
        for (int64_t i = 0; i < dim; ++i) {
            out[i] = out[i] + r[i] * tj[i];
        }
    }
    std::memcpy(rb + (pos % len) * rs, x, static_cast<size_t>(dim) * sizeof(float));
}

// Fused CPU apply_taps: same f32 operations in the same order as the graph
// below (raw * t0, then + ring(pos - j * dil) * tj), and writes raw into the
// ring slot for pos. src: raw [dim, S], ring [dim, len, S], taps [dim, n_taps],
// pos [S] I32 (all equal); userdata: the dilation.
void taps_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const int64_t       dil  = static_cast<int64_t>(reinterpret_cast<intptr_t>(ud));
    const ggml_tensor * raw  = dst->src[0];
    ggml_tensor *       ring = dst->src[1];
    const ggml_tensor * taps = dst->src[2];
    const int64_t       pos  = input_pos(dst->src[3]);
    const int64_t       dim = raw->ne[0], S = raw->ne[1], len = ring->ne[1], nt = taps->ne[1];
    int64_t             b, e;
    split_rows(S, ith, nth, b, e);
    const float * t = static_cast<const float *>(taps->data);
    for (int64_t s = b; s < e; ++s) {
        taps_row(reinterpret_cast<const float *>(static_cast<const char *>(raw->data) + s * raw->nb[1]),
                 static_cast<char *>(ring->data) + s * ring->nb[2], ring->nb[1], len, t, nt, dim, pos, dil,
                 reinterpret_cast<float *>(static_cast<char *>(dst->data) + s * dst->nb[1]));
    }
}

// Per-thread score scratch for the fused attention ops.
struct AttnScratch {
    float              stack[4096];
    std::vector<float> heap;

    float * get(int64_t n) {
        if (n <= 4096) {
            return stack;
        }
        heap.resize(static_cast<size_t>(n));
        return heap.data();
    }
};

inline const float * f32_at(const ggml_tensor * t, size_t off) {
    return reinterpret_cast<const float *>(static_cast<const char *>(t->data) + off);
}

// Fused decode self-attention over cache positions [0, pos]: QK^T, softmax,
// PV, parallel over (seq, KV head); the G query heads of a KV head share its
// K / V loads. src: Q [qk, G, nkv, S], K cache [qk, n_ctx, nkv, S], V cache
// [n_ctx4, vd, nkv, S] (transposed), pos [S] I32, gate [vd * G * nkv, S], and
// this step's k [qk, nkv, S] and v [vd * nkv, S] (each task first writes its
// own (seq, KV head) row of them into the caches at pos).
// dst [vd, G, nkv, S] = attention * sigmoid(gate).
void self_attn_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * Q  = dst->src[0];
    const ggml_tensor * kc = dst->src[1];
    const ggml_tensor * vc = dst->src[2];
    const int           n  = input_pos(dst->src[3]) + 1;
    const int64_t       qk = Q->ne[0], G = Q->ne[1], nkv = Q->ne[2], S = Q->ne[3], vd = dst->ne[0];
    const float         scl = 1.0f / std::sqrt(static_cast<float>(qk));
    AttnScratch         scratch;
    float *             sc = scratch.get(G * (n + 8));
    int64_t             b, e;
    split_rows(S * nkv, ith, nth, b, e);
    for (int64_t r = b; r < e; ++r) {
        const int64_t s = r / nkv, h = r % nkv;
        const size_t  ko = s * kc->nb[3] + h * kc->nb[2], vo = s * vc->nb[3] + h * vc->nb[2];
        if (dst->src[5] != nullptr) {  // KV write at pos
            const ggml_tensor * kn = dst->src[5];
            const ggml_tensor * vn = dst->src[6];
            std::memcpy(static_cast<char *>(kc->data) + ko + (n - 1) * kc->nb[1],
                        f32_at(kn, h * kn->nb[1] + s * kn->nb[2]), static_cast<size_t>(qk) * sizeof(float));
            const float * vr = f32_at(vn, s * vn->nb[1] + h * vd * sizeof(float));
            char *        vw = static_cast<char *>(vc->data) + vo + (n - 1) * vc->nb[0];
            for (int64_t i = 0; i < vd; ++i) {
                std::memcpy(vw + i * vc->nb[1], vr + i, sizeof(float));
            }
        }
        avx2::attn_multi(static_cast<int>(G), f32_at(Q, h * Q->nb[2] + s * Q->nb[3]), Q->nb[1] / sizeof(float),
                         f32_at(kc, ko), kc->nb[1] / sizeof(float), static_cast<int>(qk), f32_at(vc, vo),
                         vc->nb[1] / sizeof(float), static_cast<int>(vd), n, nullptr, 0, scl, sc,
                         reinterpret_cast<float *>(static_cast<char *>(dst->data) + r * G * vd * sizeof(float)), vd,
                         nullptr, 0);
        float *       o = reinterpret_cast<float *>(static_cast<char *>(dst->data) + r * G * vd * sizeof(float));
        const float * g = f32_at(dst->src[4], s * dst->src[4]->nb[1] + h * G * vd * sizeof(float));
        avx2::sigmoid_mul(o, g, o, G * vd);
    }
}

// Fused decode cross-attention, parallel over (head, utterance); the beams of
// an utterance share the head's K / V loads. src: q [qk, nh, S],
// K [qk, T4, nh, n_utt], V [T4, vd, nh, n_utt] (transposed), mask [T4, 1, 1,
// S] (0 / -inf), gate [vd * nh, S]. userdata: T, the rows attended (max
// utterance length). dst [vd * nh (+ T * nh), S]: per seq the output [vd, nh]
// times sigmoid(gate), then (when dst is wider) the probabilities [T, nh] for
// word timing.
void cross_attn_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * kc   = dst->src[1];
    const ggml_tensor * vc   = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int           T    = static_cast<int>(reinterpret_cast<intptr_t>(ud));
    const int64_t       qk = q->ne[0], nh = q->ne[1], S = q->ne[2], vd = vc->ne[1];
    const int64_t       per_utt = S / kc->ne[3];
    const bool          cap     = dst->ne[0] > vd * nh;
    const float         scl     = 1.0f / std::sqrt(static_cast<float>(qk));
    AttnScratch         scratch;
    float *             sc = scratch.get(per_utt * (T + 8));
    const size_t        rs = dst->nb[1] / sizeof(float);
    int64_t             b, e;
    split_rows(nh * kc->ne[3], ith, nth, b, e);
    for (int64_t r = b; r < e; ++r) {
        const int64_t h = r % nh, u = r / nh, s0 = u * per_utt;
        const size_t  ko = u * kc->nb[3] + h * kc->nb[2], vo = u * vc->nb[3] + h * vc->nb[2];
        float *       row = reinterpret_cast<float *>(static_cast<char *>(dst->data) + s0 * dst->nb[1]);
        avx2::attn_multi(static_cast<int>(per_utt), f32_at(q, h * q->nb[1] + s0 * q->nb[2]), q->nb[2] / sizeof(float),
                         f32_at(kc, ko), kc->nb[1] / sizeof(float), static_cast<int>(qk), f32_at(vc, vo),
                         vc->nb[1] / sizeof(float), static_cast<int>(vd), T, f32_at(mask, s0 * mask->nb[3]),
                         mask->nb[3] / sizeof(float), scl, sc, row + h * vd, rs, cap ? row + vd * nh + h * T : nullptr,
                         rs);
        for (int64_t s = s0; s < s0 + per_utt; ++s) {
            float * o = reinterpret_cast<float *>(static_cast<char *>(dst->data) + s * dst->nb[1]) + h * vd;
            avx2::sigmoid_mul(o, f32_at(dst->src[4], s * dst->src[4]->nb[1] + h * vd * sizeof(float)), o, vd);
        }
    }
}

// Fused CPU u + gate * (rms_norm(o) * w): ggml_rms_norm's f32 math (double
// sum of squares, f32 scale) followed by the same elementwise ops.
// src: u [d, S], o [d, S], w [d] (already 1 + scale); userdata: &gate.
void post_norm_residual_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const float         gate = *static_cast<const float *>(ud);
    const ggml_tensor * u    = dst->src[0];
    const ggml_tensor * o    = dst->src[1];
    const float *       w    = static_cast<const float *>(dst->src[2]->data);
    const int64_t       d = o->ne[0], S = o->ne[1];
    int64_t             b, e;
    split_rows(S, ith, nth, b, e);
    for (int64_t s = b; s < e; ++s) {
        const float * ur  = reinterpret_cast<const float *>(static_cast<const char *>(u->data) + s * u->nb[1]);
        const float * orw = reinterpret_cast<const float *>(static_cast<const char *>(o->data) + s * o->nb[1]);
        float *       out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + s * dst->nb[1]);
        double        sum = 0.0;
        for (int64_t i = 0; i < d; ++i) {
            sum += static_cast<double>(orw[i] * orw[i]);
        }
        const float mean  = static_cast<float>(sum / static_cast<double>(d));
        const float scale = 1.0f / sqrtf(mean + 1e-6f);
        for (int64_t i = 0; i < d; ++i) {
            out[i] = ur[i] + ((orw[i] * scale) * w[i]) * gate;
        }
    }
}

// x_tapped = sum_j taps[j] * raw(pos - j); writes raw into the ring at pos.
// pos_in (CPU fused path): the position is read from this input at compute
// time, so the graph does not depend on it; otherwise `pos` is baked in.
ggml_tensor * apply_taps(ggml_context * ctx,
                         ggml_cgraph *  gf,
                         ggml_tensor *  raw,
                         ggml_tensor *  ring,
                         ggml_tensor *  taps,
                         int            pos,
                         int            dil,
                         ggml_tensor *  pos_in) {
    if (pos_in != nullptr) {
        ggml_tensor * args[4] = { raw, ring, taps, pos_in };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, raw->ne[0], raw->ne[1], 1, 1, args, 4, taps_cpu, GGML_N_TASKS_MAX,
                              reinterpret_cast<void *>(static_cast<intptr_t>(dil)));
    }
    const int len = static_cast<int>(ring->ne[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, raw, ring_slot(ctx, ring, pos % len)));
    ggml_tensor * acc = ggml_mul(ctx, raw, tap_row(ctx, taps, 0));
    for (int j = 1; j < taps->ne[1]; ++j) {
        if (pos - j * dil < 0) {
            break;
        }
        acc = ggml_add(ctx, acc, ggml_mul(ctx, ring_slot(ctx, ring, (pos - j * dil) % len), tap_row(ctx, taps, j)));
    }
    return acc;
}

// u + gate * ZCRMS(o; w).
ggml_tensor * post_norm_residual(ggml_context * ctx,
                                 ggml_tensor *  u,
                                 ggml_tensor *  o,
                                 ggml_tensor *  w,
                                 const float &  gate,
                                 bool           cpu_fused) {
    if (cpu_fused) {
        ggml_tensor * args[3] = { u, o, w };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, u->ne[0], u->ne[1], 1, 1, args, 3, post_norm_residual_cpu,
                              GGML_N_TASKS_MAX, const_cast<float *>(&gate));
    }
    return ggml_add(ctx, u, ggml_scale(ctx, zcrms(ctx, o, w), gate));
}

}  // namespace

StepBuild build_step_graph(ggml_context *          ctx,
                           const WhistleWeights &  w,
                           const WhistleHParams &  hp,
                           const WhistleAux &      aux,
                           const WhistleDecCache & cache,
                           int                     pos,
                           bool                    want_cross_attn,
                           bool                    static_pos) {
    StepBuild sb{};
    sb.static_pos   = static_pos;
    const int64_t d = hp.d_model, qk = hp.qk_head_dim, vd = hp.v_head_dim, nh = hp.n_heads, nkv = hp.n_kv_heads;
    const int64_t S  = cache.n_seq;
    const int64_t T  = cache.T_enc;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    sb.graph         = gf;

    sb.tok_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(sb.tok_in, "dec.tok");
    ggml_set_input(sb.tok_in);
    sb.pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(sb.pos_in, "dec.pos");
    ggml_set_input(sb.pos_in);
    sb.eg_idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) hp.engram_n_tables * S);
    ggml_set_name(sb.eg_idx_in, "dec.engram.idx");
    ggml_set_input(sb.eg_idx_in);
    const bool eg_paired = !w.engrams.empty() && w.engrams[0].paired;
    if (eg_paired) {
        sb.eg_lo_in = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, (int64_t) hp.engram_n_tables * S);
        ggml_set_name(sb.eg_lo_in, "dec.engram.idx_lo");
        ggml_set_input(sb.eg_lo_in);
    }
    sb.eg_mask_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, hp.engram_n_tables, 1);
    ggml_set_name(sb.eg_mask_in, "dec.engram.mask");
    ggml_set_input(sb.eg_mask_in);
    // Cross scores span the cache's T4 rows (a multiple of 8: the generic score
    // product takes the tiled f32 GEMM, the fused CPU op reads whole vectors);
    // the mask hides the zero rows past each utterance's length.
    const int64_t T4 = cache.k_cross.empty() ? T : cache.k_cross[0]->ne[1];
    sb.xmask_in      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T4, 1, 1, S);
    ggml_set_name(sb.xmask_in, "dec.cross.mask");
    ggml_set_input(sb.xmask_in);
    ggml_tensor * pos_dyn = aux.cpu_fused_ops ? sb.pos_in : nullptr;

    // Hadamard-domain embedding rows: H_g is involutory, so H_g recovers them.
    ggml_tensor * x = ggml_scale(ctx, hada_in(ctx, ggml_get_rows(ctx, w.token_embd, sb.tok_in), w.hada),
                                 std::sqrt(static_cast<float>(d)));
    ggml_tensor * stream =
        ggml_repeat(ctx, ggml_reshape_3d(ctx, x, d, 1, S), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, hp.mhc_lanes, S));

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const bool  cpu        = aux.cpu_fused_ops;

    for (int l = 0; l < hp.dec_n_layers; ++l) {
        const WhistleDecBlock & b    = w.dec_blocks[l];
        int                     site = -1;
        for (size_t s = 0; s < hp.engram_sites.size(); ++s) {
            if (hp.engram_sites[s] == l) {
                site = static_cast<int>(s);
            }
        }
        MhcLayer ml;
        ml.mhc       = &w.dec_mhc;
        ml.pre_bias  = aux.dec_pre_bias;
        ml.post_bias = aux.dec_post_bias;
        ml.a_pre     = aux.dec_a_pre[l];
        ml.a_post    = aux.dec_a_post[l];
        ml.a_res     = aux.dec_a_res[l];
        ml.layer     = l;
        ml.cpu_fused = aux.cpu_fused_ops;
        ml.host      = aux.cpu_fused_ops ? &aux.dec_mhc_host[l] : nullptr;
        ml.hada      = &w.hada;
        ml.weights   = &w;

        stream = mhc_step(ctx, stream, ml, [&](ggml_tensor * u) {
            // ----- Engram -----
            // No hada_in here: Hadamard-domain table rows feed Hadamard-domain
            // key/value projections block for block (one 128-row = one input
            // group), and H_g H_g = I, so the two rotations cancel.
            if (site >= 0) {
                const WhistleEngram & eg = w.engrams[site];
                ggml_tensor *         e  = ggml_get_rows(ctx, eg.tables, sb.eg_idx_in);  // [sub, n_tables*S]
                if (eg_paired) {  // [2*sub, M] row pairs -> pick the half per row: [sub, 1, M]
                    e = ggml_get_rows(ctx, ggml_reshape_3d(ctx, e, hp.engram_sub_dim, 2, e->ne[1]), sb.eg_lo_in);
                }
                e                  = ggml_reshape_3d(ctx, e, hp.engram_sub_dim, hp.engram_n_tables, S);
                e                  = ggml_mul(ctx, e, sb.eg_mask_in);
                e                  = ggml_reshape_2d(ctx, e, (int64_t) hp.engram_sub_dim * hp.engram_n_tables, S);
                ggml_tensor * ek   = linear(ctx, w, eg.key_proj, e);
                ggml_tensor * vraw = linear(ctx, w, eg.value_proj, e);
                ggml_tensor * ev =
                    apply_taps(ctx, gf, vraw, cache.eg_val[site], eg.conv_taps, pos, hp.engram_conv_dilation, pos_dyn);
                ggml_tensor * dot =
                    ggml_sum_rows(ctx, ggml_mul(ctx, rms_norm(ctx, u, cpu), rms_norm(ctx, ek, cpu)));  // [1, S]
                ggml_tensor * alpha = ggml_sigmoid(ctx, ggml_scale(ctx, dot, 1.0f / std::sqrt(static_cast<float>(d))));
                u                   = ggml_add(ctx, u, ggml_mul(ctx, ev, alpha));
            }

            // ----- self-attention with causal taps -----
            {
                const WhistleAttn & a = b.attn;
                ggml_tensor *       pq, *pk, *pv, *pg;
                if (a.fused != nullptr) {  // q | k | v | gate in one matmul
                    ggml_tensor * y = linear(ctx, w, a.fused, hada_in(ctx, zcrms(ctx, u, b.norm_in, cpu), w.hada));
                    pq              = out_rows(ctx, y, 0, nh * qk);
                    pk              = out_rows(ctx, y, nh * qk, nkv * qk);
                    pv              = out_rows(ctx, y, (nh + nkv) * qk, nkv * vd);
                    pg              = out_rows(ctx, y, (nh + nkv) * qk + nkv * vd, nh * vd);
                } else {
                    ggml_tensor * xn = hada_in(ctx, zcrms(ctx, u, b.norm_in, cpu), w.hada);
                    pq               = linear(ctx, w, a.q, xn);
                    pk               = linear(ctx, w, a.k, xn);
                    pv               = linear(ctx, w, a.v, xn);
                    pg               = linear(ctx, w, a.gate, xn);
                }
                ggml_tensor * q = apply_taps(ctx, gf, pq, cache.tap_q[l], a.q_taps, pos, 1, pos_dyn);
                ggml_tensor * k = apply_taps(ctx, gf, pk, cache.tap_k[l], a.k_taps, pos, 1, pos_dyn);
                ggml_tensor * v = apply_taps(ctx, gf, pv, cache.tap_v[l], a.v_taps, pos, 1, pos_dyn);
                q               = zcrms(ctx, ggml_reshape_3d(ctx, q, qk, nh, S), a.q_norm, cpu);
                k               = zcrms(ctx, ggml_reshape_3d(ctx, k, qk, nkv, S), a.k_norm, cpu);
                q = ggml_rope_ext(ctx, q, sb.pos_in, nullptr, qk, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(ctx, k, sb.pos_in, nullptr, qk, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f);

                ggml_tensor * kc = cache.k_self[l];  // [qk, n_ctx, nkv, S]
                ggml_tensor * vc = cache.v_self[l];  // [n_ctx, vd, nkv, S]
                ggml_tensor * Q  = ggml_reshape_4d(ctx, q, qk, nh / nkv, nkv, S);
                ggml_tensor * o  = nullptr;
                if (static_pos) {
                    // KV write at pos_in + fused attention over [0, pos].
                    ggml_tensor * vv    = ggml_n_dims(v) == 2 ? v : ggml_reshape_2d(ctx, v, nkv * vd, S);
                    ggml_tensor * aa[7] = { Q, kc, vc, sb.pos_in, pg, k, vv };
                    o = ggml_custom_4d(ctx, GGML_TYPE_F32, vd, nh / nkv, nkv, S, aa, 7, self_attn_cpu, GGML_N_TASKS_MAX,
                                       nullptr);
                } else {
                    ggml_tensor * kw = ggml_view_4d(ctx, kc, qk, 1, nkv, S, kc->nb[1], kc->nb[2], kc->nb[3],
                                                    static_cast<size_t>(pos) * kc->nb[1]);
                    ggml_tensor * vw = ggml_view_4d(ctx, vc, 1, vd, nkv, S, vc->nb[1], vc->nb[2], vc->nb[3],
                                                    static_cast<size_t>(pos) * vc->nb[0]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_4d(ctx, k, qk, 1, nkv, S), kw));
                    ggml_build_forward_expand(
                        gf, ggml_cpy(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, v), 1, vd, nkv, S), vw));

                    // The nh / nkv query heads sharing a KV head form the columns of
                    // one product (head h -> KV head h / (nh / nkv), as ggml broadcasts),
                    // and the position contraction of probs . V is zero-padded to a
                    // multiple of 4 (zero probs over the zeroed V slack): both let
                    // ggml's tiled f32 GEMM run instead of per-element dot products.
                    const int64_t P4 = std::min<int64_t>((pos + 4) / 4 * 4, vc->ne[0]);
                    ggml_tensor * K  = ggml_view_4d(ctx, kc, qk, pos + 1, nkv, S, kc->nb[1], kc->nb[2], kc->nb[3], 0);
                    ggml_tensor * Vt = ggml_view_4d(ctx, vc, P4, vd, nkv, S, vc->nb[1], vc->nb[2], vc->nb[3], 0);
                    ggml_tensor * s  = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), nullptr, attn_scale, 0.0f);
                    if (P4 > pos + 1) {
                        s = ggml_pad(ctx, s, static_cast<int>(P4 - (pos + 1)), 0, 0, 0);
                    }
                    o = ggml_mul_mat(ctx, Vt, s);  // [vd, nh / nkv, nkv, S]
                }
                o = ggml_reshape_2d(ctx, o, vd * nh, S);
                if (!static_pos) {  // the fused op applies the gate
                    o = ggml_mul(ctx, o, ggml_sigmoid(ctx, pg));
                }
                o = linear(ctx, w, a.out, hada_in(ctx, o, w.hada));
                u = post_norm_residual(ctx, u, o, b.norm_post_attn, aux.dec_attn_gate[l], aux.cpu_fused_ops);
            }

            // ----- cross-attention -----
            {
                const WhistleAttn & c = b.cross;
                ggml_tensor *       pq, *pg;
                if (c.fused != nullptr) {  // q | gate in one matmul
                    ggml_tensor * y = linear(ctx, w, c.fused, hada_in(ctx, zcrms(ctx, u, b.norm_cross, cpu), w.hada));
                    pq              = out_rows(ctx, y, 0, nh * qk);
                    pg              = out_rows(ctx, y, nh * qk, nh * vd);
                } else {
                    ggml_tensor * xn = hada_in(ctx, zcrms(ctx, u, b.norm_cross, cpu), w.hada);
                    pq               = linear(ctx, w, c.q, xn);
                    pg               = linear(ctx, w, c.gate, xn);
                }
                // The CPU norm reads the strided q rows in place; the generic
                // path reshapes a contiguous copy.
                ggml_tensor * q = cpu ? zcrms(ctx, ggml_view_3d(ctx, pq, qk, nh, S, qk * sizeof(float), pq->nb[1], 0),
                                              c.q_norm, cpu) :
                                        zcrms(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, pq), qk, nh, S), c.q_norm, cpu);
                ggml_tensor * o = nullptr;
                if (static_pos) {
                    ggml_tensor * aa[5] = { q, cache.k_cross[l], cache.v_cross[l], sb.xmask_in, pg };
                    ggml_tensor * r = ggml_custom_4d(ctx, GGML_TYPE_F32, vd * nh + (want_cross_attn ? T * nh : 0), S, 1,
                                                     1, aa, 5, cross_attn_cpu, GGML_N_TASKS_MAX,
                                                     reinterpret_cast<void *>(static_cast<intptr_t>(T)));
                    o               = r;
                    if (want_cross_attn) {  // [T, beam, nh, n_utt] view of the probabilities
                        const int64_t nb = S / cache.n_utt;
                        ggml_tensor * pr =
                            ggml_cont(ctx, ggml_view_4d(ctx, r, T, nb, nh, cache.n_utt, r->nb[1], T * sizeof(float),
                                                        nb * r->nb[1], static_cast<size_t>(vd * nh) * sizeof(float)));
                        ggml_set_output(pr);
                        sb.cross_attn.push_back(pr);
                        o = ggml_cont(ctx, ggml_view_2d(ctx, r, vd * nh, S, r->nb[1], 0));
                    }
                } else {
                    // The beams of one utterance share its cross K/V: put them in
                    // the columns of one product per (head, utterance), and pad the
                    // T contraction of probs . V to the cache's multiple of 4.
                    const int64_t nb = S / cache.n_utt;
                    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, qk, nh, nb, cache.n_utt),
                                                                  0, 2, 1, 3));  // [qk, nb, nh, n_utt]
                    ggml_tensor * s = ggml_mul_mat(ctx, cache.k_cross[l], Q);    // [T4, nb, nh, n_utt]
                    s               = ggml_add(ctx, s, ggml_reshape_4d(ctx, sb.xmask_in, T4, nb, 1, cache.n_utt));
                    s               = ggml_soft_max_ext(ctx, s, nullptr, attn_scale, 0.0f);
                    if (want_cross_attn) {  // [T, beam, nh, n_utt]
                        ggml_tensor * sc = ggml_cont(ctx, ggml_view_4d(ctx, s, T, s->ne[1], s->ne[2], s->ne[3],
                                                                       s->nb[1], s->nb[2], s->nb[3], 0));
                        ggml_set_output(sc);
                        sb.cross_attn.push_back(sc);
                    }
                    ggml_tensor * vx = cache.v_cross[l];                                  // [T4, vd, nh, n_utt]
                    o                = ggml_mul_mat(ctx, vx, s);                          // [vd, nb, nh, n_utt]
                    o                = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [vd, nh, nb, n_utt]
                    o                = ggml_reshape_2d(ctx, o, vd * nh, S);
                    o                = ggml_mul(ctx, o, ggml_sigmoid(ctx, pg));  // the fused op applies the gate
                }
                o = linear(ctx, w, c.out, hada_in(ctx, o, w.hada));
                u = post_norm_residual(ctx, u, o, b.norm_post_cross, aux.dec_cross_gate[l], aux.cpu_fused_ops);
            }

            // ----- HadamardMLP -----
            u = ggml_add(ctx, u, hmlp(ctx, u, b.hmlp, aux, b.norm_hmlp));
            return u;
        });
    }

    ggml_tensor * h = hada_in(ctx, zcrms(ctx, lane_mean(ctx, stream), w.dec_final_norm, cpu), w.hada);
    if (w.head_rp != nullptr || w.head_q8 != nullptr) {  // padded repacked head: drop the zero rows
        ggml_tensor * full = w.head_rp != nullptr ? ggml_mul_mat(ctx, w.head_rp, h) : linear(ctx, w, w.head_q8, h);
        sb.logits          = ggml_cont(ctx, ggml_view_2d(ctx, full, hp.vocab_size, full->ne[1], full->nb[1], 0));
    } else {
        sb.logits = ggml_mul_mat(ctx, w.token_embd, h);  // [vocab, S]
    }
    ggml_set_name(sb.logits, "dec.logits");
    ggml_set_output(sb.logits);
    ggml_build_forward_expand(gf, sb.logits);
    for (ggml_tensor * t : sb.cross_attn) {
        ggml_build_forward_expand(gf, t);
    }
    return sb;
}

ReorderBuild build_reorder_graph(ggml_context * ctx, const WhistleDecCache & cache, int n_pos) {
    ReorderBuild  rb{};
    const int64_t S = cache.n_seq;
    rb.graph        = ggml_new_graph_custom(ctx, 1024, false);
    rb.src_in       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(rb.src_in, "dec.reorder.src");
    ggml_set_input(rb.src_in);
    const int64_t nkv = cache.k_self.empty() ? 1 : cache.k_self[0]->ne[2];
    const int64_t nvr = cache.v_self.empty() ? 1 : cache.v_self[0]->ne[1] * cache.v_self[0]->ne[2];
    rb.src_k          = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, S, nkv);
    ggml_set_name(rb.src_k, "dec.reorder.src_k");
    ggml_set_input(rb.src_k);
    rb.src_v = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, S, nvr);
    ggml_set_name(rb.src_v, "dec.reorder.src_v");
    ggml_set_input(rb.src_v);

    auto reorder = [&](ggml_tensor * t) {
        const int64_t slab = ggml_nelements(t) / S;
        ggml_tensor * t2   = ggml_reshape_2d(ctx, t, slab, S);
        ggml_build_forward_expand(rb.graph, ggml_cpy(ctx, ggml_get_rows(ctx, t2, rb.src_in), t2));
    };
    // Gather along the sequence axis of a 3-D view whose ne1 is the sequence
    // (rows of ne0 contiguous floats), then write back through the same view.
    auto reorder_view = [&](ggml_tensor * view, ggml_tensor * idx) {
        ggml_build_forward_expand(rb.graph, ggml_cpy(ctx, ggml_get_rows(ctx, view, idx), view));
    };
    for (size_t l = 0; l < cache.k_self.size(); ++l) {
        ggml_tensor * kc = cache.k_self[l];  // [qk, n_ctx, nkv, S]: per (head, seq) the prefix is qk * n_pos floats
        reorder_view(ggml_view_3d(ctx, kc, kc->ne[0] * n_pos, S, nkv, kc->nb[3], kc->nb[2], 0), rb.src_k);
        ggml_tensor * vc = cache.v_self[l];  // [n_ctx, vd, nkv, S]: per (row, seq) the prefix is n_pos floats
        reorder_view(ggml_view_3d(ctx, vc, n_pos, S, nvr, vc->nb[3], vc->nb[1], 0), rb.src_v);
        reorder(cache.tap_q[l]);
        reorder(cache.tap_k[l]);
        reorder(cache.tap_v[l]);
    }
    for (ggml_tensor * t : cache.eg_val) {
        reorder(t);
    }
    return rb;
}

}  // namespace transcribe::whistle
