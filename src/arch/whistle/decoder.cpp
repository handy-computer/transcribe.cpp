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

#include "encoder.h"
#include "ggml.h"
#include "whistle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

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
        ggml_tensor *       k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, c.k, mem), qk, nh, T_utt);
        k                     = zcrms(ctx, k, c.k_norm);
        k                     = ggml_permute(ctx, k, 0, 2, 1, 3);  // [qk, T, nh]
        ggml_tensor * kc      = cache.k_cross[l];
        ggml_tensor * kv =
            ggml_view_3d(ctx, kc, qk, T_utt, nh, kc->nb[1], kc->nb[2], static_cast<size_t>(utt) * kc->nb[3]);
        ggml_build_forward_expand(cb.graph, ggml_cpy(ctx, k, kv));

        ggml_tensor * v  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, c.v, mem), vd, nh, T_utt);
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

// Positions handed to the fused CPU ops through their userdata pointer (a
// static table, so the pointer outlives every graph).
constexpr int kMaxPos = 4096;

const int * pos_ptr(int pos) {
    static const auto table = [] {
        std::array<int, kMaxPos> t{};
        for (int i = 0; i < kMaxPos; ++i) {
            t[static_cast<size_t>(i)] = i;
        }
        return t;
    }();
    return &table[static_cast<size_t>(pos)];
}

inline void split_rows(int64_t total, int ith, int nth, int64_t & b, int64_t & e) {
    const int64_t per = (total + nth - 1) / nth;
    b                 = std::min(total, per * ith);
    e                 = std::min(total, b + per);
}

// Fused CPU apply_taps: same f32 operations in the same order as the graph
// below (raw * t0, then + ring(pos - j) * tj), and writes raw into the ring
// slot for pos. src: raw [dim, S], ring [dim, len, S], taps [dim, n_taps].
void taps_cpu(ggml_tensor * dst, int ith, int nth, void * ud) {
    const int           pos  = *static_cast<const int *>(ud);
    const ggml_tensor * raw  = dst->src[0];
    ggml_tensor *       ring = dst->src[1];
    const ggml_tensor * taps = dst->src[2];
    const int64_t       dim = raw->ne[0], S = raw->ne[1], len = ring->ne[1], nt = taps->ne[1];
    int64_t             b, e;
    split_rows(S, ith, nth, b, e);
    const float * t = static_cast<const float *>(taps->data);
    for (int64_t s = b; s < e; ++s) {
        const float * x   = reinterpret_cast<const float *>(static_cast<const char *>(raw->data) + s * raw->nb[1]);
        char *        rb  = static_cast<char *>(ring->data) + s * ring->nb[2];
        float *       out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + s * dst->nb[1]);
        for (int64_t i = 0; i < dim; ++i) {
            out[i] = x[i] * t[i];
        }
        for (int64_t j = 1; j < nt && pos - j >= 0; ++j) {
            const float * r  = reinterpret_cast<const float *>(rb + ((pos - j) % len) * ring->nb[1]);
            const float * tj = t + j * dim;
            for (int64_t i = 0; i < dim; ++i) {
                out[i] = out[i] + r[i] * tj[i];
            }
        }
        std::memcpy(rb + (pos % len) * ring->nb[1], x, static_cast<size_t>(dim) * sizeof(float));
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
ggml_tensor * apply_taps(ggml_context * ctx,
                         ggml_cgraph *  gf,
                         ggml_tensor *  raw,
                         ggml_tensor *  ring,
                         ggml_tensor *  taps,
                         int            pos,
                         bool           cpu_fused) {
    if (cpu_fused && pos < kMaxPos) {
        ggml_tensor * args[3] = { raw, ring, taps };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, raw->ne[0], raw->ne[1], 1, 1, args, 3, taps_cpu, GGML_N_TASKS_MAX,
                              const_cast<int *>(pos_ptr(pos)));
    }
    const int len = static_cast<int>(ring->ne[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, raw, ring_slot(ctx, ring, pos % len)));
    ggml_tensor * acc = ggml_mul(ctx, raw, tap_row(ctx, taps, 0));
    for (int j = 1; j < taps->ne[1]; ++j) {
        if (pos - j < 0) {
            break;
        }
        acc = ggml_add(ctx, acc, ggml_mul(ctx, ring_slot(ctx, ring, (pos - j) % len), tap_row(ctx, taps, j)));
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
                           bool                    want_cross_attn) {
    StepBuild     sb{};
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
    if (cache.n_utt > 1) {
        sb.xmask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, 1, 1, S);
        ggml_set_name(sb.xmask_in, "dec.cross.mask");
        ggml_set_input(sb.xmask_in);
    }

    // Hadamard-domain embedding rows: H_g is involutory, so H_g recovers them.
    ggml_tensor * x = ggml_scale(ctx, hada_in(ctx, ggml_get_rows(ctx, w.token_embd, sb.tok_in), w.hada),
                                 std::sqrt(static_cast<float>(d)));
    ggml_tensor * stream =
        ggml_repeat(ctx, ggml_reshape_3d(ctx, x, d, 1, S), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, hp.mhc_lanes, S));

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(qk));

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
                ggml_tensor * ek   = ggml_mul_mat(ctx, eg.key_proj, e);
                ggml_tensor * vraw = ggml_mul_mat(ctx, eg.value_proj, e);
                ggml_tensor * ring = cache.eg_val[site];
                const int     len  = static_cast<int>(ring->ne[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, vraw, ring_slot(ctx, ring, pos % len)));
                ggml_tensor * ev = ggml_mul(ctx, vraw, tap_row(ctx, eg.conv_taps, 0));
                for (int j = 1; j < eg.conv_taps->ne[1]; ++j) {
                    const int p = pos - j * hp.engram_conv_dilation;
                    if (p < 0) {
                        break;
                    }
                    ev = ggml_add(ctx, ev, ggml_mul(ctx, ring_slot(ctx, ring, p % len), tap_row(ctx, eg.conv_taps, j)));
                }
                ggml_tensor * dot = ggml_sum_rows(
                    ctx, ggml_mul(ctx, ggml_rms_norm(ctx, u, 1e-6f), ggml_rms_norm(ctx, ek, 1e-6f)));  // [1, S]
                ggml_tensor * alpha = ggml_sigmoid(ctx, ggml_scale(ctx, dot, 1.0f / std::sqrt(static_cast<float>(d))));
                u                   = ggml_add(ctx, u, ggml_mul(ctx, ev, alpha));
            }

            // ----- self-attention with causal taps -----
            {
                const WhistleAttn & a  = b.attn;
                ggml_tensor *       xn = hada_in(ctx, zcrms(ctx, u, b.norm_in), w.hada);
                ggml_tensor *       pq, *pk, *pv, *pg;
                if (a.fused != nullptr) {  // q | k | v | gate in one matmul
                    ggml_tensor * y = ggml_mul_mat(ctx, a.fused, xn);
                    pq              = out_rows(ctx, y, 0, nh * qk);
                    pk              = out_rows(ctx, y, nh * qk, nkv * qk);
                    pv              = out_rows(ctx, y, (nh + nkv) * qk, nkv * vd);
                    pg              = out_rows(ctx, y, (nh + nkv) * qk + nkv * vd, nh * vd);
                } else {
                    pq = ggml_mul_mat(ctx, a.q, xn);
                    pk = ggml_mul_mat(ctx, a.k, xn);
                    pv = ggml_mul_mat(ctx, a.v, xn);
                    pg = ggml_mul_mat(ctx, a.gate, xn);
                }
                ggml_tensor * q = apply_taps(ctx, gf, pq, cache.tap_q[l], a.q_taps, pos, aux.cpu_fused_ops);
                ggml_tensor * k = apply_taps(ctx, gf, pk, cache.tap_k[l], a.k_taps, pos, aux.cpu_fused_ops);
                ggml_tensor * v = apply_taps(ctx, gf, pv, cache.tap_v[l], a.v_taps, pos, aux.cpu_fused_ops);
                q               = zcrms(ctx, ggml_reshape_3d(ctx, q, qk, nh, S), a.q_norm);
                k               = zcrms(ctx, ggml_reshape_3d(ctx, k, qk, nkv, S), a.k_norm);
                q = ggml_rope_ext(ctx, q, sb.pos_in, nullptr, qk, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(ctx, k, sb.pos_in, nullptr, qk, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f);

                ggml_tensor * kc = cache.k_self[l];  // [qk, n_ctx, nkv, S]
                ggml_tensor * vc = cache.v_self[l];  // [n_ctx, vd, nkv, S]
                ggml_tensor * kw = ggml_view_4d(ctx, kc, qk, 1, nkv, S, kc->nb[1], kc->nb[2], kc->nb[3],
                                                static_cast<size_t>(pos) * kc->nb[1]);
                ggml_tensor * vw = ggml_view_4d(ctx, vc, 1, vd, nkv, S, vc->nb[1], vc->nb[2], vc->nb[3],
                                                static_cast<size_t>(pos) * vc->nb[0]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_4d(ctx, k, qk, 1, nkv, S), kw));
                ggml_build_forward_expand(gf,
                                          ggml_cpy(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, v), 1, vd, nkv, S), vw));

                // The nh / nkv query heads sharing a KV head form the columns of
                // one product (head h -> KV head h / (nh / nkv), as ggml broadcasts),
                // and the position contraction of probs . V is zero-padded to a
                // multiple of 4 (zero probs over the zeroed V slack): both let
                // ggml's tiled f32 GEMM run instead of per-element dot products.
                const int64_t P4 = std::min<int64_t>((pos + 4) / 4 * 4, vc->ne[0]);
                ggml_tensor * K  = ggml_view_4d(ctx, kc, qk, pos + 1, nkv, S, kc->nb[1], kc->nb[2], kc->nb[3], 0);
                ggml_tensor * Vt = ggml_view_4d(ctx, vc, P4, vd, nkv, S, vc->nb[1], vc->nb[2], vc->nb[3], 0);
                ggml_tensor * Q  = ggml_reshape_4d(ctx, q, qk, nh / nkv, nkv, S);
                ggml_tensor * s  = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), nullptr, attn_scale, 0.0f);
                if (P4 > pos + 1) {
                    s = ggml_pad(ctx, s, static_cast<int>(P4 - (pos + 1)), 0, 0, 0);
                }
                ggml_tensor * o = ggml_mul_mat(ctx, Vt, s);  // [vd, nh / nkv, nkv, S]
                o               = ggml_reshape_2d(ctx, o, vd * nh, S);
                o               = ggml_mul(ctx, o, ggml_sigmoid(ctx, pg));
                o               = ggml_mul_mat(ctx, a.out, hada_in(ctx, o, w.hada));
                u = post_norm_residual(ctx, u, o, b.norm_post_attn, aux.dec_attn_gate[l], aux.cpu_fused_ops);
            }

            // ----- cross-attention -----
            {
                const WhistleAttn & c  = b.cross;
                ggml_tensor *       xn = hada_in(ctx, zcrms(ctx, u, b.norm_cross), w.hada);
                ggml_tensor *       pq, *pg;
                if (c.fused != nullptr) {  // q | gate in one matmul
                    ggml_tensor * y = ggml_mul_mat(ctx, c.fused, xn);
                    pq              = ggml_cont(ctx, out_rows(ctx, y, 0, nh * qk));
                    pg              = out_rows(ctx, y, nh * qk, nh * vd);
                } else {
                    pq = ggml_mul_mat(ctx, c.q, xn);
                    pg = ggml_mul_mat(ctx, c.gate, xn);
                }
                ggml_tensor * q  = zcrms(ctx, ggml_reshape_3d(ctx, pq, qk, nh, S), c.q_norm);
                // The beams of one utterance share its cross K/V: put them in
                // the columns of one product per (head, utterance), and pad the
                // T contraction of probs . V to the cache's multiple of 4.
                const int64_t nb = S / cache.n_utt;
                ggml_tensor * Q  = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, qk, nh, nb, cache.n_utt), 0,
                                                               2, 1, 3));   // [qk, nb, nh, n_utt]
                ggml_tensor * s  = ggml_mul_mat(ctx, cache.k_cross[l], Q);  // [T, nb, nh, n_utt]
                if (sb.xmask_in != nullptr) {
                    s = ggml_add(ctx, s, ggml_reshape_4d(ctx, sb.xmask_in, T, nb, 1, cache.n_utt));
                }
                s = ggml_soft_max_ext(ctx, s, nullptr, attn_scale, 0.0f);
                if (want_cross_attn) {  // single hypothesis: [T, 1, nh, 1]
                    ggml_set_output(s);
                    sb.cross_attn.push_back(s);
                }
                ggml_tensor * vx = cache.v_cross[l];  // [T4, vd, nh, n_utt]
                if (vx->ne[0] > T) {
                    s = ggml_pad(ctx, s, static_cast<int>(vx->ne[0] - T), 0, 0, 0);
                }
                ggml_tensor * o = ggml_mul_mat(ctx, vx, s);                          // [vd, nb, nh, n_utt]
                o               = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [vd, nh, nb, n_utt]
                o               = ggml_reshape_2d(ctx, o, vd * nh, S);
                o               = ggml_mul(ctx, o, ggml_sigmoid(ctx, pg));
                o               = ggml_mul_mat(ctx, c.out, hada_in(ctx, o, w.hada));
                u = post_norm_residual(ctx, u, o, b.norm_post_cross, aux.dec_cross_gate[l], aux.cpu_fused_ops);
            }

            // ----- HadamardMLP -----
            u = ggml_add(ctx, u, hmlp(ctx, zcrms(ctx, u, b.norm_hmlp), b.hmlp, aux));
            return u;
        });
    }

    ggml_tensor * h = hada_in(ctx, zcrms(ctx, lane_mean(ctx, stream), w.dec_final_norm), w.hada);
    if (w.head_rp != nullptr) {  // padded repacked head: drop the zero rows
        ggml_tensor * full = ggml_mul_mat(ctx, w.head_rp, h);
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
