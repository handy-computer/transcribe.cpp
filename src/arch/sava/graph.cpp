// arch/sava/graph.cpp - the step graph, the stream state it reads and writes,
// and its host inputs.
//
// A step graph runs one stream over a block of n_real frames, or one frame of
// each of a list of streams. Activations are ne [C, n], one column per frame;
// stream j owns columns [j k, j k + k), the first n_real of them real:
//
//   block  one stream, k = n = max(n_real, k_min_graph_frames); the padding columns
//          come last and, everything being causal, never feed the real ones
//   batch  k = n_real = 1, n = max(count, k_min_graph_frames); padding columns
//          repeat stream 0
//
// Weight matmuls, norms and element-wise ops run once over all n columns.
// Per-stream ops (the mixer row and speech carries, the Mamba-2 conv and
// scan, the ring writes and attention) run on that stream's state, the
// attention query padded to k_min_graph_frames columns, and its state writes read its
// last real column. Every matmul so has at least k_min_graph_frames columns and takes
// the same CPU kernel in the same reduction order, so every frame gets the
// same bits whatever the block size or the batch.
//
// State, in a backend buffer of the stream:
//   mixer_tail   the previous frame's mixer row
//   conv / ssm   per Mamba-2 block: the last K - 1 raw xBC columns and the SSM
//                state (ggml_ssm_scan keeps the state after each of the last
//                k - n_real + 1 steps; the write takes the last real one)
//   k/v ring     per attention layer: keys (normed, rotated) and values of
//                position p at slot p % ring, ring = window + k_block_frames,
//                so a block writes its keys before attending without
//                evicting one its queries read. A -inf mask keeps positions
//                [p - window + 1, p]; the slot order, hence the reduction
//                order, depends on p alone.
//   prev_speech  sigmoid of the last on-time speech logit

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "sava.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace transcribe::sava {

namespace {

constexpr const char * kTag = "sava";

// Frames per block graph; also the slack of every KV ring past its window.
constexpr int k_block_frames     = 64;
// ggml's CPU matmul takes tinyBLAS only from 4 columns on NEON (2 on x86),
// below that another kernel with a different reduction order.
constexpr int k_min_graph_frames = 4;

std::atomic<uint64_t> g_stream_uid{ 1 };

ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

ggml_tensor * dump_point(ggml_tensor * t) {
    if (debug::enabled()) {
        ggml_set_output(t);
    }
    return t;
}

// Columns [c, c + w) of t ne [C, n].
ggml_tensor * cols(ggml_context * ctx, ggml_tensor * t, int64_t c, int64_t w) {
    return ggml_view_2d(ctx, t, t->ne[0], w, t->nb[1], static_cast<size_t>(c) * t->nb[1]);
}

// parts concatenated along dim, padded to n with copies of parts[0].
ggml_tensor * cat_pad(ggml_context * ctx, const std::vector<ggml_tensor *> & parts, int64_t n, int dim) {
    ggml_tensor * r = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        r = ggml_concat(ctx, r, parts[i], dim);
    }
    while (r->ne[dim] < n) {
        r = ggml_concat(ctx, r, parts[0], dim);
    }
    return r;
}

// SwiGLU FFN with its pre-norm and residual.
ggml_tensor * build_mlp(ggml_context * ctx, ggml_tensor * x, const MlpWeights & w, float eps) {
    ggml_tensor * h = rms(ctx, x, w.norm, eps);
    ggml_tensor * g = ggml_silu(ctx, ggml_mul_mat(ctx, w.gate, h));
    ggml_tensor * u = ggml_mul_mat(ctx, w.up, h);
    return ggml_add(ctx, x, ggml_mul_mat(ctx, w.down, ggml_mul(ctx, g, u)));
}

// Rotate-half RoPE of x ne [hd, heads, n] with cos / sin ne [hd / 2, 1, n].
ggml_tensor * rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cs, ggml_tensor * sn) {
    const int64_t half = x->ne[0] / 2;
    ggml_tensor * x1   = ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    ggml_tensor * x2 =
        ggml_view_3d(ctx, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], static_cast<size_t>(half) * sizeof(float));
    ggml_tensor * r1 = ggml_sub(ctx, ggml_mul(ctx, x1, cs), ggml_mul(ctx, x2, sn));
    ggml_tensor * r2 = ggml_add(ctx, ggml_mul(ctx, x2, cs), ggml_mul(ctx, x1, sn));
    return ggml_concat(ctx, r1, r2, 0);
}

// Ring writes and attention of one stream. Keys k and values v ne [KV hd, nw]
// are written to slots (I64 [nw]); queries q ne [hd, H, nq] read the ring
// through mask ne [ring, nq]. Returns ne [hd, H, nq].
ggml_tensor * attend(ggml_context *      ctx,
                     const AttnState &   as,
                     ggml_tensor *       slots,
                     ggml_tensor *       mask,
                     ggml_tensor *       q,
                     ggml_tensor *       k,
                     ggml_tensor *       v,
                     const AttnHParams & a) {
    const int64_t hd = a.head_dim;
    const int64_t KV = a.n_kv_heads;
    const size_t  fs = sizeof(float);

    ggml_tensor * kr = ggml_set_rows(ctx, as.k_ring, k, slots);
    // One-element rows of the transposed V ring: element (slot, c) gets v[c].
    ggml_tensor * vt = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), 1, k->ne[1], hd * KV);
    ggml_tensor * vr = ggml_set_rows(ctx, ggml_reshape_3d(ctx, as.v_ring, 1, as.ring, hd * KV), vt, slots);

    // scores [ring, nq, H]; query head i reads key head i / (H / KV). The
    // query is made contiguous so this matmul takes tinyBLAS like the others.
    ggml_tensor * K = ggml_permute(ctx, ggml_view_3d(ctx, kr, hd, KV, as.ring, hd * fs, kr->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * P = ggml_mul_mat(ctx, K, Q);
    P               = ggml_soft_max_ext(ctx, P, mask, 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);

    ggml_tensor * V = ggml_reshape_3d(ctx, vr, as.ring, hd, KV);
    return ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, V, P), 0, 2, 1, 3));
}

// Exclusive self attention (remove alpha times each head's projection on its
// own frame's value) and the output projection. O ne [hd, H, n] and v ne
// [hd, KV, n], both contiguous.
ggml_tensor * attn_out(ggml_context *      ctx,
                       ggml_tensor *       O,
                       ggml_tensor *       v,
                       const AttnWeights & w,
                       const AttnHParams & a) {
    const int64_t n  = O->ne[2];
    const int64_t hd = a.head_dim;
    const int64_t H  = a.n_heads;
    const int64_t KV = a.n_kv_heads;
    if (a.xsa) {
        ggml_tensor * O4   = ggml_reshape_4d(ctx, O, hd, H / KV, KV, n);
        ggml_tensor * v4   = ggml_reshape_4d(ctx, v, hd, 1, KV, n);
        ggml_tensor * dot  = ggml_sum_rows(ctx, ggml_mul(ctx, O4, v4));  // [1, G, KV, n]
        ggml_tensor * nrm  = ggml_clamp(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, v4)), 1e-12f, FLT_MAX);
        ggml_tensor * coef = ggml_div(ctx, dot, nrm);
        if (a.xsa_alpha != 1.0f) {
            coef = ggml_scale(ctx, coef, a.xsa_alpha);
        }
        O = ggml_sub(ctx, O4, ggml_mul(ctx, ggml_repeat(ctx, v4, O4), coef));
    }
    return ggml_mul_mat(ctx, w.out, ggml_reshape_2d(ctx, O, hd * H, n));
}

// Builds a step graph's body (see the file comment) into g.t.
struct Builder {
    ggml_context *             ctx;
    StepGraph &                g;
    const HParams &            hp;
    Stream * const *           streams;
    int                        count;
    std::vector<ggml_tensor *> writes;  // state writes, expanded after every read

    // t ne [C, n] one frame late per stream: each stream's first column is
    // its state, which gets its last real column.
    ggml_tensor * carry(ggml_tensor * t, ggml_tensor * Stream::* state) {
        std::vector<ggml_tensor *> parts;
        for (int j = 0; j < count; ++j) {
            ggml_tensor * s = streams[j]->*state;
            ggml_tensor * p = ggml_reshape_2d(ctx, s, t->ne[0], 1);
            if (g.k > 1) {
                p = ggml_concat(ctx, p, cols(ctx, t, static_cast<int64_t>(j) * g.k, g.k - 1), 1);
            }
            parts.push_back(p);
            writes.push_back(ggml_cpy(ctx, cols(ctx, t, static_cast<int64_t>(j) * g.k + g.n_real - 1, 1), s));
        }
        return cat_pad(ctx, parts, g.n, 1);
    }

    // Mamba-2 sublayer of trunk block i (pre-norm to output projection).
    ggml_tensor * ssm(size_t i, ggml_tensor * x, const SsmWeights & w) {
        const int64_t n     = g.n;
        const int64_t k     = g.k;
        const int64_t inner = hp.ssm_inner;
        const int64_t cdim  = hp.ssm_conv_dim;
        const int64_t heads = hp.ssm_heads;
        const int64_t hd    = hp.ssm_head_dim;
        const int64_t N     = hp.ssm_state;
        const int64_t G     = hp.ssm_groups;
        const size_t  fs    = sizeof(float);

        ggml_tensor * u     = ggml_mul_mat(ctx, w.in, rms(ctx, x, w.norm, hp.rms_eps));  // [inner + cdim + heads, n]
        ggml_tensor * z     = ggml_cont(ctx, ggml_view_2d(ctx, u, inner, n, u->nb[1], 0));
        ggml_tensor * xbc_t = ggml_cont(ctx, ggml_transpose(ctx, ggml_view_2d(ctx, u, cdim, n, u->nb[1], inner * fs)));
        ggml_tensor * dt    = ggml_view_2d(ctx, u, heads, n, u->nb[1], (inner + cdim) * fs);

        std::vector<ggml_tensor *> ys;
        for (int j = 0; j < count; ++j) {
            const SsmState & ss = streams[j]->ssm[i];
            const int64_t    c0 = static_cast<int64_t>(j) * k;
            // Causal depthwise conv over [state (K - 1) | columns], time first.
            ggml_tensor *    sx = ggml_concat(
                ctx, ss.conv, ggml_view_2d(ctx, xbc_t, k, cdim, xbc_t->nb[1], static_cast<size_t>(c0) * fs), 0);
            ggml_tensor * c =
                ggml_silu(ctx, ggml_add(ctx, ggml_ssm_conv(ctx, sx, w.conv_w), w.conv_b));  // [cdim, k, 1]
            ggml_tensor * xs   = ggml_view_4d(ctx, c, hd, heads, k, 1, hd * fs, c->nb[1], c->nb[1] * k, 0);
            ggml_tensor * B    = ggml_view_4d(ctx, c, N, G, k, 1, N * fs, c->nb[1], c->nb[1] * k, inner * fs);
            ggml_tensor * C    = ggml_view_4d(ctx, c, N, G, k, 1, N * fs, c->nb[1], c->nb[1] * k, (inner + G * N) * fs);
            // dt's softplus is the scan's.
            ggml_tensor * dtb  = ggml_reshape_3d(ctx, ggml_add(ctx, cols(ctx, dt, c0, k), w.dt_b), heads, k, 1);
            ggml_tensor * scan = ggml_ssm_scan(ctx, ss.ssm, xs, dtb, w.a, B, C, g.t.ssm_ids, k - g.n_real + 1);
            ggml_tensor * y3   = ggml_view_3d(ctx, scan, hd, heads, k, hd * fs, inner * fs, 0);
            ggml_tensor * yd   = ggml_add(ctx, y3, ggml_mul(ctx, xs, ggml_reshape_2d(ctx, w.d, 1, heads)));
            ys.push_back(ggml_reshape_2d(ctx, yd, inner, k));
            if (j == 0) {
                g.t.ssm_y[i] = dump_point(ggml_view_2d(ctx, scan, inner, k, inner * fs, 0));
            }
            writes.push_back(
                ggml_cpy(ctx, ggml_view_2d(ctx, sx, hp.ssm_conv_kernel - 1, cdim, sx->nb[1], g.n_real * fs), ss.conv));
            const size_t  at    = static_cast<size_t>(inner * k) * fs + (k - g.n_real) * ggml_nbytes(ss.ssm);
            ggml_tensor * state = ggml_view_4d(ctx, scan, ss.ssm->ne[0], ss.ssm->ne[1], ss.ssm->ne[2], 1, ss.ssm->nb[1],
                                               ss.ssm->nb[2], ss.ssm->nb[3], at);
            writes.push_back(ggml_cpy(ctx, state, ss.ssm));
        }
        ggml_tensor * yg = ggml_mul(ctx, cat_pad(ctx, ys, n, 1), ggml_silu(ctx, z));
        yg               = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, yg, inner / G, G, n), hp.rms_eps);
        yg               = ggml_mul(ctx, ggml_reshape_2d(ctx, yg, inner, n), w.norm_gated);
        return ggml_mul_mat(ctx, w.out, yg);
    }

    // Attention sublayer (pre-norm to output projection) of attention layer li.
    ggml_tensor * attention(size_t li, ggml_tensor * x, const AttnWeights & w, const AttnHParams & a) {
        const int64_t n   = g.n;
        const int64_t k   = g.k;
        const int64_t hd  = a.head_dim;
        const int64_t H   = a.n_heads;
        const int64_t KV  = a.n_kv_heads;
        const float   eps = hp.rms_eps;
        const size_t  fs  = sizeof(float);

        ggml_tensor * qkv = ggml_mul_mat(ctx, w.qkv, rms(ctx, x, w.norm, eps));  // [(H + 2 KV) hd, n]
        ggml_tensor * q   = ggml_view_3d(ctx, qkv, hd, H, n, hd * fs, qkv->nb[1], 0);
        ggml_tensor * kk  = ggml_view_3d(ctx, qkv, hd, KV, n, hd * fs, qkv->nb[1], H * hd * fs);
        ggml_tensor * v   = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, KV, n, hd * fs, qkv->nb[1], (H + KV) * hd * fs));
        q                 = rope(ctx, rms(ctx, q, w.q_norm, eps), g.t.rope_cos, g.t.rope_sin);
        kk = ggml_reshape_2d(ctx, rope(ctx, rms(ctx, kk, w.k_norm, eps), g.t.rope_cos, g.t.rope_sin), hd * KV, n);
        ggml_tensor * v2 = ggml_reshape_2d(ctx, v, hd * KV, n);

        std::vector<ggml_tensor *> outs;
        for (int j = 0; j < count; ++j) {
            const int64_t c0 = static_cast<int64_t>(j) * k;
            ggml_tensor * qj = ggml_view_3d(ctx, q, hd, H, k, q->nb[1], q->nb[2], static_cast<size_t>(c0) * q->nb[2]);
            if (k < k_min_graph_frames) {
                qj = ggml_repeat_4d(ctx, qj, hd, H, k_min_graph_frames, 1);
            }
            ggml_tensor * O =
                attend(ctx, streams[j]->attn[li], g.t.slots[li][static_cast<size_t>(j)],
                       g.t.masks[li][static_cast<size_t>(j)], qj, cols(ctx, kk, c0, k), cols(ctx, v2, c0, k), a);
            outs.push_back(ggml_view_3d(ctx, O, hd, H, k, O->nb[1], O->nb[2], 0));
        }
        return attn_out(ctx, cat_pad(ctx, outs, n, 2), v, w, a);
    }
};

// The graph of count streams with n_real frames each (count 1 or n_real 1).
transcribe_status build_graph(StepGraph & g, const Model & m, Stream * const * streams, int count, int n_real) {
    const HParams & hp  = m.hparams;
    const Weights & w   = m.weights;
    const float     eps = hp.rms_eps;

    // Tensors (nodes, views, inputs) per layer, trunk and the two head
    // attentions: measured at most ~20 shared and ~28 per stream (an
    // attention layer), plus ~100 for the mixer and heads.
    const size_t n_layers   = w.trunk.size() + 2;
    const size_t graph_size = 512 + n_layers * (64 + 40 * static_cast<size_t>(count));
    if (g.ctx != nullptr) {
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }
    if (g.sched != nullptr && g.sched_size < graph_size) {
        safe_sched_free(g.sched);
        g.sched = nullptr;
    }
    g.key.clear();
    g.t         = GraphTensors{};
    g.n_real    = n_real;
    g.k         = count == 1 ? std::max(n_real, k_min_graph_frames) : 1;
    g.n         = count == 1 ? g.k : std::max(count, k_min_graph_frames);
    const int n = g.n;

    ggml_init_params ip{};
    ip.mem_size = ggml_graph_overhead_custom(graph_size, false) + graph_size * ggml_tensor_overhead();
    ip.no_alloc = true;
    g.ctx       = ggml_init(ip);
    if (g.ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_context * ctx   = g.ctx;
    GraphTensors & t     = g.t;
    auto           input = [](ggml_tensor * x) {
        ggml_set_input(x);
        return x;
    };
    t.mel_in            = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.mel_width, n));
    t.log_energy        = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n));
    t.ssm_ids           = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1));
    t.rope_cos          = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.attn.head_dim / 2, 1, n));
    t.rope_sin          = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.attn.head_dim / 2, 1, n));
    const size_t n_attn = streams[0]->attn.size();
    t.slots.assign(n_attn, {});
    t.masks.assign(n_attn, {});
    for (size_t li = 0; li < n_attn; ++li) {
        for (int j = 0; j < count; ++j) {
            const int64_t ring = streams[j]->attn[li].ring;
            t.slots[li].push_back(input(ggml_new_tensor_1d(ctx, GGML_TYPE_I64, g.k)));
            t.masks[li].push_back(
                input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ring, std::max(g.k, k_min_graph_frames))));
        }
    }
    Builder b{ ctx, g, hp, streams, count, {} };

    // ---- mixer: [row_{t-1} | row_t], row = [mel | log energy] ----
    ggml_tensor * row = ggml_concat(ctx, t.mel_in, t.log_energy, 0);  // [row_width, n]
    ggml_tensor * x   = ggml_concat(ctx, b.carry(row, &Stream::mixer_tail), row, 0);
    x                 = rms(ctx, ggml_mul_mat(ctx, w.mixer_proj, x), w.mixer_norm, eps);
    t.mixer           = dump_point(x);

    // ---- trunk ----
    t.trunk_mix.assign(w.trunk.size(), nullptr);
    t.trunk_out.assign(w.trunk.size(), nullptr);
    t.ssm_y.assign(w.trunk.size(), nullptr);
    size_t li = 0;
    for (size_t i = 0; i < w.trunk.size(); ++i) {
        const Block & blk = w.trunk[i];
        ggml_tensor * mix = blk.is_ssm ? b.ssm(i, x, blk.ssm) : b.attention(li++, x, blk.attn, hp.attn);
        t.trunk_mix[i]    = dump_point(mix);
        x                 = dump_point(build_mlp(ctx, ggml_add(ctx, x, mix), blk.mlp, eps));
        t.trunk_out[i]    = x;
    }
    ggml_tensor * hn = dump_point(rms(ctx, x, w.trunk_norm, eps));
    t.trunk_norm     = hn;

    // ---- VAD head ----
    ggml_tensor * v = ggml_mul_mat(ctx, w.vad_down, rms(ctx, hn, w.vad_norm_in, eps));
    v               = ggml_add(ctx, v, b.attention(li++, v, w.vad_attn, hp.vad_attn));
    v               = dump_point(rms(ctx, build_mlp(ctx, v, w.vad_mlp, eps), w.vad_norm_out, eps));
    t.vad_hidden    = v;
    t.logit         = ggml_add(ctx, ggml_mul_mat(ctx, w.vad_ro_w, v), w.vad_ro_b);  // [1, n]
    t.logit_la      = ggml_add(ctx, ggml_mul_mat(ctx, w.vad_la_w, v), w.vad_la_b);  // [1, n]

    // ---- turn head: speech_in = sigmoid(on-time logit of the step before) ----
    t.speech_in = dump_point(b.carry(ggml_sigmoid(ctx, t.logit), &Stream::prev_speech));
    ggml_tensor * emb =
        ggml_mul(ctx, ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, w.turn_emb_w, hp.vad_embed, 1), hp.vad_embed, n, 1, 1),
                 t.speech_in);
    t.turn_emb       = dump_point(ggml_add(ctx, emb, w.turn_emb_b));
    t.turn_in        = dump_point(ggml_concat(ctx, rms(ctx, hn, w.turn_norm_in, eps), t.turn_emb, 0));
    ggml_tensor * tt = ggml_mul_mat(ctx, w.turn_down, t.turn_in);
    tt               = ggml_add(ctx, tt, b.attention(li++, tt, w.turn_attn, hp.turn_attn));
    t.turn_hidden    = dump_point(rms(ctx, build_mlp(ctx, tt, w.turn_mlp, eps), w.turn_norm_out, eps));
    t.turn           = ggml_add(ctx, ggml_mul_mat(ctx, w.turn_ro_w, t.turn_hidden), w.turn_ro_b);  // [n_horizons, n]

    ggml_set_output(t.logit);
    ggml_set_output(t.logit_la);
    ggml_set_output(t.turn);
    t.graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(t.graph, t.turn);
    ggml_build_forward_expand(t.graph, t.logit_la);
    if (debug::enabled()) {
        for (ggml_tensor * d : { t.mixer, t.trunk_norm, t.vad_hidden, t.speech_in, t.turn_in, t.turn_emb }) {
            ggml_build_forward_expand(t.graph, d);
        }
        for (size_t i = 0; i < w.trunk.size(); ++i) {
            ggml_build_forward_expand(t.graph, t.trunk_mix[i]);
            ggml_build_forward_expand(t.graph, t.trunk_out[i]);
            if (t.ssm_y[i] != nullptr) {
                ggml_build_forward_expand(t.graph, t.ssm_y[i]);
            }
        }
    }
    for (ggml_tensor * wr : b.writes) {
        ggml_build_forward_expand(t.graph, wr);
    }

    if (g.sched == nullptr) {
        auto & backends = const_cast<std::vector<ggml_backend_t> &>(m.plan.scheduler_list);
        g.sched = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()), graph_size,
                                         /*parallel=*/false, /*op_offload=*/true);
        if (g.sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: scheduler allocation failed", kTag);
            return TRANSCRIBE_ERR_OOM;
        }
        g.sched_size = graph_size;
    }
    ggml_backend_sched_reset(g.sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(g.sched, t.graph, kTag)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: graph allocation failed (%d columns)", kTag, n);
        return TRANSCRIBE_ERR_OOM;
    }
    for (int j = 0; j < count; ++j) {
        g.key.push_back(streams[j]->uid);
    }
    return TRANSCRIBE_OK;
}

// Append the first n_real columns of t (ne [C, n]) to dst.
void append_cols(std::vector<float> & dst, const ggml_tensor * t, int n_real) {
    if (t == nullptr) {
        return;
    }
    const size_t n   = static_cast<size_t>(t->ne[0]) * static_cast<size_t>(n_real);
    const size_t off = dst.size();
    dst.resize(off + n);
    ggml_backend_tensor_get(t, dst.data() + off, 0, n * sizeof(float));
}

// Complex FFT of n real samples x[0], x[stride], ... into (re, im)[0, n),
// fp64: radix 2 down to an odd length, then a direct DFT. tw_re / tw_im hold
// exp(-2 pi i k / n0) for k < n0, and n0 / n is the twiddle stride at this
// level. scratch holds at least 2 n doubles.
void fft(const double * x, int n, int stride, const Model & m, int n0, double * re, double * im, double * scratch) {
    const double * tw_re = m.twiddle_re.data();
    const double * tw_im = m.twiddle_im.data();
    const int      step  = n0 / n;
    if (n % 2 != 0) {
        for (int k = 0; k < n; ++k) {
            double sr = 0.0;
            double si = 0.0;
            for (int j = 0; j < n; ++j) {
                const int    t = static_cast<int>((static_cast<int64_t>(k) * j) % n) * step;
                const double v = x[static_cast<size_t>(j) * static_cast<size_t>(stride)];
                sr += v * tw_re[t];
                si += v * tw_im[t];
            }
            re[k] = sr;
            im[k] = si;
        }
        return;
    }
    const int half = n / 2;
    double *  e_re = scratch;
    double *  e_im = scratch + half;
    double *  o_re = scratch + n;
    double *  o_im = scratch + n + half;
    // The halves' own recursions use re / im (free until the butterflies).
    fft(x, half, 2 * stride, m, n0, e_re, e_im, re);
    fft(x + stride, half, 2 * stride, m, n0, o_re, o_im, re);
    for (int k = 0; k < half; ++k) {
        const double wr = tw_re[k * step];
        const double wi = tw_im[k * step];
        const double tr = wr * o_re[k] - wi * o_im[k];
        const double ti = wr * o_im[k] + wi * o_re[k];
        re[k]           = e_re[k] + tr;
        im[k]           = e_im[k] + ti;
        re[k + half]    = e_re[k] - tr;
        im[k + half]    = e_im[k] - ti;
    }
}

// One frame's causal log-mel (hop-major, mel_width values) and log energy,
// advancing the stream's audio context.
void frame_features(Stream & st, const Model & m, const float * src, float * mel, float * log_energy) {
    const HParams & hp    = m.hparams;
    const size_t    nfft  = static_cast<size_t>(hp.n_fft);
    const size_t    hops  = static_cast<size_t>(hp.hops_per_frame);
    const size_t    frame = static_cast<size_t>(hp.frame_samples);
    const size_t    ctxn  = static_cast<size_t>(hp.context_samples);
    const size_t    bins  = static_cast<size_t>(hp.n_bins);
    const size_t    mels  = static_cast<size_t>(hp.n_mels);

    st.fe_buf.resize(ctxn + frame);
    st.fe_win.resize(nfft);
    st.fft_re.resize(nfft);
    st.fft_im.resize(nfft);
    st.fft_scratch.resize(4 * nfft);
    st.fe_power.resize(bins);
    std::memcpy(st.fe_buf.data(), st.context.data(), ctxn * sizeof(float));
    std::memcpy(st.fe_buf.data() + ctxn, src, frame * sizeof(float));
    // Window j ends where hop j ends: [ctx | frame][hop j : hop j + n_fft].
    for (size_t j = 0; j < hops; ++j) {
        const float * w0 = st.fe_buf.data() + j * static_cast<size_t>(hp.hop);
        for (size_t i = 0; i < nfft; ++i) {
            st.fe_win[i] = static_cast<double>(w0[i]) * m.window[i];
        }
        fft(st.fe_win.data(), hp.n_fft, 1, m, hp.n_fft, st.fft_re.data(), st.fft_im.data(), st.fft_scratch.data());
        for (size_t k = 0; k < bins; ++k) {
            st.fe_power[k] = st.fft_re[k] * st.fft_re[k] + st.fft_im[k] * st.fft_im[k];
        }
        float * out = mel + j * mels;
        for (size_t b = 0; b < mels; ++b) {
            const double * row = m.mel_bank.data() + b * bins;
            double         acc = 0.0;
            for (size_t k = 0; k < bins; ++k) {
                acc += row[k] * st.fe_power[k];
            }
            acc    = std::max(acc, static_cast<double>(hp.mel_floor));
            out[b] = static_cast<float>((std::log10(acc) + 4.0) / 4.0);
        }
    }
    std::memcpy(st.context.data(), st.fe_buf.data() + frame, ctxn * sizeof(float));
    double sum = 0.0;
    for (size_t i = 0; i < frame; ++i) {
        sum += static_cast<double>(src[i]) * static_cast<double>(src[i]);
    }
    const float r = static_cast<float>(std::sqrt(sum / static_cast<double>(frame)));
    *log_energy   = std::log(std::max(r, hp.log_energy_floor));
}

// RoPE cos / sin (head_dim / 2 values each) of position p: fp64 angles,
// rounded to fp32.
void rope_angles(const Model & m, int64_t p, float * cs, float * sn) {
    const size_t half = static_cast<size_t>(m.hparams.attn.head_dim / 2);
    for (size_t i = 0; i < half; ++i) {
        const double ang = static_cast<double>(p) * static_cast<double>(m.rope_freqs[i]);
        cs[i]            = static_cast<float>(std::cos(ang));
        sn[i]            = static_cast<float>(std::sin(ang));
    }
}

// The mask row of a query at position p over a ring of `ring` slots: 0 on the
// slots holding positions [p - window + 1, p], -inf elsewhere.
void mask_row(int64_t p, int64_t ring, int64_t window, float * row) {
    std::fill(row, row + ring, -INFINITY);
    for (int64_t dist = 0; dist < window && dist <= p; ++dist) {
        row[(p - dist) % ring] = 0.0f;
    }
}

}  // namespace

transcribe_status stream_reset(Stream & st, const Model & m) {
    const HParams & hp = m.hparams;
    if (st.state_buf == nullptr) {
        std::vector<const AttnHParams *> attn;
        for (bool is_ssm : hp.layer_is_ssm) {
            if (!is_ssm) {
                attn.push_back(&hp.attn);
            }
        }
        attn.push_back(&hp.vad_attn);
        attn.push_back(&hp.turn_attn);
        const size_t n_tensors = 2 + 2 * hp.layer_is_ssm.size() + 2 * attn.size();

        // A previous reset whose buffer allocation failed left its context.
        if (st.state_ctx != nullptr) {
            ggml_free(st.state_ctx);
            st.state_ctx = nullptr;
        }
        ggml_init_params ip{};
        ip.mem_size  = (n_tensors + 1) * ggml_tensor_overhead();
        ip.no_alloc  = true;
        st.state_ctx = ggml_init(ip);
        if (st.state_ctx == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
        st.mixer_tail  = ggml_new_tensor_1d(st.state_ctx, GGML_TYPE_F32, hp.row_width);
        st.prev_speech = ggml_new_tensor_1d(st.state_ctx, GGML_TYPE_F32, 1);
        st.ssm.assign(hp.layer_is_ssm.size(), SsmState{});
        for (size_t i = 0; i < hp.layer_is_ssm.size(); ++i) {
            if (!hp.layer_is_ssm[i]) {
                continue;
            }
            st.ssm[i].conv = ggml_new_tensor_2d(st.state_ctx, GGML_TYPE_F32, hp.ssm_conv_kernel - 1, hp.ssm_conv_dim);
            st.ssm[i].ssm =
                ggml_new_tensor_4d(st.state_ctx, GGML_TYPE_F32, hp.ssm_state, hp.ssm_head_dim, hp.ssm_heads, 1);
        }
        st.attn.assign(attn.size(), AttnState{});
        for (size_t i = 0; i < attn.size(); ++i) {
            const int64_t kv  = static_cast<int64_t>(attn[i]->n_kv_heads) * attn[i]->head_dim;
            st.attn[i].window = attn[i]->window;
            st.attn[i].ring   = attn[i]->window + k_block_frames;
            st.attn[i].k_ring = ggml_new_tensor_2d(st.state_ctx, GGML_TYPE_F32, kv, st.attn[i].ring);
            st.attn[i].v_ring = ggml_new_tensor_2d(st.state_ctx, GGML_TYPE_F32, st.attn[i].ring, kv);
        }
        st.state_buf = ggml_backend_alloc_ctx_tensors(st.state_ctx, m.plan.primary);
        if (st.state_buf == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: stream state allocation failed", kTag);
            return TRANSCRIBE_ERR_OOM;
        }
    }
    // Zero: the reference's fresh state, and finite ring slots (a masked
    // slot still enters the value product with weight 0).
    ggml_backend_buffer_clear(st.state_buf, 0);
    st.context.assign(static_cast<size_t>(hp.context_samples), 0.0f);
    st.pos = 0;
    return TRANSCRIBE_OK;
}

transcribe_status ensure_state(Stream & st, const Model & m) {
    return st.state_buf != nullptr ? TRANSCRIBE_OK : stream_reset(st, m);
}

transcribe_status step(StepGraph &           g,
                       const Model &         m,
                       Stream * const *      streams,
                       SessionCore * const * cores,
                       int                   count,
                       int                   n_real,
                       int                   n_threads,
                       const float * const * pcm,
                       float *               turn,
                       float *               speech) {
    const HParams & hp    = m.hparams;
    const int64_t   t0    = ggml_time_us();
    bool            match = g.n_real == n_real && g.key.size() == static_cast<size_t>(count);
    for (int j = 0; match && j < count; ++j) {
        match = g.key[static_cast<size_t>(j)] == streams[j]->uid;
    }
    if (!match) {
        if (auto rc = build_graph(g, m, streams, count, n_real); rc != TRANSCRIBE_OK) {
            g.release();
            return rc;
        }
    }
    configure_sched_n_threads(g.sched, n_threads);
    const GraphTensors & t    = g.t;
    const size_t         n    = static_cast<size_t>(g.n);
    const int            k    = g.k;
    const size_t         mw   = static_cast<size_t>(hp.mel_width);
    const size_t         half = static_cast<size_t>(hp.attn.head_dim / 2);

    // Host inputs. Stream j's frame f is column j k + f; padding columns
    // take no audio and stream 0's position.
    g.mel.assign(n * mw, 0.0f);
    g.log_energy.assign(n, 0.0f);
    g.rope.assign(2 * half * n, 0.0f);
    for (int j = 0; j < count; ++j) {
        for (int f = 0; f < n_real; ++f) {
            const size_t c = static_cast<size_t>(j * k + f);
            frame_features(*streams[j], m, pcm[j] + static_cast<size_t>(f) * static_cast<size_t>(hp.frame_samples),
                           g.mel.data() + c * mw, &g.log_energy[c]);
        }
    }
    for (size_t c = 0; c < n; ++c) {
        const bool    real = c < static_cast<size_t>(count * k);
        const int64_t p =
            real ? streams[c / static_cast<size_t>(k)]->pos + static_cast<int64_t>(c % k) : streams[0]->pos;
        rope_angles(m, p, g.rope.data() + c * half, g.rope.data() + (n + c) * half);
    }
    ggml_backend_tensor_set(t.mel_in, g.mel.data(), 0, ggml_nbytes(t.mel_in));
    ggml_backend_tensor_set(t.log_energy, g.log_energy.data(), 0, ggml_nbytes(t.log_energy));
    const int32_t zero = 0;
    ggml_backend_tensor_set(t.ssm_ids, &zero, 0, sizeof(zero));
    ggml_backend_tensor_set(t.rope_cos, g.rope.data(), 0, ggml_nbytes(t.rope_cos));
    ggml_backend_tensor_set(t.rope_sin, g.rope.data() + half * n, 0, ggml_nbytes(t.rope_sin));
    // Ring slots of the stream's k positions; the mask rows of the padded
    // query (past k) repeat its last position.
    const int rows = std::max(k, k_min_graph_frames);
    g.slots.resize(static_cast<size_t>(k));
    for (size_t li = 0; li < t.slots.size(); ++li) {
        for (int j = 0; j < count; ++j) {
            const AttnState & as = streams[j]->attn[li];
            const int64_t     p  = streams[j]->pos;
            g.mask.resize(static_cast<size_t>(as.ring) * static_cast<size_t>(rows));
            for (int f = 0; f < k; ++f) {
                g.slots[static_cast<size_t>(f)] = (p + f) % as.ring;
            }
            for (int r = 0; r < rows; ++r) {
                mask_row(p + std::min(r, k - 1), as.ring, as.window,
                         g.mask.data() + static_cast<size_t>(r) * static_cast<size_t>(as.ring));
            }
            ggml_backend_tensor_set(t.slots[li][static_cast<size_t>(j)], g.slots.data(), 0,
                                    g.slots.size() * sizeof(int64_t));
            ggml_backend_tensor_set(t.masks[li][static_cast<size_t>(j)], g.mask.data(), 0,
                                    g.mask.size() * sizeof(float));
        }
    }

    if (const ggml_status gs = ggml_backend_sched_graph_compute(g.sched, t.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: graph compute failed (%d)", kTag, static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    const size_t H = static_cast<size_t>(hp.n_horizons);
    g.logit.resize(n);
    g.logit_la.resize(n);
    g.turn.resize(n * H);
    ggml_backend_tensor_get(t.logit, g.logit.data(), 0, ggml_nbytes(t.logit));
    ggml_backend_tensor_get(t.logit_la, g.logit_la.data(), 0, ggml_nbytes(t.logit_la));
    ggml_backend_tensor_get(t.turn, g.turn.data(), 0, ggml_nbytes(t.turn));
    const int64_t dt = ggml_time_us() - t0;
    for (int j = 0; j < count; ++j) {
        Stream & s = *streams[j];
        for (int f = 0; f < n_real; ++f) {
            const size_t c = static_cast<size_t>(j * k + f);
            const size_t o = static_cast<size_t>(j * n_real + f);
            // The first lookahead_frames steps have no frame to judge yet: the
            // model emits no_speech_logit there (kept in g.logit_la for the trace).
            if (s.pos + f < hp.lookahead_frames) {
                g.logit_la[c] = hp.no_speech_logit;
            }
            if (speech != nullptr) {
                speech[o] = sigmoid(g.logit_la[c]);
            }
            if (turn != nullptr) {
                turn[o] = sigmoid(g.turn[c * H]);
            }
        }
        s.pos += n_real;
        cores[j]->t_encode_us += dt;
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_step(SessionCore & core,
                              Stream &      st,
                              const Model & m,
                              const float * pcm,
                              int64_t       n_frames,
                              float *       turn,
                              float *       speech) {
    const HParams & hp = m.hparams;
    debug::init();
    if (auto rc = ensure_state(st, m); rc != TRANSCRIBE_OK) {
        return rc;
    }
    Stream *      sp = &st;
    SessionCore * cp = &core;
    for (int64_t f0 = 0; f0 < n_frames; f0 += k_block_frames) {
        if (core.poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int     n_real = static_cast<int>(std::min<int64_t>(k_block_frames, n_frames - f0));
        const float * p      = pcm + f0 * hp.frame_samples;
        if (auto rc = step(st.g, m, &sp, &cp, 1, n_real, core.n_threads, &p, turn != nullptr ? turn + f0 : nullptr,
                           speech != nullptr ? speech + f0 : nullptr);
            rc != TRANSCRIBE_OK) {
            return rc;
        }
        if (debug::enabled()) {
            const StepGraph &    g  = st.g;
            const GraphTensors & t  = g.t;
            DumpTrace &          tr = st.trace;
            const size_t         H  = static_cast<size_t>(hp.n_horizons);
            const auto           nr = static_cast<ptrdiff_t>(n_real);
            tr.mel.insert(tr.mel.end(), g.mel.begin(), g.mel.begin() + nr * hp.mel_width);
            tr.log_energy.insert(tr.log_energy.end(), g.log_energy.begin(), g.log_energy.begin() + nr);
            tr.logit.insert(tr.logit.end(), g.logit.begin(), g.logit.begin() + nr);
            tr.logit_la.insert(tr.logit_la.end(), g.logit_la.begin(), g.logit_la.begin() + nr);
            tr.turn.insert(tr.turn.end(), g.turn.begin(), g.turn.begin() + nr * static_cast<ptrdiff_t>(H));
            append_cols(tr.mixer, t.mixer, n_real);
            tr.trunk_mix.resize(t.trunk_mix.size());
            tr.trunk_out.resize(t.trunk_out.size());
            tr.ssm_y.resize(t.ssm_y.size());
            for (size_t i = 0; i < t.trunk_mix.size(); ++i) {
                append_cols(tr.trunk_mix[i], t.trunk_mix[i], n_real);
                append_cols(tr.trunk_out[i], t.trunk_out[i], n_real);
                append_cols(tr.ssm_y[i], t.ssm_y[i], n_real);
            }
            append_cols(tr.trunk_norm, t.trunk_norm, n_real);
            append_cols(tr.vad_hidden, t.vad_hidden, n_real);
            append_cols(tr.speech_in, t.speech_in, n_real);
            append_cols(tr.turn_in, t.turn_in, n_real);
            append_cols(tr.turn_emb, t.turn_emb, n_real);
            append_cols(tr.turn_hidden, t.turn_hidden, n_real);
            tr.n_steps += n_real;
        }
    }
    return TRANSCRIBE_OK;
}

void DumpTrace::clear() {
    for (std::vector<float> * v : { &mel, &log_energy, &mixer, &trunk_norm, &vad_hidden, &speech_in, &turn_in,
                                    &turn_emb, &turn_hidden, &logit, &logit_la, &turn }) {
        v->clear();
    }
    trunk_mix.clear();
    trunk_out.clear();
    ssm_y.clear();
    n_steps = 0;
}

void StepGraph::release() noexcept {
    release_compute_scratch(sched, ctx);
    sched_size = 0;
    key.clear();
    n_real = 0;
    t      = GraphTensors{};
}

Stream::Stream() : uid(g_stream_uid.fetch_add(1)) {}

Stream::~Stream() {
    g.release();
    if (state_buf != nullptr) {
        safe_buffer_free(state_buf);
    }
    if (state_ctx != nullptr) {
        ggml_free(state_ctx);
    }
}

}  // namespace transcribe::sava
