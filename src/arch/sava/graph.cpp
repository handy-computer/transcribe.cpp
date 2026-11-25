// arch/sava/graph.cpp - the step graph over a block of frames, the
// persistent stream state it reads and writes, and the host-side inputs.
//
// Layout: activations are ne [C, n], one column per frame. A block of n_real
// frames runs as a graph of n = max(n_real, k_min_graph_frames) columns; the
// padding columns come after the real ones and never feed back into them
// (everything is causal) or into the state (state writes read the last real
// column). The padding exists so ggml's CPU matmul always takes the same
// kernel: with every per-frame value computed by the same kernel in the same
// reduction order whatever n is, a stream is bit-identical to a whole run.
//
// State, in a backend buffer of the session (written by the graph):
//   mixer_tail   the previous frame's mixer row
//   conv / ssm   per Mamba-2 block: the last K - 1 raw xBC columns and the SSM
//                state. ggml_ssm_scan snapshots the state after each of the
//                last K steps, so the state after the last real column is
//                slot n - n_real.
//   k/v ring     per attention layer: keys (normed, rotated) and values of
//                position p at slot p % ring, ring = window + k_block_frames,
//                so a block's own keys are written before attention without
//                evicting a key any of its queries still reads. A query
//                reads the slots whose position is in [p - window + 1, p]
//                (an additive -inf mask); the slot order, hence the
//                reduction order, depends on p alone.
//   prev_speech  sigmoid of the last on-time speech logit (the turn head's
//                input at the next step)

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

// Frames per graph; also the slack of every KV ring past its window.
constexpr int k_block_frames     = 64;
// Smallest graph, in columns: ggml's CPU matmul takes tinyBLAS only from 4
// columns on NEON (2 on x86) and otherwise another kernel with a different
// reduction order.
constexpr int k_min_graph_frames = 4;
constexpr int k_graph_size       = 8192;

ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return w != nullptr ? ggml_mul(ctx, x, w) : x;
}

ggml_tensor * dump_point(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    if (debug::enabled()) {
        ggml_set_output(t);
    }
    return t;
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
    const int64_t hd   = x->ne[0];
    const int64_t half = hd / 2;
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
    const int64_t nw = k->ne[1];
    const size_t  fs = sizeof(float);

    ggml_tensor * kr = ggml_set_rows(ctx, as.k_ring, k, slots);
    // The V ring is stored transposed, ne [ring, KV hd], so the value product
    // reads it in place: one-element rows, element (slot, c) gets v[c].
    ggml_tensor * vt = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), 1, nw, hd * KV);
    ggml_tensor * vr = ggml_set_rows(ctx, ggml_reshape_3d(ctx, as.v_ring, 1, as.ring, hd * KV), vt, slots);

    // scores [ring, nq, H]; query head i reads key head i / (H / KV). The
    // query is made contiguous so this matmul takes tinyBLAS like the others.
    ggml_tensor * K = ggml_permute(ctx, ggml_view_3d(ctx, kr, hd, KV, as.ring, hd * fs, kr->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * P = ggml_mul_mat(ctx, K, Q);
    P               = ggml_soft_max_ext(ctx, P, mask, 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);

    ggml_tensor * V = ggml_reshape_3d(ctx, vr, as.ring, hd, KV);
    return ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, V, P), 0, 2, 1, 3));  // [hd, H, nq]
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
        ggml_tensor * proj = ggml_mul(ctx, ggml_repeat(ctx, v4, O4), coef);
        O                  = ggml_sub(ctx, O4, proj);
    }
    return ggml_mul_mat(ctx, w.out, ggml_reshape_2d(ctx, O, hd * H, n));
}

struct Qkv {
    ggml_tensor * q = nullptr;  // ne [hd, H, n], normed and rotated
    ggml_tensor * k = nullptr;  // ne [KV hd, n], normed and rotated
    ggml_tensor * v = nullptr;  // ne [hd, KV, n]
};

Qkv build_qkv(ggml_context *      ctx,
              ggml_tensor *       x,
              const AttnWeights & w,
              const AttnHParams & a,
              ggml_tensor *       rope_cos,
              ggml_tensor *       rope_sin,
              float               eps) {
    const int64_t n   = x->ne[1];
    const int64_t hd  = a.head_dim;
    const int64_t H   = a.n_heads;
    const int64_t KV  = a.n_kv_heads;
    const size_t  fs  = sizeof(float);
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.qkv, rms(ctx, x, w.norm, eps));  // [(H + 2 KV) hd, n]
    Qkv           r;
    r.q = ggml_view_3d(ctx, qkv, hd, H, n, hd * fs, qkv->nb[1], 0);
    r.k = ggml_view_3d(ctx, qkv, hd, KV, n, hd * fs, qkv->nb[1], H * hd * fs);
    r.v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, KV, n, hd * fs, qkv->nb[1], (H + KV) * hd * fs));
    r.q = rope(ctx, rms(ctx, r.q, w.q_norm, eps), rope_cos, rope_sin);
    r.k = ggml_reshape_2d(ctx, rope(ctx, rms(ctx, r.k, w.k_norm, eps), rope_cos, rope_sin), hd * KV, n);
    return r;
}

// The attention sublayer (pre-norm to output projection, no residual) of
// attention layer li over x ne [d, n].
ggml_tensor * build_attention(ggml_context *       ctx,
                              const Stream &       st,
                              const GraphTensors & g,
                              size_t               li,
                              ggml_tensor *        x,
                              const AttnWeights &  w,
                              const AttnHParams &  a,
                              float                eps) {
    const Qkv     t  = build_qkv(ctx, x, w, a, g.rope_cos, g.rope_sin, eps);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, t.v, static_cast<int64_t>(a.head_dim) * a.n_kv_heads, x->ne[1]);
    ggml_tensor * O  = attend(ctx, st.attn[li], g.slots[li], g.masks[li], t.q, t.k, v2, a);
    return attn_out(ctx, O, t.v, w, a);
}

struct SsmOut {
    ggml_tensor * out   = nullptr;  // [d, n], before the residual
    ggml_tensor * y     = nullptr;  // [inner, n], scan output before the D skip
    ggml_tensor * scan  = nullptr;
    ggml_tensor * sx    = nullptr;
    ggml_tensor * x_ssm = nullptr;
};

SsmOut build_ssm(ggml_context *       ctx,
                 const SsmState &     ss,
                 const GraphTensors & g,
                 ggml_tensor *        x,
                 const SsmWeights &   w,
                 const HParams &      hp,
                 int                  n_pad) {
    const int64_t n     = x->ne[1];
    const int64_t inner = hp.ssm_inner;
    const int64_t cdim  = hp.ssm_conv_dim;
    const int64_t heads = hp.ssm_heads;
    const int64_t hd    = hp.ssm_head_dim;
    const int64_t N     = hp.ssm_state;
    const int64_t G     = hp.ssm_groups;
    const size_t  fs    = sizeof(float);
    SsmOut        r;

    ggml_tensor * u   = ggml_mul_mat(ctx, w.in, rms(ctx, x, w.norm, hp.rms_eps));  // [inner + cdim + heads, n]
    ggml_tensor * z   = ggml_cont(ctx, ggml_view_2d(ctx, u, inner, n, u->nb[1], 0));
    ggml_tensor * xbc = ggml_view_2d(ctx, u, cdim, n, u->nb[1], inner * fs);
    ggml_tensor * dt  = ggml_view_2d(ctx, u, heads, n, u->nb[1], (inner + cdim) * fs);

    // Causal depthwise conv over [state (K - 1) | block], time first.
    r.sx            = ggml_concat(ctx, ss.conv, ggml_cont(ctx, ggml_transpose(ctx, xbc)), 0);       // [K - 1 + n, cdim]
    ggml_tensor * c = ggml_silu(ctx, ggml_add(ctx, ggml_ssm_conv(ctx, r.sx, w.conv_w), w.conv_b));  // [cdim, n, 1]

    ggml_tensor * xs  = ggml_view_4d(ctx, c, hd, heads, n, 1, hd * fs, c->nb[1], c->nb[1] * n, 0);
    ggml_tensor * B   = ggml_view_4d(ctx, c, N, G, n, 1, N * fs, c->nb[1], c->nb[1] * n, inner * fs);
    ggml_tensor * C   = ggml_view_4d(ctx, c, N, G, n, 1, N * fs, c->nb[1], c->nb[1] * n, (inner + G * N) * fs);
    ggml_tensor * dtb = ggml_reshape_3d(ctx, ggml_add(ctx, dt, w.dt_b), heads, n, 1);  // softplus is the scan's

    r.scan  = ggml_ssm_scan(ctx, ss.ssm, xs, dtb, w.a, B, C, g.ssm_ids, n_pad + 1);
    r.y     = ggml_view_2d(ctx, r.scan, inner, n, inner * fs, 0);
    r.x_ssm = xs;

    ggml_tensor * y3 = ggml_view_3d(ctx, r.scan, hd, heads, n, hd * fs, inner * fs, 0);
    ggml_tensor * yd = ggml_add(ctx, y3, ggml_mul(ctx, xs, ggml_reshape_2d(ctx, w.d, 1, heads)));
    ggml_tensor * yg = ggml_mul(ctx, ggml_reshape_2d(ctx, yd, inner, n), ggml_silu(ctx, z));
    yg               = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, yg, inner / G, G, n), hp.rms_eps);
    yg               = ggml_mul(ctx, ggml_reshape_2d(ctx, yg, inner, n), w.norm_gated);
    r.out            = ggml_mul_mat(ctx, w.out, yg);
    return r;
}

transcribe_status build_graph(SessionCore & core, Stream & st, const Model & m, int n, int n_real) {
    const HParams & hp  = m.hparams;
    const Weights & w   = m.weights;
    const float     eps = hp.rms_eps;
    const size_t    fs  = sizeof(float);

    if (core.compute_ctx != nullptr) {
        ggml_free(core.compute_ctx);
        core.compute_ctx = nullptr;
    }
    st.g          = GraphTensors{};
    st.graph_n    = 0;
    st.graph_real = 0;

    ggml_init_params ip{};
    ip.mem_size      = ggml_graph_overhead_custom(k_graph_size, false) + k_graph_size * ggml_tensor_overhead();
    ip.no_alloc      = true;
    core.compute_ctx = ggml_init(ip);
    if (core.compute_ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_context * ctx = core.compute_ctx;
    GraphTensors & g   = st.g;

    auto input = [&](ggml_tensor * t, const char * name) {
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    g.mel_in     = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.mel_width, n), "fe.mel");
    g.log_energy = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n), "fe.log_energy");
    g.ssm_ids    = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), "ssm.ids");
    g.rope_cos   = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.attn.head_dim / 2, 1, n), "rope.cos");
    g.rope_sin   = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.attn.head_dim / 2, 1, n), "rope.sin");
    for (size_t li = 0; li < st.attn.size(); ++li) {
        g.slots.push_back(input(ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n), "attn.slots"));
        g.masks.push_back(input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, st.attn[li].ring, n), "attn.mask"));
    }

    // ---- front end (host, fill_inputs): causal log-mel, hop-major per frame ----
    ggml_tensor * mel = g.mel_in;

    // ---- mixer: [row_{t-1} | row_t] ----
    ggml_tensor * row  = ggml_concat(ctx, mel, g.log_energy, 0);  // [row_width, n]
    ggml_tensor * prev = ggml_concat(ctx, ggml_reshape_2d(ctx, st.mixer_tail, hp.row_width, 1),
                                     ggml_view_2d(ctx, row, hp.row_width, n - 1, row->nb[1], 0), 1);
    ggml_tensor * x    = rms(ctx, ggml_mul_mat(ctx, w.mixer_proj, ggml_concat(ctx, prev, row, 0)), w.mixer_norm, eps);
    g.mel              = mel;
    g.mixer            = dump_point(x, "mixer.out");

    // ---- trunk ----
    std::vector<SsmOut> ssm_outs(w.trunk.size());
    size_t              li = 0;
    for (size_t i = 0; i < w.trunk.size(); ++i) {
        const Block & b   = w.trunk[i];
        ggml_tensor * mix = nullptr;
        if (b.is_ssm) {
            ssm_outs[i] = build_ssm(ctx, st.ssm[i], g, x, b.ssm, hp, n - n_real);
            mix         = ssm_outs[i].out;
            g.ssm_y.push_back(dump_point(ssm_outs[i].y, "trunk.ssm_y"));
        } else {
            mix = build_attention(ctx, st, g, li++, x, b.attn, hp.attn, eps);
            g.ssm_y.push_back(nullptr);
        }
        g.trunk_mix.push_back(dump_point(mix, "trunk.mix"));
        x = build_mlp(ctx, ggml_add(ctx, x, mix), b.mlp, eps);
        g.trunk_out.push_back(dump_point(x, "trunk.out"));
    }
    ggml_tensor * hn = dump_point(rms(ctx, x, w.trunk_norm, eps), "trunk.norm");
    g.trunk_norm     = hn;

    // ---- VAD head ----
    ggml_tensor * v = ggml_mul_mat(ctx, w.vad_down, rms(ctx, hn, w.vad_norm_in, eps));
    v               = ggml_add(ctx, v, build_attention(ctx, st, g, li++, v, w.vad_attn, hp.vad_attn, eps));
    v               = rms(ctx, build_mlp(ctx, v, w.vad_mlp, eps), w.vad_norm_out, eps);
    g.vad_hidden    = dump_point(v, "head.vad.hidden");
    g.logit         = ggml_add(ctx, ggml_mul_mat(ctx, w.vad_ro_w, v), w.vad_ro_b);  // [1, n]
    g.logit_la      = ggml_add(ctx, ggml_mul_mat(ctx, w.vad_la_w, v), w.vad_la_b);  // [1, n]

    // ---- turn head: speech_in = sigmoid(on-time logit of the step before) ----
    ggml_tensor * sp = ggml_sigmoid(ctx, g.logit);
    ggml_tensor * s_in =
        ggml_concat(ctx, ggml_reshape_2d(ctx, st.prev_speech, 1, 1), ggml_view_2d(ctx, sp, 1, n - 1, sp->nb[1], 0), 1);
    g.speech_in       = dump_point(s_in, "head.turn.speech_in");
    ggml_tensor * emb = ggml_mul(
        ctx, ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, w.turn_emb_w, hp.vad_embed, 1), hp.vad_embed, n, 1, 1), s_in);
    emb               = dump_point(ggml_add(ctx, emb, w.turn_emb_b), "head.turn.embed");
    g.turn_emb        = emb;
    ggml_tensor * tin = dump_point(ggml_concat(ctx, rms(ctx, hn, w.turn_norm_in, eps), emb, 0), "head.turn.input");
    g.turn_in         = tin;
    ggml_tensor * t   = ggml_mul_mat(ctx, w.turn_down, tin);
    t                 = ggml_add(ctx, t, build_attention(ctx, st, g, li++, t, w.turn_attn, hp.turn_attn, eps));
    t                 = rms(ctx, build_mlp(ctx, t, w.turn_mlp, eps), w.turn_norm_out, eps);
    g.turn_hidden     = dump_point(t, "head.turn.hidden");
    g.turn            = ggml_add(ctx, ggml_mul_mat(ctx, w.turn_ro_w, t), w.turn_ro_b);  // [n_horizons, n]

    ggml_set_output(g.logit);
    ggml_set_output(g.logit_la);
    ggml_set_output(g.turn);

    g.graph = ggml_new_graph_custom(ctx, k_graph_size, false);
    ggml_build_forward_expand(g.graph, g.turn);
    ggml_build_forward_expand(g.graph, g.logit_la);
    if (debug::enabled()) {
        for (ggml_tensor * d :
             { g.mixer, g.trunk_norm, g.vad_hidden, g.speech_in, g.turn_in, g.turn_emb, g.turn_hidden }) {
            ggml_build_forward_expand(g.graph, d);
        }
        for (size_t i = 0; i < w.trunk.size(); ++i) {
            ggml_build_forward_expand(g.graph, g.trunk_mix[i]);
            ggml_build_forward_expand(g.graph, g.trunk_out[i]);
            if (g.ssm_y[i] != nullptr) {
                ggml_build_forward_expand(g.graph, g.ssm_y[i]);
            }
        }
    }

    // ---- state writes, after every read of the state they replace ----
    const int64_t last = n_real - 1;
    ggml_build_forward_expand(g.graph, ggml_cpy(ctx, ggml_view_1d(ctx, sp, 1, last * sp->nb[1]), st.prev_speech));
    ggml_build_forward_expand(g.graph,
                              ggml_cpy(ctx, ggml_view_1d(ctx, row, hp.row_width, last * row->nb[1]), st.mixer_tail));
    for (size_t i = 0; i < w.trunk.size(); ++i) {
        if (!w.trunk[i].is_ssm) {
            continue;
        }
        const SsmOut &   so = ssm_outs[i];
        const SsmState & ss = st.ssm[i];
        ggml_tensor *    conv_tail =
            ggml_view_2d(ctx, so.sx, hp.ssm_conv_kernel - 1, hp.ssm_conv_dim, so.sx->nb[1], n_real * fs);
        ggml_build_forward_expand(g.graph, ggml_cpy(ctx, conv_tail, ss.conv));
        const size_t  y_bytes     = static_cast<size_t>(ggml_nelements(so.x_ssm)) * fs;
        const size_t  state_bytes = ggml_nbytes(ss.ssm);
        ggml_tensor * state =
            ggml_view_4d(ctx, so.scan, ss.ssm->ne[0], ss.ssm->ne[1], ss.ssm->ne[2], 1, ss.ssm->nb[1], ss.ssm->nb[2],
                         ss.ssm->nb[3], y_bytes + static_cast<size_t>(n - n_real) * state_bytes);
        ggml_build_forward_expand(g.graph, ggml_cpy(ctx, state, ss.ssm));
    }

    if (core.sched == nullptr) {
        auto & backends = const_cast<std::vector<ggml_backend_t> &>(m.plan.scheduler_list);
        core.sched = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()), k_graph_size,
                                            /*parallel=*/false, /*op_offload=*/true);
        if (core.sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: scheduler allocation failed", kTag);
            return TRANSCRIBE_ERR_OOM;
        }
    }
    ggml_backend_sched_reset(core.sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(core.sched, g.graph, kTag)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: graph allocation failed (%d frames)", kTag, n);
        return TRANSCRIBE_ERR_OOM;
    }
    st.graph_n    = n;
    st.graph_real = n_real;
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

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

// Host inputs for a block of n_real frames in a graph of n columns.
void fill_inputs(Stream & st, const Model & m, const float * pcm, int n, int n_real) {
    const HParams & hp = m.hparams;
    st.mel.assign(static_cast<size_t>(n) * static_cast<size_t>(hp.mel_width), 0.0f);
    st.log_energy.assign(static_cast<size_t>(n), 0.0f);
    for (int f = 0; f < n_real; ++f) {
        frame_features(st, m, pcm + static_cast<size_t>(f) * static_cast<size_t>(hp.frame_samples),
                       st.mel.data() + static_cast<size_t>(f) * static_cast<size_t>(hp.mel_width),
                       &st.log_energy[static_cast<size_t>(f)]);
    }
    const size_t half = static_cast<size_t>(hp.attn.head_dim / 2);
    st.rope.assign(2 * half * static_cast<size_t>(n), 0.0f);
    for (int j = 0; j < n; ++j) {
        rope_angles(m, st.pos + j, st.rope.data() + static_cast<size_t>(j) * half,
                    st.rope.data() + static_cast<size_t>(n + j) * half);
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

        // A previous reset whose buffer allocation failed left its context:
        // free it before building a new one.
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
    // No state buffer: the last stream_reset failed, so no step has run
    // since and the stream is still at its start. Retry the allocation.
    return st.state_buf != nullptr ? TRANSCRIBE_OK : stream_reset(st, m);
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
    for (int64_t f0 = 0; f0 < n_frames; f0 += k_block_frames) {
        if (core.poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int     n_real = static_cast<int>(std::min<int64_t>(k_block_frames, n_frames - f0));
        const int     n      = std::max(n_real, k_min_graph_frames);
        const int64_t t0     = ggml_time_us();

        if (st.graph_n != n || st.graph_real != n_real || core.compute_ctx == nullptr || st.g.graph == nullptr) {
            if (auto rc = build_graph(core, st, m, n, n_real); rc != TRANSCRIBE_OK) {
                return rc;
            }
        }
        configure_sched_n_threads(core.sched, core.n_threads);

        const int64_t p0 = st.pos;
        fill_inputs(st, m, pcm + f0 * hp.frame_samples, n, n_real);
        const GraphTensors & g = st.g;
        ggml_backend_tensor_set(g.mel_in, st.mel.data(), 0, ggml_nbytes(g.mel_in));
        ggml_backend_tensor_set(g.log_energy, st.log_energy.data(), 0, ggml_nbytes(g.log_energy));
        const int32_t zero = 0;
        ggml_backend_tensor_set(g.ssm_ids, &zero, 0, sizeof(zero));
        const size_t half_bytes = ggml_nbytes(g.rope_cos);
        ggml_backend_tensor_set(g.rope_cos, st.rope.data(), 0, half_bytes);
        ggml_backend_tensor_set(g.rope_sin, st.rope.data() + half_bytes / sizeof(float), 0, half_bytes);
        for (size_t li = 0; li < st.attn.size(); ++li) {
            const int64_t ring   = st.attn[li].ring;
            const int64_t window = st.attn[li].window;
            st.slots.resize(static_cast<size_t>(n));
            st.mask.resize(static_cast<size_t>(ring) * static_cast<size_t>(n));
            for (int j = 0; j < n; ++j) {
                const int64_t p                  = p0 + j;
                st.slots[static_cast<size_t>(j)] = p % ring;
                mask_row(p, ring, window, st.mask.data() + static_cast<size_t>(j) * static_cast<size_t>(ring));
            }
            ggml_backend_tensor_set(g.slots[li], st.slots.data(), 0, ggml_nbytes(g.slots[li]));
            ggml_backend_tensor_set(g.masks[li], st.mask.data(), 0, ggml_nbytes(g.masks[li]));
        }

        if (const ggml_status gs = ggml_backend_sched_graph_compute(core.sched, g.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: graph compute failed (%d)", kTag, static_cast<int>(gs));
            return TRANSCRIBE_ERR_BACKEND;
        }

        const size_t nr = static_cast<size_t>(n_real);
        st.logit.resize(nr);
        st.logit_la.resize(nr);
        st.turn.resize(nr * static_cast<size_t>(hp.n_horizons));
        ggml_backend_tensor_get(g.logit, st.logit.data(), 0, nr * sizeof(float));
        ggml_backend_tensor_get(g.logit_la, st.logit_la.data(), 0, nr * sizeof(float));
        ggml_backend_tensor_get(g.turn, st.turn.data(), 0, st.turn.size() * sizeof(float));
        // The first lookahead_frames steps have no frame to judge yet: the
        // model emits no_speech_logit there. The substituted value is written
        // back into st.logit_la so the trace's head.vad.logit_la matches the
        // reference dumper's (the model's emitted lookahead logit).
        for (size_t j = 0; j < nr; ++j) {
            const bool  early = p0 + static_cast<int64_t>(j) < hp.lookahead_frames;
            const float la    = early ? hp.no_speech_logit : st.logit_la[j];
            st.logit_la[j]    = la;
            if (speech != nullptr) {
                speech[f0 + static_cast<int64_t>(j)] = sigmoid(la);
            }
            if (turn != nullptr) {
                turn[f0 + static_cast<int64_t>(j)] = sigmoid(st.turn[j * static_cast<size_t>(hp.n_horizons)]);
            }
        }

        if (debug::enabled()) {
            DumpTrace & tr = st.trace;
            tr.mel.insert(tr.mel.end(), st.mel.begin(), st.mel.begin() + static_cast<ptrdiff_t>(n_real) * hp.mel_width);
            tr.log_energy.insert(tr.log_energy.end(), st.log_energy.begin(), st.log_energy.begin() + n_real);
            append_cols(tr.mixer, g.mixer, n_real);
            tr.trunk_mix.resize(g.trunk_mix.size());
            tr.trunk_out.resize(g.trunk_out.size());
            tr.ssm_y.resize(g.ssm_y.size());
            for (size_t i = 0; i < g.trunk_mix.size(); ++i) {
                append_cols(tr.trunk_mix[i], g.trunk_mix[i], n_real);
                append_cols(tr.trunk_out[i], g.trunk_out[i], n_real);
                append_cols(tr.ssm_y[i], g.ssm_y[i], n_real);
            }
            append_cols(tr.trunk_norm, g.trunk_norm, n_real);
            append_cols(tr.vad_hidden, g.vad_hidden, n_real);
            append_cols(tr.speech_in, g.speech_in, n_real);
            append_cols(tr.turn_in, g.turn_in, n_real);
            append_cols(tr.turn_emb, g.turn_emb, n_real);
            append_cols(tr.turn_hidden, g.turn_hidden, n_real);
            tr.logit.insert(tr.logit.end(), st.logit.begin(), st.logit.end());
            tr.logit_la.insert(tr.logit_la.end(), st.logit_la.begin(), st.logit_la.end());
            tr.turn.insert(tr.turn.end(), st.turn.begin(), st.turn.end());
            tr.n_steps += n_real;
        }

        st.pos += n_real;
        core.t_encode_us += ggml_time_us() - t0;
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

Stream::~Stream() {
    if (state_buf != nullptr) {
        safe_buffer_free(state_buf);
    }
    if (state_ctx != nullptr) {
        ggml_free(state_ctx);
    }
}

// ===========================================================================
// Batched steps: one frame of each of count streams in one graph.
//
// Columns are streams. The weight matmuls, norms and element-wise ops run once
// over n = max(count, k_min_graph_frames) columns; padding columns repeat
// stream 0 and write no state. The per-stream ops run on that stream's own
// state, each in the shape a 1-frame stream_step uses: the Mamba-2 conv and
// scan on 1 token, the KV ring write, and attention with the query repeated
// to k_min_graph_frames columns so K.Q and the value product take the same
// kernel. Every column therefore gets bit for bit the values stream_step
// gives its stream.
// ===========================================================================

namespace {

std::atomic<uint64_t> g_stream_uid{ 1 };

// Concatenate [C, 1]-shaped (or [hd, H, 1]) pieces along dim, padding with
// copies of the first up to n pieces.
ggml_tensor * cat_pad(ggml_context * ctx, std::vector<ggml_tensor *> parts, int n, int dim) {
    while (static_cast<int>(parts.size()) < n) {
        parts.push_back(parts[0]);
    }
    ggml_tensor * r = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        r = ggml_concat(ctx, r, parts[i], dim);
    }
    return r;
}

// Column j of a [C, n] tensor, as [C, 1].
ggml_tensor * column(ggml_context * ctx, ggml_tensor * t, int j) {
    return ggml_view_2d(ctx, t, t->ne[0], 1, t->nb[1], static_cast<size_t>(j) * t->nb[1]);
}

ggml_tensor * batch_attention(ggml_context *      ctx,
                              const BatchGraph &  bg,
                              Stream * const *    streams,
                              int                 count,
                              size_t              li,
                              ggml_tensor *       x,
                              const AttnWeights & w,
                              const AttnHParams & a,
                              float               eps) {
    const int64_t n  = x->ne[1];
    const int64_t hd = a.head_dim;
    const int64_t H  = a.n_heads;
    const int64_t KV = a.n_kv_heads;
    const Qkv     t  = build_qkv(ctx, x, w, a, bg.rope_cos, bg.rope_sin, eps);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, t.v, hd * KV, n);

    std::vector<ggml_tensor *> outs;
    for (int j = 0; j < count; ++j) {
        const size_t  js = static_cast<size_t>(j);
        ggml_tensor * qj = ggml_view_3d(ctx, t.q, hd, H, 1, t.q->nb[1], t.q->nb[2], js * t.q->nb[2]);
        qj               = ggml_repeat_4d(ctx, qj, hd, H, k_min_graph_frames, 1);
        ggml_tensor * O = attend(ctx, streams[j]->attn[li], bg.slots[li][js], bg.masks[li][js], qj, column(ctx, t.k, j),
                                 column(ctx, v2, j), a);
        outs.push_back(ggml_view_3d(ctx, O, hd, H, 1, O->nb[1], O->nb[2], 0));
    }
    return attn_out(ctx, cat_pad(ctx, outs, static_cast<int>(n), 2), t.v, w, a);
}

transcribe_status build_batch_graph(BatchGraph & bg, const Model & m, Stream * const * streams, int count) {
    const HParams & hp  = m.hparams;
    const Weights & w   = m.weights;
    const float     eps = hp.rms_eps;
    const size_t    fs  = sizeof(float);
    const int       n   = std::max(count, k_min_graph_frames);
    const int       K   = hp.ssm_conv_kernel;

    bg.release();
    // About 130 nodes per stream (per-stream conv, scan, ring writes and
    // attention) on top of the shared ~400.
    const size_t     graph_size = 1024 + 256 * static_cast<size_t>(count);
    ggml_init_params ip{};
    ip.mem_size = ggml_graph_overhead_custom(graph_size, false) + graph_size * ggml_tensor_overhead();
    ip.no_alloc = true;
    bg.ctx      = ggml_init(ip);
    if (bg.ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_context * ctx   = bg.ctx;
    auto           input = [](ggml_tensor * t) {
        ggml_set_input(t);
        return t;
    };
    bg.mel_in           = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.mel_width, n));
    bg.log_energy       = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n));
    bg.ssm_ids          = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1));
    bg.rope_cos         = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.attn.head_dim / 2, 1, n));
    bg.rope_sin         = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.attn.head_dim / 2, 1, n));
    const size_t n_attn = streams[0]->attn.size();
    bg.slots.assign(n_attn, {});
    bg.masks.assign(n_attn, {});
    for (size_t li = 0; li < n_attn; ++li) {
        for (int j = 0; j < count; ++j) {
            bg.slots[li].push_back(input(ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1)));
            bg.masks[li].push_back(
                input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, streams[j]->attn[li].ring, k_min_graph_frames)));
        }
    }
    std::vector<ggml_tensor *> writes;  // state writes, expanded after every read

    // ---- mixer ----
    ggml_tensor *              row = ggml_concat(ctx, bg.mel_in, bg.log_energy, 0);  // [row_width, n]
    std::vector<ggml_tensor *> tails;
    for (int j = 0; j < count; ++j) {
        tails.push_back(ggml_reshape_2d(ctx, streams[j]->mixer_tail, hp.row_width, 1));
        writes.push_back(ggml_cpy(ctx, column(ctx, row, j), tails.back()));
    }
    ggml_tensor * prev = cat_pad(ctx, tails, n, 1);
    ggml_tensor * x    = rms(ctx, ggml_mul_mat(ctx, w.mixer_proj, ggml_concat(ctx, prev, row, 0)), w.mixer_norm, eps);

    // ---- trunk ----
    size_t li = 0;
    for (size_t i = 0; i < w.trunk.size(); ++i) {
        const Block & b   = w.trunk[i];
        ggml_tensor * mix = nullptr;
        if (b.is_ssm) {
            const SsmWeights & sw    = b.ssm;
            const int64_t      inner = hp.ssm_inner;
            const int64_t      cdim  = hp.ssm_conv_dim;
            const int64_t      heads = hp.ssm_heads;
            const int64_t      hd    = hp.ssm_head_dim;
            const int64_t      N     = hp.ssm_state;
            const int64_t      G     = hp.ssm_groups;
            ggml_tensor *      u     = ggml_mul_mat(ctx, sw.in, rms(ctx, x, sw.norm, eps));
            ggml_tensor *      z     = ggml_cont(ctx, ggml_view_2d(ctx, u, inner, n, u->nb[1], 0));
            ggml_tensor *      xbc_t =
                ggml_cont(ctx, ggml_transpose(ctx, ggml_view_2d(ctx, u, cdim, n, u->nb[1], inner * fs)));
            std::vector<ggml_tensor *> ys;
            for (int j = 0; j < count; ++j) {
                const SsmState & ss = streams[j]->ssm[i];
                const size_t     js = static_cast<size_t>(j);
                ggml_tensor *    sx =
                    ggml_concat(ctx, ss.conv, ggml_view_2d(ctx, xbc_t, 1, cdim, xbc_t->nb[1], js * fs), 0);
                ggml_tensor * c    = ggml_silu(ctx, ggml_add(ctx, ggml_ssm_conv(ctx, sx, sw.conv_w), sw.conv_b));
                ggml_tensor * xs   = ggml_view_4d(ctx, c, hd, heads, 1, 1, hd * fs, c->nb[1], c->nb[1], 0);
                ggml_tensor * B    = ggml_view_4d(ctx, c, N, G, 1, 1, N * fs, c->nb[1], c->nb[1], inner * fs);
                ggml_tensor * C    = ggml_view_4d(ctx, c, N, G, 1, 1, N * fs, c->nb[1], c->nb[1], (inner + G * N) * fs);
                ggml_tensor * dt   = ggml_view_2d(ctx, u, heads, 1, u->nb[1], (inner + cdim) * fs + js * u->nb[1]);
                ggml_tensor * dtb  = ggml_reshape_3d(ctx, ggml_add(ctx, dt, sw.dt_b), heads, 1, 1);
                ggml_tensor * scan = ggml_ssm_scan(ctx, ss.ssm, xs, dtb, sw.a, B, C, bg.ssm_ids, 1);
                ggml_tensor * y3   = ggml_view_3d(ctx, scan, hd, heads, 1, hd * fs, inner * fs, 0);
                ggml_tensor * yd   = ggml_add(ctx, y3, ggml_mul(ctx, xs, ggml_reshape_2d(ctx, sw.d, 1, heads)));
                ys.push_back(ggml_reshape_2d(ctx, yd, inner, 1));
                writes.push_back(ggml_cpy(ctx, ggml_view_2d(ctx, sx, K - 1, cdim, sx->nb[1], fs), ss.conv));
                ggml_tensor * state =
                    ggml_view_4d(ctx, scan, ss.ssm->ne[0], ss.ssm->ne[1], ss.ssm->ne[2], 1, ss.ssm->nb[1],
                                 ss.ssm->nb[2], ss.ssm->nb[3], static_cast<size_t>(inner) * fs);
                writes.push_back(ggml_cpy(ctx, state, ss.ssm));
            }
            ggml_tensor * yg = ggml_mul(ctx, cat_pad(ctx, ys, n, 1), ggml_silu(ctx, z));
            yg               = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, yg, inner / G, G, n), eps);
            yg               = ggml_mul(ctx, ggml_reshape_2d(ctx, yg, inner, n), sw.norm_gated);
            mix              = ggml_mul_mat(ctx, sw.out, yg);
        } else {
            mix = batch_attention(ctx, bg, streams, count, li++, x, b.attn, hp.attn, eps);
        }
        x = build_mlp(ctx, ggml_add(ctx, x, mix), b.mlp, eps);
    }
    ggml_tensor * hn = rms(ctx, x, w.trunk_norm, eps);

    // ---- VAD head ----
    ggml_tensor * v = ggml_mul_mat(ctx, w.vad_down, rms(ctx, hn, w.vad_norm_in, eps));
    v               = ggml_add(ctx, v, batch_attention(ctx, bg, streams, count, li++, v, w.vad_attn, hp.vad_attn, eps));
    v               = rms(ctx, build_mlp(ctx, v, w.vad_mlp, eps), w.vad_norm_out, eps);
    bg.logit        = ggml_add(ctx, ggml_mul_mat(ctx, w.vad_ro_w, v), w.vad_ro_b);
    bg.logit_la     = ggml_add(ctx, ggml_mul_mat(ctx, w.vad_la_w, v), w.vad_la_b);

    // ---- turn head ----
    ggml_tensor *              sp = ggml_sigmoid(ctx, bg.logit);
    std::vector<ggml_tensor *> prevs;
    for (int j = 0; j < count; ++j) {
        prevs.push_back(ggml_reshape_2d(ctx, streams[j]->prev_speech, 1, 1));
        writes.push_back(
            ggml_cpy(ctx, ggml_view_1d(ctx, sp, 1, static_cast<size_t>(j) * sp->nb[1]), streams[j]->prev_speech));
    }
    ggml_tensor * s_in = cat_pad(ctx, prevs, n, 1);
    ggml_tensor * emb  = ggml_mul(
        ctx, ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, w.turn_emb_w, hp.vad_embed, 1), hp.vad_embed, n, 1, 1), s_in);
    emb               = ggml_add(ctx, emb, w.turn_emb_b);
    ggml_tensor * tin = ggml_concat(ctx, rms(ctx, hn, w.turn_norm_in, eps), emb, 0);
    ggml_tensor * t   = ggml_mul_mat(ctx, w.turn_down, tin);
    t       = ggml_add(ctx, t, batch_attention(ctx, bg, streams, count, li++, t, w.turn_attn, hp.turn_attn, eps));
    t       = rms(ctx, build_mlp(ctx, t, w.turn_mlp, eps), w.turn_norm_out, eps);
    bg.turn = ggml_add(ctx, ggml_mul_mat(ctx, w.turn_ro_w, t), w.turn_ro_b);

    ggml_set_output(bg.logit);
    ggml_set_output(bg.logit_la);
    ggml_set_output(bg.turn);
    bg.graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(bg.graph, bg.turn);
    ggml_build_forward_expand(bg.graph, bg.logit_la);
    for (ggml_tensor * wr : writes) {
        ggml_build_forward_expand(bg.graph, wr);
    }

    auto & backends = const_cast<std::vector<ggml_backend_t> &>(m.plan.scheduler_list);
    bg.sched        = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()), graph_size,
                                             /*parallel=*/false, /*op_offload=*/true);
    if (bg.sched == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: scheduler allocation failed", kTag);
        return TRANSCRIBE_ERR_OOM;
    }
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(bg.sched, bg.graph, kTag)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: batch graph allocation failed (%d streams)", kTag, count);
        return TRANSCRIBE_ERR_OOM;
    }
    bg.n = n;
    for (int j = 0; j < count; ++j) {
        bg.key.push_back(streams[j]->uid);
    }
    return TRANSCRIBE_OK;
}

}  // namespace

void BatchGraph::release() noexcept {
    release_compute_scratch(sched, ctx);
    graph = nullptr;
    key.clear();
    n = 0;
    slots.clear();
    masks.clear();
}

transcribe_status stream_step_batch(BatchGraph &          bg,
                                    const Model &         m,
                                    Stream * const *      streams,
                                    SessionCore * const * cores,
                                    int                   count,
                                    int                   n_threads,
                                    const float * const * pcm,
                                    float *               turn,
                                    float *               speech) {
    const HParams & hp = m.hparams;
    for (int j = 0; j < count; ++j) {
        if (streams[j]->state_buf == nullptr) {
            return TRANSCRIBE_ERR_OOM;  // stream_reset failed
        }
        if (cores[j]->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
    }
    const int64_t t0    = ggml_time_us();
    bool          match = bg.graph != nullptr && bg.key.size() == static_cast<size_t>(count);
    for (int j = 0; match && j < count; ++j) {
        match = bg.key[static_cast<size_t>(j)] == streams[j]->uid;
    }
    if (!match) {
        if (auto rc = build_batch_graph(bg, m, streams, count); rc != TRANSCRIBE_OK) {
            bg.release();
            return rc;
        }
    }
    configure_sched_n_threads(bg.sched, n_threads);
    const int    n    = bg.n;
    const size_t half = static_cast<size_t>(hp.attn.head_dim / 2);

    // Host inputs; padding columns repeat stream 0's position, with no audio.
    bg.mel.assign(static_cast<size_t>(n) * static_cast<size_t>(hp.mel_width), 0.0f);
    bg.log_energy_h.assign(static_cast<size_t>(n), 0.0f);
    bg.rope.assign(2 * half * static_cast<size_t>(n), 0.0f);
    for (int j = 0; j < n; ++j) {
        const size_t js = static_cast<size_t>(j);
        if (j < count) {
            frame_features(*streams[j], m, pcm[j], bg.mel.data() + js * static_cast<size_t>(hp.mel_width),
                           &bg.log_energy_h[js]);
        }
        rope_angles(m, streams[j < count ? j : 0]->pos, bg.rope.data() + js * half,
                    bg.rope.data() + (static_cast<size_t>(n) + js) * half);
    }
    ggml_backend_tensor_set(bg.mel_in, bg.mel.data(), 0, ggml_nbytes(bg.mel_in));
    ggml_backend_tensor_set(bg.log_energy, bg.log_energy_h.data(), 0, ggml_nbytes(bg.log_energy));
    const int32_t zero = 0;
    ggml_backend_tensor_set(bg.ssm_ids, &zero, 0, sizeof(zero));
    ggml_backend_tensor_set(bg.rope_cos, bg.rope.data(), 0, ggml_nbytes(bg.rope_cos));
    ggml_backend_tensor_set(bg.rope_sin, bg.rope.data() + half * static_cast<size_t>(n), 0, ggml_nbytes(bg.rope_sin));
    for (size_t li = 0; li < bg.slots.size(); ++li) {
        for (int j = 0; j < count; ++j) {
            const AttnState & as   = streams[j]->attn[li];
            const int64_t     p    = streams[j]->pos;
            const int64_t     slot = p % as.ring;
            bg.mask.resize(static_cast<size_t>(as.ring) * k_min_graph_frames);
            for (int r = 0; r < k_min_graph_frames; ++r) {
                mask_row(p, as.ring, as.window, bg.mask.data() + static_cast<size_t>(r) * static_cast<size_t>(as.ring));
            }
            ggml_backend_tensor_set(bg.slots[li][static_cast<size_t>(j)], &slot, 0, sizeof(slot));
            ggml_backend_tensor_set(bg.masks[li][static_cast<size_t>(j)], bg.mask.data(), 0,
                                    bg.mask.size() * sizeof(float));
        }
    }

    if (const ggml_status gs = ggml_backend_sched_graph_compute(bg.sched, bg.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: batch graph compute failed (%d)", kTag, static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    const size_t nc = static_cast<size_t>(count);
    bg.logit_h.resize(nc);
    bg.logit_la_h.resize(nc);
    bg.turn_h.resize(nc * static_cast<size_t>(hp.n_horizons));
    ggml_backend_tensor_get(bg.logit, bg.logit_h.data(), 0, nc * sizeof(float));
    ggml_backend_tensor_get(bg.logit_la, bg.logit_la_h.data(), 0, nc * sizeof(float));
    ggml_backend_tensor_get(bg.turn, bg.turn_h.data(), 0, bg.turn_h.size() * sizeof(float));
    const int64_t dt = ggml_time_us() - t0;
    for (int j = 0; j < count; ++j) {
        const size_t js    = static_cast<size_t>(j);
        const bool   early = streams[j]->pos < hp.lookahead_frames;
        if (speech != nullptr) {
            speech[j] = sigmoid(early ? hp.no_speech_logit : bg.logit_la_h[js]);
        }
        if (turn != nullptr) {
            turn[j] = sigmoid(bg.turn_h[js * static_cast<size_t>(hp.n_horizons)]);
        }
        streams[j]->pos += 1;
        cores[j]->t_encode_us += dt;
    }
    return TRANSCRIBE_OK;
}

Stream::Stream() : uid(g_stream_uid.fetch_add(1)) {}

}  // namespace transcribe::sava
