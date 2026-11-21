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
    const int64_t n  = x->ne[1];
    const int64_t hd = a.head_dim;
    const int64_t H  = a.n_heads;
    const int64_t KV = a.n_kv_heads;
    const size_t  fs = sizeof(float);

    ggml_tensor * h   = rms(ctx, x, w.norm, eps);
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.qkv, h);  // [(H + 2 KV) hd, n]
    ggml_tensor * q   = ggml_view_3d(ctx, qkv, hd, H, n, hd * fs, qkv->nb[1], 0);
    ggml_tensor * k   = ggml_view_3d(ctx, qkv, hd, KV, n, hd * fs, qkv->nb[1], H * hd * fs);
    ggml_tensor * v   = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, KV, n, hd * fs, qkv->nb[1], (H + KV) * hd * fs));

    q = rope(ctx, rms(ctx, q, w.q_norm, eps), g.rope_cos, g.rope_sin);  // [hd, H, n]
    k = rope(ctx, rms(ctx, k, w.k_norm, eps), g.rope_cos, g.rope_sin);  // [hd, KV, n]

    const AttnState & as = st.attn[li];
    ggml_tensor *     kr = ggml_set_rows(ctx, as.k_ring, ggml_reshape_2d(ctx, k, hd * KV, n), g.slots[li]);
    ggml_tensor *     vr = ggml_set_rows(ctx, as.v_ring, ggml_reshape_2d(ctx, v, hd * KV, n), g.slots[li]);

    // scores [ring, n, H]; query head i reads key head i / (H / KV).
    ggml_tensor * K = ggml_permute(ctx, ggml_view_3d(ctx, kr, hd, KV, as.ring, hd * fs, kr->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * P = ggml_mul_mat(ctx, K, Q);
    P               = ggml_soft_max_ext(ctx, P, g.masks[li], 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);

    // out [hd, n, H] -> [hd, H, n]
    ggml_tensor * V =
        ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, vr, hd, KV, as.ring, hd * fs, vr->nb[1], 0), 1, 2, 0, 3));
    ggml_tensor * O = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, V, P), 0, 2, 1, 3));

    if (a.xsa) {
        // Exclusive self attention: remove alpha times each head's projection
        // on its own frame's value.
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

// Host inputs for a block of n_real frames in a graph of n columns.
void fill_inputs(Stream & st, const Model & m, const float * pcm, int n, int n_real) {
    const HParams & hp    = m.hparams;
    const size_t    nfft  = static_cast<size_t>(hp.n_fft);
    const size_t    hops  = static_cast<size_t>(hp.hops_per_frame);
    const size_t    frame = static_cast<size_t>(hp.frame_samples);
    const size_t    ctxn  = static_cast<size_t>(hp.context_samples);
    const size_t    bins  = static_cast<size_t>(hp.n_bins);
    const size_t    mels  = static_cast<size_t>(hp.n_mels);

    st.mel.assign(static_cast<size_t>(n) * static_cast<size_t>(hp.mel_width), 0.0f);
    st.log_energy.assign(static_cast<size_t>(n), 0.0f);
    std::vector<float>  buf(ctxn + frame);
    std::vector<double> win(nfft);
    st.fft_re.resize(nfft);
    st.fft_im.resize(nfft);
    st.fft_scratch.resize(4 * nfft);
    std::vector<double> power(bins);
    for (int f = 0; f < n_real; ++f) {
        const float * src = pcm + static_cast<size_t>(f) * frame;
        std::memcpy(buf.data(), st.context.data(), ctxn * sizeof(float));
        std::memcpy(buf.data() + ctxn, src, frame * sizeof(float));
        // Window j ends where hop j ends: [ctx | frame][hop j : hop j + n_fft].
        for (size_t j = 0; j < hops; ++j) {
            const float * w0 = buf.data() + j * static_cast<size_t>(hp.hop);
            for (size_t i = 0; i < nfft; ++i) {
                win[i] = static_cast<double>(w0[i]) * m.window[i];
            }
            fft(win.data(), hp.n_fft, 1, m, hp.n_fft, st.fft_re.data(), st.fft_im.data(), st.fft_scratch.data());
            for (size_t k = 0; k < bins; ++k) {
                power[k] = st.fft_re[k] * st.fft_re[k] + st.fft_im[k] * st.fft_im[k];
            }
            float * out = st.mel.data() + static_cast<size_t>(f) * static_cast<size_t>(hp.mel_width) + j * mels;
            for (size_t b = 0; b < mels; ++b) {
                const double * row = m.mel_bank.data() + b * bins;
                double         acc = 0.0;
                for (size_t k = 0; k < bins; ++k) {
                    acc += row[k] * power[k];
                }
                acc    = std::max(acc, static_cast<double>(hp.mel_floor));
                out[b] = static_cast<float>((std::log10(acc) + 4.0) / 4.0);
            }
        }
        std::memcpy(st.context.data(), buf.data() + frame, ctxn * sizeof(float));
        double sum = 0.0;
        for (size_t i = 0; i < frame; ++i) {
            sum += static_cast<double>(src[i]) * static_cast<double>(src[i]);
        }
        const float r                         = static_cast<float>(std::sqrt(sum / static_cast<double>(frame)));
        st.log_energy[static_cast<size_t>(f)] = std::log(std::max(r, hp.log_energy_floor));
    }

    // RoPE angles in fp64 from the absolute position, rounded to fp32.
    const size_t half = static_cast<size_t>(hp.attn.head_dim / 2);
    st.rope.assign(2 * half * static_cast<size_t>(n), 0.0f);
    for (int j = 0; j < n; ++j) {
        const double p = static_cast<double>(st.pos + j);
        for (size_t i = 0; i < half; ++i) {
            const double ang                                 = p * static_cast<double>(m.rope_freqs[i]);
            st.rope[static_cast<size_t>(j) * half + i]       = static_cast<float>(std::cos(ang));
            st.rope[(static_cast<size_t>(n + j)) * half + i] = static_cast<float>(std::sin(ang));
        }
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
            st.attn[i].v_ring = ggml_new_tensor_2d(st.state_ctx, GGML_TYPE_F32, kv, st.attn[i].ring);
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

transcribe_status stream_step(SessionCore & core,
                              Stream &      st,
                              const Model & m,
                              const float * pcm,
                              int64_t       n_frames,
                              float *       turn,
                              float *       speech) {
    const HParams & hp = m.hparams;
    debug::init();
    if (st.state_buf == nullptr) {
        return TRANSCRIBE_ERR_OOM;  // stream_reset failed
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
            st.mask.assign(static_cast<size_t>(ring) * static_cast<size_t>(n), -INFINITY);
            for (int j = 0; j < n; ++j) {
                const int64_t p                  = p0 + j;
                st.slots[static_cast<size_t>(j)] = p % ring;
                float * row                      = st.mask.data() + static_cast<size_t>(j) * static_cast<size_t>(ring);
                for (int64_t dist = 0; dist < window && dist <= p; ++dist) {
                    row[(p - dist) % ring] = 0.0f;
                }
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

}  // namespace transcribe::sava
