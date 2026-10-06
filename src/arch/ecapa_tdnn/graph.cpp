// arch/ecapa_tdnn/graph.cpp - the ECAPA-TDNN forward graph.
//
// Mirrors speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN.forward followed by
// Xvector.Classifier, with the four conversion-time rewrites already applied
// (scripts/convert-ecapa_tdnn.py): BatchNorm as an affine scale/shift, the three
// activation-free BatchNorms folded forward into the following linear map,
// and the ASP / MFA input concatenations split into per-operand weight
// blocks.
//
// Two graphs share this file, selected by build_graph's `cpu_ops`:
//   - the stock-op graph (GPU backends, and any multi-backend schedule):
//     plain ggml ops; reductions over time run on a contiguous transposed
//     copy [T, C] because ggml reduces over ne[0].
//   - the CPU graph (ggml CPU backend only): the same math through the
//     fused kernels in cpu_ops.h (epilogues, time reductions on [C, T]
//     without transposes, Res2Net without concats) and, where the model
//     packed its weights, the AVX2 GEMM in cpu_gemm.h with the epilogue
//     fused into the store. On a Ryzen 7 4750U this is ~2.4x faster than
//     the stock graph for the F16 file at 8 threads.
// Stage 0 is one im2col matmul in both (see build_blk0_im2col).
//
// Layout: every activation is ggml ne = [C, T].
//
// Helper contract (named / linear / bn_affine / reflect_rows / conv_taps):
// an activation is ggml `ne = [C, T]`, CHANNEL-INNERMOST, frames are rows.
// That makes
//   - a 1x1 convolution a plain `ggml_mul_mat(w[IC, OC], x[IC, T])`,
//   - a bias / BatchNorm vector `[C]` a free broadcast over T,
//   - reflect padding a single `ggml_get_rows` over the frame axis,
//   - a dilated k>1 convolution a sum of matmuls against row-slices of the
//     padded activation, with no transposes (stage 0 alone runs as one
//     matmul over a host-built im2col; see build_blk0_im2col).
// Time-axis reductions (means, softmax, weighted sums) act on ggml's ne[0],
// so the caller transposes to `[T, C]` first.
//
//   mel [n_mels, T]
//     -> blk.0        TDNNBlock  n_mels -> C, k=K0, reflect pad
//     -> blk.1..3     SERes2Net  C -> C, dilation 2 / 3 / 4
//     -> mfa          w1@x1 + w2@x2 + w3@x3 + b, ReLU, BN     [3C, T]
//     -> asp          attentive statistics pooling            [6C]
//     -> fc           6C -> emb (asp_bn folded in)            [emb]
//     -> classifier   LeakyReLU, emb -> hid, LeakyReLU, hid -> n_labels

#include "graph.h"

#include "cpu_ops.h"
#include "ecapa_tdnn.h"
#include "ggml.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"

#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <string>

namespace transcribe::ecapa_tdnn {

namespace {

// Name a tensor and return it; NULL-safe.
ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr && name != nullptr) {
        ggml_set_name(t, name);
    }
    return t;
}

// y = W x (+ b): w is ne = [IC, OC], x is ne = [IC, T] or [IC]; b may be null.
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (w->type == GGML_TYPE_F16) {
        // F32 accumulation for F16 weights: the reductions here are 1024 and
        // 3072 wide and the parity regime is measured with an F32
        // accumulator, so never let a backend pick the F16 one.
        ggml_prec_set_acc(y, GGML_PREC_F32);
    }
    if (b != nullptr) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// BatchNorm stored as an affine map: y = x * scale + shift, both ne = [C].
ggml_tensor * bn_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift) {
    ggml_tensor * y = ggml_mul(ctx, x, scale);
    y               = ggml_add(ctx, y, shift);
    return y;
}

// Reflect padding along the frame axis as one gather. idx is I32 of length
// T + 2p holding [p, ..., 1, 0, 1, ..., T-1, T-2, ..., T-1-p] (torch "reflect",
// edge excluded); the result ne = [C, T + 2p] is contiguous. Requires T > p.
ggml_tensor * reflect_rows(ggml_context * ctx, ggml_tensor * x, ggml_tensor * idx) {
    return ggml_get_rows(ctx, x, idx);
}

// Dilated 1-D conv with "same" padding as a sum over taps: xpad is the
// reflect-padded ne = [IC, T + 2p] (p = dilation * (K - 1) / 2), w the
// tap-major ne = [IC, OC, K]. y[t] = sum_k w_k . xpad[t + k*dilation], which
// is torch's Conv1d(padding="same", padding_mode="reflect"). ne = [OC, T].
ggml_tensor * conv_taps(ggml_context * ctx, ggml_tensor * xpad, ggml_tensor * w, ggml_tensor * b, int dilation, int T) {
    const int64_t IC = w->ne[0];
    const int64_t K  = w->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t k = 0; k < K; ++k) {
        // Tap k's operand: rows [k*d, k*d + T) of the padded activation. The
        // row stride is xpad->nb[1] and the offset is a whole number of rows,
        // so the view is contiguous and usable as a matmul src1 everywhere.
        ggml_tensor * v  = ggml_view_2d(ctx, xpad, IC, T, xpad->nb[1], static_cast<size_t>(k * dilation) * xpad->nb[1]);
        // Tap k's kernel: the contiguous [IC, OC] slice at ne[2] == k.
        ggml_tensor * wk = ggml_view_2d(ctx, w, IC, w->ne[1], w->nb[1], static_cast<size_t>(k) * w->nb[2]);

        ggml_tensor * y = ggml_mul_mat(ctx, wk, v);
        if (w->type == GGML_TYPE_F16) {
            ggml_prec_set_acc(y, GGML_PREC_F32);
        }
        acc = (acc == nullptr) ? y : ggml_add(ctx, acc, y);
    }

    if (b != nullptr) {
        acc = ggml_add(ctx, acc, b);
    }
    return acc;
}

// Nodes reserved in the cgraph. The real graph is 566 nodes for the published
// configuration and is independent of T.
constexpr size_t kGraphSize = 2048;

// Name a tensor, mark it as a graph output so the scheduler cannot reuse its
// buffer before the post-compute dump pass reads it (a no-op unless
// TRANSCRIBE_DUMP_DIR is set), and stash the pointer.
void mark_dump(ggml_tensor *& slot, ggml_tensor * t, const char * name) {
    named(t, name);
    debug::mark_tensor_for_dump(t);
    if (t->view_src != nullptr) {
        // In-place CPU kernels return a view; the allocator frees the view's
        // backing tensor, so that is what has to be kept alive.
        debug::mark_tensor_for_dump(t->view_src);
    }
    slot = t;
}

// One TDNNBlock with k = 1: 1x1 conv -> ReLU -> BatchNorm.
ggml_tensor * tdnn_1x1(ggml_context * ctx, const TdnnLayer & l, ggml_tensor * x) {
    ggml_tensor * y = linear(ctx, l.w, x, l.b);
    y               = ggml_relu(ctx, y);
    return bn_affine(ctx, y, l.bn.scale, l.bn.shift);
}

// One TDNNBlock with k > 1 over an already reflect-padded input:
// dilated conv -> ReLU -> BatchNorm.
ggml_tensor * tdnn_conv(ggml_context *   ctx,
                        ggml_tensor *    xpad,
                        ggml_tensor *    w,
                        ggml_tensor *    b,
                        const BnAffine & bn,
                        int              dilation,
                        int              T) {
    ggml_tensor * y = conv_taps(ctx, xpad, w, b, dilation, T);
    y               = ggml_relu(ctx, y);
    return bn_affine(ctx, y, bn.scale, bn.shift);
}

// Mean over the time axis of x ne = [C, T], returned as ne = [C].
ggml_tensor * mean_over_time(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * xT = ggml_cont(ctx, ggml_transpose(ctx, x));  // [T, C]
    return ggml_reshape_1d(ctx, ggml_mean(ctx, xT), x->ne[0]);
}

// SpeechBrain Res2NetBlock with scale S: split the C channels into S chunks
// of C/S, pass the first through unchanged, and run each remaining chunk
// through its own dilated conv after adding the previous chunk's output.
//
//   y0 = x0
//   y1 = B0(x1)
//   yi = B(i-1)(xi + y(i-1))    i = 2 .. S-1
//   out = concat(y0 .. y(S-1))
//
// Chunk views are strided (stride C, width C/S), so each needs a ggml_cont
// before it can serve as a matmul operand.
ggml_tensor * res2net(ggml_context *         ctx,
                      const SeRes2NetBlock & blk,
                      ggml_tensor *          h,
                      ggml_tensor *          idx,
                      int                    dilation,
                      int                    T,
                      int                    chunk) {
    const size_t chunk_bytes = static_cast<size_t>(chunk) * ggml_element_size(h);

    ggml_tensor * acc  = nullptr;  // concatenated output so far
    ggml_tensor * prev = nullptr;  // y(i-1)

    for (int i = 0; i < kRes2NetScale; ++i) {
        ggml_tensor * ci =
            ggml_cont(ctx, ggml_view_2d(ctx, h, chunk, T, h->nb[1], static_cast<size_t>(i) * chunk_bytes));

        ggml_tensor * yi = nullptr;
        if (i == 0) {
            yi = ci;
        } else {
            ggml_tensor *   in = (i == 1) ? ci : ggml_add(ctx, ci, prev);
            const Res2Sub & s  = blk.res2[i - 1];
            yi                 = tdnn_conv(ctx, reflect_rows(ctx, in, idx), s.w, s.b, s.bn, dilation, T);
        }

        prev = yi;
        acc  = (acc == nullptr) ? yi : ggml_concat(ctx, acc, yi, 0);
    }

    return acc;
}

// SpeechBrain SEBlock: s = mean_T(x) -> 1x1 -> ReLU -> 1x1 -> sigmoid, then
// x * s broadcast over T.
ggml_tensor * se_block(ggml_context * ctx, const SeBlock & se, ggml_tensor * x) {
    ggml_tensor * s = mean_over_time(ctx, x);
    s               = ggml_relu(ctx, linear(ctx, se.c1_w, s, se.c1_b));
    s               = ggml_sigmoid(ctx, linear(ctx, se.c2_w, s, se.c2_b));
    return ggml_mul(ctx, x, s);
}

// ---- CPU-backend variants (cpu_ops.h) -------------------------------------
//
// Same math as the stock helpers above; see cpu_ops.h for why they exist.

// F32 accumulation for F16 weights (see linear()).
ggml_tensor * mul_mat_acc(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (w->type == GGML_TYPE_F16) {
        ggml_prec_set_acc(y, GGML_PREC_F32);
    }
    return y;
}

// TDNNBlock k=1: with packed weights one GEMM with the bias/ReLU/BN fused
// into its store; otherwise ggml_mul_mat plus one fused epilogue pass.
ggml_tensor * tdnn_1x1_cpu(ggml_context * ctx, gemm::Arena * ar, const TdnnLayer & l, ggml_tensor * x) {
    if (ar != nullptr) {
        const gemm::Segment sg{ l.w, 0, x, 0 };
        return gemm::mul_mat(ctx, *ar, &sg, 1, x->ne[1], gemm::Epilogue::ReluBn, l.b, l.bn.scale, l.bn.shift);
    }
    return cpu::epilogue(ctx, mul_mat_acc(ctx, l.w, x), l.b, l.bn.scale, l.bn.shift, false, true);
}

// Res2Net on the CPU. Each sub-conv is ONE matmul of its stacked taps
// (the [w, K*w] view of the tap-major [w, w, K] weight) against the padded
// chunk input, and one fused kernel that sums the shifted tap rows, applies
// the epilogue, and builds the next chunk's padded input; the output is
// assembled once at the end instead of through S - 1 growing concats.
//
// With packed weights (the GEMM path) the stacked-tap trick is replaced by a
// true dilated conv: one GEMM with one segment per tap, tap k reading the
// padded input k*d rows further on, with the epilogue fused into the store.
// The padded input for the next chunk (h[next] + y, reflect padded) is then
// a small pad kernel.
ggml_tensor * res2net_cpu(ggml_context *         ctx,
                          gemm::Arena *          ar,
                          const SeRes2NetBlock & blk,
                          ggml_tensor *          h,
                          int                    kernel,
                          int                    dilation,
                          int                    chunk) {
    ggml_tensor * ys[kRes2NetSubs] = {};
    const int64_t T                = h->ne[1];

    cpu::Res2Step st = cpu::res2_step(ctx, h, /*next=*/1, chunk, nullptr, nullptr, nullptr, nullptr, kernel, dilation);
    for (int j = 0; j < kRes2NetSubs; ++j) {
        const Res2Sub & s  = blk.res2[j];
        const int       nx = (j + 2 < kRes2NetScale) ? j + 2 : -1;
        if (ar != nullptr) {
            gemm::Segment sg[gemm::kMaxSegs];
            for (int k = 0; k < kernel; ++k) {
                sg[k] = { s.w, k, st.pad, static_cast<int64_t>(k) * dilation };
            }
            ys[j] = gemm::mul_mat(ctx, *ar, sg, kernel, T, gemm::Epilogue::ReluBn, s.b, s.bn.scale, s.bn.shift);
            if (nx >= 0) {
                st = cpu::res2_pad(ctx, h, nx, chunk, ys[j], kernel, dilation);
            }
            continue;
        }
        ggml_tensor * ws = ggml_reshape_2d(ctx, s.w, chunk, chunk * kernel);  // [w, K*w], tap-major rows
        ggml_tensor * z  = mul_mat_acc(ctx, ws, st.pad);                      // [K*w, T + 2p]
        st               = cpu::res2_step(ctx, h, nx, chunk, z, s.b, s.bn.scale, s.bn.shift, kernel, dilation);
        ys[j]            = st.y;
    }
    return cpu::res2_gather(ctx, h, ys, kRes2NetSubs);
}

// SEBlock fused with the residual add: x * s + residual.
ggml_tensor * se_block_cpu(ggml_context * ctx, const SeBlock & se, ggml_tensor * x, ggml_tensor * residual) {
    ggml_tensor * s = cpu::time_mean(ctx, x);
    s               = ggml_relu(ctx, linear(ctx, se.c1_w, s, se.c1_b));
    s               = ggml_sigmoid(ctx, linear(ctx, se.c2_w, s, se.c2_b));
    return cpu::scale_add(ctx, x, s, residual);
}

// MFA + attentive statistics pooling, stock ops. Returns pooled [2*Cm].
ggml_tensor * mfa_asp(ggml_context *        ctx,
                      GraphBuild &          gb,
                      const HParams &       hp,
                      const Weights &       w,
                      ggml_tensor * const * blk_out,
                      int                   Cm) {
    ggml_tensor * pooled = nullptr;
    // ---- MFA: 1x1 over concat(blk1, blk2, blk3), split into three blocks --
    ggml_tensor * m      = nullptr;
    for (int i = 0; i < kNumSeBlocks; ++i) {
        ggml_tensor * part = ggml_mul_mat(ctx, w.mfa_w[i], blk_out[i]);
        if (w.mfa_w[i]->type == GGML_TYPE_F16) {
            ggml_prec_set_acc(part, GGML_PREC_F32);
        }
        m = (m == nullptr) ? part : ggml_add(ctx, m, part);
    }
    m = ggml_add(ctx, m, w.mfa_b);
    m = ggml_relu(ctx, m);
    m = bn_affine(ctx, m, w.mfa_bn.scale, w.mfa_bn.shift);
    mark_dump(gb.dumps.mfa_out, m, "enc.mfa.out");

    // ---- attentive statistics pooling ------------------------------------
    //
    // mean / std are the utterance statistics over T with the uniform
    // (unpadded) weighting SpeechBrain uses when wav_lens is all-ones; the
    // variance is the biased 1/T one, and the clamp before the sqrt is
    // SpeechBrain's `.clamp(eps)` and must not be dropped.
    ggml_tensor * mT   = ggml_cont(ctx, ggml_transpose(ctx, m));  // [T, Cm]
    ggml_tensor * mean = ggml_mean(ctx, mT);                      // [1, Cm]
    ggml_tensor * var  = ggml_mean(ctx, ggml_sqr(ctx, ggml_sub(ctx, mT, mean)));
    ggml_tensor * sd   = ggml_sqrt(ctx, ggml_clamp(ctx, var, hp.asp_eps, FLT_MAX));

    ggml_tensor * mean1 = ggml_reshape_1d(ctx, mean, Cm);
    ggml_tensor * sd1   = ggml_reshape_1d(ctx, sd, Cm);

    // The context term wm@mean + ws@std + b is a [att] vector broadcast over
    // T, so the 3*Cm -> att convolution costs one Cm x T matmul instead of
    // three.
    ggml_tensor * cvec = ggml_add(ctx, linear(ctx, w.asp_wm, mean1, nullptr), linear(ctx, w.asp_ws, sd1, nullptr));
    cvec               = ggml_add(ctx, cvec, w.asp_b);

    ggml_tensor * a = ggml_add(ctx, linear(ctx, w.asp_wx, m, nullptr), cvec);  // [att, T]
    a               = ggml_relu(ctx, a);
    a               = bn_affine(ctx, a, w.asp_bn.scale, w.asp_bn.shift);
    a               = ggml_tanh(ctx, a);

    ggml_tensor * al = linear(ctx, w.asp_attn_w, a, w.asp_attn_b);  // [Cm, T]
    // Marked BEFORE the transpose: ggml ne = [Cm, T] lands on disk as
    // [T, Cm], the orientation the reference dumps.
    mark_dump(gb.dumps.asp_attn_logits, al, "enc.asp.attn_logits");

    ggml_tensor * alT  = ggml_cont(ctx, ggml_transpose(ctx, al));     // [T, Cm]
    ggml_tensor * attn = ggml_soft_max(ctx, alT);                     // over ne[0] = T

    ggml_tensor * mu  = ggml_sum_rows(ctx, ggml_mul(ctx, attn, mT));  // [1, Cm]
    ggml_tensor * dev = ggml_sqr(ctx, ggml_sub(ctx, mT, mu));
    ggml_tensor * sig = ggml_sum_rows(ctx, ggml_mul(ctx, attn, dev));
    sig               = ggml_sqrt(ctx, ggml_clamp(ctx, sig, hp.asp_eps, FLT_MAX));

    pooled = ggml_concat(ctx, ggml_reshape_1d(ctx, mu, Cm), ggml_reshape_1d(ctx, sig, Cm), /*dim=*/0);  // [2*Cm]
    mark_dump(gb.dumps.asp_out, pooled, "enc.asp.out");

    return pooled;
}

// MFA + attentive statistics pooling on the CPU: the three MFA products and
// their epilogue are one fused pass, and every reduction over time runs on
// the [C, T] layout directly (no transposes). Returns pooled [2*Cm].
ggml_tensor * mfa_asp_cpu(ggml_context *        ctx,
                          gemm::Arena *         ar,
                          GraphBuild &          gb,
                          const HParams &       hp,
                          const Weights &       w,
                          ggml_tensor * const * blk_out) {
    const int     Cm = hp.c_mfa();
    const int64_t T  = blk_out[0]->ne[1];

    ggml_tensor * m = nullptr;
    if (ar != nullptr) {
        // All three products accumulate in one GEMM (three segments).
        const gemm::Segment sg[kNumSeBlocks] = {
            { w.mfa_w[0], 0, blk_out[0], 0 },
            { w.mfa_w[1], 0, blk_out[1], 0 },
            { w.mfa_w[2], 0, blk_out[2], 0 }
        };
        m = gemm::mul_mat(ctx, *ar, sg, kNumSeBlocks, T, gemm::Epilogue::ReluBn, w.mfa_b, w.mfa_bn.scale,
                          w.mfa_bn.shift);
    } else {
        m = cpu::epilogue_sum3(ctx, mul_mat_acc(ctx, w.mfa_w[0], blk_out[0]), mul_mat_acc(ctx, w.mfa_w[1], blk_out[1]),
                               mul_mat_acc(ctx, w.mfa_w[2], blk_out[2]), w.mfa_b, w.mfa_bn.scale, w.mfa_bn.shift, true);
    }
    mark_dump(gb.dumps.mfa_out, m, "enc.mfa.out");

    // [mean, std] over T, then the context term wm@mean + ws@std + b.
    ggml_tensor * st    = cpu::time_stats(ctx, m, &hp.asp_eps);
    ggml_tensor * mean1 = ggml_view_1d(ctx, st, Cm, 0);
    ggml_tensor * sd1   = ggml_view_1d(ctx, st, Cm, static_cast<size_t>(Cm) * sizeof(float));
    ggml_tensor * cvec  = ggml_add(ctx, linear(ctx, w.asp_wm, mean1, nullptr), linear(ctx, w.asp_ws, sd1, nullptr));
    cvec                = ggml_add(ctx, cvec, w.asp_b);

    // a = tanh(BN(ReLU(wx@m + cvec))): cvec plays the bias.
    ggml_tensor * a  = nullptr;
    ggml_tensor * al = nullptr;
    if (ar != nullptr) {
        const gemm::Segment sx{ w.asp_wx, 0, m, 0 };
        a = gemm::mul_mat(ctx, *ar, &sx, 1, T, gemm::Epilogue::ReluBnTanh, cvec, w.asp_bn.scale, w.asp_bn.shift);
        const gemm::Segment sa{ w.asp_attn_w, 0, a, 0 };
        al = gemm::mul_mat(ctx, *ar, &sa, 1, T, gemm::Epilogue::None, nullptr, nullptr, nullptr);
    } else {
        a  = cpu::epilogue(ctx, mul_mat_acc(ctx, w.asp_wx, m), cvec, w.asp_bn.scale, w.asp_bn.shift,
                           /*tanh_after=*/true, true);
        al = mul_mat_acc(ctx, w.asp_attn_w, a);
    }
    // al: attention logits [Cm, T] WITHOUT the bias; attn_stats adds it per
    // channel before the softmax over time. The dump point needs the biased
    // logits, so the dump build adds it explicitly on a side branch.
    if (debug::enabled()) {
        mark_dump(gb.dumps.asp_attn_logits, ggml_add(ctx, al, w.asp_attn_b), "enc.asp.attn_logits");
        ggml_build_forward_expand(gb.graph, gb.dumps.asp_attn_logits);
    }
    ggml_tensor * pooled = cpu::attn_stats(ctx, m, al, w.asp_attn_b, &hp.asp_eps);  // [2*Cm] = [mu, sigma]
    mark_dump(gb.dumps.asp_out, pooled, "enc.asp.out");
    return pooled;
}

}  // namespace

void build_blk0_im2col(const HParams & hp, const float * mel, int T, std::vector<float> & out) {
    const int    n_mels = hp.mel_n_mels;
    const int    K      = hp.kernel_sizes[0];
    const int    d      = hp.dilations[0];
    const int    p      = hp.pad(0);
    const size_t cols   = static_cast<size_t>(hp.blk0_cols());
    out.assign(static_cast<size_t>(T) * cols, 0.0f);
    for (int t = 0; t < T; ++t) {
        float * row = out.data() + static_cast<size_t>(t) * cols;
        for (int k = 0; k < K; ++k) {
            int src = t + k * d - p;  // torch "reflect", edge not repeated
            if (src < 0) {
                src = -src;
            } else if (src >= T) {
                src = 2 * (T - 1) - src;
            }
            const float * m = mel + static_cast<size_t>(src) * static_cast<size_t>(n_mels);
            std::copy(m, m + n_mels, row + static_cast<size_t>(k) * static_cast<size_t>(n_mels));
        }
    }
}

GraphBuild build_graph(ggml_context * ctx, const Model & model, int T, bool cpu_ops, gemm::Arena * arena) {
    GraphBuild gb{};

    // Per weight group: the arena when that group was packed, else null
    // (null routes the group through ggml_mul_mat).
    gemm::Arena * ar_conv = cpu_ops && model.gemm_conv ? arena : nullptr;
    gemm::Arena * ar_lin  = cpu_ops && model.gemm_lin ? arena : nullptr;

    const HParams & hp = model.hparams;
    const Weights & w  = model.weights;

    if (ctx == nullptr || T <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "ecapa_tdnn graph: invalid arg (ctx=%p, T=%d)", static_cast<void *>(ctx),
                T);
        return gb;
    }

    const int Cm    = hp.c_mfa();
    const int chunk = hp.c_chunk();

    // reflect_rows needs T > p so the mirrored indices stay in range. The
    // 500 ms LANGID minimum gives T >= 51 frames; the largest p here is 4.
    for (int i = 0; i < kNumSeBlocks; ++i) {
        if (T <= hp.pad(i + 1)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "ecapa_tdnn graph: T=%d is too short for reflect padding %d", T,
                    hp.pad(i + 1));
            return gb;
        }
    }

    gb.graph = ggml_new_graph_custom(ctx, kGraphSize, /*grads=*/false);
    if (gb.graph == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "ecapa_tdnn graph: ggml_new_graph_custom failed");
        return gb;
    }

    // ---- inputs ----------------------------------------------------------
    gb.blk0_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.blk0_cols(), T);
    named(gb.blk0_in, "fe.mel.im2col");
    ggml_set_input(gb.blk0_in);

    if (!cpu_ops) {
        for (int i = 0; i < kNumSeBlocks; ++i) {
            char name[32];
            std::snprintf(name, sizeof(name), "reflect.idx.%d", i);
            gb.idx[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T + 2 * hp.pad(i + 1));
            named(gb.idx[i], name);
            ggml_set_input(gb.idx[i]);
        }
    }

    // ---- stage 0: TDNNBlock n_mels -> C ----------------------------------
    // One matmul over the host-built im2col (K0 taps x n_mels, zero padded to
    // a multiple of 32): n_mels = 60 alone is not a multiple of any CPU
    // tinyBLAS tile, and ggml's fallback GEMM for that shape is ~15x slower.
    ggml_tensor * x = nullptr;
    if (ar_conv != nullptr) {
        const gemm::Segment sg{ w.blk0_w_im2col, 0, gb.blk0_in, 0 };
        x = gemm::mul_mat(ctx, *ar_conv, &sg, 1, T, gemm::Epilogue::ReluBn, w.blk0_b, w.blk0_bn.scale, w.blk0_bn.shift);
    } else if (cpu_ops) {
        x = cpu::epilogue(ctx, mul_mat_acc(ctx, w.blk0_w_im2col, gb.blk0_in), w.blk0_b, w.blk0_bn.scale,
                          w.blk0_bn.shift, false, true);
    } else {
        x = linear(ctx, w.blk0_w_im2col, gb.blk0_in, w.blk0_b);
        x = ggml_relu(ctx, x);
        x = bn_affine(ctx, x, w.blk0_bn.scale, w.blk0_bn.shift);
    }
    mark_dump(gb.dumps.blk0_out, x, "enc.blk.0.out");

    // ---- stages 1..3: SERes2Net ------------------------------------------
    ggml_tensor * blk_out[kNumSeBlocks] = { nullptr, nullptr, nullptr };
    for (int i = 0; i < kNumSeBlocks; ++i) {
        const SeRes2NetBlock & blk = w.blocks[i];
        const int              d   = hp.dilations[static_cast<size_t>(i + 1)];

        ggml_tensor * residual = x;

        ggml_tensor * h = cpu_ops ? tdnn_1x1_cpu(ctx, ar_lin, blk.tdnn1, x) : tdnn_1x1(ctx, blk.tdnn1, x);
        if (i == 0) {
            mark_dump(gb.dumps.blk1_tdnn1_out, h, "enc.blk.1.tdnn1.out");
        }

        if (cpu_ops) {
            h = res2net_cpu(ctx, ar_conv, blk, h, hp.kernel_sizes[static_cast<size_t>(i + 1)], d, chunk);
        } else {
            h = res2net(ctx, blk, h, gb.idx[i], d, T, chunk);
        }
        if (i == 0) {
            mark_dump(gb.dumps.blk1_res2_out, h, "enc.blk.1.res2.out");
        }

        h = cpu_ops ? tdnn_1x1_cpu(ctx, ar_lin, blk.tdnn2, h) : tdnn_1x1(ctx, blk.tdnn2, h);

        if (cpu_ops) {
            // The SE output and the residual add are one kernel here, so the
            // enc.blk.1.se.out dump point does not exist in this graph.
            x = se_block_cpu(ctx, blk.se, h, residual);
        } else {
            h = se_block(ctx, blk.se, h);
            if (i == 0) {
                mark_dump(gb.dumps.blk1_se_out, h, "enc.blk.1.se.out");
            }
            x = ggml_add(ctx, h, residual);
        }

        char name[32];
        std::snprintf(name, sizeof(name), "enc.blk.%d.out", i + 1);
        mark_dump(gb.dumps.blk_out[i], x, name);
        blk_out[i] = x;
    }

    ggml_tensor * pooled =
        cpu_ops ? mfa_asp_cpu(ctx, ar_lin, gb, hp, w, blk_out) : mfa_asp(ctx, gb, hp, w, blk_out, Cm);

    // ---- embedding head (asp_bn folded into fc) ---------------------------
    ggml_tensor * emb = linear(ctx, w.fc_w, pooled, w.fc_b);
    mark_dump(gb.dumps.emb, emb, "enc.emb");

    // ---- classifier -------------------------------------------------------
    // LeakyReLU comes FIRST, before the BatchNorm that is folded into the
    // following linear map; getting that order wrong is invisible until the
    // logits are compared.
    ggml_tensor * e = ggml_leaky_relu(ctx, emb, hp.leaky_slope, /*inplace=*/false);
    ggml_tensor * hcls =
        ggml_leaky_relu(ctx, linear(ctx, w.cls_l1_w, e, w.cls_l1_b), hp.leaky_slope, /*inplace=*/false);
    mark_dump(gb.dumps.cls_hidden, hcls, "cls.hidden");

    ggml_tensor * z = linear(ctx, w.cls_out_w, hcls, w.cls_out_b);
    mark_dump(gb.dumps.cls_logits, z, "cls.logits_raw");
    gb.logits = z;

    ggml_set_output(gb.logits);
    ggml_build_forward_expand(gb.graph, gb.logits);

    return gb;
}

}  // namespace transcribe::ecapa_tdnn
