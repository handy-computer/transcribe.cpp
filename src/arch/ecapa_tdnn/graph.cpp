// arch/ecapa_tdnn/graph.cpp - the ECAPA-TDNN forward graph.
//
// SpeechBrain ECAPA_TDNN.forward + Xvector.Classifier, after the converter's
// rewrites: BatchNorm as scale/shift, activation-free BatchNorms folded into
// the next linear map, and the MFA / ASP concats split into weight blocks.
//
// Activations are ne = [C, T]: a 1x1 conv is one mul_mat, reflect padding is
// one get_rows over frames, and time reductions run on a [T, C] transpose.
//
//   mel [n_mels, T]
//     -> blk.0        TDNNBlock  n_mels -> C, k=K0 (host im2col + one matmul)
//     -> blk.1..3     SERes2Net  C -> C
//     -> mfa          1x1 over concat(blk.1..3)                [3C, T]
//     -> asp          attentive statistics pooling             [6C]
//     -> fc           6C -> emb                                [emb]
//     -> classifier   LeakyReLU, emb -> hid, LeakyReLU, hid -> n_labels

#include "graph.h"

#include "ecapa_tdnn.h"
#include "ggml.h"
#include "transcribe-debug.h"

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

// Dilated 1-D conv as a sum over taps: xpad is the reflect-padded
// ne = [IC, T + 2p], w the tap-major ne = [IC, OC, K].
// y[t] = sum_k w_k . xpad[t + k*dilation], ne = [OC, T].
ggml_tensor * conv_taps(ggml_context * ctx, ggml_tensor * xpad, ggml_tensor * w, ggml_tensor * b, int dilation, int T) {
    const int64_t IC = w->ne[0];
    const int64_t K  = w->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t k = 0; k < K; ++k) {
        // Rows [k*d, k*d + T) of xpad, and tap k's [IC, OC] slice of w.
        ggml_tensor * v  = ggml_view_2d(ctx, xpad, IC, T, xpad->nb[1], static_cast<size_t>(k * dilation) * xpad->nb[1]);
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

// Name a tensor, mark it for dumping (no-op unless TRANSCRIBE_DUMP_DIR is
// set), and stash the pointer.
void mark_dump(ggml_tensor *& slot, ggml_tensor * t, const char * name) {
    named(t, name);
    debug::mark_tensor_for_dump(t);
    if (t->view_src != nullptr) {
        // In-place ops return a view; keep its backing tensor alive too.
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

// SpeechBrain Res2NetBlock with scale S over chunks x0..x(S-1) of C/S:
//   y0 = x0,  y1 = B0(x1),  yi = B(i-1)(xi + y(i-1)),  out = concat(y0..y(S-1))
// Each yi is written over chunk i of h in place with set_rows (h viewed as
// [C/S, S, T]), so there is no concat.
ggml_tensor * res2net(ggml_context *         ctx,
                      const SeRes2NetBlock & blk,
                      ggml_tensor *          h,
                      ggml_tensor *          idx,
                      ggml_tensor *          chunk_ids,
                      int                    dilation,
                      int                    T,
                      int                    chunk) {
    const size_t chunk_bytes = static_cast<size_t>(chunk) * ggml_element_size(h);

    ggml_tensor * out  = ggml_reshape_3d(ctx, h, chunk, kRes2NetScale, T);
    ggml_tensor * prev = nullptr;  // y(i-1)

    for (int i = 1; i < kRes2NetScale; ++i) {
        ggml_tensor *   ci = ggml_view_2d(ctx, out, chunk, T, out->nb[2], static_cast<size_t>(i) * chunk_bytes);
        ggml_tensor *   in = (i == 1) ? ci : ggml_add(ctx, ci, prev);
        const Res2Sub & s  = blk.res2[i - 1];
        ggml_tensor *   yi = tdnn_conv(ctx, ggml_get_rows(ctx, in, idx), s.w, s.b, s.bn, dilation, T);

        out  = ggml_set_rows(ctx, out, ggml_reshape_3d(ctx, yi, chunk, 1, T),
                             ggml_view_1d(ctx, chunk_ids, 1, static_cast<size_t>(i) * sizeof(int32_t)));
        prev = yi;
    }

    return ggml_reshape_2d(ctx, out, h->ne[0], T);
}

// SpeechBrain SEBlock: s = mean_T(x) -> 1x1 -> ReLU -> 1x1 -> sigmoid, then
// x * s broadcast over T.
ggml_tensor * se_block(ggml_context * ctx, const SeBlock & se, ggml_tensor * x) {
    ggml_tensor * s = mean_over_time(ctx, x);
    s               = ggml_relu(ctx, linear(ctx, se.c1_w, s, se.c1_b));
    s               = ggml_sigmoid(ctx, linear(ctx, se.c2_w, s, se.c2_b));
    return ggml_mul(ctx, x, s);
}

// MFA + attentive statistics pooling, stock ops. Returns pooled [2*Cm].
ggml_tensor * mfa_asp(ggml_context *        ctx,
                      GraphBuild &          gb,
                      const HParams &       hp,
                      const Weights &       w,
                      ggml_tensor * const * blk_out,
                      int                   Cm) {
    const int64_t T = blk_out[0]->ne[1];

    // ---- MFA: 1x1 over concat(blk1, blk2, blk3), split into three blocks --
    // Expand all three products first and start the sum from part[2] so the
    // two adds stay adjacent (Vulkan fuses them).
    ggml_tensor * part[kNumSeBlocks] = {};
    for (int i = 0; i < kNumSeBlocks; ++i) {
        part[i] = linear(ctx, w.mfa_w[i], blk_out[i], nullptr);
        ggml_build_forward_expand(gb.graph, part[i]);
    }
    ggml_tensor * m = ggml_add(ctx, ggml_add(ctx, part[2], part[0]), part[1]);
    m               = ggml_add(ctx, m, w.mfa_b);
    m               = ggml_relu(ctx, m);
    m               = bn_affine(ctx, m, w.mfa_bn.scale, w.mfa_bn.shift);
    mark_dump(gb.dumps.mfa_out, m, "enc.mfa.out");

    // ---- attentive statistics pooling ------------------------------------
    // Uniform mean / biased std over T, std clamped at eps (SpeechBrain).
    ggml_tensor * mT   = ggml_cont(ctx, ggml_transpose(ctx, m));  // [T, Cm]
    ggml_tensor * mean = ggml_mean(ctx, mT);                      // [1, Cm]
    ggml_tensor * d    = ggml_sub(ctx, mT, mean);                 // [T, Cm]
    ggml_tensor * d2   = ggml_sqr(ctx, d);
    ggml_tensor * var  = ggml_mean(ctx, d2);
    ggml_tensor * sd   = ggml_sqrt(ctx, ggml_clamp(ctx, var, hp.asp_eps, FLT_MAX));

    ggml_tensor * mean1 = ggml_reshape_1d(ctx, mean, Cm);
    ggml_tensor * sd1   = ggml_reshape_1d(ctx, sd, Cm);

    // wm@mean + ws@std + b is constant over T: one [att] vector, broadcast.
    ggml_tensor * cvec = ggml_add(ctx, linear(ctx, w.asp_wm, mean1, nullptr), linear(ctx, w.asp_ws, sd1, nullptr));
    cvec               = ggml_add(ctx, cvec, w.asp_b);

    ggml_tensor * a = ggml_add(ctx, linear(ctx, w.asp_wx, m, nullptr), cvec);  // [att, T]
    a               = ggml_relu(ctx, a);
    a               = bn_affine(ctx, a, w.asp_bn.scale, w.asp_bn.shift);
    a               = ggml_tanh(ctx, a);

    // The per-channel bias cancels in the softmax over T; only the dump adds it.
    ggml_tensor * al = linear(ctx, w.asp_attn_w, a, nullptr);  // [Cm, T]
    if (debug::enabled()) {
        mark_dump(gb.dumps.asp_attn_logits, ggml_add(ctx, al, w.asp_attn_b), "enc.asp.attn_logits");
        ggml_build_forward_expand(gb.graph, gb.dumps.asp_attn_logits);
    }

    ggml_tensor * alT  = ggml_cont(ctx, ggml_transpose(ctx, al));  // [T, Cm]
    ggml_tensor * attn = ggml_soft_max(ctx, alT);                  // over ne[0] = T

    // Weighted moments about the uniform mean (sum_t attn = 1):
    //   e1 = sum_t attn d,  e2 = sum_t attn d^2,  mu = mean + e1,  sigma^2 = e2 - e1^2
    ggml_tensor * attn3 = ggml_reshape_3d(ctx, attn, T, 1, Cm);
    ggml_tensor * e1    = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, attn3, ggml_reshape_3d(ctx, d, T, 1, Cm)), 1, Cm);
    ggml_tensor * e2    = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, attn3, ggml_reshape_3d(ctx, d2, T, 1, Cm)), 1, Cm);
    ggml_tensor * mu    = ggml_add(ctx, mean, e1);  // [1, Cm]
    ggml_tensor * sig   = ggml_sub(ctx, e2, ggml_sqr(ctx, e1));
    sig                 = ggml_sqrt(ctx, ggml_clamp(ctx, sig, hp.asp_eps, FLT_MAX));

    ggml_tensor * pooled =
        ggml_concat(ctx, ggml_reshape_1d(ctx, mu, Cm), ggml_reshape_1d(ctx, sig, Cm), /*dim=*/0);  // [2*Cm]
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

GraphBuild build_graph(ggml_context * ctx, const Model & model, int T) {
    GraphBuild gb{};

    const HParams & hp = model.hparams;
    const Weights & w  = model.weights;

    const int Cm    = hp.c_mfa();
    const int chunk = hp.c_chunk();

    gb.graph = ggml_new_graph_custom(ctx, kGraphSize, /*grads=*/false);

    // ---- inputs ----------------------------------------------------------
    gb.blk0_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.blk0_cols(), T);
    named(gb.blk0_in, "fe.mel.im2col");
    ggml_set_input(gb.blk0_in);

    for (int i = 0; i < kNumSeBlocks; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "reflect.idx.%d", i);
        gb.idx[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T + 2 * hp.pad(i + 1));
        named(gb.idx[i], name);
        ggml_set_input(gb.idx[i]);
    }
    gb.chunk_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kRes2NetScale);
    named(gb.chunk_ids, "res2.chunk_ids");
    ggml_set_input(gb.chunk_ids);

    // ---- stage 0: TDNNBlock n_mels -> C ----------------------------------
    // One matmul over the host im2col, inner dim padded to a multiple of 32
    // for the CPU tinyBLAS tiles.
    ggml_tensor * x = linear(ctx, w.blk0_w_im2col, gb.blk0_in, w.blk0_b);
    x               = ggml_relu(ctx, x);
    x               = bn_affine(ctx, x, w.blk0_bn.scale, w.blk0_bn.shift);
    mark_dump(gb.dumps.blk0_out, x, "enc.blk.0.out");

    // ---- stages 1..3: SERes2Net ------------------------------------------
    ggml_tensor * blk_out[kNumSeBlocks] = { nullptr, nullptr, nullptr };
    for (int i = 0; i < kNumSeBlocks; ++i) {
        const SeRes2NetBlock & blk = w.blocks[i];
        const int              d   = hp.dilations[static_cast<size_t>(i + 1)];

        ggml_tensor * residual = x;

        ggml_tensor * h = tdnn_1x1(ctx, blk.tdnn1, x);
        if (i == 0) {
            mark_dump(gb.dumps.blk1_tdnn1_out, h, "enc.blk.1.tdnn1.out");
        }

        // Res2Net writes in place; copy so the tdnn1 dump survives.
        if (i == 0 && debug::enabled()) {
            h = ggml_cont(ctx, h);
        }
        h = res2net(ctx, blk, h, gb.idx[i], gb.chunk_ids, d, T, chunk);
        if (i == 0) {
            mark_dump(gb.dumps.blk1_res2_out, h, "enc.blk.1.res2.out");
        }

        h = tdnn_1x1(ctx, blk.tdnn2, h);
        h = se_block(ctx, blk.se, h);
        if (i == 0) {
            mark_dump(gb.dumps.blk1_se_out, h, "enc.blk.1.se.out");
        }
        x = ggml_add(ctx, h, residual);

        char name[32];
        std::snprintf(name, sizeof(name), "enc.blk.%d.out", i + 1);
        mark_dump(gb.dumps.blk_out[i], x, name);
        blk_out[i] = x;
    }

    ggml_tensor * pooled = mfa_asp(ctx, gb, hp, w, blk_out, Cm);

    // ---- embedding head (asp_bn folded into fc) ---------------------------
    ggml_tensor * emb = linear(ctx, w.fc_w, pooled, w.fc_b);
    mark_dump(gb.dumps.emb, emb, "enc.emb");

    // ---- classifier -------------------------------------------------------
    // LeakyReLU precedes each (folded) BatchNorm.
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
