// arch/ecapa_tdnn/graph.cpp - the ECAPA-TDNN forward graph.
//
// Mirrors speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN.forward followed by
// Xvector.Classifier, with the four conversion-time rewrites already applied
// (scripts/convert-ecapa_tdnn.py): BatchNorm as an affine scale/shift, the three
// activation-free BatchNorms folded forward into the following linear map,
// and the ASP / MFA input concatenations split into per-operand weight
// blocks.
//
// Layout: every activation is ggml ne = [C, T]. Reductions over time need
// ggml's ne[0], so they run on a contiguous transposed copy [T, C] and the
// result is reshaped back. The three SE transposes and two ASP transposes
// per pass are negligible next to the 1x1 convolutions.
//
// Helper contract (named / linear / bn_affine / reflect_rows / conv_taps):
// an activation is ggml `ne = [C, T]`, CHANNEL-INNERMOST, frames are rows.
// That makes
//   - a 1x1 convolution a plain `ggml_mul_mat(w[IC, OC], x[IC, T])`,
//   - a bias / BatchNorm vector `[C]` a free broadcast over T,
//   - reflect padding a single `ggml_get_rows` over the frame axis,
//   - a dilated k>1 convolution a sum of matmuls against row-slices of the
//     padded activation, with no im2col and no transposes.
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
// configuration and is independent of T (TRANSCRIBE_ECAPA_GRAPH_STATS
// reports it).
constexpr size_t kGraphSize = 2048;

// Name a tensor, mark it as a graph output so the scheduler cannot reuse its
// buffer before the post-compute dump pass reads it (a no-op unless
// TRANSCRIBE_DUMP_DIR is set), and stash the pointer.
void mark_dump(ggml_tensor *& slot, ggml_tensor * t, const char * name) {
    named(t, name);
    debug::mark_tensor_for_dump(t);
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

}  // namespace

GraphBuild build_graph(ggml_context * ctx, const Model & model, int T) {
    GraphBuild gb{};

    const HParams & hp = model.hparams;
    const Weights & w  = model.weights;

    if (ctx == nullptr || T <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "ecapa_tdnn graph: invalid arg (ctx=%p, T=%d)", static_cast<void *>(ctx),
                T);
        return gb;
    }

    const int C     = hp.c_block();
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
    gb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.mel_n_mels, T);
    named(gb.mel_in, "fe.mel.in");
    ggml_set_input(gb.mel_in);

    for (int i = 0; i < kNumSeBlocks; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "reflect.idx.%d", i);
        gb.idx[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T + 2 * hp.pad(i + 1));
        named(gb.idx[i], name);
        ggml_set_input(gb.idx[i]);
    }

    // ---- stage 0: TDNNBlock n_mels -> C ----------------------------------
    // read_hparams guarantees pad(0) == pad(1), so idx[0] is the right gather.
    ggml_tensor * x =
        tdnn_conv(ctx, reflect_rows(ctx, gb.mel_in, gb.idx[0]), w.blk0_w, w.blk0_b, w.blk0_bn, hp.dilations[0], T);
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

        h = res2net(ctx, blk, h, gb.idx[i], d, T, chunk);
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

    // ---- MFA: 1x1 over concat(blk1, blk2, blk3), split into three blocks --
    ggml_tensor * m = nullptr;
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

    ggml_tensor * pooled =
        ggml_concat(ctx, ggml_reshape_1d(ctx, mu, Cm), ggml_reshape_1d(ctx, sig, Cm), /*dim=*/0);  // [2*Cm]
    mark_dump(gb.dumps.asp_out, pooled, "enc.asp.out");

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

    (void) C;
    return gb;
}

}  // namespace transcribe::ecapa_tdnn
