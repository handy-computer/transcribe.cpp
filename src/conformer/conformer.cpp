// src/conformer/conformer.cpp - shared Conformer encoder helpers.
//
// Extracted from parakeet/encoder.cpp and cohere/encoder.cpp. The two
// files had grown to ~70% overlap, and the per-op helpers (layer_norm,
// feed_forward, the f32-friendly conv helpers, rel_shift, conv_module,
// rel_pos_mhsa, build_conformer_block, build_pre_encode) were byte-
// identical or trivially parameterizable. This module owns the
// canonical implementation; each family's encoder.cpp keeps only the
// glue that binds the shared helpers to its own weights / hparams /
// graph-driver contract.
//
// Per-family policy knobs (ConvPolicy, BlockParams) are passed in by
// the family — no environment variables are read here except by the
// shared detect_direct_pw, which is identical across families.

#include "conformer/conformer.h"
#include "transcribe-log.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace transcribe::conformer {

// ===========================================================================
// Low-level helpers
// ===========================================================================

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr) {
        ggml_set_name(t, name);
    }
    return t;
}

// LayerNorm with affine: y = gamma * (x - mean) / sqrt(var + eps) + beta.
//
// ggml_norm normalizes along ne[0], which for our [d_model, T, B]
// encoder activation is the d_model axis — exactly what LayerNorm
// wants. The affine multiply + add use ggml's broadcasting: gamma /
// beta have ne=[d_model, 1, 1, 1], which broadcasts naturally across
// the T and B axes of the activation.
//
// `gamma` is required; `beta` may be null for bias-free LN. Both
// Parakeet and Cohere ship with affine (gamma + beta) on every LN,
// but the helper supports no-bias for future families.
ggml_tensor * layer_norm(ggml_context * ctx,
                         ggml_tensor *  x,
                         ggml_tensor *  gamma,
                         ggml_tensor *  beta)
{
    ggml_tensor * y = ggml_norm(ctx, x, kLayerNormEps);
    y = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) {
        y = ggml_add(ctx, y, beta);
    }
    return y;
}

// FeedForward: y = Linear2(SiLU(Linear1(x)))
//
// ggml_mul_mat(W, x) reads W ne[0] as the input dim, so with
// W ne=[d_in, d_out], x ne=[d_in, T, B] -> result ne=[d_out, T, B].
// Both converters store ff*_lin1_w as ne=[d_model, d_ff] and
// ff*_lin2_w as ne=[d_ff, d_model] to match this convention. Bias
// tensors, when present, are [d_out] and broadcast along T and B.
//
// Either or both biases may be null — the optional-bias pattern is
// what lets Parakeet (bias-free) and Cohere (all biases) share the
// same code.
ggml_tensor * feed_forward(ggml_context * ctx,
                           ggml_tensor *  x,
                           ggml_tensor *  lin1_w,
                           ggml_tensor *  lin1_b,
                           ggml_tensor *  lin2_w,
                           ggml_tensor *  lin2_b)
{
    ggml_tensor * y = ggml_mul_mat(ctx, lin1_w, x);
    if (lin1_b != nullptr) {
        y = ggml_add(ctx, y, lin1_b);
    }
    y = ggml_silu(ctx, y);
    y = ggml_mul_mat(ctx, lin2_w, y);
    if (lin2_b != nullptr) {
        y = ggml_add(ctx, y, lin2_b);
    }
    return y;
}

// Macaron half-residual feed-forward: x + 0.5 * FF(LN(x)).
// Used at the start (FF1) and end (FF2) of each conformer block.
// The 0.5 scale is the macaron weighting; the alternative residual
// (full 1.0) is used for the MHSA and conv sub-blocks.
ggml_tensor * macaron_ff_residual(ggml_context * ctx,
                                  ggml_tensor *  x,
                                  ggml_tensor *  norm_w,
                                  ggml_tensor *  norm_b,
                                  ggml_tensor *  lin1_w,
                                  ggml_tensor *  lin1_b,
                                  ggml_tensor *  lin2_w,
                                  ggml_tensor *  lin2_b)
{
    ggml_tensor * y = layer_norm(ctx, x, norm_w, norm_b);
    y = feed_forward(ctx, y, lin1_w, lin1_b, lin2_w, lin2_b);
    y = ggml_scale(ctx, y, 0.5f);
    return ggml_add(ctx, x, y);
}

// Shaw / Transformer-XL relative-position skew. Transforms the
// position scoring matrix from [pos_len, T_q, H, 1] to the same shape
// but with each row rotated so that column k holds the score for
// relative offset k.
ggml_tensor * rel_shift(ggml_context * ctx, ggml_tensor * x) {
    const int64_t pos_len = x->ne[0];
    const int64_t T_q     = x->ne[1];
    const int64_t H       = x->ne[2];
    const int64_t B       = x->ne[3];

    // Step 1: prepend a [1, T_q, H, B] column of zeros along axis 0.
    // ggml_fill takes a shape template (need not be backed by real
    // data — only the metadata is used) and emits a new tensor of
    // the same shape filled with the given constant. ggml_concat
    // along dim=0 produces [pos_len+1, T_q, H, B] with the first
    // axis-0 element being our zero column.
    ggml_tensor * zero_template = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                     1, T_q, H, B);
    ggml_tensor * zeros = ggml_fill(ctx, zero_template, 0.0f);
    ggml_tensor * y = ggml_concat(ctx, zeros, x, /*dim=*/0);

    // Step 2: reshape (swap inner two axes; total elements unchanged).
    y = ggml_reshape_4d(ctx, y, T_q, pos_len + 1, H, B);

    // Step 3: drop the first row of axis 1.
    y = ggml_view_4d(ctx, y, T_q, pos_len, H, B,
                     y->nb[1], y->nb[2], y->nb[3],
                     /*offset=*/y->nb[1]);

    // Step 4: materialize as contiguous (the view above is sparse:
    // its nb[1] still reflects the parent's stride which assumes
    // pos_len+1 elements per row, so reshape can't read it directly).
    y = ggml_cont(ctx, y);

    // Step 5: reshape back to [pos_len, T_q, H, B].
    y = ggml_reshape_4d(ctx, y, pos_len, T_q, H, B);
    return y;
}

// f32-friendly Conv1D (mirrors ggml_conv_1d but passes the kernel's
// real type to im2col instead of forcing f16). The vendored ggml's
// ggml_conv_1d hardcodes GGML_TYPE_F16 for the im2col output, which
// triggers a runtime assertion when fed an f32 kernel — and both the
// Parakeet and Cohere weight catalogs are fp32 for these convs.
ggml_tensor * conv_1d_f32(ggml_context * ctx,
                          ggml_tensor *  kernel,
                          ggml_tensor *  data,
                          int            stride,
                          int            padding,
                          int            dilation)
{
    // im2col: kernel shape [K, IC, OC] in ggml ne; data shape
    // [W, IC, N] in ggml ne. Output ne[0] = IC*K, ne[1] = OW (output
    // width), ne[2] = N (batch).
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, data,
                                       stride, /*s1=*/0,
                                       padding, /*p1=*/0,
                                       dilation, /*d1=*/0,
                                       /*is_2D=*/false,
                                       /*dst_type=*/kernel->type);
    // im2col ne = [IC*K, OW, N].

    const int64_t N = im2col->ne[2];
    ggml_tensor * kernel_2d = ggml_reshape_2d(ctx, kernel,
                                              kernel->ne[0] * kernel->ne[1],
                                              kernel->ne[2]);  // [IC*K, OC]

    // F32 accumulation when the kernel is F16. The same CUDA cuBLAS
    // COMPUTE_16F saturation that hits the in-block pointwise convs also
    // hits this im2col + mul_mat for the subsampling conv stem when the
    // converter routes those kernels to F16 (medasr / cohere / parakeet).
    const bool kernel_needs_f32_acc = (kernel->type == GGML_TYPE_F16);

    if (N == 1) {
        // Single-shot path (bit-identical to the pre-batch code): flatten
        // OW*N (== OW) and reshape the [OW, OC] result back to [OW, OC, 1].
        ggml_tensor * result = ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1]),
            kernel_2d);  // [OW, OC]
        if (kernel_needs_f32_acc) {
            ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        }
        result = ggml_reshape_3d(ctx, result,
                                 im2col->ne[1], kernel->ne[2], 1);
        return result;
    }

    // Batched path. Flattening OW*N interleaves batch with width
    // (ow_n = b*OW + ow), so a [OW*N, OC] -> [OW, OC, N] reshape would
    // mislabel memory. Instead mul_mat the 3-D im2col directly: the kernel
    // (ne[2]=1) broadcasts across the batch, giving [OC, OW, N], then
    // permute to the [OW, OC, N] = [time, channels, batch] convention. The
    // per-element dot products match the N==1 path exactly (same reduction
    // over IC*K), so each utterance is bit-identical to single-shot.
    ggml_tensor * result = ggml_mul_mat(ctx, kernel_2d, im2col);  // [OC, OW, N]
    if (kernel_needs_f32_acc) {
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
    return result;  // [OW, OC, N]
}

// f32-friendly 2D depthwise conv. Mirrors ggml_conv_2d_dw exactly
// but passes the kernel's real type to im2col instead of forcing
// f16. The vendored ggml's ggml_conv_2d_dw hardcodes
// GGML_TYPE_F16 for the im2col output, which then makes the
// downstream matmul try to use a `kernel_mul_mv_f32_f16_short`
// Metal kernel that isn't compiled into this ggml's Metal
// library. Forcing f32 lands the matmul on
// `kernel_mul_mv_f32_f32_short` which IS available, unblocking
// the Metal path.
//
// The alternative `ggml_conv_2d_dw_direct` hits an even harder
// wall: its CONV_2D_DW op is reported as unsupported by
// ggml_metal_op_encode_impl, so it can't run on Metal at all in
// this version. The im2col + mul_mat path here is what works.
ggml_tensor * conv_2d_dw_f32(ggml_context * ctx,
                             ggml_tensor *  kernel,  // [KW, KH, 1, C]
                             ggml_tensor *  data,    // [W, H, C, N]
                             int            s0, int s1,
                             int            p0, int p1,
                             int            d0, int d1)
{
    // Same op sequence as ggml_conv_2d_dw, only dst_type changes.
    ggml_tensor * new_a = ggml_reshape_4d(ctx, kernel,
        kernel->ne[0], kernel->ne[1], 1,
        kernel->ne[2] * kernel->ne[3]);
    ggml_tensor * data_4d = ggml_reshape_4d(ctx, data,
        data->ne[0], data->ne[1], 1,
        data->ne[2] * data->ne[3]);
    ggml_tensor * im2col = ggml_im2col(ctx, new_a, data_4d,
        s0, s1, p0, p1, d0, d1,
        /*is_2D=*/true,
        /*dst_type=*/kernel->type);
    ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col,
        im2col->ne[0], im2col->ne[2] * im2col->ne[1],
        data->ne[2], data->ne[3]);
    new_a = ggml_reshape_4d(ctx, new_a,
        new_a->ne[0] * new_a->ne[1], new_a->ne[2], new_a->ne[3], 1);
    ggml_tensor * result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result,
        im2col->ne[1], im2col->ne[2],
        data->ne[2], data->ne[3]);
    return result;
}

// f32-friendly depthwise Conv1D (mirrors ggml_conv_1d_dw but
// passes the kernel's real type to im2col). Same fix as above.
ggml_tensor * conv_1d_dw_f32(ggml_context * ctx,
                             ggml_tensor *  kernel,
                             ggml_tensor *  data,
                             int            stride,
                             int            padding,
                             int            dilation)
{
    // Offline batch: data ne=[W, C, B] with B>1. The im2col path below
    // collapses the batch axis (its final reshape hardcodes ne[2]=1), so
    // route B>1 through the direct depthwise-2D op (W, H=1, C, N=B), which
    // threads the utterance batch at ne[3]. This keeps the B==1 path —
    // every current caller (parakeet/cohere single-shot, AR granite) —
    // byte-identical to the pre-batch code.
    const int64_t B = data->ne[2];
    if (B > 1) {
        const int64_t k = kernel->ne[0];
        const int64_t C = data->ne[1];
        // ggml_conv_2d_dw_direct misbehaves for non-f32 kernels (see the
        // canary_qwen note); the depthwise kernel is tiny, so cast to f32.
        ggml_tensor * knl = kernel;
        if (knl->type != GGML_TYPE_F32) {
            knl = ggml_cast(ctx, knl, GGML_TYPE_F32);
        }
        knl = ggml_reshape_4d(ctx, knl, k, 1, 1, C);
        ggml_tensor * d4 = ggml_reshape_4d(ctx, data, data->ne[0], 1, C, B);
        ggml_tensor * o = ggml_conv_2d_dw_direct(ctx, knl, d4,
                                                 /*s0=*/stride, /*s1=*/1,
                                                 /*p0=*/padding, /*p1=*/0,
                                                 /*d0=*/dilation, /*d1=*/1);
        return ggml_reshape_3d(ctx, o, o->ne[0], C, B);  // [W_out, C, B]
    }

    // The depthwise wrapper reshapes the data into a 2D image with
    // H=1 so the same im2col path can be reused. ggml_conv_1d_dw
    // does this; we replicate it here with the dst_type override.
    ggml_tensor * data_4d = ggml_reshape_4d(ctx, data,
                                            data->ne[0], 1,
                                            data->ne[1], data->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, data_4d,
                                       stride, /*s1=*/0,
                                       padding, /*p1=*/0,
                                       dilation, /*d1=*/0,
                                       /*is_2D=*/false,
                                       /*dst_type=*/kernel->type);
    ggml_tensor * result = ggml_mul_mat(ctx, im2col, kernel);
    result = ggml_reshape_3d(ctx, result,
                             result->ne[0], result->ne[2], 1);
    return result;
}

// Fused BatchNorm: y = x * scale + bias, where scale and bias are
// precomputed at load time from the raw BN parameters. Replaces the
// 4-op batch_norm (sub, div/sqrt, mul, add) with 2 ops (mul, add).
// The 1-D tensors [d_model] are reshaped to [1, d_model, 1, 1] to
// broadcast across the time axis of [T, d_model, 1, 1].
ggml_tensor * fused_batch_norm(ggml_context * ctx,
                               ggml_tensor *  x,
                               ggml_tensor *  scale_1d,
                               ggml_tensor *  bias_1d)
{
    const int64_t C = scale_1d->ne[0];
    ggml_tensor * scale_4d = ggml_reshape_4d(ctx, scale_1d, 1, C, 1, 1);
    ggml_tensor * bias_4d  = ggml_reshape_4d(ctx, bias_1d,  1, C, 1, 1);
    ggml_tensor * y = ggml_mul(ctx, x, scale_4d);
    y = ggml_add(ctx, y, bias_4d);
    return y;
}

// Add a 1-D bias [C] to a 4-D conv output [W, H, C, N] by reshaping
// the bias to [1, 1, C, 1] so ggml_add broadcasts the W, H, and N
// axes for free. The reshape is a metadata-only view because the
// 1-D tensor is contiguous; no data copy.
ggml_tensor * add_conv_bias(ggml_context * ctx,
                            ggml_tensor *  conv_out,
                            ggml_tensor *  bias_1d)
{
    if (bias_1d == nullptr) return conv_out;
    const int64_t channels = bias_1d->ne[0];
    ggml_tensor * bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, 1, channels, 1);
    return ggml_add(ctx, conv_out, bias_4d);
}

// ===========================================================================
// Policy detection
// ===========================================================================

bool detect_direct_pw(const char * backend) {
    const char * env = std::getenv("TRANSCRIBE_CONV_NO_DIRECT_PW");
    if (env != nullptr) return false; // user override
    env = std::getenv("TRANSCRIBE_CONV_DIRECT_PW");
    if (env != nullptr) return true;  // user override
    // Vulkan: controlled A/B on AMD Renoir showed the im2col + mul_mat
    // path is ~200 ms per encode faster than the direct mul_mat path
    // for f32 weights, so keep im2col as the default. Metal and CPU
    // benefit from the direct path.
    if (backend != nullptr && std::strstr(backend, "Vulkan") != nullptr) {
        return false;
    }
    return true;
}

// ===========================================================================
// Conv module (pointwise -> GLU -> depthwise -> BN -> SiLU -> pointwise)
// ===========================================================================

// Operates on the post-LayerNorm activation; the LN is applied OUTSIDE
// this helper by the block forward. By default, pointwise convs (k=1)
// are direct mul_mats in [d_model, T] layout, avoiding im2col overhead.
// Set TRANSCRIBE_CONV_NO_DIRECT_PW=1 to fall back to the im2col path.
// BatchNorm uses precomputed fused scale + bias (2 ops instead of 4);
// LayerNorm uses an unfused on-the-fly per-channel normalize plus
// affine, with weights from BlockView::conv_ln_*.
ggml_tensor * conv_module(ggml_context *      ctx,
                          ggml_tensor *       x,
                          const BlockView &   b,
                          const BlockParams & params)
{
    const int          conv_kernel = params.conv_kernel;
    const ConvPolicy & policy      = params.policy;

    // Centred (k-1)/2 by default; (conv_context_left, conv_context_right)
    // exactly when both are >= 0 (caller asked for causal or otherwise
    // asymmetric padding).
    int pad_left  = (conv_kernel - 1) / 2;
    int pad_right = (conv_kernel - 1) / 2;
    if (params.conv_context_left >= 0 && params.conv_context_right >= 0) {
        pad_left  = params.conv_context_left;
        pad_right = params.conv_context_right;
    }

    const int64_t d_model = x->ne[0];
    // Utterance batch lives at ne[2] of the [d_model, T, B] activation.
    // B == 1 for single-shot (parakeet / cohere today); the offline
    // batched encoder passes B > 1. Every view/reshape below threads B so
    // the B == 1 path is bit-identical to the pre-batch code.
    const int64_t B = x->ne[2];

    if (policy.direct_pw) {
        // Pointwise conv 1 as direct mul_mat in [d_model, T, B] layout.
        // Kernel ne=[1, d_model, 2*d_model] → reshape to [d_model, 2*d_model].
        // F32 accumulation when the weight is F16: on CUDA, F16 weights take
        // the cuBLAS COMPUTE_16F path whose accumulator saturates at ~6.5e4.
        // Families with large activation magnitudes (e.g. medasr's scaled
        // residuals push intermediates to ~2e6) overflow to NaN. set_prec
        // forces F32 accumulation; no-op on CPU/Metal, slight perf cost on
        // CUDA. Skip for non-F16 weights (BF16 already COMPUTE_32F; quant
        // dispatches MMQ).
        {
            ggml_tensor * pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w,
                                                d_model, 2 * d_model);
            x = ggml_mul_mat(ctx, pw1, x);  // [2*d_model, T, B]
            if (b.conv_pw1_w->type == GGML_TYPE_F16) {
                ggml_mul_mat_set_prec(x, GGML_PREC_F32);
            }
            if (b.conv_pw1_b != nullptr) {
                x = ggml_add(ctx, x, b.conv_pw1_b);
            }
        }

        // GLU: split ne[0] in half, gate * sigmoid(value). The views carry
        // the batch axis (ne[2]) so the split is per-utterance.
        {
            const int64_t T    = x->ne[1];
            const int64_t half = x->ne[0] / 2;
            ggml_tensor * gate  = ggml_view_3d(ctx, x, half, T, B,
                                               x->nb[1], x->nb[2],
                                               /*offset=*/0);
            ggml_tensor * value = ggml_view_3d(ctx, x, half, T, B,
                                               x->nb[1], x->nb[2],
                                               half * ggml_element_size(x));
            x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
        }
        // x ne = [d_model, T, B]

        // Transpose for depthwise conv: [d_model, T, B] -> [T, d_model, B].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    } else {
        // Fallback: im2col path in [T, d_model] layout.
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        x = conv_1d_f32(ctx, b.conv_pw1_w, x, /*s=*/1, /*p=*/0, /*d=*/1);

        // Add pointwise1 bias in [T, 2*d_model] layout.
        if (b.conv_pw1_b != nullptr) {
            // conv_1d_f32 returns ne=[T, 2*d_model, 1]. Reshape the
            // 1-D bias to [1, 2*d_model] so ggml_add broadcasts.
            x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
            ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_pw1_b,
                                                   1, 2 * d_model);
            x = ggml_add(ctx, x, bias_r);
        }

        const int64_t half = x->ne[1] / 2;
        ggml_tensor * gate = ggml_view_4d(ctx, x,
                                          x->ne[0], half, 1, 1,
                                          x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * value = ggml_view_4d(ctx, x,
                                           x->ne[0], half, 1, 1,
                                           x->nb[1], x->nb[2], x->nb[3],
                                           x->nb[1] * half);
        x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));

        x = ggml_cont(ctx, x);
    }

    // NeMo zeros padded-overhang frames after pointwise_conv1 + GLU and
    // before the depthwise convolution. Without this, buffered streaming
    // overhang frames can leak backward through the conv kernel.
    if (params.conv_pad_mask != nullptr) {
        x = ggml_mul(ctx, x, params.conv_pad_mask);
    }

    // Depthwise conv: kernel size from hparams, groups=d_model. The
    // ggml depthwise ops accept a single symmetric padding value on
    // each axis; for asymmetric padding (causal: [k-1, 0]) we prepend
    // / append zero-frames along the time axis explicitly and call the
    // op with p0=0. At this point x has ne = [T, d_model, 1, 1] (the
    // permute above moved T into ne[0]), so we concat along dim 0.
    //
    // Streaming carry: when params.streaming_time_in is set, the left
    // pad is the previous chunk's last (k-1) post-pw1+GLU frames
    // (NeMo's cache_last_time) instead of zeros. The new cache slot
    // for the next chunk is the LAST pad_left frames of the
    // concatenated (prev_cache, current_x) sequence — NeMo's
    //   cache_next = concat(prev_cache, x)[-pad_left:]
    // which works regardless of whether T_now >= pad_left. The
    // earlier "tail of x" formulation broke at small chunk sizes
    // (R<13 in nemotron-streaming produces T_q_new < pad_left=8).
    const bool symmetric_pad = (pad_left == pad_right);
    if (!symmetric_pad) {
        const bool streaming = (params.streaming_time_in != nullptr);

        if (streaming) {
            if (pad_left > 0) {
                // streaming_time_in has ne = [pad_left, d_model, 1, 1]
                // (matches the zero-pad shape on this axis).
                x = ggml_concat(ctx, params.streaming_time_in, x, /*dim=*/0);
            }

            // After the concat, x ne = [pad_left + T_now, d_model, 1, 1].
            // Take the LAST pad_left frames as the next-chunk cache slot.
            // pad_left always <= x->ne[0] here because the concat above
            // contributes exactly pad_left frames of prev_cache.
            if (params.streaming_time_out != nullptr &&
                params.streaming_graph != nullptr)
            {
                const int64_t T_padded = x->ne[0];
                ggml_tensor * tail = ggml_view_4d(
                    ctx, x,
                    pad_left, x->ne[1], x->ne[2], x->ne[3],
                    x->nb[1], x->nb[2], x->nb[3],
                    (T_padded - pad_left) * x->nb[0]);
                ggml_tensor * cpy = ggml_cpy(ctx, tail,
                                             params.streaming_time_out);
                // Cache writes are SIDE outputs of the encoder graph:
                // not reachable from `eb.out`, so the scheduler won't
                // schedule them unless we expand them explicitly.
                ggml_build_forward_expand(params.streaming_graph, cpy);
            }
        } else if (pad_left > 0) {
            ggml_tensor * pad_l = ggml_new_tensor_4d(ctx, x->type,
                                                     pad_left, x->ne[1],
                                                     x->ne[2], x->ne[3]);
            pad_l = ggml_fill(ctx, pad_l, 0.0f);
            x = ggml_concat(ctx, pad_l, x, /*dim=*/0);
        }
        if (pad_right > 0) {
            ggml_tensor * pad_r = ggml_new_tensor_4d(ctx, x->type,
                                                     pad_right, x->ne[1],
                                                     x->ne[2], x->ne[3]);
            pad_r = ggml_fill(ctx, pad_r, 0.0f);
            x = ggml_concat(ctx, x, pad_r, /*dim=*/0);
        }
    }
    const int padding_op = symmetric_pad ? pad_left : 0;
    if (policy.direct_dw_in_block) {
        // Fused single-op depthwise conv (no im2col). Input is [T, d_model, B]
        // from the transpose above. Reshape to 4D [T, 1, d_model, B] for
        // ggml_conv_2d_dw_direct which expects [W, H, C, N] (N == B, the
        // utterance batch). Kernel [k, 1, d_model] → [k, 1, 1, d_model].
        ggml_tensor * knl = ggml_reshape_4d(ctx, b.conv_dw_w,
                                            conv_kernel, 1, 1, d_model);
        ggml_tensor * data = ggml_reshape_4d(ctx, x,
                                             x->ne[0], 1, x->ne[1], B);
        x = ggml_conv_2d_dw_direct(ctx, knl, data,
                                   /*s0=*/1, /*s1=*/1,
                                   /*p0=*/padding_op, /*p1=*/0,
                                   /*d0=*/1, /*d1=*/1);
        // Output: [T_out, 1, d_model, B] → [T_out, d_model, B].
        x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[2], x->ne[3]);
    } else {
        // im2col depthwise (cohere on Metal/CPU); single-utterance only.
        x = conv_1d_dw_f32(ctx, b.conv_dw_w, x,
                           /*s=*/1, /*p=*/padding_op, /*d=*/1);
    }

    // Add depthwise bias in [T, d_model] layout (nullable).
    if (b.conv_dw_b != nullptr) {
        ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_dw_b, 1, d_model);
        x = ggml_add(ctx, x, bias_r);
    }

    // Post-depthwise normalisation: BN (fused mul + add) or LN
    // (per-channel mean/std normalize + affine). Both produce the same
    // [T, d_model] shape; the SiLU below is identical.
    //
    // At this point x has ne = [T, d_model, 1, 1]. NeMo's LayerNorm
    // normalises over the feature axis (d_model), matching the
    // `layer_norm` helper exactly when applied to a [T, d_model]
    // tensor — but `layer_norm` reads ne[0] as the feature axis, so we
    // permute to [d_model, T] first, apply, and permute back.
    if (params.conv_norm_type == BlockParams::ConvNormType::LayerNorm) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        x = layer_norm(ctx, x, b.conv_ln_w, b.conv_ln_b);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    } else {
        x = fused_batch_norm(ctx, x,
                             b.conv_bn_fused_scale, b.conv_bn_fused_bias);
    }

    // SiLU activation.
    x = ggml_silu(ctx, x);

    if (policy.direct_pw) {
        // Transpose back: [T, d_model] -> [d_model, T].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        // Pointwise conv 2 as direct mul_mat in [d_model, T] layout. See
        // the pw1 comment above for the F16 / CUDA COMPUTE_16F rationale.
        ggml_tensor * pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w,
                                            d_model, d_model);
        x = ggml_mul_mat(ctx, pw2, x);
        if (b.conv_pw2_w->type == GGML_TYPE_F16) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        if (b.conv_pw2_b != nullptr) {
            x = ggml_add(ctx, x, b.conv_pw2_b);
        }
    } else {
        x = conv_1d_f32(ctx, b.conv_pw2_w, x, /*s=*/1, /*p=*/0, /*d=*/1);
        if (b.conv_pw2_b != nullptr) {
            x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
            ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_pw2_b,
                                                   1, d_model);
            x = ggml_add(ctx, x, bias_r);
        }
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    }

    return x;
}

// ===========================================================================
// Relative-position multi-head self-attention
// ===========================================================================

// use_flash = true  -> ggml_flash_attn_ext (fused GPU kernel, needs
//                      a dk template in the Metal backend)
// use_flash = false -> manual mul_mat + soft_max_ext + mul_mat (works
//                      on any backend; use this when the head dim has
//                      no flash kernel, e.g. Cohere's encoder dk=160)
//
// ggml_flash_attn_ext expected layouts:
//   Q:    [head_dim, T_q,    n_head, 1]
//   K:    [head_dim, T_k,    n_head, 1]
//   V:    [head_dim, T_k,    n_head, 1]  (NOT transposed)
//   mask: [T_k,      T_q,    n_head, 1]  F16, additive
//   out:  [head_dim, n_head, T_q,    1]  (already permuted)
//
// Score formula:
//   flash_attn computes: softmax(q_u @ k^T * scale + mask) @ v
//   We set mask = rel_shift(q_v @ p^T)[:T_k] * scale
//   Which gives: softmax((q_u @ k^T + rel_pos_bias) * scale) @ v
ggml_tensor * rel_pos_mhsa(ggml_context *      ctx,
                           ggml_tensor *       x,
                           ggml_tensor *       pos_emb,
                           const BlockView &   b,
                           const BlockParams & params)
{
    const int     d_model  = params.d_model;
    const int     n_head   = params.n_head;
    const ggml_type kv_type  = params.kv_type;
    const bool      use_flash = params.use_flash;
    const int     att_context_left  = params.att_context_left;
    const int     att_context_right = params.att_context_right;
    const int     head_dim = d_model / n_head;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T_q      = x->ne[1];
    const int64_t pos_len  = pos_emb->ne[1];
    // Utterance batch at ne[2] of the [d_model, T, B] activation. After the
    // head split below the batch moves to ne[3] (heads take ne[2]). When
    // batched we force the manual attention path: ggml_flash_attn_ext's
    // additive mask (matrix_bd, which is content-dependent and therefore
    // differs per utterance) is not validated for a per-batch ne[3] here,
    // whereas the manual mul_mat + soft_max path broadcasts over the batch
    // cleanly. Single-shot (B == 1) keeps its existing flash path exactly.
    const int64_t B         = x->ne[2];
    // Flash attention works batched: ggml_flash_attn_ext accepts the
    // per-utterance rel-pos mask at ne[3] == B, and the batched encoder
    // output is bit-identical to single-shot flash (verified on CPU and
    // Metal). So the batch axis does not change the attention path.
    // TRANSCRIBE_NO_FLASH=1 forces the manual mul_mat + soft_max path for
    // both single-shot and batched (used by the bit-exact CPU tensor gate,
    // since flash casts the rel-pos mask to F16 while manual stays F32).
    const bool    flash     = use_flash;

    // Local-attention bookkeeping. With both window sides non-negative
    // in the Regular style, pos_emb arrives at the smaller
    // [left+right+1, d] length and matrix_bd needs zero/-inf padding to
    // keep the rel_shift trick intact. ChunkedLimited keeps pos_emb at
    // its full 2T-1 length and uses the external chunked mask instead.
    const bool is_chunked =
        (params.att_context_style == BlockParams::AttContextStyle::ChunkedLimited);
    const bool is_local =
        (!is_chunked) && (att_context_left >= 0 && att_context_right >= 0);
    const int  W_left  = is_local ? att_context_left  : 0;
    const int  W_right = is_local ? att_context_right : 0;

    // ----- Q, K, V, P projections ---------------------------------
    // Q, K, V may have bias (Cohere) or not (Parakeet). Pos projection
    // (attn_pos_w) never has a bias in either family.
    ggml_tensor * q = ggml_mul_mat(ctx, b.attn_q_w, x);
    if (b.attn_q_b != nullptr) q = ggml_add(ctx, q, b.attn_q_b);
    ggml_tensor * k = ggml_mul_mat(ctx, b.attn_k_w, x);
    if (b.attn_k_b != nullptr) k = ggml_add(ctx, k, b.attn_k_b);
    ggml_tensor * v = ggml_mul_mat(ctx, b.attn_v_w, x);
    if (b.attn_v_b != nullptr) v = ggml_add(ctx, v, b.attn_v_b);
    ggml_tensor * p = ggml_mul_mat(ctx, b.attn_pos_w, pos_emb);

    // ----- Split heads --------------------------------------------
    // pos_bias_u/v broadcast onto [head_dim, n_head, T, 1] BEFORE
    // the permute that moves T past n_head.
    // Head split carries the batch axis at ne[3]: [head_dim, n_head, T, B].
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, T_q, B);
    ggml_tensor * q_u = ggml_add(ctx, q, b.attn_pos_u);
    ggml_tensor * q_v = ggml_add(ctx, q, b.attn_pos_v);

    // All of Q, K, V need [head_dim, T, n_head, B] for the attention. The
    // permute is a zero-cost view (no data copy). flash_attn_ext accesses
    // data via strides, so contiguous materialization (cont) is not
    // required for q_u/k/v. q_v and p still need cont because they feed
    // into mul_mat for the position score computation.
    q_u = ggml_permute(ctx, q_u, 0, 2, 1, 3);
    q_v = ggml_cont(ctx, ggml_permute(ctx, q_v, 0, 2, 1, 3));

    k = ggml_reshape_4d(ctx, k, head_dim, n_head, T_q, B);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);

    v = ggml_reshape_4d(ctx, v, head_dim, n_head, T_q, B);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Position scores are batch-independent (pos_emb has no batch axis), so
    // p keeps ne[3] == 1 and broadcasts across the batch in the mul_mat.
    p = ggml_reshape_4d(ctx, p, head_dim, n_head, pos_len, 1);
    p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    // ----- Position mask / bias -----------------------------------
    // matrix_bd = rel_shift(q_v @ p^T), truncated to [T_q, T_q].
    ggml_tensor * matrix_bd = ggml_mul_mat(ctx, p, q_v);

    // Local-attention pad/slice. The standard rel_shift trick assumes
    // matrix_bd has shape [2T_q-1, T_q]: row r corresponds to relative
    // offset (T_q-1-r). For local attention pos_emb is shorter
    // ([W_left+W_right+1]) where row r corresponds to offset (W_left-r).
    // Bring matrix_bd back to the [2T_q-1, T_q] shape by:
    //   - prepending (T_q-1-W_left) rows of -INF (or slicing them off
    //     when the audio is so short the window already covers it),
    //   - appending  (T_q-1-W_right) rows of -INF (or slicing).
    // After this, rel_shift + the existing T_q×T_q view land each
    // out-of-window position at -INF, which softmax zeroes out. With
    // both window sides == -1 (full attention) this block is skipped.
    if (is_local) {
        const int top_pad = static_cast<int>(T_q) - 1 - W_left;
        if (top_pad > 0) {
            ggml_tensor * top_template = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F32, top_pad, T_q, n_head, B);
            ggml_tensor * top = ggml_fill(ctx, top_template, -INFINITY);
            matrix_bd = ggml_concat(ctx, top, matrix_bd, /*dim=*/0);
        } else if (top_pad < 0) {
            const int kept = static_cast<int>(matrix_bd->ne[0]) + top_pad;
            matrix_bd = ggml_view_4d(ctx, matrix_bd,
                                     kept, T_q, n_head, B,
                                     matrix_bd->nb[1], matrix_bd->nb[2],
                                     matrix_bd->nb[3],
                                     (-top_pad) * matrix_bd->nb[0]);
            matrix_bd = ggml_cont(ctx, matrix_bd);
        }
        const int bot_pad = static_cast<int>(T_q) - 1 - W_right;
        if (bot_pad > 0) {
            ggml_tensor * bot_template = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F32, bot_pad, T_q, n_head, B);
            ggml_tensor * bot = ggml_fill(ctx, bot_template, -INFINITY);
            matrix_bd = ggml_concat(ctx, matrix_bd, bot, /*dim=*/0);
        } else if (bot_pad < 0) {
            const int kept = static_cast<int>(matrix_bd->ne[0]) + bot_pad;
            matrix_bd = ggml_view_4d(ctx, matrix_bd,
                                     kept, T_q, n_head, B,
                                     matrix_bd->nb[1], matrix_bd->nb[2],
                                     matrix_bd->nb[3], /*offset=*/0);
            matrix_bd = ggml_cont(ctx, matrix_bd);
        }
    }

    matrix_bd = rel_shift(ctx, matrix_bd);
    matrix_bd = ggml_view_4d(ctx, matrix_bd,
                             T_q, T_q, n_head, B,
                             matrix_bd->nb[1], matrix_bd->nb[2],
                             matrix_bd->nb[3], /*offset=*/0);
    // The view is non-contiguous (nb[1] stays at parent's
    // pos_len*es), but it IS contiguous-rows. The flash path calls
    // ggml_scale which wants full contiguity, so cont there. The
    // manual path only feeds matrix_bd into ggml_add(kq, matrix_bd),
    // which handles contiguous-rows inputs on both CPU and Metal.
    if (flash) {
        matrix_bd = ggml_cont(ctx, matrix_bd);
    }

    // ChunkedLimited mask. Caller provided a [T_q, T_q, 1, 1] F32
    // tensor with 0 on allowed (q, k) pairs and -INF outside the
    // [q_chunk - left_chunks, q_chunk] band. Broadcasts across n_head.
    // -INF and 0 are scale-invariant so this can be added before the
    // pre-scale that the flash path applies below.
    if (is_chunked && params.attn_chunked_mask != nullptr) {
        matrix_bd = ggml_add(ctx, matrix_bd, params.attn_chunked_mask);
    }

    // Variable-length batch key-padding mask. [T_k, 1, 1, B] additive
    // (-INF on padded keys) broadcasts over queries and heads. Added here
    // so it applies on both the flash and manual paths (matrix_bd is the
    // flash mask and the manual additive bias alike). -INF / 0 are
    // scale-invariant, so adding before the flash pre-scale is fine.
    if (params.attn_pad_mask != nullptr) {
        matrix_bd = ggml_add(ctx, matrix_bd, params.attn_pad_mask);
    }

    ggml_tensor * o;

    if (flash) {
        matrix_bd = ggml_scale(ctx, matrix_bd, scale);
        matrix_bd = ggml_cast(ctx, matrix_bd, GGML_TYPE_F16);

        // Optionally cast K/V activations to a narrower type to
        // reduce bandwidth in the attention kernel. GGML_TYPE_COUNT
        // means "auto" (f16 for quantized weights, f32 for f32
        // weights). The flash_attn accumulator is f32 internally
        // regardless of K/V type.
        {
            ggml_type effective_kv = kv_type;
            if (effective_kv == GGML_TYPE_COUNT) {
                effective_kv = (b.attn_k_w->type != GGML_TYPE_F32)
                             ? GGML_TYPE_F16 : GGML_TYPE_F32;
            }
            if (effective_kv != GGML_TYPE_F32) {
                k = ggml_cast(ctx, k, effective_kv);
                v = ggml_cast(ctx, v, effective_kv);
            }
        }

        o = ggml_flash_attn_ext(ctx, q_u, k, v, matrix_bd,
                                scale, /*max_bias=*/0.0f,
                                /*logit_softcap=*/0.0f);
        // Output is [head_dim, n_head, T_q, 1] — already permuted.
    } else {
        // Manual attention path: explicit matmuls + softmax.
        //
        //   attn_scores  = q_u @ k^T                    [T_q, T_q, n_head, 1]
        //   attn_scores  = (attn_scores + matrix_bd) * scale
        //   attn_weights = softmax(attn_scores)
        //   o            = attn_weights @ v
        //
        // K is [head_dim, T_q, n_head, 1] from permute(0,2,1,3).
        // mul_mat(K, Q_u) computes Q_u^T @ K for each head, giving
        // [T_q, T_q, n_head, 1] attention scores.
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q_u);

        // Add relative position bias, then scale inside soft_max_ext.
        kq = ggml_add(ctx, kq, matrix_bd);

        // ggml_soft_max_ext fuses the scale into softmax.
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr,
                                                  scale, /*max_bias=*/0.0f);

        // V needs to be transposed for the value matmul.
        // V is [head_dim, T_q, n_head, 1]. We need [T_q, head_dim, n_head, 1].
        ggml_tensor * v_t = ggml_cont(ctx,
                                      ggml_permute(ctx, v, 1, 0, 2, 3));

        // attn_weights @ V: [head_dim, T_q, n_head, 1]
        o = ggml_mul_mat(ctx, v_t, kq_soft);

        // Merge heads: permute [head_dim, T_q, n_head] -> [head_dim,
        // n_head, T_q] then reshape to [d_model, T_q]. The permute
        // makes heads contiguous for the reshape.
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
    }
    // Collapse heads back to d_model, keeping the batch at ne[2].
    // reshape_3d with B == 1 is equivalent to the prior reshape_2d.
    o = ggml_reshape_3d(ctx, o, d_model, T_q, B);

    // ----- Output linear -----------------------------------------
    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    if (b.attn_out_b != nullptr) o = ggml_add(ctx, o, b.attn_out_b);
    return o;
}

// ===========================================================================
// Conformer block (FF1 -> attn -> conv -> FF2 -> LN_out)
// ===========================================================================

ggml_tensor * build_conformer_block(ggml_context *        ctx,
                                    ggml_tensor *         x,
                                    ggml_tensor *         pos_emb,
                                    const BlockView &     b,
                                    const BlockParams &   params,
                                    const BlockObserver * obs)
{
    // Observer dispatch helper. Inlines to nothing when `obs` is null
    // or its callback is null, so blocks without instrumentation pay
    // zero overhead. Tags are compile-time string literals.
    const auto notify = [&](const char * tag, ggml_tensor * t) {
        if (obs != nullptr && obs->on_point != nullptr) {
            obs->on_point(obs->user, tag, t);
        }
    };

    // Macaron FF1: x = x + 0.5 * FF1(LN_ff1(x))
    x = macaron_ff_residual(ctx, x,
                            b.norm_ff1_w, b.norm_ff1_b,
                            b.ff1_lin1_w, b.ff1_lin1_b,
                            b.ff1_lin2_w, b.ff1_lin2_b);
    notify("after_ff1", x);

    // Self-attention with relative position. Full residual.
    //
    // Streaming carry (params.streaming_channel_in != nullptr):
    //   1. Compute the new cache as
    //        new_cache = concat(prev_cache[T_q_new:], x_norm)
    //      and cpy it into streaming_channel_out (per NeMo:
    //      q_keep_size = T_q_new with cache_drop_size = 0; the cache
    //      stays at size T_cache).
    //   2. Virtualize x_norm for the attention call:
    //        x_norm_virtual = concat(prev_cache, x_norm)
    //      so rel_pos_mhsa runs on T_virtual = T_cache + T_q_new
    //      positions. The pos_emb and chunked_mask the caller built
    //      are sized for T_virtual.
    //   3. Slice the attention output to the last T_q_new rows so it
    //      lines up with the running residual (which still has
    //      T_q_new frames).
    //
    // T_q_new is taken from the running x's ne[1] (x is the post-pre-
    // encode + post-FF1 running tensor; its time dim hasn't changed
    // yet at this point).
    {
        const bool streaming = (params.streaming_channel_in != nullptr);
        const int64_t T_q_new = streaming ? x->ne[1] : 0;
        const int64_t T_cache = streaming
            ? params.streaming_channel_in->ne[1]
            : 0;
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_attn_w, b.norm_attn_b);

        if (streaming) {
            // Emit the new cache slot before we virtualize x_norm
            // for attention. Source: prev_cache tail (T_cache - T_q_new
            // rows) concatenated with this chunk's x_norm (T_q_new
            // rows) — total T_cache rows along ne[1].
            //
            // For T_q_new < T_cache (the common case; nemotron has
            // T_q_new=12, T_cache=70), prev_cache_tail is a [d_model,
            // T_cache - T_q_new] view. For T_q_new >= T_cache the
            // new cache is just the last T_cache rows of x_norm — not
            // exercised today but the guard is cheap.
            if (params.streaming_channel_out != nullptr &&
                params.streaming_graph != nullptr)
            {
                ggml_tensor * new_cache = nullptr;
                if (T_q_new < T_cache) {
                    ggml_tensor * prev_tail = ggml_view_2d(
                        ctx, params.streaming_channel_in,
                        params.streaming_channel_in->ne[0],
                        T_cache - T_q_new,
                        params.streaming_channel_in->nb[1],
                        /*offset=*/T_q_new *
                            params.streaming_channel_in->nb[1]);
                    new_cache = ggml_concat(ctx, prev_tail, x_norm,
                                            /*dim=*/1);
                } else {
                    // x_norm holds at least T_cache rows — take its
                    // last T_cache.
                    new_cache = ggml_view_2d(
                        ctx, x_norm,
                        x_norm->ne[0], T_cache,
                        x_norm->nb[1],
                        (T_q_new - T_cache) * x_norm->nb[1]);
                }
                ggml_tensor * cpy = ggml_cpy(ctx, new_cache,
                                             params.streaming_channel_out);
                ggml_build_forward_expand(params.streaming_graph, cpy);
            }

            // Virtual-T attention: prepend the previous cache to
            // x_norm and let rel_pos_mhsa run as usual.
            x_norm = ggml_concat(ctx, params.streaming_channel_in,
                                 x_norm, /*dim=*/1);
        }

        ggml_tensor * attn_out = rel_pos_mhsa(ctx, x_norm, pos_emb, b, params);

        if (streaming) {
            // attn_out: [d_model, T_virtual]. Slice to last T_q_new.
            const int64_t T_virt = attn_out->ne[1];
            attn_out = ggml_view_2d(
                ctx, attn_out,
                attn_out->ne[0], T_q_new,
                attn_out->nb[1],
                (T_virt - T_q_new) * attn_out->nb[1]);
            attn_out = ggml_cont(ctx, attn_out);
        }

        x = ggml_add(ctx, x, attn_out);
    }
    notify("after_attn", x);

    // Convolution module. Full residual.
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_conv_w, b.norm_conv_b);
        ggml_tensor * conv_out = conv_module(ctx, x_norm, b, params);
        x = ggml_add(ctx, x, conv_out);
    }
    notify("after_conv", x);

    // Macaron FF2.
    x = macaron_ff_residual(ctx, x,
                            b.norm_ff2_w, b.norm_ff2_b,
                            b.ff2_lin1_w, b.ff2_lin1_b,
                            b.ff2_lin2_w, b.ff2_lin2_b);
    notify("after_ff2", x);

    // Final per-block LayerNorm (the block's output transform).
    x = layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
    notify("out", x);
    return x;
}

// ===========================================================================
// Pre-encode (DwStridingSubsampling)
// ===========================================================================

namespace {

// Name a tensor as `<prefix>.<suffix>` if prefix is non-null. Returns
// t unchanged. Used below to reproduce Parakeet's dump-point naming
// inside the shared helper without branching on family.
ggml_tensor * name_prefixed(ggml_tensor * t, const char * prefix,
                            const char * suffix)
{
    if (t == nullptr || prefix == nullptr) return t;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s.%s", prefix, suffix);
    ggml_set_name(t, buf);
    return t;
}

} // namespace

// Pre-encode subsampling stack. See conformer.py:206-328
// (DwStridingSubsampling). The op order matches NeMo:
//
//   conv0 (standard, 1->C, k=3 s=2 p=1)  -> bias -> ReLU
//   conv2 (depthwise, C->C, groups=C)    -> bias -> conv3 (pointwise, C->C) -> bias -> ReLU
//   conv5 (depthwise, C->C, groups=C)    -> bias -> conv6 (pointwise, C->C) -> bias -> ReLU
//   permute + cont + reshape to flatten freq*channels into a single
//   feature axis matching `pre_encode_in = channels * (n_mels / 8)`
//   linear (out_w, out_b) -> [d_model, T_enc, 1, 1]
//
// In NeMo's torch.nn.Sequential, the bias is fused into each conv.
// We replicate that here with an explicit ggml_add per conv
// (add_conv_bias is a no-op when the bias slot is null).
//
// The 2-D depthwise convs (conv2, conv5) honor
// policy.direct_dw_in_pre_encode. Parakeet keeps this false (uses the
// f32-friendly im2col helper on every backend, including ones where
// the direct path would work); Cohere follows its detect_direct_dw
// result. See the policy comment in conformer.h for why the two
// families historically disagree on this.
ggml_tensor * build_pre_encode(ggml_context *        ctx,
                               const PreEncodeView & pe,
                               ggml_tensor *         mel_in,
                               const ConvPolicy &    policy,
                               const char *          name_prefix,
                               const char *          error_tag,
                               PreEncodeValidMasks * valid_masks)
{
    // Utterance batch and a helper that, in the masked-subsampling path,
    // zeros each conv intermediate's padded time region after a ReLU. The
    // mask is a graph input ne=[1, H_stage, 1, B] broadcasting over freq
    // (ne[0]) and channels (ne[2]); the driver fills it per utterance. `slot`
    // receives the allocated handle. No-op when valid_masks is null.
    const int64_t pe_batch = mel_in->ne[3];
    auto apply_valid_mask = [&](ggml_tensor * x, ggml_tensor ** slot,
                                const char * mname) -> ggml_tensor * {
        if (valid_masks == nullptr) return x;
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                             1, x->ne[1], 1, pe_batch);
        ggml_set_name(m, mname);
        ggml_set_input(m);
        *slot = m;
        return ggml_mul(ctx, x, m);
    };

    // Transpose the mel input from its natural [T_mel, n_mels, 1, 1]
    // layout (matching the C++ row-major MelFrontend buffer) to
    // [n_mels, T_mel, 1, 1] = [W=F, H=T, IC, N] which is what
    // ggml_conv_2d expects under the NHWC `[B, H=T, W=F, C]`
    // convention. The conv kernel is not spatially symmetric, so the
    // axis order matters — if F and T are swapped the conv math
    // diverges. ggml_permute returns a non-contiguous view; the
    // ggml_cont materializes it for the conv.
    ggml_tensor * x = ggml_permute(ctx, mel_in,
                                   /*axis0=*/1, /*axis1=*/0,
                                   /*axis2=*/2, /*axis3=*/3);
    x = ggml_cont(ctx, x);
    x = name_prefixed(x, name_prefix, "mel_t");

    // Causal pre_encode: NeMo's CausalConv2D applies F.pad(left=k-1,
    // right=stride-1, left=k-1, right=stride-1) on both spatial axes
    // before the underlying conv (kernel=3, stride=2, padding=0). We
    // replicate that with explicit zero pads on dim 0 (W = freq) and
    // dim 1 (H = time), then call the conv with p=0. The op-side
    // (k-1)/2 symmetric padding path is what every offline variant
    // takes.
    const bool causal_pe = policy.causal_pre_encode;
    const int  pe_p_op   = causal_pe ? 0 : 1;
    auto pad_causal = [&](ggml_tensor * t) {
        if (!causal_pe) return t;
        // For k=3 / s=2: left=2, right=1 on both axes. Pad as zero
        // F32 leaves; ggml_fill writes the constant after compute-
        // buffer alloc, identical to how rel_pos_mhsa zero-pads.
        const int left = 2, right = 1;
        auto make_pad = [&](int width_w, int width_h) {
            ggml_tensor * p = ggml_new_tensor_4d(
                ctx, t->type,
                width_w > 0 ? width_w : t->ne[0],
                width_h > 0 ? width_h : t->ne[1],
                t->ne[2], t->ne[3]);
            return ggml_fill(ctx, p, 0.0f);
        };
        // dim 0 (W = freq).
        ggml_tensor * pad_l = make_pad(left, /*h=*/0);
        t = ggml_concat(ctx, pad_l, t, /*dim=*/0);
        ggml_tensor * pad_r = make_pad(right, /*h=*/0);
        t = ggml_concat(ctx, t, pad_r, /*dim=*/0);
        // dim 1 (H = time). At this point ne[0] grew by left+right.
        ggml_tensor * pad_t = ggml_new_tensor_4d(
            ctx, t->type, t->ne[0], left, t->ne[2], t->ne[3]);
        pad_t = ggml_fill(ctx, pad_t, 0.0f);
        t = ggml_concat(ctx, pad_t, t, /*dim=*/1);
        ggml_tensor * pad_b = ggml_new_tensor_4d(
            ctx, t->type, t->ne[0], right, t->ne[2], t->ne[3]);
        pad_b = ggml_fill(ctx, pad_b, 0.0f);
        t = ggml_concat(ctx, t, pad_b, /*dim=*/1);
        return t;
    };

    // conv0 (standard 2D conv: 1 in, channels out, k=3 s=2)
    x = pad_causal(x);
    x = ggml_conv_2d(ctx, pe.conv0_w, x,
                     /*s0=*/2, /*s1=*/2,
                     /*p0=*/pe_p_op, /*p1=*/pe_p_op,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv0_b);
    x = name_prefixed(x, name_prefix, "conv0");
    x = ggml_relu(ctx, x);
    x = name_prefixed(x, name_prefix, "relu0");
    x = apply_valid_mask(x, valid_masks ? &valid_masks->mask_s1 : nullptr,
                         "pre_encode.valid_mask.s1");

    // conv2 (depthwise: channels -> channels, groups=channels, k=3 s=2)
    // See conv_2d_dw_f32 above for the Metal backstory on why the
    // im2col path is used when direct_dw_in_pre_encode is false.
    x = pad_causal(x);
    if (policy.direct_dw_in_pre_encode) {
        x = ggml_conv_2d_dw_direct(ctx, pe.conv2_w, x,
                                   /*s0=*/2, /*s1=*/2,
                                   /*p0=*/pe_p_op, /*p1=*/pe_p_op,
                                   /*d0=*/1, /*d1=*/1);
    } else {
        x = conv_2d_dw_f32(ctx, pe.conv2_w, x,
                           /*s0=*/2, /*s1=*/2,
                           /*p0=*/pe_p_op, /*p1=*/pe_p_op,
                           /*d0=*/1, /*d1=*/1);
    }
    x = add_conv_bias(ctx, x, pe.conv2_b);
    x = name_prefixed(x, name_prefix, "conv2");

    // conv3 (pointwise: channels -> channels, k=1 s=1 p=0)
    x = ggml_conv_2d(ctx, pe.conv3_w, x,
                     /*s0=*/1, /*s1=*/1,
                     /*p0=*/0, /*p1=*/0,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv3_b);
    x = name_prefixed(x, name_prefix, "conv3");
    x = ggml_relu(ctx, x);
    x = name_prefixed(x, name_prefix, "relu3");
    x = apply_valid_mask(x, valid_masks ? &valid_masks->mask_s2 : nullptr,
                         "pre_encode.valid_mask.s2");

    // conv5 (depthwise) -> conv6 (pointwise) -> ReLU
    x = pad_causal(x);
    if (policy.direct_dw_in_pre_encode) {
        x = ggml_conv_2d_dw_direct(ctx, pe.conv5_w, x,
                                   /*s0=*/2, /*s1=*/2,
                                   /*p0=*/pe_p_op, /*p1=*/pe_p_op,
                                   /*d0=*/1, /*d1=*/1);
    } else {
        x = conv_2d_dw_f32(ctx, pe.conv5_w, x,
                           /*s0=*/2, /*s1=*/2,
                           /*p0=*/pe_p_op, /*p1=*/pe_p_op,
                           /*d0=*/1, /*d1=*/1);
    }
    x = add_conv_bias(ctx, x, pe.conv5_b);
    x = name_prefixed(x, name_prefix, "conv5");

    x = ggml_conv_2d(ctx, pe.conv6_w, x,
                     /*s0=*/1, /*s1=*/1,
                     /*p0=*/0, /*p1=*/0,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv6_b);
    x = name_prefixed(x, name_prefix, "conv6");
    x = ggml_relu(ctx, x);
    x = name_prefixed(x, name_prefix, "relu6");
    x = apply_valid_mask(x, valid_masks ? &valid_masks->mask_s3 : nullptr,
                         "pre_encode.valid_mask.s3");

    // At this point ne = [F'=16, T_enc, channels=256, 1] where
    // T_enc = floor(T_mel / 8) (with the per-conv floor formula).
    // We need to flatten (F', C) into a single feature axis matching
    // pre_encode_in = channels * (n_mels / subsampling_factor).
    //
    // The reference flattens with C as the slower sub-axis and F'
    // as the faster, giving flat index = c*F' + f'. In ggml
    // fast-to-slow ne, that's the layout you get by permuting to put
    // F' on axis 0 (already there) and C on axis 1 (currently axis
    // 2), then collapsing axes 0 and 1.
    //
    //   permute (0, 2, 1, 3): [F', T, C, 1] -> [F', C, T, 1]
    //   ggml_cont:            (reshape requires a contiguous tensor)
    //   reshape_2d to         [F'*C, T, 1, 1] = [pre_encode_in, T_enc, 1, 1]
    const int64_t F_prime = x->ne[0];
    const int64_t T_enc   = x->ne[1];
    const int64_t C       = x->ne[2];
    // Utterance batch rides the conv "N" axis (ne[3]) through the whole
    // stem; after the flatten + linear below it lands on the activation's
    // ne[2] == B convention the conformer blocks expect. B == 1 for
    // single-shot, so the 3-D reshape collapses to the prior 2-D shape.
    const int64_t B       = x->ne[3];
    const int64_t pre_encode_in = F_prime * C;

    // Sanity check against the catalog: the linear layer expects
    // exactly this many input features. If the math drifts (e.g. a
    // future converter changes the subsampling layout), surface it
    // here as a clear assertion rather than letting mul_mat fail
    // with a confusing shape mismatch.
    if (pe.out_w != nullptr && pre_encode_in != pe.out_w->ne[0]) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s encoder: pre_encode_in mismatch: "
                     "F'*C=%lld but out_w expects %lld",
                     error_tag != nullptr ? error_tag : "conformer",
                     static_cast<long long>(pre_encode_in),
                     static_cast<long long>(pe.out_w->ne[0]));
        return nullptr;
    }

    x = ggml_permute(ctx, x, /*axis0=*/0, /*axis1=*/2,
                              /*axis2=*/1, /*axis3=*/3);
    x = name_prefixed(x, name_prefix, "permuted");
    x = ggml_cont(ctx, x);
    x = name_prefixed(x, name_prefix, "flat_pre");
    // [F', C, T_enc, B] -> [F'*C, T_enc, B]. B at ne[2] after the collapse.
    x = ggml_reshape_3d(ctx, x, pre_encode_in, T_enc, B);
    x = name_prefixed(x, name_prefix, "flat");

    // Linear projection to d_model. ggml_mul_mat(W, x):
    //   W ne=[in, out],  x ne=[in, T]  ->  out ne=[out, T]
    // The catalog stores out_w as ne=[pre_encode_in, d_model] which
    // matches this convention exactly.
    x = ggml_mul_mat(ctx, pe.out_w, x);
    x = name_prefixed(x, name_prefix, "linear");

    // Add bias [d_model] -> broadcast to [d_model, T_enc, 1, 1].
    // Same trick as the conv biases above: reshape to a 4-D view
    // with the channel axis in position 0 and size-1 dims elsewhere.
    if (pe.out_b != nullptr) {
        const int64_t d_model = pe.out_b->ne[0];
        ggml_tensor * bias_4d = ggml_reshape_4d(ctx, pe.out_b,
                                                d_model, 1, 1, 1);
        x = ggml_add(ctx, x, bias_4d);
    }
    x = name_prefixed(x, name_prefix, "out");

    return x;
}

} // namespace transcribe::conformer
