// arch/parakeet/encoder.cpp - Parakeet Conformer encoder graph builder.
//
// See encoder.h for the API contract and the sub-stage roadmap. This
// file currently implements sub-stage 3a (pre_encode subsampling
// stack). Subsequent sub-stages append the conformer blocks.
//
// Reference: parakeet-mlx
//   /tmp/parakeet-mlx/parakeet_mlx/conformer.py:206-328  (DwStridingSubsampling)
//   /tmp/parakeet-mlx/parakeet_mlx/conformer.py:392-423  (Conformer.__call__)
//
// Layout cheat sheet:
//
//   MelFrontend output  : row-major [n_mels=128, n_frames=T_mel] f32
//   ggml mel_in tensor  : ne=[T_mel, 128, 1, 1] f32  (matches the
//                         row-major buffer byte-for-byte: element
//                         [m, t] in the C++ buffer lives at the
//                         same offset as logical (i0=t, i1=m) in
//                         the ggml tensor)
//   conv input form     : [W=F, H=T, IC, N] per ggml_conv_2d. The
//                         catalog kernels are stored in MLX/PyTorch
//                         OIHW order which translates to ggml ne
//                         [KW, KH, IC, OC] where KW is aligned with
//                         the FREQ axis and KH with the TIME axis
//                         (this is the MLX NHWC convention: H=T,
//                         W=F). The 3x3 conv kernel is NOT spatially
//                         symmetric, so the data must be presented
//                         with W=F and H=T to match MLX's math; we
//                         transpose the natural mel layout via
//                         ggml_permute(1,0,2,3) + ggml_cont before
//                         the first conv.
//
// After 3 stride-2 convs (kernel=3, padding=1), W and H both shrink
// by floor((d + 2 - 3)/2) + 1 per layer:
//   F (ggml W) =  128 ->  64 ->  32 -> 16
//   T (ggml H) = 1101 -> 551 -> 276 -> 138    (jfk.wav at 16 kHz, 11 s)
// So the post-conv tensor has ne = [F'=16, T_enc=138, channels=256, 1].
//
// To match parakeet-mlx's `[B, T_enc, channels*16] -> linear ->
// [B, T_enc, d_model]`, we permute (0, 2, 1, 3) to get
// [F'=16, channels=256, T_enc, 1], make contiguous, reshape to
// [4096, T_enc, 1, 1], then mul_mat with `out_w` (ne=[4096, 1024])
// yielding [d_model=1024, T_enc, 1, 1]. The bias `out_b` (ne=[1024])
// is added after broadcasting via reshape to [d_model, 1, 1, 1].
//
// Why F' goes inner in the flatten: MLX flattens with C as the
// slower sub-axis and F' as the faster, giving flat index
// = c*F' + f'. In ggml fast-to-slow ne, that's the layout you get
// by permuting to put F' on axis 0 and C on axis 1.

#include "encoder.h"

#include "weights.h"

#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace transcribe::parakeet {

namespace {

// LayerNorm + BatchNorm epsilon. Both default to 1e-5 in MLX, NOT
// stored in the GGUF — hardcoded here to match. If a future variant
// changes either, the value moves to ParakeetHParams.
constexpr float kLayerNormEps = 1e-5f;

// Set a debug name on a tensor so dump points and ggml's own logs
// can identify intermediates. ggml_set_name copies the string into a
// fixed-size buffer on the tensor; truncation is fine for our names.
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
// wants. The affine multiply + add use ggml's broadcasting:
// gamma/beta have ne=[d_model, 1, 1, 1], which broadcasts naturally
// across the T and B axes of the activation.
//
// `gamma` is required; `beta` may be null for bias-free LN
// (Parakeet uses gamma+beta on every LN, but the helper supports
// no-bias for future families).
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

// FeedForward: y = Linear[d_ff -> d_model](SiLU(Linear[d_model -> d_ff](x)))
//
// Both linears are bias-free per Parakeet's enc_use_bias=false.
// ggml_mul_mat(W, x) reads W ne[0] as the input dim, so with
// W ne=[d_in, d_out], x ne=[d_in, T, B] -> result ne=[d_out, T, B].
// The catalog stores ff*_lin1_w as ne=[d_model, d_ff] and
// ff*_lin2_w as ne=[d_ff, d_model] which match this convention.
ggml_tensor * feed_forward(ggml_context * ctx,
                           ggml_tensor *  x,
                           ggml_tensor *  lin1_w,
                           ggml_tensor *  lin2_w)
{
    ggml_tensor * y = ggml_mul_mat(ctx, lin1_w, x);
    y = ggml_silu(ctx, y);
    y = ggml_mul_mat(ctx, lin2_w, y);
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
                                  ggml_tensor *  lin2_w)
{
    ggml_tensor * y = layer_norm(ctx, x, norm_w, norm_b);
    y = feed_forward(ctx, y, lin1_w, lin2_w);
    y = ggml_scale(ctx, y, 0.5f);
    return ggml_add(ctx, x, y);
}

// Shaw / Transformer-XL relative-position skew. See attention.py:82-91
// in parakeet-mlx. Transforms the position scoring matrix from
// [pos_len, T_q, H, 1] to the same shape but with each row rotated
// so that column k holds the score for relative offset k.
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
// triggers a runtime assertion when fed an f32 kernel — and the
// Parakeet weight catalog is fp32 throughout.
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

    // Reshape im2col to [IC*K, OW*N] and kernel to [IC*K, OC] for
    // mul_mat. Result ne = [OC, OW*N].
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col,
                        im2col->ne[0],
                        im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel,
                        kernel->ne[0] * kernel->ne[1],
                        kernel->ne[2]));

    // Reshape back to [OW, OC, N]. The semantic is the same as
    // ggml_conv_1d's final reshape; this matches the [time,
    // channels, batch] layout the conformer module expects.
    result = ggml_reshape_3d(ctx, result,
                             im2col->ne[1],
                             kernel->ne[2],
                             im2col->ne[2]);
    return result;
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

// Conformer convolution module (conformer.py:51-103). Operates on
// the post-LayerNorm activation; the LN is applied OUTSIDE this
// helper by the block forward.
//
// By default, pointwise convs (k=1) are direct mul_mats in [d_model, T]
// layout, avoiding im2col overhead. Set TRANSCRIBE_CONV_NO_DIRECT_PW=1
// to fall back to the im2col path (useful for backends where the
// reshape-based path is slower, e.g. some Vulkan drivers).
//
// BatchNorm uses precomputed fused scale + bias (2 ops instead of 4).

bool use_direct_pw() {
    static const bool val = (std::getenv("TRANSCRIBE_CONV_NO_DIRECT_PW") == nullptr);
    return val;
}

ggml_tensor * conv_module(ggml_context * ctx,
                          ggml_tensor *  x,
                          const ParakeetBlock & b,
                          int            conv_kernel)
{
    const int padding = (conv_kernel - 1) / 2;
    const int64_t d_model = x->ne[0];

    if (use_direct_pw()) {
        // Pointwise conv 1 as direct mul_mat in [d_model, T] layout.
        // Kernel ne=[1, d_model, 2*d_model] → reshape to [d_model, 2*d_model].
        {
            ggml_tensor * pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w,
                                                d_model, 2 * d_model);
            x = ggml_mul_mat(ctx, pw1, x);  // [2*d_model, T]
        }

        // GLU: split ne[0] in half, gate * sigmoid(value).
        {
            const int64_t T    = x->ne[1];
            const int64_t half = x->ne[0] / 2;
            ggml_tensor * gate  = ggml_view_2d(ctx, x, half, T,
                                               x->nb[1], /*offset=*/0);
            ggml_tensor * value = ggml_view_2d(ctx, x, half, T,
                                               x->nb[1],
                                               half * ggml_element_size(x));
            x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
        }
        // x ne = [d_model, T]

        // Transpose for depthwise conv: [d_model, T] -> [T, d_model].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    } else {
        // Fallback: im2col path in [T, d_model] layout.
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        x = conv_1d_f32(ctx, b.conv_pw1_w, x, /*s=*/1, /*p=*/0, /*d=*/1);

        const int64_t half = x->ne[1] / 2;
        ggml_tensor * gate = ggml_view_4d(ctx, x,
                                          x->ne[0], half, x->ne[2], x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * value = ggml_view_4d(ctx, x,
                                           x->ne[0], half, x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3],
                                           x->nb[1] * half);
        x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));

        x = ggml_cont(ctx, x);
    }

    // Depthwise conv: kernel size from hparams, groups=d_model.
    x = conv_1d_dw_f32(ctx, b.conv_dw_w, x,
                       /*s=*/1, /*p=*/padding, /*d=*/1);

    // Fused BatchNorm (precomputed scale + bias).
    x = fused_batch_norm(ctx, x,
                         b.conv_bn_fused_scale, b.conv_bn_fused_bias);

    // SiLU activation.
    x = ggml_silu(ctx, x);

    if (use_direct_pw()) {
        // Transpose back: [T, d_model] -> [d_model, T].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        // Pointwise conv 2 as direct mul_mat in [d_model, T] layout.
        ggml_tensor * pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w,
                                            d_model, d_model);
        x = ggml_mul_mat(ctx, pw2, x);
    } else {
        x = conv_1d_f32(ctx, b.conv_pw2_w, x, /*s=*/1, /*p=*/0, /*d=*/1);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    }

    return x;
}

// Forward declaration so build_conformer_block can call into it.
ggml_tensor * rel_pos_mhsa(ggml_context * ctx,
                           ggml_tensor *  x,
                           ggml_tensor *  pos_emb,
                           const ParakeetBlock & b,
                           int            d_model,
                           int            n_head,
                           ggml_type      kv_type);

// One conformer block forward (conformer.py:186-203). The macaron
// structure: FF1 (0.5x residual), MHSA (full residual), Conv (full
// residual), FF2 (0.5x residual), final per-block LayerNorm.
//
// `pos_emb` is shared across every block in the encoder, so the
// caller computes it once and threads it through.
ggml_tensor * build_conformer_block(ggml_context * ctx,
                                    ggml_tensor *  x,
                                    ggml_tensor *  pos_emb,
                                    const ParakeetBlock & b,
                                    int            d_model,
                                    int            n_head,
                                    int            conv_kernel,
                                    ggml_type      kv_type)
{
    // Macaron FF1: x = x + 0.5 * FF1(LN_ff1(x))
    x = macaron_ff_residual(ctx, x,
                            b.norm_ff1_w, b.norm_ff1_b,
                            b.ff1_lin1_w, b.ff1_lin2_w);

    // Self-attention with relative position. Full residual.
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_attn_w, b.norm_attn_b);
        ggml_tensor * attn_out = rel_pos_mhsa(ctx, x_norm, pos_emb,
                                              b, d_model, n_head, kv_type);
        x = ggml_add(ctx, x, attn_out);
    }

    // Convolution module. Full residual.
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_conv_w, b.norm_conv_b);
        ggml_tensor * conv_out = conv_module(ctx, x_norm, b, conv_kernel);
        x = ggml_add(ctx, x, conv_out);
    }

    // Macaron FF2.
    x = macaron_ff_residual(ctx, x,
                            b.norm_ff2_w, b.norm_ff2_b,
                            b.ff2_lin1_w, b.ff2_lin2_w);

    // Final per-block LayerNorm (the block's output transform).
    x = layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
    return x;
}

// Relative-position multi-head self-attention using flash attention.
//
// Uses ggml_flash_attn_ext to fuse Q@K^T, scale, softmax, and @V into
// a single op. The relative position contribution is precomputed and
// passed as the additive mask (F16).
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
ggml_tensor * rel_pos_mhsa(ggml_context * ctx,
                           ggml_tensor *  x,
                           ggml_tensor *  pos_emb,
                           const ParakeetBlock & b,
                           int            d_model,
                           int            n_head,
                           ggml_type      kv_type)
{
    const int     head_dim = d_model / n_head;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T_q      = x->ne[1];
    const int64_t pos_len  = pos_emb->ne[1];

    // ----- Q, K, V, P projections (all bias-free) ------------------
    ggml_tensor * q = ggml_mul_mat(ctx, b.attn_q_w, x);
    ggml_tensor * k = ggml_mul_mat(ctx, b.attn_k_w, x);
    ggml_tensor * v = ggml_mul_mat(ctx, b.attn_v_w, x);
    ggml_tensor * p = ggml_mul_mat(ctx, b.attn_pos_w, pos_emb);

    // ----- Split heads ---------------------------------------------
    // pos_bias_u/v broadcast onto [head_dim, n_head, T, 1] BEFORE
    // the permute that moves T past n_head.
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, T_q, 1);
    ggml_tensor * q_u = ggml_add(ctx, q, b.attn_pos_u);
    ggml_tensor * q_v = ggml_add(ctx, q, b.attn_pos_v);

    // All of Q, K, V need [head_dim, T, n_head, 1] for flash_attn.
    q_u = ggml_cont(ctx, ggml_permute(ctx, q_u, 0, 2, 1, 3));
    q_v = ggml_cont(ctx, ggml_permute(ctx, q_v, 0, 2, 1, 3));

    k = ggml_reshape_4d(ctx, k, head_dim, n_head, T_q, 1);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    v = ggml_reshape_4d(ctx, v, head_dim, n_head, T_q, 1);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    p = ggml_reshape_4d(ctx, p, head_dim, n_head, pos_len, 1);
    p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    // ----- Position mask for flash attention -----------------------
    // Compute rel_shift(q_v @ p^T)[:T_k] * scale, cast to F16.
    ggml_tensor * matrix_bd = ggml_mul_mat(ctx, p, q_v);
    matrix_bd = rel_shift(ctx, matrix_bd);
    matrix_bd = ggml_view_4d(ctx, matrix_bd,
                             T_q, T_q, n_head, 1,
                             matrix_bd->nb[1], matrix_bd->nb[2],
                             matrix_bd->nb[3], /*offset=*/0);
    matrix_bd = ggml_cont(ctx, matrix_bd);
    matrix_bd = ggml_scale(ctx, matrix_bd, scale);
    matrix_bd = ggml_cast(ctx, matrix_bd, GGML_TYPE_F16);

    // ----- Flash attention -----------------------------------------
    // Optionally cast K/V activations to a narrower type to reduce
    // bandwidth in the attention kernel. GGML_TYPE_COUNT means "auto"
    // (f16 for quantized weights, f32 for f32 weights). The flash_attn
    // accumulator is f32 internally regardless of K/V type.
    {
        ggml_type effective_kv = kv_type;
        if (effective_kv == GGML_TYPE_COUNT) {
            // Auto: f16 for quantized models, f32 for f32 models.
            effective_kv = (b.attn_k_w->type != GGML_TYPE_F32)
                         ? GGML_TYPE_F16 : GGML_TYPE_F32;
        }
        if (effective_kv != GGML_TYPE_F32) {
            k = ggml_cast(ctx, k, effective_kv);
            v = ggml_cast(ctx, v, effective_kv);
        }
    }

    ggml_tensor * o = ggml_flash_attn_ext(ctx, q_u, k, v, matrix_bd,
                                          scale, /*max_bias=*/0.0f,
                                          /*logit_softcap=*/0.0f);
    // Output is [head_dim, n_head, T_q, 1] — already permuted.
    o = ggml_reshape_2d(ctx, o, d_model, T_q);

    // ----- Output linear -------------------------------------------
    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    return o;
}

// Add a 1-D bias [C] to a 4-D conv output [W, H, C, N] by reshaping
// the bias to [1, 1, C, 1] so ggml_add broadcasts the W, H, and N
// axes for free. The reshape is a metadata-only view because the
// 1-D tensor is contiguous; no data copy.
ggml_tensor * add_conv_bias(ggml_context * ctx,
                            ggml_tensor *  conv_out,
                            ggml_tensor *  bias_1d)
{
    const int64_t channels = bias_1d->ne[0];
    ggml_tensor * bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, 1, channels, 1);
    return ggml_add(ctx, conv_out, bias_4d);
}

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
// In NeMo's torch.nn.Sequential, the bias is fused into each conv
// (Conv2d emits its own bias add). We replicate that here with an
// explicit ggml_add per conv. Activations sit BETWEEN the
// (depthwise+pointwise) pairs, not after every individual conv —
// matching the indices in the catalog (conv1 and conv4 are the ReLU
// modules, hence skipped from the weight list).
ggml_tensor * build_pre_encode(ggml_context *          ctx,
                               const ParakeetWeights & w,
                               const ParakeetHParams & hp,
                               ggml_tensor *           mel_in)
{
    const auto & pe = w.pre_encode;

    // Transpose the mel input from its natural [T_mel, n_mels, 1, 1]
    // layout (matching the C++ row-major MelFrontend buffer) to
    // [n_mels, T_mel, 1, 1] = [W=F, H=T, IC, N] which is what
    // ggml_conv_2d expects to match MLX's NHWC `[B, H=T, W=F, C]`
    // convention. The conv kernel is not spatially symmetric, so the
    // data axes have to be presented in MLX order or the conv math
    // diverges. ggml_permute returns a non-contiguous view; the
    // ggml_cont materializes it for the conv.
    ggml_tensor * x = ggml_permute(ctx, mel_in,
                                   /*axis0=*/1, /*axis1=*/0,
                                   /*axis2=*/2, /*axis3=*/3);
    x = ggml_cont(ctx, x);
    x = named(x, "enc.pre_encode.mel_t");

    // conv0 (standard 2D conv: 1 in, channels out, k=3 s=2 p=1)
    x = ggml_conv_2d(ctx, pe.conv0_w, x,
                     /*s0=*/2, /*s1=*/2,
                     /*p0=*/1, /*p1=*/1,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv0_b);
    x = named(x, "enc.pre_encode.conv0");
    x = ggml_relu(ctx, x);
    x = named(x, "enc.pre_encode.relu0");

    // conv2 (depthwise: channels -> channels, groups=channels, k=3 s=2 p=1)
    // ggml_conv_2d_dw expects the same kernel layout the catalog
    // stores: ne=[KW=3, KH=3, IC_per_group=1, OC=channels].
    // Depthwise 2D conv. Both ggml's built-in options are broken
    // on Metal in this ggml version: ggml_conv_2d_dw forces f16
    // im2col output and lands on a missing
    // `kernel_mul_mv_f32_f16_short`; ggml_conv_2d_dw_direct lands
    // on a CONV_2D_DW op that ggml_metal_op_encode_impl reports
    // as unsupported. Use the local f32-friendly helper which
    // does im2col with the kernel's real f32 type and matmuls
    // against `kernel_mul_mv_f32_f32_short` (which IS compiled
    // into the Metal lib).
    x = conv_2d_dw_f32(ctx, pe.conv2_w, x,
                       /*s0=*/2, /*s1=*/2,
                       /*p0=*/1, /*p1=*/1,
                       /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv2_b);
    x = named(x, "enc.pre_encode.conv2");

    // conv3 (pointwise: channels -> channels, k=1 s=1 p=0)
    x = ggml_conv_2d(ctx, pe.conv3_w, x,
                     /*s0=*/1, /*s1=*/1,
                     /*p0=*/0, /*p1=*/0,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv3_b);
    x = named(x, "enc.pre_encode.conv3");
    x = ggml_relu(ctx, x);
    x = named(x, "enc.pre_encode.relu3");

    // conv5 (depthwise) -> conv6 (pointwise) -> ReLU
    x = conv_2d_dw_f32(ctx, pe.conv5_w, x,
                       /*s0=*/2, /*s1=*/2,
                       /*p0=*/1, /*p1=*/1,
                       /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv5_b);
    x = named(x, "enc.pre_encode.conv5");

    x = ggml_conv_2d(ctx, pe.conv6_w, x,
                     /*s0=*/1, /*s1=*/1,
                     /*p0=*/0, /*p1=*/0,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv6_b);
    x = named(x, "enc.pre_encode.conv6");
    x = ggml_relu(ctx, x);
    x = named(x, "enc.pre_encode.relu6");

    // At this point ne = [F'=16, T_enc, channels=256, 1] where
    // T_enc = floor(T_mel / 8) (with the per-conv floor formula).
    // We need to flatten (F', C) into a single feature axis matching
    // pre_encode_in = channels * (n_mels / subsampling_factor).
    //
    // MLX flattens with C as the slower sub-axis and F' as the
    // faster, giving flat index = c*F' + f'. In ggml fast-to-slow
    // ne, that's the layout you get by permuting to put F' on
    // axis 0 (already there) and C on axis 1 (currently axis 2),
    // then collapsing axes 0 and 1.
    //
    //   permute (0, 2, 1, 3): [F', T, C, 1] -> [F', C, T, 1]
    //   ggml_cont:            (reshape requires a contiguous tensor)
    //   reshape_2d to         [F'*C, T, 1, 1] = [pre_encode_in, T_enc, 1, 1]
    const int64_t F_prime = x->ne[0];
    const int64_t T_enc   = x->ne[1];
    const int64_t C       = x->ne[2];
    const int64_t pre_encode_in = F_prime * C;

    // Sanity check against the catalog: the linear layer expects
    // exactly this many input features. If the math drifts (e.g. a
    // future converter changes the subsampling layout), surface it
    // here as a clear assertion rather than letting mul_mat fail
    // with a confusing shape mismatch.
    if (pre_encode_in != pe.out_w->ne[0]) {
        std::fprintf(stderr,
                     "parakeet encoder: pre_encode_in mismatch: "
                     "F'*C=%lld but out_w expects %lld\n",
                     static_cast<long long>(pre_encode_in),
                     static_cast<long long>(pe.out_w->ne[0]));
        return nullptr;
    }
    (void)hp; // hp is captured for future sub-stages; reading
              // hp.enc_subsampling_channels here would just duplicate
              // the catalog value we already validated above.

    x = ggml_permute(ctx, x, /*axis0=*/0, /*axis1=*/2,
                              /*axis2=*/1, /*axis3=*/3);
    x = named(x, "enc.pre_encode.permuted");
    x = ggml_cont(ctx, x);
    x = named(x, "enc.pre_encode.flat_pre");
    x = ggml_reshape_2d(ctx, x, pre_encode_in, T_enc);
    x = named(x, "enc.pre_encode.flat");

    // Linear projection to d_model. ggml_mul_mat(W, x):
    //   W ne=[in, out],  x ne=[in, T]  ->  out ne=[out, T]
    // The catalog stores out_w as ne=[pre_encode_in, d_model] which
    // matches this convention exactly.
    x = ggml_mul_mat(ctx, pe.out_w, x);
    x = named(x, "enc.pre_encode.linear");

    // Add bias [d_model] -> broadcast to [d_model, T_enc, 1, 1].
    // Same trick as the conv biases above: reshape to a 4-D view
    // with the channel axis in position 0 and size-1 dims elsewhere.
    {
        const int64_t d_model = pe.out_b->ne[0];
        ggml_tensor * bias_4d = ggml_reshape_4d(ctx, pe.out_b,
                                                d_model, 1, 1, 1);
        x = ggml_add(ctx, x, bias_4d);
    }
    x = named(x, "enc.pre_encode.out");

    return x;
}

} // namespace

EncoderBuild build_encoder_graph(ggml_context *          ctx,
                                 const ParakeetWeights & w,
                                 const ParakeetHParams & hp,
                                 int                     n_mel_frames,
                                 ggml_type               kv_type)
{
    EncoderBuild eb {};

    if (ctx == nullptr || n_mel_frames <= 0) {
        std::fprintf(stderr,
                     "parakeet encoder: invalid arg "
                     "(ctx=%p, n_mel_frames=%d)\n",
                     static_cast<void *>(ctx), n_mel_frames);
        return eb;
    }

    // Sub-stage 3a only supports the catalog's locked-in
    // factor=8/n_mels=128 layout. The full encoder will assert this
    // upstream in init_context once step 5 wires the run path; for
    // now we re-check here so the encoder builder is self-contained
    // for sub-stage validation.
    if (hp.enc_subsampling_factor != 8 || hp.fe_num_mels != 128) {
        std::fprintf(stderr,
                     "parakeet encoder: unsupported geometry "
                     "subsampling_factor=%d num_mels=%d "
                     "(only 8/128 implemented)\n",
                     hp.enc_subsampling_factor, hp.fe_num_mels);
        return eb;
    }

    // Mel input handle. ne=[T_mel, n_mels, 1, 1] matches the C++
    // MelFrontend's row-major [n_mels, n_frames] storage byte for
    // byte (see the layout cheat sheet at the top of the file).
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                   n_mel_frames, hp.fe_num_mels);
    if (eb.mel_in == nullptr) {
        std::fprintf(stderr,
                     "parakeet encoder: failed to allocate mel_in tensor\n");
        return eb;
    }
    ggml_set_name(eb.mel_in, "mel.in");
    ggml_set_input(eb.mel_in);

    // Build the pre_encode subgraph (sub-stage 3a).
    ggml_tensor * x = build_pre_encode(ctx, w, hp, eb.mel_in);
    if (x == nullptr) {
        // build_pre_encode already logged the diagnostic.
        return eb;
    }
    eb.dumps.pre_encode_out = x;

    // ----- Conformer block 0 ---------------------------------------
    //
    // Sub-stages 3b-3e wire each sub-step of block 0 in turn; sub-stage
    // 3f loops over all 24 blocks. For now we hand-build block 0 only.

    if (!w.blocks.empty()) {
        const auto & b0 = w.blocks[0];

        // After pre_encode, x ne = [d_model, T_enc, 1, 1]. T_enc is
        // needed by the positional embedding (whose pos_len = 2*T_enc - 1)
        // and by the encoder accuracy harness for shape sanity.
        const int64_t T_enc = x->ne[1];

        // Create the pos_emb input tensor here, once we know T_enc.
        // The driver fills it via ggml_backend_tensor_set after the
        // compute buffer is allocated. ne = [d_model, pos_len, 1, 1]
        // — slow-to-fast shape (numpy) is (pos_len, d_model), matching
        // parakeet-mlx's `enc.pos_emb` dump.
        const int64_t pos_len = 2 * T_enc - 1;
        eb.pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           hp.enc_d_model, pos_len);
        ggml_set_name(eb.pos_emb_in, "pos_emb.in");
        ggml_set_input(eb.pos_emb_in);

        // Sub-stage 3b: macaron FF1.
        x = macaron_ff_residual(ctx, x,
                                b0.norm_ff1_w, b0.norm_ff1_b,
                                b0.ff1_lin1_w, b0.ff1_lin2_w);
        x = named(x, "enc.block.0.after_ff1");
        eb.dumps.block0_after_ff1 = x;

        // Sub-stage 3c: relative-position multi-head self-attention.
        // Full residual (no 0.5 scale).
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              b0.norm_attn_w, b0.norm_attn_b);
            ggml_tensor * attn_out = rel_pos_mhsa(ctx, x_norm, eb.pos_emb_in,
                                                  b0,
                                                  hp.enc_d_model, hp.enc_n_heads,
                                                  kv_type);
            x = ggml_add(ctx, x, attn_out);
            x = named(x, "enc.block.0.after_attn");
            eb.dumps.block0_after_attn = x;
        }

        // Sub-stage 3d: convolution module. Full residual.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              b0.norm_conv_w, b0.norm_conv_b);
            ggml_tensor * conv_out = conv_module(ctx, x_norm, b0,
                                                 hp.enc_conv_kernel);
            x = ggml_add(ctx, x, conv_out);
            x = named(x, "enc.block.0.after_conv");
            eb.dumps.block0_after_conv = x;
        }

        // Sub-stage 3e: macaron FF2 + final per-block LayerNorm.
        x = macaron_ff_residual(ctx, x,
                                b0.norm_ff2_w, b0.norm_ff2_b,
                                b0.ff2_lin1_w, b0.ff2_lin2_w);
        x = named(x, "enc.block.0.after_ff2");
        eb.dumps.block0_after_ff2 = x;

        // Final per-block LayerNorm. This is the block's output
        // transform (applied OUTSIDE the residual chain).
        x = layer_norm(ctx, x, b0.norm_out_w, b0.norm_out_b);
        x = named(x, "enc.block.0.out");
        eb.dumps.block0_out = x;
    }

    // Sub-stage 3f: loop over blocks 1..N-1.
    //
    // Block 0 was hand-built above with named dump points for the
    // 3b-3e bring-up loop; the remaining blocks use the
    // build_conformer_block helper. Spot-check dumps for blocks 12
    // and 23 give us mid-encoder and final-encoder validation
    // without flooding the dump dir with all 24 outputs.
    for (size_t i = 1; i < w.blocks.size(); ++i) {
        x = build_conformer_block(ctx, x, eb.pos_emb_in,
                                  w.blocks[i],
                                  hp.enc_d_model,
                                  hp.enc_n_heads,
                                  hp.enc_conv_kernel,
                                  kv_type);
        if (i == 12) {
            x = named(x, "enc.block.12.out");
            eb.dumps.block12_out = x;
        }
        if (i == 23) {
            x = named(x, "enc.block.23.out");
            eb.dumps.block23_out = x;
        }
    }

    // Final encoder output. With all 24 blocks wired, this is now
    // the true encoder exit point (= block 23's norm_out output)
    // and the comparator's `enc.final` row will compare against
    // parakeet-mlx's true encoder output.
    eb.dumps.final_out = x;
    x = named(x, "enc.final");

    // The encoder's final output is whatever the last sub-stage
    // produced. eb.out is the graph endpoint (always); eb.dumps.final_out
    // is only set once the full 24-block encoder is wired in
    // sub-stage 3f, so the comparator doesn't false-fail the
    // `enc.final` row against parakeet-mlx's true encoder output
    // during 3b-3e bring-up.
    eb.out = x;
    ggml_set_output(eb.out);

    // Build the forward cgraph. ggml_new_graph_custom allocates the
    // cgraph out of the same compute_ctx, so it has to be sized for
    // it. The default GGML_DEFAULT_GRAPH_SIZE (2048 nodes) is too
    // small for the full encoder: each conformer block contributes
    // ~90 ops (FF + attn + conv + FF + LN, including all the
    // permute/cont overhead) and 24 blocks pushes the count past
    // 2200. Use 8192 to leave headroom for future sub-stages and
    // any helper-internal expansion.
    eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (eb.graph == nullptr) {
        std::fprintf(stderr,
                     "parakeet encoder: ggml_new_graph_custom failed\n");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);

    return eb;
}

} // namespace transcribe::parakeet
