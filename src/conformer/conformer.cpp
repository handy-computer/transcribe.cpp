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
//
// Reference: parakeet-mlx
//   /tmp/parakeet-mlx/parakeet_mlx/conformer.py:206-328  (DwStridingSubsampling)
//   /tmp/parakeet-mlx/parakeet_mlx/conformer.py:392-423  (Conformer.__call__)

#include "conformer/conformer.h"

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

    // Reshape im2col to [IC*K, OW*N] and kernel to [IC*K, OC] for
    // mul_mat. ggml_mul_mat returns ne = [OW*N, OC].
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
// BatchNorm uses precomputed fused scale + bias (2 ops instead of 4).
ggml_tensor * conv_module(ggml_context *     ctx,
                          ggml_tensor *      x,
                          const BlockView &  b,
                          int                conv_kernel,
                          const ConvPolicy & policy)
{
    const int     padding = (conv_kernel - 1) / 2;
    const int64_t d_model = x->ne[0];

    if (policy.direct_pw) {
        // Pointwise conv 1 as direct mul_mat in [d_model, T] layout.
        // Kernel ne=[1, d_model, 2*d_model] → reshape to [d_model, 2*d_model].
        {
            ggml_tensor * pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w,
                                                d_model, 2 * d_model);
            x = ggml_mul_mat(ctx, pw1, x);  // [2*d_model, T]
            if (b.conv_pw1_b != nullptr) {
                x = ggml_add(ctx, x, b.conv_pw1_b);
            }
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

    // Depthwise conv: kernel size from hparams, groups=d_model.
    if (policy.direct_dw_in_block) {
        // Fused single-op depthwise conv (no im2col). Input is [T, d_model]
        // from the transpose above. Reshape to 4D [T, 1, d_model, 1] for
        // ggml_conv_2d_dw_direct which expects [W, H, C, N].
        // Kernel [k, 1, d_model] → [k, 1, 1, d_model] (KW, KH, 1, C).
        ggml_tensor * knl = ggml_reshape_4d(ctx, b.conv_dw_w,
                                            conv_kernel, 1, 1, d_model);
        ggml_tensor * data = ggml_reshape_4d(ctx, x,
                                             x->ne[0], 1, x->ne[1], 1);
        x = ggml_conv_2d_dw_direct(ctx, knl, data,
                                   /*s0=*/1, /*s1=*/1,
                                   /*p0=*/padding, /*p1=*/0,
                                   /*d0=*/1, /*d1=*/1);
        // Output: [T_out, 1, d_model, 1] → [T, d_model].
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[2]);
    } else {
        x = conv_1d_dw_f32(ctx, b.conv_dw_w, x,
                           /*s=*/1, /*p=*/padding, /*d=*/1);
    }

    // Add depthwise bias in [T, d_model] layout (nullable).
    if (b.conv_dw_b != nullptr) {
        ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_dw_b, 1, d_model);
        x = ggml_add(ctx, x, bias_r);
    }

    // Fused BatchNorm (precomputed scale + bias).
    x = fused_batch_norm(ctx, x,
                         b.conv_bn_fused_scale, b.conv_bn_fused_bias);

    // SiLU activation.
    x = ggml_silu(ctx, x);

    if (policy.direct_pw) {
        // Transpose back: [T, d_model] -> [d_model, T].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        // Pointwise conv 2 as direct mul_mat in [d_model, T] layout.
        ggml_tensor * pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w,
                                            d_model, d_model);
        x = ggml_mul_mat(ctx, pw2, x);
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
ggml_tensor * rel_pos_mhsa(ggml_context *    ctx,
                           ggml_tensor *     x,
                           ggml_tensor *     pos_emb,
                           const BlockView & b,
                           int               d_model,
                           int               n_head,
                           ggml_type         kv_type,
                           bool              use_flash)
{
    const int     head_dim = d_model / n_head;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T_q      = x->ne[1];
    const int64_t pos_len  = pos_emb->ne[1];

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
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, T_q, 1);
    ggml_tensor * q_u = ggml_add(ctx, q, b.attn_pos_u);
    ggml_tensor * q_v = ggml_add(ctx, q, b.attn_pos_v);

    // All of Q, K, V need [head_dim, T, n_head, 1] for flash_attn.
    // The permute is a zero-cost view (no data copy). flash_attn_ext
    // accesses data via strides, so contiguous materialization (cont)
    // is not required for q_u/k/v. q_v and p still need cont because
    // they feed into mul_mat for the position score computation.
    q_u = ggml_permute(ctx, q_u, 0, 2, 1, 3);
    q_v = ggml_cont(ctx, ggml_permute(ctx, q_v, 0, 2, 1, 3));

    k = ggml_reshape_4d(ctx, k, head_dim, n_head, T_q, 1);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);

    v = ggml_reshape_4d(ctx, v, head_dim, n_head, T_q, 1);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    p = ggml_reshape_4d(ctx, p, head_dim, n_head, pos_len, 1);
    p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    // ----- Position mask / bias -----------------------------------
    // matrix_bd = rel_shift(q_v @ p^T), truncated to [T_q, T_q].
    ggml_tensor * matrix_bd = ggml_mul_mat(ctx, p, q_v);
    matrix_bd = rel_shift(ctx, matrix_bd);
    matrix_bd = ggml_view_4d(ctx, matrix_bd,
                             T_q, T_q, n_head, 1,
                             matrix_bd->nb[1], matrix_bd->nb[2],
                             matrix_bd->nb[3], /*offset=*/0);
    // The view is non-contiguous (nb[1] stays at parent's
    // pos_len*es), but it IS contiguous-rows. The flash path calls
    // ggml_scale which wants full contiguity, so cont there. The
    // manual path only feeds matrix_bd into ggml_add(kq, matrix_bd),
    // which handles contiguous-rows inputs on both CPU and Metal.
    if (use_flash) {
        matrix_bd = ggml_cont(ctx, matrix_bd);
    }

    ggml_tensor * o;

    if (use_flash) {
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
    o = ggml_reshape_2d(ctx, o, d_model, T_q);

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
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_attn_w, b.norm_attn_b);
        ggml_tensor * attn_out = rel_pos_mhsa(ctx, x_norm, pos_emb, b,
                                              params.d_model, params.n_head,
                                              params.kv_type, params.use_flash);
        x = ggml_add(ctx, x, attn_out);
    }
    notify("after_attn", x);

    // Convolution module. Full residual.
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_conv_w, b.norm_conv_b);
        ggml_tensor * conv_out = conv_module(ctx, x_norm, b,
                                             params.conv_kernel,
                                             params.policy);
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
                               const char *          error_tag)
{
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
    x = name_prefixed(x, name_prefix, "mel_t");

    // conv0 (standard 2D conv: 1 in, channels out, k=3 s=2 p=1)
    x = ggml_conv_2d(ctx, pe.conv0_w, x,
                     /*s0=*/2, /*s1=*/2,
                     /*p0=*/1, /*p1=*/1,
                     /*d0=*/1, /*d1=*/1);
    x = add_conv_bias(ctx, x, pe.conv0_b);
    x = name_prefixed(x, name_prefix, "conv0");
    x = ggml_relu(ctx, x);
    x = name_prefixed(x, name_prefix, "relu0");

    // conv2 (depthwise: channels -> channels, groups=channels, k=3 s=2 p=1)
    // See conv_2d_dw_f32 above for the Metal backstory on why the
    // im2col path is used when direct_dw_in_pre_encode is false.
    if (policy.direct_dw_in_pre_encode) {
        x = ggml_conv_2d_dw_direct(ctx, pe.conv2_w, x,
                                   /*s0=*/2, /*s1=*/2,
                                   /*p0=*/1, /*p1=*/1,
                                   /*d0=*/1, /*d1=*/1);
    } else {
        x = conv_2d_dw_f32(ctx, pe.conv2_w, x,
                           /*s0=*/2, /*s1=*/2,
                           /*p0=*/1, /*p1=*/1,
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

    // conv5 (depthwise) -> conv6 (pointwise) -> ReLU
    if (policy.direct_dw_in_pre_encode) {
        x = ggml_conv_2d_dw_direct(ctx, pe.conv5_w, x,
                                   /*s0=*/2, /*s1=*/2,
                                   /*p0=*/1, /*p1=*/1,
                                   /*d0=*/1, /*d1=*/1);
    } else {
        x = conv_2d_dw_f32(ctx, pe.conv5_w, x,
                           /*s0=*/2, /*s1=*/2,
                           /*p0=*/1, /*p1=*/1,
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
    if (pe.out_w != nullptr && pre_encode_in != pe.out_w->ne[0]) {
        std::fprintf(stderr,
                     "%s encoder: pre_encode_in mismatch: "
                     "F'*C=%lld but out_w expects %lld\n",
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
    x = ggml_reshape_2d(ctx, x, pre_encode_in, T_enc);
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
