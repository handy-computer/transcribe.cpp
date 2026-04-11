// arch/cohere/encoder.cpp - Cohere ASR Conformer encoder graph builder.
//
// Forked from parakeet/encoder.cpp. Key differences:
//   - FFN layers have bias (add after each mul_mat)
//   - Attention Q, K, V, out projections have bias
//   - Conv pointwise and depthwise ops have bias
//
// The pre_encode structure and positional encoding are identical.

#include "encoder.h"

#include "weights.h"

#include "transcribe-debug.h"

#include "ggml.h"

// Helper: mark a tensor as a graph output so the scheduler preserves
// its buffer. Only done when the debug dumper is active.
static void mark_for_dump(ggml_tensor * t) {
    if (t != nullptr && transcribe::debug::enabled()) {
        ggml_set_output(t);
    }
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace transcribe::cohere {

namespace {

constexpr float kLayerNormEps = 1e-5f;

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr) {
        ggml_set_name(t, name);
    }
    return t;
}

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

// FeedForward with bias: y = Linear2(SiLU(Linear1(x)))
// where Linear(x) = mul_mat(W, x) + b.
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

ggml_tensor * rel_shift(ggml_context * ctx, ggml_tensor * x) {
    const int64_t pos_len = x->ne[0];
    const int64_t T_q     = x->ne[1];
    const int64_t H       = x->ne[2];
    const int64_t B       = x->ne[3];

    ggml_tensor * zero_template = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                     1, T_q, H, B);
    ggml_tensor * zeros = ggml_fill(ctx, zero_template, 0.0f);
    ggml_tensor * y = ggml_concat(ctx, zeros, x, /*dim=*/0);

    y = ggml_reshape_4d(ctx, y, T_q, pos_len + 1, H, B);

    y = ggml_view_4d(ctx, y, T_q, pos_len, H, B,
                     y->nb[1], y->nb[2], y->nb[3],
                     /*offset=*/y->nb[1]);

    y = ggml_cont(ctx, y);
    y = ggml_reshape_4d(ctx, y, pos_len, T_q, H, B);
    return y;
}

// f32-friendly Conv1D.
ggml_tensor * conv_1d_f32(ggml_context * ctx,
                          ggml_tensor *  kernel,
                          ggml_tensor *  data,
                          int            stride,
                          int            padding,
                          int            dilation)
{
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, data,
                                       stride, 0,
                                       padding, 0,
                                       dilation, 0,
                                       false,
                                       kernel->type);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col,
                        im2col->ne[0],
                        im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel,
                        kernel->ne[0] * kernel->ne[1],
                        kernel->ne[2]));
    result = ggml_reshape_3d(ctx, result,
                             im2col->ne[1],
                             kernel->ne[2],
                             im2col->ne[2]);
    return result;
}

// f32-friendly 2D depthwise conv.
ggml_tensor * conv_2d_dw_f32(ggml_context * ctx,
                             ggml_tensor *  kernel,
                             ggml_tensor *  data,
                             int            s0, int s1,
                             int            p0, int p1,
                             int            d0, int d1)
{
    ggml_tensor * new_a = ggml_reshape_4d(ctx, kernel,
        kernel->ne[0], kernel->ne[1], 1,
        kernel->ne[2] * kernel->ne[3]);
    ggml_tensor * data_4d = ggml_reshape_4d(ctx, data,
        data->ne[0], data->ne[1], 1,
        data->ne[2] * data->ne[3]);
    ggml_tensor * im2col = ggml_im2col(ctx, new_a, data_4d,
        s0, s1, p0, p1, d0, d1,
        true,
        kernel->type);
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

// f32-friendly depthwise Conv1D.
ggml_tensor * conv_1d_dw_f32(ggml_context * ctx,
                             ggml_tensor *  kernel,
                             ggml_tensor *  data,
                             int            stride,
                             int            padding,
                             int            dilation)
{
    ggml_tensor * data_4d = ggml_reshape_4d(ctx, data,
                                            data->ne[0], 1,
                                            data->ne[1], data->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, data_4d,
                                       stride, 0,
                                       padding, 0,
                                       dilation, 0,
                                       false,
                                       kernel->type);
    ggml_tensor * result = ggml_mul_mat(ctx, im2col, kernel);
    result = ggml_reshape_3d(ctx, result,
                             result->ne[0], result->ne[2], 1);
    return result;
}

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

bool detect_direct_pw(const char * backend) {
    const char * env = std::getenv("TRANSCRIBE_CONV_NO_DIRECT_PW");
    if (env != nullptr) return false;
    env = std::getenv("TRANSCRIBE_CONV_DIRECT_PW");
    if (env != nullptr) return true;
    if (std::strstr(backend, "Vulkan") != nullptr) return false;
    return true;
}

bool detect_direct_dw(const char * backend) {
    const char * env = std::getenv("TRANSCRIBE_CONV_DIRECT_DW");
    if (env != nullptr) return true;
    env = std::getenv("TRANSCRIBE_CONV_NO_DIRECT_DW");
    if (env != nullptr) return false;
    // Vulkan and CUDA have native conv_2d_dw kernels.
    // Metal does NOT — use the im2col + mul_mat path to stay on GPU.
    if (std::strstr(backend, "Vulkan") != nullptr) return true;
    if (std::strstr(backend, "CUDA")   != nullptr) return true;
    return false;
}

// Add a 1-D bias [C] to a 4-D conv output [W, H, C, N].
ggml_tensor * add_conv_bias(ggml_context * ctx,
                            ggml_tensor *  conv_out,
                            ggml_tensor *  bias_1d)
{
    if (bias_1d == nullptr) return conv_out;
    const int64_t channels = bias_1d->ne[0];
    ggml_tensor * bias_4d = ggml_reshape_4d(ctx, bias_1d, 1, 1, channels, 1);
    return ggml_add(ctx, conv_out, bias_4d);
}

// Convolution module with bias on all ops.
ggml_tensor * conv_module(ggml_context * ctx,
                          ggml_tensor *  x,
                          const CohereBlock & b,
                          int            conv_kernel,
                          bool           direct_pw,
                          bool           direct_dw)
{
    const int padding = (conv_kernel - 1) / 2;
    const int64_t d_model = x->ne[0];

    if (direct_pw) {
        // Pointwise conv 1 as direct mul_mat.
        {
            ggml_tensor * pw1 = ggml_reshape_2d(ctx, b.conv_pw1_w,
                                                d_model, 2 * d_model);
            x = ggml_mul_mat(ctx, pw1, x);
            if (b.conv_pw1_b != nullptr) {
                x = ggml_add(ctx, x, b.conv_pw1_b);
            }
        }

        // GLU.
        {
            const int64_t T    = x->ne[1];
            const int64_t half = x->ne[0] / 2;
            ggml_tensor * gate  = ggml_view_2d(ctx, x, half, T,
                                               x->nb[1], 0);
            ggml_tensor * value = ggml_view_2d(ctx, x, half, T,
                                               x->nb[1],
                                               half * ggml_element_size(x));
            x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
        }

        // Transpose for depthwise conv: [d_model, T] -> [T, d_model].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    } else {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        x = conv_1d_f32(ctx, b.conv_pw1_w, x, 1, 0, 1);
        // Add pointwise1 bias in [T, 2*d_model] layout.
        if (b.conv_pw1_b != nullptr) {
            // conv output is [T, 2*d_model, 1]. Bias is [2*d_model].
            // Need to reshape bias for broadcast.
            x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
            ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_pw1_b, 1, 2 * d_model);
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

    // Depthwise conv.
    if (direct_dw) {
        ggml_tensor * knl = ggml_reshape_4d(ctx, b.conv_dw_w,
                                            conv_kernel, 1, 1, d_model);
        ggml_tensor * data = ggml_reshape_4d(ctx, x,
                                             x->ne[0], 1, x->ne[1], 1);
        x = ggml_conv_2d_dw_direct(ctx, knl, data,
                                   1, 1, padding, 0, 1, 0);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[2]);
    } else {
        x = conv_1d_dw_f32(ctx, b.conv_dw_w, x, 1, padding, 1);
    }

    // Add depthwise bias in [T, d_model] layout.
    if (b.conv_dw_b != nullptr) {
        ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_dw_b, 1, d_model);
        x = ggml_add(ctx, x, bias_r);
    }

    // Fused BatchNorm.
    x = fused_batch_norm(ctx, x,
                         b.conv_bn_fused_scale, b.conv_bn_fused_bias);

    // SiLU activation.
    x = ggml_silu(ctx, x);

    if (direct_pw) {
        // Transpose back: [T, d_model] -> [d_model, T].
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        ggml_tensor * pw2 = ggml_reshape_2d(ctx, b.conv_pw2_w,
                                            d_model, d_model);
        x = ggml_mul_mat(ctx, pw2, x);
        if (b.conv_pw2_b != nullptr) {
            x = ggml_add(ctx, x, b.conv_pw2_b);
        }
    } else {
        x = conv_1d_f32(ctx, b.conv_pw2_w, x, 1, 0, 1);
        // Add pointwise2 bias.
        if (b.conv_pw2_b != nullptr) {
            x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
            ggml_tensor * bias_r = ggml_reshape_2d(ctx, b.conv_pw2_b, 1, d_model);
            x = ggml_add(ctx, x, bias_r);
        }
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    }

    return x;
}

// Relative-position multi-head self-attention with bias.
//
// use_flash = true  -> ggml_flash_attn_ext (fused GPU kernel, needs
//                      dk template in Metal backend)
// use_flash = false -> manual mul_mat + softmax + mul_mat (works on
//                      any backend, useful for comparison or when the
//                      head dim has no flash kernel)
ggml_tensor * rel_pos_mhsa(ggml_context * ctx,
                           ggml_tensor *  x,
                           ggml_tensor *  pos_emb,
                           const CohereBlock & b,
                           int            d_model,
                           int            n_head,
                           ggml_type      kv_type,
                           bool           use_flash)
{
    const int     head_dim = d_model / n_head;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T_q      = x->ne[1];
    const int64_t pos_len  = pos_emb->ne[1];

    // Q, K, V, P projections with bias.
    ggml_tensor * q = ggml_mul_mat(ctx, b.attn_q_w, x);
    if (b.attn_q_b != nullptr) q = ggml_add(ctx, q, b.attn_q_b);
    ggml_tensor * k = ggml_mul_mat(ctx, b.attn_k_w, x);
    if (b.attn_k_b != nullptr) k = ggml_add(ctx, k, b.attn_k_b);
    ggml_tensor * v = ggml_mul_mat(ctx, b.attn_v_w, x);
    if (b.attn_v_b != nullptr) v = ggml_add(ctx, v, b.attn_v_b);
    ggml_tensor * p = ggml_mul_mat(ctx, b.attn_pos_w, pos_emb);

    // Split heads.
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, T_q, 1);
    ggml_tensor * q_u = ggml_add(ctx, q, b.attn_pos_u);
    ggml_tensor * q_v = ggml_add(ctx, q, b.attn_pos_v);

    q_u = ggml_permute(ctx, q_u, 0, 2, 1, 3);
    q_v = ggml_permute(ctx, q_v, 0, 2, 1, 3);

    k = ggml_reshape_4d(ctx, k, head_dim, n_head, T_q, 1);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);

    v = ggml_reshape_4d(ctx, v, head_dim, n_head, T_q, 1);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    p = ggml_reshape_4d(ctx, p, head_dim, n_head, pos_len, 1);
    p = ggml_permute(ctx, p, 0, 2, 1, 3);

    // Relative position bias: BD = rel_shift(q_v @ p^T), truncated to [T_q, T_q].
    ggml_tensor * matrix_bd = ggml_mul_mat(ctx, p, q_v);
    matrix_bd = rel_shift(ctx, matrix_bd);
    matrix_bd = ggml_view_4d(ctx, matrix_bd,
                             T_q, T_q, n_head, 1,
                             matrix_bd->nb[1], matrix_bd->nb[2],
                             matrix_bd->nb[3], 0);
    // The view is non-contiguous (nb[1] stays at parent's pos_len*es),
    // but it IS contiguous-rows. The flash path calls ggml_scale on
    // CPU which wants full contiguity, so cont there. The standard
    // path only feeds matrix_bd into ggml_add(kq, matrix_bd), which
    // handles contiguous-rows inputs on both CPU and Metal (Metal
    // add and soft_max both pass nb11/12/13 through to the kernel).
    if (use_flash) {
        matrix_bd = ggml_cont(ctx, matrix_bd);
    }

    ggml_tensor * o;

    if (use_flash) {
        // Flash attention path: fused Q@K^T + mask + softmax + @V.
        // matrix_bd is pre-scaled and cast to f16 for the mask arg.
        matrix_bd = ggml_scale(ctx, matrix_bd, scale);
        matrix_bd = ggml_cast(ctx, matrix_bd, GGML_TYPE_F16);

        // Optional K/V type narrowing.
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
                                scale, 0.0f, 0.0f);
    } else {
        // Manual attention path: explicit matmuls + softmax.
        //
        //   attn_scores = q_u @ k^T                    [T_q, T_q, n_head, 1]
        //   attn_scores = (attn_scores + matrix_bd) * scale
        //   attn_weights = softmax(attn_scores)
        //   o = attn_weights @ v
        //
        // K is [head_dim, T_q, n_head, 1] from permute(0,2,1,3).
        // mul_mat(K, Q_u) computes Q_u^T @ K for each head, giving
        // [T_q, T_q, n_head, 1] attention scores.
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q_u);

        // Add relative position bias, then scale.
        kq = ggml_add(ctx, kq, matrix_bd);

        // ggml_soft_max_ext fuses the scale into softmax.
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, nullptr,
                                                   scale, 0.0f);

        // V needs to be transposed for the value matmul.
        // V is [head_dim, T_q, n_head, 1]. We need [T_q, head_dim, n_head, 1].
        ggml_tensor * v_t = ggml_cont(ctx,
                                       ggml_permute(ctx, v, 1, 0, 2, 3));

        // attn_weights @ V: [head_dim, T_q, n_head, 1]
        o = ggml_mul_mat(ctx, v_t, kq_soft);

        // Merge heads: permute [head_dim, T_q, n_head] -> [head_dim, n_head, T_q]
        // then reshape to [d_model, T_q]. The permute makes heads contiguous
        // for the reshape.
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
    }
    o = ggml_reshape_2d(ctx, o, d_model, T_q);

    // Output linear with bias.
    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    if (b.attn_out_b != nullptr) o = ggml_add(ctx, o, b.attn_out_b);
    return o;
}

// One conformer block forward.
ggml_tensor * build_conformer_block(ggml_context * ctx,
                                    ggml_tensor *  x,
                                    ggml_tensor *  pos_emb,
                                    const CohereBlock & b,
                                    int            d_model,
                                    int            n_head,
                                    int            conv_kernel,
                                    ggml_type      kv_type,
                                    bool           use_flash,
                                    bool           direct_pw,
                                    bool           direct_dw)
{
    // Macaron FF1: x = x + 0.5 * FF1(LN(x))
    x = macaron_ff_residual(ctx, x,
                            b.norm_ff1_w, b.norm_ff1_b,
                            b.ff1_lin1_w, b.ff1_lin1_b,
                            b.ff1_lin2_w, b.ff1_lin2_b);

    // Self-attention. Full residual.
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_attn_w, b.norm_attn_b);
        ggml_tensor * attn_out = rel_pos_mhsa(ctx, x_norm, pos_emb,
                                              b, d_model, n_head,
                                              kv_type, use_flash);
        x = ggml_add(ctx, x, attn_out);
    }

    // Convolution module. Full residual.
    {
        ggml_tensor * x_norm = layer_norm(ctx, x,
                                          b.norm_conv_w, b.norm_conv_b);
        ggml_tensor * conv_out = conv_module(ctx, x_norm, b, conv_kernel,
                                             direct_pw, direct_dw);
        x = ggml_add(ctx, x, conv_out);
    }

    // Macaron FF2.
    x = macaron_ff_residual(ctx, x,
                            b.norm_ff2_w, b.norm_ff2_b,
                            b.ff2_lin1_w, b.ff2_lin1_b,
                            b.ff2_lin2_w, b.ff2_lin2_b);

    // Final per-block LayerNorm.
    x = layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
    return x;
}

// Pre-encode subsampling stack (identical to Parakeet).
ggml_tensor * build_pre_encode(ggml_context *         ctx,
                               const CohereWeights &  w,
                               const CohereHParams &  hp,
                               ggml_tensor *          mel_in)
{
    const auto & pe = w.pre_encode;

    ggml_tensor * x = ggml_permute(ctx, mel_in, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

    // conv0
    x = ggml_conv_2d(ctx, pe.conv0_w, x, 2, 2, 1, 1, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv0_b);
    x = ggml_relu(ctx, x);

    // conv2 (depthwise) -> conv3 (pointwise) -> ReLU
    x = conv_2d_dw_f32(ctx, pe.conv2_w, x, 2, 2, 1, 1, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv2_b);

    x = ggml_conv_2d(ctx, pe.conv3_w, x, 1, 1, 0, 0, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv3_b);
    x = ggml_relu(ctx, x);

    // conv5 (depthwise) -> conv6 (pointwise) -> ReLU
    x = conv_2d_dw_f32(ctx, pe.conv5_w, x, 2, 2, 1, 1, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv5_b);

    x = ggml_conv_2d(ctx, pe.conv6_w, x, 1, 1, 0, 0, 1, 1);
    x = add_conv_bias(ctx, x, pe.conv6_b);
    x = ggml_relu(ctx, x);

    // Flatten [F', T_enc, C, 1] -> [F'*C, T_enc]
    const int64_t F_prime = x->ne[0];
    const int64_t T_enc   = x->ne[1];
    const int64_t C       = x->ne[2];
    const int64_t pre_encode_in = F_prime * C;

    if (pre_encode_in != pe.out_w->ne[0]) {
        std::fprintf(stderr,
                     "cohere encoder: pre_encode_in mismatch: "
                     "F'*C=%lld but out_w expects %lld\n",
                     static_cast<long long>(pre_encode_in),
                     static_cast<long long>(pe.out_w->ne[0]));
        return nullptr;
    }
    (void)hp;

    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    x = ggml_cont(ctx, x);
    x = ggml_reshape_2d(ctx, x, pre_encode_in, T_enc);

    // Linear projection to d_model.
    x = ggml_mul_mat(ctx, pe.out_w, x);
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

EncoderBuild build_encoder_graph(ggml_context *         ctx,
                                 const CohereWeights &  w,
                                 const CohereHParams &  hp,
                                 int                    n_mel_frames,
                                 ggml_type              kv_type,
                                 bool                   use_flash,
                                 const char *           backend_name)
{
    const bool direct_pw = detect_direct_pw(backend_name);
    const bool direct_dw = detect_direct_dw(backend_name);

    EncoderBuild eb {};

    if (ctx == nullptr || n_mel_frames <= 0) {
        std::fprintf(stderr,
                     "cohere encoder: invalid arg "
                     "(ctx=%p, n_mel_frames=%d)\n",
                     static_cast<void *>(ctx), n_mel_frames);
        return eb;
    }

    if (hp.enc_subsampling_factor != 8 || hp.fe_num_mels != 128) {
        std::fprintf(stderr,
                     "cohere encoder: unsupported geometry "
                     "subsampling_factor=%d num_mels=%d "
                     "(only 8/128 implemented)\n",
                     hp.enc_subsampling_factor, hp.fe_num_mels);
        return eb;
    }

    // Mel input handle.
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                   n_mel_frames, hp.fe_num_mels);
    if (eb.mel_in == nullptr) return eb;
    ggml_set_name(eb.mel_in, "mel.in");
    ggml_set_input(eb.mel_in);

    // Pre-encode.
    ggml_tensor * x = build_pre_encode(ctx, w, hp, eb.mel_in);
    if (x == nullptr) return eb;
    eb.dumps.pre_encode_out = x;
    mark_for_dump(x);

    // Conformer blocks.
    if (!w.blocks.empty()) {
        const int64_t T_enc = x->ne[1];

        // Positional embedding input.
        const int64_t pos_len = 2 * T_enc - 1;
        eb.pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           hp.enc_d_model, pos_len);
        ggml_set_name(eb.pos_emb_in, "pos_emb.in");
        ggml_set_input(eb.pos_emb_in);

        // Block 0 (hand-built for detailed dumps).
        {
            const auto & b0 = w.blocks[0];
            x = build_conformer_block(ctx, x, eb.pos_emb_in,
                                      b0, hp.enc_d_model, hp.enc_n_heads,
                                      hp.enc_conv_kernel, kv_type,
                                      use_flash, direct_pw, direct_dw);
            x = named(x, "enc.block.0.out");
            eb.dumps.block0_out = x;
            mark_for_dump(x);
        }

        // Blocks 1..N-1.
        for (size_t i = 1; i < w.blocks.size(); ++i) {
            x = build_conformer_block(ctx, x, eb.pos_emb_in,
                                      w.blocks[i],
                                      hp.enc_d_model, hp.enc_n_heads,
                                      hp.enc_conv_kernel, kv_type,
                                      use_flash, direct_pw, direct_dw);
            // Spot-check at the middle block (n_layers/2 - 1 to match ref).
            if (i == static_cast<size_t>(hp.enc_n_layers / 2 - 1)) {
                char bname[64];
                std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
                x = named(x, bname);
                eb.dumps.block_mid_out = x;
                mark_for_dump(x);
            }
            // Spot-check last block.
            if (i == w.blocks.size() - 1) {
                char bname[64];
                std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
                x = named(x, bname);
                eb.dumps.block_last_out = x;
                mark_for_dump(x);
            }
        }
    }

    eb.dumps.final_out = x;
    x = named(x, "enc.final");
    mark_for_dump(x);

    // Encoder-decoder projection.
    {
        ggml_tensor * proj = ggml_mul_mat(ctx, w.enc_dec_proj.weight, x);
        proj = ggml_add(ctx, proj, w.enc_dec_proj.bias);
        proj = named(proj, "enc_dec_proj.out");
        eb.dumps.enc_dec_proj_out = proj;
        mark_for_dump(proj);
        x = proj;
    }

    eb.out = x;
    ggml_set_output(eb.out);

    // Build the forward cgraph. 48 blocks + decoder means a large graph.
    // 16384 should be plenty.
    eb.graph = ggml_new_graph_custom(ctx, 16384, false);
    if (eb.graph == nullptr) {
        std::fprintf(stderr,
                     "cohere encoder: ggml_new_graph_custom failed\n");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);

    return eb;
}

} // namespace transcribe::cohere
