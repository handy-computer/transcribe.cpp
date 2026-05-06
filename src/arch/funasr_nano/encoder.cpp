// arch/funasr_nano/encoder.cpp - SAN-M encoder graph (no prefix prepend,
// no CTC head). Pruned fork of arch/sensevoice/encoder.cpp.
//
// Forward shape (single-utterance, batch=1):
//
//   frontend.in    [d_input=560, T_lfr]
//     -> scale by sqrt(d_model=512)
//     -> add sinusoidal PE (depth = d_input, 1-based positions)
//                                                       = enc.embed.out
//     -> encoders0[0] (SAN-M block, projection 560 -> 512, no attn-residual)
//                                                       = enc.encoders0.0.out
//     -> encoders[0..n-2] x49 (residual SAN-M blocks)
//                                                       = enc.encoders.{0,24,48}.out
//     -> after_norm (LayerNorm eps=1e-12)               = enc.after_norm.out
//     -> tp_encoders[0..tp-1] x20 (residual SAN-M)      = enc.tp_encoders.{0,10,19}.out
//     -> tp_norm (LayerNorm eps=1e-12)                  = enc.tp_norm.out
//
// SAN-M block layout: identical to sensevoice's. See the head comment in
// arch/sensevoice/encoder.cpp for the per-block math.

#include "encoder.h"

#include "weights.h"

#include "transcribe-debug.h"
#include "conformer/conformer.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace transcribe::funasr_nano {

namespace {

namespace conf = transcribe::conformer;
using conf::named;

constexpr float kLayerNormEps = 1e-12f;

ggml_tensor * sv_layer_norm(ggml_context * ctx,
                            ggml_tensor *  x,
                            ggml_tensor *  gamma,
                            ggml_tensor *  beta)
{
    ggml_tensor * y = ggml_norm(ctx, x, kLayerNormEps);
    y = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) y = ggml_add(ctx, y, beta);
    return y;
}

ggml_tensor * fsmn_branch(ggml_context * ctx,
                          ggml_tensor *  v_pre,        // [d_model, T]
                          ggml_tensor *  fsmn_w,       // [K, 1, d_model]
                          int            kernel)
{
    ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_pre)); // [T, d_model]
    const int padding = (kernel - 1) / 2;
    ggml_tensor * fsmn = conf::conv_1d_dw_f32(
        ctx, fsmn_w, v_t, /*stride=*/1, padding, /*dilation=*/1);
    fsmn = ggml_reshape_2d(ctx, fsmn, fsmn->ne[0], fsmn->ne[1]); // [T, d_model]
    fsmn = ggml_cont(ctx, ggml_transpose(ctx, fsmn));            // [d_model, T]
    fsmn = ggml_add(ctx, fsmn, v_pre);
    return fsmn;
}

ggml_tensor * sanm_attention(ggml_context *  ctx,
                             ggml_tensor *   x,         // [d_in, T]
                             const EncBlock & b,
                             int              n_heads,
                             int              d_model,
                             int              kernel)
{
    const int head_dim = d_model / n_heads;
    const int64_t T    = x->ne[1];
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_tensor * qkv = ggml_mul_mat(ctx, b.attn_qkv_w, x);
    qkv = ggml_add(ctx, qkv, b.attn_qkv_b);

    const size_t qkv_nb1 = qkv->nb[1];
    ggml_tensor * q = ggml_view_2d(ctx, qkv, d_model, T, qkv_nb1, 0);
    ggml_tensor * k = ggml_view_2d(ctx, qkv, d_model, T, qkv_nb1,
                                   static_cast<size_t>(d_model) * sizeof(float));
    ggml_tensor * v = ggml_view_2d(ctx, qkv, d_model, T, qkv_nb1,
                                   static_cast<size_t>(2 * d_model) * sizeof(float));

    ggml_tensor * v_pre = ggml_cont(ctx, v);
    ggml_tensor * fsmn  = fsmn_branch(ctx, v_pre, b.attn_fsmn_w, kernel);

    auto split_heads = [&](ggml_tensor * t) {
        t = ggml_cont(ctx, t);
        return ggml_reshape_4d(ctx, t, head_dim, n_heads, T, 1);
    };
    ggml_tensor * qh = split_heads(q);
    ggml_tensor * kh = split_heads(k);
    ggml_tensor * vh = split_heads(v);

    auto to_attn_layout = [&](ggml_tensor * t) {
        t = ggml_permute(ctx, t, 0, 2, 1, 3);   // [head_dim, T, n_heads, 1]
        return ggml_cont(ctx, t);
    };
    qh = to_attn_layout(qh);
    kh = to_attn_layout(kh);
    vh = to_attn_layout(vh);

    ggml_tensor * o = ggml_flash_attn_ext(
        ctx, qh, kh, vh, /*mask=*/nullptr,
        scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    o = ggml_cont(ctx, o);
    o = ggml_reshape_2d(ctx, o, d_model, T);

    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    o = ggml_add(ctx, o, b.attn_out_b);
    return ggml_add(ctx, o, fsmn);
}

ggml_tensor * sanm_ffn(ggml_context * ctx,
                       ggml_tensor *  x,
                       const EncBlock & b)
{
    ggml_tensor * h = ggml_mul_mat(ctx, b.ffn_fc1_w, x);
    h = ggml_add(ctx, h, b.ffn_fc1_b);
    h = ggml_relu(ctx, h);
    h = ggml_mul_mat(ctx, b.ffn_fc2_w, h);
    h = ggml_add(ctx, h, b.ffn_fc2_b);
    return h;
}

ggml_tensor * sanm_block_residual(ggml_context * ctx,
                                  ggml_tensor *  x,
                                  const EncBlock & b,
                                  int n_heads, int d_model, int kernel)
{
    ggml_tensor * y = sv_layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = sanm_attention(ctx, y, b, n_heads, d_model, kernel);
    x = ggml_add(ctx, x, y);

    ggml_tensor * z = sv_layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
    z = sanm_ffn(ctx, z, b);
    return ggml_add(ctx, x, z);
}

ggml_tensor * sanm_block_projection(ggml_context * ctx,
                                    ggml_tensor *  x,
                                    const EncBlock & b,
                                    int n_heads, int d_model, int kernel)
{
    ggml_tensor * y = sv_layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = sanm_attention(ctx, y, b, n_heads, d_model, kernel);
    // projection 560 → 512: no attn residual.
    x = y;

    ggml_tensor * z = sv_layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
    z = sanm_ffn(ctx, z, b);
    return ggml_add(ctx, x, z);
}

void mark_dump(ggml_tensor *& slot, ggml_tensor * t, const char * name) {
    named(t, name);
    transcribe::debug::mark_tensor_for_dump(t);
    slot = t;
}

} // namespace

EncoderBuild build_encoder_graph(ggml_context *             ctx,
                                 const FunAsrNanoWeights &  w,
                                 const FunAsrNanoHParams &  hp,
                                 int                        n_lfr_frames)
{
    EncoderBuild eb {};
    if (ctx == nullptr || n_lfr_frames <= 0) {
        std::fprintf(stderr,
                     "funasr_nano encoder: invalid arg "
                     "(ctx=%p, n_lfr_frames=%d)\n",
                     static_cast<void *>(ctx), n_lfr_frames);
        return eb;
    }

    const int d_input = hp.enc_d_input;
    const int d_model = hp.enc_d_model;
    const int n_heads = hp.enc_n_heads;
    const int kernel  = hp.enc_kernel;
    const int T       = n_lfr_frames;

    eb.frontend_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_input, T);
    named(eb.frontend_in, "frontend.in");
    ggml_set_input(eb.frontend_in);
    mark_dump(eb.dumps.frontend_out, eb.frontend_in, "frontend.fbank.lfr.cmvn.out");

    eb.pe_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_input, T);
    named(eb.pe_in, "pe.in");
    ggml_set_input(eb.pe_in);

    // Embedding scale + PE add.
    const float embed_scale = std::sqrt(static_cast<float>(d_model));
    ggml_tensor * x = ggml_scale(ctx, eb.frontend_in, embed_scale);
    x = ggml_add(ctx, x, eb.pe_in);
    mark_dump(eb.dumps.embed_out, x, "enc.embed.out");

    // encoders0[0] (560 → 512 projection).
    x = sanm_block_projection(ctx, x, w.encoders0, n_heads, d_model, kernel);
    mark_dump(eb.dumps.encoders0_0_out, x, "enc.encoders0.0.out");

    const int n_main = static_cast<int>(w.encoders.size());
    if (n_main > 0) {
        const int last_idx = n_main - 1;
        const int mid_idx  = n_main / 2;
        for (int i = 0; i < n_main; ++i) {
            x = sanm_block_residual(ctx, x, w.encoders[i],
                                    n_heads, d_model, kernel);
            if (i == 0) {
                mark_dump(eb.dumps.encoders_first, x, "enc.encoders.0.out");
            } else if (i == mid_idx) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "enc.encoders.%d.out", i);
                mark_dump(eb.dumps.encoders_mid, x, nm);
            } else if (i == last_idx) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "enc.encoders.%d.out", i);
                mark_dump(eb.dumps.encoders_last, x, nm);
            }
        }
    }

    x = sv_layer_norm(ctx, x, w.after_norm_w, w.after_norm_b);
    mark_dump(eb.dumps.after_norm_out, x, "enc.after_norm.out");

    const int n_tp = static_cast<int>(w.tp_encoders.size());
    if (n_tp > 0) {
        const int last_idx = n_tp - 1;
        const int mid_idx  = n_tp / 2;
        for (int i = 0; i < n_tp; ++i) {
            x = sanm_block_residual(ctx, x, w.tp_encoders[i],
                                    n_heads, d_model, kernel);
            if (i == 0) {
                mark_dump(eb.dumps.tp_encoders_first, x, "enc.tp_encoders.0.out");
            } else if (i == mid_idx) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "enc.tp_encoders.%d.out", i);
                mark_dump(eb.dumps.tp_encoders_mid, x, nm);
            } else if (i == last_idx) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "enc.tp_encoders.%d.out", i);
                mark_dump(eb.dumps.tp_encoders_last, x, nm);
            }
        }
    }

    x = sv_layer_norm(ctx, x, w.tp_norm_w, w.tp_norm_b);
    mark_dump(eb.dumps.tp_norm_out, x, "enc.tp_norm.out");

    eb.out = x;
    ggml_set_output(eb.out);

    eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (eb.graph == nullptr) {
        std::fprintf(stderr, "funasr_nano encoder: ggml_new_graph_custom failed\n");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);
    return eb;
}

} // namespace transcribe::funasr_nano
