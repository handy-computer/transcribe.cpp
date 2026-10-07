#include "encoder.h"

#include "ggml.h"
#include "transcribe-log.h"

#include <cmath>

namespace transcribe::arkasr {

namespace {

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight), bias);
}

ggml_tensor * conv_bias(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
}

ggml_tensor * encoder_attention(ggml_context *         ctx,
                                ggml_tensor *          x,
                                const ArkAsrEncBlock & b,
                                const ArkAsrHParams &  hp,
                                ggml_tensor *          positions) {
    const int head_dim = hp.enc_head_dim;
    const int heads    = hp.enc_n_heads;
    const int tokens   = static_cast<int>(x->ne[1]);

    auto * q = ggml_add(ctx, ggml_mul_mat(ctx, b.q_w, x), b.q_b);
    auto * k = ggml_mul_mat(ctx, b.k_w, x);
    auto * v = ggml_add(ctx, ggml_mul_mat(ctx, b.v_w, x), b.v_b);

    q = ggml_reshape_3d(ctx, q, head_dim, heads, tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, tokens);

    q = ggml_rope_ext(ctx, q, positions, nullptr, hp.enc_rot_dim, GGML_ROPE_TYPE_NORMAL, 0, hp.enc_rope_theta, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, hp.enc_rot_dim, GGML_ROPE_TYPE_NORMAL, 0, hp.enc_rope_theta, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    auto * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
    out        = ggml_reshape_2d(ctx, out, hp.enc_d_model, tokens);
    return ggml_add(ctx, ggml_mul_mat(ctx, b.o_w, out), b.o_b);
}

}  // namespace

EncoderBuild build_encoder_graph(ggml_context *        ctx,
                                 const ArkAsrWeights & w,
                                 const ArkAsrHParams & hp,
                                 int                   n_mel_frames) {
    EncoderBuild out{};
    if (ctx == nullptr || n_mel_frames <= 0) {
        return out;
    }

    out.graph = ggml_new_graph_custom(ctx, 16384, false);
    if (out.graph == nullptr) {
        return {};
    }

    out.mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_mel_frames, hp.enc_num_mels);
    ggml_set_name(out.mel, "ark.mel");
    ggml_set_input(out.mel);

    auto * x             = ggml_conv_1d(ctx, w.conv1_w, out.mel, 1, 1, 1);
    x                    = ggml_gelu_erf(ctx, ggml_add(ctx, x, conv_bias(ctx, w.conv1_b)));
    x                    = ggml_conv_1d(ctx, w.conv2_w, x, 2, 1, 1);
    x                    = ggml_gelu_erf(ctx, ggml_add(ctx, x, conv_bias(ctx, w.conv2_b)));
    const int enc_tokens = static_cast<int>(x->ne[0]);
    x                    = ggml_cont(ctx, ggml_transpose(ctx, x));

    out.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, enc_tokens);
    ggml_set_name(out.positions, "ark.encoder_positions");
    ggml_set_input(out.positions);

    for (int layer = 0; layer < hp.enc_n_layers; ++layer) {
        const auto & b        = w.enc_blocks[layer];
        auto *       residual = x;
        auto *       h        = layer_norm(ctx, x, b.attn_ln_w, b.attn_ln_b, hp.enc_ln_eps);
        x                     = ggml_add(ctx, residual, encoder_attention(ctx, h, b, hp, out.positions));

        residual = x;
        h        = layer_norm(ctx, x, b.ffn_ln_w, b.ffn_ln_b, hp.enc_ln_eps);
        h        = ggml_add(ctx, ggml_mul_mat(ctx, b.fc1_w, h), b.fc1_b);
        h        = ggml_gelu_erf(ctx, h);
        h        = ggml_add(ctx, ggml_mul_mat(ctx, b.fc2_w, h), b.fc2_b);
        x        = ggml_add(ctx, residual, h);
    }

    x                = layer_norm(ctx, x, w.adapter_ln_w, w.adapter_ln_b, hp.enc_ln_eps);
    const int usable = (enc_tokens / hp.merge_factor) * hp.merge_factor;
    if (usable <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: audio is too short after encoder subsampling");
        return {};
    }
    if (usable != enc_tokens) {
        x = ggml_cont(ctx, ggml_view_2d(ctx, x, hp.enc_d_model, usable, x->nb[1], 0));
    }

    out.n_audio = usable / hp.merge_factor;
    x           = ggml_reshape_2d(ctx, x, hp.enc_d_model * hp.merge_factor, out.n_audio);
    x           = ggml_add(ctx, ggml_mul_mat(ctx, w.adapter_fc1_w, x), w.adapter_fc1_b);
    x           = ggml_gelu_erf(ctx, x);
    x           = ggml_add(ctx, ggml_mul_mat(ctx, w.adapter_fc2_w, x), w.adapter_fc2_b);

    out.output = x;
    ggml_set_name(out.output, "ark.audio_embeddings");
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.graph, out.output);
    return out;
}

}  // namespace transcribe::arkasr
