// arch/moss/encoder.cpp - MOSS Whisper encoder + VQAdaptor graph builders.
//
// The encoder is stock HF WhisperEncoder (see arch/whisper/encoder.cpp): a
// 2-layer Conv1d stem (GELU-erf) + learned positional embedding + N pre-LN
// transformer blocks (LayerNorm, gelu FFN; q/v/out carry bias, k does not) +
// final LayerNorm. The adaptor consumes the 4x-time-merged, per-chunk-trimmed,
// concatenated encoder output.

#include "encoder.h"

#include "conformer/conformer.h"
#include "ggml.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"

#include <cmath>
#include <cstdio>

namespace transcribe::moss {

namespace {

namespace conf = transcribe::conformer;
using conf::layer_norm;
using conf::named;

// LayerNorm with an explicit eps (the VQAdaptor uses 1e-6, not conformer's 1e-5).
ggml_tensor * layer_norm_eps(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gamma, ggml_tensor * beta, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y               = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) {
        y = ggml_add(ctx, y, beta);
    }
    return y;
}

// Reshape a 1D conv bias [Cout] to [1, Cout, 1, 1] so it broadcasts across T
// against a ggml_conv_1d output [T_out, Cout, 1, 1].
ggml_tensor * add_conv1d_bias(ggml_context * ctx, ggml_tensor * conv_out, ggml_tensor * bias_1d) {
    if (bias_1d == nullptr) {
        return conv_out;
    }
    const int64_t channels = bias_1d->ne[0];
    ggml_tensor * bias_4d  = ggml_reshape_4d(ctx, bias_1d, 1, channels, 1, 1);
    return ggml_add(ctx, conv_out, bias_4d);
}

// Bidirectional multi-head self-attention. x: [d_model, T] -> [d_model, T].
// Whisper quirk: q/v/out carry bias, k does not.
ggml_tensor * mha_encoder(ggml_context * ctx,
                          ggml_tensor *  x,
                          const MossEncBlock & b,
                          int            n_heads,
                          int            d_model,
                          bool           use_flash) {
    const int     head_dim = d_model / n_heads;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T        = x->ne[1];

    ggml_tensor * q = ggml_mul_mat(ctx, b.attn_q_w, x);
    q               = ggml_add(ctx, q, b.attn_q_b);
    ggml_tensor * k = ggml_mul_mat(ctx, b.attn_k_w, x);
    ggml_tensor * v = ggml_mul_mat(ctx, b.attn_v_w, x);
    v               = ggml_add(ctx, v, b.attn_v_b);

    q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, n_heads, T), 0, 2, 1, 3);
    k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, n_heads, T), 0, 2, 1, 3);
    v = ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, n_heads, T), 0, 2, 1, 3);

    ggml_tensor * o;
    if (use_flash) {
        ggml_tensor * q_c = ggml_cont(ctx, q);
        ggml_tensor * k_c = ggml_cont(ctx, k);
        ggml_tensor * v_c = ggml_cont(ctx, v);
        o                 = ggml_flash_attn_ext(ctx, q_c, k_c, v_c, nullptr, scale, 0.0f, 0.0f);
        o                 = ggml_reshape_2d(ctx, o, d_model, T);
    } else {
        ggml_tensor * kq      = ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, q));
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * v_t     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        o                     = ggml_mul_mat(ctx, v_t, kq_soft);
        o                     = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
        o                     = ggml_reshape_2d(ctx, o, d_model, T);
    }

    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    o = ggml_add(ctx, o, b.attn_out_b);
    return o;
}

// y = x + MHSA(LN(x)); y = y + FFN(LN(y)).  FFN: fc2(GELU-erf(fc1(x))).
ggml_tensor * build_block(ggml_context * ctx, ggml_tensor * x, const MossEncBlock & b, int n_heads, int d_model,
                          bool use_flash) {
    {
        ggml_tensor * y = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
        y               = mha_encoder(ctx, y, b, n_heads, d_model, use_flash);
        x               = ggml_add(ctx, x, y);
    }
    {
        ggml_tensor * y = layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
        y               = ggml_mul_mat(ctx, b.ffn_fc1_w, y);
        y               = ggml_add(ctx, y, b.ffn_fc1_b);
        y               = ggml_gelu_erf(ctx, y);
        y               = ggml_mul_mat(ctx, b.ffn_fc2_w, y);
        y               = ggml_add(ctx, y, b.ffn_fc2_b);
        x               = ggml_add(ctx, x, y);
    }
    return x;
}

}  // namespace

int whisper_enc_len(int n_mel_frames) {
    return n_mel_frames / 2;
}

int audio_token_length(int num_samples, const MossHParams & hp) {
    const int stride = hp.fe_hop_length * 2 * hp.audio_merge_size;
    if (stride <= 0 || num_samples <= 0) {
        return 0;
    }
    return (num_samples - 1) / stride + 1;
}

EncoderBuild build_encoder_graph(ggml_context * ctx, const MossWeights & w, const MossHParams & hp, int n_mel_frames,
                                 bool use_flash) {
    EncoderBuild eb{};
    if (ctx == nullptr || n_mel_frames <= 0 || n_mel_frames % 2 != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encoder: invalid n_mel_frames=%d", n_mel_frames);
        return eb;
    }

    const int d_model = hp.enc_d_model;
    const int n_mels  = hp.enc_num_mel_bins;
    const int n_heads = hp.enc_n_heads;
    const int T_enc   = n_mel_frames / 2;
    if (T_enc > hp.enc_max_source_positions) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encoder: T_enc=%d exceeds max_source_positions=%d", T_enc,
                hp.enc_max_source_positions);
        return eb;
    }

    // mel_in ne=[T, n_mels]: MelFrontend emits row-major [num_mels, n_frames]
    // (t-fastest), which IS ggml [T, n_mels] — and exactly the [W, IC] layout
    // ggml_conv_1d wants, so no leading transpose (cf. qwen3_asr).
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_mel_frames, n_mels);
    named(eb.mel_in, "enc.mel.in");
    ggml_set_input(eb.mel_in);

    // conv stem.
    ggml_tensor * x = conf::conv_1d_f32(ctx, w.enc_stem.conv0_w, eb.mel_in, 1, 1, 1);  // [T, d_model]
    x               = add_conv1d_bias(ctx, x, w.enc_stem.conv0_b);
    x               = ggml_gelu_erf(ctx, x);

    x = conf::conv_1d_f32(ctx, w.enc_stem.conv1_w, x, 2, 1, 1);  // [T_enc, d_model]
    x = add_conv1d_bias(ctx, x, w.enc_stem.conv1_b);
    x = ggml_gelu_erf(ctx, x);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [d_model, T_enc]

    // + learned positional embedding (prefix view when T_enc < max).
    ggml_tensor * pos_emb = w.enc_top.pos_emb_w;
    if (T_enc != hp.enc_max_source_positions) {
        pos_emb = ggml_view_2d(ctx, w.enc_top.pos_emb_w, d_model, T_enc, w.enc_top.pos_emb_w->nb[1], 0);
    }
    x = ggml_add(ctx, x, pos_emb);
    named(x, "enc.pos_add.out");
    eb.dumps.pos_add_out = x;
    transcribe::debug::mark_tensor_for_dump(x);

    const int n_blocks = static_cast<int>(w.enc_blocks.size());
    for (int i = 0; i < n_blocks; ++i) {
        x = build_block(ctx, x, w.enc_blocks[i], n_heads, d_model, use_flash);
        if (i == 0 || i == n_blocks - 1) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            named(x, nm);
            transcribe::debug::mark_tensor_for_dump(x);
            if (i == 0) {
                eb.dumps.block_0_out = x;
            } else {
                eb.dumps.block_last_out = x;
            }
        }
    }

    x = layer_norm(ctx, x, w.enc_top.final_norm_w, w.enc_top.final_norm_b);
    named(x, "enc.ln_post.out");
    eb.dumps.ln_post_out = x;
    transcribe::debug::mark_tensor_for_dump(x);

    eb.out = x;
    ggml_set_output(eb.out);

    eb.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (eb.graph == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encoder: ggml_new_graph_custom failed");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);
    if (eb.dumps.pos_add_out) {
        ggml_build_forward_expand(eb.graph, eb.dumps.pos_add_out);
    }
    if (eb.dumps.block_0_out) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_0_out);
    }
    if (eb.dumps.block_last_out) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_last_out);
    }
    return eb;
}

AdaptorBuild build_adaptor_graph(ggml_context * ctx, const MossWeights & w, const MossHParams & hp, int T_trim) {
    AdaptorBuild ab{};
    const int merge = hp.audio_merge_size;
    if (ctx == nullptr || T_trim <= 0 || merge <= 0 || T_trim % merge != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss adaptor: invalid T_trim=%d (merge=%d)", T_trim, merge);
        return ab;
    }
    const int d_model = hp.enc_d_model;
    const int T_enc   = T_trim / merge;
    ab.T_enc          = T_enc;

    ab.in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_trim);
    named(ab.in, "enc.merge.in");
    ggml_set_input(ab.in);

    // 4x time merge: [d_model, T_trim] -> [d_model*merge, T_enc]. Row-major
    // group-of-merge over consecutive time frames (contiguous input). cont so
    // the tensor is materialized (not a view of the input, whose buffer the
    // scheduler recycles after fc1 consumes it — which would corrupt the dump).
    ggml_tensor * merged = ggml_cont(ctx, ggml_reshape_2d(ctx, ab.in, static_cast<int64_t>(d_model) * merge, T_enc));
    named(merged, "enc.merge.out");
    ab.dumps.merge_out = merged;
    transcribe::debug::mark_tensor_for_dump(merged);

    // VQAdaptor: fc1 + SiLU + fc2 + LayerNorm(eps 1e-6).
    ggml_tensor * y = ggml_mul_mat(ctx, w.adaptor.fc1_w, merged);
    y               = ggml_add(ctx, y, w.adaptor.fc1_b);
    y               = ggml_silu(ctx, y);
    y               = ggml_mul_mat(ctx, w.adaptor.fc2_w, y);
    y               = ggml_add(ctx, y, w.adaptor.fc2_b);
    y = layer_norm_eps(ctx, y, w.adaptor.norm_out_w, w.adaptor.norm_out_b, hp.dec_rms_norm_eps);
    named(y, "enc.adaptor.out");
    ab.dumps.adaptor_out = y;
    transcribe::debug::mark_tensor_for_dump(y);

    ab.out = y;
    ggml_set_output(ab.out);

    ab.graph = ggml_new_graph_custom(ctx, 2048, false);
    if (ab.graph == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss adaptor: ggml_new_graph_custom failed");
        return ab;
    }
    ggml_build_forward_expand(ab.graph, ab.out);
    if (ab.dumps.merge_out) {
        ggml_build_forward_expand(ab.graph, ab.dumps.merge_out);
    }
    return ab;
}

}  // namespace transcribe::moss
