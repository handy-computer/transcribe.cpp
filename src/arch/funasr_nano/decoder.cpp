// arch/funasr_nano/decoder.cpp - Qwen3-0.6B LM prefill / step graphs.
//
// Pruned fork of arch/qwen3_asr/decoder.cpp. The Qwen3 block math is
// identical (pre-LN RMSNorm, GQA, per-head Q/K-RMSNorm, NeoX RoPE,
// SwiGLU). Audio injection differs in surface only — funasr_nano splices
// the first `fake_token_len` rows of the adaptor output at the
// `fbank_beg` position via a three-way concat
// [prefix | audio | suffix]. Token-id stream contains zero placeholders
// at the audio positions (the embedding lookup is computed for the
// dump but the rows are then overridden by the concat).

#include "decoder.h"

#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace transcribe::funasr_nano {

namespace {

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr && name != nullptr) ggml_set_name(t, name);
    return t;
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x,
                       ggml_tensor * weight, float eps)
{
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

} // namespace

PrefillBuild build_prefill_graph(ggml_context *             ctx,
                                 const FunAsrNanoWeights &  weights,
                                 const FunAsrNanoHParams &  hp,
                                 KvCache &                  kv_cache,
                                 int                        T_prompt,
                                 int                        T_audio,
                                 int                        prefix_len,
                                 int                        suffix_len,
                                 bool                       use_flash,
                                 bool                       slice_last)
{
    PrefillBuild pb {};
    pb.T_prompt   = T_prompt;
    pb.T_audio    = T_audio;
    pb.prefix_len = prefix_len;
    pb.suffix_len = suffix_len;

    if (ctx == nullptr || T_prompt <= 0 || T_audio < 0) {
        std::fprintf(stderr,
                     "funasr_nano decoder: invalid arg (T_prompt=%d, T_audio=%d)\n",
                     T_prompt, T_audio);
        return pb;
    }
    if (prefix_len < 0 || suffix_len < 0 ||
        prefix_len + T_audio + suffix_len != T_prompt)
    {
        std::fprintf(stderr,
                     "funasr_nano decoder: prefix_len(%d) + T_audio(%d) + suffix_len(%d) != T_prompt(%d)\n",
                     prefix_len, T_audio, suffix_len, T_prompt);
        return pb;
    }
    if (kv_cache.self_k == nullptr || kv_cache.self_v == nullptr) {
        std::fprintf(stderr, "funasr_nano decoder: kv_cache not initialized\n");
        return pb;
    }
    if (T_prompt > kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "funasr_nano decoder: T_prompt=%d exceeds kv_cache.n_ctx=%d\n",
                     T_prompt, kv_cache.n_ctx);
        return pb;
    }

    const int64_t hidden     = hp.dec_hidden;
    const int64_t n_heads    = hp.dec_n_heads;
    const int64_t n_kv_heads = hp.dec_n_kv_heads;
    const int64_t n_groups   = n_heads / n_kv_heads;
    const int64_t head_dim   = hp.dec_head_dim;
    const int64_t q_dim      = n_heads    * head_dim;
    const int64_t kv_dim     = n_kv_heads * head_dim;
    const int64_t vocab      = hp.dec_vocab_size;
    const int     n_layer    = hp.dec_n_layers;
    const int     n_ctx      = kv_cache.n_ctx;
    const float   rms_eps    = hp.dec_rms_norm_eps;
    const float   rope_theta = hp.dec_rope_theta;
    const float   scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    // ---------- Inputs ----------
    pb.input_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
    named(pb.input_ids_in, "dec.input_ids");
    ggml_set_input(pb.input_ids_in);

    if (T_audio > 0) {
        pb.audio_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T_audio);
        named(pb.audio_in, "dec.audio_in");
        ggml_set_input(pb.audio_in);
    }

    pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
    named(pb.positions_in, "dec.positions");
    ggml_set_input(pb.positions_in);

    pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_prompt, T_prompt);
    named(pb.mask_in, "dec.attn_mask");
    ggml_set_input(pb.mask_in);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
    if (gf == nullptr) {
        std::fprintf(stderr,
                     "funasr_nano decoder: ggml_new_graph_custom failed\n");
        return pb;
    }
    pb.graph = gf;

    // ---------- Token embedding ----------
    ggml_tensor * token_emb_all = ggml_get_rows(
        ctx, weights.dec_embed.token_w, pb.input_ids_in);
    named(token_emb_all, "dec.token_emb");
    pb.dumps.token_emb = token_emb_all;
    transcribe::debug::mark_tensor_for_dump(token_emb_all);

    // ---------- Audio injection via concat ----------
    const size_t emb_elem = ggml_element_size(token_emb_all);
    ggml_tensor * x_prefix = nullptr;
    if (prefix_len > 0) {
        x_prefix = ggml_view_2d(ctx, token_emb_all,
                                hidden, prefix_len,
                                emb_elem * hidden, /*off=*/0);
        x_prefix = ggml_cont(ctx, x_prefix);
    }

    ggml_tensor * x_suffix = nullptr;
    if (suffix_len > 0) {
        x_suffix = ggml_view_2d(ctx, token_emb_all,
                                hidden, suffix_len,
                                emb_elem * hidden,
                                emb_elem * hidden *
                                    static_cast<size_t>(prefix_len + T_audio));
        x_suffix = ggml_cont(ctx, x_suffix);
    }

    ggml_tensor * x;
    if (T_audio > 0) {
        x = pb.audio_in;
        if (x_prefix != nullptr) x = ggml_concat(ctx, x_prefix, x, /*dim=*/1);
        if (x_suffix != nullptr) x = ggml_concat(ctx, x, x_suffix, /*dim=*/1);
    } else {
        x = token_emb_all;
    }
    named(x, "dec.inputs_embeds.with_audio");
    pb.dumps.audio_injected = x;
    transcribe::debug::mark_tensor_for_dump(x);

    for (int il = 0; il < n_layer; ++il) {
        const auto & w = weights.dec_blocks[il];

        // ---- Attention sub-layer ----
        ggml_tensor * x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

        ggml_tensor * Q = ggml_mul_mat(ctx, w.attn_q_w, x_norm);
        ggml_tensor * K = ggml_mul_mat(ctx, w.attn_k_w, x_norm);
        ggml_tensor * V = ggml_mul_mat(ctx, w.attn_v_w, x_norm);

        Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads,    T_prompt, 1);
        K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_prompt, 1);
        V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_prompt, 1);

        Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
        K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);

        Q = ggml_rope_ext(ctx, Q, pb.positions_in, nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, pb.positions_in, nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // KV cache write.
        {
            const size_t layer_off = static_cast<size_t>(il) * n_ctx * kv_dim;
            const size_t n_elem    = static_cast<size_t>(T_prompt) * kv_dim;
            ggml_tensor * k_dst = ggml_view_1d(ctx, kv_cache.self_k,
                                               n_elem, k_elem * layer_off);
            ggml_tensor * v_dst = ggml_view_1d(ctx, kv_cache.self_v,
                                               n_elem, v_elem * layer_off);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
        }

        const size_t layer_off_bytes =
            k_elem * static_cast<size_t>(il) * n_ctx * kv_dim;
        ggml_tensor * K_att = ggml_view_3d(
            ctx, kv_cache.self_k,
            head_dim, T_prompt, n_kv_heads,
            k_elem * kv_dim, k_elem * head_dim,
            layer_off_bytes);
        ggml_tensor * V_att = ggml_view_3d(
            ctx, kv_cache.self_v,
            head_dim, T_prompt, n_kv_heads,
            v_elem * kv_dim, v_elem * head_dim,
            v_elem * static_cast<size_t>(il) * n_ctx * kv_dim);

        ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

        ggml_tensor * o;
        if (use_flash) {
            o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, pb.mask_in,
                                    scale_attn, /*max_bias=*/0.0f,
                                    /*logit_softcap=*/0.0f);
            o = ggml_reshape_2d(ctx, o, q_dim, T_prompt);
        } else {
            ggml_tensor * K_att_c = ggml_cont(ctx, K_att);
            ggml_tensor * V_att_c = ggml_cont(ctx, V_att);
            ggml_tensor * K_4d = ggml_reshape_4d(
                ctx, K_att_c, head_dim, T_prompt, 1, n_kv_heads);
            ggml_tensor * V_4d = ggml_reshape_4d(
                ctx, V_att_c, head_dim, T_prompt, 1, n_kv_heads);
            ggml_tensor * K_rep_template = ggml_new_tensor_4d(
                ctx, K_att->type, head_dim, T_prompt, n_groups, n_kv_heads);
            ggml_tensor * V_rep_template = ggml_new_tensor_4d(
                ctx, V_att->type, head_dim, T_prompt, n_groups, n_kv_heads);
            ggml_tensor * K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
            ggml_tensor * V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
            ggml_tensor * K_full = ggml_reshape_3d(ctx, K_rep, head_dim, T_prompt, n_heads);
            ggml_tensor * V_full = ggml_reshape_3d(ctx, V_rep, head_dim, T_prompt, n_heads);

            ggml_tensor * kq = ggml_mul_mat(ctx, K_full, Q_att);
            ggml_tensor * kq_soft = ggml_soft_max_ext(
                ctx, kq, pb.mask_in, scale_attn, /*max_bias=*/0.0f);
            ggml_tensor * V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
            o = ggml_mul_mat(ctx, V_t, kq_soft);
            o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
            o = ggml_reshape_2d(ctx, o, q_dim, T_prompt);
        }

        o = ggml_mul_mat(ctx, w.attn_o_w, o);
        x = ggml_add(ctx, x, o);

        if (slice_last && il == n_layer - 1) {
            const size_t elem = ggml_element_size(x);
            x = ggml_view_2d(ctx, x, hidden, 1,
                             elem * hidden,
                             elem * hidden * static_cast<size_t>(T_prompt - 1));
            x = ggml_cont(ctx, x);
        }

        ggml_tensor * ff_norm  = rms_norm(ctx, x, w.norm_ffn_w, rms_eps);
        ggml_tensor * gate_up  = ggml_mul_mat(ctx, w.ffn_gate_up_w, ff_norm);
        ggml_tensor * ff       = ggml_swiglu(ctx, gate_up);
        ff = ggml_mul_mat(ctx, w.ffn_down_w, ff);

        x = ggml_add(ctx, x, ff);

        if (il == 0) {
            named(x, "dec.block.0.out");
            pb.dumps.block_0_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        } else if (il == n_layer - 1) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "dec.block.%d.out", il);
            named(x, nm);
            pb.dumps.block_last_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
    }

    x = rms_norm(ctx, x, weights.dec_final.norm_w, rms_eps);
    named(x, "dec.out_before_head");
    pb.dumps.out_before_head = x;
    transcribe::debug::mark_tensor_for_dump(x);

    ggml_tensor * last_x;
    if (slice_last) {
        last_x = x;
    } else {
        last_x = ggml_view_2d(
            ctx, x, hidden, 1,
            ggml_element_size(x) * hidden,
            ggml_element_size(x) * hidden * static_cast<size_t>(T_prompt - 1));
        last_x = ggml_cont(ctx, last_x);
    }

    ggml_tensor * logits = ggml_mul_mat(ctx, weights.dec_embed.token_w, last_x);
    logits = ggml_reshape_1d(ctx, logits, vocab);
    named(logits, "dec.logits_raw.prefill");
    pb.dumps.logits_raw = logits;
    transcribe::debug::mark_tensor_for_dump(logits);

    pb.out = logits;
    ggml_set_output(pb.out);

    ggml_build_forward_expand(gf, pb.out);
    if (pb.dumps.token_emb)       ggml_build_forward_expand(gf, pb.dumps.token_emb);
    if (pb.dumps.audio_injected)  ggml_build_forward_expand(gf, pb.dumps.audio_injected);
    if (pb.dumps.block_0_out)     ggml_build_forward_expand(gf, pb.dumps.block_0_out);
    if (pb.dumps.block_last_out)  ggml_build_forward_expand(gf, pb.dumps.block_last_out);
    if (pb.dumps.out_before_head) ggml_build_forward_expand(gf, pb.dumps.out_before_head);

    return pb;
}

StepBuild build_step_graph(ggml_context *             ctx,
                           const FunAsrNanoWeights &  weights,
                           const FunAsrNanoHParams &  hp,
                           KvCache &                  kv_cache,
                           int                        max_n_kv,
                           bool                       use_flash)
{
    StepBuild sb {};
    sb.max_n_kv = max_n_kv;

    if (ctx == nullptr || max_n_kv <= 0) {
        std::fprintf(stderr,
                     "funasr_nano step: invalid arg (max_n_kv=%d)\n", max_n_kv);
        return sb;
    }
    if (kv_cache.self_k == nullptr) {
        std::fprintf(stderr, "funasr_nano step: kv_cache not initialized\n");
        return sb;
    }
    if (max_n_kv > kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "funasr_nano step: max_n_kv=%d exceeds kv_cache.n_ctx=%d\n",
                     max_n_kv, kv_cache.n_ctx);
        return sb;
    }

    const int64_t n_heads    = hp.dec_n_heads;
    const int64_t n_kv_heads = hp.dec_n_kv_heads;
    const int64_t n_groups   = n_heads / n_kv_heads;
    const int64_t head_dim   = hp.dec_head_dim;
    const int64_t q_dim      = n_heads    * head_dim;
    const int64_t kv_dim     = n_kv_heads * head_dim;
    const int64_t vocab      = hp.dec_vocab_size;
    const int     n_layer    = hp.dec_n_layers;
    const int     n_ctx      = kv_cache.n_ctx;
    const float   rms_eps    = hp.dec_rms_norm_eps;
    const float   rope_theta = hp.dec_rope_theta;
    const float   scale_attn = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    sb.input_id_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(sb.input_id_in, "step.input_id");
    ggml_set_input(sb.input_id_in);

    sb.position_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(sb.position_in, "step.position");
    ggml_set_input(sb.position_in);

    sb.kv_idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_set_name(sb.kv_idx_in, "step.kv_idx");
    ggml_set_input(sb.kv_idx_in);

    sb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_n_kv, 1);
    ggml_set_name(sb.mask_in, "step.mask");
    ggml_set_input(sb.mask_in);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (gf == nullptr) {
        std::fprintf(stderr,
                     "funasr_nano step: ggml_new_graph_custom failed\n");
        return sb;
    }
    sb.graph = gf;

    ggml_tensor * x = ggml_get_rows(ctx, weights.dec_embed.token_w,
                                    sb.input_id_in);

    for (int il = 0; il < n_layer; ++il) {
        const auto & w = weights.dec_blocks[il];

        ggml_tensor * x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

        ggml_tensor * Q = ggml_mul_mat(ctx, w.attn_q_w, x_norm);
        ggml_tensor * K = ggml_mul_mat(ctx, w.attn_k_w, x_norm);
        ggml_tensor * V = ggml_mul_mat(ctx, w.attn_v_w, x_norm);

        Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads,    1, 1);
        K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, 1, 1);
        V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, 1, 1);

        Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
        K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);

        Q = ggml_rope_ext(ctx, Q, sb.position_in, nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, sb.position_in, nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        {
            const size_t layer_off_k =
                k_elem * static_cast<size_t>(il) * n_ctx * kv_dim;
            const size_t layer_off_v =
                v_elem * static_cast<size_t>(il) * n_ctx * kv_dim;

            ggml_tensor * k_layer = ggml_view_2d(ctx, kv_cache.self_k,
                                                 kv_dim, n_ctx,
                                                 k_elem * kv_dim, layer_off_k);
            ggml_tensor * v_layer = ggml_view_2d(ctx, kv_cache.self_v,
                                                 kv_dim, n_ctx,
                                                 v_elem * kv_dim, layer_off_v);
            ggml_tensor * K_row = ggml_reshape_2d(ctx, K, kv_dim, 1);
            ggml_tensor * V_row = ggml_reshape_2d(ctx, V, kv_dim, 1);

            ggml_build_forward_expand(
                gf, ggml_set_rows(ctx, k_layer, K_row, sb.kv_idx_in));
            ggml_build_forward_expand(
                gf, ggml_set_rows(ctx, v_layer, V_row, sb.kv_idx_in));
        }

        const size_t layer_off_bytes =
            k_elem * static_cast<size_t>(il) * n_ctx * kv_dim;
        ggml_tensor * K_att = ggml_view_3d(
            ctx, kv_cache.self_k,
            head_dim, max_n_kv, n_kv_heads,
            k_elem * kv_dim, k_elem * head_dim,
            layer_off_bytes);
        ggml_tensor * V_att = ggml_view_3d(
            ctx, kv_cache.self_v,
            head_dim, max_n_kv, n_kv_heads,
            v_elem * kv_dim, v_elem * head_dim,
            v_elem * static_cast<size_t>(il) * n_ctx * kv_dim);

        ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

        ggml_tensor * o;
        if (use_flash) {
            o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, sb.mask_in,
                                    scale_attn, 0.0f, 0.0f);
            o = ggml_reshape_2d(ctx, o, q_dim, 1);
        } else {
            ggml_tensor * K_att_c = ggml_cont(ctx, K_att);
            ggml_tensor * V_att_c = ggml_cont(ctx, V_att);
            ggml_tensor * K_4d = ggml_reshape_4d(ctx, K_att_c, head_dim, max_n_kv, 1, n_kv_heads);
            ggml_tensor * V_4d = ggml_reshape_4d(ctx, V_att_c, head_dim, max_n_kv, 1, n_kv_heads);
            ggml_tensor * K_rep_template = ggml_new_tensor_4d(
                ctx, K_att->type, head_dim, max_n_kv, n_groups, n_kv_heads);
            ggml_tensor * V_rep_template = ggml_new_tensor_4d(
                ctx, V_att->type, head_dim, max_n_kv, n_groups, n_kv_heads);
            ggml_tensor * K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
            ggml_tensor * V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
            ggml_tensor * K_full = ggml_reshape_3d(ctx, K_rep, head_dim, max_n_kv, n_heads);
            ggml_tensor * V_full = ggml_reshape_3d(ctx, V_rep, head_dim, max_n_kv, n_heads);

            ggml_tensor * kq = ggml_mul_mat(ctx, K_full, Q_att);
            ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, sb.mask_in, scale_attn, 0.0f);
            ggml_tensor * V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
            o = ggml_mul_mat(ctx, V_t, kq_soft);
            o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
            o = ggml_reshape_2d(ctx, o, q_dim, 1);
        }
        o = ggml_mul_mat(ctx, w.attn_o_w, o);
        x = ggml_add(ctx, x, o);

        ggml_tensor * ff_norm = rms_norm(ctx, x, w.norm_ffn_w, rms_eps);
        ggml_tensor * gate_up = ggml_mul_mat(ctx, w.ffn_gate_up_w, ff_norm);
        ggml_tensor * ff      = ggml_swiglu(ctx, gate_up);
        ff = ggml_mul_mat(ctx, w.ffn_down_w, ff);

        x = ggml_add(ctx, x, ff);
    }

    x = rms_norm(ctx, x, weights.dec_final.norm_w, rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, weights.dec_embed.token_w, x);
    logits = ggml_reshape_1d(ctx, logits, vocab);
    ggml_set_name(logits, "step.logits");

    ggml_tensor * amax = ggml_argmax(ctx, logits);
    ggml_set_name(amax, "step.argmax");

    sb.out = amax;
    sb.logits = logits;
    ggml_set_output(sb.out);
    ggml_set_output(sb.logits);
    ggml_build_forward_expand(gf, sb.out);
    ggml_build_forward_expand(gf, logits);
    return sb;
}

} // namespace transcribe::funasr_nano
