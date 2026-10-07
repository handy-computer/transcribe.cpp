#include "decoder.h"

#include "ggml.h"
#include "transcribe-log.h"

namespace transcribe::arkasr {

namespace {

causal_lm::BlockView block_view(const ArkAsrDecBlock & b) {
    causal_lm::BlockView v{};
    v.norm_attn_w   = b.attn_norm_w;
    v.norm_ffn_w    = b.ffn_norm_w;
    v.attn_q_w      = b.q_w;
    v.attn_k_w      = b.k_w;
    v.attn_v_w      = b.v_w;
    v.attn_o_w      = b.o_w;
    v.attn_q_b      = b.q_b;
    v.attn_k_b      = b.k_b;
    v.attn_v_b      = b.v_b;
    v.ffn_gate_up_w = b.ffn_gate_up_w;
    v.ffn_down_w    = b.ffn_down_w;
    return v;
}

causal_lm::BlockParams block_params(const ArkAsrHParams & hp) {
    causal_lm::BlockParams p{};
    p.n_heads      = hp.dec_n_heads;
    p.n_kv_heads   = hp.dec_n_kv_heads;
    p.head_dim     = hp.dec_head_dim;
    p.max_position = hp.dec_max_position;
    p.rms_eps      = hp.dec_rms_eps;
    p.rope_theta   = hp.dec_rope_theta;
    return p;
}

ggml_tensor * final_logits(ggml_context * ctx, ggml_tensor * x, const ArkAsrWeights & w, const ArkAsrHParams & hp) {
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp.dec_rms_eps), w.dec_norm_w);
    x = ggml_mul_mat(ctx, w.dec_embed_w, x);
    return ggml_reshape_1d(ctx, x, hp.dec_vocab_size);
}

}  // namespace

PrefillBuild build_prefill_graph(ggml_context *                   ctx,
                                 const ArkAsrWeights &            weights,
                                 const ArkAsrHParams &            hp,
                                 transcribe::causal_lm::KvCache & kv_cache,
                                 int                              prompt_len,
                                 int                              audio_len,
                                 int                              audio_start,
                                 bool                             use_flash) {
    PrefillBuild out{};
    if (ctx == nullptr || prompt_len <= 0 || audio_len <= 0 || audio_start < 0 ||
        audio_start + audio_len > prompt_len || prompt_len > kv_cache.n_ctx) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: invalid prefill dimensions");
        return out;
    }

    out.input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prompt_len);
    out.audio     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.dec_hidden, audio_len);
    out.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prompt_len);
    out.mask      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, prompt_len, prompt_len);
    for (auto * t : { out.input_ids, out.audio, out.positions, out.mask }) {
        ggml_set_input(t);
    }
    ggml_set_name(out.input_ids, "ark.input_ids");
    ggml_set_name(out.audio, "ark.audio_embeddings");
    ggml_set_name(out.positions, "ark.positions");
    ggml_set_name(out.mask, "ark.mask");

    out.graph = ggml_new_graph_custom(ctx, 16384, false);
    if (out.graph == nullptr) {
        return {};
    }

    ggml_tensor * token_embeddings = ggml_get_rows(ctx, weights.dec_embed_w, out.input_ids);
    const size_t  elem             = ggml_element_size(token_embeddings);
    ggml_tensor * x                = out.audio;
    if (audio_start > 0) {
        auto * prefix = ggml_view_2d(ctx, token_embeddings, hp.dec_hidden, audio_start, elem * hp.dec_hidden, 0);
        x             = ggml_concat(ctx, ggml_cont(ctx, prefix), x, 1);
    }
    const int suffix_len = prompt_len - audio_start - audio_len;
    if (suffix_len > 0) {
        auto * suffix = ggml_view_2d(ctx, token_embeddings, hp.dec_hidden, suffix_len, elem * hp.dec_hidden,
                                     elem * hp.dec_hidden * static_cast<size_t>(audio_start + audio_len));
        x             = ggml_concat(ctx, x, ggml_cont(ctx, suffix), 1);
    }

    const auto params = block_params(hp);
    for (int layer = 0; layer < hp.dec_n_layers; ++layer) {
        causal_lm::BlockOpts opts{};
        opts.use_flash             = use_flash;
        opts.slice_last_before_ffn = layer == hp.dec_n_layers - 1;
        x = causal_lm::block_prefill(ctx, out.graph, x, block_view(weights.dec_blocks[layer]), params, kv_cache, layer,
                                     prompt_len, out.mask, out.positions, opts);
    }

    out.logits = final_logits(ctx, x, weights, hp);
    ggml_set_name(out.logits, "ark.prefill_logits");
    ggml_set_output(out.logits);
    ggml_build_forward_expand(out.graph, out.logits);
    return out;
}

StepBuild build_step_graph(ggml_context *                   ctx,
                           const ArkAsrWeights &            weights,
                           const ArkAsrHParams &            hp,
                           transcribe::causal_lm::KvCache & kv_cache,
                           int                              max_kv,
                           bool                             use_flash) {
    StepBuild out{};
    if (ctx == nullptr || max_kv <= 0 || max_kv > kv_cache.n_ctx) {
        return out;
    }

    out.input_id   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    out.position   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    out.kv_index   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    out.mask       = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, max_kv, 1);
    out.logit_mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.dec_vocab_size);
    for (auto * t : { out.input_id, out.position, out.kv_index, out.mask, out.logit_mask }) {
        ggml_set_input(t);
    }

    out.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (out.graph == nullptr) {
        return {};
    }

    ggml_tensor * x      = ggml_get_rows(ctx, weights.dec_embed_w, out.input_id);
    const auto    params = block_params(hp);
    for (int layer = 0; layer < hp.dec_n_layers; ++layer) {
        x = causal_lm::block_step(ctx, out.graph, x, block_view(weights.dec_blocks[layer]), params, kv_cache, layer,
                                  max_kv, out.mask, out.position, out.kv_index, use_flash);
    }
    auto * logits = final_logits(ctx, x, weights, hp);
    logits        = ggml_add(ctx, logits, out.logit_mask);
    out.token     = ggml_argmax(ctx, logits);
    ggml_set_output(out.token);
    ggml_build_forward_expand(out.graph, out.token);
    return out;
}

}  // namespace transcribe::arkasr
