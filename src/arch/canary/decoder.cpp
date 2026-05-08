// arch/canary/decoder.cpp - Canary autoregressive Transformer decoder.
//
// Modeled on src/arch/cohere/decoder.cpp. Differences from cohere:
//   - per-sublayer dump points: dec.layer.{i}.{self_attn,cross_attn,ffn}.out
//     so the C++ matches the reference dumper's hooks at layers
//     {0, n_layers/2, n_layers-1}
//   - LM head is UNTIED: applies w.head.weight + w.head.bias rather
//     than reusing the token embedding
//   - tensor names follow canary's GGUF (norm1/2/3, q/k/v/o, ffn.up/down)

#include "decoder.h"

#include "canary.h"
#include "weights.h"

#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace transcribe::canary {

namespace {

constexpr float kLayerNormEps = 1e-5f;

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr) ggml_set_name(t, name);
    return t;
}

ggml_tensor * layer_norm(ggml_context * ctx,
                         ggml_tensor *  x,
                         ggml_tensor *  gamma,
                         ggml_tensor *  beta)
{
    ggml_tensor * y = ggml_norm(ctx, x, kLayerNormEps);
    y = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) y = ggml_add(ctx, y, beta);
    return y;
}

// Self-attention with KV cache. Mirrors cohere's mha_self_cached.
ggml_tensor * mha_self_cached(
    ggml_context * ctx,
    ggml_cgraph *  gf,
    ggml_tensor *  x,
    CanaryKvCache & kv_cache,
    ggml_tensor *  mask,
    ggml_tensor *  q_w, ggml_tensor * q_b,
    ggml_tensor *  k_w, ggml_tensor * k_b,
    ggml_tensor *  v_w, ggml_tensor * v_b,
    ggml_tensor *  out_w, ggml_tensor * out_b,
    int            n_heads,
    int            hidden,
    int            il,
    int            n_past,
    int            n_tokens,
    bool           use_flash)
{
    const int head_dim = hidden / n_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int n_ctx    = kv_cache.n_ctx;
    const int n_kv     = n_past + n_tokens;

    ggml_tensor * Qcur = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) Qcur = ggml_add(ctx, Qcur, q_b);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, k_w, x);
    if (k_b != nullptr) Kcur = ggml_add(ctx, Kcur, k_b);

    ggml_tensor * Vcur = ggml_mul_mat(ctx, v_w, x);
    if (v_b != nullptr) Vcur = ggml_add(ctx, Vcur, v_b);

    ggml_tensor * Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

    {
        ggml_tensor * k_dst = ggml_view_1d(ctx, kv_cache.self_k,
            static_cast<int64_t>(n_tokens) * hidden,
            ggml_element_size(kv_cache.self_k) * static_cast<size_t>(
                static_cast<int64_t>(il) * n_ctx * hidden +
                static_cast<int64_t>(n_past) * hidden));

        ggml_tensor * v_dst = ggml_view_1d(ctx, kv_cache.self_v,
            static_cast<int64_t>(n_tokens) * hidden,
            ggml_element_size(kv_cache.self_v) * static_cast<size_t>(
                static_cast<int64_t>(il) * n_ctx * hidden +
                static_cast<int64_t>(n_past) * hidden));

        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v_dst));
    }

    ggml_tensor * K = ggml_view_3d(ctx, kv_cache.self_k,
        head_dim, n_kv, n_heads,
        ggml_element_size(kv_cache.self_k) * hidden,
        ggml_element_size(kv_cache.self_k) * head_dim,
        ggml_element_size(kv_cache.self_k) * static_cast<size_t>(
            static_cast<int64_t>(il) * n_ctx * hidden));

    ggml_tensor * V = ggml_view_3d(ctx, kv_cache.self_v,
        head_dim, n_kv, n_heads,
        ggml_element_size(kv_cache.self_v) * hidden,
        ggml_element_size(kv_cache.self_v) * head_dim,
        ggml_element_size(kv_cache.self_v) * static_cast<size_t>(
            static_cast<int64_t>(il) * n_ctx * hidden));

    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, v_t, kq_soft);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    }

    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

// Cross-attention reading from pre-populated KV cache.
ggml_tensor * mha_cross_cached(
    ggml_context * ctx,
    ggml_tensor *  x,
    CanaryKvCache & kv_cache,
    ggml_tensor *  q_w, ggml_tensor * q_b,
    ggml_tensor *  out_w, ggml_tensor * out_b,
    int            n_heads,
    int            hidden,
    int            il,
    int            T_enc,
    bool           use_flash)
{
    const int head_dim = hidden / n_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t n_tokens = x->ne[1];

    ggml_tensor * Qcur = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) Qcur = ggml_add(ctx, Qcur, q_b);

    ggml_tensor * Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

    ggml_tensor * K = ggml_view_3d(ctx, kv_cache.cross_k,
        head_dim, T_enc, n_heads,
        ggml_element_size(kv_cache.cross_k) * hidden,
        ggml_element_size(kv_cache.cross_k) * head_dim,
        ggml_element_size(kv_cache.cross_k) * static_cast<size_t>(
            static_cast<int64_t>(il) * T_enc * hidden));

    ggml_tensor * V = ggml_view_3d(ctx, kv_cache.cross_v,
        head_dim, T_enc, n_heads,
        ggml_element_size(kv_cache.cross_v) * hidden,
        ggml_element_size(kv_cache.cross_v) * head_dim,
        ggml_element_size(kv_cache.cross_v) * static_cast<size_t>(
            static_cast<int64_t>(il) * T_enc * hidden));

    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, v_t, kq_soft);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    }

    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

// Layers we attach per-sublayer dump points to. Mirrors the reference
// dumper's _block_indices convention: {0, n_layers/2, n_layers-1}, with
// dedup if the set collapses (n_layers <= 2).
bool is_dump_layer(int i, int n_layers) {
    if (n_layers <= 0) return false;
    if (i == 0 || i == n_layers - 1) return true;
    return i == n_layers / 2;
}

} // namespace

DecoderBuild build_cross_kv_graph(ggml_context *         ctx,
                                  const CanaryWeights &  w,
                                  const CanaryHParams &  hp,
                                  CanaryKvCache &        kv_cache,
                                  int                    T_enc)
{
    DecoderBuild db {};

    if (ctx == nullptr || T_enc <= 0) {
        std::fprintf(stderr,
                     "canary cross_kv: invalid arg (ctx=%p, T_enc=%d)\n",
                     static_cast<void *>(ctx), T_enc);
        return db;
    }

    const int64_t hidden = hp.dec_d_model;

    db.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T_enc);
    ggml_set_name(db.encoder_out_in, "dec.encoder_out");
    ggml_set_input(db.encoder_out_in);

    db.graph = ggml_new_graph_custom(ctx, 4096, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr, "canary cross_kv: ggml_new_graph_custom failed\n");
        return db;
    }

    for (int il = 0; il < hp.dec_n_layers; ++il) {
        const auto & blk = w.dec_blocks[il];

        ggml_tensor * Kcross = ggml_mul_mat(ctx, blk.cross_k_w, db.encoder_out_in);
        if (blk.cross_k_b != nullptr) Kcross = ggml_add(ctx, Kcross, blk.cross_k_b);

        ggml_tensor * Vcross = ggml_mul_mat(ctx, blk.cross_v_w, db.encoder_out_in);
        if (blk.cross_v_b != nullptr) Vcross = ggml_add(ctx, Vcross, blk.cross_v_b);

        ggml_tensor * k_dst = ggml_view_1d(ctx, kv_cache.cross_k,
            static_cast<int64_t>(T_enc) * hidden,
            ggml_element_size(kv_cache.cross_k) * static_cast<size_t>(
                static_cast<int64_t>(il) * T_enc * hidden));

        ggml_tensor * v_dst = ggml_view_1d(ctx, kv_cache.cross_v,
            static_cast<int64_t>(T_enc) * hidden,
            ggml_element_size(kv_cache.cross_v) * static_cast<size_t>(
                static_cast<int64_t>(il) * T_enc * hidden));

        ggml_build_forward_expand(db.graph, ggml_cpy(ctx, Kcross, k_dst));
        ggml_build_forward_expand(db.graph, ggml_cpy(ctx, Vcross, v_dst));
    }

    return db;
}

DecoderBuild build_decoder_graph_kv(ggml_context *         ctx,
                                    const CanaryWeights &  w,
                                    const CanaryHParams &  hp,
                                    CanaryKvCache &        kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    bool                   skip_log_softmax,
                                    bool                   use_flash)
{
    DecoderBuild db {};

    if (ctx == nullptr || n_tokens <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "canary decoder_kv: invalid arg "
                     "(ctx=%p, n_tokens=%d, T_enc=%d)\n",
                     static_cast<void *>(ctx), n_tokens, T_enc);
        return db;
    }

    const int64_t hidden  = hp.dec_d_model;
    const int     n_heads = hp.dec_n_heads;
    const int     n_kv    = n_past + n_tokens;

    db.token_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(db.token_ids_in, "dec.token_ids");
    ggml_set_input(db.token_ids_in);

    db.pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(db.pos_ids_in, "dec.pos_ids");
    ggml_set_input(db.pos_ids_in);

    ggml_tensor * tok_emb = ggml_get_rows(ctx, w.dec_embed.token_w, db.token_ids_in);
    tok_emb = named(tok_emb, "dec.token_emb");
    ggml_set_output(tok_emb);
    db.dumps.token_emb = tok_emb;

    ggml_tensor * pos_emb = ggml_get_rows(ctx, w.dec_embed.pos_enc, db.pos_ids_in);
    pos_emb = named(pos_emb, "dec.pos_emb");
    ggml_set_output(pos_emb);
    db.dumps.pos_emb = pos_emb;

    // Embedding layer norm: NeMo applies it AFTER tok+pos sum.
    ggml_tensor * x = ggml_add(ctx, tok_emb, pos_emb);
    x = layer_norm(ctx, x, w.dec_embed.norm_w, w.dec_embed.norm_b);
    x = named(x, "dec.embed_norm");
    ggml_set_output(x);
    db.dumps.embed_norm = x;

    // Causal mask for self-attention. Shape: [n_kv, n_tokens].
    ggml_tensor * causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
        ggml_set_name(causal_mask, "dec.causal_mask");
        ggml_set_input(causal_mask);
        causal_mask = ggml_cast(ctx, causal_mask, GGML_TYPE_F16);
    }

    db.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr, "canary decoder_kv: ggml_new_graph_custom failed\n");
        return db;
    }

    for (int i = 0; i < hp.dec_n_layers; ++i) {
        const auto & blk = w.dec_blocks[i];
        const bool dump_this = is_dump_layer(i, hp.dec_n_layers) && i < 64;

        // norm1 + self-attention. Reference dump hooks first_sub_layer
        // output BEFORE the residual add — see TransformerDecoderBlock
        // forward_preln in NeMo, which does
        //   self_attn_output = first_sub_layer(LN(x), ...)
        //   self_attn_output += residual
        // and the forward_hook captures self_attn_output BEFORE the +=.
        // So we dump self_out (the sublayer's contribution), not x.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x, blk.norm1_w, blk.norm1_b);
            ggml_tensor * self_out = mha_self_cached(
                ctx, db.graph, x_norm, kv_cache, causal_mask,
                blk.self_q_w, blk.self_q_b,
                blk.self_k_w, blk.self_k_b,
                blk.self_v_w, blk.self_v_b,
                blk.self_o_w, blk.self_o_b,
                n_heads, static_cast<int>(hidden),
                i, n_past, n_tokens,
                use_flash);
            if (dump_this) {
                char bname[64];
                std::snprintf(bname, sizeof(bname), "dec.layer.%d.self_attn.out", i);
                self_out = named(self_out, bname);
                ggml_set_output(self_out);
                db.dumps.sub_self[i] = self_out;
            }
            x = ggml_add(ctx, x, self_out);
        }

        // norm2 + cross-attention. Same pattern — second_sub_layer
        // returns the raw cross-attn output before NeMo adds the
        // residual.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x, blk.norm2_w, blk.norm2_b);
            ggml_tensor * cross_out = mha_cross_cached(
                ctx, x_norm, kv_cache,
                blk.cross_q_w, blk.cross_q_b,
                blk.cross_o_w, blk.cross_o_b,
                n_heads, static_cast<int>(hidden),
                i, T_enc,
                use_flash);
            if (dump_this) {
                char bname[64];
                std::snprintf(bname, sizeof(bname), "dec.layer.%d.cross_attn.out", i);
                cross_out = named(cross_out, bname);
                ggml_set_output(cross_out);
                db.dumps.sub_cross[i] = cross_out;
            }
            x = ggml_add(ctx, x, cross_out);
        }

        // norm3 + FFN. third_sub_layer (PositionWiseFF) returns the FFN
        // output before NeMo adds the residual.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x, blk.norm3_w, blk.norm3_b);
            ggml_tensor * ff = ggml_mul_mat(ctx, blk.ffn_up_w, x_norm);
            if (blk.ffn_up_b != nullptr) ff = ggml_add(ctx, ff, blk.ffn_up_b);
            if (hp.dec_activation == "relu") {
                ff = ggml_relu(ctx, ff);
            } else {
                ff = ggml_silu(ctx, ff);
            }
            ff = ggml_mul_mat(ctx, blk.ffn_down_w, ff);
            if (blk.ffn_down_b != nullptr) ff = ggml_add(ctx, ff, blk.ffn_down_b);
            if (dump_this) {
                char bname[64];
                std::snprintf(bname, sizeof(bname), "dec.layer.%d.ffn.out", i);
                ff = named(ff, bname);
                ggml_set_output(ff);
                db.dumps.sub_ffn[i] = ff;
            }
            x = ggml_add(ctx, x, ff);
        }
    }

    // Final LayerNorm (NeMo applies this after the last block).
    x = layer_norm(ctx, x, w.dec_final.norm_w, w.dec_final.norm_b);
    x = named(x, "dec.out_before_head");
    db.dumps.out_before_head = x;

    // Untied LM head: head.weight is its own tensor, not the token embedding.
    ggml_tensor * logits = ggml_mul_mat(ctx, w.head.weight, x);
    if (w.head.bias != nullptr) {
        logits = ggml_add(ctx, logits, w.head.bias);
    }

    logits = named(logits, "dec.logits_raw");
    ggml_set_output(logits);
    db.dumps.logits_raw = logits;

    if (!skip_log_softmax) {
        logits = ggml_log(ctx, ggml_soft_max(ctx, logits));
    }
    logits = named(logits, "dec.logits");
    db.dumps.logits = logits;

    db.out = logits;
    ggml_set_output(db.out);

    if (skip_log_softmax) {
        ggml_tensor * last_logits = logits;
        if (n_tokens > 1) {
            const int64_t vocab = logits->ne[0];
            const size_t  row_bytes = ggml_element_size(logits) * vocab;
            last_logits = ggml_view_2d(ctx, logits, vocab, /*n=*/1,
                                       row_bytes, row_bytes * (n_tokens - 1));
            last_logits = ggml_cont(ctx, last_logits);
        }
        db.argmax_out = ggml_argmax(ctx, last_logits);
        ggml_set_name(db.argmax_out, "dec.argmax");
        ggml_set_output(db.argmax_out);
        ggml_build_forward_expand(db.graph, db.argmax_out);
    } else {
        ggml_build_forward_expand(db.graph, db.out);
    }

    return db;
}

} // namespace transcribe::canary
