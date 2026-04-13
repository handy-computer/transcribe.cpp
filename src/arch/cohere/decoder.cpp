// arch/cohere/decoder.cpp - Cohere ASR Transformer decoder graph builder.
//
// Implements three graph builders:
//
// 1. build_decoder_graph() -- Original prompt-pass graph (no KV cache).
//    Kept for reference/validation. Processes all tokens at once with
//    a causal self-attention mask.
//
// 2. build_cross_kv_graph() -- Computes cross-attention K/V for all
//    decoder layers from the encoder output. Run once per utterance.
//    Writes into the cross-attention KV cache via ggml_cpy.
//
// 3. build_decoder_graph_kv() -- KV-cached decoder graph. Works for
//    both prompt pass (n_tokens > 1) and step pass (n_tokens = 1).
//    Reads/writes the self-attention KV cache; reads cross-attention
//    KV cache.
//
// The decoder forward pass:
//   1. Token embedding lookup + positional embedding lookup + LayerNorm
//   2. For each layer:
//      a. Pre-LN -> causal self-attention -> residual add
//      b. Pre-LN -> cross-attention (encoder as K/V) -> residual add
//      c. Pre-LN -> FFN (ReLU) -> residual add
//   3. Final LayerNorm
//   4. Linear projection (tied weight = token embedding^T) + bias
//   5. Log-softmax

#include "decoder.h"

#include "cohere.h"
#include "weights.h"

#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace transcribe::cohere {

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

// Standard multi-head attention (no relative position encoding).
// x:         [hidden, seq_len]
// context:   [hidden, T_ctx] (for cross-attn) or nullptr (for self-attn)
// mask:      additive mask in f16, or nullptr
// use_flash: true  -> ggml_flash_attn_ext (fused GPU kernel)
//            false -> manual mul_mat + soft_max_ext + mul_mat
// Returns:   [hidden, seq_len]
ggml_tensor * mha(ggml_context * ctx,
                  ggml_tensor *  x,
                  ggml_tensor *  context,
                  ggml_tensor *  mask,
                  ggml_tensor *  q_w, ggml_tensor * q_b,
                  ggml_tensor *  k_w, ggml_tensor * k_b,
                  ggml_tensor *  v_w, ggml_tensor * v_b,
                  ggml_tensor *  out_w, ggml_tensor * out_b,
                  int            n_heads,
                  int            hidden,
                  bool           use_flash)
{
    const int head_dim = hidden / n_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T_q  = x->ne[1];

    ggml_tensor * source = (context != nullptr) ? context : x;
    const int64_t T_k = source->ne[1];

    // Q, K, V projections with bias.
    ggml_tensor * q = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) q = ggml_add(ctx, q, q_b);
    ggml_tensor * k = ggml_mul_mat(ctx, k_w, source);
    if (k_b != nullptr) k = ggml_add(ctx, k, k_b);
    ggml_tensor * v = ggml_mul_mat(ctx, v_w, source);
    if (v_b != nullptr) v = ggml_add(ctx, v, v_b);

    // Split into heads: [head_dim, n_heads, T, 1] -> permute to
    // [head_dim, T, n_heads, 1] for flash_attn / standard.
    q = ggml_reshape_4d(ctx, q, head_dim, n_heads, T_q, 1);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);

    k = ggml_reshape_4d(ctx, k, head_dim, n_heads, T_k, 1);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);

    v = ggml_reshape_4d(ctx, v, head_dim, n_heads, T_k, 1);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * o;
    if (use_flash) {
        // Flash attention path: fused Q @ K^T + mask + softmax + @V.
        o = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        // [head_dim, n_heads, T_q, 1] -> [hidden, T_q]
        o = ggml_reshape_2d(ctx, o, hidden, T_q);
    } else {
        // Manual attention path: explicit matmuls + soft_max_ext.
        //   K is [head_dim, T_k, n_heads, 1] after the permute above.
        //   mul_mat(K, Q) computes Q^T @ K per head -> [T_k, T_q, n_heads, 1]
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        // soft_max_ext fuses scale + optional mask into softmax. Mask
        // shape must be [T_k, T_q] (or broadcastable); the caller
        // already prepared it as f16 at that shape.
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask,
                                                   scale, 0.0f);

        // V needs transposing for the value matmul.
        //   V is [head_dim, T_k, n_heads, 1]; we need
        //        [T_k, head_dim, n_heads, 1].
        ggml_tensor * v_t = ggml_cont(ctx,
                                       ggml_permute(ctx, v, 1, 0, 2, 3));

        // attn_weights @ V -> [head_dim, T_q, n_heads, 1]
        o = ggml_mul_mat(ctx, v_t, kq_soft);

        // Merge heads: permute [head_dim, T_q, n_heads] ->
        //                      [head_dim, n_heads, T_q], then reshape
        // to [hidden, T_q]. The permute makes heads contiguous for
        // the reshape, as in the encoder's manual path.
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, hidden, T_q);
    }

    // Output projection with bias.
    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

// Multi-head self-attention with KV cache.
//
// x:         [hidden, n_tokens] -- input for new tokens only
// kv_cache:  the pre-allocated KV cache
// mask:      [n_kv, n_tokens] additive mask (f16), or nullptr
// il:        layer index (for cache offset calculation)
// n_past:    number of tokens already in cache
// n_tokens:  number of new tokens being processed
// n_ctx:     max sequence length (cache size per layer)
//
// Writes new K/V into cache, reads full [0..n_past+n_tokens) for attn.
// Returns: [hidden, n_tokens]
ggml_tensor * mha_self_cached(
    ggml_context * ctx,
    ggml_cgraph *  gf,
    ggml_tensor *  x,
    CohereKvCache & kv_cache,
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

    // Q, K, V projections from new tokens.
    ggml_tensor * Qcur = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) Qcur = ggml_add(ctx, Qcur, q_b);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, k_w, x);
    if (k_b != nullptr) Kcur = ggml_add(ctx, Kcur, k_b);

    ggml_tensor * Vcur = ggml_mul_mat(ctx, v_w, x);
    if (v_b != nullptr) Vcur = ggml_add(ctx, Vcur, v_b);

    // Build Q for attention: [head_dim, n_tokens, n_heads].
    ggml_tensor * Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    // Q is now [head_dim, n_tokens, n_heads, 1]

    // Write new K/V into cache.
    // Cache layout (flash_attn mode): K and V are stored contiguously
    // per-layer as [hidden, n_ctx] (i.e., n_state elements per position).
    // Layer il occupies offset il*n_ctx*hidden in the flat 1D buffer.
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

    // Read K from cache: [head_dim, n_kv, n_heads].
    ggml_tensor * K = ggml_view_3d(ctx, kv_cache.self_k,
        head_dim, n_kv, n_heads,
        ggml_element_size(kv_cache.self_k) * hidden,       // nb1: stride between positions
        ggml_element_size(kv_cache.self_k) * head_dim,     // nb2: stride between heads
        ggml_element_size(kv_cache.self_k) * static_cast<size_t>(
            static_cast<int64_t>(il) * n_ctx * hidden));   // offset to layer

    // Read V from cache: [head_dim, n_kv, n_heads].
    ggml_tensor * V = ggml_view_3d(ctx, kv_cache.self_v,
        head_dim, n_kv, n_heads,
        ggml_element_size(kv_cache.self_v) * hidden,
        ggml_element_size(kv_cache.self_v) * head_dim,
        ggml_element_size(kv_cache.self_v) * static_cast<size_t>(
            static_cast<int64_t>(il) * n_ctx * hidden));

    // Attention. Flash path is fused; manual path is the usual
    // mul_mat + soft_max_ext + mul_mat triple. See `mha` above for
    // the shape argument.
    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    } else {
        // K is [head_dim, n_kv, n_heads, 1] from the cache view.
        // mul_mat(K, Q) -> [n_kv, n_tokens, n_heads, 1]
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask,
                                                   scale, 0.0f);

        // V is [head_dim, n_kv, n_heads, 1]; we need
        //      [n_kv, head_dim, n_heads, 1] for the value matmul.
        ggml_tensor * v_t = ggml_cont(ctx,
                                       ggml_permute(ctx, V, 1, 0, 2, 3));

        // attn_weights @ V -> [head_dim, n_tokens, n_heads, 1]
        o = ggml_mul_mat(ctx, v_t, kq_soft);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    }

    // Output projection.
    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

// Multi-head cross-attention reading from pre-computed KV cache.
//
// x:        [hidden, n_tokens] -- query input
// kv_cache: the cross-attention cache (already populated)
// il:       layer index
// T_enc:    number of encoder frames
//
// Returns: [hidden, n_tokens]
ggml_tensor * mha_cross_cached(
    ggml_context * ctx,
    ggml_tensor *  x,
    CohereKvCache & kv_cache,
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

    // Only Q is computed from current input. K/V come from cache.
    ggml_tensor * Qcur = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) Qcur = ggml_add(ctx, Qcur, q_b);

    ggml_tensor * Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

    // Read K from cross cache: [head_dim, T_enc, n_heads].
    ggml_tensor * K = ggml_view_3d(ctx, kv_cache.cross_k,
        head_dim, T_enc, n_heads,
        ggml_element_size(kv_cache.cross_k) * hidden,
        ggml_element_size(kv_cache.cross_k) * head_dim,
        ggml_element_size(kv_cache.cross_k) * static_cast<size_t>(
            static_cast<int64_t>(il) * T_enc * hidden));

    // Read V from cross cache: [head_dim, T_enc, n_heads].
    ggml_tensor * V = ggml_view_3d(ctx, kv_cache.cross_v,
        head_dim, T_enc, n_heads,
        ggml_element_size(kv_cache.cross_v) * hidden,
        ggml_element_size(kv_cache.cross_v) * head_dim,
        ggml_element_size(kv_cache.cross_v) * static_cast<size_t>(
            static_cast<int64_t>(il) * T_enc * hidden));

    // No mask for cross-attention. Flash path is the fused kernel;
    // manual path is the usual triple.
    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q, K, V, nullptr,
                                scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    } else {
        // K is [head_dim, T_enc, n_heads, 1] from the cache view.
        // mul_mat(K, Q) -> [T_enc, n_tokens, n_heads, 1]
        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, nullptr,
                                                   scale, 0.0f);

        // V is [head_dim, T_enc, n_heads, 1]; transpose for the
        // value matmul.
        ggml_tensor * v_t = ggml_cont(ctx,
                                       ggml_permute(ctx, V, 1, 0, 2, 3));

        // attn_weights @ V -> [head_dim, n_tokens, n_heads, 1]
        o = ggml_mul_mat(ctx, v_t, kq_soft);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, hidden, n_tokens);
    }

    // Output projection.
    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

} // namespace

// ---------------------------------------------------------------------------
// Original non-cached prompt-pass graph (kept for reference/validation).
// ---------------------------------------------------------------------------

DecoderBuild build_decoder_graph(ggml_context *         ctx,
                                 const CohereWeights &  w,
                                 const CohereHParams &  hp,
                                 int                    seq_len,
                                 int                    T_enc,
                                 bool                   use_flash)
{
    DecoderBuild db {};

    if (ctx == nullptr || seq_len <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "cohere decoder: invalid arg "
                     "(ctx=%p, seq_len=%d, T_enc=%d)\n",
                     static_cast<void *>(ctx), seq_len, T_enc);
        return db;
    }

    const int64_t hidden   = hp.dec_hidden;
    const int     n_heads  = hp.dec_n_heads;

    // Input tensors.
    db.token_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(db.token_ids_in, "dec.token_ids");
    ggml_set_input(db.token_ids_in);

    db.pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(db.pos_ids_in, "dec.pos_ids");
    ggml_set_input(db.pos_ids_in);

    db.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T_enc);
    ggml_set_name(db.encoder_out_in, "dec.encoder_out");
    ggml_set_input(db.encoder_out_in);

    // Token embedding: lookup from [hidden, vocab_size] table.
    // ggml_get_rows takes a table of ne=[hidden, vocab_size] and indices
    // of ne=[seq_len], producing [hidden, seq_len].
    ggml_tensor * tok_emb = ggml_get_rows(ctx, w.dec_embed.token_w,
                                          db.token_ids_in);
    tok_emb = named(tok_emb, "dec.token_emb");
    ggml_set_output(tok_emb);
    db.dumps.token_emb = tok_emb;

    // Positional embedding: lookup from [hidden, max_seq] table.
    ggml_tensor * pos_emb = ggml_get_rows(ctx, w.dec_embed.pos_enc,
                                          db.pos_ids_in);
    pos_emb = named(pos_emb, "dec.pos_emb");
    ggml_set_output(pos_emb);
    db.dumps.pos_emb = pos_emb;

    // x = LayerNorm(tok_emb + pos_emb)
    ggml_tensor * x = ggml_add(ctx, tok_emb, pos_emb);
    x = layer_norm(ctx, x, w.dec_embed.norm_w, w.dec_embed.norm_b);
    x = named(x, "dec.embed_norm");
    ggml_set_output(x);
    db.dumps.embed_norm = x;

    // Build causal self-attention mask. Shape [seq_len, seq_len] f16.
    // mask[i][j] = 0 if j <= i, else -inf.
    // For flash_attn, mask shape should be [T_k, T_q, 1, 1].
    ggml_tensor * causal_mask = nullptr;
    if (seq_len > 1) {
        // ggml_new_tensor_2d(ctx, F32, T_k=seq_len, T_q=seq_len)
        // filled with causal pattern, then cast to F16.
        causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                         seq_len, seq_len);
        ggml_set_name(causal_mask, "dec.causal_mask");
        ggml_set_input(causal_mask);
        // Will be filled by the driver with proper causal mask values.
        // Cast to f16 for flash_attn.
        causal_mask = ggml_cast(ctx, causal_mask, GGML_TYPE_F16);
    }

    // Decoder layers.
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        const auto & blk = w.dec_blocks[i];

        // Pre-LN self-attention with causal mask.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              blk.norm_self_w, blk.norm_self_b);
            ggml_tensor * self_out = mha(ctx, x_norm, nullptr, causal_mask,
                                         blk.self_q_w, blk.self_q_b,
                                         blk.self_k_w, blk.self_k_b,
                                         blk.self_v_w, blk.self_v_b,
                                         blk.self_out_w, blk.self_out_b,
                                         n_heads, static_cast<int>(hidden),
                                         use_flash);
            x = ggml_add(ctx, x, self_out);
        }

        // Pre-LN cross-attention (encoder output as K/V, no mask).
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              blk.norm_cross_w, blk.norm_cross_b);
            ggml_tensor * cross_out = mha(ctx, x_norm, db.encoder_out_in, nullptr,
                                          blk.cross_q_w, blk.cross_q_b,
                                          blk.cross_k_w, blk.cross_k_b,
                                          blk.cross_v_w, blk.cross_v_b,
                                          blk.cross_out_w, blk.cross_out_b,
                                          n_heads, static_cast<int>(hidden),
                                          use_flash);
            x = ggml_add(ctx, x, cross_out);
        }

        // Pre-LN FFN with ReLU (or silu).
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              blk.norm_ff_w, blk.norm_ff_b);
            ggml_tensor * ff = ggml_mul_mat(ctx, blk.ff_in_w, x_norm);
            if (blk.ff_in_b != nullptr) ff = ggml_add(ctx, ff, blk.ff_in_b);
            if (hp.dec_activation == "relu") {
                ff = ggml_relu(ctx, ff);
            } else {
                ff = ggml_silu(ctx, ff);
            }
            ff = ggml_mul_mat(ctx, blk.ff_out_w, ff);
            if (blk.ff_out_b != nullptr) ff = ggml_add(ctx, ff, blk.ff_out_b);
            x = ggml_add(ctx, x, ff);
        }
    }

    // Final LayerNorm.
    x = layer_norm(ctx, x, w.dec_final.norm_w, w.dec_final.norm_b);
    x = named(x, "dec.out_before_head");
    db.dumps.out_before_head = x;

    // Head: tied weight (transpose of token embedding) + bias + log-softmax.
    // ggml_mul_mat(W, x) where W=[hidden, vocab_size] and x=[hidden, seq_len]
    // gives [vocab_size, seq_len]. The token embedding is [hidden, vocab_size]
    // which is exactly what we need.
    ggml_tensor * logits = ggml_mul_mat(ctx, w.dec_embed.token_w, x);
    if (w.head.bias != nullptr) {
        logits = ggml_add(ctx, logits, w.head.bias);
    }

    if (hp.head_log_softmax) {
        logits = ggml_log(ctx, ggml_soft_max(ctx, logits));
    }
    logits = named(logits, "dec.logits");
    db.dumps.logits = logits;

    db.out = logits;
    ggml_set_output(db.out);

    // Build the forward graph.
    db.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr,
                     "cohere decoder: ggml_new_graph_custom failed\n");
        return db;
    }
    ggml_build_forward_expand(db.graph, db.out);

    return db;
}

// ---------------------------------------------------------------------------
// Cross-attention KV computation graph.
// ---------------------------------------------------------------------------

DecoderBuild build_cross_kv_graph(ggml_context *         ctx,
                                  const CohereWeights &  w,
                                  const CohereHParams &  hp,
                                  CohereKvCache &        kv_cache,
                                  int                    T_enc)
{
    DecoderBuild db {};

    if (ctx == nullptr || T_enc <= 0) {
        std::fprintf(stderr,
                     "cohere cross_kv: invalid arg (ctx=%p, T_enc=%d)\n",
                     static_cast<void *>(ctx), T_enc);
        return db;
    }

    const int64_t hidden  = hp.dec_hidden;

    // Input: encoder output [hidden, T_enc].
    db.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, T_enc);
    ggml_set_name(db.encoder_out_in, "dec.encoder_out");
    ggml_set_input(db.encoder_out_in);

    db.graph = ggml_new_graph_custom(ctx, 4096, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr,
                     "cohere cross_kv: ggml_new_graph_custom failed\n");
        return db;
    }

    // For each decoder layer, compute K and V from encoder output,
    // then write into the cross-attention cache.
    for (int il = 0; il < hp.dec_n_layers; ++il) {
        const auto & blk = w.dec_blocks[il];

        // K = cross_k_w * encoder_out + cross_k_b
        ggml_tensor * Kcross = ggml_mul_mat(ctx, blk.cross_k_w,
                                            db.encoder_out_in);
        if (blk.cross_k_b != nullptr) {
            Kcross = ggml_add(ctx, Kcross, blk.cross_k_b);
        }

        // V = cross_v_w * encoder_out + cross_v_b
        ggml_tensor * Vcross = ggml_mul_mat(ctx, blk.cross_v_w,
                                            db.encoder_out_in);
        if (blk.cross_v_b != nullptr) {
            Vcross = ggml_add(ctx, Vcross, blk.cross_v_b);
        }

        // Write into cross cache. Layout: flat [hidden * T_enc] per layer.
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

// ---------------------------------------------------------------------------
// KV-cached decoder graph (prompt + step passes).
// ---------------------------------------------------------------------------

DecoderBuild build_decoder_graph_kv(ggml_context *         ctx,
                                    const CohereWeights &  w,
                                    const CohereHParams &  hp,
                                    CohereKvCache &        kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    bool                   skip_log_softmax,
                                    bool                   use_flash)
{
    DecoderBuild db {};

    if (ctx == nullptr || n_tokens <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "cohere decoder_kv: invalid arg "
                     "(ctx=%p, n_tokens=%d, T_enc=%d)\n",
                     static_cast<void *>(ctx), n_tokens, T_enc);
        return db;
    }

    const int64_t hidden  = hp.dec_hidden;
    const int     n_heads = hp.dec_n_heads;
    const int     n_kv    = n_past + n_tokens;

    // Input tensors (only for new tokens).
    db.token_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(db.token_ids_in, "dec.token_ids");
    ggml_set_input(db.token_ids_in);

    db.pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(db.pos_ids_in, "dec.pos_ids");
    ggml_set_input(db.pos_ids_in);

    // Token embedding.
    ggml_tensor * tok_emb = ggml_get_rows(ctx, w.dec_embed.token_w,
                                          db.token_ids_in);
    tok_emb = named(tok_emb, "dec.token_emb");
    ggml_set_output(tok_emb);
    db.dumps.token_emb = tok_emb;

    // Positional embedding.
    ggml_tensor * pos_emb = ggml_get_rows(ctx, w.dec_embed.pos_enc,
                                          db.pos_ids_in);
    pos_emb = named(pos_emb, "dec.pos_emb");
    ggml_set_output(pos_emb);
    db.dumps.pos_emb = pos_emb;

    // x = LayerNorm(tok_emb + pos_emb)
    ggml_tensor * x = ggml_add(ctx, tok_emb, pos_emb);
    x = layer_norm(ctx, x, w.dec_embed.norm_w, w.dec_embed.norm_b);
    x = named(x, "dec.embed_norm");
    ggml_set_output(x);
    db.dumps.embed_norm = x;

    // Causal mask for self-attention.
    // Shape: [n_kv, n_tokens] -- for each new query position, which
    // cache positions are visible.
    //
    // For the prompt pass (n_past=0, n_tokens>1):
    //   mask[k, q] = 0 if k <= q, else -inf
    //
    // For step pass (n_tokens=1):
    //   All n_kv positions are visible (mask is all zeros), so we can
    //   skip the mask entirely. But for correctness with n_tokens>1 in
    //   prompt pass, we always create the mask.
    ggml_tensor * causal_mask = nullptr;
    if (n_tokens > 1) {
        // Need a mask of shape [n_kv, n_tokens].
        causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
        ggml_set_name(causal_mask, "dec.causal_mask");
        ggml_set_input(causal_mask);
        causal_mask = ggml_cast(ctx, causal_mask, GGML_TYPE_F16);
    }

    // Pre-create the graph (we need it for ggml_build_forward_expand in
    // the self-attention cache writes).
    db.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr,
                     "cohere decoder_kv: ggml_new_graph_custom failed\n");
        return db;
    }

    // Decoder layers.
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        const auto & blk = w.dec_blocks[i];

        // Pre-LN self-attention with KV cache.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              blk.norm_self_w, blk.norm_self_b);
            ggml_tensor * self_out = mha_self_cached(
                ctx, db.graph, x_norm, kv_cache, causal_mask,
                blk.self_q_w, blk.self_q_b,
                blk.self_k_w, blk.self_k_b,
                blk.self_v_w, blk.self_v_b,
                blk.self_out_w, blk.self_out_b,
                n_heads, static_cast<int>(hidden),
                i, n_past, n_tokens,
                use_flash);
            x = ggml_add(ctx, x, self_out);
        }

        // Pre-LN cross-attention reading from cached K/V.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              blk.norm_cross_w, blk.norm_cross_b);
            ggml_tensor * cross_out = mha_cross_cached(
                ctx, x_norm, kv_cache,
                blk.cross_q_w, blk.cross_q_b,
                blk.cross_out_w, blk.cross_out_b,
                n_heads, static_cast<int>(hidden),
                i, T_enc,
                use_flash);
            x = ggml_add(ctx, x, cross_out);
        }

        // Pre-LN FFN.
        {
            ggml_tensor * x_norm = layer_norm(ctx, x,
                                              blk.norm_ff_w, blk.norm_ff_b);
            ggml_tensor * ff = ggml_mul_mat(ctx, blk.ff_in_w, x_norm);
            if (blk.ff_in_b != nullptr) ff = ggml_add(ctx, ff, blk.ff_in_b);
            if (hp.dec_activation == "relu") {
                ff = ggml_relu(ctx, ff);
            } else {
                ff = ggml_silu(ctx, ff);
            }
            ff = ggml_mul_mat(ctx, blk.ff_out_w, ff);
            if (blk.ff_out_b != nullptr) ff = ggml_add(ctx, ff, blk.ff_out_b);
            x = ggml_add(ctx, x, ff);
        }

        // Per-layer output dump.
        {
            char bname[32];
            std::snprintf(bname, sizeof(bname), "dec.block.%d.out", i);
            x = named(x, bname);
            ggml_set_output(x);
            db.dumps.block_out[i] = x;
        }
    }

    // Final LayerNorm.
    x = layer_norm(ctx, x, w.dec_final.norm_w, w.dec_final.norm_b);
    x = named(x, "dec.out_before_head");
    db.dumps.out_before_head = x;

    // Head: tied weight + bias + optional log-softmax.
    ggml_tensor * logits = ggml_mul_mat(ctx, w.dec_embed.token_w, x);
    if (w.head.bias != nullptr) {
        logits = ggml_add(ctx, logits, w.head.bias);
    }

    // Pre-softmax logits for numerically stable comparison.
    logits = named(logits, "dec.logits_raw");
    ggml_set_output(logits);
    db.dumps.logits_raw = logits;

    if (hp.head_log_softmax && !skip_log_softmax) {
        logits = ggml_log(ctx, ggml_soft_max(ctx, logits));
    }
    logits = named(logits, "dec.logits");
    db.dumps.logits = logits;

    db.out = logits;
    ggml_set_output(db.out);

    // When skipping log_softmax (prompt and step passes), add argmax on
    // GPU so we only need to read back a single int32 instead of the
    // full [vocab_size x n_tokens] logits tensor. For n_tokens > 1 we
    // only need the argmax of the last position (the next token to
    // generate), so take a view of the last row before argmax.
    if (skip_log_softmax) {
        ggml_tensor * last_logits = logits;
        if (n_tokens > 1) {
            const int64_t vocab = logits->ne[0];
            const size_t  row_bytes =
                ggml_element_size(logits) * vocab;
            last_logits = ggml_view_2d(ctx, logits,
                                       vocab, /*n=*/1,
                                       row_bytes,
                                       row_bytes * (n_tokens - 1));
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

} // namespace transcribe::cohere
