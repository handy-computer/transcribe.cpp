// arch/whisper/decoder.cpp - Whisper decoder graph builder (prefill).
//
// Mirrors the transformers reference (WhisperDecoderLayer):
//
//   x = embed_tokens(ids) + embed_positions(weight[0:seq_len])
//   for layer in decoder.layers:
//       x = x + self_attn(LN_self(x), causal_mask)
//       x = x + cross_attn(LN_cross(x), encoder_hidden)
//       x = x + ffn(LN_ffn(x))
//   x = layer_norm(x)
//   logits_raw = proj_out(x)              # tied: W = embed_tokens.weight
//   logits     = log_softmax(logits_raw)
//
// Whisper has NO post-embed LayerNorm (cohere does — that is the
// notable shape difference vs. cohere/decoder.cpp).
//
// q/v/out carry bias on both self and cross attention; k_proj has NO
// bias. The shared `mha_decoder` helper takes nullable bias slots and
// skips the add when null, matching encoder.cpp's `mha_encoder`.

#include "decoder.h"

#include "encoder.h"      // for transcribe::conformer namespace alias context
#include "weights.h"
#include "whisper.h"

#include "conformer/conformer.h"
#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace transcribe::whisper {

namespace {

namespace conf = transcribe::conformer;
using conf::named;
using conf::layer_norm;

// Standard multi-head attention used by both self- and cross-attention
// in the decoder. context==nullptr selects self-attention (Q/K/V from x);
// non-null context selects cross-attention (Q from x, K/V from context).
//
// x:        [d_model, T_q]
// context:  [d_model, T_k]   or nullptr for self-attention (T_k = T_q)
// mask:     additive mask (f16) shape [T_k, T_q] or nullptr
// Returns:  [d_model, T_q]
ggml_tensor * mha_decoder(ggml_context * ctx,
                          ggml_tensor *  x,
                          ggml_tensor *  context,
                          ggml_tensor *  mask,
                          ggml_tensor *  q_w, ggml_tensor * q_b,
                          ggml_tensor *  k_w,
                          ggml_tensor *  v_w, ggml_tensor * v_b,
                          ggml_tensor *  out_w, ggml_tensor * out_b,
                          int            n_heads,
                          int            d_model,
                          bool           use_flash)
{
    const int     head_dim = d_model / n_heads;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T_q      = x->ne[1];

    ggml_tensor * source = (context != nullptr) ? context : x;
    const int64_t T_k    = source->ne[1];

    ggml_tensor * q = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) q = ggml_add(ctx, q, q_b);

    ggml_tensor * k = ggml_mul_mat(ctx, k_w, source);
    // k has no bias.

    ggml_tensor * v = ggml_mul_mat(ctx, v_w, source);
    if (v_b != nullptr) v = ggml_add(ctx, v, v_b);

    // [d_model, T] -> [head_dim, n_heads, T] -> permute to
    // [head_dim, T, n_heads] for both flash and manual paths.
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T_q);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);

    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T_k);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);

    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, T_k);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor * o;
    if (use_flash) {
        ggml_tensor * q_c = ggml_cont(ctx, q);
        ggml_tensor * k_c = ggml_cont(ctx, k);
        ggml_tensor * v_c = ggml_cont(ctx, v);
        o = ggml_flash_attn_ext(ctx, q_c, k_c, v_c, mask, scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, d_model, T_q);
    } else {
        ggml_tensor * kq      = ggml_mul_mat(ctx, ggml_cont(ctx, k),
                                             ggml_cont(ctx, q));
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

        ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, v_t, kq_soft);

        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, d_model, T_q);
    }

    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

// FFN: fc1 -> GELU -> fc2. Both projections carry bias.
ggml_tensor * ffn(ggml_context * ctx,
                  ggml_tensor *  x,
                  ggml_tensor *  fc1_w, ggml_tensor * fc1_b,
                  ggml_tensor *  fc2_w, ggml_tensor * fc2_b) {
    ggml_tensor * h = ggml_mul_mat(ctx, fc1_w, x);
    if (fc1_b != nullptr) h = ggml_add(ctx, h, fc1_b);
    // HF Whisper uses nn.functional.gelu (exact erf form); ggml_gelu
    // is the tanh approximation and drifts ~1e-4 per element.
    h = ggml_gelu_erf(ctx, h);
    ggml_tensor * o = ggml_mul_mat(ctx, fc2_w, h);
    if (fc2_b != nullptr) o = ggml_add(ctx, o, fc2_b);
    return o;
}

ggml_tensor * build_block(ggml_context *          ctx,
                          ggml_tensor *           x,
                          ggml_tensor *           encoder_hidden,
                          ggml_tensor *           causal_mask,
                          const WhisperDecBlock & b,
                          int                     n_heads,
                          int                     d_model,
                          bool                    use_flash)
{
    // Self-attention (causal).
    {
        ggml_tensor * y = layer_norm(ctx, x, b.norm_self_w, b.norm_self_b);
        y = mha_decoder(ctx, y, /*context=*/nullptr, causal_mask,
                        b.self_q_w, b.self_q_b,
                        b.self_k_w,
                        b.self_v_w, b.self_v_b,
                        b.self_out_w, b.self_out_b,
                        n_heads, d_model, use_flash);
        x = ggml_add(ctx, x, y);
    }
    // Cross-attention (no mask; queries decoder state against encoder).
    {
        ggml_tensor * y = layer_norm(ctx, x, b.norm_cross_w, b.norm_cross_b);
        y = mha_decoder(ctx, y, encoder_hidden, /*mask=*/nullptr,
                        b.cross_q_w, b.cross_q_b,
                        b.cross_k_w,
                        b.cross_v_w, b.cross_v_b,
                        b.cross_out_w, b.cross_out_b,
                        n_heads, d_model, use_flash);
        x = ggml_add(ctx, x, y);
    }
    // FFN.
    {
        ggml_tensor * y = layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
        y = ffn(ctx, y,
                b.ffn_fc1_w, b.ffn_fc1_b,
                b.ffn_fc2_w, b.ffn_fc2_b);
        x = ggml_add(ctx, x, y);
    }
    return x;
}

// ---------------------------------------------------------------------------
// KV-cached helpers
// ---------------------------------------------------------------------------
//
// Cache layout (flat 1D tensors, one slot per layer, per role):
//
//   self_k / self_v   : [hidden * n_layer * n_ctx]
//       layer il offset:  il * n_ctx * hidden
//       position  n  slot within layer:
//                         il * n_ctx * hidden + n * hidden
//
//   cross_k / cross_v : [hidden * n_layer * T_enc]
//       layer il offset:  il * T_enc * hidden
//
// Shape follows cohere's layout exactly so the ggml patterns transfer
// unchanged; whisper's only quirk is k_proj has no bias on both self
// and cross (cohere has k_b which may also be null).

// Self-attention reading/writing the self KV cache.
//
// x:         [d_model, n_tokens] input for new tokens only
// mask:      [n_kv, n_tokens] f16 additive mask, or nullptr
// il:        layer index (for cache offset)
// n_past:    number of tokens already in cache
// n_tokens:  number of new tokens being processed
//
// Writes new K/V into the cache at [il, n_past..n_past+n_tokens),
// then reads the full [il, 0..n_past+n_tokens) window for attention.
// Returns [d_model, n_tokens].
ggml_tensor * mha_self_cached(ggml_context *   ctx,
                              ggml_cgraph *    gf,
                              ggml_tensor *    x,
                              WhisperKvCache & kv_cache,
                              ggml_tensor *    mask,
                              ggml_tensor *    q_w, ggml_tensor * q_b,
                              ggml_tensor *    k_w,
                              ggml_tensor *    v_w, ggml_tensor * v_b,
                              ggml_tensor *    out_w, ggml_tensor * out_b,
                              int              n_heads,
                              int              d_model,
                              int              il,
                              int              n_past,
                              int              n_tokens,
                              bool             use_flash)
{
    const int head_dim = d_model / n_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int n_ctx    = kv_cache.n_ctx;
    const int n_kv     = n_past + n_tokens;

    // Q / K / V projections for the *new* tokens only.
    ggml_tensor * Qcur = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) Qcur = ggml_add(ctx, Qcur, q_b);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, k_w, x);
    // k has no bias on whisper.

    ggml_tensor * Vcur = ggml_mul_mat(ctx, v_w, x);
    if (v_b != nullptr) Vcur = ggml_add(ctx, Vcur, v_b);

    // Q in [head_dim, n_tokens, n_heads] for flash / manual paths.
    ggml_tensor * Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

    // Write new K/V into the cache. Layout is [hidden, n_ctx] per layer,
    // stored as a flat 1D buffer. Layer il occupies
    //   offset = il * n_ctx * hidden  (in elements).
    {
        const size_t k_elem = ggml_element_size(kv_cache.self_k);
        const size_t v_elem = ggml_element_size(kv_cache.self_v);

        ggml_tensor * k_dst = ggml_view_1d(
            ctx, kv_cache.self_k,
            static_cast<int64_t>(n_tokens) * d_model,
            k_elem * static_cast<size_t>(
                static_cast<int64_t>(il) * n_ctx * d_model +
                static_cast<int64_t>(n_past) * d_model));

        ggml_tensor * v_dst = ggml_view_1d(
            ctx, kv_cache.self_v,
            static_cast<int64_t>(n_tokens) * d_model,
            v_elem * static_cast<size_t>(
                static_cast<int64_t>(il) * n_ctx * d_model +
                static_cast<int64_t>(n_past) * d_model));

        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v_dst));
    }

    // Read K / V as [head_dim, n_kv, n_heads] for attention.
    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    ggml_tensor * K = ggml_view_3d(
        ctx, kv_cache.self_k,
        head_dim, n_kv, n_heads,
        k_elem * d_model,        // nb1: stride between positions (one full d_model row)
        k_elem * head_dim,       // nb2: stride between heads inside a position
        k_elem * static_cast<size_t>(
            static_cast<int64_t>(il) * n_ctx * d_model));

    const size_t v_elem = ggml_element_size(kv_cache.self_v);
    ggml_tensor * V = ggml_view_3d(
        ctx, kv_cache.self_v,
        head_dim, n_kv, n_heads,
        v_elem * d_model,
        v_elem * head_dim,
        v_elem * static_cast<size_t>(
            static_cast<int64_t>(il) * n_ctx * d_model));

    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
    } else {
        ggml_tensor * kq      = ggml_mul_mat(ctx, K, Q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * v_t     = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, v_t, kq_soft);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
    }

    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

// Cross-attention reading the pre-populated cross KV cache.
//
// x:        [d_model, n_tokens] query input
// il:       layer index
// T_enc:    number of encoder frames
//
// Returns: [d_model, n_tokens].
ggml_tensor * mha_cross_cached(ggml_context *   ctx,
                               ggml_tensor *    x,
                               WhisperKvCache & kv_cache,
                               ggml_tensor *    q_w, ggml_tensor * q_b,
                               ggml_tensor *    out_w, ggml_tensor * out_b,
                               int              n_heads,
                               int              d_model,
                               int              il,
                               int              T_enc,
                               bool             use_flash)
{
    const int head_dim = d_model / n_heads;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t n_tokens = x->ne[1];

    // Q from current input; K / V read from the cross cache.
    ggml_tensor * Qcur = ggml_mul_mat(ctx, q_w, x);
    if (q_b != nullptr) Qcur = ggml_add(ctx, Qcur, q_b);

    ggml_tensor * Q = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_tokens);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

    const size_t k_elem = ggml_element_size(kv_cache.cross_k);
    ggml_tensor * K = ggml_view_3d(
        ctx, kv_cache.cross_k,
        head_dim, T_enc, n_heads,
        k_elem * d_model,
        k_elem * head_dim,
        k_elem * static_cast<size_t>(
            static_cast<int64_t>(il) * T_enc * d_model));

    const size_t v_elem = ggml_element_size(kv_cache.cross_v);
    ggml_tensor * V = ggml_view_3d(
        ctx, kv_cache.cross_v,
        head_dim, T_enc, n_heads,
        v_elem * d_model,
        v_elem * head_dim,
        v_elem * static_cast<size_t>(
            static_cast<int64_t>(il) * T_enc * d_model));

    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
    } else {
        ggml_tensor * kq      = ggml_mul_mat(ctx, K, Q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * v_t     = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, v_t, kq_soft);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
        o = ggml_reshape_2d(ctx, o, d_model, n_tokens);
    }

    o = ggml_mul_mat(ctx, out_w, o);
    if (out_b != nullptr) o = ggml_add(ctx, o, out_b);
    return o;
}

} // namespace

DecoderBuild build_decoder_prefill_graph(ggml_context *         ctx,
                                         const WhisperWeights & w,
                                         const WhisperHParams & hp,
                                         int                    seq_len,
                                         int                    T_enc,
                                         bool                   use_flash)
{
    DecoderBuild db {};

    if (ctx == nullptr || seq_len <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "whisper decoder: invalid arg "
                     "(ctx=%p, seq_len=%d, T_enc=%d)\n",
                     static_cast<void *>(ctx), seq_len, T_enc);
        return db;
    }
    if (seq_len > hp.dec_max_target_positions) {
        std::fprintf(stderr,
                     "whisper decoder: seq_len=%d exceeds "
                     "max_target_positions=%d\n",
                     seq_len, hp.dec_max_target_positions);
        return db;
    }

    const int d_model = hp.dec_d_model;
    const int n_heads = hp.dec_n_heads;
    const int vocab   = hp.dec_vocab_size;

    // ---- Inputs ------------------------------------------------------
    db.token_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    named(db.token_ids_in, "dec.token_ids");
    ggml_set_input(db.token_ids_in);

    db.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_enc);
    named(db.encoder_out_in, "dec.encoder_out");
    ggml_set_input(db.encoder_out_in);

    // Causal mask declared as F32 input; cast to F16 inside the graph
    // because flash_attn_ext / soft_max_ext both want f16.
    ggml_tensor * causal_mask = nullptr;
    if (seq_len > 1) {
        db.causal_mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                               seq_len, seq_len);
        named(db.causal_mask_in, "dec.causal_mask");
        ggml_set_input(db.causal_mask_in);
        causal_mask = ggml_cast(ctx, db.causal_mask_in, GGML_TYPE_F16);
    }

    // ---- Token embedding --------------------------------------------
    // token_embd_w is [d_model, vocab]; ggml_get_rows with i32 indices
    // [seq_len] yields [d_model, seq_len].
    ggml_tensor * tok_emb = ggml_get_rows(ctx, w.dec_top.token_embd_w,
                                          db.token_ids_in);
    named(tok_emb, "dec.token_emb");
    transcribe::debug::mark_tensor_for_dump(tok_emb);
    db.dumps.token_emb = tok_emb;

    // ---- Positional embedding ---------------------------------------
    // pos_emb_w is [d_model, max_target_positions]. The transformers
    // reference takes weight[past:past+seq_len]; for prefill past=0 so
    // a 2D view of the leading seq_len rows is exactly the same data
    // (and avoids needing an i32 input for position ids).
    ggml_tensor * pos_emb = w.dec_top.pos_emb_w;
    if (seq_len != hp.dec_max_target_positions) {
        pos_emb = ggml_view_2d(ctx, w.dec_top.pos_emb_w,
                               d_model, seq_len,
                               w.dec_top.pos_emb_w->nb[1], 0);
    }
    named(pos_emb, "dec.pos_emb");
    transcribe::debug::mark_tensor_for_dump(pos_emb);
    db.dumps.pos_emb = pos_emb;

    // ---- Embedding sum (no LayerNorm) -------------------------------
    ggml_tensor * x = ggml_add(ctx, tok_emb, pos_emb);
    named(x, "dec.embed_sum");
    transcribe::debug::mark_tensor_for_dump(x);
    db.dumps.embed_sum = x;

    // ---- Decoder blocks ---------------------------------------------
    const int n_blocks = static_cast<int>(w.dec_blocks.size());
    db.dumps.block_outs.reserve(static_cast<size_t>(n_blocks));
    for (int i = 0; i < n_blocks; ++i) {
        x = build_block(ctx, x, db.encoder_out_in, causal_mask,
                        w.dec_blocks[i], n_heads, d_model, use_flash);

        char bname[64];
        std::snprintf(bname, sizeof(bname), "dec.block.%d.out", i);
        named(x, bname);
        transcribe::debug::mark_tensor_for_dump(x);
        db.dumps.block_outs.push_back(x);
    }

    // ---- Final LayerNorm --------------------------------------------
    x = layer_norm(ctx, x, w.dec_top.final_norm_w, w.dec_top.final_norm_b);
    named(x, "dec.out_before_head");
    transcribe::debug::mark_tensor_for_dump(x);
    db.dumps.out_before_head = x;

    // ---- Logits head (tied weight, no bias) -------------------------
    // mul_mat(W, x) where W=[d_model, vocab] and x=[d_model, seq_len]
    // gives [vocab, seq_len].
    ggml_tensor * logits_raw = ggml_mul_mat(ctx, w.dec_top.token_embd_w, x);
    named(logits_raw, "dec.logits_raw");
    transcribe::debug::mark_tensor_for_dump(logits_raw);
    db.dumps.logits_raw = logits_raw;

    ggml_tensor * logits = ggml_log(ctx, ggml_soft_max(ctx, logits_raw));
    named(logits, "dec.logits");
    transcribe::debug::mark_tensor_for_dump(logits);
    db.dumps.logits = logits;

    db.out = logits;
    ggml_set_output(db.out);

    // 8192 leaves headroom for 4-block whisper-tiny (and for larger
    // variants up to whisper-large-v3 at 32 blocks; each block adds
    // ~20 ops with self+cross+ffn).
    db.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr,
                     "whisper decoder: ggml_new_graph_custom failed\n");
        return db;
    }
    ggml_build_forward_expand(db.graph, db.out);

    // Suppress unused-variable warning when vocab is only used in the
    // bounds check above (kept for documentation in the catalog).
    (void)vocab;

    return db;
}

// ---------------------------------------------------------------------------
// KV-cached path: cross-KV precompute + prompt/step decoder graph.
// ---------------------------------------------------------------------------

DecoderBuild build_cross_kv_graph(ggml_context *         ctx,
                                  const WhisperWeights & w,
                                  const WhisperHParams & hp,
                                  WhisperKvCache &       kv_cache,
                                  int                    T_enc)
{
    DecoderBuild db {};

    if (ctx == nullptr || T_enc <= 0) {
        std::fprintf(stderr,
                     "whisper cross_kv: invalid arg (ctx=%p, T_enc=%d)\n",
                     static_cast<void *>(ctx), T_enc);
        return db;
    }

    const int d_model = hp.dec_d_model;

    // Input: encoder output [d_model, T_enc]. Caller uploads via
    // ggml_backend_tensor_set after alloc_graph.
    db.encoder_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_enc);
    named(db.encoder_out_in, "dec.encoder_out");
    ggml_set_input(db.encoder_out_in);

    db.graph = ggml_new_graph_custom(ctx, 4096, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr,
                     "whisper cross_kv: ggml_new_graph_custom failed\n");
        return db;
    }

    const int n_layers = static_cast<int>(w.dec_blocks.size());
    for (int il = 0; il < n_layers; ++il) {
        const auto & blk = w.dec_blocks[il];

        // Whisper: cross k has no bias; cross v has a bias.
        ggml_tensor * Kcross = ggml_mul_mat(ctx, blk.cross_k_w,
                                            db.encoder_out_in);
        ggml_tensor * Vcross = ggml_mul_mat(ctx, blk.cross_v_w,
                                            db.encoder_out_in);
        if (blk.cross_v_b != nullptr) {
            Vcross = ggml_add(ctx, Vcross, blk.cross_v_b);
        }

        // Write the [d_model, T_enc] projections into the flat cache
        // slot for this layer. Offset in elements = il * T_enc * d_model.
        const size_t k_elem = ggml_element_size(kv_cache.cross_k);
        const size_t v_elem = ggml_element_size(kv_cache.cross_v);

        ggml_tensor * k_dst = ggml_view_1d(
            ctx, kv_cache.cross_k,
            static_cast<int64_t>(T_enc) * d_model,
            k_elem * static_cast<size_t>(
                static_cast<int64_t>(il) * T_enc * d_model));

        ggml_tensor * v_dst = ggml_view_1d(
            ctx, kv_cache.cross_v,
            static_cast<int64_t>(T_enc) * d_model,
            v_elem * static_cast<size_t>(
                static_cast<int64_t>(il) * T_enc * d_model));

        ggml_build_forward_expand(db.graph, ggml_cpy(ctx, Kcross, k_dst));
        ggml_build_forward_expand(db.graph, ggml_cpy(ctx, Vcross, v_dst));
    }

    return db;
}

DecoderBuild build_decoder_graph_kv(ggml_context *         ctx,
                                    const WhisperWeights & w,
                                    const WhisperHParams & hp,
                                    WhisperKvCache &       kv_cache,
                                    int                    n_tokens,
                                    int                    n_past,
                                    int                    T_enc,
                                    bool                   skip_log_softmax,
                                    bool                   use_flash)
{
    DecoderBuild db {};

    if (ctx == nullptr || n_tokens <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "whisper decoder_kv: invalid arg "
                     "(ctx=%p, n_tokens=%d, T_enc=%d)\n",
                     static_cast<void *>(ctx), n_tokens, T_enc);
        return db;
    }
    const int n_kv = n_past + n_tokens;
    if (n_kv > kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "whisper decoder_kv: n_kv=%d exceeds n_ctx=%d\n",
                     n_kv, kv_cache.n_ctx);
        return db;
    }

    const int d_model = hp.dec_d_model;
    const int n_heads = hp.dec_n_heads;

    // ---- Inputs -----------------------------------------------------
    db.token_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    named(db.token_ids_in, "dec.token_ids");
    ggml_set_input(db.token_ids_in);

    // Position ids as an input so the step loop can upload a single
    // int32 (the absolute position) without rebuilding the graph
    // structure. For prompt pass (n_past=0, n_tokens=4) the caller
    // uploads [0, 1, 2, 3].
    ggml_tensor * pos_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    named(pos_ids_in, "dec.pos_ids");
    ggml_set_input(pos_ids_in);

    // Causal mask only needed when n_tokens > 1 (prompt pass). For
    // step pass, the current token can attend to everything in the
    // cache, so mask is all zeros and we skip it.
    //
    // The mask-null branch is only sound when exactly one new token is
    // being added to the cache — otherwise later tokens in the batch
    // would see each other and leak future context. Enforce the
    // invariant explicitly: if a beam-search caller ever passes
    // n_tokens > 1 with n_past > 0 it must also build a mask.
    ggml_tensor * causal_mask = nullptr;
    if (n_tokens > 1) {
        db.causal_mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                               n_kv, n_tokens);
        named(db.causal_mask_in, "dec.causal_mask");
        ggml_set_input(db.causal_mask_in);
        causal_mask = ggml_cast(ctx, db.causal_mask_in, GGML_TYPE_F16);
    } else {
        GGML_ASSERT(n_tokens == 1 &&
                    "decoder_kv mask-null branch requires single-token step");
    }

    // ---- Token embedding -------------------------------------------
    ggml_tensor * tok_emb = ggml_get_rows(ctx, w.dec_top.token_embd_w,
                                          db.token_ids_in);
    named(tok_emb, "dec.token_emb");
    if (n_past == 0) {
        transcribe::debug::mark_tensor_for_dump(tok_emb);
        db.dumps.token_emb = tok_emb;
    }

    // ---- Positional embedding --------------------------------------
    ggml_tensor * pos_emb = ggml_get_rows(ctx, w.dec_top.pos_emb_w,
                                          pos_ids_in);
    named(pos_emb, "dec.pos_emb");
    if (n_past == 0) {
        transcribe::debug::mark_tensor_for_dump(pos_emb);
        db.dumps.pos_emb = pos_emb;
    }

    // ---- Embed sum (no LayerNorm) ----------------------------------
    ggml_tensor * x = ggml_add(ctx, tok_emb, pos_emb);
    named(x, "dec.embed_sum");
    if (n_past == 0) {
        transcribe::debug::mark_tensor_for_dump(x);
        db.dumps.embed_sum = x;
    }

    // Pre-create the graph so mha_self_cached can use
    // ggml_build_forward_expand to wire the cache writes into it.
    db.graph = ggml_new_graph_custom(ctx, 8192, false);
    if (db.graph == nullptr) {
        std::fprintf(stderr,
                     "whisper decoder_kv: ggml_new_graph_custom failed\n");
        return db;
    }

    // ---- Decoder blocks --------------------------------------------
    const int n_blocks = static_cast<int>(w.dec_blocks.size());
    db.dumps.block_outs.reserve(static_cast<size_t>(n_blocks));
    for (int i = 0; i < n_blocks; ++i) {
        const auto & b = w.dec_blocks[i];

        // Self-attention (pre-LN, causal via mask when prompt pass).
        {
            ggml_tensor * y = layer_norm(ctx, x, b.norm_self_w, b.norm_self_b);
            y = mha_self_cached(
                ctx, db.graph, y, kv_cache, causal_mask,
                b.self_q_w, b.self_q_b,
                b.self_k_w,
                b.self_v_w, b.self_v_b,
                b.self_out_w, b.self_out_b,
                n_heads, d_model,
                i, n_past, n_tokens, use_flash);
            x = ggml_add(ctx, x, y);
        }
        // Cross-attention (pre-LN, reads cross cache).
        {
            ggml_tensor * y = layer_norm(ctx, x, b.norm_cross_w, b.norm_cross_b);
            y = mha_cross_cached(
                ctx, y, kv_cache,
                b.cross_q_w, b.cross_q_b,
                b.cross_out_w, b.cross_out_b,
                n_heads, d_model, i, T_enc, use_flash);
            x = ggml_add(ctx, x, y);
        }
        // FFN.
        {
            ggml_tensor * y = layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
            y = ffn(ctx, y,
                    b.ffn_fc1_w, b.ffn_fc1_b,
                    b.ffn_fc2_w, b.ffn_fc2_b);
            x = ggml_add(ctx, x, y);
        }

        char bname[64];
        std::snprintf(bname, sizeof(bname), "dec.block.%d.out", i);
        named(x, bname);
        if (n_past == 0) {
            transcribe::debug::mark_tensor_for_dump(x);
            db.dumps.block_outs.push_back(x);
        }
    }

    // ---- Final LayerNorm -------------------------------------------
    x = layer_norm(ctx, x, w.dec_top.final_norm_w, w.dec_top.final_norm_b);
    named(x, "dec.out_before_head");
    if (n_past == 0) {
        transcribe::debug::mark_tensor_for_dump(x);
        db.dumps.out_before_head = x;
    }

    // ---- Logits head (tied weight, no bias) ------------------------
    ggml_tensor * logits_raw = ggml_mul_mat(ctx, w.dec_top.token_embd_w, x);
    named(logits_raw, "dec.logits_raw");
    if (n_past == 0) {
        transcribe::debug::mark_tensor_for_dump(logits_raw);
        db.dumps.logits_raw = logits_raw;
    }

    ggml_tensor * logits = logits_raw;
    if (!skip_log_softmax) {
        logits = ggml_log(ctx, ggml_soft_max(ctx, logits_raw));
        named(logits, "dec.logits");
        if (n_past == 0) {
            transcribe::debug::mark_tensor_for_dump(logits);
            db.dumps.logits = logits;
        }
    }

    db.out = logits;
    ggml_set_output(db.out);
    ggml_build_forward_expand(db.graph, db.out);

    return db;
}

} // namespace transcribe::whisper
