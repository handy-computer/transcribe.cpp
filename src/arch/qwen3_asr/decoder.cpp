// arch/qwen3_asr/decoder.cpp - Qwen3-ASR LM prefill/step graph builders.
//
// Reference: Qwen3ASRThinkerTextModel in modeling_qwen3_asr.py.
//
// Per-block flow (pre-LN):
//   residual = x
//   x = rms_norm(x) * norm_attn_w
//   Q = mul_mat(q_w, x) -> reshape [D, H,   T, 1]
//   K = mul_mat(k_w, x) -> reshape [D, Hkv, T, 1]
//   V = mul_mat(v_w, x) -> reshape [D, Hkv, T, 1]
//   Q = rms_norm_head_dim(Q) * q_norm_w
//   K = rms_norm_head_dim(K) * k_norm_w
//   Q, K = rope_ext_neox(Q, K, positions)
//   cache_write: cpy(K -> self_k[layer, :T, :, :]),
//                cpy(V -> self_v[layer, :T, :, :])
//   K_full = repeat(K_view_from_cache, Hkv->H)
//   V_full = repeat(V_view_from_cache, Hkv->H)
//   kq = mul_mat(K_full_permuted, Q_permuted) -> [T_k, T_q, H]
//   attn = soft_max_ext(kq, mask, scale) @ V_full_permuted
//   o = o_proj(merge_heads(attn))
//   x = residual + o
//
//   residual = x
//   x = rms_norm(x) * norm_ffn_w
//   ff = down_proj(silu(gate_proj(x)) * up_proj(x))
//   x = residual + ff
//
// KV cache layout (per layer il):
//   flat offset = il * n_ctx * kv_dim
//   within layer: position t at offset t * kv_dim; head h at h * D.
//   Matches the memory emitted by reshape_4d(Kproj, D, Hkv, T, 1).
//
// MRoPE reduction: for text-only ASR every position has a single
// (T, H, W) coordinate so the interleaved MRoPE collapses to NeoX
// RoPE on the temporal axis. We ggml_rope_ext with GGML_ROPE_TYPE_NEOX.
// If a future variant breaks this assumption we'll switch to
// ggml_rope_multi.

#include "decoder.h"

#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>

namespace transcribe::qwen3_asr {

// Whether to use flash attention in the LM graphs. head_dim=128 is
// supported on every backend we ship (CPU/Metal/Vulkan), and flash
// handles GQA natively (n_kv_heads on K/V, n_heads on Q), which lets
// us skip the repeat-interleave step entirely. The kv_type cast to
// F16 follows the same pattern as the Cohere decoder.
static constexpr bool kUseFlashAttn = true;

namespace {

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr && name != nullptr) ggml_set_name(t, name);
    return t;
}

// Qwen3 RMSNorm: weight * rsqrt(mean(x^2) + eps) * x.
ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x,
                       ggml_tensor * weight, float eps)
{
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

} // namespace

PrefillBuild build_prefill_graph(ggml_context *         ctx,
                                 const QwenAsrWeights & weights,
                                 const QwenAsrHParams & hp,
                                 QwenAsrKvCache &       kv_cache,
                                 int                    T_prompt,
                                 int                    T_enc,
                                 int                    prefix_len,
                                 int                    suffix_len)
{
    PrefillBuild pb {};
    pb.T_prompt   = T_prompt;
    pb.T_enc      = T_enc;
    pb.prefix_len = prefix_len;
    pb.suffix_len = suffix_len;

    if (ctx == nullptr || T_prompt <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "qwen3_asr decoder: invalid arg (T_prompt=%d, T_enc=%d)\n",
                     T_prompt, T_enc);
        return pb;
    }
    if (prefix_len < 0 || suffix_len < 0 ||
        prefix_len + T_enc + suffix_len != T_prompt)
    {
        std::fprintf(stderr,
                     "qwen3_asr decoder: prefix_len(%d) + T_enc(%d) + "
                     "suffix_len(%d) != T_prompt(%d)\n",
                     prefix_len, T_enc, suffix_len, T_prompt);
        return pb;
    }
    if (kv_cache.self_k == nullptr || kv_cache.self_v == nullptr) {
        std::fprintf(stderr, "qwen3_asr decoder: kv_cache not initialized\n");
        return pb;
    }
    if (T_prompt > kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "qwen3_asr decoder: T_prompt=%d exceeds kv_cache.n_ctx=%d\n",
                     T_prompt, kv_cache.n_ctx);
        return pb;
    }

    const int64_t hidden      = hp.dec_hidden;
    const int64_t n_heads     = hp.dec_n_heads;
    const int64_t n_kv_heads  = hp.dec_n_kv_heads;
    const int64_t n_groups    = n_heads / n_kv_heads;
    const int64_t head_dim    = hp.dec_head_dim;
    const int64_t q_dim       = n_heads    * head_dim;
    const int64_t kv_dim      = n_kv_heads * head_dim;
    const int64_t vocab       = hp.dec_vocab_size;
    const int     n_layer     = hp.dec_n_layers;
    const int     n_ctx       = kv_cache.n_ctx;
    const float   rms_eps     = hp.dec_rms_norm_eps;
    const float   rope_theta  = hp.dec_rope_theta;
    const float   scale_attn  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int     enc_out_dim = hp.enc_output_dim;

    if (enc_out_dim != hidden) {
        std::fprintf(stderr,
                     "qwen3_asr decoder: enc_output_dim (%d) != dec_hidden (%lld) — "
                     "audio injection requires matching dims\n",
                     enc_out_dim, static_cast<long long>(hidden));
        return pb;
    }

    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    // ---------- Graph inputs ----------
    pb.input_ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
    named(pb.input_ids_in, "dec.input_ids");
    ggml_set_input(pb.input_ids_in);

    pb.enc_out_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc_out_dim, T_enc);
    named(pb.enc_out_in, "dec.enc_out");
    ggml_set_input(pb.enc_out_in);

    pb.positions_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prompt);
    named(pb.positions_in, "dec.positions");
    ggml_set_input(pb.positions_in);

    pb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_prompt, T_prompt);
    named(pb.mask_in, "dec.attn_mask");
    ggml_set_input(pb.mask_in);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
    if (gf == nullptr) {
        std::fprintf(stderr,
                     "qwen3_asr decoder: ggml_new_graph_custom failed\n");
        return pb;
    }
    pb.graph = gf;

    // ---------- Token embedding (all positions, for the dump) ----------
    ggml_tensor * token_emb_all = ggml_get_rows(
        ctx, weights.dec_embed.token_w, pb.input_ids_in);
    named(token_emb_all, "dec.token_emb");
    pb.dumps.token_emb = token_emb_all;
    transcribe::debug::mark_tensor_for_dump(token_emb_all);

    // ---------- Audio injection via concat ----------
    // Assumes a single contiguous audio block at positions
    // [prefix_len, prefix_len + T_enc). Sliced views of token_emb_all
    // yield the prefix and suffix token embeddings; enc_out_in supplies
    // the audio block. Two ggml_concat calls splice them back into a
    // [hidden, T_prompt] tensor.
    const size_t emb_elem = ggml_element_size(token_emb_all);
    ggml_tensor * x_prefix = nullptr;
    if (prefix_len > 0) {
        x_prefix = ggml_view_2d(
            ctx, token_emb_all,
            hidden, prefix_len,
            emb_elem * hidden,        // nb[1]: row stride
            /*off=*/0);
        x_prefix = ggml_cont(ctx, x_prefix);
    }
    ggml_tensor * x_audio = pb.enc_out_in;

    ggml_tensor * x_suffix = nullptr;
    if (suffix_len > 0) {
        x_suffix = ggml_view_2d(
            ctx, token_emb_all,
            hidden, suffix_len,
            emb_elem * hidden,
            emb_elem * hidden * static_cast<size_t>(prefix_len + T_enc));
        x_suffix = ggml_cont(ctx, x_suffix);
    }

    ggml_tensor * x = x_audio;
    if (x_prefix != nullptr) {
        x = ggml_concat(ctx, x_prefix, x, /*dim=*/1);
    }
    if (x_suffix != nullptr) {
        x = ggml_concat(ctx, x, x_suffix, /*dim=*/1);
    }
    named(x, "dec.audio_injected");
    pb.dumps.audio_injected = x;
    transcribe::debug::mark_tensor_for_dump(x);

    // The audio-injected tensor is used as a source by every block's
    // rms_norm; the scheduler keeps it alive through the graph, so
    // the dump-preservation mark above is sufficient.

    for (int il = 0; il < n_layer; ++il) {
        const auto & w = weights.dec_blocks[il];

        // ---- Attention sub-layer ----
        ggml_tensor * x_norm = rms_norm(ctx, x, w.norm_attn_w, rms_eps);

        // Q/K/V projections (bias-free on Qwen3).
        ggml_tensor * Q = ggml_mul_mat(ctx, w.attn_q_w, x_norm);
        ggml_tensor * K = ggml_mul_mat(ctx, w.attn_k_w, x_norm);
        ggml_tensor * V = ggml_mul_mat(ctx, w.attn_v_w, x_norm);

        Q = ggml_reshape_4d(ctx, Q, head_dim, n_heads,    T_prompt, 1);
        K = ggml_reshape_4d(ctx, K, head_dim, n_kv_heads, T_prompt, 1);
        V = ggml_reshape_4d(ctx, V, head_dim, n_kv_heads, T_prompt, 1);

        // Per-head Q/K RMSNorm (reduces along ne[0]=head_dim).
        Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, rms_eps), w.attn_q_norm);
        K = ggml_mul(ctx, ggml_rms_norm(ctx, K, rms_eps), w.attn_k_norm);

        // RoPE (NeoX rotate_half) on Q and K.
        Q = ggml_rope_ext(ctx, Q, pb.positions_in, /*c=*/nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta,
                          1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, pb.positions_in, nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta,
                          1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Cache write: K, V currently have ne=[D, Hkv, T_prompt, 1];
        // memory order (fastest→slowest) is D, Hkv, T. The cache stores
        // position-major within each layer: per position, all (D*Hkv)
        // values are contiguous. Same layout. A 1D cpy handles it.
        {
            const size_t layer_off = static_cast<size_t>(il) * n_ctx * kv_dim;
            const size_t n_elem    = static_cast<size_t>(T_prompt) * kv_dim;

            ggml_tensor * k_dst = ggml_view_1d(
                ctx, kv_cache.self_k, n_elem, k_elem * layer_off);
            ggml_tensor * v_dst = ggml_view_1d(
                ctx, kv_cache.self_v, n_elem, v_elem * layer_off);

            ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
        }

        // Read K, V back from the cache for attention. Memory layout
        // per layer: position t at offset t*kv_dim, head h at h*D
        // within that. We view the cache DIRECTLY as [D, T, Hkv] via
        // stride tricks — no permute + cont needed. mul_mat and
        // flash_attn_ext both accept strided inputs. Identical pattern
        // to Cohere's mha_self_cached.
        const size_t layer_off_bytes =
            k_elem * static_cast<size_t>(il) * n_ctx * kv_dim;
        ggml_tensor * K_att = ggml_view_3d(
            ctx, kv_cache.self_k,
            head_dim,
            T_prompt,
            n_kv_heads,
            /*nb1=*/k_elem * kv_dim,     // stride between positions
            /*nb2=*/k_elem * head_dim,   // stride between kv heads
            layer_off_bytes);
        ggml_tensor * V_att = ggml_view_3d(
            ctx, kv_cache.self_v,
            head_dim,
            T_prompt,
            n_kv_heads,
            v_elem * kv_dim,
            v_elem * head_dim,
            v_elem * static_cast<size_t>(il) * n_ctx * kv_dim);

        // Permute Q for attention: [D, H, T, 1] → [D, T, H, 1].
        ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

        ggml_tensor * o;
        if (kUseFlashAttn) {
            // ggml_flash_attn_ext handles GQA natively when K/V have
            // n_kv_heads along axis 2 and Q has n_heads — so we skip
            // the repeat_interleave step. Mask must be F16.
            ggml_tensor * mask_f16 = ggml_cast(ctx, pb.mask_in, GGML_TYPE_F16);
            o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask_f16,
                                    scale_attn, /*max_bias=*/0.0f,
                                    /*logit_softcap=*/0.0f);
            // Output is [D, H, T_q, 1] — already permuted.
            o = ggml_reshape_2d(ctx, o, q_dim, T_prompt);
        } else {
            // Manual GQA path: emulate repeat_interleave via
            // reshape-into-(1,Hkv) → repeat along the new size-1 axis
            // → collapse back. Keeps the numerics identical to the
            // reference's explicit repeat_kv.
            ggml_tensor * K_4d = ggml_reshape_4d(
                ctx, K_att, head_dim, T_prompt, 1, n_kv_heads);
            ggml_tensor * V_4d = ggml_reshape_4d(
                ctx, V_att, head_dim, T_prompt, 1, n_kv_heads);
            ggml_tensor * K_rep_template = ggml_new_tensor_4d(
                ctx, K_att->type, head_dim, T_prompt, n_groups, n_kv_heads);
            ggml_tensor * V_rep_template = ggml_new_tensor_4d(
                ctx, V_att->type, head_dim, T_prompt, n_groups, n_kv_heads);
            ggml_tensor * K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
            ggml_tensor * V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
            ggml_tensor * K_full = ggml_reshape_3d(
                ctx, K_rep, head_dim, T_prompt, n_heads);
            ggml_tensor * V_full = ggml_reshape_3d(
                ctx, V_rep, head_dim, T_prompt, n_heads);

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

        // ---- MLP sub-layer (SwiGLU) ----
        ggml_tensor * ff_norm = rms_norm(ctx, x, w.norm_ffn_w, rms_eps);
        ggml_tensor * gate    = ggml_mul_mat(ctx, w.ffn_gate_w, ff_norm);
        ggml_tensor * up      = ggml_mul_mat(ctx, w.ffn_up_w,   ff_norm);
        gate = ggml_silu(ctx, gate);
        ggml_tensor * ff = ggml_mul(ctx, gate, up);
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

    // ---------- Final RMSNorm + tied lm_head ----------
    x = rms_norm(ctx, x, weights.dec_final.norm_w, rms_eps);
    named(x, "dec.out_before_head");
    pb.dumps.out_before_head = x;
    transcribe::debug::mark_tensor_for_dump(x);

    // Slice the last position and multiply by the tied embedding.
    ggml_tensor * last_x = ggml_view_2d(
        ctx, x,
        hidden, 1,
        ggml_element_size(x) * hidden,
        ggml_element_size(x) * hidden * static_cast<size_t>(T_prompt - 1));
    last_x = ggml_cont(ctx, last_x);

    ggml_tensor * logits = ggml_mul_mat(ctx, weights.dec_embed.token_w, last_x);
    logits = ggml_reshape_1d(ctx, logits, vocab);
    named(logits, "dec.logits_raw");
    pb.dumps.logits_raw = logits;
    transcribe::debug::mark_tensor_for_dump(logits);

    pb.out = logits;
    ggml_set_output(pb.out);

    ggml_build_forward_expand(gf, pb.out);

    // Preserve intermediate dumps.
    if (pb.dumps.token_emb)       ggml_build_forward_expand(gf, pb.dumps.token_emb);
    if (pb.dumps.audio_injected)  ggml_build_forward_expand(gf, pb.dumps.audio_injected);
    if (pb.dumps.block_0_out)     ggml_build_forward_expand(gf, pb.dumps.block_0_out);
    if (pb.dumps.block_last_out)  ggml_build_forward_expand(gf, pb.dumps.block_last_out);
    if (pb.dumps.out_before_head) ggml_build_forward_expand(gf, pb.dumps.out_before_head);

    return pb;
}

// ---------------------------------------------------------------------------
// Step graph (single-token decode)
// ---------------------------------------------------------------------------

StepBuild build_step_graph(ggml_context *         ctx,
                           const QwenAsrWeights & weights,
                           const QwenAsrHParams & hp,
                           QwenAsrKvCache &       kv_cache,
                           int                    n_past)
{
    StepBuild sb {};
    sb.n_past = n_past;

    if (ctx == nullptr || n_past < 0) {
        std::fprintf(stderr,
                     "qwen3_asr step: invalid arg (n_past=%d)\n", n_past);
        return sb;
    }
    if (kv_cache.self_k == nullptr) {
        std::fprintf(stderr, "qwen3_asr step: kv_cache not initialized\n");
        return sb;
    }
    const int n_kv = n_past + 1;
    if (n_kv > kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "qwen3_asr step: n_kv=%d exceeds kv_cache.n_ctx=%d\n",
                     n_kv, kv_cache.n_ctx);
        return sb;
    }

    const int64_t hidden      = hp.dec_hidden;
    const int64_t n_heads     = hp.dec_n_heads;
    const int64_t n_kv_heads  = hp.dec_n_kv_heads;
    const int64_t n_groups    = n_heads / n_kv_heads;
    const int64_t head_dim    = hp.dec_head_dim;
    const int64_t q_dim       = n_heads    * head_dim;
    const int64_t kv_dim      = n_kv_heads * head_dim;
    const int64_t vocab       = hp.dec_vocab_size;
    const int     n_layer     = hp.dec_n_layers;
    const int     n_ctx       = kv_cache.n_ctx;
    const float   rms_eps     = hp.dec_rms_norm_eps;
    const float   rope_theta  = hp.dec_rope_theta;
    const float   scale_attn  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t k_elem = ggml_element_size(kv_cache.self_k);
    const size_t v_elem = ggml_element_size(kv_cache.self_v);

    // ---------- Graph inputs ----------
    sb.input_id_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(sb.input_id_in, "step.input_id");
    ggml_set_input(sb.input_id_in);

    sb.position_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(sb.position_in, "step.position");
    ggml_set_input(sb.position_in);

    sb.mask_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, 1);
    ggml_set_name(sb.mask_in, "step.mask");
    ggml_set_input(sb.mask_in);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (gf == nullptr) {
        std::fprintf(stderr,
                     "qwen3_asr step: ggml_new_graph_custom failed\n");
        return sb;
    }
    sb.graph = gf;

    // ---------- Token embedding ----------
    ggml_tensor * x = ggml_get_rows(ctx, weights.dec_embed.token_w,
                                    sb.input_id_in);  // [hidden, 1]

    for (int il = 0; il < n_layer; ++il) {
        const auto & w = weights.dec_blocks[il];

        // Attention sub-layer.
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
                          rope_theta,
                          1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_ext(ctx, K, sb.position_in, nullptr,
                          static_cast<int>(head_dim),
                          GGML_ROPE_TYPE_NEOX,
                          hp.dec_max_position_embeddings,
                          rope_theta,
                          1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Cache write: a single token at position n_past.
        {
            const size_t layer_off = static_cast<size_t>(il) * n_ctx * kv_dim;
            const size_t pos_off   = static_cast<size_t>(n_past) * kv_dim;

            ggml_tensor * k_dst = ggml_view_1d(
                ctx, kv_cache.self_k, kv_dim,
                k_elem * (layer_off + pos_off));
            ggml_tensor * v_dst = ggml_view_1d(
                ctx, kv_cache.self_v, kv_dim,
                v_elem * (layer_off + pos_off));

            ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
        }

        // Read the full cache window [0, n_kv) for this layer.
        // Direct strided view of the KV cache as [D, T, Hkv] — no
        // permute + cont. Same pattern as the prefill path.
        const size_t layer_off_bytes =
            k_elem * static_cast<size_t>(il) * n_ctx * kv_dim;
        ggml_tensor * K_att = ggml_view_3d(
            ctx, kv_cache.self_k,
            head_dim, n_kv, n_kv_heads,
            k_elem * kv_dim,
            k_elem * head_dim,
            layer_off_bytes);
        ggml_tensor * V_att = ggml_view_3d(
            ctx, kv_cache.self_v,
            head_dim, n_kv, n_kv_heads,
            v_elem * kv_dim,
            v_elem * head_dim,
            v_elem * static_cast<size_t>(il) * n_ctx * kv_dim);

        ggml_tensor * Q_att = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

        ggml_tensor * o;
        if (kUseFlashAttn) {
            ggml_tensor * mask_f16 = ggml_cast(ctx, sb.mask_in, GGML_TYPE_F16);
            o = ggml_flash_attn_ext(ctx, Q_att, K_att, V_att, mask_f16,
                                    scale_attn, /*max_bias=*/0.0f,
                                    /*logit_softcap=*/0.0f);
            o = ggml_reshape_2d(ctx, o, q_dim, 1);
        } else {
            ggml_tensor * K_4d = ggml_reshape_4d(ctx, K_att, head_dim, n_kv, 1, n_kv_heads);
            ggml_tensor * V_4d = ggml_reshape_4d(ctx, V_att, head_dim, n_kv, 1, n_kv_heads);
            ggml_tensor * K_rep_template = ggml_new_tensor_4d(
                ctx, K_att->type, head_dim, n_kv, n_groups, n_kv_heads);
            ggml_tensor * V_rep_template = ggml_new_tensor_4d(
                ctx, V_att->type, head_dim, n_kv, n_groups, n_kv_heads);
            ggml_tensor * K_rep = ggml_repeat(ctx, K_4d, K_rep_template);
            ggml_tensor * V_rep = ggml_repeat(ctx, V_4d, V_rep_template);
            ggml_tensor * K_full = ggml_reshape_3d(ctx, K_rep, head_dim, n_kv, n_heads);
            ggml_tensor * V_full = ggml_reshape_3d(ctx, V_rep, head_dim, n_kv, n_heads);

            ggml_tensor * kq = ggml_mul_mat(ctx, K_full, Q_att);
            ggml_tensor * kq_soft = ggml_soft_max_ext(
                ctx, kq, sb.mask_in, scale_attn, /*max_bias=*/0.0f);
            ggml_tensor * V_t = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 0, 2, 3));
            o = ggml_mul_mat(ctx, V_t, kq_soft);
            o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
            o = ggml_reshape_2d(ctx, o, q_dim, 1);
        }
        o = ggml_mul_mat(ctx, w.attn_o_w, o);

        x = ggml_add(ctx, x, o);

        // MLP sub-layer.
        ggml_tensor * ff_norm = rms_norm(ctx, x, w.norm_ffn_w, rms_eps);
        ggml_tensor * gate = ggml_mul_mat(ctx, w.ffn_gate_w, ff_norm);
        ggml_tensor * up   = ggml_mul_mat(ctx, w.ffn_up_w,   ff_norm);
        gate = ggml_silu(ctx, gate);
        ggml_tensor * ff = ggml_mul(ctx, gate, up);
        ff = ggml_mul_mat(ctx, w.ffn_down_w, ff);

        x = ggml_add(ctx, x, ff);
    }

    x = rms_norm(ctx, x, weights.dec_final.norm_w, rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, weights.dec_embed.token_w, x);
    logits = ggml_reshape_1d(ctx, logits, vocab);
    ggml_set_name(logits, "step.logits");

    // argmax runs on the accelerator — produces a single i32 so the
    // host-side readback is 4 bytes instead of a 600 KB vocab vector.
    // argmax is monotone-invariant so we don't need softmax/log.
    ggml_tensor * amax = ggml_argmax(ctx, logits);
    ggml_set_name(amax, "step.argmax");

    sb.out = amax;
    ggml_set_output(sb.out);

    ggml_build_forward_expand(gf, sb.out);
    // Also keep the raw logits alive in case the caller wants to sample
    // (not used today, but harmless — the scheduler pins the allocation).
    ggml_build_forward_expand(gf, logits);

    return sb;
}

} // namespace transcribe::qwen3_asr
