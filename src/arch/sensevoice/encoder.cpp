// arch/sensevoice/encoder.cpp - SenseVoice SAN-M encoder graph builder.
//
// Forward shape (single-utterance, batch=1, channel-innermost ggml ne):
//
//   frontend.in       [d_input=560, T_lfr]
//     -> prefix prepend (lid + event_emo + textnorm via ggml_get_rows)
//                       [d_input, 4 + T_lfr]              = enc.input.with_prefix
//     -> scale by sqrt(d_model=512)
//     -> add sinusoidal PE (depth = current width = d_input = 560,
//                           1-based positions)
//                       [d_input, 4 + T_lfr]              = enc.embed.out
//     -> encoders0[0]  (SAN-M block, projection 560 -> 512, no attn-residual)
//                       [d_model, 4 + T_lfr]              = enc.encoders0.0.out
//     -> encoders[0..n-2] x49 (SAN-M block, in==out==d_model)
//                       [d_model, 4 + T_lfr]              = enc.encoders.{0,24,48}.out
//     -> after_norm (LayerNorm eps=1e-12)
//                       [d_model, 4 + T_lfr]              = enc.after_norm.out
//     -> tp_encoders[0..tp-1] x20 (SAN-M block)
//                       [d_model, 4 + T_lfr]              = enc.tp_encoders.{0,10,19}.out
//     -> tp_norm (LayerNorm eps=1e-12)
//                       [d_model, 4 + T_lfr]              = enc.tp_norm.out
//     -> ctc.head linear [d_model, vocab] + bias [vocab]
//                       [vocab, 4 + T_lfr]                = ctc.logits.raw
//
// SAN-M block layout (residual variant — encoders[*], tp_encoders[*]):
//
//   x_in
//     y = layer_norm(x_in, gamma_attn, beta_attn, eps=1e-12)
//     qkv = mul_mat(W_qkv, y) + b_qkv                        [3*d_model, T]
//     split into q, k, v along the channel axis (d_model each)
//     reshape q,k,v -> [head_dim, n_heads, T, 1]
//     v_pre  = ungrouped v in [d_model, T] (post-mask in batched eval;
//              batch=1 means mask is all-1, so the masking is a no-op)
//
//     -- FSMN parallel branch on v_pre --
//     v_t = transpose(v_pre) -> [T, d_model]
//     fsmn = depthwise_conv1d(W_fsmn, v_t, k=11, pad=5)   [T, d_model]
//     fsmn = transpose(fsmn) -> [d_model, T]
//     fsmn = fsmn + v_pre
//
//     -- SDPA --
//     scores = q (scaled by 1/sqrt(d_k)) @ k^T -> softmax -> @ v
//     merge heads -> [d_model, T]
//     att = mul_mat(W_out, sdpa) + b_out
//
//     attn_branch = att + fsmn
//     x = x_in + attn_branch                       (residual when in==out)
//     z = layer_norm(x, gamma_ffn, beta_ffn, eps=1e-12)
//     y = mul_mat(W_fc1, z) + b_fc1; y = relu(y);
//         y = mul_mat(W_fc2, y) + b_fc2
//     x = x + y
//
// Projection variant (encoders0[0] only):
//   - norm_attn input dim is d_input (=560), not d_model.
//   - q_k_v input dim is d_input.
//   - The block-level attention residual is SKIPPED (x = attn_branch
//     directly), because the channel count changes 560 -> 512.
//   - The FFN residual still adds.
//
// Mask is omitted: batch=1 inference always has every position valid,
// so the mask is all 1s and the masked_fill / softmax-mask / FSMN
// V-mask are no-ops. Re-add when batched inference lands.
//
// LayerNorm epsilon is 1e-12 (FunASR's
// transformer.layer_norm.LayerNorm), NOT the conformer-shared 1e-5.

#include "encoder.h"

#include "weights.h"

#include "transcribe-debug.h"
#include "conformer/conformer.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace transcribe::sensevoice {

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
    if (beta != nullptr) {
        y = ggml_add(ctx, y, beta);
    }
    return y;
}

// Depthwise 1D conv with kernel-size symmetric padding, mirroring the
// reference's ConstantPad1d(left=(K-1)/2, right=(K-1)/2) +
// nn.Conv1d(groups=C). Input layout: [d_model, T] in ggml ne
// (channel-innermost).
ggml_tensor * fsmn_branch(ggml_context * ctx,
                          ggml_tensor *  v_pre,        // [d_model, T]
                          ggml_tensor *  fsmn_w,       // ne=[K, 1, d_model]
                          int            kernel)
{
    // The reference op is depthwise Conv1d on a channels-first tensor
    // ([B, C, L]). Our layout is channels-innermost. Transpose to
    // [T, d_model] so conv_1d_dw_f32 (which expects ne=[T, C, B]) sees
    // the right shape, then transpose back.
    ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_pre));  // [T, d_model]

    const int padding = (kernel - 1) / 2;  // 5 for K=11, sanm_shift=0
    ggml_tensor * fsmn = conf::conv_1d_dw_f32(
        ctx, fsmn_w, v_t,
        /*stride=*/1,
        /*padding=*/padding,
        /*dilation=*/1);
    // fsmn ne=[T, d_model, 1]. Drop the singleton batch and transpose back.
    fsmn = ggml_reshape_2d(ctx, fsmn, fsmn->ne[0], fsmn->ne[1]); // [T, d_model]
    fsmn = ggml_cont(ctx, ggml_transpose(ctx, fsmn));            // [d_model, T]

    // Residual within the FSMN: x_fsmn += masked_v.
    fsmn = ggml_add(ctx, fsmn, v_pre);
    return fsmn;
}

// SAN-M attention sub-block. Returns the post-projection branch tensor
// (= sdpa_proj + fsmn_memory) at [d_model, T].
ggml_tensor * sanm_attention(ggml_context *          ctx,
                             ggml_tensor *           x,         // [d_in, T]
                             const SenseVoiceBlock & b,
                             int                     n_heads,
                             int                     d_model,
                             int                     kernel)
{
    const int head_dim = d_model / n_heads;
    const int64_t T    = x->ne[1];
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Fused QKV projection. W_qkv ne=[d_in, 3*d_model] (PyTorch
    // [3*d_model, d_in] stored ggml-style as [d_in, 3*d_model]).
    ggml_tensor * qkv = ggml_mul_mat(ctx, b.attn_qkv_w, x);    // [3*d_model, T]
    qkv = ggml_add(ctx, qkv, b.attn_qkv_b);

    // Split QKV along the channel axis (ne[0]). Each view is
    // contiguous in time (ne[1]) but the channel slices live at
    // offsets 0, d_model, 2*d_model.
    const size_t qkv_nb1 = qkv->nb[1];
    ggml_tensor * q = ggml_view_2d(
        ctx, qkv, d_model, T, qkv_nb1, /*offset=*/0);
    ggml_tensor * k = ggml_view_2d(
        ctx, qkv, d_model, T, qkv_nb1,
        /*offset=*/static_cast<size_t>(d_model) * sizeof(float));
    ggml_tensor * v = ggml_view_2d(
        ctx, qkv, d_model, T, qkv_nb1,
        /*offset=*/static_cast<size_t>(2 * d_model) * sizeof(float));

    // The FSMN consumes V *before* head reshape, in the [d_model, T]
    // layout. Make a contiguous copy so the post-FSMN add does not see
    // a strided alias of V.
    ggml_tensor * v_pre = ggml_cont(ctx, v);

    // ----- FSMN branch (parallel to SDPA) -----
    ggml_tensor * fsmn = fsmn_branch(ctx, v_pre, b.attn_fsmn_w, kernel);

    // ----- SDPA -----
    // Reshape Q,K,V to [head_dim, n_heads, T, 1] for ggml_flash_attn_ext.
    // The fused QKV split above is non-contiguous along the channel
    // axis (each split view sees nb[0]=4 within the slice but nb[1] is
    // the full QKV row stride), so we cont each before reshape.
    auto split_heads = [&](ggml_tensor * t) {
        t = ggml_cont(ctx, t);
        return ggml_reshape_4d(ctx, t, head_dim, n_heads, T, 1);
    };
    ggml_tensor * qh = split_heads(q);
    ggml_tensor * kh = split_heads(k);
    ggml_tensor * vh = split_heads(v);

    // ggml_flash_attn_ext expects q,k,v with ne=[head_dim, T*, n_heads, 1]
    // (head_dim innermost, time at ne[1], heads at ne[2]).
    auto to_attn_layout = [&](ggml_tensor * t) {
        t = ggml_permute(ctx, t, 0, 2, 1, 3);   // [head_dim, T, n_heads, 1]
        return ggml_cont(ctx, t);
    };
    qh = to_attn_layout(qh);
    kh = to_attn_layout(kh);
    vh = to_attn_layout(vh);

    ggml_tensor * o = ggml_flash_attn_ext(
        ctx, qh, kh, vh, /*mask=*/nullptr,
        /*scale=*/scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    // ggml_flash_attn_ext returns ne=[head_dim, n_heads, T, 1] — fast-to-
    // slow that's already time-major-with-heads-concatenated in memory
    // (for each T, the 4*head_dim values = head_0_dims | head_1_dims |
    // ...). Reshape directly to [d_model, T] without permuting.
    o = ggml_cont(ctx, o);
    o = ggml_reshape_2d(ctx, o, d_model, T);

    // Output projection.
    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    o = ggml_add(ctx, o, b.attn_out_b);

    // SAN-M attention output: SDPA + FSMN.
    return ggml_add(ctx, o, fsmn);
}

ggml_tensor * sanm_ffn(ggml_context *          ctx,
                       ggml_tensor *           x,
                       const SenseVoiceBlock & b)
{
    ggml_tensor * h = ggml_mul_mat(ctx, b.ffn_fc1_w, x);
    h = ggml_add(ctx, h, b.ffn_fc1_b);
    h = ggml_relu(ctx, h);
    h = ggml_mul_mat(ctx, b.ffn_fc2_w, h);
    h = ggml_add(ctx, h, b.ffn_fc2_b);
    return h;
}

// Residual SAN-M block (encoders[*] and tp_encoders[*]).
ggml_tensor * sanm_block_residual(ggml_context *          ctx,
                                  ggml_tensor *           x,
                                  const SenseVoiceBlock & b,
                                  int                     n_heads,
                                  int                     d_model,
                                  int                     kernel)
{
    ggml_tensor * y = sv_layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = sanm_attention(ctx, y, b, n_heads, d_model, kernel);
    x = ggml_add(ctx, x, y);

    ggml_tensor * z = sv_layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
    z = sanm_ffn(ctx, z, b);
    x = ggml_add(ctx, x, z);
    return x;
}

// Projection SAN-M block (encoders0[0] only). in_size != size, so the
// attention sub-block has no residual — the projection itself is the
// residual stream.
ggml_tensor * sanm_block_projection(ggml_context *          ctx,
                                    ggml_tensor *           x,        // [d_input, T]
                                    const SenseVoiceBlock & b,
                                    int                     n_heads,
                                    int                     d_model,
                                    int                     kernel)
{
    ggml_tensor * y = sv_layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    y = sanm_attention(ctx, y, b, n_heads, d_model, kernel);
    // No attention residual — the channel count changes 560 -> 512.
    x = y;

    ggml_tensor * z = sv_layer_norm(ctx, x, b.norm_ffn_w, b.norm_ffn_b);
    z = sanm_ffn(ctx, z, b);
    x = ggml_add(ctx, x, z);
    return x;
}

// Mark a tensor as live for the post-compute dump pass and stash the
// pointer in the dumps slot. The mark is a no-op when
// TRANSCRIBE_DUMP_DIR is unset.
void mark_dump(ggml_tensor *& slot, ggml_tensor * t, const char * name) {
    named(t, name);
    transcribe::debug::mark_tensor_for_dump(t);
    slot = t;
}

} // namespace

EncoderBuild build_encoder_graph(ggml_context *            ctx,
                                 const SenseVoiceWeights & w,
                                 const SenseVoiceHParams & hp,
                                 int                       n_lfr_frames)
{
    EncoderBuild eb {};

    if (ctx == nullptr || n_lfr_frames <= 0) {
        std::fprintf(stderr,
                     "sensevoice encoder: invalid arg "
                     "(ctx=%p, n_lfr_frames=%d)\n",
                     static_cast<void *>(ctx), n_lfr_frames);
        return eb;
    }

    const int d_input = hp.enc_d_input;
    const int d_model = hp.enc_d_model;
    const int n_heads = hp.enc_n_heads;
    const int kernel  = hp.enc_kernel;
    const int T_in    = n_lfr_frames;
    const int T       = T_in + 4;          // post-prefix length

    // ----- inputs ----------------------------------------------------
    eb.frontend_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_input, T_in);
    named(eb.frontend_in, "frontend.in");
    ggml_set_input(eb.frontend_in);
    mark_dump(eb.dumps.frontend_out, eb.frontend_in, "frontend.fbank.lfr.cmvn.out");

    eb.lid_idx       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    named(eb.lid_idx, "prefix.lid_idx");
    ggml_set_input(eb.lid_idx);

    eb.event_emo_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    named(eb.event_emo_idx, "prefix.event_emo_idx");
    ggml_set_input(eb.event_emo_idx);

    eb.textnorm_idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    named(eb.textnorm_idx, "prefix.textnorm_idx");
    ggml_set_input(eb.textnorm_idx);

    // Sinusoidal PE input (host fills with 1-based positions encoded
    // at depth=d_input).
    ggml_tensor * pe_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_input, T);
    named(pe_in, "pe.in");
    ggml_set_input(pe_in);

    // ----- prefix prepend --------------------------------------------
    ggml_tensor * lid_emb       = ggml_get_rows(ctx, w.embed, eb.lid_idx);       // [d_in, 1]
    ggml_tensor * event_emo_emb = ggml_get_rows(ctx, w.embed, eb.event_emo_idx); // [d_in, 2]
    ggml_tensor * textnorm_emb  = ggml_get_rows(ctx, w.embed, eb.textnorm_idx);  // [d_in, 1]

    mark_dump(eb.dumps.prefix_lid,       lid_emb,       "enc.prefix.lid_emb");
    mark_dump(eb.dumps.prefix_event_emo, event_emo_emb, "enc.prefix.event_emo_emb");
    mark_dump(eb.dumps.prefix_textnorm,  textnorm_emb,  "enc.prefix.textnorm_emb");

    // Reference order:
    //   speech = cat(textnorm,     speech)
    //   speech = cat(lid, ev_emo,  speech)
    // Concat along ne[1] (= time axis after batch was squeezed out).
    ggml_tensor * tn_then_speech = ggml_concat(ctx, textnorm_emb, eb.frontend_in, /*dim=*/1);
    ggml_tensor * lid_evev       = ggml_concat(ctx, lid_emb, event_emo_emb, /*dim=*/1);
    ggml_tensor * x              = ggml_concat(ctx, lid_evev, tn_then_speech, /*dim=*/1);
    mark_dump(eb.dumps.input_with_prefix, x, "enc.input.with_prefix");

    // ----- embedding scale + PE add ----------------------------------
    // Scale by sqrt(d_model). NOTE: x is still 560-d here, but the
    // scale is sqrt(d_model=512), per the reference's encoder.forward().
    const float embed_scale = std::sqrt(static_cast<float>(d_model));
    x = ggml_scale(ctx, x, embed_scale);
    x = ggml_add(ctx, x, pe_in);
    mark_dump(eb.dumps.embed_out, x, "enc.embed.out");

    // ----- encoders0[0] (560 -> 512 projection) ----------------------
    x = sanm_block_projection(ctx, x, w.encoders0,
                              n_heads, d_model, kernel);
    mark_dump(eb.dumps.encoders0_0_out, x, "enc.encoders0.0.out");

    // ----- encoders[0..n-2] ------------------------------------------
    const int n_main = static_cast<int>(w.encoders.size());
    if (n_main > 0) {
        const int last_idx = n_main - 1;
        const int mid_idx  = n_main / 2;
        for (int i = 0; i < n_main; ++i) {
            x = sanm_block_residual(ctx, x, w.encoders[static_cast<size_t>(i)],
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

    // ----- after_norm ------------------------------------------------
    x = sv_layer_norm(ctx, x, w.after_norm_w, w.after_norm_b);
    mark_dump(eb.dumps.after_norm_out, x, "enc.after_norm.out");

    // ----- tp_encoders[0..tp-1] --------------------------------------
    const int n_tp = static_cast<int>(w.tp_encoders.size());
    if (n_tp > 0) {
        const int last_idx = n_tp - 1;
        const int mid_idx  = n_tp / 2;
        for (int i = 0; i < n_tp; ++i) {
            x = sanm_block_residual(ctx, x, w.tp_encoders[static_cast<size_t>(i)],
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

    // ----- tp_norm ---------------------------------------------------
    x = sv_layer_norm(ctx, x, w.tp_norm_w, w.tp_norm_b);
    mark_dump(eb.dumps.tp_norm_out, x, "enc.tp_norm.out");

    // ----- CTC head --------------------------------------------------
    ggml_tensor * logits = ggml_mul_mat(ctx, w.ctc_head_w, x);
    logits = ggml_add(ctx, logits, w.ctc_head_b);
    mark_dump(eb.dumps.ctc_logits, logits, "ctc.logits.raw");

    // log_softmax over the vocab axis (ne[0]).
    ggml_tensor * log_probs = ggml_log(ctx, ggml_soft_max(ctx, logits));
    mark_dump(eb.dumps.ctc_log_probs, log_probs, "ctc.log_probs");

    eb.out = log_probs;
    ggml_set_output(eb.out);

    // 70 SAN-M blocks * ~30 ops/block + frontend/PE/CTC ≈ 2300 nodes.
    // 8192 leaves ample headroom.
    eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (eb.graph == nullptr) {
        std::fprintf(stderr,
                     "sensevoice encoder: ggml_new_graph_custom failed\n");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);

    return eb;
}

} // namespace transcribe::sensevoice
