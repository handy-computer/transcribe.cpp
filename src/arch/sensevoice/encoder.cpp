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
// SAN-M block topology lives in src/sanm/sanm.h (shared with funasr_nano).
// This file owns the surrounding shape — prefix prepend, sinusoidal PE
// add, after_norm / tp_norm, CTC head, and dump-marker wiring.
//
// Mask is omitted: batch=1 inference always has every position valid.

#include "encoder.h"

#include "weights.h"

#include "conformer/conformer.h"
#include "sanm/sanm.h"
#include "transcribe-debug.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace transcribe::sensevoice {

namespace {

namespace conf = transcribe::conformer;
namespace sanm = transcribe::sanm;
using conf::named;

sanm::SanmBlockView to_sanm_view(const SenseVoiceBlock & b) {
    sanm::SanmBlockView v;
    v.norm_attn_w = b.norm_attn_w;
    v.norm_attn_b = b.norm_attn_b;
    v.attn_qkv_w  = b.attn_qkv_w;
    v.attn_qkv_b  = b.attn_qkv_b;
    v.attn_out_w  = b.attn_out_w;
    v.attn_out_b  = b.attn_out_b;
    v.attn_fsmn_w = b.attn_fsmn_w;
    v.norm_ffn_w  = b.norm_ffn_w;
    v.norm_ffn_b  = b.norm_ffn_b;
    v.ffn_fc1_w   = b.ffn_fc1_w;
    v.ffn_fc1_b   = b.ffn_fc1_b;
    v.ffn_fc2_w   = b.ffn_fc2_w;
    v.ffn_fc2_b   = b.ffn_fc2_b;
    return v;
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
    const int T_in    = n_lfr_frames;
    const int T       = T_in + 4;          // post-prefix length

    const sanm::SanmBlockParams block_params {
        /*n_heads=*/hp.enc_n_heads,
        /*d_model=*/d_model,
        /*kernel=*/hp.enc_kernel,
    };

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
    x = sanm::sanm_block_projection(ctx, x, to_sanm_view(w.encoders0), block_params);
    mark_dump(eb.dumps.encoders0_0_out, x, "enc.encoders0.0.out");

    // ----- encoders[0..n-2] ------------------------------------------
    const int n_main = static_cast<int>(w.encoders.size());
    if (n_main > 0) {
        const int last_idx = n_main - 1;
        const int mid_idx  = n_main / 2;
        for (int i = 0; i < n_main; ++i) {
            x = sanm::sanm_block_residual(
                ctx, x, to_sanm_view(w.encoders[static_cast<size_t>(i)]),
                block_params);
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
    x = sanm::sv_layer_norm(ctx, x, w.after_norm_w, w.after_norm_b);
    mark_dump(eb.dumps.after_norm_out, x, "enc.after_norm.out");

    // ----- tp_encoders[0..tp-1] --------------------------------------
    const int n_tp = static_cast<int>(w.tp_encoders.size());
    if (n_tp > 0) {
        const int last_idx = n_tp - 1;
        const int mid_idx  = n_tp / 2;
        for (int i = 0; i < n_tp; ++i) {
            x = sanm::sanm_block_residual(
                ctx, x, to_sanm_view(w.tp_encoders[static_cast<size_t>(i)]),
                block_params);
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
    x = sanm::sv_layer_norm(ctx, x, w.tp_norm_w, w.tp_norm_b);
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
