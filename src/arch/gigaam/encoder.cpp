// arch/gigaam/encoder.cpp - GigaAM Conformer encoder graph builder.
//
// Reuses transcribe::conformer helpers for the macaron FF, conv module,
// and the LayerNorm primitive. Adds gigaam-specific pieces:
//
//   - 2-conv1d pre_encode (not parakeet's 4-conv2d stack).
//   - Rotary self-attention with PRE-projection rotation (the rotation
//     is applied to x before Wq/Wk; same rotated x feeds both, Wv gets
//     unrotated x).
//
// Layout convention:
//   - mel_in ne=[T_mel, n_mels, 1, 1] f32 (matches the reference dumper's
//     [n_mels, T_mel] row-major layout byte-for-byte).
//   - After pre_encode: [d_model, T_enc, 1, 1] where T_enc = floor((T_mel-1)/4)
//     (two stride-2 conv1d with kernel=5, sym pad=2).
//   - All conformer block intermediates stay at [d_model, T_enc, 1, 1].

#include "encoder.h"

#include "weights.h"

#include "conformer/conformer.h"
#include "transcribe-debug.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace transcribe::gigaam {

namespace {

namespace conf = transcribe::conformer;

// Project a GigaamBlock onto the shared BlockView so we can reuse
// conf::conv_module. The attention-half fields stay nullptr because we
// run a custom rotary path; the conv module fields are the only ones
// `conf::conv_module` reads.
conf::BlockView to_conv_view(const GigaamBlock & b) {
    conf::BlockView v;
    v.conv_pw1_w  = b.conv_pw1_w;
    v.conv_pw1_b  = b.conv_pw1_b;
    v.conv_dw_w   = b.conv_dw_w;
    v.conv_dw_b   = b.conv_dw_b;
    v.conv_pw2_w  = b.conv_pw2_w;
    v.conv_pw2_b  = b.conv_pw2_b;
    v.conv_ln_w   = b.conv_ln_w;
    v.conv_ln_b   = b.conv_ln_b;
    return v;
}

// Build the 2-conv1d pre-encode stack.
//
// Input  mel_in:    ne=[T_mel, n_mels, 1, 1]
// Output pre_enc:   ne=[d_model, T_enc, 1, 1] with T_enc ≈ T_mel / 4
//
// Reference (gigaam.encoder.StridingSubsampling):
//   x = audio.transpose(1, 2)            # [B, n_mels, T]
//   x = self.conv(x)                     # Sequential(Conv1d(64→768, k=5, s=2),
//                                        #            ReLU,
//                                        #            Conv1d(768→768, k=5, s=2),
//                                        #            ReLU)
//   x = x.transpose(1, 2)                # [B, T_enc, d_model]
//
// In ggml: mel_in is already [T_mel, n_mels] (matching PyTorch [n_mels, T] when
// considering row-major + ggml's fast-to-slow convention). conv_1d_f32 expects
// data ne=[W=T, IC, N] and kernel ne=[K, IC, OC] — both match the GGUF layout.
// After two stride-2 conv1d the result is [T_enc, d_model, 1] which we
// transpose+contiguous to [d_model, T_enc, 1, 1] to match the conformer
// downstream layout.
ggml_tensor * build_pre_encode(ggml_context *          ctx,
                               const GigaamPreEncode & pe,
                               ggml_tensor *           mel_in,
                               int                     subs_kernel_size,
                               const char *            name_prefix)
{
    const int padding = (subs_kernel_size - 1) / 2; // sym pad, e.g. k=5 -> 2
    char buf[64];

    auto named = [&](ggml_tensor * t, const char * tag) {
        if (name_prefix != nullptr) {
            std::snprintf(buf, sizeof(buf), "%s.%s", name_prefix, tag);
            ggml_set_name(t, buf);
        }
        return t;
    };

    // Conv 0: in=feat_in, out=d_model. Output ne=[T_mel/2, d_model, 1].
    ggml_tensor * x = conf::conv_1d_f32(ctx, pe.conv0_w, mel_in,
                                        /*stride=*/2, padding, /*dilation=*/1);
    // conv_1d_f32 returns ne=[T_out, OC, N]. Reshape to add bias on the
    // OC axis.
    x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    {
        ggml_tensor * bias_r = ggml_reshape_2d(ctx, pe.conv0_b, 1, x->ne[1]);
        x = ggml_add(ctx, x, bias_r);
    }
    x = ggml_relu(ctx, x);

    // conv_1d_f32 expects data ne=[T, IC, B]; we have 2D [T, IC]. Reshape
    // to 3D for the next conv.
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);

    // Conv 2: in=d_model, out=d_model.
    x = conf::conv_1d_f32(ctx, pe.conv2_w, x,
                          /*stride=*/2, padding, /*dilation=*/1);
    x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    {
        ggml_tensor * bias_r = ggml_reshape_2d(ctx, pe.conv2_b, 1, x->ne[1]);
        x = ggml_add(ctx, x, bias_r);
    }
    x = ggml_relu(ctx, x);

    // Materialize the relu output as a 3D tensor and transpose into
    // the [d_model, T_enc, 1] layout the conformer blocks consume.
    // The cont after permute is what does the actual byte transpose.
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    return x;
}

// Build the rotary attention sub-block.
//
// GigaAM applies rotary to the PRE-projection x: reshape x to
// [d_k, n_head, T, B], rotate the d_k axis with NEOX split-halves, then
// reshape back to [d_model, T, B] and apply Wq / Wk. Wv uses the
// UNROTATED x. Output projection at the end.
//
// Input  x:       [d_model, T, 1, 1]
// Output:        [d_model, T, 1, 1]
ggml_tensor * build_rotary_attn(ggml_context *      ctx,
                                ggml_tensor *       x,
                                ggml_tensor *       positions,
                                const GigaamBlock & b,
                                int                 d_model,
                                int                 n_head,
                                bool                use_flash)
{
    const int     head_dim = d_model / n_head;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t T        = x->ne[1];

    // Apply rotary to x BEFORE Wq/Wk projection. The rotation operates
    // on ne[0] (which must be the head_dim axis), so reshape x to
    // [head_dim, n_head, T, 1] first. NEOX mode = split-halves, matching
    // gigaam's `rtt_half(x) = [-x[..., d_k/2:], x[..., :d_k/2]]`.
    ggml_tensor * x_rot = ggml_reshape_4d(ctx, x, head_dim, n_head, T, 1);
    // GigaAM passes pos_emb_max_len (default 5000) as the rotary `base`
    // (theta). That is NOT the standard 10000 — see
    // RotaryPositionalEmbedding(d_model // n_heads, pos_emb_max_len) and
    // PositionalEncoding.__init__(dim, base). Verified empirically: the
    // dumped enc.pos_emb tensor matches base=5000 to ~1e-6 and base=10000
    // is off by ~0.1.
    x_rot = ggml_rope_ext(ctx, x_rot, positions, /*c=*/nullptr,
                          /*n_dims=*/head_dim,
                          GGML_ROPE_TYPE_NEOX,
                          /*n_ctx_orig=*/0,
                          /*theta=*/5000.0f,
                          /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f,
                          /*attn_factor=*/1.0f,
                          /*beta_fast=*/32.0f,
                          /*beta_slow=*/1.0f);
    // Fold heads back to [d_model, T, 1] for the Wq/Wk projection.
    x_rot = ggml_cont(ctx, ggml_reshape_3d(ctx, x_rot, d_model, T, 1));

    // Wq, Wk on the rotated input; Wv on the original x.
    ggml_tensor * q = ggml_mul_mat(ctx, b.attn_q_w, x_rot);
    if (b.attn_q_b != nullptr) q = ggml_add(ctx, q, b.attn_q_b);
    ggml_tensor * k = ggml_mul_mat(ctx, b.attn_k_w, x_rot);
    if (b.attn_k_b != nullptr) k = ggml_add(ctx, k, b.attn_k_b);
    ggml_tensor * v = ggml_mul_mat(ctx, b.attn_v_w, x);
    if (b.attn_v_b != nullptr) v = ggml_add(ctx, v, b.attn_v_b);

    // Reshape for SDPA. Target layout: [head_dim, T, n_head, 1] for the
    // flash_attn path; [head_dim, T, n_head, 1] also for the manual path
    // (k will be transposed for the score matmul).
    auto to_attn = [&](ggml_tensor * t) -> ggml_tensor * {
        // t: [d_model, T, 1] -> [head_dim, n_head, T, 1] -> [head_dim, T, n_head, 1]
        t = ggml_reshape_4d(ctx, t, head_dim, n_head, T, 1);
        t = ggml_permute(ctx, t, 0, 2, 1, 3);
        return ggml_cont(ctx, t);
    };
    q = to_attn(q);
    k = to_attn(k);
    v = to_attn(v);

    ggml_tensor * o;
    if (use_flash) {
        o = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr,
                                scale, 0.0f, 0.0f);
        // ggml_flash_attn_ext returns [head_dim, n_head, T, 1].
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont(ctx, o);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr,
                                                  scale, 0.0f);
        ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        o = ggml_mul_mat(ctx, v_t, kq_soft);
    }
    // o: [head_dim, T, n_head, 1] -> [head_dim, n_head, T, 1] -> [d_model, T, 1]
    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, d_model, T, 1);

    // Output projection.
    o = ggml_mul_mat(ctx, b.attn_out_w, o);
    if (b.attn_out_b != nullptr) o = ggml_add(ctx, o, b.attn_out_b);
    return o;
}

// Build one Conformer block with rotary attention. Returns the
// post-norm-out tensor (= the block output).
ggml_tensor * build_block(ggml_context *      ctx,
                          ggml_tensor *       x,
                          ggml_tensor *       positions,
                          const GigaamBlock & b,
                          int                 d_model,
                          int                 n_head,
                          int                 conv_kernel,
                          bool                use_flash,
                          const conf::ConvPolicy & policy)
{
    // Macaron FF1: x + 0.5 * FF(LN(x)).
    x = conf::macaron_ff_residual(ctx, x,
                                   b.norm_ff1_w, b.norm_ff1_b,
                                   b.ff1_lin1_w, b.ff1_lin1_b,
                                   b.ff1_lin2_w, b.ff1_lin2_b);

    // Self-attention residual: x + attn(LN(x)).
    {
        ggml_tensor * y = conf::layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
        y = build_rotary_attn(ctx, y, positions, b, d_model, n_head, use_flash);
        x = ggml_add(ctx, x, y);
    }

    // Conv module residual: x + conv(LN(x)).
    {
        ggml_tensor * y = conf::layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);
        conf::BlockView cv = to_conv_view(b);
        conf::BlockParams cp {};
        cp.d_model        = d_model;
        cp.n_head         = n_head;
        cp.conv_kernel    = conv_kernel;
        cp.kv_type        = GGML_TYPE_F32;
        cp.use_flash      = use_flash;
        cp.policy         = policy;
        cp.conv_norm_type = conf::BlockParams::ConvNormType::LayerNorm;
        y = conf::conv_module(ctx, y, cv, cp);
        x = ggml_add(ctx, x, y);
    }

    // Macaron FF2.
    x = conf::macaron_ff_residual(ctx, x,
                                   b.norm_ff2_w, b.norm_ff2_b,
                                   b.ff2_lin1_w, b.ff2_lin1_b,
                                   b.ff2_lin2_w, b.ff2_lin2_b);

    // Final per-block LayerNorm. NOTE: the reference returns norm_out(x)
    // (not the residual stream); the block output IS the post-LN value.
    x = conf::layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
    return x;
}

} // namespace

EncoderBuild build_encoder_graph(ggml_context *        ctx,
                                 const GigaamWeights & w,
                                 const GigaamHParams & hp,
                                 int                   n_mel_frames,
                                 ggml_type             /*kv_type*/,
                                 const char *          backend_name)
{
    EncoderBuild eb {};
    if (ctx == nullptr || n_mel_frames <= 0) {
        std::fprintf(stderr,
                     "gigaam encoder: invalid arg "
                     "(ctx=%p, n_mel_frames=%d)\n",
                     static_cast<void *>(ctx), n_mel_frames);
        return eb;
    }

    // Mel input handle. ne=[T_mel, n_mels, 1, 1].
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                   n_mel_frames, hp.fe_num_mels);
    if (eb.mel_in == nullptr) {
        std::fprintf(stderr,
                     "gigaam encoder: failed to allocate mel_in tensor\n");
        return eb;
    }
    ggml_set_name(eb.mel_in, "mel.in");
    ggml_set_input(eb.mel_in);

    // Pre-encode: 2 stride-2 conv1d.
    ggml_tensor * x = build_pre_encode(ctx, w.pre_encode, eb.mel_in,
                                       hp.enc_subs_kernel_size,
                                       "enc.pre_encode");
    if (x == nullptr) return eb;
    ggml_set_name(x, "enc.subsample.out");
    eb.dumps.pre_encode_out = x;
    transcribe::debug::mark_tensor_for_dump(x);

    if (w.blocks.empty()) {
        eb.out = x;
        eb.dumps.final_out = x;
        eb.graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(eb.graph, x);
        return eb;
    }

    const int64_t T_enc = x->ne[1];

    // Positions tensor for rotary. Allocate here, fill via
    // ggml_backend_tensor_set after compute alloc.
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_enc);
    ggml_set_name(positions, "enc.positions");
    ggml_set_input(positions);
    eb.dumps.pos_emb = positions; // documentation only

    // Conv policy. GigaAM's depthwise k=5 — Metal direct path works.
    conf::ConvPolicy policy {};
    policy.direct_pw               = conf::detect_direct_pw(backend_name);
    policy.direct_dw_in_block      = true;
    policy.direct_dw_in_pre_encode = false;
    policy.causal_pre_encode       = false;

    // Use flash attention on GPU backends; manual path on CPU.
    const bool use_flash = (backend_name != nullptr &&
                            std::string(backend_name).find("CPU") == std::string::npos);

    const int n_layers = hp.enc_n_layers;
    eb.dumps.all_block_outs.reserve(n_layers);
    eb.dumps.mid_block_idx  = n_layers / 2 - 1;     // matches dumper (8 for 16-layer)
    eb.dumps.last_block_idx = n_layers - 1;

    for (int i = 0; i < n_layers; ++i) {
        x = build_block(ctx, x, positions, w.blocks[i],
                        hp.enc_d_model, hp.enc_n_heads,
                        hp.enc_conv_kernel, use_flash, policy);

        eb.dumps.all_block_outs.push_back(x);

        if (i == 0) {
            ggml_set_name(x, "enc.block.0.out");
            eb.dumps.block0_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        if (i == eb.dumps.mid_block_idx) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            ggml_set_name(x, nm);
            eb.dumps.mid_block_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        if (i == eb.dumps.last_block_idx) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            ggml_set_name(x, nm);
            eb.dumps.last_block_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
    }

    // x has ne=[d_model, T_enc, 1] = PyTorch [B, T, D]. The RNN-T / CTC
    // decoder consumes this layout directly (matches the reference's
    // `encoded.transpose(1, 2)` for the joint network). Expose it
    // before the final transpose.
    ggml_set_name(x, "rnnt.encoded");
    eb.dumps.rnnt_encoded = x;
    transcribe::debug::mark_tensor_for_dump(x);

    // The reference ConformerEncoder.forward returns
    // `audio_signal.transpose(1, 2)` — [B, T, D] -> [B, D, T]. Apply
    // the final transpose to match the reference enc.out byte layout.
    // CRITICAL: do NOT wrap in a reshape view afterwards; ggml_set_output
    // on a view doesn't preserve the materialized cont buffer (verified
    // experimentally during M2 bring-up of pre_encode).
    ggml_tensor * enc_out_t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_set_name(enc_out_t, "enc.out");
    eb.out = enc_out_t;
    eb.dumps.final_out = enc_out_t;
    transcribe::debug::mark_tensor_for_dump(enc_out_t);

    eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    ggml_build_forward_expand(eb.graph, eb.out);
    return eb;
}

} // namespace transcribe::gigaam
