// arch/parakeet/encoder.cpp - Parakeet Conformer encoder graph builder.
//
// Thin glue over the shared Conformer helpers in src/conformer/. The
// per-op helpers (layer_norm, f32-friendly conv helpers, rel_shift,
// conv_module, rel_pos_mhsa, build_conformer_block, build_pre_encode)
// now live in transcribe::conformer — see src/conformer/conformer.{h,cpp}
// for the full implementations and the Metal backstory on the
// conv_*_f32 helpers.
//
// Parakeet-specific pieces kept in this file:
//   - detect_direct_dw (different policy than Cohere)
//   - to_view() helpers that project ParakeetPreEncode / ParakeetBlock
//     onto the shared nullable-pointer views. Parakeet's conformer
//     blocks ship without biases on the FFN/attention/conv-module
//     linear layers, so those slots in BlockView stay nullptr — but
//     pre_encode has real conv biases (conv0_b..conv6_b, out_b) and
//     those ARE copied through by to_view(ParakeetPreEncode).
//   - build_encoder_graph orchestration — every block goes through
//     conf::build_conformer_block; block 0 passes a BlockObserver
//     that names intermediates and stashes them in the EncoderDumps
//     struct so the 3b-3e bring-up harness can diff every sub-step
//
// Reference: parakeet-mlx
//   /tmp/parakeet-mlx/parakeet_mlx/conformer.py:206-328  (DwStridingSubsampling)
//   /tmp/parakeet-mlx/parakeet_mlx/conformer.py:392-423  (Conformer.__call__)
//
// Layout cheat sheet:
//
//   MelFrontend output  : row-major [n_mels=128, n_frames=T_mel] f32
//   ggml mel_in tensor  : ne=[T_mel, 128, 1, 1] f32  (matches the
//                         row-major buffer byte-for-byte)
//   conv input form     : [W=F, H=T, IC, N] per ggml_conv_2d. The
//                         catalog kernels are stored in PyTorch OIHW
//                         order which translates to ggml ne
//                         [KW, KH, IC, OC] where KW is aligned with
//                         the FREQ axis and KH with the TIME axis
//                         (MLX NHWC convention: H=T, W=F). The 3x3
//                         conv kernel is not spatially symmetric so
//                         the data must be presented with W=F, H=T.
//
// After 3 stride-2 convs (kernel=3, padding=1), W and H shrink by
// floor((d + 2 - 3)/2) + 1 per layer:
//   F (ggml W) =  128 ->  64 ->  32 -> 16
//   T (ggml H) = 1101 -> 551 -> 276 -> 138    (jfk.wav, 11 s)
// So the post-conv tensor has ne = [F'=16, T_enc=138, channels=256, 1].

#include "encoder.h"

#include "weights.h"

#include "conformer/conformer.h"
#include "transcribe-debug.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace transcribe::parakeet {

namespace {

namespace conf = transcribe::conformer;

// ----- Per-family depthwise dispatch policy -----------------------
//
// Parakeet uses the direct conv_2d_dw path on every backend for the
// in-block depthwise conv (Vulkan: native kernel; Metal/CPU: the
// direct path is faster than the 31x im2col expansion). The
// pre_encode site historically hardcoded the im2col f32 helper
// regardless of the in-block flag, so direct_dw_in_pre_encode stays
// false here to preserve that behavior exactly. Env vars map onto
// both sites because Parakeet has no history of splitting them.
bool detect_direct_dw_in_block(const char * backend) {
    const char * env = std::getenv("TRANSCRIBE_CONV_DIRECT_DW");
    if (env != nullptr) return true;  // user override
    env = std::getenv("TRANSCRIBE_CONV_NO_DIRECT_DW");
    if (env != nullptr) return false; // user override
    // Vulkan and CPU have native CONV_2D_DW; Metal does too via the
    // direct path in this version. CPU also benefits from direct dw
    // (avoids the 31x im2col expansion).
    (void)backend;
    return true;
}

// ----- Views ------------------------------------------------------

conf::PreEncodeView to_view(const ParakeetPreEncode & pe) {
    conf::PreEncodeView v;
    v.conv0_w = pe.conv0_w; v.conv0_b = pe.conv0_b;
    v.conv2_w = pe.conv2_w; v.conv2_b = pe.conv2_b;
    v.conv3_w = pe.conv3_w; v.conv3_b = pe.conv3_b;
    v.conv5_w = pe.conv5_w; v.conv5_b = pe.conv5_b;
    v.conv6_w = pe.conv6_w; v.conv6_b = pe.conv6_b;
    v.out_w   = pe.out_w;   v.out_b   = pe.out_b;
    return v;
}

conf::BlockView to_view(const ParakeetBlock & b) {
    conf::BlockView v;

    // FF1 (no biases on the linear layers).
    v.norm_ff1_w = b.norm_ff1_w;
    v.norm_ff1_b = b.norm_ff1_b;
    v.ff1_lin1_w = b.ff1_lin1_w;
    v.ff1_lin1_b = nullptr;
    v.ff1_lin2_w = b.ff1_lin2_w;
    v.ff1_lin2_b = nullptr;

    // Self-attention (no biases on Q/K/V/out in Parakeet).
    v.norm_attn_w = b.norm_attn_w;
    v.norm_attn_b = b.norm_attn_b;
    v.attn_q_w    = b.attn_q_w;   v.attn_q_b   = nullptr;
    v.attn_k_w    = b.attn_k_w;   v.attn_k_b   = nullptr;
    v.attn_v_w    = b.attn_v_w;   v.attn_v_b   = nullptr;
    v.attn_out_w  = b.attn_out_w; v.attn_out_b = nullptr;
    v.attn_pos_w  = b.attn_pos_w;
    v.attn_pos_u  = b.attn_pos_u;
    v.attn_pos_v  = b.attn_pos_v;

    // Conv module (no biases on pw1/dw/pw2 in Parakeet).
    v.norm_conv_w         = b.norm_conv_w;
    v.norm_conv_b         = b.norm_conv_b;
    v.conv_pw1_w          = b.conv_pw1_w;
    v.conv_pw1_b          = nullptr;
    v.conv_dw_w           = b.conv_dw_w;
    v.conv_dw_b           = nullptr;
    v.conv_pw2_w          = b.conv_pw2_w;
    v.conv_pw2_b          = nullptr;
    v.conv_bn_fused_scale = b.conv_bn_fused_scale;
    v.conv_bn_fused_bias  = b.conv_bn_fused_bias;

    // FF2 (no biases).
    v.norm_ff2_w = b.norm_ff2_w;
    v.norm_ff2_b = b.norm_ff2_b;
    v.ff2_lin1_w = b.ff2_lin1_w;
    v.ff2_lin1_b = nullptr;
    v.ff2_lin2_w = b.ff2_lin2_w;
    v.ff2_lin2_b = nullptr;

    v.norm_out_w = b.norm_out_w;
    v.norm_out_b = b.norm_out_b;
    return v;
}

// ----- Block 0 observer -------------------------------------------
//
// Drives the 3b-3e bring-up harness. For each of the five canonical
// points build_conformer_block fires on, we name the tensor
// `enc.block.0.<tag>` (matching the old hand-built block's naming
// exactly, so existing graph-inspection workflows keep working) and
// stash the pointer in the right EncoderDumps slot so Parakeet::run's
// try_dump loop finds it.
//
// NOTE: the dump FILE names in model.cpp's try_dump table
// (`enc.block.0.ff1`, etc.) are independent of the ggml names here
// (`enc.block.0.after_ff1`, etc.). That pre-dates the observer —
// both names existed on the hand-built block as well. Preserved
// as-is to avoid any behavioral change during the refactor.
struct Block0DumpSink {
    EncoderDumps * dumps;
};

void parakeet_block0_observer_cb(void *        user,
                                 const char *  tag,
                                 ggml_tensor * t)
{
    auto * sink = static_cast<Block0DumpSink *>(user);
    if (sink == nullptr || sink->dumps == nullptr || t == nullptr) return;

    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "enc.block.0.%s", tag);
    ggml_set_name(t, name_buf);

    if (std::strcmp(tag, "after_ff1") == 0) {
        sink->dumps->block0_after_ff1 = t;
    } else if (std::strcmp(tag, "after_attn") == 0) {
        sink->dumps->block0_after_attn = t;
    } else if (std::strcmp(tag, "after_conv") == 0) {
        sink->dumps->block0_after_conv = t;
    } else if (std::strcmp(tag, "after_ff2") == 0) {
        sink->dumps->block0_after_ff2 = t;
    } else if (std::strcmp(tag, "out") == 0) {
        sink->dumps->block0_out = t;
    }
}

} // namespace

EncoderBuild build_encoder_graph(ggml_context *          ctx,
                                 const ParakeetWeights & w,
                                 const ParakeetHParams & hp,
                                 int                     n_mel_frames,
                                 ggml_type               kv_type,
                                 const char *            backend_name)
{
    // Per-family dispatch policy. Parakeet keeps the historical split
    // where the in-block depthwise uses the direct path on every
    // backend but the pre_encode depthwise uses the f32-friendly
    // im2col helper unconditionally. See detect_direct_dw_in_block
    // above and the ConvPolicy comment in conformer.h.
    conf::ConvPolicy policy {};
    policy.direct_pw               = conf::detect_direct_pw(backend_name);
    policy.direct_dw_in_block      = detect_direct_dw_in_block(backend_name);
    policy.direct_dw_in_pre_encode = false;

    EncoderBuild eb {};

    if (ctx == nullptr || n_mel_frames <= 0) {
        std::fprintf(stderr,
                     "parakeet encoder: invalid arg "
                     "(ctx=%p, n_mel_frames=%d)\n",
                     static_cast<void *>(ctx), n_mel_frames);
        return eb;
    }

    // Sub-stage 3a only supports the catalog's locked-in
    // factor=8/n_mels=128 layout. The full encoder will assert this
    // upstream in init_context once step 5 wires the run path; for
    // now we re-check here so the encoder builder is self-contained
    // for sub-stage validation.
    if (hp.enc_subsampling_factor != 8 || hp.fe_num_mels != 128) {
        std::fprintf(stderr,
                     "parakeet encoder: unsupported geometry "
                     "subsampling_factor=%d num_mels=%d "
                     "(only 8/128 implemented)\n",
                     hp.enc_subsampling_factor, hp.fe_num_mels);
        return eb;
    }

    // Mel input handle. ne=[T_mel, n_mels, 1, 1] matches the C++
    // MelFrontend's row-major [n_mels, n_frames] storage byte for
    // byte (see the layout cheat sheet at the top of the file).
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                   n_mel_frames, hp.fe_num_mels);
    if (eb.mel_in == nullptr) {
        std::fprintf(stderr,
                     "parakeet encoder: failed to allocate mel_in tensor\n");
        return eb;
    }
    ggml_set_name(eb.mel_in, "mel.in");
    ggml_set_input(eb.mel_in);

    // Build the pre_encode subgraph.
    ggml_tensor * x = conf::build_pre_encode(ctx, to_view(w.pre_encode),
                                             eb.mel_in, policy,
                                             /*name_prefix=*/"enc.pre_encode",
                                             /*error_tag=*/"parakeet");
    if (x == nullptr) {
        // build_pre_encode already logged the diagnostic.
        return eb;
    }
    eb.dumps.pre_encode_out = x;

    // ----- Conformer blocks -----------------------------------------
    //
    // Every block goes through conf::build_conformer_block — there is
    // no hand-built path anymore. Block 0 passes a BlockObserver that
    // names intermediates and stashes them in eb.dumps so the 3b-3e
    // bring-up harness can diff every sub-step. Blocks 12 and 23 get
    // their final-LN outputs named after return; the loop is plain
    // otherwise so the dump dir doesn't fill with all 24 outputs.

    if (!w.blocks.empty()) {
        // After pre_encode, x ne = [d_model, T_enc, 1, 1]. T_enc is
        // needed by the positional embedding (whose pos_len =
        // 2*T_enc - 1) and by the encoder accuracy harness for shape
        // sanity.
        const int64_t T_enc = x->ne[1];

        // Create the pos_emb input tensor here, once we know T_enc.
        // The driver fills it via ggml_backend_tensor_set after the
        // compute buffer is allocated. ne = [d_model, pos_len, 1, 1]
        // — slow-to-fast shape (numpy) is (pos_len, d_model),
        // matching parakeet-mlx's `enc.pos_emb` dump.
        const int64_t pos_len = 2 * T_enc - 1;
        eb.pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           hp.enc_d_model, pos_len);
        ggml_set_name(eb.pos_emb_in, "pos_emb.in");
        ggml_set_input(eb.pos_emb_in);

        conf::BlockParams bparams {};
        bparams.d_model     = hp.enc_d_model;
        bparams.n_head      = hp.enc_n_heads;
        bparams.conv_kernel = hp.enc_conv_kernel;
        bparams.kv_type     = kv_type;
        bparams.use_flash   = true;
        bparams.policy      = policy;

        // Block 0: run through the shared helper with the dump
        // observer. The observer writes into eb.dumps; no sub-step
        // references are lost vs. the old hand-built path.
        {
            Block0DumpSink sink { &eb.dumps };
            conf::BlockObserver obs {};
            obs.on_point = parakeet_block0_observer_cb;
            obs.user     = &sink;
            x = conf::build_conformer_block(ctx, x, eb.pos_emb_in,
                                            to_view(w.blocks[0]),
                                            bparams, &obs);
        }

        // Blocks 1..N-1: no observer. Name the final-LN outputs of
        // the mid-encoder and last-block spot-check points so the
        // dump harness finds them; every other block stays anonymous.
        for (size_t i = 1; i < w.blocks.size(); ++i) {
            x = conf::build_conformer_block(ctx, x, eb.pos_emb_in,
                                            to_view(w.blocks[i]), bparams);
            if (i == 12) {
                x = conf::named(x, "enc.block.12.out");
                eb.dumps.block12_out = x;
            }
            if (i == 23) {
                x = conf::named(x, "enc.block.23.out");
                eb.dumps.block23_out = x;
            }
        }
    }

    // Final encoder output. With all 24 blocks wired, this is now
    // the true encoder exit point (= block 23's norm_out output) and
    // the comparator's `enc.final` row will compare against
    // parakeet-mlx's true encoder output.
    eb.dumps.final_out = x;
    x = conf::named(x, "enc.final");

    eb.out = x;
    ggml_set_output(eb.out);

    // Build the forward cgraph. ggml_new_graph_custom allocates the
    // cgraph out of the same compute_ctx, so it has to be sized for
    // it. The default GGML_DEFAULT_GRAPH_SIZE (2048 nodes) is too
    // small for the full encoder: each conformer block contributes
    // ~90 ops (FF + attn + conv + FF + LN, including all the
    // permute/cont overhead) and 24 blocks pushes the count past
    // 2200. Use 8192 to leave headroom.
    eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (eb.graph == nullptr) {
        std::fprintf(stderr,
                     "parakeet encoder: ggml_new_graph_custom failed\n");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);

    return eb;
}

} // namespace transcribe::parakeet
