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
//                         (NHWC convention: H=T, W=F). The 3x3 conv
//                         kernel is not spatially symmetric so the
//                         data must be presented with W=F, H=T.
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

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

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

    // FF1.
    v.norm_ff1_w = b.norm_ff1_w;
    v.norm_ff1_b = b.norm_ff1_b;
    v.ff1_lin1_w = b.ff1_lin1_w;
    v.ff1_lin1_b = b.ff1_lin1_b; // null when use_bias=false
    v.ff1_lin2_w = b.ff1_lin2_w;
    v.ff1_lin2_b = b.ff1_lin2_b;

    // Self-attention. Q/K/V/out can carry biases when use_bias=true
    // (1.1B / rnnt / ctc variants); v2/v3 ship without. linear_pos is
    // bias-free in NeMo regardless.
    v.norm_attn_w = b.norm_attn_w;
    v.norm_attn_b = b.norm_attn_b;
    v.attn_q_w    = b.attn_q_w;   v.attn_q_b   = b.attn_q_b;
    v.attn_k_w    = b.attn_k_w;   v.attn_k_b   = b.attn_k_b;
    v.attn_v_w    = b.attn_v_w;   v.attn_v_b   = b.attn_v_b;
    v.attn_out_w  = b.attn_out_w; v.attn_out_b = b.attn_out_b;
    v.attn_pos_w  = b.attn_pos_w;
    v.attn_pos_u  = b.attn_pos_u;
    v.attn_pos_v  = b.attn_pos_v;

    // Conv module.
    v.norm_conv_w         = b.norm_conv_w;
    v.norm_conv_b         = b.norm_conv_b;
    v.conv_pw1_w          = b.conv_pw1_w;
    v.conv_pw1_b          = b.conv_pw1_b;
    v.conv_dw_w           = b.conv_dw_w;
    v.conv_dw_b           = b.conv_dw_b;
    v.conv_pw2_w          = b.conv_pw2_w;
    v.conv_pw2_b          = b.conv_pw2_b;
    v.conv_bn_fused_scale = b.conv_bn_fused_scale;
    v.conv_bn_fused_bias  = b.conv_bn_fused_bias;
    // For the LayerNorm path (streaming variants) the GGUF stores the
    // affine scale/bias under the same conv.bn.weight/bias tensor names
    // (NeMo keeps the Python attribute named `batch_norm` even when the
    // module is swapped to LayerNorm). Point conv_ln_* at the raw
    // tensors; the conformer block only reads them when
    // params.conv_norm_type == LayerNorm.
    v.conv_ln_w           = b.conv_bn_w;
    v.conv_ln_b           = b.conv_bn_b;

    // FF2.
    v.norm_ff2_w = b.norm_ff2_w;
    v.norm_ff2_b = b.norm_ff2_b;
    v.ff2_lin1_w = b.ff2_lin1_w;
    v.ff2_lin1_b = b.ff2_lin1_b;
    v.ff2_lin2_w = b.ff2_lin2_w;
    v.ff2_lin2_b = b.ff2_lin2_b;

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
        transcribe::debug::mark_tensor_for_dump(t);
    } else if (std::strcmp(tag, "after_attn") == 0) {
        sink->dumps->block0_after_attn = t;
        transcribe::debug::mark_tensor_for_dump(t);
    } else if (std::strcmp(tag, "after_conv") == 0) {
        sink->dumps->block0_after_conv = t;
        transcribe::debug::mark_tensor_for_dump(t);
    } else if (std::strcmp(tag, "after_ff2") == 0) {
        sink->dumps->block0_after_ff2 = t;
        transcribe::debug::mark_tensor_for_dump(t);
    } else if (std::strcmp(tag, "out") == 0) {
        sink->dumps->block0_out = t;
        transcribe::debug::mark_tensor_for_dump(t);
    }
}

// Generalized sub-block observer for blocks 1..N-1, gated by
// TRANSCRIBE_DUMP_SUB_BLOCKS=<comma-separated indices>. Names tensors
// `enc.block.<i>.{ff1,attn,conv,ff2,out}` (mapping the helper's "after_*"
// tags to short names matching the reference dumper). The "out" tag
// here corresponds to the post-norm-out tensor — i.e. block.<i>.out.
struct SubBlockSink {
    int            block_idx;
    EncoderDumps * dumps;
};

void parakeet_subblock_observer_cb(void *        user,
                                   const char *  tag,
                                   ggml_tensor * t)
{
    auto * sink = static_cast<SubBlockSink *>(user);
    if (sink == nullptr || sink->dumps == nullptr || t == nullptr) return;

    const char * short_tag = tag;
    if (std::strcmp(tag, "after_ff1") == 0)  short_tag = "ff1";
    else if (std::strcmp(tag, "after_attn") == 0) short_tag = "attn";
    else if (std::strcmp(tag, "after_conv") == 0) short_tag = "conv";
    else if (std::strcmp(tag, "after_ff2") == 0)  short_tag = "ff2";
    else if (std::strcmp(tag, "out") == 0) {
        // Skip "out" — block.<i>.out is already covered by the
        // mid_block_out / last_block_out / all_block_outs paths.
        return;
    }

    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "enc.block.%d.%s",
                  sink->block_idx, short_tag);
    ggml_set_name(t, name_buf);
    transcribe::debug::mark_tensor_for_dump(t);
    sink->dumps->sub_block_dumps.emplace_back(std::string(name_buf), t);
}

// Parse "0,12,22,23" into a sorted unique vector of block indices.
// Returns empty if the env var is unset or empty.
std::vector<int> parse_sub_block_env(int n_layers)
{
    std::vector<int> out;
    const char * raw = std::getenv("TRANSCRIBE_DUMP_SUB_BLOCKS");
    if (raw == nullptr || *raw == '\0') return out;
    const char * p = raw;
    while (*p != '\0') {
        // Skip leading separators.
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        if (*p == '\0') break;
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;  // no progress — malformed
        if (v >= 0 && v < n_layers) {
            out.push_back(static_cast<int>(v));
        }
        p = end;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

} // namespace

EncoderBuild build_encoder_graph(ggml_context *                      ctx,
                                 const ParakeetWeights &             w,
                                 const ParakeetHParams &             hp,
                                 int                                 n_mel_frames,
                                 ggml_type                           kv_type,
                                 const char *                        backend_name,
                                 const BufferedStreamMaskOverride *  buf_mask)
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
    // Cache-aware streaming variants (NeMo `causal_downsampling=true`)
    // use CausalConv2D for the pre-encode subsample stack (left=k-1,
    // right=stride-1 pad on both spatial axes). We infer this from the
    // attention style: ChunkedLimited (cache-aware) implies causal
    // subsample; Regular (offline) and ChunkedLimitedWithRc (buffered
    // streaming) both use the standard non-causal symmetric pad. The
    // pre-encode kernel (k=3) and the conformer conv-module kernel
    // (k=9) are independent — hp.enc_conv_context_{left,right} are
    // for the latter and don't say anything about the former.
    policy.causal_pre_encode =
        (hp.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimited);

    EncoderBuild eb {};

    if (ctx == nullptr || n_mel_frames <= 0) {
        std::fprintf(stderr,
                     "parakeet encoder: invalid arg "
                     "(ctx=%p, n_mel_frames=%d)\n",
                     static_cast<void *>(ctx), n_mel_frames);
        return eb;
    }

    // We support subsampling_factor=8 (every published Parakeet variant)
    // and any n_mels divisible by the subsampling factor on the freq axis
    // (3 stride-2 convs, kernel=3, padding=1). All current variants ship
    // n_mels=80 or 128; both pass cleanly. If a future variant ships a
    // factor != 8, build_pre_encode would need a structural rework, so
    // we fail loudly here.
    if (hp.enc_subsampling_factor != 8) {
        std::fprintf(stderr,
                     "parakeet encoder: unsupported subsampling_factor=%d "
                     "(only 8 implemented)\n",
                     hp.enc_subsampling_factor);
        return eb;
    }
    if (hp.fe_num_mels <= 0 || (hp.fe_num_mels % hp.enc_subsampling_factor) != 0) {
        std::fprintf(stderr,
                     "parakeet encoder: n_mels=%d not divisible by "
                     "subsampling_factor=%d\n",
                     hp.fe_num_mels, hp.enc_subsampling_factor);
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
    transcribe::debug::mark_tensor_for_dump(x);

    // ----- xscaling ---------------------------------------------------
    //
    // NeMo's RelPositionalEncoding.forward applies x = x * sqrt(d_model)
    // when its `xscaling` flag is true. This multiplication happens AFTER
    // pre_encode and BEFORE the first conformer block — i.e. it
    // operates on the running residual, not just on the pos_emb input.
    // ctc-0.6b/1.1b, rnnt-0.6b/1.1b, and unified-en-0.6b ship with
    // xscaling=True; v2/v3 and the TDT/TDT_CTC variants ship with
    // xscaling=False.
    //
    // The same shape comes out (the scale factor is applied uniformly
    // along d_model). LayerNorms in every sub-block normalize the
    // scale away when computing FF/attn/conv outputs, so the only
    // place the scale matters is the residual carried between
    // sub-blocks and the running residual into norm_out at the end of
    // each block. Without this multiplication on a True-flagged model
    // the residual is sqrt(d_model)x smaller, the corrections are the
    // same magnitude (because they go through LN first), and the
    // residual+correction mix is structurally different — the
    // accumulated drift past 24 blocks is enough to produce empty
    // transcripts on unified-en-0.6b and ~0.04% WER drift on ctc-0.6b
    // / rnnt-0.6b before this fix landed.
    if (hp.enc_xscaling) {
        const float scale = std::sqrt(static_cast<float>(hp.enc_d_model));
        x = ggml_scale(ctx, x, scale);
        x = conf::named(x, "enc.pre_encode.xscaled");
    }

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
        // matching the reference `enc.pos_emb` dump.
        //
        // Local attention (Regular style with both att_context_{left,right}
        // >= 0) shortens pos_emb to (left+right+1) so the buffer covers
        // exactly the attended relative-position range — matches NeMo's
        // LocalAttRelPositionalEncoding. ChunkedLimited style keeps pos_emb
        // at the full 2T-1 and uses a separate attention-position mask
        // (built below).
        // ChunkedLimited (2-tuple cache-aware) always engages the mask.
        // ChunkedLimitedWithRc (3-tuple buffered streaming) only engages
        // when the caller passes a non-null buf_mask override — offline
        // load of a unified-en GGUF runs the encoder with full attention
        // even though the style declares chunked_limited_with_rc.
        const bool is_chunked =
            (hp.enc_att_context_style ==
                 ParakeetHParams::AttContextStyle::ChunkedLimited) ||
            (hp.enc_att_context_style ==
                 ParakeetHParams::AttContextStyle::ChunkedLimitedWithRc
             && buf_mask != nullptr);
        const bool is_local_pe =
            (!is_chunked) &&
            (hp.enc_att_context_left >= 0 && hp.enc_att_context_right >= 0);
        const int64_t pos_len = is_local_pe
            ? static_cast<int64_t>(hp.enc_att_context_left
                                   + hp.enc_att_context_right + 1)
            : (2 * T_enc - 1);
        eb.pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           hp.enc_d_model, pos_len);
        ggml_set_name(eb.pos_emb_in, "pos_emb.in");
        ggml_set_input(eb.pos_emb_in);

        // ChunkedLimited mask. [T_enc, T_enc, 1, 1] F32; the driver
        // fills it host-side after the compute buffer is allocated.
        // Broadcasts across heads inside rel_pos_mhsa.
        ggml_tensor * chunked_mask_in = nullptr;
        if (is_chunked) {
            chunked_mask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                 T_enc, T_enc, 1, 1);
            ggml_set_name(chunked_mask_in, "attn.chunked_mask.in");
            ggml_set_input(chunked_mask_in);
            eb.chunked_mask_in = chunked_mask_in;
        }

        conf::BlockParams bparams {};
        bparams.d_model           = hp.enc_d_model;
        bparams.n_head            = hp.enc_n_heads;
        bparams.conv_kernel       = hp.enc_conv_kernel;
        bparams.kv_type           = kv_type;
        bparams.use_flash         = true;
        bparams.policy            = policy;
        bparams.att_context_left  = hp.enc_att_context_left;
        bparams.att_context_right = hp.enc_att_context_right;
        bparams.att_context_style = is_chunked
            ? conf::BlockParams::AttContextStyle::ChunkedLimited
            : conf::BlockParams::AttContextStyle::Regular;
        bparams.attn_chunked_mask = chunked_mask_in;
        bparams.conv_context_left  = hp.enc_conv_context_left;
        bparams.conv_context_right = hp.enc_conv_context_right;
        bparams.conv_norm_type = (hp.enc_conv_norm_type
                                  == ParakeetHParams::ConvNormType::LayerNorm)
            ? conf::BlockParams::ConvNormType::LayerNorm
            : conf::BlockParams::ConvNormType::BatchNorm;

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

        // Blocks 1..N-1: no observer. Mark the mid- and last-layer
        // outputs for dump so the harness can spot-check them; every
        // other block stays anonymous. Mid + last indices scale with
        // n_layers so the dump points match the per-variant oracle
        // (0/n_half/n-1). For 24-layer v2/v3 that's 0/12/23; for
        // 42-layer 1.1B it's 0/21/41; for 17-layer tdt_ctc-110m it's
        // 0/8/16. We only mark for dump here — the actual file name
        // is synthesized in model.cpp from the saved layer index,
        // because conf::named's trailing "enc.final" rename would
        // otherwise clobber the last-block name in place.
        const int n_layers   = static_cast<int>(w.blocks.size());
        const int mid_layer  = n_layers / 2;
        const int last_layer = n_layers - 1;
        eb.dumps.mid_block_idx  = mid_layer;
        eb.dumps.last_block_idx = last_layer;
        eb.dumps.all_block_outs.assign(n_layers, nullptr);
        eb.dumps.all_block_outs[0] = eb.dumps.block0_out;
        // Per-block dump opt-in for the layer-by-layer divergence
        // bisect. The reference dumper supports `--blocks 0,1,...,N-1`;
        // setting TRANSCRIBE_DUMP_ALL_BLOCKS=1 makes the C++ side
        // dump every block to enable a 1:1 frame-by-frame compare.
        const bool dump_all_blocks =
            std::getenv("TRANSCRIBE_DUMP_ALL_BLOCKS") != nullptr;
        if (dump_all_blocks && eb.dumps.block0_out != nullptr) {
            transcribe::debug::mark_tensor_for_dump(eb.dumps.block0_out);
        }
        // Sub-block observation gate. TRANSCRIBE_DUMP_SUB_BLOCKS=
        // "12,22,23" installs the sub-block observer on each listed
        // index >= 1 (block 0 already has its dedicated observer above).
        // Stored in a local set for O(1) lookup in the loop.
        const std::vector<int> sub_blocks = parse_sub_block_env(n_layers);
        // Persistent sinks — one per requested block, kept alive for the
        // duration of the encoder build. The observer pointer captures
        // their address.
        std::vector<SubBlockSink> sub_sinks;
        sub_sinks.reserve(sub_blocks.size());
        for (int idx : sub_blocks) sub_sinks.push_back(SubBlockSink{idx, &eb.dumps});
        for (int i = 1; i < n_layers; ++i) {
            // If this block is requested for sub-block dumping, install
            // the generalized observer; otherwise run plain.
            const auto it = std::lower_bound(sub_blocks.begin(),
                                             sub_blocks.end(), i);
            const bool with_obs =
                (it != sub_blocks.end() && *it == i);
            if (with_obs) {
                const size_t sink_idx =
                    static_cast<size_t>(it - sub_blocks.begin());
                conf::BlockObserver obs {};
                obs.on_point = parakeet_subblock_observer_cb;
                obs.user     = &sub_sinks[sink_idx];
                x = conf::build_conformer_block(ctx, x, eb.pos_emb_in,
                                                to_view(w.blocks[i]),
                                                bparams, &obs);
            } else {
                x = conf::build_conformer_block(ctx, x, eb.pos_emb_in,
                                                to_view(w.blocks[i]), bparams);
            }
            eb.dumps.all_block_outs[i] = x;
            if (dump_all_blocks) {
                transcribe::debug::mark_tensor_for_dump(x);
            }
            if (i == mid_layer) {
                eb.dumps.mid_block_out = x;
                transcribe::debug::mark_tensor_for_dump(x);
            }
            if (i == last_layer) {
                eb.dumps.last_block_out = x;
                transcribe::debug::mark_tensor_for_dump(x);
            }
        }
    }

    // Final encoder output. With all 24 blocks wired, this is now
    // the true encoder exit point (= block 23's norm_out output) and
    // the comparator's `enc.final` row will compare against the
    // reference's true encoder output.
    eb.dumps.final_out = x;
    x = conf::named(x, "enc.final");
    transcribe::debug::mark_tensor_for_dump(x);

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

// ---------------------------------------------------------------------------
// Streaming encoder graph (cache-aware, per-chunk feed)
// ---------------------------------------------------------------------------
//
// Mirrors build_encoder_graph but:
//   - Operates on a mel chunk (n_mel_chunk_frames), not the whole audio.
//   - Slices off `drop_extra_pre_encoded` post-subsample frames.
//   - Threads per-layer cache_last_channel + cache_last_time tensors
//     through BlockParams, and emits per-layer cache outputs into
//     fresh compute_ctx tensors that the driver reads back.
//   - Sizes pos_emb and chunked_mask for T_virtual = T_cache + T_q_new
//     (the per-block attention runs on the cache+new union).
//
// Shapes:
//   eb.mel_in           [n_mel_chunk_frames, n_mels, 1, 1]
//   eb.pos_emb_in       [d_model, 2*T_virtual - 1, 1, 1]
//   eb.chunked_mask_in  [T_virtual, T_virtual, 1, 1]
//   eb.out              [d_model, T_q_new, 1, 1]
//   cache_io.channel_out[i]   [d_model, T_cache, 1, 1]   (fresh)
//   cache_io.time_out[i]      [k_minus_1, d_model, 1, 1] (fresh)
//
// All cache input tensors (cache_io.channel_in / time_in) must come
// from the persistent ParakeetStreamingCaches buffer and have the
// shapes above. The graph DOES treat them as inputs (the scheduler
// walks back from cpy ops and eb.out to find them).
EncoderBuild build_encoder_graph_streaming(
    ggml_context *          ctx,
    const ParakeetWeights & w,
    const ParakeetHParams & hp,
    int                     n_mel_chunk_frames,
    int                     drop_extra_pre_encoded,
    StreamingEncoderCacheIO & cache_io,
    ggml_type               kv_type,
    const char *            backend_name)
{
    conf::ConvPolicy policy {};
    policy.direct_pw               = conf::detect_direct_pw(backend_name);
    policy.direct_dw_in_block      = detect_direct_dw_in_block(backend_name);
    policy.direct_dw_in_pre_encode = false;
    // See the offline analog in build_encoder_graph: causal pre-encode
    // is the cache-aware streaming convention only.
    policy.causal_pre_encode =
        (hp.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimited);

    EncoderBuild eb {};

    if (ctx == nullptr || n_mel_chunk_frames <= 0) {
        std::fprintf(stderr,
                     "parakeet streaming encoder: invalid arg "
                     "(ctx=%p, n_mel_chunk_frames=%d)\n",
                     static_cast<void *>(ctx), n_mel_chunk_frames);
        return eb;
    }

    if (hp.enc_subsampling_factor != 8) {
        std::fprintf(stderr,
                     "parakeet streaming encoder: unsupported "
                     "subsampling_factor=%d (only 8 implemented)\n",
                     hp.enc_subsampling_factor);
        return eb;
    }
    if (hp.enc_att_context_style !=
            ParakeetHParams::AttContextStyle::ChunkedLimited)
    {
        std::fprintf(stderr,
                     "parakeet streaming encoder: requires "
                     "att_context_style=ChunkedLimited\n");
        return eb;
    }

    const int n_layers = static_cast<int>(w.blocks.size());
    if (static_cast<int>(cache_io.channel_in.size()) != n_layers ||
        static_cast<int>(cache_io.time_in.size())    != n_layers)
    {
        std::fprintf(stderr,
                     "parakeet streaming encoder: cache_io.in vectors "
                     "must be sized to n_layers=%d (got channel=%zu, "
                     "time=%zu)\n",
                     n_layers, cache_io.channel_in.size(),
                     cache_io.time_in.size());
        return eb;
    }

    // ----- Mel chunk input -----
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                   n_mel_chunk_frames, hp.fe_num_mels);
    ggml_set_name(eb.mel_in, "stream.mel.in");
    ggml_set_input(eb.mel_in);

    // ----- Pre-encode + (optional) xscaling -----
    ggml_tensor * x = conf::build_pre_encode(ctx, to_view(w.pre_encode),
                                             eb.mel_in, policy,
                                             /*name_prefix=*/"stream.enc.pre_encode",
                                             /*error_tag=*/"parakeet stream");
    if (x == nullptr) return eb;

    if (hp.enc_xscaling) {
        const float scale = std::sqrt(static_cast<float>(hp.enc_d_model));
        x = ggml_scale(ctx, x, scale);
    }

    // x ne = [d_model, T_pre_enc]. Slice off the first
    // drop_extra_pre_encoded frames (NeMo's mechanism to handle the
    // 8x-subsample overlap after the mel-history prepend).
    if (drop_extra_pre_encoded > 0) {
        const int64_t T_pre = x->ne[1];
        if (drop_extra_pre_encoded >= T_pre) {
            std::fprintf(stderr,
                         "parakeet streaming encoder: drop_extra=%d >= "
                         "T_pre_enc=%lld\n",
                         drop_extra_pre_encoded, (long long)T_pre);
            return eb;
        }
        x = ggml_view_2d(ctx, x,
                         x->ne[0], T_pre - drop_extra_pre_encoded,
                         x->nb[1],
                         drop_extra_pre_encoded * x->nb[1]);
        x = ggml_cont(ctx, x);
    }

    const int64_t T_q_new  = x->ne[1];
    const int64_t T_cache  = cache_io.channel_in[0]->ne[1];
    const int64_t T_virt   = T_cache + T_q_new;

    // ----- Pos_emb sized for T_virtual -----
    const int64_t pos_len = 2 * T_virt - 1;
    eb.pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                       hp.enc_d_model, pos_len);
    ggml_set_name(eb.pos_emb_in, "stream.pos_emb.in");
    ggml_set_input(eb.pos_emb_in);

    // Chunked mask: [T_virtual, T_virtual]. Built host-side.
    eb.chunked_mask_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                            T_virt, T_virt, 1, 1);
    ggml_set_name(eb.chunked_mask_in, "stream.attn.chunked_mask.in");
    ggml_set_input(eb.chunked_mask_in);

    // ----- Create the graph BEFORE allocating cache_out tensors, so
    // the helpers can ggml_build_forward_expand their cpy nodes onto
    // it. -----
    eb.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);
    if (eb.graph == nullptr) return eb;

    // ----- BlockParams (per-block, but only streaming_*_in/_out and
    // streaming_T_q_new change per layer) -----
    conf::BlockParams bparams {};
    bparams.d_model           = hp.enc_d_model;
    bparams.n_head            = hp.enc_n_heads;
    bparams.conv_kernel       = hp.enc_conv_kernel;
    bparams.kv_type           = kv_type;
    bparams.use_flash         = true;
    bparams.policy            = policy;
    bparams.att_context_left  = hp.enc_att_context_left;
    bparams.att_context_right = hp.enc_att_context_right;
    bparams.att_context_style = conf::BlockParams::AttContextStyle::ChunkedLimited;
    bparams.attn_chunked_mask = eb.chunked_mask_in;
    bparams.conv_context_left  = hp.enc_conv_context_left;
    bparams.conv_context_right = hp.enc_conv_context_right;
    bparams.conv_norm_type = (hp.enc_conv_norm_type
                              == ParakeetHParams::ConvNormType::LayerNorm)
        ? conf::BlockParams::ConvNormType::LayerNorm
        : conf::BlockParams::ConvNormType::BatchNorm;
    bparams.streaming_graph    = eb.graph;
    bparams.streaming_T_q_new  = static_cast<int>(T_q_new);

    cache_io.channel_out.assign(n_layers, nullptr);
    cache_io.time_out.assign(n_layers, nullptr);
    const int k_minus_1 = static_cast<int>(cache_io.time_in[0]->ne[0]);

    for (int i = 0; i < n_layers; ++i) {
        // Per-layer cache I/O. The "in" tensors are from the
        // persistent cache buffer (passed in by the driver). The
        // "out" tensors are fresh in compute_ctx; the helpers fill
        // them via ggml_cpy and the driver reads them after compute.
        ggml_tensor * cache_ch_out = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, hp.enc_d_model, T_cache);
        ggml_tensor * cache_tm_out = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, k_minus_1, hp.enc_d_model);
        if (cache_ch_out == nullptr || cache_tm_out == nullptr) return eb;
        char nm[64];
        std::snprintf(nm, sizeof(nm), "stream.cache_ch_out.%d", i);
        ggml_set_name(cache_ch_out, nm);
        std::snprintf(nm, sizeof(nm), "stream.cache_tm_out.%d", i);
        ggml_set_name(cache_tm_out, nm);
        ggml_set_output(cache_ch_out);
        ggml_set_output(cache_tm_out);

        bparams.streaming_channel_in  = cache_io.channel_in[i];
        bparams.streaming_channel_out = cache_ch_out;
        bparams.streaming_time_in     = cache_io.time_in[i];
        bparams.streaming_time_out    = cache_tm_out;

        x = conf::build_conformer_block(ctx, x, eb.pos_emb_in,
                                        to_view(w.blocks[i]),
                                        bparams,
                                        /*obs=*/nullptr);

        cache_io.channel_out[i] = cache_ch_out;
        cache_io.time_out[i]    = cache_tm_out;
    }

    eb.out = x;
    ggml_set_output(eb.out);
    ggml_build_forward_expand(eb.graph, eb.out);
    return eb;
}

} // namespace transcribe::parakeet
