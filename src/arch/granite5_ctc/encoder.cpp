// arch/granite5_ctc/encoder.cpp - see encoder.h for the contract.

#include "encoder.h"

#include "conformer/conformer.h"
#include "ggml.h"
#include "granite_conformer/shaw_attn.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace transcribe::granite5_ctc {

namespace {

// nn.LayerNorm default eps. Not stored in the GGUF.
constexpr float kLayerNormEps = 1e-5f;

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr && name != nullptr) {
        ggml_set_name(t, name);
    }
    return t;
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gamma, ggml_tensor * beta) {
    ggml_tensor * y = ggml_norm(ctx, x, kLayerNormEps);
    y               = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) {
        y = ggml_add(ctx, y, beta);
    }
    return y;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

}  // namespace

// Host-side frontend.

int encoder_frames_for(const Granite5CtcHParams & hp, int64_t n_samples) {
    if (n_samples <= 0 || hp.fe_hop_length <= 0 || hp.fe_stack_factor <= 0) {
        return 0;
    }
    const int64_t stack      = hp.fe_stack_factor;
    const int64_t mel_frames = n_samples / hp.fe_hop_length;
    const int64_t num_frames = stack * ((mel_frames + stack - 1) / stack);
    return static_cast<int>(num_frames / stack);
}

transcribe_status compute_encoder_input(const transcribe::MelFrontend & mel,
                                        const Granite5CtcHParams &      hp,
                                        const float *                   pcm,
                                        int64_t                         n_samples,
                                        int                             n_threads,
                                        std::vector<float> &            out_feats,
                                        int &                           out_t_enc) {
    if (pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int hop   = hp.fe_hop_length;
    const int stack = hp.fe_stack_factor;

    // Frame budget. Reference (_extract_features):
    //   mel_frames         = audio.shape[1] // hop_length
    //   num_frames         = stack * ceil(mel_frames / stack)
    //   num_samples_needed = (num_frames - 1) * hop_length + 1
    //   if shorter: right-pad the WAVEFORM with zeros to that length
    //   mel  = mel_filters(audio)[..., :num_frames]
    //
    // The pad lands on the waveform, not on the features, so the last
    // frame's reflect-padded window sees zeros rather than a mirror of
    // the tail. That only bites when n_samples is an exact multiple of
    // hop; otherwise the clip is already long enough and no pad happens.
    const int64_t mel_frames = n_samples / hop;
    if (mel_frames <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t num_frames = static_cast<int64_t>(stack) * ((mel_frames + stack - 1) / stack);
    const int64_t need       = (num_frames - 1) * hop + 1;

    const float *      mel_pcm = pcm;
    int64_t            mel_n   = n_samples;
    std::vector<float> padded;
    if (n_samples < need) {
        padded.assign(static_cast<size_t>(need), 0.0f);
        std::memcpy(padded.data(), pcm, static_cast<size_t>(n_samples) * sizeof(float));
        mel_pcm = padded.data();
        mel_n   = need;
    }

    // per_utterance gives log10 -> floor at (global max - 8 dB) -> (v+4)/4,
    // which is exactly the reference's clamp/div/add chain. The explicit
    // frame count is the part that differs from every other family: the
    // mode's own rule drops the trailing center-pad frame, which is right
    // when mel_frames is even (num_frames == mel_frames) and wrong when it
    // is odd (num_frames == mel_frames + 1, and the reference keeps that
    // frame). The per-utterance max is taken over the emitted frames only,
    // matching `mel[..., :num_frames].amax()`.
    std::vector<float> logmel;
    int                n_mels   = 0;
    int                n_frames = 0;
    if (const transcribe_status st = mel.compute(mel_pcm, static_cast<size_t>(mel_n), logmel, n_mels, n_frames,
                                                 n_threads, static_cast<int>(num_frames));
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (n_frames != static_cast<int>(num_frames) || n_mels != hp.fe_num_mels) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: frontend produced [%d, %d], expected [%d, %d]", n_mels,
                n_frames, hp.fe_num_mels, static_cast<int>(num_frames));
        return TRANSCRIBE_ERR_GGUF;
    }

    // First-order deltas. torchaudio.functional.compute_deltas with
    // win_length=3 reduces to n=1, denom=2 and replicate edge padding:
    //     d[t] = (x[min(t+1, T-1)] - x[max(t-1, 0)]) / 2
    // (the conv1d there is a cross-correlation with kernel [-1, 0, 1]).
    const int          T       = n_frames;
    const int          n_delta = hp.fe_deltas ? n_mels : 0;
    std::vector<float> deltas;
    if (hp.fe_deltas) {
        deltas.resize(static_cast<size_t>(n_mels) * T);
        for (int m = 0; m < n_mels; ++m) {
            const float * src = logmel.data() + static_cast<size_t>(m) * T;
            float *       dst = deltas.data() + static_cast<size_t>(m) * T;
            for (int t = 0; t < T; ++t) {
                const float next = src[(t + 1 < T) ? (t + 1) : (T - 1)];
                const float prev = src[(t > 0) ? (t - 1) : 0];
                dst[t]           = 0.5f * (next - prev);
            }
        }
    }

    // Channel concat + 2-frame stack. The reference does
    //   cat([logmel, deltas], dim=-2)   ->  [160, T]
    //   .transpose(-1, -2)              ->  [T, 160]
    //   .reshape(B, -1, 2 * 160)        ->  [T/2, 320]
    // so row t is [logmel(2t) || delta(2t) || logmel(2t+1) || delta(2t+1)].
    const int mel_ch    = n_mels + n_delta;
    const int input_dim = mel_ch * stack;
    if (input_dim != hp.enc_input_dim) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: stacked width %d != encoder input_dim %d", input_dim,
                hp.enc_input_dim);
        return TRANSCRIBE_ERR_GGUF;
    }
    const int t_enc = T / stack;
    if (t_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    out_feats.assign(static_cast<size_t>(t_enc) * input_dim, 0.0f);
    for (int t = 0; t < t_enc; ++t) {
        float * row = out_feats.data() + static_cast<size_t>(t) * input_dim;
        for (int s = 0; s < stack; ++s) {
            const int frame = t * stack + s;
            float *   dst   = row + static_cast<size_t>(s) * mel_ch;
            for (int m = 0; m < n_mels; ++m) {
                dst[m] = logmel[static_cast<size_t>(m) * T + frame];
            }
            if (n_delta > 0) {
                for (int m = 0; m < n_delta; ++m) {
                    dst[n_mels + m] = deltas[static_cast<size_t>(m) * T + frame];
                }
            }
        }
    }
    out_t_enc = t_enc;
    return TRANSCRIBE_OK;
}

// Shaw indices + pad mask.

std::vector<int32_t> precompute_pos_rows(int context_size, int max_pos_emb) {
    const int            n = 2 * context_size - 1;
    std::vector<int32_t> rows(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        // rel_shift reads index d = key - query + context_size - 1, so the
        // relative offset (query - key) this slot must carry is
        // context_size - 1 - d.
        int d                        = context_size - 1 - i;
        d                            = std::max(d, -context_size);
        d                            = std::min(d, context_size);
        rows[static_cast<size_t>(i)] = static_cast<int32_t>(d + max_pos_emb);
    }
    return rows;
}

std::vector<float> precompute_pad_mask(int context_size, int t_len, const std::vector<int> & real_lens) {
    const int    n_blocks = (t_len + context_size - 1) / context_size;
    const int    B        = std::max<int>(1, static_cast<int>(real_lens.size()));
    // Large finite negative rather than -INF; see the header.
    const float  kMasked  = -1e30f;
    const size_t per_blk  = static_cast<size_t>(context_size) * context_size;

    std::vector<float> mask(per_blk * n_blocks * B, 0.0f);
    for (int b = 0; b < B; ++b) {
        // Two independent sources of pad keys, both handled by the same
        // comparison: this utterance may be shorter than the batch's
        // T_max, and T_max itself is rounded up to a whole number of
        // context blocks. `real` is the absolute index of the first
        // non-real key either way.
        const int real = real_lens.empty() ? t_len : std::min(real_lens[b], t_len);
        for (int blk = 0; blk < n_blocks; ++blk) {
            const int blk_start = blk * context_size;
            const int first_pad = std::max(0, std::min(context_size, real - blk_start));
            if (first_pad >= context_size) {
                continue;  // every key in this block is real
            }
            const size_t base = (static_cast<size_t>(b) * n_blocks + blk) * per_blk;
            for (int q = 0; q < context_size; ++q) {
                for (int k = first_pad; k < context_size; ++k) {
                    mask[base + static_cast<size_t>(q) * context_size + k] = kMasked;
                }
            }
        }
    }
    return mask;
}

std::vector<float> precompute_frame_mask(int t_len, const std::vector<int> & real_lens) {
    const int          B = std::max<int>(1, static_cast<int>(real_lens.size()));
    std::vector<float> mask(static_cast<size_t>(t_len) * B, 0.0f);
    for (int b = 0; b < B; ++b) {
        const int real = real_lens.empty() ? t_len : std::min(real_lens[b], t_len);
        for (int t = 0; t < real; ++t) {
            mask[static_cast<size_t>(b) * t_len + t] = 1.0f;
        }
    }
    return mask;
}

// Per-block builders.

namespace {

// Conv module: LN -> pointwise1 (hidden -> 2*inner) -> GLU (-> inner)
// -> depthwise k, stride s -> fused BN -> SiLU -> pointwise2
// (inner -> hidden). Input/output [hidden, T].
//
// Unlike granite 4.x the two pointwise weights are 2-D nn.Linear here,
// so they go straight into mul_mat with no reshape.
ggml_tensor * conv_module(ggml_context *              ctx,
                          ggml_tensor *               x,
                          const Granite5CtcEncBlock & b,
                          int                         conv_kernel,
                          int                         inner_dim,
                          int                         stride,
                          ggml_tensor *               frame_mask) {
    const int64_t d_model = x->ne[0];
    const int64_t T       = x->ne[1];
    const int64_t B       = x->ne[2];

    x = layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);

    x = ggml_mul_mat(ctx, b.conv_pointwise1_w, x);  // [2*inner, T, B]
    x = ggml_add(ctx, x, b.conv_pointwise1_b);

    // GLU over ne[0]: first half * sigmoid(second half). Matches
    // F.glu(dim=-1) on the reference's [B, T, 2*inner] layout.
    {
        ggml_tensor * gate = ggml_view_3d(ctx, x, inner_dim, T, B, x->nb[1], x->nb[2], /*offset=*/0);
        ggml_tensor * value =
            ggml_view_3d(ctx, x, inner_dim, T, B, x->nb[1], x->nb[2], inner_dim * ggml_element_size(x));
        x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    }

    // Zero the padded frames before the depthwise conv, matching the
    // reference's `masked_fill(~attention_mask.unsqueeze(-1), 0.0)`. This
    // is what keeps one utterance's padding out of its neighbours' real
    // frames through the kernel-7 receptive field. Null (single-shot or
    // an even-length batch) leaves the graph unchanged.
    if (frame_mask != nullptr) {
        x = ggml_mul(ctx, x, frame_mask);
    }

    // [inner, T, B] -> [T, inner, B] for the depthwise conv.
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    // SAME padding. At stride 2 this emits ceil(T/2) frames, one more
    // than the pooled residual when T is odd — the caller trims.
    //
    // Deliberately NOT conformer::conv_1d_dw_f32: that helper switches
    // algorithm on the batch size (im2col + mul_mat at B == 1, the direct
    // depthwise op at B > 1), which makes a batched utterance differ from
    // its single-shot self by ~7e-3 relative — enough to fail the batch
    // tensor-parity gate even though the text is byte-identical. Both
    // helpers below take ONE path at every batch size, so single-shot and
    // batched stay bit-identical. [T, inner, B] is reshaped to the
    // (W, H=1, C, N) image they want.
    //
    // The direct op replaced conv_2d_dw_f32 (im2col + mul_mat) purely for
    // speed: im2col inflates the activation by conv_kernel to feed a
    // matmul whose K is that same conv_kernel, which is nearly all memory
    // traffic. M4 / Metal / F32, jfk: encoder 114.7 ms -> 101.4 ms, the
    // depthwise conv itself 13.3 ms -> ~0. The two forms are not bitwise
    // equal (different accumulation), so the swap was re-gated: see the
    // family doc's "Depthwise conv: direct vs im2col" note.
    const int     padding = (conv_kernel - 1) / 2;
    const int64_t T_in    = x->ne[0];
    {
        // conv_2d_dw_direct_f32 promotes the kernel to F32 — the direct op
        // requires it, and the stored type is F16 at the BF16 reference
        // tier. Under im2col that promotion was tried and reverted for
        // being drift-neutral and costly; here it is not optional, and it
        // is free because the direct op never materialises a KW-fold
        // buffer to double.
        ggml_tensor * knl = ggml_reshape_4d(ctx, b.conv_depthwise_w, conv_kernel, 1, 1, inner_dim);
        ggml_tensor * img = ggml_reshape_4d(ctx, x, T_in, 1, inner_dim, B);
        x                 = transcribe::conformer::conv_2d_dw_direct_f32(ctx, knl, img,
                                                                         /*s0=*/stride, /*s1=*/1,
                                                                         /*p0=*/padding, /*p1=*/0,
                                                                         /*d0=*/1, /*d1=*/1);
        x                 = ggml_reshape_3d(ctx, x, x->ne[0], inner_dim, B);
    }

    x = transcribe::conformer::fused_batch_norm(ctx, x, b.conv_bn_fused_scale, b.conv_bn_fused_bias);
    x = ggml_silu(ctx, x);

    // [T_out, inner, B] -> [inner, T_out, B].
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    x = ggml_mul_mat(ctx, b.conv_pointwise2_w, x);  // [hidden, T_out, B]
    x = ggml_add(ctx, x, b.conv_pointwise2_b);
    (void) d_model;
    return x;
}

// Mean-pool consecutive frame pairs along the time axis.
//
// Reference: `hidden_states.unfold(1, 2, 2).mean(-1)`, which produces
// floor(T/2) frames and silently DROPS a trailing odd frame. Built from
// two strided views plus an add so it needs no pooling kernel and works
// on every backend.
ggml_tensor * mean_pool_pairs(ggml_context * ctx, ggml_tensor * x) {
    const int64_t C      = x->ne[0];
    const int64_t T      = x->ne[1];
    const int64_t B      = x->ne[2];
    const int64_t T_half = T / 2;
    ggml_tensor * even   = ggml_cont(ctx, ggml_view_3d(ctx, x, C, T_half, B, 2 * x->nb[1], x->nb[2], /*offset=*/0));
    ggml_tensor * odd    = ggml_cont(ctx, ggml_view_3d(ctx, x, C, T_half, B, 2 * x->nb[1], x->nb[2], x->nb[1]));
    return ggml_scale(ctx, ggml_add(ctx, even, odd), 0.5f);
}

// Keep the leading `t_keep` frames of [C, T, B].
ggml_tensor * trim_time(ggml_context * ctx, ggml_tensor * x, int64_t t_keep) {
    if (x->ne[1] == t_keep) {
        return x;
    }
    return ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], t_keep, x->ne[2], x->nb[1], x->nb[2], /*offset=*/0));
}

}  // namespace

// Encoder graph.

EncoderBuild build_encoder_graph(ggml_context *             ctx,
                                 const Granite5CtcWeights & weights,
                                 const Granite5CtcHParams & hp,
                                 int                        T_enc,
                                 const std::vector<int> &   real_lens) {
    EncoderBuild eb{};
    eb.n_batch = std::max<int>(1, static_cast<int>(real_lens.size()));

    // Uneven lengths are the only thing that turns masking on. An even
    // batch (and every single-utterance run) takes the identical graph
    // the single-shot path has always built.
    bool var_len = false;
    for (int len : real_lens) {
        if (len != T_enc) {
            var_len = true;
            break;
        }
    }
    const int64_t B = eb.n_batch;

    const int64_t d_model   = hp.enc_hidden;
    const int64_t input_dim = hp.enc_input_dim;
    const int     inner_dim = hp.enc_hidden * hp.enc_conv_expansion;
    const int     conv_k    = hp.enc_conv_kernel_size;
    const int     n_heads   = hp.enc_n_heads;
    const int     head_dim  = hp.enc_head_dim;
    const int     ctx_size  = hp.enc_context_size;
    const int     n_layers  = static_cast<int>(weights.enc_blocks.size());

    auto is_subsample = [&](int i) {
        return std::find(hp.enc_subsample_layers.begin(), hp.enc_subsample_layers.end(), i) !=
               hp.enc_subsample_layers.end();
    };

    // Plan the per-block sequence lengths first. Attention in block i
    // runs at the length ENTERING the block; a subsampling block halves
    // it on the way out (floor, matching the residual pooling). Each
    // distinct length needs its own pad mask and zero-pad tile.
    eb.block_stage.assign(n_layers, 0);
    // Per-utterance valid lengths, halved at every subsampling block. The
    // reference halves the mask with AND-over-pairs
    // (`downsample_attention_mask`), which on a prefix mask is exactly
    // floor(len / 2) — the same rounding the residual pooling uses, so a
    // frame pooled from one real and one pad source lands outside the new
    // valid range instead of silently entering it.
    std::vector<int>              stage_lens = real_lens.empty() ? std::vector<int>{ T_enc } : real_lens;
    std::vector<std::vector<int>> stage_real;
    {
        int t = T_enc;
        for (int i = 0; i < n_layers; ++i) {
            int stage = -1;
            for (size_t s = 0; s < eb.stages.size(); ++s) {
                if (eb.stages[s].t_len == t) {
                    stage = static_cast<int>(s);
                    break;
                }
            }
            if (stage < 0) {
                AttnStage st{};
                st.t_len    = t;
                st.n_blocks = (t + ctx_size - 1) / ctx_size;
                eb.stages.push_back(st);
                stage_real.push_back(stage_lens);
                stage = static_cast<int>(eb.stages.size()) - 1;
            }
            eb.block_stage[i] = stage;
            if (is_subsample(i)) {
                for (int & len : stage_lens) {
                    len /= 2;
                }
                t /= 2;
                if (t <= 0) {
                    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "granite5_ctc encoder: audio too short — the sequence collapses to 0 "
                            "frames at subsampling block %d (T_enc=%d)",
                            i, T_enc);
                    return eb;
                }
            }
        }
        eb.t_out         = t;
        eb.real_lens_out = stage_lens;
    }

    // Graph inputs.
    eb.feats_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_dim, T_enc, B);
    ggml_set_input(eb.feats_in);

    eb.pos_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * static_cast<int64_t>(ctx_size) - 1);
    named(eb.pos_rows, "enc.pos_rows");
    ggml_set_input(eb.pos_rows);

    for (size_t s = 0; s < eb.stages.size(); ++s) {
        AttnStage & st    = eb.stages[s];
        const int   t_pad = st.n_blocks * ctx_size;
        char        nbuf[64];

        st.pad_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ctx_size, ctx_size, st.n_blocks * B);
        std::snprintf(nbuf, sizeof(nbuf), "enc.pad_mask.%zu", s);
        named(st.pad_mask, nbuf);
        ggml_set_input(st.pad_mask);

        if (t_pad > st.t_len) {
            st.zero_pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_model, t_pad - st.t_len, B);
            std::snprintf(nbuf, sizeof(nbuf), "enc.zero_pad.%zu", s);
            named(st.zero_pad, nbuf);
            ggml_set_input(st.zero_pad);
        }

        if (var_len) {
            st.frame_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, st.t_len, B);
            std::snprintf(nbuf, sizeof(nbuf), "enc.frame_mask.%zu", s);
            named(st.frame_mask, nbuf);
            ggml_set_input(st.frame_mask);
        }
    }

    auto record = [&](const char * name, ggml_tensor * t) {
        named(t, name);
        transcribe::debug::mark_tensor_for_dump(t);
        eb.dump_list.emplace_back(name, t);
    };

    record("mel.in", eb.feats_in);

    // input_linear: 320 -> 1024.
    ggml_tensor * x = linear(ctx, eb.feats_in, weights.enc_top.input_linear_w, weights.enc_top.input_linear_b);
    // Reference: `hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)`
    // right after input_linear, so the bias does not leave padded frames
    // carrying a constant.
    if (eb.stages[0].frame_mask != nullptr) {
        x = ggml_mul(ctx, x, eb.stages[0].frame_mask);
    }
    record("enc.input_linear.out", x);

    // Residual-stream sub-step dumps. Emitted for every subsampling block
    // plus the first block after them — the range where the time axis
    // changes and where a pooling/trim mistake is localizable. Beyond
    // that the blocks are all identical, so per-block sub-steps would
    // only cost validation memory. mark_tensor_for_dump is a no-op when
    // dumping is off, so this is free in production.
    int last_sub = -1;
    for (int32_t idx : hp.enc_subsample_layers) {
        last_sub = std::max(last_sub, idx);
    }
    const int sub_dump_through = last_sub + 1;

    for (int i = 0; i < n_layers; ++i) {
        const auto &      b        = weights.enc_blocks[i];
        const AttnStage & stage    = eb.stages[eb.block_stage[i]];
        const bool        sub      = is_subsample(i);
        const bool        dump_sub = (i <= sub_dump_through);
        char              nbuf[64];

        // FF1 macaron half: x + 0.5 * FF(LN(x)).
        x = transcribe::conformer::macaron_ff_residual(ctx, x, b.norm_ff1_w, b.norm_ff1_b, b.ff1_lin1_w, b.ff1_lin1_b,
                                                       b.ff1_lin2_w, b.ff1_lin2_b);
        if (dump_sub) {
            std::snprintf(nbuf, sizeof(nbuf), "enc.block.%d.post_ff1", i);
            record(nbuf, x);
        }

        // Block-local Shaw attention at the stage's length.
        const transcribe::granite_conformer::ShawAttnWeights shaw_w{
            b.norm_attn_w, b.norm_attn_b, b.attn_q_w, b.attn_kv_w, b.attn_rel_pos_emb, b.attn_out_w, b.attn_out_b,
        };
        ggml_tensor * attn_out = transcribe::granite_conformer::shaw_block_attn(
            ctx, x, stage.zero_pad, /*dists=*/nullptr, stage.pad_mask, shaw_w, n_heads, head_dim, ctx_size,
            stage.n_blocks, stage.t_len, kLayerNormEps, eb.pos_rows);
        if (attn_out == nullptr) {
            return eb;
        }
        x = ggml_add(ctx, x, attn_out);
        if (dump_sub) {
            std::snprintf(nbuf, sizeof(nbuf), "enc.block.%d.post_attn", i);
            record(nbuf, x);
        }

        // Conv module + residual. On a subsampling block the residual is
        // mean-pooled over frame pairs (dropping a trailing odd frame)
        // and the stride-2 conv output is trimmed down to that length —
        // never the other way round, since ceil(T/2) >= floor(T/2).
        ggml_tensor * conv_out = conv_module(ctx, x, b, conv_k, inner_dim, sub ? 2 : 1, stage.frame_mask);
        if (sub) {
            ggml_tensor * pooled = mean_pool_pairs(ctx, x);
            conv_out             = trim_time(ctx, conv_out, pooled->ne[1]);
            x                    = ggml_add(ctx, pooled, conv_out);
        } else {
            x = ggml_add(ctx, x, conv_out);
        }
        if (dump_sub) {
            std::snprintf(nbuf, sizeof(nbuf), "enc.block.%d.post_conv", i);
            record(nbuf, x);
        }

        // FF2 macaron half.
        x = transcribe::conformer::macaron_ff_residual(ctx, x, b.norm_ff2_w, b.norm_ff2_b, b.ff2_lin1_w, b.ff2_lin1_b,
                                                       b.ff2_lin2_w, b.ff2_lin2_b);
        if (dump_sub) {
            std::snprintf(nbuf, sizeof(nbuf), "enc.block.%d.post_ff2", i);
            record(nbuf, x);
        }

        // Per-block final LayerNorm. This IS the block output the
        // reference hooks, so it is named and dumped before the
        // self-conditioning injection below.
        x = layer_norm(ctx, x, b.norm_out_w, b.norm_out_b);
        std::snprintf(nbuf, sizeof(nbuf), "enc.block.%d.out", i);
        record(nbuf, x);

        // Self-conditioned CTC. Reference fires when
        // `layer_idx + 1 == num_hidden_layers // 2`, i.e. on the output
        // of block (self_cond_layer - 1), using the SAME projection as
        // the final CTC head (tied weights).
        if (i + 1 == hp.enc_self_cond_layer) {
            ggml_tensor * mid_logits = linear(ctx, x, weights.enc_top.ctc_proj_w, weights.enc_top.ctc_proj_b);
            record("enc.ctc.mid_logits", mid_logits);

            ggml_tensor * mid_soft  = ggml_soft_max(ctx, mid_logits);
            ggml_tensor * injection = linear(ctx, mid_soft, weights.enc_top.ctc_bypass_w, weights.enc_top.ctc_bypass_b);
            record("enc.ctc.mid_injection", injection);

            x = ggml_add(ctx, x, injection);
        }
    }

    // NOTE: on a variant whose last block is not followed by an injection
    // this is the SAME tensor as `enc.block.<N-1>.out`. Both names are in
    // dump_list, which is why dumping goes through that list instead of
    // the tensor's single ggml name.
    eb.out = x;
    named(eb.out, "enc.out");
    ggml_set_output(eb.out);
    eb.dump_list.emplace_back("enc.out", eb.out);

    // Tied CTC head: the same projection as the mid-layer one.
    eb.ctc_logits = linear(ctx, x, weights.enc_top.ctc_proj_w, weights.enc_top.ctc_proj_b);
    record("enc.ctc_logits", eb.ctc_logits);
    ggml_set_output(eb.ctc_logits);

    eb.graph = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
    if (eb.graph == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc encoder: ggml_new_graph_custom failed");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.ctc_logits);
    ggml_build_forward_expand(eb.graph, eb.out);

    return eb;
}

}  // namespace transcribe::granite5_ctc
