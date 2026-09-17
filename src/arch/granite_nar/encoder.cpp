// arch/granite_nar/encoder.cpp - NLE Conformer encoder + BPE CTC head.
//
// Structurally identical to the AR granite encoder (block-local Shaw
// self-attention, GLU conv module with conv_expansion=2, macaron FFN,
// mid-layer self-conditioned CTC bypass). NLE additions:
//
//   - A second CTC head over a BPE vocab. Its posterior-weighted encoder
//     states are projected in bounded chunks after this graph; each chunk
//     performs greedy argmax before returning to host.
//   - All-hidden-states capture: we tap the per-block POST-LN output at
//     indices specified by hp.enc_layer_indices (e.g. [4, 8, 12, -1]
//     1-indexed → block outputs after 3, 7, 11, 15). These are
//     concatenated along the channel axis and returned as the
//     projector's input.

#include "encoder.h"

#include "conformer/conformer.h"
#include "ggml.h"
#include "granite_conformer/shaw_attn.h"
#include "granite_nar.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace transcribe::granite_nar {

namespace {

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

// Host-side mel + 2-frame stack.

transcribe_status compute_mel_encoder_input(const transcribe::MelFrontend & mel,
                                            const float *                   pcm,
                                            int                             n_samples,
                                            int                             n_threads,
                                            std::vector<float> &            out_mel,
                                            int &                           out_t_enc) {
    if (pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    std::vector<float> raw;
    int                n_mels   = 0;
    int                n_frames = 0;
    if (const transcribe_status st = mel.compute(pcm, static_cast<size_t>(n_samples), raw, n_mels, n_frames, n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }

    if ((n_frames % 2) == 1) {
        --n_frames;
    }
    const int t_enc = n_frames / 2;
    if (t_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int stride = static_cast<int>(raw.size() / static_cast<size_t>(n_mels));
    if (stride <= 0 || stride < n_frames) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite_nar: mel buffer stride (%d) < expected frames (%d)", stride,
                n_frames);
        return TRANSCRIBE_ERR_GGUF;
    }

    const int input_dim = 2 * n_mels;
    out_mel.assign(static_cast<size_t>(t_enc) * input_dim, 0.0f);
    for (int t = 0; t < t_enc; ++t) {
        float * dst = out_mel.data() + static_cast<size_t>(t) * input_dim;
        for (int m = 0; m < n_mels; ++m) {
            dst[m]          = raw[static_cast<size_t>(m) * stride + 2 * t];
            dst[m + n_mels] = raw[static_cast<size_t>(m) * stride + 2 * t + 1];
        }
    }
    out_t_enc = t_enc;
    return TRANSCRIBE_OK;
}

// Shaw bookkeeping (matches AR granite).

std::vector<int32_t> precompute_pos_rows(int context_size, int max_pos_emb) {
    // One rel_pos_emb row per distinct relative offset, replacing the full
    // [context_size, context_size] table. The table's value at
    // (ne0 = key, ne1 = query) is clamp(query - key, +/- context_size) +
    // max_pos_emb, which depends on (query - key) alone, so only the
    // 2*context_size - 1 reachable offsets are needed.
    //
    // shaw_block_attn's skew path indexes this vector by
    // d = key - query + context_size - 1 (the layout conformer::rel_shift
    // rotates), so slot d must carry the offset context_size - 1 - d.
    const int            n = 2 * context_size - 1;
    std::vector<int32_t> rows(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        int d = context_size - 1 - i;
        if (d < -context_size) {
            d = -context_size;
        }
        if (d > context_size) {
            d = context_size;
        }
        rows[static_cast<size_t>(i)] = static_cast<int32_t>(d + max_pos_emb);
    }
    return rows;
}

std::vector<float> precompute_last_block_mask(int context_size, int t_enc_remainder) {
    std::vector<float> mask(static_cast<size_t>(context_size) * context_size, 0.0f);
    if (t_enc_remainder <= 0 || t_enc_remainder >= context_size) {
        return mask;
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int q = 0; q < context_size; ++q) {
        for (int k = t_enc_remainder; k < context_size; ++k) {
            mask[static_cast<size_t>(q) * context_size + k] = neg_inf;
        }
    }
    return mask;
}

// Per-block builders (identical to AR granite, parameterised over the
// granite_nar weight struct).

namespace {

ggml_tensor * macaron(ggml_context * ctx, ggml_tensor * x, const GraniteNarEncBlock & b, bool is_ff1) {
    if (is_ff1) {
        return transcribe::conformer::macaron_ff_residual(ctx, x, b.norm_ff1_w, b.norm_ff1_b, b.ff1_up_w, b.ff1_up_b,
                                                          b.ff1_down_w, b.ff1_down_b);
    }
    return transcribe::conformer::macaron_ff_residual(ctx, x, b.norm_ff2_w, b.norm_ff2_b, b.ff2_up_w, b.ff2_up_b,
                                                      b.ff2_down_w, b.ff2_down_b);
}

ggml_tensor * conv_module(ggml_context *             ctx,
                          ggml_tensor *              x,
                          const GraniteNarEncBlock & b,
                          ggml_tensor *              bn_fused_scale,
                          ggml_tensor *              bn_fused_bias,
                          int                        conv_kernel,
                          int                        inner_dim,
                          bool                       direct_depthwise) {
    const int64_t d_model = x->ne[0];
    const int64_t T       = x->ne[1];

    x = layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);
    {
        ggml_tensor * pw1 = ggml_reshape_2d(ctx, b.conv_pointwise1_w, d_model, 2 * inner_dim);
        x                 = ggml_mul_mat(ctx, pw1, x);
        x                 = ggml_add(ctx, x, b.conv_pointwise1_b);
    }
    {
        ggml_tensor * gate  = ggml_view_2d(ctx, x, inner_dim, T, x->nb[1], 0);
        ggml_tensor * value = ggml_view_2d(ctx, x, inner_dim, T, x->nb[1], inner_dim * ggml_element_size(x));
        x                   = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    const int padding = (conv_kernel - 1) / 2;
    if (direct_depthwise) {
        ggml_tensor * kernel = ggml_reshape_4d(ctx, b.conv_depthwise_w, conv_kernel, 1, 1, inner_dim);
        ggml_tensor * data   = ggml_reshape_4d(ctx, x, T, 1, inner_dim, 1);
        x                    = transcribe::conformer::conv_2d_dw_direct_f32(ctx, kernel, data,
                                                                            /*s0=*/1, /*s1=*/1,
                                                                            /*p0=*/padding, /*p1=*/0,
                                                                            /*d0=*/1, /*d1=*/1);
        x                    = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[2], x->ne[3]);
    } else {
        constexpr int64_t          kTimeChunk = 256;
        std::vector<ggml_tensor *> chunks;
        chunks.reserve(static_cast<size_t>((T + kTimeChunk - 1) / kTimeChunk));
        for (int64_t out0 = 0; out0 < T; out0 += kTimeChunk) {
            const int64_t n_out     = std::min<int64_t>(kTimeChunk, T - out0);
            const int64_t src0      = out0 - padding;
            const int64_t src1      = src0 + n_out + conv_kernel - 1;
            const int64_t view0     = std::max<int64_t>(0, src0);
            const int64_t view1     = std::min<int64_t>(T, src1);
            const int64_t pad_left  = view0 - src0;
            const int64_t pad_right = src1 - view1;
            ggml_tensor * input =
                ggml_view_3d(ctx, x, view1 - view0, inner_dim, 1, x->nb[1], x->nb[2], view0 * x->nb[0]);
            input = ggml_pad_ext(ctx, input, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
            chunks.push_back(transcribe::conformer::conv_1d_dw_f32(ctx, b.conv_depthwise_w, input,
                                                                   /*stride=*/1, /*padding=*/0,
                                                                   /*dilation=*/1));
        }
        while (chunks.size() > 1) {
            std::vector<ggml_tensor *> next;
            next.reserve((chunks.size() + 1) / 2);
            for (size_t i = 0; i < chunks.size(); i += 2) {
                next.push_back(i + 1 < chunks.size() ? ggml_concat(ctx, chunks[i], chunks[i + 1], /*dim=*/0) :
                                                       chunks[i]);
            }
            chunks.swap(next);
        }
        x = chunks.front();
    }

    x = transcribe::conformer::fused_batch_norm(ctx, x, bn_fused_scale, bn_fused_bias);
    x = ggml_silu(ctx, x);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    {
        ggml_tensor * pw2 = ggml_reshape_2d(ctx, b.conv_pointwise2_w, inner_dim, d_model);
        x                 = ggml_mul_mat(ctx, pw2, x);
        x                 = ggml_add(ctx, x, b.conv_pointwise2_b);
    }
    return x;
}

}  // namespace

EncoderBuild build_encoder_graph(ggml_context *            ctx,
                                 const GraniteNarWeights & weights,
                                 const GraniteNarHParams & hp,
                                 int                       T_enc,
                                 bool /*use_flash*/,
                                 const char * backend_name) {
    EncoderBuild eb{};

    const bool backend_direct   = backend_name != nullptr && (std::strstr(backend_name, "Vulkan") != nullptr ||
                                                              std::strstr(backend_name, "CUDA") != nullptr ||
                                                              std::strstr(backend_name, "ROCm") != nullptr);
    const bool direct_depthwise = transcribe::conformer::resolve_conv_direct(
        "TRANSCRIBE_CONV_DIRECT_DW", "TRANSCRIBE_CONV_NO_DIRECT_DW", backend_direct);
    eb.n_blocks_local = (T_enc + hp.enc_context_size - 1) / hp.enc_context_size;
    const int T_pad   = eb.n_blocks_local * hp.enc_context_size;
    eb.last_block_rem = T_enc - (eb.n_blocks_local - 1) * hp.enc_context_size;

    const int64_t d_model   = hp.enc_hidden;
    const int64_t input_dim = hp.enc_input_dim;
    const int64_t inner_dim = static_cast<int64_t>(hp.enc_hidden) * hp.enc_conv_expansion;
    const int     conv_k    = hp.enc_conv_kernel_size;
    const int     n_heads   = hp.enc_n_heads;
    const int     head_dim  = hp.enc_head_dim;
    const int     ctx_size  = hp.enc_context_size;
    const int     n_layers  = hp.enc_n_layers;

    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, T_enc);
    named(eb.mel_in, "enc.mel_in");
    ggml_set_input(eb.mel_in);

    eb.pos_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * static_cast<int64_t>(ctx_size) - 1);
    named(eb.pos_rows, "enc.pos_rows");
    ggml_set_input(eb.pos_rows);

    eb.last_block_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ctx_size, ctx_size, eb.n_blocks_local);
    named(eb.last_block_mask, "enc.last_block_mask");
    ggml_set_input(eb.last_block_mask);

    if (T_pad > T_enc) {
        eb.zero_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_pad - T_enc);
        named(eb.zero_pad, "enc.zero_pad");
        ggml_set_input(eb.zero_pad);
    }

    // Input linear (160 -> 1024).
    ggml_tensor * x = linear(ctx, eb.mel_in, weights.enc_top.input_linear_w, weights.enc_top.input_linear_b);
    named(x, "enc.input_linear.out");
    eb.dumps.input_linear_out = x;
    transcribe::debug::mark_tensor_for_dump(x);

    const int bypass_after = hp.enc_n_layers / 2;  // 1-indexed boundary

    // Resolve enc_layer_indices to 0-indexed block outputs we want to
    // capture for the projector. Negative values count from the end
    // (-1 -> last block). The all_hidden_states array in the reference
    // is 1-indexed over (input_linear, block1, block2, ...) — converting
    // to "post-block i (0-indexed)" gives idx - 1 for positive values
    // and (n_layers - 1 + idx + 1) - 1 = n_layers + idx for negatives.
    std::vector<int> capture_idx;
    capture_idx.reserve(hp.enc_layer_indices.size());
    for (int32_t li : hp.enc_layer_indices) {
        if (li > 0) {
            capture_idx.push_back(li - 1);
        } else if (li < 0) {
            capture_idx.push_back(n_layers + li);
        } else {
            capture_idx.push_back(0);
        }
    }

    std::vector<ggml_tensor *> captures(capture_idx.size(), nullptr);
    for (size_t k = 0; k < capture_idx.size(); ++k) {
        if (capture_idx[k] == n_layers - 1) {
            eb.final_capture_offset = static_cast<int64_t>(k) * d_model;
            break;
        }
    }
    if (eb.final_capture_offset < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite_nar encoder: projector captures do not include final layer");
        return eb;
    }

    for (int i = 0; i < n_layers; ++i) {
        const auto & b = weights.enc_blocks[i];

        x = macaron(ctx, x, b, /*is_ff1=*/true);
        // Sub-step dump: post-FF1 residual on block 0.
        if (i == 0) {
            named(x, "enc.block.0.post_ff1");
            eb.dumps.block_0_post_ff1 = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }

        const transcribe::granite_conformer::ShawAttnWeights shaw_w{
            b.norm_attn_w, b.norm_attn_b, b.attn_q_w, b.attn_kv_w, b.attn_rel_pos_emb, b.attn_out_w, b.attn_out_b,
        };
        ggml_tensor * attn_out = transcribe::granite_conformer::shaw_block_attn(
            ctx, x, eb.zero_pad, /*dists=*/nullptr, eb.last_block_mask, shaw_w, n_heads, head_dim, ctx_size,
            eb.n_blocks_local, T_enc, kLayerNormEps, eb.pos_rows);
        if (attn_out == nullptr) {
            return eb;
        }
        x = ggml_add(ctx, x, attn_out);
        if (i == 0) {
            named(x, "enc.block.0.post_attn");
            eb.dumps.block_0_post_attn = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }

        ggml_tensor * conv_out = conv_module(ctx, x, b, b.conv_bn_fused_scale, b.conv_bn_fused_bias, conv_k,
                                             static_cast<int>(inner_dim), direct_depthwise);
        x                      = ggml_add(ctx, x, conv_out);
        if (i == 0) {
            named(x, "enc.block.0.post_conv");
            eb.dumps.block_0_post_conv = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }

        x = macaron(ctx, x, b, /*is_ff1=*/false);
        if (i == 0) {
            named(x, "enc.block.0.post_ff2");
            eb.dumps.block_0_post_ff2 = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        x = layer_norm(ctx, x, b.norm_post_w, b.norm_post_b);

        // Per-block dump taps. The reference dumper hooks the layer
        // module itself, so `enc.block.{i}.out` is captured BEFORE the
        // CTC bypass injection (which fires outside the block in the
        // encoder forward loop).
        if (i == 0) {
            named(x, "enc.block.0.out");
            eb.dumps.block_0_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        if (i == bypass_after - 1) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            named(x, nm);
            eb.dumps.block_mid_pre = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        if (i == n_layers - 1) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            named(x, nm);
            eb.dumps.block_last_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }

        // Self-conditioned CTC bypass at layer N/2 (1-indexed). In the
        // reference forward this fires BEFORE all_hidden_states is
        // appended, so the multi-layer capture at layer index 8 sees
        // the post-bypass value. We also expose softmax(mid_ctc)[blank]
        // as a separate graph output so the host-side BPE pool can use
        // it for `importance = 1 - blank_prob_mid` per the reference.
        if ((i + 1) == bypass_after) {
            ggml_tensor * cl = ggml_mul_mat(ctx, weights.enc_top.ctc_proj_w, x);
            cl               = ggml_add(ctx, cl, weights.enc_top.ctc_proj_b);
            // This is mid_logits in the reference modeling code:
            //   mid_logits = encoder.out(dropout(hidden))  at self_conditioning_layer
            // — the only place the char-CTC head is invoked by the model
            // forward. Dump it here as `enc.ctc_logits` so the C++ and ref
            // dumps capture the same semantic tensor.
            named(cl, "enc.ctc_logits");
            eb.ctc_logits       = cl;
            eb.dumps.ctc_logits = cl;
            ggml_set_output(cl);
            transcribe::debug::mark_tensor_for_dump(cl);
            ggml_tensor * cs         = ggml_soft_max(ctx, cl);
            // Slice the blank-token probability (channel 0). cs ne =
            // [n_ctc_vocab, T_enc]; view a [1, T_enc] window at offset 0
            // and reshape to [T_enc]. Element 0 of ne[0] is the blank.
            ggml_tensor * blank_prob = ggml_view_2d(ctx, cs, 1, T_enc, cs->nb[1], 0);
            blank_prob               = ggml_cont(ctx, blank_prob);
            blank_prob               = ggml_reshape_1d(ctx, blank_prob, T_enc);
            named(blank_prob, "enc.mid_blank_probs");
            eb.mid_blank_probs = blank_prob;
            ggml_set_output(blank_prob);

            ggml_tensor * bypass = ggml_mul_mat(ctx, weights.enc_top.ctc_bypass_w, cs);
            bypass               = ggml_add(ctx, bypass, weights.enc_top.ctc_bypass_b);
            x                    = ggml_add(ctx, x, bypass);
        }

        // Multi-layer capture (matches reference's all_hidden_states
        // append placement: AFTER the bypass injection, so layer 7's
        // capture sees the bypass result).
        for (size_t k = 0; k < capture_idx.size(); ++k) {
            if (capture_idx[k] == i) {
                captures[k] = x;
            }
        }

        // enc.block.8.out tap (after the bypass).
        if (i == bypass_after) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "enc.block.%d.out", i);
            named(x, nm);
            eb.dumps.block_mid_post = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
    }

    // Channel concat of captured layer outputs. Walk captures in order
    // (capture_idx ordered by hp.enc_layer_indices entry).
    ggml_tensor * cat = nullptr;
    for (size_t k = 0; k < captures.size(); ++k) {
        if (captures[k] == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite_nar encoder: missing capture for layer index %d",
                    (int) hp.enc_layer_indices[k]);
            return eb;
        }
        if (cat == nullptr) {
            cat = captures[k];
        } else {
            cat = ggml_concat(ctx, cat, captures[k], /*dim=*/0);
        }
    }
    named(cat, "enc.cat_out");
    eb.cat_out = cat;
    ggml_set_output(eb.cat_out);

    eb.graph = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
    if (eb.graph == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite_nar encoder: ggml_new_graph_custom failed");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.cat_out);
    ggml_build_forward_expand(eb.graph, eb.ctc_logits);
    if (eb.mid_blank_probs) {
        ggml_build_forward_expand(eb.graph, eb.mid_blank_probs);
    }
    if (eb.dumps.input_linear_out) {
        ggml_build_forward_expand(eb.graph, eb.dumps.input_linear_out);
    }
    if (eb.dumps.block_0_out) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_0_out);
    }
    if (eb.dumps.block_0_post_ff1) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_0_post_ff1);
    }
    if (eb.dumps.block_0_post_attn) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_0_post_attn);
    }
    if (eb.dumps.block_0_post_conv) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_0_post_conv);
    }
    if (eb.dumps.block_0_post_ff2) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_0_post_ff2);
    }
    if (eb.dumps.block_mid_pre) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_mid_pre);
    }
    if (eb.dumps.block_mid_post) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_mid_post);
    }
    if (eb.dumps.block_last_out) {
        ggml_build_forward_expand(eb.graph, eb.dumps.block_last_out);
    }

    return eb;
}

BpeCtcBuild build_bpe_ctc_graph(ggml_context *            ctx,
                                const GraniteNarWeights & weights,
                                const GraniteNarHParams & hp,
                                int                       n_windows) {
    BpeCtcBuild bb{};
    if (ctx == nullptr || weights.enc_top.ctc_bpe_w == nullptr || weights.enc_top.ctc_bpe_b == nullptr ||
        hp.enc_hidden <= 0 || hp.enc_bpe_pool_window <= 0 || n_windows <= 0) {
        return bb;
    }

    bb.n_windows = n_windows;

    bb.hidden_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.enc_hidden, n_windows);
    named(bb.hidden_in, "enc.ctc_bpe.hidden_in");
    ggml_set_input(bb.hidden_in);

    ggml_tensor * logits = ggml_mul_mat(ctx, weights.enc_top.ctc_bpe_w, bb.hidden_in);
    logits               = ggml_add(ctx, logits, weights.enc_top.ctc_bpe_b);

    bb.token_ids = ggml_argmax(ctx, logits);
    named(bb.token_ids, "enc.ctc_bpe.token_ids");
    ggml_set_output(bb.token_ids);

    bb.graph = ggml_new_graph_custom(ctx, /*size=*/1024, /*grads=*/false);
    if (bb.graph == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite_nar BPE CTC: ggml_new_graph_custom failed");
        return {};
    }
    ggml_build_forward_expand(bb.graph, bb.token_ids);
    return bb;
}

}  // namespace transcribe::granite_nar
