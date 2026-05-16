// arch/granite/encoder.cpp - Granite Speech audio encoder (Conformer).
//
// Reference: GraniteSpeechCTCEncoder and GraniteSpeechConformerBlock in
// transformers/models/granite_speech/modeling_granite_speech.py.
//
// The encoder is structurally a NeMo-style Conformer (macaron FFN +
// rel-pos MHSA + GLU conv module + post-block LN) with two granite
// deviations:
//
//   1. Block-local Shaw attention. Each block reshapes the sequence
//      into context_size=200 chunks and attends only within each chunk.
//      The relative-position bias is a learned [2*max_pos_emb+1,
//      head_dim] embedding looked up by (k - q + max_pos_emb).
//   2. Self-conditioned CTC bypass. After the N/2-th block (0-indexed
//      idx N/2 - 1), the running hidden state is projected through
//      a 1024 -> 348 CTC head, softmaxed over the channel axis, and
//      projected back via a 348 -> 1024 head. The result is added as
//      a residual to the running hidden state.
//
// The Conformer macaron FFN and GLU conv module *shape* matches
// `transcribe::conformer::*`, but the helper there hardcodes
// inner_dim == d_model (parakeet / cohere / canary all use
// conv_expansion=1). Granite uses conv_expansion=2 — the pointwise1
// kernel maps d_model → 4*d_model and GLU halves to 2*d_model. We
// inline the conv module here rather than parameterise the shared
// helper. The macaron FF helper is fine to reuse (it only depends on
// the FF expansion ratio implicit in the tensor shapes, not on d_model).

#include "encoder.h"

#include "granite.h"
#include "weights.h"

#include "conformer/conformer.h"
#include "transcribe-debug.h"
#include "transcribe-mel.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace transcribe::granite {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

namespace {

// LayerNorm epsilon. Granite's encoder uses nn.LayerNorm whose default
// eps is 1e-5, matching the conformer helper's kLayerNormEps.
constexpr float kLayerNormEps = 1e-5f;

ggml_tensor * named(ggml_tensor * t, const char * name) {
    if (t != nullptr && name != nullptr) {
        ggml_set_name(t, name);
    }
    return t;
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                         ggml_tensor * gamma, ggml_tensor * beta)
{
    ggml_tensor * y = ggml_norm(ctx, x, kLayerNormEps);
    y = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) {
        y = ggml_add(ctx, y, beta);
    }
    return y;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x,
                     ggml_tensor * w, ggml_tensor * b)
{
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

} // namespace

// ---------------------------------------------------------------------------
// Host-side mel + 2-frame stack
// ---------------------------------------------------------------------------

transcribe_status compute_mel_encoder_input(
    const transcribe::MelFrontend & mel,
    const float *                   pcm,
    int                             n_samples,
    int                             n_threads,
    std::vector<float> &            out_mel,
    int &                           out_t_enc)
{
    if (pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    std::vector<float> raw;
    int n_mels   = 0;
    int n_frames = 0;
    if (const transcribe_status st = mel.compute(pcm,
                                                 static_cast<size_t>(n_samples),
                                                 raw, n_mels, n_frames,
                                                 n_threads);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // MelFrontend (whisper-mode) returns [n_mels, n_frames] row-major,
    // with the last (center-pad) frame already dropped. Granite then
    // drops the last frame if the resulting count is still odd:
    if ((n_frames % 2) == 1) {
        --n_frames;
    }
    const int t_enc = n_frames / 2;
    if (t_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Reshape to [t_enc, 2*n_mels] row-major. Output row t is
    // [logmel(t=2t, 0..n_mels-1) || logmel(t=2t+1, 0..n_mels-1)].
    // raw[m * n_frames_original + t] addresses bin m at frame t. Note
    // that whisper-mode's drop reduces n_frames AFTER compute(), so the
    // stride here is the ORIGINAL n_frames it computed against. We can
    // recover that as raw.size() / n_mels.
    const int stride = static_cast<int>(raw.size() / static_cast<size_t>(n_mels));
    if (stride <= 0 || stride < n_frames) {
        std::fprintf(stderr,
                     "granite: mel buffer stride (%d) < expected frames (%d)\n",
                     stride, n_frames);
        return TRANSCRIBE_ERR_GGUF;
    }

    const int input_dim = 2 * n_mels;
    out_mel.assign(static_cast<size_t>(t_enc) * input_dim, 0.0f);
    for (int t = 0; t < t_enc; ++t) {
        float * dst = out_mel.data() + static_cast<size_t>(t) * input_dim;
        for (int m = 0; m < n_mels; ++m) {
            dst[m]            = raw[static_cast<size_t>(m) * stride + 2 * t    ];
            dst[m + n_mels]   = raw[static_cast<size_t>(m) * stride + 2 * t + 1];
        }
    }
    out_t_enc = t_enc;
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Shaw attention_dists + last-block mask
// ---------------------------------------------------------------------------

std::vector<int32_t> precompute_attention_dists(int context_size, int max_pos_emb)
{
    // Reference (modeling_granite_speech.py:295-297):
    //   seq = torch.arange(context_size)
    //   relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
    //   attention_dists = clamp(relpos_dist, ±context_size) + max_pos_emb
    //
    // `seq.view(-1, 1) - seq.view(1, -1)` broadcasts to a [c, r] matrix
    // whose element at (c, r) is `c - r` — i.e., the offset from the
    // KEY position back to the QUERY position. (i in the row index is
    // the query position; j in the column index is the key position.)
    //
    // We were writing `r - c` here — the opposite sign — which mirrored
    // the Shaw bias across the diagonal and corrupted block-local
    // attention from the first encoder layer.
    std::vector<int32_t> dists(static_cast<size_t>(context_size) * context_size);
    for (int c = 0; c < context_size; ++c) {
        for (int r = 0; r < context_size; ++r) {
            int d = c - r;
            if (d < -context_size) d = -context_size;
            if (d >  context_size) d =  context_size;
            // Row-major: dists[c * context_size + r].
            dists[static_cast<size_t>(c) * context_size + r] =
                static_cast<int32_t>(d + max_pos_emb);
        }
    }
    return dists;
}

std::vector<float> precompute_last_block_mask(int context_size, int t_enc_remainder)
{
    // t_enc_remainder is the number of REAL positions in the last
    // block. Positions in [t_enc_remainder, context_size) are pad.
    //
    // We mask ONLY the pad-K columns (force them to -INF so they
    // drop out of every valid-Q row's softmax). Pad-Q rows are NOT
    // masked: doing so would set every score in those rows to -INF
    // and softmax(-INF, -INF, ...) is NaN, which then propagates to
    // neighbouring valid positions via the depthwise conv. Pad-Q
    // row outputs are garbage but bounded (softmax-weighted V averages),
    // and they're sliced off before any caller-visible tensor.
    //
    // Mask shape ggml ne order: [k=context_size, q=context_size].
    // ne[0]=k is the innermost, so mask[q * context_size + k] addresses
    // (q, k). Linearize: -INF if k >= remainder, else 0. (Independent of q.)
    std::vector<float> mask(static_cast<size_t>(context_size) * context_size, 0.0f);
    if (t_enc_remainder <= 0 || t_enc_remainder >= context_size) {
        return mask;  // no pad in last block — all-zero mask is a no-op
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int q = 0; q < context_size; ++q) {
        for (int k = t_enc_remainder; k < context_size; ++k) {
            mask[static_cast<size_t>(q) * context_size + k] = neg_inf;
        }
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Per-block builders
// ---------------------------------------------------------------------------

namespace {

// FFN (macaron half): pre-norm -> up_proj -> SiLU -> down_proj.
// Then residual: x = 0.5 * ff(x) + x.
//
// Reuses transcribe::conformer::macaron_ff_residual which has the
// exact same shape (pre-LN with bias, two biased linears, SiLU between,
// 0.5x residual scale).
ggml_tensor * granite_macaron(ggml_context * ctx,
                              ggml_tensor *  x,
                              const GraniteEncBlock & b,
                              bool           is_ff1)
{
    if (is_ff1) {
        return transcribe::conformer::macaron_ff_residual(
            ctx, x,
            b.norm_ff1_w, b.norm_ff1_b,
            b.ff1_up_w,   b.ff1_up_b,
            b.ff1_down_w, b.ff1_down_b);
    } else {
        return transcribe::conformer::macaron_ff_residual(
            ctx, x,
            b.norm_ff2_w, b.norm_ff2_b,
            b.ff2_up_w,   b.ff2_up_b,
            b.ff2_down_w, b.ff2_down_b);
    }
}

// Conv module: LN -> pointwise1 (d_model -> 2*inner_dim) -> GLU split
// (-> inner_dim) -> depthwise (k=15) -> BN -> SiLU -> pointwise2
// (inner_dim -> d_model). Operates on [d_model, T] (ne ordering).
//
// inner_dim = d_model * conv_expansion. For granite conv_expansion=2,
// so inner_dim = 2*d_model and the pw1 kernel maps d_model -> 4*d_model.
//
// The conformer::conv_module helper hardcodes inner_dim == d_model
// (parakeet / cohere / canary all use conv_expansion=1). We inline the
// module here. The shape of operations is otherwise identical.
ggml_tensor * granite_conv_module(ggml_context *          ctx,
                                  ggml_tensor *           x,
                                  const GraniteEncBlock & b,
                                  ggml_tensor *           bn_fused_scale,
                                  ggml_tensor *           bn_fused_bias,
                                  int                     conv_kernel,
                                  int                     inner_dim)
{
    const int64_t d_model = x->ne[0];
    const int64_t T       = x->ne[1];

    // Pre-LayerNorm with bias.
    x = layer_norm(ctx, x, b.norm_conv_w, b.norm_conv_b);

    // Pointwise1: [1, d_model, 2*inner_dim] kernel as a direct mul_mat.
    // After reshape to [d_model, 2*inner_dim] and mul_mat with
    // [d_model, T], output is [2*inner_dim, T].
    {
        ggml_tensor * pw1 = ggml_reshape_2d(ctx, b.conv_pointwise1_w,
                                            d_model, 2 * inner_dim);
        x = ggml_mul_mat(ctx, pw1, x);  // [2*inner_dim, T]
        x = ggml_add(ctx, x, b.conv_pointwise1_b);
    }

    // GLU: split ne[0] in half, gate * sigmoid(value).
    {
        ggml_tensor * gate  = ggml_view_2d(ctx, x, inner_dim, T,
                                           x->nb[1], /*offset=*/0);
        ggml_tensor * value = ggml_view_2d(ctx, x, inner_dim, T,
                                           x->nb[1],
                                           inner_dim * ggml_element_size(x));
        x = ggml_mul(ctx, gate, ggml_sigmoid(ctx, value));
    }
    // x ne = [inner_dim, T]

    // Transpose for depthwise conv: [inner_dim, T] -> [T, inner_dim].
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    // Depthwise conv1d: kernel [k, 1, inner_dim], symmetric pad (k-1)/2
    // (= 7 for k=15). Granite's depthwise has no bias.
    const int padding = (conv_kernel - 1) / 2;
    x = transcribe::conformer::conv_1d_dw_f32(
        ctx, b.conv_depthwise_w, x,
        /*stride=*/1, /*padding=*/padding, /*dilation=*/1);

    // Fused BatchNorm: y = x * fused_scale + fused_bias, with 1-D scale
    // and bias broadcast across the time axis.
    x = transcribe::conformer::fused_batch_norm(ctx, x,
                                                bn_fused_scale,
                                                bn_fused_bias);

    // SiLU.
    x = ggml_silu(ctx, x);

    // Transpose back: [T, inner_dim] -> [inner_dim, T].
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

    // Pointwise2: [1, inner_dim, d_model] kernel as a direct mul_mat.
    // After reshape to [inner_dim, d_model] and mul_mat with
    // [inner_dim, T], output is [d_model, T].
    {
        ggml_tensor * pw2 = ggml_reshape_2d(ctx, b.conv_pointwise2_w,
                                            inner_dim, d_model);
        x = ggml_mul_mat(ctx, pw2, x);
        x = ggml_add(ctx, x, b.conv_pointwise2_b);
    }

    return x;
}

// Block-local Shaw self-attention.
//
// Mirrors GraniteSpeechConformerAttention.forward exactly:
//
//   h = pre_norm(x)                                # at T_enc
//   if remainder > 0: h = F.pad(h, (0, 0, 0, pad)) # zero-pad to T_pad
//   q = to_q(h); kv = to_kv(h); k, v = kv.chunk(2, dim=-1)
//   ... block-local attention with Shaw bias ...
//   out = out.reshape(..., T_pad, inner_dim)
//   out = to_out(out[:, :num_features, :])         # slice back to T_enc
//   return out                                     # at T_enc
//
// The slice happens BEFORE to_out (the upstream order), which means
// the conv module that follows sees an unpadded tensor — critical
// because the depthwise conv would otherwise pull pad-row garbage
// into valid frames near the tail.
//
//   x:           [d_model, T_enc]  (the pre-norm input).
//   zero_pad:    [d_model, pad_n] = [d_model, T_pad - T_enc] f32 zeros,
//                or nullptr when T_enc % context_size == 0.
//   dists:       [context_size * context_size] int32, Shaw lookup indices.
//   pad_mask_3d: [context_size, context_size, num_blocks] additive mask.
//                All zero except the last slice for non-aligned T_enc
//                (pad-K columns at -INF; pad-Q rows are NOT masked,
//                see precompute_last_block_mask comments).
//
// Output: [d_model, T_enc] (matches input shape).
ggml_tensor * granite_shaw_block_attn(ggml_context *          ctx,
                                      ggml_tensor *           x,
                                      ggml_tensor *           zero_pad,
                                      ggml_tensor *           dists,
                                      ggml_tensor *           pad_mask_3d,
                                      const GraniteEncBlock & b,
                                      int                     n_heads,
                                      int                     head_dim,
                                      int                     context_size,
                                      int                     num_blocks,
                                      int                     T_enc)
{
    const int64_t inner_dim = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t d_model   = x->ne[0];
    const int64_t T_pad     = static_cast<int64_t>(context_size) * num_blocks;
    const float   scale     = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Pre-LayerNorm on the T_enc input.
    ggml_tensor * h = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);

    // Pad to T_pad along the time axis with zeros (graph input).
    if (T_pad > T_enc) {
        if (zero_pad == nullptr) {
            std::fprintf(stderr,
                         "granite encoder: zero_pad is null but T_pad > T_enc\n");
            return nullptr;
        }
        h = ggml_concat(ctx, h, zero_pad, /*dim=*/1);
    }
    // h ne = [d_model, T_pad]
    (void)d_model;

    // QKV projections. q has no bias; kv is fused (2*inner_dim) with no
    // bias; out has bias. The shape after projection in [feature, T_pad]
    // layout: q is [inner_dim, T_pad], kv is [2*inner_dim, T_pad].
    ggml_tensor * q  = ggml_mul_mat(ctx, b.attn_q_w,  h);  // [inner_dim, T_pad]
    ggml_tensor * kv = ggml_mul_mat(ctx, b.attn_kv_w, h);  // [2*inner_dim, T_pad]

    // Split kv into k, v along ne[0]. The mul_mat output is contiguous,
    // so views with a strided ne[0] are safe.
    ggml_tensor * k = ggml_view_2d(ctx, kv, inner_dim, kv->ne[1],
                                   kv->nb[1], /*offset=*/0);
    ggml_tensor * v = ggml_view_2d(ctx, kv, inner_dim, kv->ne[1],
                                   kv->nb[1],
                                   inner_dim * ggml_element_size(kv));
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Reshape into block-local form. We want q, k, v with ne layout
    // [head_dim, context_size, n_heads, num_blocks] so that
    // ggml_mul_mat(k, q) does per-(head, block) batched attention.
    //
    // Source layout: [inner_dim = n_heads*head_dim, T_pad = context_size*num_blocks].
    //
    // Step 1: reshape to [head_dim, n_heads, context_size, num_blocks].
    //   ne[0] = head_dim (inner-most of inner_dim split)
    //   ne[1] = n_heads
    //   ne[2] = context_size (inner-most of T_pad split)
    //   ne[3] = num_blocks
    auto reshape_qkv = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * r = ggml_reshape_4d(ctx, t,
                                          head_dim, n_heads,
                                          context_size, num_blocks);
        // Step 2: permute to [head_dim, context_size, n_heads, num_blocks].
        // ggml_permute new[a_i] = old[i]: we want new dims = (head_dim,
        // context_size, n_heads, num_blocks) coming from old dims
        // (head_dim, n_heads, context_size, num_blocks) at positions
        // (0, 2, 1, 3). So old[0] -> new[0], old[1] -> new[2],
        // old[2] -> new[1], old[3] -> new[3]: ggml_permute(r, 0, 2, 1, 3).
        r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
        return r;
    };
    q = reshape_qkv(q);
    k = reshape_qkv(k);
    v = reshape_qkv(v);

    // ----- Shaw positional bias -----
    // rel_pos_emb is [head_dim, 2*max_pos_emb+1]. attention_dists is
    // [context_size, context_size] int32. We materialise the per-(c, r)
    // lookup as a [head_dim, context_size*context_size] tensor, then
    // reshape to [head_dim, context_size=r, context_size=c, 1].
    //
    // ggml_get_rows(a, b) where a has ne[0..1] = [K, N] and b is int32
    // [B] returns [K, B] with row B[i] = a[:, b[i]].
    //
    // Flatten attention_dists to [context_size * context_size] via a
    // 1-D view.
    ggml_tensor * dists_flat = ggml_reshape_1d(ctx, dists,
                                               static_cast<int64_t>(context_size) * context_size);
    ggml_tensor * rel_lookup = ggml_get_rows(ctx, b.attn_rel_pos_emb, dists_flat);
    // rel_lookup ne = [head_dim, context_size*context_size]. We treat
    // it as [head_dim, context_size=r, context_size=c] (row-major over
    // (c, r) pairs — attention_dists is stored as c-then-r).
    rel_lookup = ggml_reshape_3d(ctx, rel_lookup,
                                 head_dim, context_size, context_size);

    // pos_attn[h, b, c, r] = sum_d q[h, b, c, d] * rel_lookup[c, r, d].
    //
    // Permute q from [head_dim, context_size=c, n_heads, num_blocks]
    // to [head_dim, n_heads, context_size=c, num_blocks] so that c is
    // ne[2]. Then ggml_mul_mat(rel_lookup, q_perm) with
    //   rel_lookup: [K=head_dim, M=r, batch=c, 1]
    //   q_perm:     [K=head_dim, N=n_heads, batch=c, num_blocks]
    // gives output [M=r, N=n_heads, c, num_blocks]. The "batch=c" axis
    // must match between the two operands. We need to broadcast
    // num_blocks across the rel_lookup tensor, which has ne[3] = 1.
    // ggml's mul_mat broadcasts the second operand's outer dims onto
    // the first by repeat if shapes are compatible (1 or matching).
    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    // q_perm ne = [head_dim, n_heads, context_size, num_blocks]

    // rel_lookup is [head_dim, r=context_size, c=context_size, 1]. We
    // want this to act as the LHS with K=head_dim and batch=c. Then
    // mul_mat over K. ggml_mul_mat treats ne[2:] as batch dims; the
    // second operand's ne[2:] must broadcast against the first's.
    // rel_lookup has ne[2]=c and ne[3]=1; q_perm has ne[2]=c and ne[3]=num_blocks.
    // Broadcast: ne[3]=1 of LHS broadcasts to num_blocks of RHS.

    ggml_tensor * pos_attn = ggml_mul_mat(ctx, rel_lookup, q_perm);
    // pos_attn ne = [M=r=context_size, N=n_heads, c=context_size, num_blocks]

    // ----- QK^T -----
    // ggml_mul_mat(K, Q) with K, Q both [head_dim, context_size,
    // n_heads, num_blocks] produces [context_size=k, context_size=q,
    // n_heads, num_blocks].
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    // kq ne = [k=context_size, q=context_size, n_heads, num_blocks]

    // To add pos_attn into kq, both must have the same ne layout.
    // pos_attn is [r=context_size, n_heads, c=context_size, num_blocks]
    // (r = key index, c = query index in our naming). kq is
    // [k, q, n_heads, num_blocks]. To match, permute pos_attn so
    // r becomes ne[0] (matches kq's k), c becomes ne[1] (matches q),
    // n_heads ne[2], num_blocks ne[3].
    //
    // Source: pos_attn (r, n_heads, c, num_blocks) at positions (0,1,2,3).
    // Target: (r, c, n_heads, num_blocks).
    // Mapping: old[0]->new[0], old[1]->new[2], old[2]->new[1], old[3]->new[3].
    // ggml_permute args (0, 2, 1, 3).
    pos_attn = ggml_cont(ctx, ggml_permute(ctx, pos_attn, 0, 2, 1, 3));

    // Combine: scores = (kq + pos_attn) * scale + pad_mask
    ggml_tensor * scores = ggml_add(ctx, kq, pos_attn);
    scores = ggml_scale(ctx, scores, scale);

    // Add per-block additive mask (all-zero for non-last blocks; -INF
    // pad cells in last block). pad_mask_3d ne =
    // [context_size, context_size, num_blocks, 1]. Broadcast across
    // n_heads (ne[2] of scores) by adding to a permuted view.
    //
    // scores ne = [k, q, n_heads, num_blocks]
    // pad_mask_3d ne = [k, q, num_blocks, 1]
    //
    // For the add to broadcast across n_heads, we'd want pad_mask shape
    // [k, q, 1, num_blocks]. ggml_add broadcasts dims that are 1.
    // Reshape pad_mask_3d:
    ggml_tensor * pad_mask_4d = ggml_reshape_4d(ctx, pad_mask_3d,
                                                context_size, context_size,
                                                1, num_blocks);
    scores = ggml_add(ctx, scores, pad_mask_4d);

    // Softmax over the k axis (ne[0]).
    ggml_tensor * attn = ggml_soft_max(ctx, scores);
    // attn ne = [k=context_size, q=context_size, n_heads, num_blocks]

    // out = attn @ V along k. We want output ne = [head_dim, q, n_heads,
    // num_blocks]. ggml_mul_mat with V transposed (V^T is implied by
    // mul_mat semantics): we need V as [head_dim, k, n_heads, num_blocks]
    // and attn as [k, q, n_heads, num_blocks]. Computing
    // ggml_mul_mat(V_T, attn) where V_T has the contraction axis k on
    // ne[0]. V's existing layout is [head_dim, k=context_size, n_heads,
    // num_blocks]. We need [k, head_dim, ...] so K matches.
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    // v_t ne = [k, head_dim, n_heads, num_blocks]

    ggml_tensor * out = ggml_mul_mat(ctx, v_t, attn);
    // out ne = [head_dim, q, n_heads, num_blocks]

    // Reshape back to [inner_dim, T_pad]. out has ne (head_dim, q,
    // n_heads, num_blocks). We want (head_dim * n_heads = inner_dim,
    // q * num_blocks = T_pad). First permute to (head_dim, n_heads,
    // q, num_blocks) so the collapse-to-2D below interleaves head_dim
    // and n_heads correctly.
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    // out ne = [head_dim, n_heads, q=context_size, num_blocks]
    out = ggml_reshape_2d(ctx, out, inner_dim, T_pad);
    // out ne = [inner_dim, T_pad]

    // Slice off pad rows BEFORE the out_proj, mirroring the reference's
    // `out = self.to_out(out[:, :num_features, :])` order. This is the
    // critical step that keeps pad-row garbage from leaking into the
    // post-attention conv module via the depthwise kernel.
    if (T_pad > T_enc) {
        out = ggml_view_2d(ctx, out, inner_dim, T_enc, out->nb[1], 0);
        out = ggml_cont(ctx, out);
    }

    // Final out_proj (with bias). Operates at T_enc.
    out = ggml_mul_mat(ctx, b.attn_out_w, out);
    out = ggml_add(ctx, out, b.attn_out_b);
    // out ne = [d_model, T_enc]
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Encoder graph
// ---------------------------------------------------------------------------

EncoderBuild build_encoder_graph(ggml_context *           ctx,
                                 const GraniteWeights &   weights,
                                 const GraniteHParams &   hp,
                                 int                      T_enc,
                                 bool                     /*use_flash*/)
{
    EncoderBuild eb {};
    eb.n_blocks_local = (T_enc + hp.enc_context_size - 1) / hp.enc_context_size;
    const int T_pad   = eb.n_blocks_local * hp.enc_context_size;
    eb.last_block_rem = T_enc - (eb.n_blocks_local - 1) * hp.enc_context_size;

    const int64_t d_model    = hp.enc_hidden;
    const int64_t input_dim  = hp.enc_input_dim;
    const int64_t inner_dim  = static_cast<int64_t>(hp.enc_hidden) *
                                hp.enc_conv_expansion;
    const int     conv_k     = hp.enc_conv_kernel_size;
    const int     n_heads    = hp.enc_n_heads;
    const int     head_dim   = hp.enc_head_dim;
    const int     ctx_size   = hp.enc_context_size;

    // ----- Graph inputs -----
    // mel_in: [input_dim, T_enc]. Encoder operates at T_enc throughout;
    // the only T_pad expansion happens INSIDE the Shaw block-local
    // attention helper, mirroring the reference's
    // `F.pad(hidden_states, (0, 0, 0, context_size - remainder))`
    // step which sits between pre_norm and the QKV projections.
    eb.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, T_enc);
    named(eb.mel_in, "enc.mel_in");
    ggml_set_input(eb.mel_in);

    // attention_dists: [ctx_size, ctx_size] int32, row-major over (c, r).
    eb.attention_dists = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                                            static_cast<int64_t>(ctx_size) * ctx_size);
    named(eb.attention_dists, "enc.attention_dists");
    ggml_set_input(eb.attention_dists);

    // last_block_mask: [ctx_size, ctx_size, n_blocks_local] additive
    // mask, all zero except last slice which carries -INF in pad cells.
    eb.last_block_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                            ctx_size, ctx_size,
                                            eb.n_blocks_local);
    named(eb.last_block_mask, "enc.last_block_mask");
    ggml_set_input(eb.last_block_mask);

    // zero_pad: only needed when T_enc is not aligned to context_size.
    // [d_model, T_pad - T_enc] f32 zeros, uploaded by the caller. This
    // is the buffer that attention's internal pre-norm output is
    // concatenated against to reach T_pad.
    if (T_pad > T_enc) {
        eb.zero_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                         d_model, T_pad - T_enc);
        named(eb.zero_pad, "enc.zero_pad");
        ggml_set_input(eb.zero_pad);
    }

    // ----- Input linear (160 -> 1024) -----
    ggml_tensor * x = linear(ctx, eb.mel_in,
                             weights.enc_top.input_linear_w,
                             weights.enc_top.input_linear_b);
    // x ne = [d_model, T_enc]
    named(x, "enc.input_linear.out");
    eb.dumps.input_linear_out = x;
    transcribe::debug::mark_tensor_for_dump(x);

    // BN fusion happens at load time and we read scale/bias from the
    // model (filled in model.cpp). For Phase 2 we accept that the
    // fused tensors come from external storage (via build_encoder_graph
    // caller wiring); we look them up off the granite weights struct.
    //
    // Each block in granite has its own [inner_dim] BN gamma/beta plus
    // running stats. The graph references the FUSED scale/bias which
    // model.cpp computes and writes once. They live as separate
    // ggml_tensor*s outside the GraniteWeights struct (in
    // GraniteModel::bn_fused); see model.cpp::fuse_batch_norm().
    //
    // For Phase 2 we accept a callback-style override: the caller must
    // ensure GraniteEncBlock::conv_bn_w / conv_bn_b / conv_bn_mean /
    // conv_bn_var have been "fused" into a pair of [inner_dim] f32
    // tensors and exposed through a side table. To keep build_encoder
    // self-contained we recompute the fusion on the fly here using
    // ggml ops:
    //   scale_i = bn_gamma_i / sqrt(bn_var_i + eps)
    //   bias_i  = bn_beta_i  - bn_mean_i * scale_i
    // These are 1-D fp32 tensors that fold cleanly into the graph.

    const int n_layers = static_cast<int>(weights.enc_blocks.size());
    const int bypass_after = hp.enc_n_layers / 2;  // 1-indexed boundary

    // cat_hidden_layers captures (post-LN, pre-bypass). Empty for 1b/2b;
    // size 1 with idx=3 for granite-speech-4.1-2b-plus. Holds raw ggml
    // tensor pointers; they remain valid for the lifetime of the
    // compute context.
    std::vector<ggml_tensor *> exported_hidden_states;
    exported_hidden_states.reserve(hp.enc_cat_hidden_layers.size());

    for (int i = 0; i < n_layers; ++i) {
        const auto & b = weights.enc_blocks[i];

        // --- FF1 macaron half ---
        x = granite_macaron(ctx, x, b, /*is_ff1=*/true);

        // --- Block-local Shaw self-attention ---
        // attention takes x at T_enc, pads internally to T_pad for the
        // block-local compute, and returns at T_enc — mirroring the
        // reference where to_out runs on the unpadded slice. This is
        // load-bearing: the conv module below cannot see pad-row
        // garbage near the tail (depthwise kernel would spread it).
        ggml_tensor * attn_out = granite_shaw_block_attn(
            ctx, x,
            eb.zero_pad, eb.attention_dists, eb.last_block_mask,
            b, n_heads, head_dim, ctx_size, eb.n_blocks_local, T_enc);
        if (attn_out == nullptr) {
            return eb;
        }
        x = ggml_add(ctx, x, attn_out);

        // --- Conv module (with pre-fused BN scale/bias) ---
        // model.cpp::fuse_batch_norm() precomputes
        //   scale = bn_w / sqrt(bn_var + eps)
        //   bias  = bn_b - bn_mean * scale
        // into [inner_dim] tensors stashed under conv_bn_fused_*.
        ggml_tensor * conv_out = granite_conv_module(
            ctx, x, b,
            b.conv_bn_fused_scale, b.conv_bn_fused_bias,
            conv_k, static_cast<int>(inner_dim));
        x = ggml_add(ctx, x, conv_out);

        // --- FF2 macaron half ---
        x = granite_macaron(ctx, x, b, /*is_ff1=*/false);

        // --- Post-block LayerNorm ---
        x = layer_norm(ctx, x, b.norm_post_w, b.norm_post_b);

        // --- cat_hidden_layers capture (-plus) ---
        // HF GraniteSpeechPlusCTCEncoder.forward:
        //   for idx, layer in enumerate(self.layers, start=1):
        //       hidden_states = layer(...)            # post_norm included
        //       if idx in cat_layers:
        //           exported_hidden_states.append(hidden_states)
        //       if idx == num_layers // 2:
        //           hidden_states += bypass
        // We are 0-indexed (i = idx - 1); the HF semantic is to capture
        // BEFORE the CTC bypass injection (which fires below).
        for (int32_t k : hp.enc_cat_hidden_layers) {
            if ((i + 1) == k) {
                exported_hidden_states.push_back(x);
                break;
            }
        }

        // --- Self-conditioned CTC bypass at layer N/2 ---
        // Reference: triggered when (i + 1) == num_layers // 2.
        if ((i + 1) == bypass_after) {
            // x: [d_model, T_pad]. ctc_proj: d_model -> output_dim.
            ggml_tensor * ctc_logits = ggml_mul_mat(ctx,
                                                    weights.enc_top.ctc_proj_w,
                                                    x);
            ctc_logits = ggml_add(ctx, ctc_logits, weights.enc_top.ctc_proj_b);
            // softmax over the channel axis (ne[0]).
            ggml_tensor * ctc_soft = ggml_soft_max(ctx, ctc_logits);
            // ctc_bypass: output_dim -> d_model.
            ggml_tensor * bypass = ggml_mul_mat(ctx,
                                                weights.enc_top.ctc_bypass_w,
                                                ctc_soft);
            bypass = ggml_add(ctx, bypass, weights.enc_top.ctc_bypass_b);
            x = ggml_add(ctx, x, bypass);
        }

        // --- Dumps ---
        // x is already [d_model, T_enc] (no pad), matching the
        // reference's hook on the bare layer output.
        if (i == 0) {
            named(x, "enc.block.0.out");
            eb.dumps.block_0_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        if (i == bypass_after) {
            // i.e., block index N/2 (0-indexed). For n=16, this is
            // block 8 — Stage 2 dumps `enc.block.8.out` after block 8's
            // forward completes (with the bypass residual already
            // injected at block-8 input).
            char bname[64];
            std::snprintf(bname, sizeof(bname), "enc.block.%d.out", i);
            named(x, bname);
            eb.dumps.block_mid_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
        if (i == n_layers - 1) {
            char bname[64];
            std::snprintf(bname, sizeof(bname), "enc.block.%d.out", i);
            named(x, bname);
            eb.dumps.block_last_out = x;
            transcribe::debug::mark_tensor_for_dump(x);
        }
    }

    // ----- cat_hidden_layers channel concat (-plus only) -----
    // HF: hidden_states = torch.cat([*exported_hidden_states, hidden_states], dim=-1)
    // i.e. earlier-layer captures FIRST, final hidden LAST, along the
    // channel axis. ggml dim=0 is the channel axis for [hidden, T].
    // Build right-to-left so the final order matches HF: walk captures
    // in reverse, each prepended via concat(exp, acc).
    for (auto it = exported_hidden_states.rbegin();
         it != exported_hidden_states.rend(); ++it)
    {
        x = ggml_concat(ctx, *it, x, /*dim=*/0);
    }

    // ----- Final output (already at T_enc) -----
    named(x, "enc.out");
    eb.out = x;
    eb.dumps.out_named = x;
    ggml_set_output(eb.out);
    transcribe::debug::mark_tensor_for_dump(eb.out);

    eb.graph = ggml_new_graph_custom(ctx, /*size=*/16384, /*grads=*/false);
    if (eb.graph == nullptr) {
        std::fprintf(stderr, "granite encoder: ggml_new_graph_custom failed\n");
        return eb;
    }
    ggml_build_forward_expand(eb.graph, eb.out);
    if (eb.dumps.input_linear_out) ggml_build_forward_expand(eb.graph, eb.dumps.input_linear_out);
    if (eb.dumps.block_0_out)      ggml_build_forward_expand(eb.graph, eb.dumps.block_0_out);
    if (eb.dumps.block_mid_out)    ggml_build_forward_expand(eb.graph, eb.dumps.block_mid_out);
    if (eb.dumps.block_last_out)   ggml_build_forward_expand(eb.graph, eb.dumps.block_last_out);

    return eb;
}

} // namespace transcribe::granite
