// src/conformer/conformer.h - shared Conformer encoder helpers.
//
// Both Parakeet and Cohere-ASR (and any future family built on a
// NeMo-style Conformer encoder) share the same block topology:
//
//   pre_encode (DwStridingSubsampling: 1 standard conv + 2 depthwise
//               -separable conv pairs, ReLU-interleaved, then a linear
//               projection to d_model)
//   N x conformer block {
//     macaron FF1 (0.5x residual)
//     rel-pos MHSA (1.0x residual)
//     conv module (pointwise-GLU -> depthwise -> BN -> SiLU -> pointwise)
//     macaron FF2 (0.5x residual)
//     final per-block LayerNorm
//   }
//
// This header exposes that machinery as a set of family-agnostic
// helpers driven by three structs:
//
//   - PreEncodeView / BlockView : nullable-pointer projections over
//     whatever per-family weight struct the family's weights.h
//     defines. Families with biases (Cohere) fill the bias slots;
//     bias-free families (Parakeet) leave them nullptr. Every helper
//     that takes a bias pointer treats nullptr as "no bias" and skips
//     the add — this is the optional-bias pattern.
//
//   - ConvPolicy : per-family dispatch choice for the pointwise and
//     depthwise convs. The pointwise policy is identical across
//     families (detect_direct_pw lives here, shared) but the
//     depthwise policy splits between "inside the conformer block"
//     and "inside pre_encode" because the two sites have different
//     tensor shapes and historical backend behavior. Each family
//     populates its own ConvPolicy; the shared helpers never read
//     environment variables directly for dw dispatch.
//
//   - BlockParams : the scalar knobs the block forward needs
//     (d_model, n_head, conv_kernel, kv_type, use_flash, policy).
//
// Sub-helpers (macaron_ff_residual, rel_pos_mhsa, conv_module) are
// also exposed at namespace scope so a family that hand-builds one or
// more blocks for debug-dump purposes (currently Parakeet's block 0)
// can drive the same code paths without going through
// build_conformer_block. No helper in this module ever calls
// ggml_set_name or ggml_set_output on intermediates — naming and
// output marking are the family's responsibility, and the one helper
// that does name intermediates (build_pre_encode) takes an explicit
// name prefix.
//
// The f32-friendly conv helpers (conv_1d_f32, conv_2d_dw_f32,
// conv_1d_dw_f32) exist to work around ggml kernel limitations on
// Metal — see the block comment above conv_2d_dw_f32 in conformer.cpp
// for the full story. Callers should use these, NOT ggml's built-in
// ggml_conv_1d / ggml_conv_2d_dw, whenever the kernel weights are
// f32 and the compute must run on Metal.

#pragma once

#include "ggml.h"

namespace transcribe::conformer {

// ===========================================================================
// Constants
// ===========================================================================

// LayerNorm + BatchNorm epsilon. Both default to 1e-5 in NeMo and
// are NOT stored in the GGUF. Hardcoded here to match. If a future
// variant changes either value, plumb it through BlockParams rather
// than bumping this constant.
constexpr float kLayerNormEps = 1e-5f;

// ===========================================================================
// Views over per-family weight structs
// ===========================================================================

// Nullable-pointer projection over the pre_encode weights. The family
// (parakeet / cohere) fills this from its own weights struct at the
// call site — see to_view() helpers in each encoder.cpp.
struct PreEncodeView {
    ggml_tensor * conv0_w = nullptr; // [KW=3, KH=3, IC=1,        OC=channels]
    ggml_tensor * conv0_b = nullptr; // [channels]
    ggml_tensor * conv2_w = nullptr; // depthwise [KW=3, KH=3, 1, OC=channels]
    ggml_tensor * conv2_b = nullptr; // [channels]
    ggml_tensor * conv3_w = nullptr; // pointwise [1, 1, channels, channels]
    ggml_tensor * conv3_b = nullptr; // [channels]
    ggml_tensor * conv5_w = nullptr; // depthwise
    ggml_tensor * conv5_b = nullptr;
    ggml_tensor * conv6_w = nullptr; // pointwise
    ggml_tensor * conv6_b = nullptr;
    // Final projection from [channels * (num_mels / subsampling)] to d_model.
    ggml_tensor * out_w   = nullptr; // [pre_encode_in, d_model]
    ggml_tensor * out_b   = nullptr; // [d_model]
};

// Nullable-pointer projection over one Conformer block's weights.
// Every `_b` (bias) slot is genuinely optional — leave nullptr for
// families that ship without that bias. The matching helper skips the
// add when the pointer is null.
struct BlockView {
    // Macaron FF1.
    ggml_tensor * norm_ff1_w = nullptr; // [d_model]
    ggml_tensor * norm_ff1_b = nullptr; // [d_model]
    ggml_tensor * ff1_lin1_w = nullptr; // [d_model, d_ff]
    ggml_tensor * ff1_lin1_b = nullptr; // [d_ff], nullable
    ggml_tensor * ff1_lin2_w = nullptr; // [d_ff, d_model]
    ggml_tensor * ff1_lin2_b = nullptr; // [d_model], nullable

    // Self-attention with relative positional encoding.
    ggml_tensor * norm_attn_w = nullptr; // [d_model]
    ggml_tensor * norm_attn_b = nullptr; // [d_model]
    ggml_tensor * attn_q_w    = nullptr; // [d_model, d_model]
    ggml_tensor * attn_q_b    = nullptr; // [d_model], nullable
    ggml_tensor * attn_k_w    = nullptr; // [d_model, d_model]
    ggml_tensor * attn_k_b    = nullptr; // [d_model], nullable
    ggml_tensor * attn_v_w    = nullptr; // [d_model, d_model]
    ggml_tensor * attn_v_b    = nullptr; // [d_model], nullable
    ggml_tensor * attn_out_w  = nullptr; // [d_model, d_model]
    ggml_tensor * attn_out_b  = nullptr; // [d_model], nullable
    ggml_tensor * attn_pos_w  = nullptr; // [d_model, d_model] linear_pos (no bias)
    ggml_tensor * attn_pos_u  = nullptr; // [n_head, head_dim]
    ggml_tensor * attn_pos_v  = nullptr; // [n_head, head_dim]

    // Convolution module. Pointwise conv1 is 2*d_model for GLU.
    ggml_tensor * norm_conv_w         = nullptr; // [d_model]
    ggml_tensor * norm_conv_b         = nullptr; // [d_model]
    ggml_tensor * conv_pw1_w          = nullptr; // [1, d_model, 2*d_model]
    ggml_tensor * conv_pw1_b          = nullptr; // [2*d_model], nullable
    ggml_tensor * conv_dw_w           = nullptr; // [k, 1, d_model]
    ggml_tensor * conv_dw_b           = nullptr; // [d_model], nullable
    ggml_tensor * conv_pw2_w          = nullptr; // [1, d_model, d_model]
    ggml_tensor * conv_pw2_b          = nullptr; // [d_model], nullable
    // Post-depthwise normalisation. Exactly one pair is populated per
    // block, depending on BlockParams::conv_norm_type.
    //   BatchNorm path: conv_bn_fused_scale / conv_bn_fused_bias
    //                   (precomputed at load from raw BN weight/bias +
    //                   running_mean/var). Offline Conformer variants.
    //   LayerNorm path: conv_ln_w / conv_ln_b (raw LN scale/bias; the
    //                   per-channel mean/std is computed at inference).
    //                   Streaming variants (nemotron-speech-streaming).
    ggml_tensor * conv_bn_fused_scale = nullptr; // [d_model]
    ggml_tensor * conv_bn_fused_bias  = nullptr; // [d_model]
    ggml_tensor * conv_ln_w           = nullptr; // [d_model]
    ggml_tensor * conv_ln_b           = nullptr; // [d_model]

    // Macaron FF2.
    ggml_tensor * norm_ff2_w = nullptr;
    ggml_tensor * norm_ff2_b = nullptr;
    ggml_tensor * ff2_lin1_w = nullptr;
    ggml_tensor * ff2_lin1_b = nullptr; // nullable
    ggml_tensor * ff2_lin2_w = nullptr;
    ggml_tensor * ff2_lin2_b = nullptr; // nullable

    // Final per-block layer norm.
    ggml_tensor * norm_out_w = nullptr;
    ggml_tensor * norm_out_b = nullptr;
};

// ===========================================================================
// Policy and params
// ===========================================================================

// Per-family conv dispatch policy. direct_pw is shared
// (detect_direct_pw produces the same answer for every family today)
// but direct_dw splits between block-internal and pre_encode sites:
//
//   - direct_dw_in_block     : the conformer block's 1-D depthwise
//                              conv (after GLU). Parakeet takes the
//                              direct path on every backend; Cohere
//                              takes it only on Vulkan/CUDA and uses
//                              im2col on Metal/CPU. See each family's
//                              detect_direct_dw() for the exact
//                              policy.
//
//   - direct_dw_in_pre_encode : the pre_encode 2-D depthwise conv
//                              (stride-2 downsample). Historically
//                              Parakeet hardcoded the im2col path here
//                              (direct_dw_in_pre_encode = false)
//                              regardless of the in-block flag;
//                              Cohere routes both sites through the
//                              same detect_direct_dw value. Splitting
//                              the two lets each family keep its
//                              historical behavior exactly, no quiet
//                              unification.
// Defaults are deliberately conservative: direct_pw is true because
// pointwise convs are direct mul_mat on every backend, but both
// direct_dw_* default to false (im2col path). A future family that
// default-initializes ConvPolicy{} and forgets to set these will get
// the im2col path — slower than direct on Vulkan but safe on Metal,
// where the direct 2D depthwise kernel is not implemented for all
// shapes and the im2col + mul_mat path is the reliable fallback.
// Parakeet and Cohere populate all three fields explicitly at the
// call site in build_encoder_graph, so the defaults only matter for
// new callers.
struct ConvPolicy {
    bool direct_pw                = true;
    bool direct_dw_in_block       = false;
    bool direct_dw_in_pre_encode  = false;

    // Causal pre_encode convolutions. NeMo's cache-aware streaming
    // (`is_causal=true`) swaps every Conv2d in ConvSubsampling for
    // CausalConv2D, which applies `F.pad(left=k-1, right=stride-1)` on
    // both spatial axes before calling the underlying conv with p=0.
    // For k=3 / s=2 that is (left=2, right=1) per axis — different
    // total padding from the offline (k-1)/2 symmetric path, and
    // shifts both the freq output dim (e.g. 128 → 65 → 33 → 17
    // instead of 64 → 32 → 16) and the time output dim. False on every
    // offline Parakeet variant; true on nemotron-speech-streaming-en.
    bool causal_pre_encode        = false;
};

// Pointwise conv dispatch auto-detect. Identical policy across
// families today (Parakeet and Cohere return the same answer for
// every backend) so this lives here rather than in each encoder.cpp.
// Env overrides: TRANSCRIBE_CONV_NO_DIRECT_PW=1 forces im2col,
// TRANSCRIBE_CONV_DIRECT_PW=1 forces direct.
//
// Policy note (Vulkan): on AMD Renoir, a controlled A/B showed the
// im2col + mul_mat path is ~200 ms per encode faster than the direct
// mul_mat path for f32 weights, so Vulkan defaults to im2col. Metal
// and CPU prefer direct.
bool detect_direct_pw(const char * backend);

// Scalar knobs for the block forward. All fields required except the
// optional local-attention window (att_context_left / att_context_right).
struct BlockParams {
    int        d_model;
    int        n_head;
    int        conv_kernel;
    ggml_type  kv_type;      // GGML_TYPE_COUNT = auto (f16 if quant, f32 if f32)
    bool       use_flash;    // false -> manual mul_mat + soft_max + mul_mat
    ConvPolicy policy;

    // Local attention window. -1 / -1 = full attention (default).
    // Semantics depend on att_context_style below.
    int att_context_left  = -1;
    int att_context_right = -1;

    // Self-attention context style.
    //
    //   Regular         — when both att_context_left/right are >= 0,
    //     pos_emb is expected to have length (left+right+1) and
    //     rel_pos_mhsa pads matrix_bd with -inf rows so positions
    //     outside the band become -inf in the attention scores after
    //     softmax. Matches NeMo's LocalAttRelPositionalEncoding when
    //     T <= 2W+1 and remains correct (band-restricted) when T
    //     exceeds the window. Default for every offline variant.
    //
    //   ChunkedLimited  — pos_emb stays at the full 2T-1 length and
    //     the caller supplies a precomputed F16 mask of shape
    //     [T_k, T_q, 1, 1] in `attn_chunked_mask` that
    //     rel_pos_mhsa adds onto matrix_bd before flash_attn. Chunk
    //     size = right+1, left_chunks = left/chunk_size; the mask is
    //     0 inside the allowed [q_chunk - left_chunks, q_chunk] band
    //     and -INF elsewhere. NeMo cache-aware streaming.
    enum class AttContextStyle { Regular, ChunkedLimited };
    AttContextStyle att_context_style = AttContextStyle::Regular;

    // Optional precomputed mask for ChunkedLimited. The caller builds
    // this as a graph input shape [T_k, T_q, 1, 1] F16 (broadcasts
    // across heads) and uploads the host-computed pattern after the
    // compute buffer is allocated. Ignored for AttContextStyle::Regular.
    ggml_tensor * attn_chunked_mask = nullptr;

    // Optional key-padding mask for variable-length offline batching.
    // Shape [T_k, 1, 1, B] F32 (0 on real keys, -INF on padded keys);
    // broadcasts across queries (ne[1]) and heads (ne[2]) and is added
    // onto the attention score matrix in rel_pos_mhsa before flash /
    // soft_max, exactly like attn_chunked_mask. Null for single-shot and
    // same-length batches. Keeps a real query from attending to another
    // utterance's padded tail frames.
    ggml_tensor * attn_pad_mask = nullptr;

    // Depthwise conv padding. -1 / -1 means "centred (k-1)/2" on both
    // sides (every offline variant). Otherwise the depthwise conv
    // uses (conv_context_left, conv_context_right) directly. NeMo's
    // `conv_context_size = "causal"` emits (kernel-1, 0).
    int conv_context_left  = -1;
    int conv_context_right = -1;

    // Optional post-GLU valid-frame mask for the conv module. Shape
    // [T, 1, 1, 1], values 1.0 for valid frames and 0.0 for padded
    // overhang. NeMo applies pad_mask after pointwise_conv1+GLU and
    // before the depthwise convolution; null keeps the historical path.
    ggml_tensor * conv_pad_mask = nullptr;

    // Conv-module normalisation choice. BatchNorm uses fused scale +
    // bias precomputed at load time (BlockView::conv_bn_fused_*).
    // LayerNorm computes per-channel mean/std at inference and uses
    // BlockView::conv_ln_* as the affine scale/bias.
    enum class ConvNormType { BatchNorm, LayerNorm };
    ConvNormType conv_norm_type = ConvNormType::BatchNorm;

    // ---- Streaming (cache-aware) inputs/outputs ----
    //
    // All four pointers are nullptr in offline mode and the block
    // runs the existing graph topology. When non-null, the block
    // runs the streaming path:
    //
    //   streaming_channel_in   per-layer cache_last_channel tensor
    //     from the previous chunk (persistent backend buffer). Shape
    //     [d_model, T_cache, 1, 1]. Concat-prepended onto the post-
    //     attn-LN x before Q/K/V projection ("virtual T" approach):
    //     the block runs attention on T_virtual = T_cache + T_q_new
    //     positions and slices the output to the last T_q_new rows.
    //
    //   streaming_channel_out  output tensor (allocated by the caller
    //     in the per-call compute_ctx) that rel_pos_mhsa fills via
    //     ggml_cpy with the tail of x_norm (size T_q_new, the new
    //     cache slot per NeMo's q_keep_size = T_new rule with
    //     cache_drop_size = 0). After graph_compute, the driver
    //     rotates this into the persistent cache buffer.
    //
    //   streaming_time_in      per-layer cache_last_time tensor from
    //     the previous chunk. Shape [k_minus_1, d_model, 1, 1].
    //     Replaces the zero left-pad in the depthwise conv (the
    //     causal pad on streaming variants is (k-1, 0); the k-1 left
    //     frames are now the previous chunk's post-pw1+GLU tail).
    //
    //   streaming_time_out     output tensor (compute_ctx) that
    //     conv_module fills with the last k_minus_1 frames of the
    //     post-pw1+GLU input (which becomes the next chunk's left
    //     pad).
    //
    // The pos_emb tensor (passed separately to rel_pos_mhsa) and the
    // attn_chunked_mask (above) are sized for the virtual T when
    // streaming, not the offline T_enc. The caller is responsible for
    // building them at the correct size.
    ggml_tensor * streaming_channel_in  = nullptr;
    ggml_tensor * streaming_channel_out = nullptr;
    ggml_tensor * streaming_time_in     = nullptr;
    ggml_tensor * streaming_time_out    = nullptr;

    // When streaming, the number of new (post-pre-encode) encoder
    // frames being produced this call. Used to:
    //   - slice rel_pos_mhsa's attention output back to the new rows
    //   - decide the source slice for streaming_channel_out / _time_out
    // Ignored in offline mode (streaming_channel_in == nullptr).
    int streaming_T_q_new = 0;

    // Optional streaming KV cache for this block (all four set
    // together, or none): persistent [d_model, T_cache] tensors holding
    // the previous chunks' pre-projected attention keys/values (bias
    // included). When set, the block computes K/V projections only for
    // the new frames, concatenates the cache in front, runs attention
    // against the combined window, and emits the rotated cache via
    // ggml_cpy into the *_out tensors (the driver copies them into the
    // persistent cache after compute, exactly like streaming_channel_*).
    // streaming_channel_in/_out are ignored in this mode — the channel
    // cache is the recompute-K/V representation of the same state.
    ggml_tensor * streaming_kv_k_in  = nullptr;
    ggml_tensor * streaming_kv_v_in  = nullptr;
    ggml_tensor * streaming_kv_k_out = nullptr;
    ggml_tensor * streaming_kv_v_out = nullptr;

    // Optional precomputed rel-pos projection for this block: the
    // [head_dim, pos_len, n_head, 1] result of
    // cont(permute(reshape(attn_pos_w @ pos_emb))) from a previous
    // chunk with identical geometry. When non-null, rel_pos_mhsa
    // consumes it directly and never touches pos_emb (which may then
    // be null). Streaming-only memoization; offline paths leave this
    // null.
    ggml_tensor * streaming_pos_proj_in = nullptr;

    // Graph the helpers use to forward-expand their cache-write cpy
    // nodes. The streaming cache outputs are SIDE outputs (not
    // reachable from the encoder's `out` tensor), so they have to be
    // added to the graph explicitly. The caller sets this to the
    // graph it'll later run on; the helpers do
    // ggml_build_forward_expand(streaming_graph, cpy_node) when they
    // emit a cache write. Required when streaming_*_out is non-null.
    ggml_cgraph * streaming_graph = nullptr;
};

// ===========================================================================
// Low-level helpers (exposed for family-specific hand-built blocks)
// ===========================================================================

// Attach a debug name to a tensor. Safe on nullptr.
ggml_tensor * named(ggml_tensor * t, const char * name);

// LayerNorm with affine: y = gamma * (x - mean) / sqrt(var + eps) + beta.
// `gamma` is required; `beta` may be null for bias-free LN.
ggml_tensor * layer_norm(ggml_context * ctx,
                         ggml_tensor *  x,
                         ggml_tensor *  gamma,
                         ggml_tensor *  beta);

// Two-linear FeedForward: y = Linear2(SiLU(Linear1(x))). Biases are
// optional — pass nullptr to skip the add.
ggml_tensor * feed_forward(ggml_context * ctx,
                           ggml_tensor *  x,
                           ggml_tensor *  lin1_w,
                           ggml_tensor *  lin1_b,
                           ggml_tensor *  lin2_w,
                           ggml_tensor *  lin2_b);

// Macaron half-residual FF: x + 0.5 * FeedForward(LayerNorm(x)).
ggml_tensor * macaron_ff_residual(ggml_context * ctx,
                                  ggml_tensor *  x,
                                  ggml_tensor *  norm_w,
                                  ggml_tensor *  norm_b,
                                  ggml_tensor *  lin1_w,
                                  ggml_tensor *  lin1_b,
                                  ggml_tensor *  lin2_w,
                                  ggml_tensor *  lin2_b);

// Shaw / Transformer-XL relative-position skew. Turns a
// [pos_len, T_q, H, 1] score matrix into a [pos_len, T_q, H, 1]
// matrix rotated so column k holds the score for relative offset k.
ggml_tensor * rel_shift(ggml_context * ctx, ggml_tensor * x);

// f32-friendly Conv1D. Use this instead of ggml_conv_1d for fp32
// kernels — see the long comment above conv_2d_dw_f32 in
// conformer.cpp for the Metal backstory.
ggml_tensor * conv_1d_f32(ggml_context * ctx,
                          ggml_tensor *  kernel,
                          ggml_tensor *  data,
                          int            stride,
                          int            padding,
                          int            dilation);

// f32-friendly 2D depthwise conv. Use this instead of
// ggml_conv_2d_dw / ggml_conv_2d_dw_direct for fp32 kernels when the
// compute may run on Metal — the full story is in the comment above
// the implementation.
ggml_tensor * conv_2d_dw_f32(ggml_context * ctx,
                             ggml_tensor *  kernel,
                             ggml_tensor *  data,
                             int            s0, int s1,
                             int            p0, int p1,
                             int            d0, int d1);

// f32-friendly 1D depthwise conv. Same Metal reasoning.
ggml_tensor * conv_1d_dw_f32(ggml_context * ctx,
                             ggml_tensor *  kernel,
                             ggml_tensor *  data,
                             int            stride,
                             int            padding,
                             int            dilation);

// Fused BatchNorm: y = x * scale + bias, with 1-D scale/bias
// reshaped for broadcast across the time axis of [T, C, 1, 1].
// Both scale_1d and bias_1d are REQUIRED (not nullable) — they come
// from the load-time BN fusion and must always be present. Callers
// that need a no-op path should skip the call entirely rather than
// pass null.
ggml_tensor * fused_batch_norm(ggml_context * ctx,
                               ggml_tensor *  x,
                               ggml_tensor *  scale_1d,
                               ggml_tensor *  bias_1d);

// Add a 1-D bias [C] to a 4-D conv output [W, H, C, N] via broadcast.
// Nullable: passing bias_1d == nullptr is a no-op that returns
// conv_out unchanged.
ggml_tensor * add_conv_bias(ggml_context * ctx,
                            ggml_tensor *  conv_out,
                            ggml_tensor *  bias_1d);

// ===========================================================================
// Sub-blocks (exposed for families that hand-build a block)
// ===========================================================================

// Convolution sub-block: pointwise -> GLU -> optional valid-frame mask
// -> depthwise -> BN/LN -> SiLU -> pointwise. Operates on post-LayerNorm
// input; the LN is applied by the caller. Reads
// BlockParams::conv_context_{left,right} (depthwise padding),
// BlockParams::conv_pad_mask, and BlockParams::conv_norm_type
// (BN vs LN).
ggml_tensor * conv_module(ggml_context *      ctx,
                          ggml_tensor *       x,
                          const BlockView &   b,
                          const BlockParams & params);

// Relative-position multi-head self-attention. Flash attention when
// use_flash is true; manual mul_mat + soft_max_ext + mul_mat fallback
// otherwise. Reads attention-context fields from BlockParams:
//
//   Regular        + att_context_{left,right} >= 0 -> local sliding
//     window. pos_emb has length (left+right+1); helper pads matrix_bd
//     with -inf rows so keys outside the band drop out of softmax.
//
//   ChunkedLimited                                  -> caller supplies
//     a precomputed [T_k, T_q] F16 mask in BlockParams::attn_chunked_mask
//     and pos_emb has full 2T-1 length. The mask is added onto
//     matrix_bd before flash_attn / soft_max_ext.
//
// Otherwise (Regular + att_context_* == -1) -> unrestricted attention.
// x_q (optional): query-only activation [d_model, T_q, B] with T_q !=
// x->ne[1]. When non-null, K/V (and the keys of the score matrix) come
// from `x` while queries come from `x_q` — the cache-aware streaming
// case where only the new frames need attention output but the cached
// frames still serve as keys/values. Requirements in this mode:
//   - pos_emb length must be T_q + T_k - 1 with the zero-offset row at
//     index T_k - 1 (row i holds relative offset (T_k - 1) - i)
//   - attn_chunked_mask (if any) must be [T_k, T_q]
//   - the flash path is bypassed (manual mul_mat + soft_max only)
// Null (default) keeps the historical self-attention behavior.
//
// k_full / v_full (optional, set together): pre-projected keys/values
// [d_model, T_k, 1, 1] (bias already applied). When set, the helper
// skips its own K/V projections entirely, queries come from `x`, and
// the same rectangular requirements as x_q apply. Streaming KV-cache
// path; mutually exclusive with x_q.
ggml_tensor * rel_pos_mhsa(ggml_context *      ctx,
                           ggml_tensor *       x,
                           ggml_tensor *       pos_emb,
                           const BlockView &   b,
                           const BlockParams & params,
                           ggml_tensor *       x_q = nullptr,
                           ggml_tensor *       k_full = nullptr,
                           ggml_tensor *       v_full = nullptr);

// ===========================================================================
// Top-level block + pre-encode
// ===========================================================================

// Optional per-block observer hook. `build_conformer_block` calls
// `on_point` at the five canonical residual / post-LN points during
// the block forward, with `tag` set to a compile-time constant
// string:
//
//   "after_ff1"  — after the FF1 macaron residual merge
//   "after_attn" — after the attention residual merge
//   "after_conv" — after the conv module residual merge
//   "after_ff2"  — after the FF2 macaron residual merge
//   "out"        — after the final per-block LayerNorm (== block out)
//
// Callers typically use this to attach a ggml name, stash the
// tensor in a dump-slot struct, and/or mark the tensor for graph
// output. The observer is the supported replacement for
// hand-building a block out of the exposed sub-helpers just to get
// named dump points: it keeps the block implementation single-source
// and removes the lockstep maintenance hazard.
//
// Pointer-of-struct + user pointer (not std::function) keeps this
// header C-ABI-friendly and free of <functional>. Pass nullptr for
// the observer argument to skip observation entirely.
struct BlockObserver {
    void (*on_point)(void *        user,
                     const char *  tag,
                     ggml_tensor * t) = nullptr;
    void * user = nullptr;
};

// Full Conformer block forward: FF1 -> attn -> conv -> FF2 -> LN_out.
// `pos_emb` is shared across every block in an encoder — the caller
// computes it once and threads it through. `obs` may be null; when
// non-null its `on_point` callback fires at each of the five
// canonical points listed on BlockObserver.
ggml_tensor * build_conformer_block(ggml_context *        ctx,
                                    ggml_tensor *         x,
                                    ggml_tensor *         pos_emb,
                                    const BlockView &     b,
                                    const BlockParams &   params,
                                    const BlockObserver * obs = nullptr);

// Per-stage valid-frame masks for variable-length offline batching of the
// pre_encode conv stem. Zero-padding utterances to a common length is NOT
// transparent through this conv stack: the convs carry bias + ReLU, so the
// padded region becomes non-zero after the first conv and then leaks into
// each utterance's last VALID frame via the next stride-2 conv's receptive
// field — corrupting exactly the boundary frame (it flips trailing CTC
// tokens). The fix (NeMo's masked subsampling) is to zero each conv
// intermediate's padded time region after every ReLU so the next conv sees
// zeros beyond each utterance's boundary, exactly like a standalone run.
//
// When a non-null PreEncodeValidMasks is passed, build_pre_encode allocates
// three time-valid mask graph inputs (one per ReLU stage) sized
// ne=[1, H_stage, 1, B] (broadcast over freq + channels), applies them, and
// writes the handles back here for the driver to fill (1.0 on valid frames,
// 0.0 on padded). Only meaningful for the non-causal (offline) pre_encode;
// the caller should skip it for causal/streaming variants whose output-length
// formula differs.
struct PreEncodeValidMasks {
    ggml_tensor * mask_s1 = nullptr;  // after relu0
    ggml_tensor * mask_s2 = nullptr;  // after relu3
    ggml_tensor * mask_s3 = nullptr;  // after relu6
};

// DwStridingSubsampling pre_encode stack. Returns the final
// [d_model, T_enc, 1, 1] tensor or nullptr on a shape-sanity failure.
// `name_prefix`, if non-null, is used to attach ggml_set_name() calls
// to every intermediate (pattern: "<prefix>.conv0", "<prefix>.relu0",
// ...). Pass nullptr to skip naming entirely.
//
// Uses pe.out_w->ne[0] as a final d_model sanity check. `error_tag`
// supplies the family name used in the diagnostic on shape mismatch.
//
// valid_masks (optional): when non-null, enables the masked-subsampling path
// above and is filled with the three mask input handles.
ggml_tensor * build_pre_encode(ggml_context *        ctx,
                               const PreEncodeView & pe,
                               ggml_tensor *         mel_in,
                               const ConvPolicy &    policy,
                               const char *          name_prefix,
                               const char *          error_tag,
                               PreEncodeValidMasks * valid_masks = nullptr);

} // namespace transcribe::conformer
