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

// LayerNorm + BatchNorm epsilon. Both default to 1e-5 in MLX and in
// NeMo, NOT stored in the GGUF. Hardcoded here to match. If a future
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
    // Fused BN (computed at load time from the raw BN params).
    ggml_tensor * conv_bn_fused_scale = nullptr; // [d_model]
    ggml_tensor * conv_bn_fused_bias  = nullptr; // [d_model]

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

// Scalar knobs for the block forward. All fields required.
struct BlockParams {
    int        d_model;
    int        n_head;
    int        conv_kernel;
    ggml_type  kv_type;      // GGML_TYPE_COUNT = auto (f16 if quant, f32 if f32)
    bool       use_flash;    // false -> manual mul_mat + soft_max + mul_mat
    ConvPolicy policy;
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

// Convolution sub-block: pointwise -> GLU -> depthwise -> BN -> SiLU
// -> pointwise. Operates on post-LayerNorm input; the LN is applied
// by the caller.
ggml_tensor * conv_module(ggml_context *    ctx,
                          ggml_tensor *     x,
                          const BlockView & b,
                          int               conv_kernel,
                          const ConvPolicy & policy);

// Relative-position multi-head self-attention. Flash attention when
// use_flash is true; manual mul_mat + soft_max_ext + mul_mat fallback
// otherwise.
ggml_tensor * rel_pos_mhsa(ggml_context *    ctx,
                           ggml_tensor *     x,
                           ggml_tensor *     pos_emb,
                           const BlockView & b,
                           int               d_model,
                           int               n_head,
                           ggml_type         kv_type,
                           bool              use_flash);

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

// DwStridingSubsampling pre_encode stack. Returns the final
// [d_model, T_enc, 1, 1] tensor or nullptr on a shape-sanity failure.
// `name_prefix`, if non-null, is used to attach ggml_set_name() calls
// to every intermediate (pattern: "<prefix>.conv0", "<prefix>.relu0",
// ...). Pass nullptr to skip naming entirely.
//
// `pre_encode_in_expected` is pre_encode.out_w->ne[0] — passed
// explicitly so this helper can log a useful family-tagged error
// without a back-reference to the weights struct. Set to -1 to skip
// the sanity check.
ggml_tensor * build_pre_encode(ggml_context *        ctx,
                               const PreEncodeView & pe,
                               ggml_tensor *         mel_in,
                               const ConvPolicy &    policy,
                               const char *          name_prefix,
                               const char *          error_tag);

} // namespace transcribe::conformer
