// src/granite_conformer/shaw_attn.h - shared Shaw block-local attention.
//
// Reused by both `src/arch/granite/encoder.cpp` (AR audio-LLM) and
// `src/arch/granite_nar/encoder.cpp` (NAR editor). The two families
// share an audio encoder up to the projector; this is the only
// Granite-specific Conformer op without an analog in `src/conformer/`.
//
// Mirrors `GraniteSpeechConformerAttention.forward`:
//
//   h = pre_norm(x)                                # at T_enc
//   if remainder > 0: h = F.pad(h, (0, 0, 0, pad)) # zero-pad to T_pad
//   q = to_q(h); kv = to_kv(h); k, v = kv.chunk(2, dim=-1)
//   ... block-local attention with Shaw bias + last-block pad mask ...
//   out = out.reshape(..., T_pad, inner_dim)
//   out = to_out(out[:, :num_features, :])         # slice back to T_enc
//   return out                                     # at T_enc
//
// The slice happens BEFORE to_out (upstream order), so the following
// conv module's depthwise kernel never pulls pad-row garbage into
// valid frames near the tail.

#pragma once

#include "ggml.h"

namespace transcribe::granite_conformer {

// Per-block Shaw attention weights. Both `GraniteEncBlock` and
// `GraniteNarEncBlock` carry the same set of tensors with the same
// field names; this packs them so the helper has no compile-time
// dependency on either family's weights struct.
struct ShawAttnWeights {
    ggml_tensor * norm_attn_w     = nullptr;  // [hidden]
    ggml_tensor * norm_attn_b     = nullptr;  // [hidden]      (nullable)
    ggml_tensor * attn_q_w        = nullptr;  // [hidden, inner_dim]
    ggml_tensor * attn_kv_w       = nullptr;  // [hidden, 2*inner_dim]  (fused K|V)
    ggml_tensor * attn_rel_pos_emb = nullptr; // [head_dim, 2*max_pos_emb+1]
    ggml_tensor * attn_out_w      = nullptr;  // [inner_dim, hidden]
    ggml_tensor * attn_out_b      = nullptr;  // [hidden]      (nullable)
};

// Build the Shaw block-local attention sub-graph.
//
// Inputs:
//   x            : [d_model, T_enc] pre-norm input.
//   zero_pad     : [d_model, T_pad - T_enc] f32 zeros, or nullptr when
//                  T_enc % context_size == 0.
//   dists        : [context_size * context_size] int32 Shaw lookup
//                  indices (`clamp(k - q + max_pos_emb, 0, 2*max_pos_emb)`).
//   pad_mask_3d  : [context_size, context_size, num_blocks] f32 additive
//                  mask (all-zero except the last slice, which has -INF
//                  on pad-K columns).
//   w            : per-block weights (see ShawAttnWeights).
//   n_heads, head_dim, context_size, num_blocks, T_enc : shape scalars.
//   layer_norm_eps : eps for the pre-norm (both Granite families use
//                    1e-5; expose as a parameter so callers stay in
//                    control).
//
// Returns: [d_model, T_enc] (matches input shape). Returns nullptr on
// shape error (with a stderr diagnostic).
ggml_tensor * shaw_block_attn(ggml_context *           ctx,
                              ggml_tensor *            x,
                              ggml_tensor *            zero_pad,
                              ggml_tensor *            dists,
                              ggml_tensor *            pad_mask_3d,
                              const ShawAttnWeights &  w,
                              int                      n_heads,
                              int                      head_dim,
                              int                      context_size,
                              int                      num_blocks,
                              int                      T_enc,
                              float                    layer_norm_eps);

} // namespace transcribe::granite_conformer
