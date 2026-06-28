// transcribe-quantize policy: per-tensor bucket classification + per-preset
// quant-type resolution.
//
// The policy table is the single source of truth for "what dtype does
// tensor X end up at under preset Y". Python converters only write
// source/reference-dtype GGUFs; every lossy quant decision lives here.
//
// See docs/tools/quantization.md for the external-facing description
// (preset table, bucket rules, per-family overrides). Keep that doc in
// sync when adding or changing presets.

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace transcribe::quantize {

// Tensor role. classify_tensor() maps a canonical GGUF tensor name to
// exactly one bucket; each preset maps each bucket to one ggml_type.
enum class Bucket {
    Linear,  // ggml_mul_mat operands — full linear allowlist.
    Embed,   // Decoder token embedding. Mixed _M presets bump to Q6_K
             // where shape-compatible; otherwise the preset's
             // linear_fallback (Q8_0 for all K presets).
    ConvPw,  // 1×1 pointwise convs in conformer blocks (F16 on im2col + mul_mat).
    Conv,    // Non-pointwise conv kernels (F32 / F16 — no quant im2col).
    Norm,    // Biases, LayerNorm/BN, pos_bias, pos_enc, frontend buffers (F32).
};

// Map a canonical tensor name to its bucket. Substring-based; family-
// specific overrides (e.g. the Cohere tied embedding) live here at the
// top of the function, clearly marked.
Bucket classify_tensor(const std::string & name);

// A single preset. Each field is the target ggml_type for one bucket
// (with a couple of bucket sub-cases: linear_fallback for dims that
// don't divide the quant's block size, and linear_attn_out for the _M
// recipe's attention-output bump).
struct Preset {
    const char * name;
    // Common case: linear operand, inner dim divides the K-quant block
    // size, not the attention output projection.
    ggml_type    linear_main;
    // Fallback when the inner dim doesn't divide linear_main's block
    // size (e.g. Parakeet predictor/joint at ne0=640 vs K-quant block
    // 256). Q8_0 for every K preset — see policy.cpp for the rationale.
    // Legacy block quants (block 32) set this to F16 as a tripwire;
    // it should never trigger in practice.
    ggml_type    linear_fallback;
    // Attention output projection (attn.linear_out.weight). _M recipes
    // bump this to Q8_0 for quality; uniform presets leave it at
    // linear_main.
    ggml_type    linear_attn_out;
    // Decoder token embedding (Embed bucket). GGML_TYPE_COUNT means "no
    // override" — resolve to linear_main.
    ggml_type    linear_embed;
    // Non-pointwise conv kernels.
    ggml_type    conv;
    // 1×1 pointwise conv kernels.
    ggml_type    conv_pw;
    // Norm-bucket tensors (always F32 today; field exists for symmetry).
    ggml_type    norm;
    // llama-style file_type tag written to general.file_type.
    uint32_t     file_type;
};

// Look up a preset by name. Case-insensitive so "Q4_K_M", "q4_k_m",
// and any mixed case all resolve the same. Returns nullptr if the name
// is not in the table.
const Preset * find_preset(const char * name);

// Iterate the preset table. Exposes the underlying array pointer +
// length so callers can print the full list (usage) without taking a
// dependency on the storage layout.
const Preset * preset_table(size_t & n_out);

// Resolve the per-tensor target ggml_type. Takes the source tensor's
// ne[0] so the linear_fallback path can trip when the inner dim
// doesn't divide the target block size.
ggml_type resolve_target_type(const Preset & preset, const std::string & name, int64_t ne0);

}  // namespace transcribe::quantize
