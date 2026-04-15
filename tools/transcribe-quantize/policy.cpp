// transcribe-quantize policy implementation.
//
// Keep this file in sync with docs/tools/quantization.md. The doc
// describes the preset table row-by-row; this file *is* that table.

#include "policy.h"

#include "ggml.h"

#include <cstring>
#include <cctype>

namespace transcribe::quantize {

namespace {

// Substring helpers. std::string::find returns npos if not found.
inline bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}
inline bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n &&
           std::memcmp(s.data() + s.size() - n, suffix, n) == 0;
}

// Case-insensitive C-string equality.
bool iequals(const char * a, const char * b) {
    while (*a && *b) {
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

} // namespace

// ---------------------------------------------------------------------------
// Family overrides (scoped, not generalized)
// ---------------------------------------------------------------------------
//
// Per-family rules live here as explicit name checks, not as pattern
// generalizations. Each override is one-line with a comment explaining
// *which* family and *why* — revisit when 3+ families share the same
// override pattern.

Bucket classify_tensor(const std::string & name) {
    // --- Cohere: tied token embedding ---
    // dec.embed.token.weight doubles as the output projection (tied
    // embedding). Under _M presets it bumps to Q6_K — without this,
    // WER regresses measurably. head.bias stays in Norm (it's a 1D
    // per-logit offset after the matmul; loader requires F32).
    if (name == "dec.embed.token.weight") {
        return Bucket::Embed;
    }

    // --- Generic rules (apply to every family) ---

    // Biases of every kind — short and precision-sensitive.
    if (ends_with(name, ".bias")) {
        return Bucket::Norm;
    }
    // BatchNorm: bn.{weight, bias, running_mean, running_var}.
    if (contains(name, ".bn.")) {
        return Bucket::Norm;
    }
    // LayerNorm scale (norm_ff1.weight, norm_attn.weight, norm_conv.weight,
    // norm_out.weight, norm_self.weight, norm_cross.weight, norm_ff.weight,
    // etc). The .bias case was already caught above.
    if (contains(name, "norm_") && ends_with(name, ".weight")) {
        return Bucket::Norm;
    }
    // Cohere: dec.embed.norm.weight and dec.final_norm.weight — layer
    // norm scale tensors that use "." instead of "_" as separator.
    if (ends_with(name, ".final_norm.weight") ||
        ends_with(name, ".embed.norm.weight"))
    {
        return Bucket::Norm;
    }
    // Per-head positional biases — added directly to fp32 q via ggml_add
    // inside rel_pos_mhsa.
    if (ends_with(name, ".pos_bias_u") || ends_with(name, ".pos_bias_v")) {
        return Bucket::Norm;
    }
    // Cohere: sinusoidal positional encoding table, precision-sensitive.
    if (ends_with(name, ".pos_enc")) {
        return Bucket::Norm;
    }
    // Cohere: mel frontend buffers (filterbank + window) — stored as
    // F32 by the converter and consumed as-is by the mel stage.
    if (name == "frontend.mel_filterbank" || name == "frontend.window") {
        return Bucket::Norm;
    }
    // Conformer-block 1×1 pointwise convs — split out from Conv so
    // they can run at F16 on the im2col+matmul path. Matches
    // "enc.blocks.<N>.conv.pointwise1.weight" and ".pointwise2.weight"
    // but intentionally NOT any pre-encode convs (different names like
    // enc.pre_encode.conv.0.weight).
    if ((ends_with(name, ".conv.pointwise1.weight") ||
         ends_with(name, ".conv.pointwise2.weight")) &&
        contains(name, "enc.blocks."))
    {
        return Bucket::ConvPw;
    }
    // Conv kernels: enc.pre_encode.conv.{0,2,3,5,6}.weight and
    // enc.blocks.{i}.conv.depthwise.weight.
    // ".conv." (with dots on both sides) distinguishes these from
    // norm_conv.weight which we already routed above.
    if (contains(name, ".conv.") && ends_with(name, ".weight")) {
        return Bucket::Conv;
    }
    return Bucket::Linear;
}

// ---------------------------------------------------------------------------
// Preset table
// ---------------------------------------------------------------------------
//
// Names match llama.cpp / llama-quantize exactly (uppercase). Columns:
//
//   name, linear_main, linear_fallback, linear_attn_out, linear_embed,
//   conv, conv_pw, norm, file_type
//
// linear_embed = GGML_TYPE_COUNT means "no override" — resolve_target_type()
// falls back to linear_main for Embed tensors in that case. This keeps
// the uniform presets (F16, Q8_0, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1)
// consistent across Linear and Embed while letting _M presets bump
// embeddings to Q6_K.
//
// For the legacy blockwise quants (Q4_0/1, Q5_0/1): attn_out stays at
// linear_main (these presets are uniform accuracy/size tradeoffs, not
// llama.cpp's mixed recipes). ConvPw stays at F16 because 1×1 pointwise
// convs benefit from f16 matmul shaders and the file-size cost is
// ~2 MB across all of them.
//
// file_type values mirror LlamaFileType / GGML_FTYPE (see
// refs/ggml-org/llama.cpp/include/llama.h).

namespace {

const Preset kPresets[] = {
    // Uniform fp tiers.
    {"F16",    GGML_TYPE_F16,  GGML_TYPE_F16, GGML_TYPE_F16,  GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_F16*/    1 },

    // Legacy blockwise quants. Block size 32 — any sensible inner dim
    // divides cleanly, so linear_fallback is a formality.
    {"Q4_0",   GGML_TYPE_Q4_0, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q4_0*/   2 },
    {"Q4_1",   GGML_TYPE_Q4_1, GGML_TYPE_F16, GGML_TYPE_Q4_1, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q4_1*/   3 },
    {"Q5_0",   GGML_TYPE_Q5_0, GGML_TYPE_F16, GGML_TYPE_Q5_0, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q5_0*/   8 },
    {"Q5_1",   GGML_TYPE_Q5_1, GGML_TYPE_F16, GGML_TYPE_Q5_1, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q5_1*/   9 },
    {"Q8_0",   GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q8_0*/   7 },

    // K-quants. Block size 256; linear_fallback drops to F16 when ne0
    // doesn't divide 256 (Parakeet predictor/joint at ne0=640).
    {"Q6_K",   GGML_TYPE_Q6_K, GGML_TYPE_F16, GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q6_K*/  18 },
    {"Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,  GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q5_K_M*/ 17 },
    {"Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,  GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q4_K_M*/ 15 },
};

constexpr size_t kPresetCount = sizeof(kPresets) / sizeof(kPresets[0]);

} // namespace

const Preset * find_preset(const char * name) {
    for (size_t i = 0; i < kPresetCount; ++i) {
        if (iequals(kPresets[i].name, name)) {
            return &kPresets[i];
        }
    }
    return nullptr;
}

const Preset * preset_table(size_t & n_out) {
    n_out = kPresetCount;
    return kPresets;
}

ggml_type resolve_target_type(const Preset & preset,
                              const std::string & name,
                              int64_t ne0)
{
    const Bucket b = classify_tensor(name);
    switch (b) {
        case Bucket::Norm:   return preset.norm;
        case Bucket::Conv:   return preset.conv;
        case Bucket::ConvPw: return preset.conv_pw;
        case Bucket::Embed: {
            // If the preset has no explicit embed override, fall back
            // to linear_main (with its own fallback when misaligned).
            ggml_type target = (preset.linear_embed != GGML_TYPE_COUNT)
                ? preset.linear_embed
                : preset.linear_main;
            const int64_t blk = ggml_blck_size(target);
            if (blk > 1 && (ne0 % blk) != 0) {
                return preset.linear_fallback;
            }
            return target;
        }
        case Bucket::Linear: {
            // attn.linear_out.weight bumped per _M recipe.
            if (ends_with(name, "attn.linear_out.weight")) {
                return preset.linear_attn_out;
            }
            // Fall back when inner dim doesn't divide the chosen
            // quant's block size.
            const int64_t blk = ggml_blck_size(preset.linear_main);
            if (blk > 1 && (ne0 % blk) != 0) {
                return preset.linear_fallback;
            }
            return preset.linear_main;
        }
    }
    return preset.norm; // unreachable
}

} // namespace transcribe::quantize
