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
    // --- Decoder token embeddings ---
    // Cohere and Qwen3-ASR tie the input embedding to the LM head. Under
    // _M presets it bumps to Q6_K — without this, WER regresses
    // measurably. Cohere uses dec.embed.token.weight; Qwen3-ASR and
    // Whisper use the llama.cpp-style dec.token_embd.weight. head.bias
    // stays in Norm (it's a 1D per-logit offset after the matmul; loader
    // requires F32).
    if (name == "dec.embed.token.weight" ||
        name == "dec.token_embd.weight")
    {
        return Bucket::Embed;
    }

    // --- Generic rules (apply to every family) ---

    // Biases of every kind — short and precision-sensitive.
    if (ends_with(name, ".bias")) {
        return Bucket::Norm;
    }
    // BatchNorm: bn.{weight, bias, running_mean, running_var}. The
    // ".bn." form matches NeMo-style tensor names; the "conv_bn." form
    // matches granite/granite_nar where the conv module uses an
    // underscore separator. Also catch running stats (running_mean /
    // running_var) which need to stay F32 for the at-load BN-fusion math.
    if (contains(name, ".bn.") ||
        contains(name, "conv_bn.") ||
        ends_with(name, ".running_mean") ||
        ends_with(name, ".running_var"))
    {
        return Bucket::Norm;
    }
    // LayerNorm scale (norm_ff1.weight, norm_attn.weight, norm_conv.weight,
    // norm_out.weight, norm_self.weight, norm_cross.weight, norm_ff.weight,
    // etc). The .bias case was already caught above.
    if (contains(name, "norm_") && ends_with(name, ".weight")) {
        return Bucket::Norm;
    }
    // Mirror-image: name ends in "_norm.weight" (e.g. granite_nar's
    // prj.out_norm.weight). Distinct from the "norm_" prefix rule above.
    if (ends_with(name, "_norm.weight")) {
        return Bucket::Norm;
    }
    // Lists of layer-norms indexed by integer: granite_nar's per-encoder-
    // layer projector LNs live at `prj.layer_norms.<N>.weight`. The "s"
    // breaks the "_norm." substring rule above.
    if (contains(name, "_norms.") && ends_with(name, ".weight")) {
        return Bucket::Norm;
    }
    // Cohere: dec.embed.norm.weight and dec.final_norm.weight — layer
    // norm scale tensors that use "." instead of "_" as separator.
    if (ends_with(name, ".final_norm.weight") ||
        ends_with(name, ".embed.norm.weight"))
    {
        return Bucket::Norm;
    }
    // Qwen3-style norm weights that don't match the "norm_" prefix rule
    // (per-head q_norm/k_norm on attention, output RMSNorm before the
    // tied head, pre/post encoder layer norms).
    if (ends_with(name, ".q_norm.weight") ||
        ends_with(name, ".k_norm.weight") ||
        ends_with(name, ".output_norm.weight") ||
        ends_with(name, ".ln_post.weight") ||
        ends_with(name, ".ln_pre.weight"))
    {
        return Bucket::Norm;
    }
    // Canary decoder layer norms: dec.layer.<N>.norm{1,2,3}.weight —
    // numbered (no underscore), so the generic "norm_" rule misses them.
    // Also dec.norm.weight (final post-decoder layer norm).
    if (ends_with(name, ".norm1.weight") ||
        ends_with(name, ".norm2.weight") ||
        ends_with(name, ".norm3.weight") ||
        name == "dec.norm.weight")
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
    // granite_nar projector learned parameters: the 3-query embedding
    // (prj.query, [hidden, 3, 1]) and the 15-frame window positional
    // bias (prj.window_positions, [hidden, 15, 1]). Both are added
    // directly into F32 activations inside the projector graph, and the
    // loader uses an F32/F16/BF16-only find_tensor for both. Keep them
    // at the same dtype as the .bias tensors.
    if (name == "prj.query" || name == "prj.window_positions") {
        return Bucket::Norm;
    }
    // Whisper: encoder sinusoidal pos_emb (1500 x d_model) and decoder
    // learned pos_emb (448 x d_model). Both are small and added directly
    // to the residual stream every layer; quantizing them costs accuracy
    // for negligible file-size savings. Keep at F32 across all presets.
    if (name == "enc.pos_emb.weight" || name == "dec.pos_emb.weight") {
        return Bucket::Norm;
    }
    // Cohere: mel frontend buffers (filterbank + window) — stored as
    // F32 by the converter and consumed as-is by the mel stage.
    if (name == "frontend.mel_filterbank" || name == "frontend.window") {
        return Bucket::Norm;
    }
    // SenseVoice: per-feature CMVN shift/scale (1D, d_input). Applied
    // additively/multiplicatively to the LFR-stacked mel frame; loader
    // requires F32, and the tensors are too small (560 elements) to
    // benefit from quantization.
    if (name == "frontend.cmvn.shift" || name == "frontend.cmvn.scale") {
        return Bucket::Norm;
    }
    // SenseVoice: encoder prefix-token embedding (16 × d_input = 16×560)
    // looked up via ggml_get_rows for the [lid, event, emo, textnorm]
    // prepend. Loader requires F32; the table is ~35 KB total so
    // quantization saves nothing.
    if (name == "enc.embed.weight") {
        return Bucket::Norm;
    }
    // SenseVoice: trailing LayerNorms after the encoder stack and after
    // the tp_encoder stack. Don't carry the `norm_` prefix the generic
    // rule looks for, and the loader requires F32.
    if (ends_with(name, ".after_norm.weight") ||
        ends_with(name, ".tp_norm.weight"))
    {
        return Bucket::Norm;
    }
    // Moonshine Streaming: learned scalar inside the asinh frontend
    // (`asinh(exp(log_k) * x)`). [1]-element F32 — smaller than every
    // quant block size, and precision-sensitive (it controls the
    // magnitude of every encoder input feature).
    if (name == "enc.embedder.comp.log_k") {
        return Bucket::Norm;
    }
    // Voxtral Realtime: baked sinusoidal time-embedding inverse-frequency
    // table (dec.time_embed.inv_freq, [time_embed_dim/2]). Consumed F32 at
    // session setup to precompute the per-layer ada-norm FFN scales; the
    // loader's GET_F32 requires F32 and the table is too small to quantize.
    if (name == "dec.time_embed.inv_freq") {
        return Bucket::Norm;
    }
    // Conformer-block 1×1 pointwise convs — split out from Conv so
    // they can run at F16 on the im2col+matmul path. Matches
    // "enc.blocks.<N>.conv.pointwise1.weight" and ".pointwise2.weight"
    // but intentionally NOT any pre-encode convs (different names like
    // enc.pre_encode.conv.0.weight). granite_nar uses the underscore
    // form `conv_pointwise{1,2}` (single-token names without a `.conv.`
    // separator) — route those to ConvPw as well.
    if ((ends_with(name, ".conv.pointwise1.weight") ||
         ends_with(name, ".conv.pointwise2.weight")) &&
        contains(name, "enc.blocks."))
    {
        return Bucket::ConvPw;
    }
    if ((ends_with(name, ".conv_pointwise1.weight") ||
         ends_with(name, ".conv_pointwise2.weight")) &&
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
    // granite_nar depthwise conv kernel uses the underscore form
    // (`conv_depthwise.weight`) without a `.conv.` separator.
    if (ends_with(name, ".conv_depthwise.weight")) {
        return Bucket::Conv;
    }
    // SenseVoice FSMN: depthwise 1D conv inside the SAN-M attention
    // block (kernel=11, groups=d_model). Named "...attn.fsmn.weight",
    // doesn't carry the ".conv." substring, but loader treats it as a
    // conv (GET_CONV → F32/F16 only).
    if (ends_with(name, ".attn.fsmn.weight")) {
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

    // K-quants. Block size 256; linear_fallback is Q8_0 for every K
    // preset, not a scaled-down legacy quant, because fallback triggers
    // frequently on ASR models (Whisper-tiny d_model=384, Parakeet
    // predictor/joint ne0=640, Qwen3-ASR encoder ne0=896 — none divide
    // 256). Two reasons Q8_0 is the right floor:
    //   1. Size. F16 fallback makes K presets larger than Q8_0 on those
    //      families, which defeats the preset entirely.
    //   2. Quality. The tensors that hit fallback are the *same* ones
    //      that were already shape-awkward; using a scaled legacy quant
    //      (Q4_K → Q4_1) would penalize them twice. Q8_0 is ~lossless
    //      vs F16 and keeps fallback quality monotonically above main.
    // The tradeoff vs Q4_1/Q5_1 fallback is a few percent on file size,
    // paid on tensors that couldn't be K-quantized anyway.
    {"Q6_K",   GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q6_K*/  18 },
    {"Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,  GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q5_K_M*/ 17 },
    {"Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,  GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, /*MOSTLY_Q4_K_M*/ 15 },
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
            // to linear_main. Shape-misaligned embeddings route through
            // the same linear_fallback as everything else — for K
            // presets that's Q8_0, which is strictly higher quality than
            // the Q6_K bump we'd have applied, so there's no special
            // case to make here.
            ggml_type target = (preset.linear_embed != GGML_TYPE_COUNT)
                ? preset.linear_embed
                : preset.linear_main;
            if (const int64_t blk = ggml_blck_size(target); blk > 1 && (ne0 % blk) != 0) {
                target = preset.linear_fallback;
            }
            // If even the fallback's block size doesn't divide ne0
            // (e.g. moonshine-streaming's frame_len=80 hitting a Q8_0
            // fallback with block 32), drop to F16 — the universal
            // floor. F16 has block size 1 so it always fits.
            if (const int64_t blk = ggml_blck_size(target); blk > 1 && (ne0 % blk) != 0) {
                target = GGML_TYPE_F16;
            }
            return target;
        }
        case Bucket::Linear: {
            // Attention output projection bumped per _M recipe. Cohere
            // and Parakeet name it "attn.linear_out.weight"; Qwen3-ASR
            // and Whisper name it "attn.out.weight" (encoder); Qwen3-ASR
            // also uses "attn.o.weight" (decoder). All land in the same
            // bump. Whisper additionally has self_attn / cross_attn
            // out projections in the decoder under those exact names.
            const bool is_attn_out =
                ends_with(name, "attn.linear_out.weight") ||
                ends_with(name, "attn.out.weight") ||
                ends_with(name, "attn.o.weight");
            ggml_type target = is_attn_out
                ? preset.linear_attn_out
                : preset.linear_main;
            // Fall back when inner dim doesn't divide the chosen
            // quant's block size. Same rule for both linear_main and
            // linear_attn_out — without it, e.g. whisper-tiny's
            // d_model=384 attn.out.weight under Q6_K would land at
            // Q6_K (block size 256) and the GGUF would refuse to load.
            if (const int64_t blk = ggml_blck_size(target); blk > 1 && (ne0 % blk) != 0) {
                target = preset.linear_fallback;
            }
            // F16 floor: if linear_fallback's block size also doesn't
            // divide ne0 (e.g. moonshine-streaming's enc.embedder.linear
            // with ne0=frame_len=80 → Q8_0 fallback block 32), drop to
            // F16. F16 has block size 1 so it is the universal floor.
            if (const int64_t blk = ggml_blck_size(target); blk > 1 && (ne0 % blk) != 0) {
                target = GGML_TYPE_F16;
            }
            return target;
        }
    }
    return preset.norm; // unreachable
}

} // namespace transcribe::quantize
