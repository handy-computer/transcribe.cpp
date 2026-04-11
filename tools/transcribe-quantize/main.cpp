// transcribe-quantize - re-quantize a transcribe.cpp GGUF.
//
// Reads an input GGUF (typically the f32 or f16 conversion of a Parakeet
// model), walks every tensor, and writes a new GGUF where each tensor's
// dtype is chosen by a per-preset table. The dequantize → fp32 → requantize
// path uses ggml's reference quantizers (the same ones llama.cpp's
// llama-quantize uses), so the output is bit-compatible with anything
// that consumes standard GGUF blocks.
//
// Why a separate C++ tool instead of doing this in convert-parakeet.py:
//
//   gguf-py 0.18 ships pure-Python implementations of F32, F16, BF16,
//   Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 — but every K-quant (Q4_K, Q5_K, Q6_K,
//   Q8_K) and every IQ-quant raises NotImplementedError. Bridging
//   ggml's C quantizers from Python via ctypes works but is fragile.
//   Following whisper.cpp / llama.cpp's pattern, the conversion path
//   is Python (one-shot from MLX → fp32 / f16 / Q8_0) and the K-quant
//   path is a small C++ binary that links libggml directly.
//
// The tool is intentionally Parakeet-aware in exactly one place:
// classify_tensor() — which mirrors the same function in
// scripts/convert-parakeet.py. If you change the bucket assignment in
// one place, change the other or the loader's GET_LIN allowlist will
// reject the result.
//
// Output writes use gguf_write_to_file() with only_meta=false: the
// gguf_context's tensor list owns its own ggml_context, and tensor data
// is held inside that context, so the writer streams everything in one
// pass without any external file handling.

#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------
//
// Mirrors classify_tensor() in scripts/convert-parakeet.py AND
// scripts/convert-cohere.py. The two implementations MUST agree on the
// bucket for every canonical Parakeet / Cohere tensor name; the C++
// side is the gate at re-quant time and the Python side is the gate at
// first conversion.

enum class Bucket {
    Linear, // ggml_mul_mat operands. Eligible for Q8_0 / Q4_K / Q5_K / etc.
    Embed,  // tied token embedding + head bias. Like Linear but mixed
            // k-quant presets bump this to Q6_K, matching llama.cpp's
            // _M convention. Falls back to Linear when a preset has
            // no explicit embed override.
    Conv,   // conv kernels. F32 / F16 only.
    Norm,   // biases, LayerNorm/BN weight+bias, pos_bias, pos_enc,
            // frontend buffers. F32.
};

// Substring helpers — std::string::find returns npos if not found.
inline bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}
inline bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && std::memcmp(s.data() + s.size() - n, suffix, n) == 0;
}

Bucket classify_tensor(const std::string & name) {
    // Cohere: tied token embedding → Embed bucket so mixed k-quant
    // presets can bump it to Q6_K. head.bias stays in the Norm
    // bucket (the loader requires it as F32 since it's a 1D
    // per-logit offset added after the matmul).
    if (name == "dec.embed.token.weight") {
        return Bucket::Embed;
    }
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
    // norm scale tensors that use a "." separator instead of "_".
    if (ends_with(name, ".final_norm.weight") ||
        ends_with(name, ".embed.norm.weight")) {
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
    // Conv kernels: enc.pre_encode.conv.{0,2,3,5,6}.weight and
    // enc.blocks.{i}.conv.{pointwise1,depthwise,pointwise2}.weight.
    // ".conv." (with dots on both sides) distinguishes these from
    // norm_conv.weight which we already routed above.
    if (contains(name, ".conv.") && ends_with(name, ".weight")) {
        return Bucket::Conv;
    }
    return Bucket::Linear;
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------
//
// Each preset is a function from (bucket, tensor name, source ne[0])
// to a target ggml_type. The "source ne[0]" hook lets a preset fall
// back from a K-quant (block size 256) to a smaller blockwise quant
// (block size 32) when the inner dim doesn't divide 256. This is the
// failsafe for the Parakeet 0.6B predictor + joint dimension 640.
//
// We special-case attn.out_w in the K-quant presets: standard
// llama.cpp _M recipe bumps the attention output projection to a
// higher-precision quant (Q8_0 here) because it's the most
// quality-sensitive matmul in the encoder block.

struct Preset {
    const char * name;
    // Linear-bucket type returned for the "common" case (inner dim
    // divides the K-quant block size 256, and the tensor is not the
    // attention output projection).
    ggml_type linear_main;
    // Linear-bucket type returned when the inner dim doesn't divide
    // 256 (predictor + joint linears at ne0=640). We'd rather drop to
    // F16 here than risk a per-block scale on a misaligned tail.
    ggml_type linear_fallback;
    // Linear-bucket type returned for the attention output projection
    // (attn.linear_out.weight). Parakeet-specific bump; cohere's
    // equivalent uses a different name so this is a no-op there.
    ggml_type linear_attn_out;
    // Embed-bucket type for the tied token embedding + head bias
    // (cohere). Mixed k-quant presets bump this to Q6_K; other
    // presets use GGML_TYPE_COUNT as "no explicit override" and fall
    // back to linear_main.
    ggml_type linear_embed;
    // Conv-bucket type. F32 today across every preset; reserved here
    // so a future "lite" preset could ship F16 conv kernels once Metal
    // covers them.
    ggml_type conv;
    // Norm-bucket type. Always F32; the field exists for symmetry.
    ggml_type norm;
    // llama-style file_type tag for general.file_type. Mirrors the
    // GGUFWriter add_uint32 the Python converter writes for f32/f16/q8_0.
    uint32_t file_type;
};

// Names mirror the Python QUANT_PRESETS keys plus the K-quant ones.
//
// f16, q8_0 are listed for completeness — re-quantizing through this
// tool is not the canonical path for those (the converter does it
// directly from MLX). They exist as cheap test cases for the round-trip
// machinery.
// GGML_TYPE_COUNT in the linear_embed slot means "no explicit override";
// resolve_target_type() falls back to linear_main for Embed tensors in
// that case. This keeps f16 / q8_0 / q6_k presets uniform across
// Linear and Embed while letting _M presets bump embeddings to Q6_K.
const Preset PRESETS[] = {
    // f16: linear bucket → F16; conv + norm → F32. Fallback irrelevant
    // (F16 has no shape constraint).
    {"f16",    GGML_TYPE_F16,  GGML_TYPE_F16, GGML_TYPE_F16,  GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F32, /*MOSTLY_F16*/    1 },
    // q8_0: linear → Q8_0 (block 32, no fallback ever needed).
    {"q8_0",   GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F32, /*MOSTLY_Q8_0*/   7 },
    // q6_k: pure Q6_K across the linear bucket, embed inherits.
    {"q6_k",   GGML_TYPE_Q6_K, GGML_TYPE_F16, GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F32, /*MOSTLY_Q6_K*/  18 },
    // q5_k_m: K-quant for the encoder, F16 fallback for tensors with
    // ne0 not divisible by 256. attn.linear_out bumped to Q8_0 per
    // the llama.cpp _M recipe. Token embedding bumped to Q6_K.
    {"q5_k_m", GGML_TYPE_Q5_K, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,  GGML_TYPE_F32, GGML_TYPE_F32, /*MOSTLY_Q5_K_M*/ 17 },
    // q4_k_m: same shape as q5_k_m, smaller bit budget on the encoder
    // matmul bucket.
    {"q4_k_m", GGML_TYPE_Q4_K, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,  GGML_TYPE_F32, GGML_TYPE_F32, /*MOSTLY_Q4_K_M*/ 15 },
};

const Preset * find_preset(const char * name) {
    for (const auto & p : PRESETS) {
        if (std::strcmp(p.name, name) == 0) {
            return &p;
        }
    }
    return nullptr;
}

// Block size for a given quant type, from ggml's type traits. Used to
// gate the K-quant fallback decision: if ne0 % blck_size != 0, the
// quantizer would crash, so we bail to the preset's fallback type.
int64_t blck_size_of(ggml_type t) {
    return ggml_blck_size(t);
}

// Resolve the per-tensor target type for a single tensor. Buckets and
// special cases live here so the main loop is just a dispatch table.
ggml_type resolve_target_type(const Preset & preset,
                              const std::string & name,
                              int64_t ne0)
{
    const Bucket b = classify_tensor(name);
    switch (b) {
        case Bucket::Norm: return preset.norm;
        case Bucket::Conv: return preset.conv;
        case Bucket::Embed: {
            // If the preset has no explicit embed override, fall back
            // to the linear_main bucket (including its ne0 fallback).
            ggml_type target = (preset.linear_embed != GGML_TYPE_COUNT)
                ? preset.linear_embed
                : preset.linear_main;
            const int64_t blk = blck_size_of(target);
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
            // quant's block size. The fallback is preset-defined and
            // is always something with a smaller block size (or no
            // block constraint at all, like F16).
            const int64_t blk = blck_size_of(preset.linear_main);
            if (blk > 1 && (ne0 % blk) != 0) {
                return preset.linear_fallback;
            }
            return preset.linear_main;
        }
    }
    return preset.norm; // unreachable
}

// ---------------------------------------------------------------------------
// Per-tensor requant
// ---------------------------------------------------------------------------

// Walk type traits to dequantize an arbitrary input tensor into fp32.
// Mirrors decoder.cpp's read_tensor_to_f32 (including the F32 fast
// path that exists because the type_traits[F32] entry leaves to_float
// null upstream — F32 dequant is the identity).
bool dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        std::fprintf(stderr, "transcribe-quantize: tensor \"%s\" has nelem=%lld\n",
                     t->name, (long long)nelem);
        return false;
    }
    out.resize((size_t)nelem);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)nelem * sizeof(float));
        return true;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: type %s has no to_float\n",
                     ggml_type_name(t->type));
        return false;
    }
    tt->to_float(t->data, out.data(), nelem);
    return true;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct PerTypeStats {
    int64_t n_tensors = 0;
    int64_t bytes     = 0;
};

const char * type_label(ggml_type t) {
    return ggml_type_name(t);
}

void print_stats(const std::map<ggml_type, PerTypeStats> & stats,
                 const char * label)
{
    int64_t total_bytes = 0;
    int64_t total_n     = 0;
    std::printf("  %-12s%-10s%s\n", "type", "tensors", "MB");
    for (const auto & kv : stats) {
        std::printf("  %-12s%-10lld%.1f\n",
                    type_label(kv.first),
                    (long long)kv.second.n_tensors,
                    (double)kv.second.bytes / (1024.0 * 1024.0));
        total_bytes += kv.second.bytes;
        total_n     += kv.second.n_tensors;
    }
    std::printf("  %-12s%-10lld%.1f total (%s)\n", "",
                (long long)total_n,
                (double)total_bytes / (1024.0 * 1024.0),
                label);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s INPUT.gguf OUTPUT.gguf --quant PRESET\n"
        "\n"
        "  --quant PRESET   one of:", argv0);
    for (const auto & p : PRESETS) {
        std::fprintf(stderr, " %s", p.name);
    }
    std::fprintf(stderr, "\n");
}

} // namespace

int main(int argc, char ** argv) {
    const char * in_path  = nullptr;
    const char * out_path = nullptr;
    const char * quant    = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quant" || a == "-q") {
            if (i + 1 >= argc) { print_usage(argv[0]); return 2; }
            quant = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]); return 0;
        } else if (in_path == nullptr) {
            in_path = argv[i];
        } else if (out_path == nullptr) {
            out_path = argv[i];
        } else {
            std::fprintf(stderr, "transcribe-quantize: unexpected argument %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (in_path == nullptr || out_path == nullptr || quant == nullptr) {
        print_usage(argv[0]);
        return 2;
    }
    const Preset * preset = find_preset(quant);
    if (preset == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: unknown preset %s\n", quant);
        print_usage(argv[0]);
        return 2;
    }

    std::printf("transcribe-quantize: %s -> %s (preset %s)\n",
                in_path, out_path, preset->name);

    // ---- Load input gguf with tensor data into a ggml_context ----
    //
    // no_alloc=false + ctx=&ctx_in lets ggml allocate one big buffer
    // and load every tensor's bytes into it. After this returns, every
    // ggml_tensor in ctx_in has a valid `data` pointer pointing into
    // that buffer. The data lives until ggml_free(ctx_in).
    ggml_context * ctx_in = nullptr;
    gguf_init_params in_params{};
    in_params.no_alloc = false;
    in_params.ctx      = &ctx_in;
    gguf_context * gguf_in = gguf_init_from_file(in_path, in_params);
    if (gguf_in == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: failed to read %s\n", in_path);
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_in);
    std::printf("input: %lld tensors\n", (long long)n_tensors);

    // ---- Plan: per-tensor target type + total output buffer size ----
    //
    // We need a fresh ggml_context for the output tensors that owns
    // their (possibly newly-quantized) data. The size is the sum of
    // ggml_row_size() across the chosen output types, plus per-tensor
    // metadata overhead, plus alignment slack.
    //
    // Iterate via the gguf side rather than ggml's object list:
    // when ggml_init_from_file uses no_alloc=false, gguf.cpp allocates
    // a single i8 "GGUF tensor data binary blob" inside ctx_in to
    // hold the raw bytes (gguf.cpp:789), then makes every real tensor
    // a view into it. ggml_get_first_tensor would return that blob
    // first; using gguf_get_tensor_name + ggml_get_tensor walks only
    // the named gguf tensors and skips the blob.
    struct PlanEntry {
        ggml_tensor * src;
        ggml_type     dst_type;
        size_t        dst_nbytes; // ggml_row_size(dst_type, ne0) * nrows
    };
    std::vector<PlanEntry> plan;
    plan.reserve((size_t)n_tensors);
    size_t total_data_bytes = 0;
    std::map<ggml_type, PerTypeStats> in_stats, out_stats;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * tname = gguf_get_tensor_name(gguf_in, i);
        ggml_tensor * t = ggml_get_tensor(ctx_in, tname);
        if (t == nullptr) {
            std::fprintf(stderr,
                         "transcribe-quantize: tensor %s missing from ctx_in\n",
                         tname);
            ggml_free(ctx_in);
            gguf_free(gguf_in);
            return 1;
        }
        const std::string name(t->name);
        const ggml_type dst_type = resolve_target_type(*preset, name, t->ne[0]);

        // ggml_row_size handles both block-quant and dense layouts.
        // For multi-row tensors the total bytes is row_size * (nrows
        // * higher_dims). We compute "all rows beyond ne[0]" by
        // multiplying ne[1..3].
        int64_t nrows = 1;
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            nrows *= t->ne[d];
        }
        const size_t row_size = ggml_row_size(dst_type, t->ne[0]);
        const size_t nb       = row_size * (size_t)nrows;

        plan.push_back({t, dst_type, nb});
        total_data_bytes += nb;
        // Round up to ggml's alignment so each tensor starts on a
        // 32-byte boundary inside the output context buffer.
        total_data_bytes = (total_data_bytes + 31) & ~size_t{31};

        in_stats[t->type].n_tensors  += 1;
        in_stats[t->type].bytes      += (int64_t)ggml_nbytes(t);
        out_stats[dst_type].n_tensors += 1;
        out_stats[dst_type].bytes     += (int64_t)nb;
    }

    // Per-tensor metadata overhead (ggml_tensor struct + bookkeeping
    // inside the context). ggml_tensor_overhead() is the canonical
    // accessor; multiply by tensor count + slack.
    const size_t per_tensor_overhead = ggml_tensor_overhead();
    const size_t mem_size =
        total_data_bytes
        + (size_t)n_tensors * per_tensor_overhead
        + 16 * 1024 * 1024; // 16 MB safety pad for alignment + bookkeeping

    std::printf("plan: %.1f MB tensor data, %.1f MB context buffer\n",
                (double)total_data_bytes / (1024.0 * 1024.0),
                (double)mem_size          / (1024.0 * 1024.0));
    std::printf("\ninput tensor types:\n");
    print_stats(in_stats,  "input");
    std::printf("\noutput tensor types:\n");
    print_stats(out_stats, "output");
    std::printf("\n");

    // ---- Allocate the output ggml_context + gguf_context ----
    ggml_init_params out_init{};
    out_init.mem_size   = mem_size;
    out_init.mem_buffer = nullptr; // ggml will malloc internally
    out_init.no_alloc   = false;
    ggml_context * ctx_out = ggml_init(out_init);
    if (ctx_out == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: ggml_init(out) failed\n");
        gguf_free(gguf_in);
        return 1;
    }

    gguf_context * gguf_out = gguf_init_empty();
    if (gguf_out == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: gguf_init_empty failed\n");
        ggml_free(ctx_out);
        gguf_free(gguf_in);
        return 1;
    }

    // Copy every KV from the input. This pulls general.architecture,
    // tokenizer.*, stt.parakeet.*, stt.frontend.*, stt.capability.*,
    // and the existing general.file_type if present.
    gguf_set_kv(gguf_out, gguf_in);

    // Override general.file_type with the new preset's tag. The C++
    // loader doesn't read this today, but downstream tools like
    // gguf-dump display it.
    gguf_set_val_u32(gguf_out, "general.file_type", preset->file_type);

    // ---- Per-tensor: dequant → fp32 → requant → add to ctx_out ----
    std::vector<float> fp32_scratch;
    int64_t requantized = 0;
    int64_t copied      = 0;

    for (const PlanEntry & e : plan) {
        ggml_tensor * src = e.src;

        // Allocate the destination tensor in ctx_out with the new
        // dtype and the same shape as the source. ggml_new_tensor
        // handles arbitrary rank up to GGML_MAX_DIMS.
        const int n_dims = ggml_n_dims(src);
        ggml_tensor * dst = ggml_new_tensor(ctx_out, e.dst_type, n_dims, src->ne);
        if (dst == nullptr) {
            std::fprintf(stderr,
                         "transcribe-quantize: ggml_new_tensor failed for %s "
                         "(out of context memory?)\n", src->name);
            ggml_free(ctx_out);
            gguf_free(gguf_in);
            gguf_free(gguf_out);
            return 1;
        }
        ggml_set_name(dst, src->name);

        // Cheap path: same dtype as source, just memcpy the bytes.
        // Catches all the F32 norm/bias tensors when re-quantizing
        // an f16 input, and the F16 conv kernels when quantizing an
        // already-f16 input, etc.
        if (src->type == dst->type) {
            std::memcpy(dst->data, src->data, ggml_nbytes(src));
            ++copied;
        } else {
            // Real requant. Step 1: dequantize the source to fp32 in
            // a host scratch buffer.
            if (!dequantize_to_f32(src, fp32_scratch)) {
                ggml_free(ctx_out);
                gguf_free(gguf_in);
                gguf_free(gguf_out);
                return 1;
            }

            // Step 2: quantize fp32 → dst_type into the destination
            // tensor's data buffer. ggml_quantize_chunk takes (type,
            // src_fp32, dst_bytes, start_elem, nrows, n_per_row,
            // imatrix). For our K-quants imatrix is unused; we only
            // use types that don't require it.
            if (ggml_quantize_requires_imatrix(dst->type)) {
                std::fprintf(stderr,
                             "transcribe-quantize: type %s requires imatrix; "
                             "this tool does not support imatrix quants yet\n",
                             ggml_type_name(dst->type));
                ggml_free(ctx_out);
                gguf_free(gguf_in);
                gguf_free(gguf_out);
                return 1;
            }
            const int64_t n_per_row = src->ne[0];
            int64_t       nrows     = 1;
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                nrows *= src->ne[d];
            }
            ggml_quantize_chunk(dst->type,
                                fp32_scratch.data(),
                                dst->data,
                                /*start=*/0,
                                nrows,
                                n_per_row,
                                /*imatrix=*/nullptr);
            ++requantized;
        }

        gguf_add_tensor(gguf_out, dst);
    }

    std::printf("processed: %lld requantized, %lld copied\n",
                (long long)requantized, (long long)copied);

    // ---- Write the output gguf ----
    if (!gguf_write_to_file(gguf_out, out_path, /*only_meta=*/false)) {
        std::fprintf(stderr, "transcribe-quantize: gguf_write_to_file failed\n");
        ggml_free(ctx_out);
        gguf_free(gguf_in);
        gguf_free(gguf_out);
        return 1;
    }

    // Final cleanup. Free in reverse order: gguf_out and ctx_out are
    // independent; gguf_in owns ctx_in (allocated via in_params.ctx)
    // and freeing the gguf context releases the ggml context too.
    ggml_free(ctx_out);
    gguf_free(gguf_out);
    gguf_free(gguf_in);
    ggml_quantize_free();

    std::printf("done. wrote %s\n", out_path);
    return 0;
}
