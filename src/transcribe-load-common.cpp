// transcribe-load-common.cpp - shared model-load scaffolding.
//
// See transcribe-load-common.h for rationale. The functions here
// are verbatim hoists of the common backend-init and tensor-stream
// logic that used to live duplicated in every per-family model.cpp.
// The pre-hoist copies differed only in the "parakeet:"/"cohere:"
// log prefix and were otherwise character-identical.

#include "transcribe-load-common.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <vector>

namespace transcribe::load_common {

transcribe_status init_backends(bool                           force_cpu,
                                const char *                   error_tag,
                                std::vector<ggml_backend_t> &  out)
{
    // GPU/iGPU backend (Metal, Vulkan, CUDA — whichever ggml found).
    //
    // Skipped entirely when the caller requested CPU-only via
    // transcribe_model_params::use_gpu == false. ACCEL backends
    // (BLAS on CPU, etc.) are NOT skipped because they run on host
    // memory and are orthogonal to the GPU/CPU split.
    if (!force_cpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto dev_type = ggml_backend_dev_type(dev);
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
                if (be != nullptr) {
                    std::fprintf(stderr, "%s: using GPU backend: %s\n",
                                 error_tag, ggml_backend_dev_name(dev));
                    out.push_back(be);
                    break; // first available GPU wins
                }
            }
        }
    }

    // ACCEL backends (BLAS on CPU, etc.). These accelerate specific
    // ops (matmul) while CPU handles the rest of the graph, so they
    // layer cleanly on top of both CPU-only and GPU configurations.
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (be != nullptr) {
                std::fprintf(stderr, "%s: using ACCEL backend: %s\n",
                             error_tag, ggml_backend_dev_name(dev));
                out.push_back(be);
            }
        }
    }

    // CPU backend (always present, always last).
    ggml_backend_t cpu_be =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_be == nullptr) {
        std::fprintf(stderr,
                     "%s: failed to initialize CPU backend\n", error_tag);
        return TRANSCRIBE_ERR_GGUF;
    }
    out.push_back(cpu_be);
    return TRANSCRIBE_OK;
}

transcribe_status stream_tensor_data(const std::string &   path,
                                     const gguf_context *  gguf_data,
                                     ggml_context *        ctx_meta,
                                     const char *          error_tag)
{
    // std::ifstream rather than going through gguf's loader a third
    // time: ifstream::seekg takes a streamoff (signed 64-bit on every
    // platform we target), so multi-GB tensor offsets work without
    // #ifdef'ing fseeko vs _fseeki64.
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::fprintf(stderr,
                     "%s: failed to reopen %s for tensor data\n",
                     error_tag, path.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    const size_t data_offset = gguf_get_data_offset(gguf_data);
    std::vector<uint8_t> staging;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta);
         t != nullptr;
         t = ggml_get_next_tensor(ctx_meta, t))
    {
        const int64_t idx = gguf_find_tensor(gguf_data, t->name);
        if (idx < 0) {
            std::fprintf(stderr,
                         "%s: tensor \"%s\" not in gguf data\n",
                         error_tag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t toffset = gguf_get_tensor_offset(gguf_data, idx);
        const size_t nbytes  = ggml_nbytes(t);

        const std::streamoff abs_offset =
            static_cast<std::streamoff>(data_offset) +
            static_cast<std::streamoff>(toffset);
        fin.seekg(abs_offset);
        if (!fin) {
            std::fprintf(stderr,
                         "%s: seek failed for tensor \"%s\"\n",
                         error_tag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }

        if (staging.size() < nbytes) {
            staging.resize(nbytes);
        }
        fin.read(reinterpret_cast<char *>(staging.data()),
                 static_cast<std::streamsize>(nbytes));
        if (!fin) {
            std::fprintf(stderr,
                         "%s: short read for tensor \"%s\" (%zu bytes)\n",
                         error_tag, t->name, nbytes);
            return TRANSCRIBE_ERR_GGUF;
        }

        // ggml_backend_tensor_set is the right call regardless of
        // backend: on host buffers (CPU + Metal unified memory on
        // Apple Silicon) it's a memcpy; on discrete GPUs it does
        // the upload.
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
    }

    return TRANSCRIBE_OK;
}

transcribe_status promote_conv_pw_f16_to_f32_on_cpu(
    const std::vector<ggml_backend_t> & backends,
    const std::vector<ConvPwF32Slot> &  slots,
    const char *                        error_tag,
    ggml_context **                     out_ctx,
    ggml_backend_buffer_t *             out_buffer)
{
    if (backends.empty()) return TRANSCRIBE_OK;

    // Only the primary backend (backends.front()) matters: that's
    // where the encoder graph actually runs. ACCEL backends only
    // accelerate specific ops on host memory; they don't change where
    // the matmul dispatches.
    ggml_backend_dev_t primary_dev =
        ggml_backend_get_device(backends.front());
    if (primary_dev == nullptr) return TRANSCRIBE_OK;
    if (ggml_backend_dev_type(primary_dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return TRANSCRIBE_OK;
    }

    if (slots.empty()) return TRANSCRIBE_OK;

    // New ctx sized for exactly the replacement tensors plus a small
    // slack. no_alloc=true — ggml_backend_alloc_ctx_tensors will
    // allocate the storage buffer separately below.
    const size_t ctx_size = slots.size() * ggml_tensor_overhead() + 256;
    ggml_init_params params = {ctx_size, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) return TRANSCRIBE_ERR_BACKEND;

    // Allocate F32 replacements in the new ctx, matching each source's
    // full n-d shape. Names are copied so debug dumps still find them.
    std::vector<ggml_tensor *> replacements;
    replacements.reserve(slots.size());
    for (const auto & s : slots) {
        ggml_tensor * r = ggml_new_tensor(
            ctx, GGML_TYPE_F32, ggml_n_dims(s.src), s.src->ne);
        if (r == nullptr) {
            ggml_free(ctx);
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_set_name(r, s.src->name);
        replacements.push_back(r);
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx, backends.front());
    if (buffer == nullptr) {
        std::fprintf(stderr,
            "%s: conv_pw f32 promotion buffer alloc failed\n", error_tag);
        ggml_free(ctx);
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Dequantize each F16 tensor into its F32 replacement.
    const auto * f16_traits = ggml_get_type_traits(GGML_TYPE_F16);
    if (f16_traits == nullptr || f16_traits->to_float == nullptr) {
        std::fprintf(stderr,
            "%s: no f16 to_float trait — skipping conv pw promotion\n",
            error_tag);
        // Partial success: the ctx + buffer are already allocated but
        // unused. Free them so the caller's outparams stay nullptr,
        // matching the "do nothing" contract.
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return TRANSCRIBE_OK;
    }

    std::vector<uint8_t> f16_staging;
    std::vector<float>   f32_staging;
    for (size_t i = 0; i < slots.size(); ++i) {
        ggml_tensor * src = slots[i].src;
        ggml_tensor * dst = replacements[i];
        const int64_t n_elem = ggml_nelements(src);
        const size_t f16_bytes = ggml_nbytes(src);
        const size_t f32_bytes = static_cast<size_t>(n_elem) * sizeof(float);

        if (f16_staging.size() < f16_bytes) f16_staging.resize(f16_bytes);
        if (f32_staging.size() < static_cast<size_t>(n_elem)) {
            f32_staging.resize(n_elem);
        }

        ggml_backend_tensor_get(src, f16_staging.data(), 0, f16_bytes);
        f16_traits->to_float(f16_staging.data(), f32_staging.data(), n_elem);
        ggml_backend_tensor_set(dst, f32_staging.data(), 0, f32_bytes);

        *slots[i].dst_slot = dst;
    }

    *out_ctx    = ctx;
    *out_buffer = buffer;

    std::fprintf(stderr,
        "%s: promoted %zu conv pointwise weights from F16 → F32 "
        "for CPU backend\n", error_tag, slots.size());
    return TRANSCRIBE_OK;
}

} // namespace transcribe::load_common
