// transcribe-load-common.cpp - shared model-load scaffolding.
//
// See transcribe-load-common.h for rationale. The functions here
// are the common backend-init and tensor-stream logic that used to
// live duplicated in every per-family model.cpp. The pre-hoist copies
// differed only in the "parakeet:"/"cohere:" log prefix and were
// otherwise character-identical.

#include "transcribe-load-common.h"

#include "transcribe-backend.h"
#include "transcribe-log.h"

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

namespace {

// Try to discover and initialize the first device whose classified
// BackendKind matches `wanted`. On success returns the initialized
// backend and writes the classified kind of the device that actually
// succeeded to out_kind (which may differ from `wanted` when
// `wanted == BackendKind::OtherGpu`). Returns nullptr if no matching
// device initializes.
//
// When `wanted == BackendKind::OtherGpu`, this acts as a "first GPU
// or IGPU of any vendor" probe — used by the AUTO path. Critically,
// out_kind is derived from the device that actually initialized, not
// from a separate post-hoc registry walk, so a failing first-GPU
// followed by a succeeding second-GPU yields the correct kind.
ggml_backend_t try_init_kind(BackendKind   wanted,
                             const char *  error_tag,
                             BackendKind & out_kind)
{
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (wanted == BackendKind::OtherGpu) {
            // "Any GPU" probe: accept the first GPU/IGPU device,
            // regardless of vendor.
            const auto dev_type = ggml_backend_dev_type(dev);
            if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
        } else if (classify_device(dev) != wanted) {
            continue;
        }

        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) continue;

        const BackendKind kind = classify_device(dev);
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using %s backend: %s",
                     error_tag,
                     kind_name(kind),
                     ggml_backend_dev_name(dev));
        out_kind = kind;
        return be;
    }
    return nullptr;
}

// Append every ACCEL device as a scheduler backend. ACCEL backends
// (BLAS, AMX, …) accelerate specific ops on host memory, so they
// layer cleanly on top of both CPU and GPU primaries. They are
// excluded only on strict-CPU requests, where the whole point is to
// avoid any backend dispatch ambiguity.
void append_accel_backends(std::vector<ggml_backend_t> & out,
                           const char *                  error_tag)
{
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) continue;

        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using accel backend: %s",
                     error_tag, ggml_backend_dev_name(dev));
        out.push_back(be);
    }
}

// Initialize the CPU backend. Always runs — it is the universal
// fallback and the strict-CPU primary.
ggml_backend_t init_cpu_backend(const char * error_tag) {
    ggml_backend_t cpu_be =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_be == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: failed to initialize CPU backend", error_tag);
    }
    return cpu_be;
}

} // namespace

transcribe_status init_backends(transcribe_backend_request requested,
                                const char *               error_tag,
                                BackendPlan &              out)
{
    out = BackendPlan{};
    out.requested = requested;

    // Explicit switch over the enum so an unknown / garbage value
    // from a C caller never silently collapses into AUTO. Unknown
    // values are a programming error on the caller's side, not a
    // fallback we want to tolerate.
    switch (requested) {
    case TRANSCRIBE_BACKEND_CPU:
    case TRANSCRIBE_BACKEND_CPU_ACCEL: {
        // CPU primary. The CPU_ACCEL variant additionally layers the
        // host-memory accelerators (BLAS/AMX/…) registered by ggml as
        // GGML_BACKEND_DEVICE_TYPE_ACCEL onto the scheduler. Both
        // variants set primary_kind == Cpu so CPU-keyed policy
        // decisions (e.g. F16→F32 conv pointwise promotion) trigger
        // identically. Strict CPU is the right choice for numerical
        // reference runs and cross-platform determinism; CPU_ACCEL is
        // for production CPU throughput on machines where the build
        // included an accel backend (e.g. GGML_BLAS=ON).
        //
        // ggml requires the CPU backend to sit last in the scheduler
        // list, so accel backends (when included) go in first.
        const bool with_accel = (requested == TRANSCRIBE_BACKEND_CPU_ACCEL);
        ggml_backend_t cpu_be = init_cpu_backend(error_tag);
        if (cpu_be == nullptr) return TRANSCRIBE_ERR_BACKEND;
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using cpu backend (%s)",
                     error_tag, with_accel ? "with accel" : "strict");
        out.primary      = cpu_be;
        out.primary_kind = BackendKind::Cpu;
        if (with_accel) {
            append_accel_backends(out.scheduler_list, error_tag);
        }
        out.scheduler_list.push_back(cpu_be);
        return TRANSCRIBE_OK;
    }

    case TRANSCRIBE_BACKEND_METAL:
    case TRANSCRIBE_BACKEND_VULKAN:
    case TRANSCRIBE_BACKEND_CUDA: {
        // Specific GPU backend request: must find a matching device
        // or fail. ACCEL is still layered on because it's host-memory
        // and orthogonal to the GPU/CPU split.
        BackendKind wanted = BackendKind::Unknown;
        switch (requested) {
            case TRANSCRIBE_BACKEND_METAL:  wanted = BackendKind::Metal;  break;
            case TRANSCRIBE_BACKEND_VULKAN: wanted = BackendKind::Vulkan; break;
            case TRANSCRIBE_BACKEND_CUDA:   wanted = BackendKind::Cuda;   break;
            default: break;
        }
        BackendKind got_kind = BackendKind::Unknown;
        ggml_backend_t gpu_be = try_init_kind(wanted, error_tag, got_kind);
        if (gpu_be == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "%s: %s backend requested but not available",
                         error_tag, kind_name(wanted));
            return TRANSCRIBE_ERR_BACKEND;
        }
        out.primary      = gpu_be;
        out.primary_kind = got_kind;
        out.scheduler_list.push_back(gpu_be);

        append_accel_backends(out.scheduler_list, error_tag);

        ggml_backend_t cpu_be = init_cpu_backend(error_tag);
        if (cpu_be == nullptr) return TRANSCRIBE_ERR_BACKEND;
        out.scheduler_list.push_back(cpu_be);
        return TRANSCRIBE_OK;
    }

    case TRANSCRIBE_BACKEND_AUTO: {
        // AUTO: take the first GPU/IGPU device that successfully
        // initializes, regardless of vendor. ggml registers devices
        // in build-time priority order (Metal on Apple, Vulkan on
        // Linux, etc.), which matches the documented preference.
        // If every GPU fails init or none is compiled in, fall
        // through to CPU + ACCEL.
        //
        // try_init_kind yields the classified kind of the device
        // that actually succeeded, so a failing-then-succeeding probe
        // can't misclassify primary_kind.
        BackendKind got_kind = BackendKind::Unknown;
        ggml_backend_t gpu_be =
            try_init_kind(BackendKind::OtherGpu, error_tag, got_kind);
        if (gpu_be != nullptr) {
            out.primary      = gpu_be;
            out.primary_kind = got_kind;
            out.scheduler_list.push_back(gpu_be);
        }

        append_accel_backends(out.scheduler_list, error_tag);

        ggml_backend_t cpu_be = init_cpu_backend(error_tag);
        if (cpu_be == nullptr) {
            // If we already have at least a GPU or ACCEL backend,
            // losing CPU is catastrophic — the scheduler needs CPU
            // as a fallback for every op it can't dispatch
            // elsewhere. Fail hard.
            return TRANSCRIBE_ERR_BACKEND;
        }

        // If the GPU probe failed and AUTO picked nothing so far, CPU
        // becomes the primary.
        if (out.primary == nullptr) {
            out.primary      = cpu_be;
            out.primary_kind = BackendKind::Cpu;
        }
        out.scheduler_list.push_back(cpu_be);
        return TRANSCRIBE_OK;
    }
    }

    // Unknown enumerator: reject loudly so callers catch ABI drift
    // during development. Do not let "everything else" silently map
    // to AUTO — that hides bugs.
    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                 "%s: invalid transcribe_backend_request value %d",
                 error_tag, static_cast<int>(requested));
    return TRANSCRIBE_ERR_INVALID_ARG;
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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: failed to reopen %s for tensor data",
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "%s: tensor \"%s\" not in gguf data",
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "%s: seek failed for tensor \"%s\"",
                         error_tag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }

        if (staging.size() < nbytes) {
            staging.resize(nbytes);
        }
        fin.read(reinterpret_cast<char *>(staging.data()),
                 static_cast<std::streamsize>(nbytes));
        if (!fin) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "%s: short read for tensor \"%s\" (%zu bytes)",
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
    const BackendPlan &                plan,
    const std::vector<ConvPwF32Slot> & slots,
    const char *                       error_tag,
    ggml_context **                    out_ctx,
    ggml_backend_buffer_t *            out_buffer)
{
    // Key off the classified primary kind, not off ACCEL/CPU
    // ordering in the backend list. This is the fix for the latent
    // bug where strict-CPU requests could be silently shadowed by
    // an ACCEL backend sitting ahead of CPU.
    if (plan.primary_kind != BackendKind::Cpu) return TRANSCRIBE_OK;
    if (plan.primary == nullptr)               return TRANSCRIBE_OK;

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
        ggml_backend_alloc_ctx_tensors(ctx, plan.primary);
    if (buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
            "%s: conv_pw f32 promotion buffer alloc failed", error_tag);
        ggml_free(ctx);
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Dequantize each F16 tensor into its F32 replacement.
    const auto * f16_traits = ggml_get_type_traits(GGML_TYPE_F16);
    if (f16_traits == nullptr || f16_traits->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
            "%s: no f16 to_float trait — skipping conv pw promotion",
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

    log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
        "%s: promoted %zu conv pointwise weights from F16 → F32 "
        "for CPU backend", error_tag, slots.size());
    return TRANSCRIBE_OK;
}

ReadF32Result read_f32_tensor_checked(
    gguf_context *        gguf_ctx,
    const std::string &   gguf_path,
    const char *          tensor_name,
    size_t                expected_elems,
    const char *          error_tag,
    std::vector<float> &  out)
{
    // Clear on entry so stale data cannot leak on any return path.
    out.clear();

    const int64_t idx = gguf_find_tensor(gguf_ctx, tensor_name);
    if (idx < 0) {
        return ReadF32Result::Absent;
    }

    // Validate type is F32.
    const enum ggml_type ttype = gguf_get_tensor_type(gguf_ctx, idx);
    if (ttype != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: tensor \"%s\" has type %d, expected F32 (%d)",
                     error_tag, tensor_name,
                     static_cast<int>(ttype),
                     static_cast<int>(GGML_TYPE_F32));
        return ReadF32Result::BadType;
    }

    const size_t nbytes = static_cast<size_t>(
        gguf_get_tensor_size(gguf_ctx, idx));

    // Validate alignment: byte count must be a multiple of sizeof(float).
    if (nbytes == 0 || (nbytes % sizeof(float)) != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: tensor \"%s\" has %zu bytes "
                     "(not a multiple of %zu)",
                     error_tag, tensor_name, nbytes, sizeof(float));
        return ReadF32Result::BadSize;
    }

    const size_t n_elems = nbytes / sizeof(float);

    // Validate expected element count when the caller knows the shape.
    if (expected_elems > 0 && n_elems != expected_elems) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: tensor \"%s\" has %zu elements, expected %zu",
                     error_tag, tensor_name, n_elems, expected_elems);
        return ReadF32Result::BadSize;
    }

    // Read from the GGUF data section. Use the same overflow-safe
    // offset pattern as stream_tensor_data: cast each addend to
    // std::streamoff individually rather than adding two size_t values
    // that could wrap on 32-bit builds (not a real platform today, but
    // consistent).
    const size_t data_off = gguf_get_data_offset(gguf_ctx);
    const size_t t_off    = gguf_get_tensor_offset(gguf_ctx, idx);

    std::ifstream fin(gguf_path, std::ios::binary);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: failed to open %s for tensor \"%s\"",
                     error_tag, gguf_path.c_str(), tensor_name);
        return ReadF32Result::ReadErr;
    }

    const std::streamoff abs_offset =
        static_cast<std::streamoff>(data_off) +
        static_cast<std::streamoff>(t_off);
    fin.seekg(abs_offset);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: seek failed for tensor \"%s\"",
                     error_tag, tensor_name);
        return ReadF32Result::ReadErr;
    }

    out.resize(n_elems);
    fin.read(reinterpret_cast<char *>(out.data()),
             static_cast<std::streamsize>(nbytes));

    if (!fin || static_cast<size_t>(fin.gcount()) != nbytes) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "%s: short read for tensor \"%s\" "
                     "(got %zu of %zu bytes)",
                     error_tag, tensor_name,
                     static_cast<size_t>(fin.gcount()), nbytes);
        out.clear();
        return ReadF32Result::ReadErr;
    }

    return ReadF32Result::Ok;
}

} // namespace transcribe::load_common
