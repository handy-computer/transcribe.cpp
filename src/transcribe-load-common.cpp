// transcribe-load-common.cpp - shared model-load scaffolding.
//
// See transcribe-load-common.h for rationale. The functions here are the
// common backend-init and tensor-stream logic shared by per-family load().

#include "transcribe-load-common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
// Under GGML_BACKEND_DL the Metal symbols live in a loadable module and are
// not link-visible here, so the capability query below is compiled out.
#if defined(TRANSCRIBE_HAS_METAL) && !defined(TRANSCRIBE_GGML_BACKEND_DL)
#    include "ggml-metal.h"
#endif
#include "transcribe-backend.h"
#include "transcribe-env.h"
#include "transcribe-log.h"
#include "transcribe-path.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <vector>

namespace transcribe::load_common {

// GPUs below MTLGPUFamilyApple7 (Intel iGPUs, AMD dGPUs on Intel Macs) have
// no simdgroup matrix multiply; ggml's fallback matmul kernels silently
// produce garbage transcripts there (Handy issue #1608). No generic ggml
// capability exposes this — MUL_MAT's supports_op only needs simdgroup
// *reduction*, which these GPUs do report — so ask Metal directly.
bool metal_backend_lacks_simdgroup_mm(ggml_backend_t be, ggml_backend_dev_t dev) {
    // Test hook: force the "missing" verdict by device-name substring
    // ("*" matches any) so the gate can be exercised on healthy hardware.
    if (const char * match = std::getenv("TRANSCRIBE_TEST_METAL_NO_SIMDGROUP_MM");
        match != nullptr && match[0] != '\0') {
        const char * name = ggml_backend_dev_name(dev);
        if (std::strcmp(match, "*") == 0 || (name != nullptr && std::strstr(name, match) != nullptr)) {
            return true;
        }
    }
#if defined(TRANSCRIBE_HAS_METAL) && !defined(TRANSCRIBE_GGML_BACKEND_DL)
    // ggml_backend_metal_supports_family asserts the backend is Metal; guard
    // so a classify/is_metal disagreement can never abort the process.
    if (ggml_backend_is_metal(be)) {
        return !ggml_backend_metal_supports_family(be, /*MTLGPUFamilyApple7=*/7);
    }
#endif
    (void) be;
    return false;
}

namespace {

// Device init can throw from GPU drivers. Treat that like nullptr so AUTO can
// keep probing and explicit backend requests fail cleanly.
//
// Test hook: TRANSCRIBE_TEST_DEV_INIT_THROW is inert when unset or empty,
// "*" matches every device, otherwise it matches by device-name substring.
ggml_backend_t dev_init_checked(ggml_backend_dev_t dev, const char * error_tag) {
    const char * name = "?";
    try {
        if (const char * n = ggml_backend_dev_name(dev); n != nullptr) {
            name = n;
        }
        if (const char * match = std::getenv("TRANSCRIBE_TEST_DEV_INIT_THROW"); match != nullptr && match[0] != '\0') {
            if (std::strcmp(match, "*") == 0 || std::strstr(name, match) != nullptr) {
                throw std::runtime_error("TRANSCRIBE_TEST_DEV_INIT_THROW fault injection");
            }
        }
        return ggml_backend_dev_init(dev, nullptr);
    } catch (const std::exception & e) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: device \"%s\" init threw: %s - skipping it", error_tag, name, e.what());
        return nullptr;
    } catch (...) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: device \"%s\" init threw an unknown exception - skipping it", error_tag,
                name);
        return nullptr;
    }
}

// Try to discover and initialize the first device whose classified
// BackendKind matches `wanted`, visiting candidates in gpu_probe_order
// (every discrete GPU before any integrated GPU). On success returns
// the initialized backend and writes the classified kind of the device
// that actually succeeded to out_kind (which may differ from `wanted`
// when `wanted == BackendKind::OtherGpu`). Returns nullptr if no
// matching device initializes.
//
// When `wanted == BackendKind::OtherGpu`, this acts as an "any GPU of
// any vendor" probe — used by the AUTO path. Critically, out_kind is
// derived from the device that actually initialized, not from a
// separate post-hoc registry walk, so a failing first-GPU followed by
// a succeeding second-GPU yields the correct kind.
ggml_backend_t try_init_kind(BackendKind wanted, const char * error_tag, BackendKind & out_kind) {
    const size_t n = ggml_backend_dev_count();

    std::vector<enum ggml_backend_dev_type> dev_types;
    dev_types.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        dev_types.push_back(ggml_backend_dev_type(ggml_backend_dev_get(i)));
    }

    for (const size_t i : gpu_probe_order(dev_types)) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (wanted != BackendKind::OtherGpu && classify_device(dev) != wanted) {
            continue;
        }

        ggml_backend_t be = dev_init_checked(dev, error_tag);
        if (be == nullptr) {
            continue;
        }

        const BackendKind kind = classify_device(dev);

        // A Metal device without simdgroup matmul yields garbage transcripts
        // (see metal_backend_lacks_simdgroup_mm): skip it under AUTO, honor
        // an explicit Metal request with a warning.
        if (kind == BackendKind::Metal && metal_backend_lacks_simdgroup_mm(be, dev)) {
            const char * dname = ggml_backend_dev_name(dev);
            if (wanted == BackendKind::OtherGpu) {
                log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                        "%s: skipping Metal device \"%s\": no simdgroup matrix multiply "
                        "(pre-Apple7 GPU)",
                        error_tag, dname != nullptr ? dname : "?");
                safe_backend_free(be);
                continue;
            }
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                    "%s: Metal device \"%s\" has no simdgroup matrix multiply "
                    "(pre-Apple7 GPU); transcription may be incorrect and very slow",
                    error_tag, dname != nullptr ? dname : "?");
        }

        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using %s backend: %s", error_tag, kind_name(kind),
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
void append_accel_backends(std::vector<ggml_backend_t> & out, const char * error_tag) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t be = dev_init_checked(dev, error_tag);
        if (be == nullptr) {
            continue;
        }

        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using accel backend: %s", error_tag, ggml_backend_dev_name(dev));
        out.push_back(be);
    }
}

// Initialize the CPU backend. Always runs — it is the universal
// fallback and the strict-CPU primary.
ggml_backend_t init_cpu_backend(const char * error_tag) {
    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_be == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to initialize CPU backend", error_tag);
    }
    return cpu_be;
}

bool valid_backend_request(int raw) {
    switch (raw) {
        case TRANSCRIBE_BACKEND_AUTO:
        case TRANSCRIBE_BACKEND_CPU:
        case TRANSCRIBE_BACKEND_METAL:
        case TRANSCRIBE_BACKEND_VULKAN:
        case TRANSCRIBE_BACKEND_CPU_ACCEL:
        case TRANSCRIBE_BACKEND_CUDA:
        case TRANSCRIBE_BACKEND_ROCM:
            return true;
    }
    return false;
}

// Resolve a BackendPlan for an exact process-local device handle. The handle
// must be one currently registered with ggml. GPU primaries retain ACCEL + CPU
// scheduler fallbacks; an exact CPU primary is strict CPU unless CPU_ACCEL was
// requested, in which case host-memory accelerators are layered ahead of it.
transcribe_status init_backends_explicit_device(transcribe_backend_request requested,
                                                transcribe_device_t        device,
                                                const char *               error_tag,
                                                BackendPlan &              out) {
    ggml_backend_dev_t dev        = reinterpret_cast<ggml_backend_dev_t>(device);
    bool               registered = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_get(i) == dev) {
            registered = true;
            break;
        }
    }
    if (!registered) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: device handle is not registered with this runtime", error_tag);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const auto dev_type = ggml_backend_dev_type(dev);
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: accelerator device %s cannot be a model primary", error_tag,
                ggml_backend_dev_name(dev));
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const BackendKind got    = classify_device(dev);
    BackendKind       wanted = BackendKind::Unknown;
    switch (requested) {
        case TRANSCRIBE_BACKEND_CPU:
        case TRANSCRIBE_BACKEND_CPU_ACCEL:
            wanted = BackendKind::Cpu;
            break;
        case TRANSCRIBE_BACKEND_METAL:
            wanted = BackendKind::Metal;
            break;
        case TRANSCRIBE_BACKEND_VULKAN:
            wanted = BackendKind::Vulkan;
            break;
        case TRANSCRIBE_BACKEND_CUDA:
            wanted = BackendKind::Cuda;
            break;
        case TRANSCRIBE_BACKEND_ROCM:
            wanted = BackendKind::Rocm;
            break;
        case TRANSCRIBE_BACKEND_AUTO:
            break;
        default:
            return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (wanted != BackendKind::Unknown && got != wanted) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: selected %s device does not match requested %s backend", error_tag,
                kind_name(got), kind_name(wanted));
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    ggml_backend_t primary = dev_init_checked(dev, error_tag);
    if (primary == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to initialize selected device %s", error_tag,
                ggml_backend_dev_name(dev));
        return TRANSCRIBE_ERR_BACKEND;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using selected %s device: %s", error_tag, kind_name(got),
            ggml_backend_dev_name(dev));

    if (got == BackendKind::Metal && metal_backend_lacks_simdgroup_mm(primary, dev)) {
        const char * dname = ggml_backend_dev_name(dev);
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "%s: Metal device \"%s\" has no simdgroup matrix multiply "
                "(pre-Apple7 GPU); transcription may be incorrect and very slow",
                error_tag, dname != nullptr ? dname : "?");
    }

    out.primary      = primary;
    out.primary_kind = got;
    if (got == BackendKind::Cpu) {
        if (requested == TRANSCRIBE_BACKEND_CPU_ACCEL) {
            append_accel_backends(out.scheduler_list, error_tag);
        }
        out.scheduler_list.push_back(primary);
        return TRANSCRIBE_OK;
    }

    out.scheduler_list.push_back(primary);
    append_accel_backends(out.scheduler_list, error_tag);
    ggml_backend_t cpu_be = init_cpu_backend(error_tag);
    if (cpu_be == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    out.scheduler_list.push_back(cpu_be);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status init_backends(transcribe_backend_request requested,
                                transcribe_device_t        device,
                                const char *               error_tag,
                                BackendPlan &              out) {
    // Read the request as raw bytes before any enum-typed load: a C caller
    // can pass any int here, and loading an out-of-range value through the
    // enum lvalue is UB in C++ (UBSan traps). The raw value is validated by
    // the switch; only valid values are stored back as the enum.
    int requested_raw = TRANSCRIBE_BACKEND_AUTO;
    std::memcpy(&requested_raw, &requested, sizeof(requested_raw));

    out           = BackendPlan{};
    out.requested = static_cast<transcribe_backend_request>(
        valid_backend_request(requested_raw) ? requested_raw : TRANSCRIBE_BACKEND_AUTO);

    // A non-NULL handle pins one exact registered device. Validate the raw
    // backend value before passing its enum representation onward.
    if (device != nullptr) {
        if (!valid_backend_request(requested_raw)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid transcribe_backend_request value %d", error_tag,
                    requested_raw);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        return init_backends_explicit_device(out.requested, device, error_tag, out);
    }

    // Explicit switch over the enum so an unknown / garbage value
    // from a C caller never silently collapses into AUTO. Unknown
    // values are a programming error on the caller's side, not a
    // fallback we want to tolerate.
    switch (requested_raw) {
        case TRANSCRIBE_BACKEND_CPU:
        case TRANSCRIBE_BACKEND_CPU_ACCEL:
            {
                // CPU primary. CPU_ACCEL additionally layers the host-memory
                // accelerators (BLAS/AMX/…, GGML_BACKEND_DEVICE_TYPE_ACCEL) onto the
                // scheduler. Both set primary_kind == Cpu so CPU-keyed policy (e.g.
                // F16→F32 conv pointwise promotion) triggers identically.
                //
                // ggml requires the CPU backend to sit last in the scheduler list,
                // so accel backends (when included) go in first.
                const bool     with_accel = (requested == TRANSCRIBE_BACKEND_CPU_ACCEL);
                ggml_backend_t cpu_be     = init_cpu_backend(error_tag);
                if (cpu_be == nullptr) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: using cpu backend (%s)", error_tag,
                        with_accel ? "with accel" : "strict");
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
        case TRANSCRIBE_BACKEND_CUDA:
        case TRANSCRIBE_BACKEND_ROCM:
            {
                // Specific GPU backend request: must find a matching device
                // or fail. ACCEL is still layered on because it's host-memory
                // and orthogonal to the GPU/CPU split.
                BackendKind wanted = BackendKind::Unknown;
                switch (requested) {
                    case TRANSCRIBE_BACKEND_METAL:
                        wanted = BackendKind::Metal;
                        break;
                    case TRANSCRIBE_BACKEND_VULKAN:
                        wanted = BackendKind::Vulkan;
                        break;
                    case TRANSCRIBE_BACKEND_CUDA:
                        wanted = BackendKind::Cuda;
                        break;
                    case TRANSCRIBE_BACKEND_ROCM:
                        wanted = BackendKind::Rocm;
                        break;
                    default:
                        break;
                }
                BackendKind    got_kind = BackendKind::Unknown;
                ggml_backend_t gpu_be   = try_init_kind(wanted, error_tag, got_kind);
                if (gpu_be == nullptr) {
                    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: %s backend requested but not available", error_tag,
                            kind_name(wanted));
                    return TRANSCRIBE_ERR_BACKEND;
                }
                out.primary      = gpu_be;
                out.primary_kind = got_kind;
                out.scheduler_list.push_back(gpu_be);

                append_accel_backends(out.scheduler_list, error_tag);

                ggml_backend_t cpu_be = init_cpu_backend(error_tag);
                if (cpu_be == nullptr) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                out.scheduler_list.push_back(cpu_be);
                return TRANSCRIBE_OK;
            }

        case TRANSCRIBE_BACKEND_AUTO:
            {
                // AUTO: take the first GPU device that successfully
                // initializes, regardless of vendor, in gpu_probe_order
                // (discrete before integrated). If every GPU fails init
                // or none is compiled in, fall through to CPU + ACCEL.
                //
                // try_init_kind yields the classified kind of the device
                // that actually succeeded, so a failing-then-succeeding probe
                // can't misclassify primary_kind.
                BackendKind    got_kind = BackendKind::Unknown;
                ggml_backend_t gpu_be   = try_init_kind(BackendKind::OtherGpu, error_tag, got_kind);
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
    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid transcribe_backend_request value %d", error_tag, requested_raw);
    return TRANSCRIBE_ERR_INVALID_ARG;
}

// ---------------------------------------------------------------------------
// Weight storage: memory-mapped where the device can use host memory,
// copied into a backend buffer otherwise.
// ---------------------------------------------------------------------------

void WeightStorage::reset() noexcept {
    // Buffers before the mapping. A host-pointer buffer only wraps the pages
    // (its free never touches them), but it must not outlive them.
    safe_buffer_free(mapped_buffer_);
    mapped_buffer_ = nullptr;
    safe_buffer_free(copy_buffer_);
    copy_buffer_ = nullptr;
    file_.reset();
}

bool WeightStorage::is_mapped(const ggml_tensor * t) const noexcept {
    return t != nullptr && mapped_buffer_ != nullptr && t->buffer == mapped_buffer_;
}

bool mmap_allowed(const transcribe_model_load_params * params) {
    if (params != nullptr && params->no_mmap) {
        return false;
    }
    return !env::flag("TRANSCRIBE_NO_MMAP");
}

namespace {

constexpr double k_mib = 1024.0 * 1024.0;

// The primary device can keep weights in a wrapped host allocation. ggml says
// so through buffer_from_host_ptr (CPU, Metal) and mmap_support (false for
// integrated CUDA/Vulkan GPUs). Pre-Apple7 Metal GPUs are also excluded: they
// include the discrete AMD GPUs of Intel Macs, whose weights belong in VRAM
// rather than in shared system memory.
bool device_can_map(const BackendPlan & plan, ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.buffer_from_host_ptr || !props.caps.mmap_support) {
        return false;
    }
    if (plan.primary_kind == BackendKind::Metal && metal_backend_lacks_simdgroup_mm(plan.primary, dev)) {
        return false;
    }
    return true;
}

struct MapCounts {
    size_t n_mapped     = 0;
    size_t mapped_bytes = 0;
    size_t n_layout     = 0;  // in-memory type/size differs from the file (converted at load)
    size_t n_misaligned = 0;  // file offset not aligned for the device
};

// Map every tensor of ctx_meta whose file bytes are usable as-is. When nothing
// can be mapped, `file` and `out_buffer` are left empty, `why` says why, and
// the result is still TRANSCRIBE_OK: the caller copies everything instead.
transcribe_status map_weights(const std::string &     path,
                              const gguf_context *    gguf_data,
                              ggml_context *          ctx_meta,
                              ggml_backend_dev_t      dev,
                              const char *            error_tag,
                              MappedFile &            file,
                              ggml_backend_buffer_t & out_buffer,
                              MapCounts &             counts,
                              std::string &           why) {
    if (!file.open(path, &why)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: cannot memory-map %s (%s); copying weights instead", error_tag,
                path.c_str(), why.c_str());
        why = "memory mapping failed";
        return TRANSCRIBE_OK;
    }

    const size_t file_size   = file.size();
    const size_t data_offset = gguf_get_data_offset(gguf_data);
    char *       base        = static_cast<char *>(file.addr());

    struct Candidate {
        ggml_tensor * t;
        char *        addr;
    };

    std::vector<Candidate> candidates;
    size_t                 max_tensor_size = 0;
    if (data_offset <= file_size) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
            if (t->data != nullptr || t->buffer != nullptr || t->view_src != nullptr) {
                continue;  // already has storage
            }
            const int64_t idx = gguf_find_tensor(gguf_data, ggml_get_name(t));
            if (idx < 0) {
                continue;  // stream_tensor_data reports it
            }
            const size_t nbytes = ggml_nbytes(t);
            if (gguf_get_tensor_type(gguf_data, idx) != t->type || gguf_get_tensor_size(gguf_data, idx) != nbytes ||
                !ggml_is_contiguous(t)) {
                ++counts.n_layout;
                continue;
            }
            const size_t toffset = gguf_get_tensor_offset(gguf_data, idx);
            if (toffset > file_size - data_offset || nbytes > file_size - data_offset - toffset) {
                continue;  // runs past EOF: the stream reports the truncation
            }
            candidates.push_back({ t, base + data_offset + toffset });
            max_tensor_size = std::max(max_tensor_size, nbytes);
        }
    }
    if (candidates.empty()) {
        file.reset();
        why = "no tensor is stored as-is in the file";
        return TRANSCRIBE_OK;
    }

    // Wrap the whole page-aligned mapping, not just the data section: ggml's
    // CPU host-pointer buffer asserts 32-byte alignment of its base, which a
    // GGUF with general.alignment < 32 would not give data_offset.
    ggml_backend_buffer_t buffer = ggml_backend_dev_buffer_from_host_ptr(dev, base, file_size, max_tensor_size);
    if (buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: %s cannot wrap the memory-mapped file; copying weights instead",
                error_tag, ggml_backend_dev_name(dev));
        file.reset();
        why = "device could not wrap the mapping";
        return TRANSCRIBE_OK;
    }

    const size_t alignment = ggml_backend_buffer_get_alignment(buffer);
    for (const Candidate & c : candidates) {
        const size_t nbytes = ggml_nbytes(c.t);
        if (alignment > 1 && reinterpret_cast<uintptr_t>(c.addr) % alignment != 0) {
            ++counts.n_misaligned;
            continue;
        }
        if (ggml_backend_buffer_get_alloc_size(buffer, c.t) != nbytes) {
            ++counts.n_layout;
            continue;
        }
        if (ggml_backend_tensor_alloc(buffer, c.t, c.addr) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: %s rejected memory-mapped tensor \"%s\"", error_tag,
                    ggml_backend_dev_name(dev), ggml_get_name(c.t));
            out_buffer = buffer;  // owned by the storage from here; tensors may point into it
            return TRANSCRIBE_ERR_BACKEND;
        }
        ++counts.n_mapped;
        counts.mapped_bytes += nbytes;
    }
    if (counts.n_mapped == 0) {
        safe_buffer_free(buffer);
        file.reset();
        why = counts.n_misaligned > 0 ? "tensors misaligned in the file" : "no tensor is stored as-is in the file";
        return TRANSCRIBE_OK;
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    out_buffer = buffer;
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status allocate_copied_weights(ggml_context *      ctx_meta,
                                          const BackendPlan & plan,
                                          const char *        error_tag,
                                          WeightStorage &     out) {
    bool any_unallocated = false;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (t->data == nullptr && t->view_src == nullptr) {
            any_unallocated = true;
            break;
        }
    }
    if (!any_unallocated) {
        return TRANSCRIBE_OK;
    }
    if (out.copy_buffer_ != nullptr || plan.primary == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: weight storage already allocated or no primary backend", error_tag);
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Allocates only the tensors that have no storage yet.
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_meta, plan.primary);
    if (buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: ggml_backend_alloc_ctx_tensors failed", error_tag);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    out.copy_buffer_ = buffer;
    return TRANSCRIBE_OK;
}

transcribe_status allocate_weights(const std::string &                  path,
                                   const gguf_context *                 gguf_data,
                                   ggml_context *                       ctx_meta,
                                   const BackendPlan &                  plan,
                                   const transcribe_model_load_params * params,
                                   const char *                         error_tag,
                                   WeightStorage &                      out) {
    if (plan.primary == nullptr || gguf_data == nullptr || ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(plan.primary);

    std::string why;
    MapCounts   counts;
    if (!mmap_allowed(params)) {
        why = "memory mapping disabled";
    } else if (!MappedFile::supported()) {
        why = "memory mapping unsupported on this platform";
    } else if (!device_can_map(plan, dev)) {
        why = "device keeps weights in its own memory";
    } else if (const transcribe_status st =
                   map_weights(path, gguf_data, ctx_meta, dev, error_tag, out.file_, out.mapped_buffer_, counts, why);
               st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = allocate_copied_weights(ctx_meta, plan, error_tag, out); st != TRANSCRIBE_OK) {
        return st;
    }

    size_t n_copied = 0, copied_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (out.copy_buffer_ != nullptr && t->buffer == out.copy_buffer_) {
            ++n_copied;
            copied_bytes += ggml_nbytes(t);
        }
    }
    if (counts.n_mapped == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: weights copied: %.1f MiB in %s (%s)", error_tag, copied_bytes / k_mib,
                out.copy_buffer_ != nullptr ? ggml_backend_buffer_name(out.copy_buffer_) : "-", why.c_str());
    } else if (n_copied == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: weights memory-mapped: %.1f MiB in %s", error_tag,
                counts.mapped_bytes / k_mib, ggml_backend_buffer_name(out.mapped_buffer_));
    } else {
        char reasons[96];
        if (counts.n_layout > 0 && counts.n_misaligned > 0) {
            std::snprintf(reasons, sizeof(reasons), "%zu converted at load, %zu misaligned in the file",
                          counts.n_layout, counts.n_misaligned);
        } else if (counts.n_misaligned > 0) {
            std::snprintf(reasons, sizeof(reasons), "misaligned in the file");
        } else {
            std::snprintf(reasons, sizeof(reasons), "converted at load");
        }
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
                "%s: weights memory-mapped: %.1f MiB in %s; %zu tensors (%.1f MiB) copied (%s)", error_tag,
                counts.mapped_bytes / k_mib, ggml_backend_buffer_name(out.mapped_buffer_), n_copied,
                copied_bytes / k_mib, reasons);
    }
    return TRANSCRIBE_OK;
}

transcribe_status load_weights(const std::string &                  path,
                               const gguf_context *                 gguf_data,
                               ggml_context *                       ctx_meta,
                               const BackendPlan &                  plan,
                               const transcribe_model_load_params * params,
                               const char *                         error_tag,
                               WeightStorage &                      out) {
    if (const transcribe_status st = allocate_weights(path, gguf_data, ctx_meta, plan, params, error_tag, out);
        st != TRANSCRIBE_OK) {
        return st;
    }
    return stream_tensor_data(path, gguf_data, ctx_meta, error_tag, &out);
}

transcribe_status stream_tensor_data(const std::string &   path,
                                     const gguf_context *  gguf_data,
                                     ggml_context *        ctx_meta,
                                     const char *          error_tag,
                                     const WeightStorage * skip) {
    // std::ifstream rather than going through gguf's loader a third
    // time: ifstream::seekg takes a streamoff (signed 64-bit on every
    // platform we target), so multi-GB tensor offsets work without
    // #ifdef'ing fseeko vs _fseeki64.
    std::ifstream fin(path_from_utf8(path), std::ios::binary);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to reopen %s for tensor data", error_tag, path.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    // Uploads go through a bounded staging buffer: ggml_backend_tensor_set
    // stages each call whole, and for device memory the host cannot map,
    // Vulkan grows a per-device staging buffer, kept for the life of the
    // process, to the largest single write (243 MiB for Qwen3-ASR-1.7B's token
    // embedding), on top of ours.
    constexpr size_t k_stage_chunk = size_t(16) << 20;

    const size_t         data_offset = gguf_get_data_offset(gguf_data);
    std::vector<uint8_t> staging;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (skip != nullptr && skip->is_mapped(t)) {
            continue;  // already backed by the file bytes
        }
        const int64_t idx = gguf_find_tensor(gguf_data, t->name);
        if (idx < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: tensor \"%s\" not in gguf data", error_tag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t toffset = gguf_get_tensor_offset(gguf_data, idx);
        const size_t nbytes  = ggml_nbytes(t);

        const std::streamoff abs_offset =
            static_cast<std::streamoff>(data_offset) + static_cast<std::streamoff>(toffset);
        fin.seekg(abs_offset);
        if (!fin) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: seek failed for tensor \"%s\"", error_tag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }

        // Host buffers (the CPU) take the file bytes directly: no staging
        // copy. Everything else (Metal, Vulkan, CUDA, ...) is uploaded in
        // chunks through ggml_backend_tensor_set, which needs a buffer type
        // that accepts partial writes: every default one does (ggml's CPU
        // repack buffers would not, and assert).
        if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) {
            fin.read(static_cast<char *>(t->data), static_cast<std::streamsize>(nbytes));
            if (!fin) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: short read for tensor \"%s\" (%zu bytes)", error_tag, t->name,
                        nbytes);
                return TRANSCRIBE_ERR_GGUF;
            }
            continue;
        }

        const size_t chunk = std::min(nbytes, k_stage_chunk);
        if (staging.size() < chunk) {
            staging.resize(chunk);
        }
        for (size_t done = 0; done < nbytes;) {
            const size_t n = std::min(nbytes - done, staging.size());
            fin.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(n));
            if (!fin) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: short read for tensor \"%s\" (%zu bytes)", error_tag, t->name,
                        nbytes);
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_tensor_set(t, staging.data(), done, n);
            done += n;
        }
    }

    return TRANSCRIBE_OK;
}

transcribe_status promote_conv_pw_f16_to_f32_on_cpu(const BackendPlan &                plan,
                                                    const std::vector<ConvPwF32Slot> & slots,
                                                    const char *                       error_tag,
                                                    ggml_context **                    out_ctx,
                                                    ggml_backend_buffer_t *            out_buffer) {
    // Key off the classified primary kind, not off ACCEL/CPU ordering in
    // the backend list, so a strict-CPU request reliably triggers promotion
    // even when an ACCEL backend sorts ahead of CPU.
    if (plan.primary_kind != BackendKind::Cpu) {
        return TRANSCRIBE_OK;
    }
    if (plan.primary == nullptr) {
        return TRANSCRIBE_OK;
    }

    if (slots.empty()) {
        return TRANSCRIBE_OK;
    }

    // New ctx sized for exactly the replacement tensors plus a small
    // slack. no_alloc=true — ggml_backend_alloc_ctx_tensors will
    // allocate the storage buffer separately below.
    const size_t     ctx_size = slots.size() * ggml_tensor_overhead() + 256;
    ggml_init_params params   = { ctx_size, nullptr, true };
    ggml_context *   ctx      = ggml_init(params);
    if (ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }

    // Allocate F32 replacements in the new ctx, matching each source's
    // full n-d shape. Names are copied so debug dumps still find them.
    std::vector<ggml_tensor *> replacements;
    replacements.reserve(slots.size());
    for (const auto & s : slots) {
        ggml_tensor * r = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(s.src), s.src->ne);
        if (r == nullptr) {
            ggml_free(ctx);
            return TRANSCRIBE_ERR_OOM;
        }
        ggml_set_name(r, s.src->name);
        replacements.push_back(r);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, plan.primary);
    if (buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: conv_pw f32 promotion buffer alloc failed", error_tag);
        ggml_free(ctx);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Dequantize each F16 tensor into its F32 replacement.
    const auto * f16_traits = ggml_get_type_traits(GGML_TYPE_F16);
    if (f16_traits == nullptr || f16_traits->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: no f16 to_float trait — skipping conv pw promotion", error_tag);
        // Partial success: the ctx + buffer are already allocated but
        // unused. Free them so the caller's outparams stay nullptr,
        // matching the "do nothing" contract.
        safe_buffer_free(buffer);
        ggml_free(ctx);
        return TRANSCRIBE_OK;
    }

    std::vector<uint8_t> f16_staging;
    std::vector<float>   f32_staging;
    for (size_t i = 0; i < slots.size(); ++i) {
        ggml_tensor * src       = slots[i].src;
        ggml_tensor * dst       = replacements[i];
        const int64_t n_elem    = ggml_nelements(src);
        const size_t  f16_bytes = ggml_nbytes(src);
        const size_t  f32_bytes = static_cast<size_t>(n_elem) * sizeof(float);

        if (f16_staging.size() < f16_bytes) {
            f16_staging.resize(f16_bytes);
        }
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
            "for CPU backend",
            error_tag, slots.size());
    return TRANSCRIBE_OK;
}

ReadF32Result read_f32_tensor_checked(gguf_context *       gguf_ctx,
                                      const std::string &  gguf_path,
                                      const char *         tensor_name,
                                      size_t               expected_elems,
                                      const char *         error_tag,
                                      std::vector<float> & out) {
    // Clear on entry so stale data cannot leak on any return path.
    out.clear();

    const int64_t idx = gguf_find_tensor(gguf_ctx, tensor_name);
    if (idx < 0) {
        return ReadF32Result::Absent;
    }

    // Validate type is F32.
    const enum ggml_type ttype = gguf_get_tensor_type(gguf_ctx, idx);
    if (ttype != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: tensor \"%s\" has type %d, expected F32 (%d)", error_tag, tensor_name,
                static_cast<int>(ttype), static_cast<int>(GGML_TYPE_F32));
        return ReadF32Result::BadType;
    }

    const size_t nbytes = static_cast<size_t>(gguf_get_tensor_size(gguf_ctx, idx));

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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: tensor \"%s\" has %zu elements, expected %zu", error_tag, tensor_name,
                n_elems, expected_elems);
        return ReadF32Result::BadSize;
    }

    // Read from the GGUF data section. Use the same overflow-safe
    // offset pattern as stream_tensor_data: cast each addend to
    // std::streamoff individually rather than adding two size_t values
    // that could wrap on 32-bit builds (not a real platform today, but
    // consistent).
    const size_t data_off = gguf_get_data_offset(gguf_ctx);
    const size_t t_off    = gguf_get_tensor_offset(gguf_ctx, idx);

    std::ifstream fin(path_from_utf8(gguf_path), std::ios::binary);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to open %s for tensor \"%s\"", error_tag, gguf_path.c_str(),
                tensor_name);
        return ReadF32Result::ReadErr;
    }

    const std::streamoff abs_offset = static_cast<std::streamoff>(data_off) + static_cast<std::streamoff>(t_off);
    fin.seekg(abs_offset);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: seek failed for tensor \"%s\"", error_tag, tensor_name);
        return ReadF32Result::ReadErr;
    }

    out.resize(n_elems);
    fin.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes));

    if (!fin || static_cast<size_t>(fin.gcount()) != nbytes) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "%s: short read for tensor \"%s\" "
                "(got %zu of %zu bytes)",
                error_tag, tensor_name, static_cast<size_t>(fin.gcount()), nbytes);
        out.clear();
        return ReadF32Result::ReadErr;
    }

    return ReadF32Result::Ok;
}

}  // namespace transcribe::load_common
