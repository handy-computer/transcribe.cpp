// transcribe.cpp - library-level public C API entry points + model dispatch.
//
// What lives here:
//   - Status strings, version, and ABI struct-size metadata.
//   - The log-sink publication / emission helpers.
//   - Backend module loading and device discovery.
//   - The single dispatch site mapping an opened GGUF to its per-family
//     Arch handler (transcribe_model_load_file), model free, and the
//     model-level queries (arch / variant / metadata / backend / device,
//     extension-kind and feature probes).
//   - The init functions for the params / output structs shared by every
//     role (model load params, device info, timings, speaker segments).
//
// What does NOT live here: the ASR session API (sessions, run, batch,
// streaming, results; transcribe-asr.cpp), GGUF reading (transcribe-loader),
// and per-family load / context / run code (src/arch/<family>/...).

#include "transcribe.h"

#include "arch/whisper/bin_load.h"
#include "ggml-backend.h"
#include "ggml.h"  // ggml_log_set: route ggml diagnostics into our sink
#include "transcribe-abi.h"
#include "transcribe-api-guard.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-model.h"
#include "transcribe-path.h"
#include "transcribe/diarize.h"
#include "transcribe/langid.h"

#if defined(TRANSCRIBE_GGML_BACKEND_DL) && defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(TRANSCRIBE_GGML_BACKEND_DL)
#    include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <mutex>
#include <set>
#include <string>

using transcribe::api_guard_status;
using transcribe::api_guard_value;
using transcribe::api_guard_void;
using transcribe::enum_field_raw;

// Status

// Parameter is `int` (not `transcribe_status`) so a caller passing an
// out-of-range value does not form a bogus enum object. See the comment
// above the declaration in include/transcribe.h.
extern "C" const char * transcribe_status_string(int status) {
    switch (status) {
        case TRANSCRIBE_OK:
            return "ok";
        case TRANSCRIBE_ERR_INVALID_ARG:
            return "invalid argument";
        case TRANSCRIBE_ERR_NOT_IMPLEMENTED:
            return "not implemented";
        case TRANSCRIBE_ERR_FILE_NOT_FOUND:
            return "file not found";
        case TRANSCRIBE_ERR_GGUF:
            return "gguf load error";
        case TRANSCRIBE_ERR_UNSUPPORTED_ARCH:
            return "unsupported architecture";
        case TRANSCRIBE_ERR_UNSUPPORTED_VARIANT:
            return "unsupported variant";
        case TRANSCRIBE_ERR_OOM:
            return "out of memory";
        case TRANSCRIBE_ERR_BACKEND:
            return "backend error";
        case TRANSCRIBE_ERR_SAMPLE_RATE:
            return "sample rate mismatch";
        case TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE:
            return "unsupported language";
        case TRANSCRIBE_ERR_UNSUPPORTED_TASK:
            return "unsupported task";
        case TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS:
            return "unsupported timestamp granularity";
        case TRANSCRIBE_ERR_ABORTED:
            return "aborted by callback";
        case TRANSCRIBE_ERR_BAD_STRUCT_SIZE:
            return "caller-owned struct missing or has bad struct_size";
        case TRANSCRIBE_ERR_UNSUPPORTED_PNC:
            return "model does not support runtime PNC control (reserved; not currently returned)";
        case TRANSCRIBE_ERR_UNSUPPORTED_ITN:
            return "model does not support runtime ITN control (reserved; not currently returned)";
        case TRANSCRIBE_ERR_INPUT_TOO_LONG:
            return "input audio too long for model context";
        case TRANSCRIBE_ERR_OUTPUT_TRUNCATED:
            return "output truncated: decode hit the context/generation cap before end-of-stream";
        case TRANSCRIBE_ERR_OUTPUT_REPETITION:
            return "output repetition: decode stopped when the output began repeating itself";
        case TRANSCRIBE_ERR_UNSUPPORTED_ROLE:
            return "model does not serve the requested role";
        case TRANSCRIBE_ERR_INPUT_TOO_SHORT:
            return "input audio too short";
        default:
            return "unknown status";
    }
}

// Version

// TRANSCRIBE_COMMIT is stamped by the build (git rev-parse --short HEAD at
// configure time); fall back to "unknown" so the accessor never returns
// NULL. TRANSCRIBE_VERSION comes from <transcribe.h>.
#ifndef TRANSCRIBE_COMMIT
#    define TRANSCRIBE_COMMIT "unknown"
#endif

extern "C" const char * transcribe_version(void) {
    return TRANSCRIBE_VERSION;
}

extern "C" const char * transcribe_version_commit(void) {
    return TRANSCRIBE_COMMIT;
}

// ABI metadata: sizeof/alignof for the public structs, so a binding can
// verify its layout against this build. Keep these switches in sync with the
// transcribe_abi_struct enum.

extern "C" size_t transcribe_abi_struct_size(transcribe_abi_struct which) {
    switch (enum_field_raw(&which)) {
        case TRANSCRIBE_ABI_MODEL_LOAD_PARAMS:
            return sizeof(struct transcribe_model_load_params);
        case TRANSCRIBE_ABI_SESSION_PARAMS:
            return sizeof(struct transcribe_session_params);
        case TRANSCRIBE_ABI_RUN_PARAMS:
            return sizeof(struct transcribe_run_params);
        case TRANSCRIBE_ABI_STREAM_PARAMS:
            return sizeof(struct transcribe_stream_params);
        case TRANSCRIBE_ABI_CAPABILITIES:
            return sizeof(struct transcribe_capabilities);
        case TRANSCRIBE_ABI_TIMINGS:
            return sizeof(struct transcribe_timings);
        case TRANSCRIBE_ABI_SEGMENT:
            return sizeof(struct transcribe_segment);
        case TRANSCRIBE_ABI_WORD:
            return sizeof(struct transcribe_word);
        case TRANSCRIBE_ABI_TOKEN:
            return sizeof(struct transcribe_token);
        case TRANSCRIBE_ABI_STREAM_UPDATE:
            return sizeof(struct transcribe_stream_update);
        case TRANSCRIBE_ABI_STREAM_TEXT:
            return sizeof(struct transcribe_stream_text);
        case TRANSCRIBE_ABI_SESSION_LIMITS:
            return sizeof(struct transcribe_session_limits);
        case TRANSCRIBE_ABI_EXT:
            return sizeof(struct transcribe_ext);
        case TRANSCRIBE_ABI_DEVICE_INFO:
            return sizeof(struct transcribe_device_info);
        case TRANSCRIBE_ABI_SPEAKER_SEGMENT:
            return sizeof(struct transcribe_speaker_segment);
        case TRANSCRIBE_ABI_BACKEND_INIT_PARAMS:
            return sizeof(struct transcribe_backend_init_params);
        case TRANSCRIBE_ABI_DIARIZE_INFO:
            return sizeof(struct transcribe_diarize_info);
        case TRANSCRIBE_ABI_DIARIZE_SESSION_PARAMS:
            return sizeof(struct transcribe_diarize_session_params);
        case TRANSCRIBE_ABI_DIARIZE_PARAMS:
            return sizeof(struct transcribe_diarize_params);
        case TRANSCRIBE_ABI_LANGID_INFO:
            return sizeof(struct transcribe_langid_info);
        case TRANSCRIBE_ABI_LANGID_SESSION_PARAMS:
            return sizeof(struct transcribe_langid_session_params);
        case TRANSCRIBE_ABI_LANGID_PARAMS:
            return sizeof(struct transcribe_langid_params);
        case TRANSCRIBE_ABI_LANGID_RESULT:
            return sizeof(struct transcribe_langid_result);
        case TRANSCRIBE_ABI_LANGID_CANDIDATE:
            return sizeof(struct transcribe_langid_candidate);
    }
    return 0;  // unknown id: "cannot verify", never a real size
}

extern "C" size_t transcribe_abi_struct_align(transcribe_abi_struct which) {
    switch (enum_field_raw(&which)) {
        case TRANSCRIBE_ABI_MODEL_LOAD_PARAMS:
            return alignof(struct transcribe_model_load_params);
        case TRANSCRIBE_ABI_SESSION_PARAMS:
            return alignof(struct transcribe_session_params);
        case TRANSCRIBE_ABI_RUN_PARAMS:
            return alignof(struct transcribe_run_params);
        case TRANSCRIBE_ABI_STREAM_PARAMS:
            return alignof(struct transcribe_stream_params);
        case TRANSCRIBE_ABI_CAPABILITIES:
            return alignof(struct transcribe_capabilities);
        case TRANSCRIBE_ABI_TIMINGS:
            return alignof(struct transcribe_timings);
        case TRANSCRIBE_ABI_SEGMENT:
            return alignof(struct transcribe_segment);
        case TRANSCRIBE_ABI_WORD:
            return alignof(struct transcribe_word);
        case TRANSCRIBE_ABI_TOKEN:
            return alignof(struct transcribe_token);
        case TRANSCRIBE_ABI_STREAM_UPDATE:
            return alignof(struct transcribe_stream_update);
        case TRANSCRIBE_ABI_STREAM_TEXT:
            return alignof(struct transcribe_stream_text);
        case TRANSCRIBE_ABI_SESSION_LIMITS:
            return alignof(struct transcribe_session_limits);
        case TRANSCRIBE_ABI_EXT:
            return alignof(struct transcribe_ext);
        case TRANSCRIBE_ABI_DEVICE_INFO:
            return alignof(struct transcribe_device_info);
        case TRANSCRIBE_ABI_SPEAKER_SEGMENT:
            return alignof(struct transcribe_speaker_segment);
        case TRANSCRIBE_ABI_BACKEND_INIT_PARAMS:
            return alignof(struct transcribe_backend_init_params);
        case TRANSCRIBE_ABI_DIARIZE_INFO:
            return alignof(struct transcribe_diarize_info);
        case TRANSCRIBE_ABI_DIARIZE_SESSION_PARAMS:
            return alignof(struct transcribe_diarize_session_params);
        case TRANSCRIBE_ABI_DIARIZE_PARAMS:
            return alignof(struct transcribe_diarize_params);
        case TRANSCRIBE_ABI_LANGID_INFO:
            return alignof(struct transcribe_langid_info);
        case TRANSCRIBE_ABI_LANGID_SESSION_PARAMS:
            return alignof(struct transcribe_langid_session_params);
        case TRANSCRIBE_ABI_LANGID_PARAMS:
            return alignof(struct transcribe_langid_params);
        case TRANSCRIBE_ABI_LANGID_RESULT:
            return alignof(struct transcribe_langid_result);
        case TRANSCRIBE_ABI_LANGID_CANDIDATE:
            return alignof(struct transcribe_langid_candidate);
    }
    return 0;
}

// Logging
//
// Two-atomic log sink. Publication stores userdata (relaxed) then cb
// (release); emission acquire-loads cb and relaxed-loads userdata. The
// acquire/release pairing on cb is the happens-before edge the supported
// "install once at startup, before threads or models exist" model needs.
// Reconfiguration is unsupported (the (cb, userdata) pair can tear and an
// in-flight emitter may still hold the old sink); the only guarantee then is
// absence of data races, not correct semantics.
//
// Three sink states, classified from one acquire load of g_log_cb:
//   nullptr        never configured -> stderr default (emit_or_stderr)
//   sentinel       disabled via transcribe_log_set(NULL) -> drop everything
//   anything else  caller's callback -> route everything
// The sentinel distinguishes "asked for silence" from "never called us"
// without a second atomic whose pairing could tear.

namespace {

// Stored in g_log_cb in place of NULL when the host explicitly disables
// logging. Never invoked; only compared against.
void transcribe_log_cb_disabled(transcribe_log_level, const char *, void *) {}

std::atomic<transcribe_log_callback> g_log_cb{ nullptr };
std::atomic<void *>                  g_log_userdata{ nullptr };

// Map a ggml log level onto the transcribe enum. The numeric values do
// NOT coincide (vendored ggml: DEBUG=1 INFO=2 WARN=3 ERROR=4;
// transcribe.h: INFO=1 WARN=2 ERROR=3 DEBUG=4), so the pass-through
// layer owns an explicit mapping — exactly the responsibility the
// transcribe_log_level contract assigns to it. Unknown future levels
// degrade to INFO rather than being dropped.
transcribe_log_level transcribe_log_level_from_ggml(int level) {
    switch (level) {
        case GGML_LOG_LEVEL_NONE:
            return TRANSCRIBE_LOG_LEVEL_NONE;
        case GGML_LOG_LEVEL_DEBUG:
            return TRANSCRIBE_LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_INFO:
            return TRANSCRIBE_LOG_LEVEL_INFO;
        case GGML_LOG_LEVEL_WARN:
            return TRANSCRIBE_LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_ERROR:
            return TRANSCRIBE_LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_CONT:
            return TRANSCRIBE_LOG_LEVEL_CONT;
        default:
            return TRANSCRIBE_LOG_LEVEL_INFO;
    }
}

void transcribe_ggml_log_bridge(enum ggml_log_level level, const char * text, void *);

}  // namespace

extern "C" void transcribe_log_set(transcribe_log_callback cb, void * userdata) {
    g_log_userdata.store(userdata, std::memory_order_relaxed);
    g_log_cb.store(cb != nullptr ? cb : &transcribe_log_cb_disabled, std::memory_order_release);
    // Route ggml's own diagnostics — above all the per-module dlopen
    // failures from dynamic backend loading, the signal a host needs to
    // see why an accelerator degraded to CPU — through the same sink.
    // Installed on configuration (not at library init) so a host that
    // never calls transcribe_log_set keeps ggml's stderr default
    // untouched. ggml_log_set itself is a plain global store; under the
    // supported install-once-at-startup model that is race-free.
    ggml_log_set(&transcribe_ggml_log_bridge, nullptr);
}

// Log callbacks have no error channel and may run from catch/noexcept paths.
// Treat a throwing callback as a host bug: drop the message, report to stderr.
static void transcribe_log_invoke(transcribe_log_callback cb,
                                  transcribe_log_level    level,
                                  const char *            msg,
                                  void *                  userdata) noexcept {
    try {
        cb(level, msg, userdata);
    } catch (...) {
        std::fprintf(stderr, "transcribe: log callback threw; message dropped: %s\n", msg);
    }
}

// Internal emission helper. Not part of the public ABI, not declared in
// any header. Used by transcribe_print_timings and the advisory-warn
// path; future logging from the loader / frontend / decode can call
// through here as well. Drops the message in both the never-configured
// and explicitly-disabled states.
static void transcribe_log_emit(transcribe_log_level level, const char * msg) {
    const auto cb = g_log_cb.load(std::memory_order_acquire);
    if (cb == nullptr || cb == &transcribe_log_cb_disabled) {
        return;
    }
    void * userdata = g_log_userdata.load(std::memory_order_relaxed);
    transcribe_log_invoke(cb, level, msg, userdata);
}

// Wrapper for messages we want surfaced even when the caller never
// configured a log sink: never-configured falls back to stderr so dev /
// CLI builds don't silently drop diagnostics, while an explicit
// transcribe_log_set(NULL, ...) silences them per the public contract.
// The single load (instead of emit-then-recheck) also closes the benign
// race where a concurrently installed callback could double-emit.
static void transcribe_log_emit_or_stderr(transcribe_log_level level, const char * msg) {
    const auto cb = g_log_cb.load(std::memory_order_acquire);
    if (cb == nullptr) {  // never configured
        std::fprintf(stderr, "%s\n", msg);
        return;
    }
    if (cb == &transcribe_log_cb_disabled) {  // explicit silence
        return;
    }
    transcribe_log_invoke(cb, level, msg, g_log_userdata.load(std::memory_order_relaxed));
}

namespace {

// ggml -> transcribe sink bridge, installed by transcribe_log_set.
// transcribe messages carry no trailing newline (the stderr fallback
// appends one), while ggml messages usually embed theirs — strip exactly
// one so a host callback sees a uniform convention. GGML_LOG_LEVEL_CONT
// fragments pass through as-is with level CONT; joining them is the
// host's choice (a fragment is still information).
void transcribe_ggml_log_bridge(enum ggml_log_level level, const char * text, void *) {
    if (text == nullptr) {
        return;
    }
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "%s", text);
    const size_t n = std::strlen(buf);
    if (n > 0 && buf[n - 1] == '\n') {
        buf[n - 1] = '\0';
    }
    transcribe_log_emit_or_stderr(transcribe_log_level_from_ggml(level), buf);
}

}  // namespace

// Internal printf-style logger declared in transcribe-log.h. Renders into
// a bounded stack buffer and forwards to the stderr-fallback emitter, so
// library internals (including per-family run() drivers) reach the
// caller's installed log sink instead of writing raw stderr. See
// transcribe-log.h and docs/input-limits.md.
namespace transcribe {
void log_msg(transcribe_log_level level, const char * fmt, ...) {
    char    buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    transcribe_log_emit_or_stderr(level, buf);
}
}  // namespace transcribe

// Params init functions
//
// Each input params struct is initialized by zero-filling and stamping
// struct_size, which works because every field's documented default is its
// zero value. struct_size == 0 is NOT a defaults form (uninitialized stack
// memory could hit the zero case); callers reach defaults via NULL only.

extern "C" void transcribe_model_load_params_init(struct transcribe_model_load_params * p) {
    transcribe::init_sized(p);
}

// Output struct init functions
//
// Output structs are caller-allocated buffers the library writes into,
// bounded by min(caller struct_size, library size). Every output field's
// zero value means "absent / unknown / false / none", so a zeroed struct +
// struct_size is the correct empty state. struct_size == 0 is rejected by
// the accessors as a "you forgot to init the buffer" error.

extern "C" void transcribe_timings_init(struct transcribe_timings * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_device_info_init(struct transcribe_device_info * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_backend_init_params_init(struct transcribe_backend_init_params * p) {
    transcribe::init_sized(p);
    if (p != nullptr) {
        p->allowed_backends = TRANSCRIBE_BACKEND_MASK_ALL;
    }
}

extern "C" void transcribe_speaker_segment_init(struct transcribe_speaker_segment * p) {
    transcribe::init_sized(p);
}

// Extension helpers

extern "C" transcribe_status transcribe_ext_check(const struct transcribe_ext * ext,
                                                  uint32_t                      expected_kind,
                                                  uint64_t                      min_size) {
    if (ext == nullptr) {
        return TRANSCRIBE_OK;
    }
    if (ext->size < sizeof(struct transcribe_ext)) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    if (ext->kind != expected_kind) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ext->size < min_size) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    return TRANSCRIBE_OK;
}

extern "C" bool transcribe_model_accepts_ext_kind(const struct transcribe_model * model,
                                                  transcribe_ext_slot             slot,
                                                  uint32_t                        kind) {
    if (model == nullptr || model->arch == nullptr) {
        return false;
    }
    if (model->arch->accepts_ext_kind == nullptr) {
        return false;
    }
    return model->arch->accepts_ext_kind(model, slot, kind);
}

extern "C" bool transcribe_model_supports(const struct transcribe_model * model, transcribe_feature feature) {
    // Raw read before any enum-typed load (see enum_field_raw): an unknown
    // feature id from a C caller answers false, per the documented probe
    // contract, instead of being UB.
    return transcribe::has_feature(model, enum_field_raw(&feature));
}

namespace {

constexpr size_t k_min_model_params_size = TRANSCRIBE_FIELD_END(transcribe_model_load_params, device);
constexpr size_t k_min_device_info_size  = TRANSCRIBE_FIELD_END(transcribe_device_info, kind);

constexpr size_t k_min_backend_init_params_size =
    TRANSCRIBE_FIELD_END(transcribe_backend_init_params, allowed_backends);

using transcribe::check_input_struct_size;
using transcribe::check_struct_size;
using transcribe::copy_out_prefix;

}  // namespace

// Backend modules and device discovery
//
// transcribe_init_backends loads ggml backend modules from ONE caller-named
// directory (a Python wheel's package dir, an app bundle, ...) via
// ggml_backend_load_all_from_path; it never widens the search. ggml tolerates
// every per-module failure (missing system deps -> dlopen fails -> module
// skipped), the ship-Vulkan-by-default degradation contract.
//
// NOTE: these definitions must sit at FILE scope, outside the anonymous
// namespace above. clang exports extern "C" functions defined inside an
// unnamed namespace; gcc gives them internal linkage, which strips them from
// the shared library on Linux (undefined references at link).

// ggml's MSVC timer (ggml_time_us/ms in ggml.c) divides QueryPerformanceCounter
// by a global frequency that ggml_time_init() fills in from
// QueryPerformanceFrequency. ggml_init() calls it on first use, but every
// family times its own load (`ggml_time_us()` at the top of <family>_load)
// BEFORE any ggml context exists — so on Windows/MSVC the divisor is still 0
// and model load faults with STATUS_INTEGER_DIVIDE_BY_ZERO (0xC0000094). POSIX
// ggml_time_* read clock_gettime directly (no divisor) so the missing init is
// silently harmless there; arm64 masks integer ÷0 as 0. ggml.h documents
// ggml_time_init() as "call this once at the beginning of the program" — do
// exactly that at the first public entry point.
static void ensure_ggml_time_init() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_time_init(); });
}

#if defined(TRANSCRIBE_GGML_BACKEND_DL)
static std::filesystem::path library_self_dir() {
#    if defined(_WIN32)
    HMODULE    module = nullptr;
    const auto flags  = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&transcribe_init_backends_default), &module)) {
        return {};
    }

    std::wstring path(1024, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (n == 0) {
            return {};
        }
        if (n < path.size()) {
            path.resize(n);
            break;
        }
        if (path.size() >= 32768) {
            return {};
        }
        path.resize(path.size() * 2);
    }
    return std::filesystem::path(path).parent_path();
#    else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&transcribe_init_backends_default), &info) == 0 ||
        info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
        return {};
    }
    std::filesystem::path path(info.dli_fname);
    std::error_code       ec;
    auto                  canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        canonical = std::filesystem::absolute(path, ec);
    }
    if (ec) {
        canonical = path;
    }
    return canonical.parent_path();
#    endif
}

static std::string path_for_c_api(const std::filesystem::path & path) {
#    if defined(_WIN32)
    const auto u8 = path.u8string();
    return std::string(u8.begin(), u8.end());
#    else
    return path.string();
#    endif
}
#endif

// Allowed-backend mask, enforced by backend_reg_filter (installed at static
// init). The first filter call fixes the mask. Mask (low 32 bits) and fixed
// flag share one atomic so _ex() cannot race the first registration.
constexpr uint64_t           k_backend_mask_fixed = uint64_t{ 1 } << 32;
static std::atomic<uint64_t> s_backend_mask_state{ TRANSCRIBE_BACKEND_MASK_ALL };

static uint32_t host_backend_mask() {
    return static_cast<uint32_t>(s_backend_mask_state.load());
}

static bool ascii_iequals(const char * a, const char * b) {
    for (; *a != '\0' && *b != '\0'; ++a, ++b) {
        if (std::tolower(static_cast<unsigned char>(*a)) != std::tolower(static_cast<unsigned char>(*b))) {
            return false;
        }
    }
    return *a == *b;
}

// ascii_iequals for a token that is not NUL-terminated.
static bool ascii_iequals_n(const char * a, size_t n, const char * b) {
    for (size_t i = 0; i < n; ++i, ++b) {
        if (*b == '\0' ||
            std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(*b))) {
            return false;
        }
    }
    return *b == '\0';
}

struct BackendMaskName {
    const char * name;
    uint32_t     bit;
};

// ggml module name -> mask bit; unlisted (incl. "external") is OTHER.
static uint32_t module_mask_bit(const char * name) {
    static const BackendMaskName k_modules[] = {
        { "cpu",    TRANSCRIBE_BACKEND_MASK_CPU    },
        { "blas",   TRANSCRIBE_BACKEND_MASK_CPU    },
        { "zendnn", TRANSCRIBE_BACKEND_MASK_CPU    },
        { "metal",  TRANSCRIBE_BACKEND_MASK_METAL  },
        { "vulkan", TRANSCRIBE_BACKEND_MASK_VULKAN },
        { "cuda",   TRANSCRIBE_BACKEND_MASK_CUDA   },
        { "hip",    TRANSCRIBE_BACKEND_MASK_ROCM   },
    };
    for (const auto & e : k_modules) {
        if (ascii_iequals(name, e.name)) {
            return e.bit;
        }
    }
    return TRANSCRIBE_BACKEND_MASK_OTHER;
}

// TRANSCRIBE_BACKENDS, parsed once; unset/empty is ALL. Runs inside the
// registry filter, so it must not allocate or throw.
static uint32_t env_backend_mask() {
    static const uint32_t mask = [] {
        static const BackendMaskName k_tokens[] = {
            { "cpu",    TRANSCRIBE_BACKEND_MASK_CPU    },
            { "metal",  TRANSCRIBE_BACKEND_MASK_METAL  },
            { "vulkan", TRANSCRIBE_BACKEND_MASK_VULKAN },
            { "cuda",   TRANSCRIBE_BACKEND_MASK_CUDA   },
            { "rocm",   TRANSCRIBE_BACKEND_MASK_ROCM   },
            { "other",  TRANSCRIBE_BACKEND_MASK_OTHER  },
            { "all",    TRANSCRIBE_BACKEND_MASK_ALL    },
        };
        const char * env = std::getenv("TRANSCRIBE_BACKENDS");
        if (env == nullptr || env[0] == '\0') {
            return TRANSCRIBE_BACKEND_MASK_ALL;
        }
        uint32_t     m       = 0;
        bool         unknown = false;
        const char * tok     = env;
        for (const char * p = env;; ++p) {
            if (*p != '\0' && *p != ',' && *p != ' ' && *p != '\t') {
                continue;
            }
            const size_t len = static_cast<size_t>(p - tok);
            if (len > 0) {
                uint32_t bit = 0;
                for (const auto & e : k_tokens) {
                    if (ascii_iequals_n(tok, len, e.name)) {
                        bit = e.bit;
                    }
                }
                unknown = unknown || bit == 0;
                m |= bit;
            }
            if (*p == '\0') {
                break;
            }
            tok = p + 1;
        }
        // Unknown names are dropped, so a typo narrows (fail-closed). Say
        // loudly what that left allowed.
        if (unknown) {
            char           allowed[64] = "all";
            const uint32_t eff         = m | TRANSCRIBE_BACKEND_MASK_CPU;
            if (eff != TRANSCRIBE_BACKEND_MASK_ALL) {
                size_t off = 0;
                for (const auto & e : k_tokens) {
                    if (e.bit != TRANSCRIBE_BACKEND_MASK_ALL && (eff & e.bit) != 0) {
                        off += static_cast<size_t>(
                            std::snprintf(allowed + off, sizeof(allowed) - off, "%s%s", off ? "," : "", e.name));
                    }
                }
            }
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "TRANSCRIBE_BACKENDS='%s': unknown backend name(s) ignored; it now allows: %s "
                                "(valid: cpu, metal, vulkan, cuda, rocm, other, all)",
                                env, allowed);
        }
        return m;
    }();
    return mask;
}

static uint32_t effective_backend_mask(uint32_t host_mask) {
    return (host_mask & env_backend_mask()) | TRANSCRIBE_BACKEND_MASK_CPU;
}

// Called from inside ggml's registry: must not touch the registry or throw.
static bool backend_reg_filter(const char * name) noexcept {
    const uint32_t host    = static_cast<uint32_t>(s_backend_mask_state.fetch_or(k_backend_mask_fixed));
    const uint32_t bit     = module_mask_bit(name != nullptr ? name : "");
    const bool     allowed = (effective_backend_mask(host) & bit) != 0;
    if (!allowed) {
        // Callback-only, like the post-scan device summary: no stderr noise.
        char msg[160];
        std::snprintf(msg, sizeof(msg), "backend '%s' not registered: excluded by allowed-backend mask",
                      name != nullptr ? name : "(null)");
        transcribe_log_emit(TRANSCRIBE_LOG_LEVEL_DEBUG, msg);
    }
    return allowed;
}

static const bool s_backend_reg_filter_installed = [] {
    ggml_backend_set_reg_filter(&backend_reg_filter);
    return true;
}();

static transcribe_status set_host_backend_mask(uint32_t mask) {
    uint64_t state = s_backend_mask_state.load();
    while ((state & k_backend_mask_fixed) == 0) {
        if (s_backend_mask_state.compare_exchange_weak(state, mask)) {
            return TRANSCRIBE_OK;
        }
    }
    const uint32_t current = effective_backend_mask(static_cast<uint32_t>(state));
    const uint32_t wanted  = effective_backend_mask(mask);
    if (current != wanted) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "transcribe_init_backends_ex: allowed-backend mask is already fixed at 0x%08x "
                            "(backends were registered earlier in this process); cannot change it to 0x%08x",
                            current, wanted);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

static transcribe_status transcribe_init_backends_impl(const char * artifact_dir) {
    ensure_ggml_time_init();
    if (artifact_dir == nullptr || artifact_dir[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    {
        std::error_code ec;
        if (!std::filesystem::is_directory(transcribe::path_from_utf8(artifact_dir), ec)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "transcribe_init_backends: %s is not an existing directory",
                                artifact_dir);
            return TRANSCRIBE_ERR_FILE_NOT_FOUND;
        }
    }

    // Idempotency: loading the same directory twice would re-dlopen and
    // re-register every module (duplicate devices). Keyed on the canonical
    // path so two spellings of one directory count as one load.
    static std::mutex            s_mutex;
    static std::set<std::string> s_loaded_dirs;
    std::lock_guard<std::mutex>  lock(s_mutex);

    std::error_code       ec;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(transcribe::path_from_utf8(artifact_dir), ec);
    const std::string     key       = ec ? std::string(artifact_dir) : canonical.u8string();
    if (s_loaded_dirs.find(key) == s_loaded_dirs.end()) {
        ggml_backend_load_all_from_path(artifact_dir);
        s_loaded_dirs.insert(key);

        // Post-scan device summary, through the installed sink only (release
        // ggml compiles its per-module load-failure diagnostics out, so this
        // line is the reliable signal of which backends made it). Callback-only
        // so a host without a sink gets no new import-time stderr noise.
        char         devices[256];
        size_t       off   = 0;
        const size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            const int wrote = std::snprintf(devices + off, sizeof(devices) - off, "%s%s", i == 0 ? "" : ", ",
                                            ggml_backend_dev_name(ggml_backend_dev_get(i)));
            if (wrote < 0 || static_cast<size_t>(wrote) >= sizeof(devices) - off) {
                break;  // truncated; the prefix is summary enough
            }
            off += static_cast<size_t>(wrote);
        }
        char msg[512];
        std::snprintf(msg, sizeof(msg),
                      "transcribe_init_backends: %zu compute device(s) "
                      "registered after scanning %s: %s",
                      n_dev, artifact_dir, off > 0 ? devices : "(none)");
        transcribe_log_emit(TRANSCRIBE_LOG_LEVEL_INFO, msg);
    }

    if (ggml_backend_dev_count() == 0) {
        // Dynamic-backend build pointed at a directory with no usable
        // modules: the process has nothing to compute on. Loud and early
        // beats a confusing model-load failure later.
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "transcribe_init_backends: no compute devices registered after "
                            "loading %s (no usable ggml backend modules found there, and no "
                            "backends are compiled in)",
                            artifact_dir);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

static transcribe_status transcribe_init_backends_default_impl(void) {
    ensure_ggml_time_init();
#if !defined(TRANSCRIBE_GGML_BACKEND_DL)
    return TRANSCRIBE_OK;
#else
    const auto dir = library_self_dir();
    if (dir.empty()) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "transcribe_init_backends_default: could not resolve the "
                            "directory containing libtranscribe");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const auto s = path_for_c_api(dir);
    return transcribe_init_backends(s.c_str());
#endif
}

static transcribe_status transcribe_init_backends_ex_impl(const struct transcribe_backend_init_params * params) {
    ensure_ggml_time_init();
    struct transcribe_backend_init_params p;
    transcribe_backend_init_params_init(&p);
    if (params != nullptr) {
        if (const auto st = check_input_struct_size(params->struct_size, k_min_backend_init_params_size);
            st != TRANSCRIBE_OK) {
            return st;
        }
        std::memcpy(&p, params, std::min<uint64_t>(params->struct_size, sizeof(p)));
    }
    if (const auto st = set_host_backend_mask(p.allowed_backends); st != TRANSCRIBE_OK) {
        return st;
    }
    const auto st = p.artifact_dir != nullptr ? transcribe_init_backends_impl(p.artifact_dir) :
                                                transcribe_init_backends_default_impl();
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    // Static builds register lazily; force it so this call fixes the mask.
    if (ggml_backend_dev_count() == 0) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "transcribe_init_backends_ex: no compute devices registered (allowed-backend mask 0x%08x)",
                            effective_backend_mask(host_backend_mask()));
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

namespace {

// Map ggml's device-type enum onto the public transcribe_device_type. GPU
// and any unexpected type (e.g. the META tensor-parallel aggregate, which we
// never construct) collapse to GPU; the dedicated CPU/IGPU/ACCEL cases are
// reported faithfully.
transcribe_device_type to_device_type(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return TRANSCRIBE_DEVICE_TYPE_CPU;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return TRANSCRIBE_DEVICE_TYPE_IGPU;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return TRANSCRIBE_DEVICE_TYPE_ACCEL;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        default:
            return TRANSCRIBE_DEVICE_TYPE_GPU;
    }
}

// Fill a caller-owned transcribe_device_info from a ggml device, honoring
// the caller's declared struct_size via copy_out_prefix. ggml_backend_dev_get_props
// queries memory live, so every call observes a fresh memory_free snapshot.
void fill_device_info(ggml_backend_dev_t dev, uint64_t caller_size, struct transcribe_device_info * out) {
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);

    const transcribe::BackendKind kind      = transcribe::classify_device(dev);
    const char *                  kind_name = transcribe::kind_name(kind);
    const char *                  name      = ggml_backend_dev_name(dev);
    if (name == nullptr || name[0] == '\0') {
        name = (props.name != nullptr && props.name[0] != '\0') ? props.name : kind_name;
    }
    const char * description = ggml_backend_dev_description(dev);
    if (description == nullptr) {
        description = props.description != nullptr ? props.description : "";
    }

    struct transcribe_device_info staged{};
    staged.struct_size  = caller_size;
    staged.name         = name;
    staged.description  = description;
    staged.kind         = kind_name;
    staged.device_id    = props.device_id;
    staged.memory_total = props.memory_total;
    staged.memory_free  = props.memory_free;
    staged.device_type  = to_device_type(props.type);
    copy_out_prefix(out, &staged, caller_size, sizeof(staged));
}

}  // namespace

static int transcribe_device_count_impl(void) {
    return static_cast<int>(ggml_backend_dev_count());
}

static transcribe_device_t transcribe_device_get_impl(int index) {
    if (index < 0 || index >= static_cast<int>(ggml_backend_dev_count())) {
        return nullptr;
    }
    return reinterpret_cast<transcribe_device_t>(ggml_backend_dev_get(static_cast<size_t>(index)));
}

static ggml_backend_dev_t device_from_handle(transcribe_device_t device) {
    if (device == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t candidate = reinterpret_cast<ggml_backend_dev_t>(device);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_get(i) == candidate) {
            return candidate;
        }
    }
    return nullptr;
}

static transcribe_status transcribe_device_get_info_impl(transcribe_device_t             device,
                                                         struct transcribe_device_info * out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_device_info_size); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_backend_dev_t dev = device_from_handle(device);
    if (dev == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    fill_device_info(dev, out->struct_size, out);
    return TRANSCRIBE_OK;
}

// Takes the request pre-read as a raw int (see enum_field_raw in the
// forwarder) so no enum-typed load of an unknown probe value ever happens;
// unknown values answer false per the documented probe contract.
static bool transcribe_backend_available_impl(int raw) {
    const size_t n = ggml_backend_dev_count();
    if (raw == TRANSCRIBE_BACKEND_AUTO) {
        return n > 0;
    }
    transcribe::BackendKind want;
    switch (raw) {
        case TRANSCRIBE_BACKEND_CPU:
        case TRANSCRIBE_BACKEND_CPU_ACCEL:
            want = transcribe::BackendKind::Cpu;
            break;
        case TRANSCRIBE_BACKEND_METAL:
            want = transcribe::BackendKind::Metal;
            break;
        case TRANSCRIBE_BACKEND_VULKAN:
            want = transcribe::BackendKind::Vulkan;
            break;
        case TRANSCRIBE_BACKEND_CUDA:
            want = transcribe::BackendKind::Cuda;
            break;
        case TRANSCRIBE_BACKEND_ROCM:
            want = transcribe::BackendKind::Rocm;
            break;
        default:
            return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (transcribe::classify_device(ggml_backend_dev_get(i)) == want) {
            return true;
        }
    }
    return false;
}

// Lifecycle

static transcribe_status transcribe_model_load_file_impl(const char *                                path,
                                                         const struct transcribe_model_load_params * params,
                                                         struct transcribe_model **                  out_model) {
    // Set up ggml's timer before any family's load-timing ggml_time_us() runs
    // (Windows/MSVC ÷0 otherwise — see ensure_ggml_time_init).
    ensure_ggml_time_init();
    if (out_model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_model = nullptr;
    if (path == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // NULL params means "all defaults". This is the version-proof
    // spelling of defaults — it carries no struct_size, so it always
    // matches the running library. A zeroed struct ({0}) is a different
    // thing: it claims struct_size == 0 and is rejected below.
    struct transcribe_model_load_params params_defaults;
    transcribe_model_load_params_init(&params_defaults);
    if (params == nullptr) {
        params = &params_defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_model_params_size); st != TRANSCRIBE_OK) {
        return st;
    }

    // device is validated where the registry is available, in
    // load_common::init_backends. NULL means automatic selection; a non-NULL
    // handle selects that exact registered device or fails.

    // Raw-validate the backend request before the families' first
    // enum-typed load of it (see enum_field_raw). init_backends re-checks
    // as defense in depth; this boundary check is what makes every
    // downstream `params->backend` read defined.
    switch (enum_field_raw(&params->backend)) {
        case TRANSCRIBE_BACKEND_AUTO:
        case TRANSCRIBE_BACKEND_CPU:
        case TRANSCRIBE_BACKEND_METAL:
        case TRANSCRIBE_BACKEND_VULKAN:
        case TRANSCRIBE_BACKEND_CPU_ACCEL:
        case TRANSCRIBE_BACKEND_CUDA:
        case TRANSCRIBE_BACKEND_ROCM:
            break;
        default:
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "transcribe_model_load_file: invalid backend request %d",
                                enum_field_raw(&params->backend));
            return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Format sniff by reading the first four bytes (not the file extension,
    // which HF/users mislabel): GGUF magic 0x46554747 is the canonical path,
    // ggml magic 0x67676d6c is a legacy whisper.cpp .bin. Public contract:
    // missing path -> ERR_FILE_NOT_FOUND; every other failure -> ERR_GGUF.
    if (!transcribe::path_is_present(path)) {
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }
    uint32_t magic = 0;
    {
        std::ifstream fin(transcribe::path_from_utf8(path), std::ios::binary);
        if (!fin) {
            // The existence pre-check said the path is reachable but
            // the open failed — permissions / type issue. Treat as a generic
            // load failure rather than FILE_NOT_FOUND.
            return TRANSCRIBE_ERR_GGUF;
        }
        fin.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (!fin || fin.gcount() != sizeof(magic)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "transcribe_model_load_file: short read on file "
                                "magic for %s",
                                path);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // 0x67676d6c ("ggml" little-endian): legacy whisper.cpp .bin. Hand off
    // to the whisper .bin adapter, which validates the hparams as
    // whisper-shaped (rejecting unrelated ggml-magic files like Silero VAD).
    if (magic == 0x67676d6cu) {
        return transcribe::whisper::load_from_bin(path, params, out_model);
    }

    // Header-only GGUF inspection. The Loader is stack-allocated; if
    // anything below bails before transferring ownership, its destructor
    // frees the gguf_context on unwind.
    transcribe::Loader loader;
    if (const transcribe_status st = loader.open(path); st != TRANSCRIBE_OK) {
        return st;
    }

    // Per-family dispatch. The architecture string came from the GGUF KV so
    // the loader guarantees it is non-null and NUL-terminated.
    const transcribe::Arch * arch = transcribe::find_arch(loader.arch().c_str());
    if (arch == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
    }

    // A registered family with no load entry point yet is treated as
    // not-implemented at the central dispatch level so a half-wired
    // handler does not crash.
    if (arch->load == nullptr) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    // Hand off. The handler may take ownership of the gguf_context via
    // loader.release_gguf(); if it doesn't, the Loader destructor frees
    // it normally on stack unwinding.
    const transcribe_status st = arch->load(loader, params, out_model);

    // Copy the string-metadata map onto the model once, centrally. The
    // loader read it during open() and outlives arch->load(), so we set it
    // here instead of in every per-family handler. `variant` is owned by the
    // family (it may default it when stt.variant was absent) and stays on its
    // own accessor; this map is the generic general.* / display surface.
    if (st == TRANSCRIBE_OK && *out_model != nullptr) {
        (*out_model)->meta = loader.meta();
    }
    return st;
}

static void transcribe_model_free_impl(struct transcribe_model * model) {
    // Polymorphic delete: the base has a virtual destructor (anchored
    // in transcribe-model.cpp), so this dispatches to the per-family
    // subclass destructor and that destructor frees gguf_context,
    // weights, etc. Passing NULL is a no-op per the public contract.
    delete model;
}

// Model introspection
//
// These accessors read directly from the base struct; per-family load()
// fills variant / backend / meta before returning success.

extern "C" const char * transcribe_model_arch_string(const struct transcribe_model * model) {
    // The dispatch token's name field has static lifetime and is the
    // canonical arch string; we never store it again on the model.
    if (model == nullptr || model->arch == nullptr) {
        return "";
    }
    return model->arch->name != nullptr ? model->arch->name : "";
}

extern "C" const char * transcribe_model_variant_string(const struct transcribe_model * model) {
    if (model == nullptr) {
        return "";
    }
    return model->variant.c_str();
}

extern "C" const char * transcribe_model_meta_val_str(const struct transcribe_model * model, const char * key) {
    // Generic GGUF string-metadata getter, mirroring llama_model_meta_val_str.
    // Returns a model-owned string (valid until the model is freed) or "" when
    // the model/key is null or the key is absent. There is no fallback to the
    // variant slug — callers that want one read transcribe_model_variant_string.
    if (model == nullptr || key == nullptr) {
        return "";
    }
    const auto it = model->meta.find(key);
    return it != model->meta.end() ? it->second.c_str() : "";
}

extern "C" uint32_t transcribe_model_roles(const struct transcribe_model * model) {
    return model != nullptr ? model->roles : 0;
}

extern "C" const char * transcribe_model_backend(const struct transcribe_model * model) {
    // Empty string means "no runtime backend bound" — see the public header
    // for the full semantic.
    if (model == nullptr) {
        return "";
    }
    return model->backend.c_str();
}

static transcribe_device_t transcribe_model_device_impl(const struct transcribe_model * model) {
    if (model == nullptr || model->primary_backend == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(model->primary_backend);
    if (dev == nullptr || device_from_handle(reinterpret_cast<transcribe_device_t>(dev)) == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<transcribe_device_t>(dev);
}

// C ABI forwarders. Entry points that allocate, reach ggml/driver state,
// or transfer ownership route through api_guard_* here. Other public symbols
// above are kept nothrow by construction: pure POD/c_str reads, prefix
// copies, plain stores, or contained callback emission.

extern "C" transcribe_status transcribe_init_backends(const char * artifact_dir) {
    return api_guard_status("transcribe_init_backends", [&] { return transcribe_init_backends_impl(artifact_dir); });
}

extern "C" transcribe_status transcribe_init_backends_default(void) {
    return api_guard_status("transcribe_init_backends_default",
                            [&] { return transcribe_init_backends_default_impl(); });
}

extern "C" transcribe_status transcribe_init_backends_ex(const struct transcribe_backend_init_params * params) {
    return api_guard_status("transcribe_init_backends_ex", [&] { return transcribe_init_backends_ex_impl(params); });
}

extern "C" uint32_t transcribe_allowed_backends(void) {
    return api_guard_value("transcribe_allowed_backends", static_cast<uint32_t>(TRANSCRIBE_BACKEND_MASK_CPU),
                           [&] { return effective_backend_mask(host_backend_mask()); });
}

extern "C" int transcribe_device_count(void) {
    return api_guard_value("transcribe_device_count", 0, [&] { return transcribe_device_count_impl(); });
}

extern "C" transcribe_device_t transcribe_device_get(int index) {
    return api_guard_value("transcribe_device_get", static_cast<transcribe_device_t>(nullptr),
                           [&] { return transcribe_device_get_impl(index); });
}

extern "C" transcribe_status transcribe_device_get_info(transcribe_device_t             device,
                                                        struct transcribe_device_info * out) {
    return api_guard_status("transcribe_device_get_info", [&] { return transcribe_device_get_info_impl(device, out); });
}

extern "C" bool transcribe_backend_available(transcribe_backend_request kind) {
    // Raw read before any enum-typed load (see enum_field_raw): the lambda
    // must not read `kind` as an enum, or an unknown probe value is UB.
    const int raw = enum_field_raw(&kind);
    return api_guard_value("transcribe_backend_available", false,
                           [&] { return transcribe_backend_available_impl(raw); });
}

extern "C" transcribe_status transcribe_model_load_file(const char *                                path,
                                                        const struct transcribe_model_load_params * params,
                                                        struct transcribe_model **                  out_model) {
    const transcribe_status st = api_guard_status("transcribe_model_load_file", [&] {
        const transcribe_status lst = transcribe_model_load_file_impl(path, params, out_model);
        return lst == TRANSCRIBE_OK && *out_model != nullptr ? transcribe::resolve_roles(*out_model) : lst;
    });
    // Boundary-owned postcondition: failure => *out_model == NULL.
    if (st != TRANSCRIBE_OK && out_model != nullptr && *out_model != nullptr) {
        transcribe_model_free(*out_model);
        *out_model = nullptr;
    }
    return st;
}

extern "C" void transcribe_model_free(struct transcribe_model * model) {
    api_guard_void("transcribe_model_free", [&] { transcribe_model_free_impl(model); });
}

extern "C" transcribe_device_t transcribe_model_device(const struct transcribe_model * model) {
    return api_guard_value("transcribe_model_device", static_cast<transcribe_device_t>(nullptr),
                           [&] { return transcribe_model_device_impl(model); });
}
