// transcribe.cpp - public C API entry points + central dispatch.
//
// What lives here:
//   - The C entry points declared in include/transcribe.h.
//   - The single dispatch site that maps an opened GGUF file to its
//     per-family Arch handler (transcribe_model_load_file).
//   - The log-sink publication / emission helpers.
//   - The factory functions that initialize each public params struct.
//
// What does NOT live here:
//   - GGUF reading itself (transcribe-loader.{h,cpp}).
//   - Per-family load / context / run code (src/arch/<family>/...).
//   - Anything ggml-specific (the public header is ggml-free; this file
//     forwards to the loader which is the only place gguf.h is included).
//
// 2B: model and context lifecycle entry points are real now. The base
// classes in transcribe-model.h / transcribe-session.h hold the data
// the introspection accessors need (caps, variant, backend, arch
// name), so this file reads them directly without going through the
// per-family Arch trait. Per-family code still owns load(),
// init_context(), and run().

#include "transcribe.h"
#include "transcribe/whisper.h"

#include "transcribe-abi.h"
#include "transcribe-arch.h"
#include "transcribe-session.h"
#include "transcribe-loader.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"

#include "arch/whisper/bin_load.h"

#include <sys/stat.h>

#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

// Parameter is `int` (not `transcribe_status`) so a caller passing an
// out-of-range value does not form a bogus enum object. See the comment
// above the declaration in include/transcribe.h.
extern "C" const char * transcribe_status_string(int status) {
    switch (status) {
        case TRANSCRIBE_OK:                       return "ok";
        case TRANSCRIBE_ERR_INVALID_ARG:          return "invalid argument";
        case TRANSCRIBE_ERR_NOT_IMPLEMENTED:      return "not implemented";
        case TRANSCRIBE_ERR_FILE_NOT_FOUND:       return "file not found";
        case TRANSCRIBE_ERR_GGUF:                 return "gguf load error";
        case TRANSCRIBE_ERR_UNSUPPORTED_ARCH:     return "unsupported architecture";
        case TRANSCRIBE_ERR_UNSUPPORTED_VARIANT:  return "unsupported variant";
        case TRANSCRIBE_ERR_OOM:                  return "out of memory";
        case TRANSCRIBE_ERR_BACKEND:              return "backend error";
        case TRANSCRIBE_ERR_SAMPLE_RATE:          return "sample rate mismatch";
        case TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE: return "unsupported language";
        case TRANSCRIBE_ERR_UNSUPPORTED_TASK:     return "unsupported task";
        case TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS:
                                                  return "unsupported timestamp granularity";
        case TRANSCRIBE_ERR_ABORTED:              return "aborted by callback";
        case TRANSCRIBE_ERR_BAD_STRUCT_SIZE:      return "caller-owned struct missing or has bad struct_size";
        case TRANSCRIBE_ERR_UNSUPPORTED_PNC:      return "model does not support runtime PNC control (reserved; not currently returned)";
        case TRANSCRIBE_ERR_UNSUPPORTED_ITN:      return "model does not support runtime ITN control (reserved; not currently returned)";
        default:                                  return "unknown status";
    }
}

namespace {

// Ordered rank for timestamp granularities, used by the dispatcher to
// compare a request against a family's advertised maximum.
//
//   NONE    < SEGMENT  < WORD     < TOKEN
//     0          1          2          3
//
// AUTO is a sentinel at the call site and is handled by the caller
// (resolved to the model's max before ranking). A value that isn't a
// known enumerator gets rank -1 so an out-of-range request compares
// strictly less than every valid max (the dispatcher's earlier
// enum-range switch has already rejected those anyway; this keeps
// the helper total).
int timestamp_rank(transcribe_timestamp_kind k) {
    switch (k) {
        case TRANSCRIBE_TIMESTAMPS_NONE:    return 0;
        case TRANSCRIBE_TIMESTAMPS_SEGMENT: return 1;
        case TRANSCRIBE_TIMESTAMPS_WORD:    return 2;
        case TRANSCRIBE_TIMESTAMPS_TOKEN:   return 3;
        case TRANSCRIBE_TIMESTAMPS_AUTO:    return -1;
    }
    return -1;
}

// Shared run-params validation used by transcribe_run and
// transcribe_stream_begin. Both call sites have already validated
// that session, session->model, and params are non-null. The translate-task
// rejection differs between the two (run mirrors supports_translate,
// streaming-begin rejects unconditionally in v1), so each caller
// applies its own translate check before reaching this helper.
transcribe_status validate_run_params_common(
    const transcribe_session * session,
    const transcribe_run_params *  params)
{
    switch (params->task) {
        case TRANSCRIBE_TASK_TRANSCRIBE:
        case TRANSCRIBE_TASK_TRANSLATE:
            break;
        default:
            return TRANSCRIBE_ERR_INVALID_ARG;
    }
    switch (params->timestamps) {
        case TRANSCRIBE_TIMESTAMPS_NONE:
        case TRANSCRIBE_TIMESTAMPS_AUTO:
        case TRANSCRIBE_TIMESTAMPS_SEGMENT:
        case TRANSCRIBE_TIMESTAMPS_WORD:
        case TRANSCRIBE_TIMESTAMPS_TOKEN:
            break;
        default:
            return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (params->timestamps != TRANSCRIBE_TIMESTAMPS_AUTO) {
        const int req_rank = timestamp_rank(params->timestamps);
        const int max_rank = timestamp_rank(session->model->caps.max_timestamp_kind);
        if (req_rank > max_rank) {
            return TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS;
        }
    }
    if (params->language != nullptr &&
        session->model->caps.n_languages > 0 &&
        session->model->caps.languages != nullptr)
    {
        bool found = false;
        for (int i = 0; i < session->model->caps.n_languages; ++i) {
            const char * entry = session->model->caps.languages[i];
            if (entry != nullptr && std::strcmp(entry, params->language) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
    }
    return TRANSCRIBE_OK;
}

} // namespace

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
//
// Two-atomic log sink. Publication of (cb, userdata) stores userdata
// (relaxed) and then cb (release). Emission acquire-loads cb, early-outs
// on null, and relaxed-loads userdata. The acquire/release pairing on
// the cb pointer establishes a happens-before edge between the install
// thread and any later emitter, which is the property the supported
// "install once at startup" usage model needs.
//
// What this implementation does NOT promise:
//   - Pair-atomic publication of (cb, userdata) under reconfiguration.
//     Two concurrent calls to transcribe_log_set can interleave their
//     stores in modification order such that an emitter observes a cb
//     from one generation and a userdata from another. Even with a
//     single writer, an emitter that still observes the previous cb
//     may already see the new userdata via its relaxed load.
//   - Lifetime safety for old sinks across reconfiguration. After
//     transcribe_log_set returns, an in-flight emission on another
//     thread may already be holding the previous (cb, userdata) on its
//     stack and about to call through it. A caller that frees the old
//     userdata immediately after reconfiguring would create a UAF.
//
// The supported model (per the threading contract in transcribe.h) is
// "install once at startup, before threads or models exist." In that
// model neither caveat above can occur. Anything beyond startup-time
// install is unsupported in 0.x and is the caller's problem; the only
// guarantee the implementation provides in that case is the absence of
// data races, not correct semantics.

namespace {
std::atomic<transcribe_log_callback> g_log_cb       { nullptr };
std::atomic<void *>                  g_log_userdata { nullptr };
} // namespace

extern "C" void transcribe_log_set(transcribe_log_callback cb, void * userdata) {
    g_log_userdata.store(userdata, std::memory_order_relaxed);
    g_log_cb.store(cb, std::memory_order_release);
}

// Internal emission helper. Not part of the public ABI, not declared in
// any header. Used by transcribe_print_timings and the advisory-warn
// path; future logging from the loader / frontend / decode can call
// through here as well.
static void transcribe_log_emit(
    transcribe_log_level level,
    const char *         msg)
{
    const auto cb = g_log_cb.load(std::memory_order_acquire);
    if (cb == nullptr) {
        return;
    }
    void * userdata = g_log_userdata.load(std::memory_order_relaxed);
    cb(level, msg, userdata);
}

// Stderr-fallback wrapper for messages we want surfaced even when the
// caller hasn't installed a log sink. Mirrors transcribe_print_timings'
// fallback so dev / CLI builds don't silently drop the diagnostic.
static void transcribe_log_emit_or_stderr(
    transcribe_log_level level,
    const char *         msg)
{
    transcribe_log_emit(level, msg);
    if (g_log_cb.load(std::memory_order_acquire) == nullptr) {
        std::fprintf(stderr, "%s\n", msg);
    }
}

// ---------------------------------------------------------------------------
// Params init functions
// ---------------------------------------------------------------------------
//
// Every input params struct is initialized by zero-filling and stamping
// struct_size. This works because each field's documented default IS its
// zero value (TASK_TRANSCRIBE, TIMESTAMPS_NONE, PNC/ITN_MODE_DEFAULT,
// BACKEND_AUTO, KV_TYPE_AUTO are all 0; gpu_device 0 = auto; NULL
// pointers; keep_special_tags false = strip). That invariant is also why
// `struct ... p = {0};` is a valid (non-canonical) defaults form, and why
// the input entry points accept struct_size == 0 as "defaults".
//
// These are one-argument by design: they assume the caller and library
// agree on the struct layout, which holds for the supported build-with-
// your-consumers distribution model. A caller passing a struct smaller
// than the library's view would be a version-skew bug handled at release
// time (soname), not in this API.

extern "C" void transcribe_model_load_params_init(struct transcribe_model_load_params * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_session_params_init(struct transcribe_session_params * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_run_params_init(struct transcribe_run_params * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_stream_params_init(struct transcribe_stream_params * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

// ---------------------------------------------------------------------------
// Output struct init functions
// ---------------------------------------------------------------------------
//
// Output structs are caller-allocated buffers the library writes into,
// bounded by min(caller struct_size, library size). The caller must
// declare its buffer size via struct_size; these init functions do that
// and zero-fill the rest. Every output field is designed so its zero
// value means "absent / unknown / false / none", so a zeroed struct +
// struct_size is the correct empty state. (Unlike input params, a {0}
// output struct is NOT accepted by the accessors — struct_size == 0 there
// is a real "you forgot to init the buffer" error.)

extern "C" void transcribe_capabilities_init(struct transcribe_capabilities * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_timings_init(struct transcribe_timings * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_stream_update_init(struct transcribe_stream_update * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_segment_init(struct transcribe_segment * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_word_init(struct transcribe_word * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

extern "C" void transcribe_token_init(struct transcribe_token * p) {
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

// Whisper telemetry + run-extension init and chunk-trace accessors live
// in arch/whisper/public.cpp (family source) so this dispatcher stays
// family-agnostic, matching parakeet/moonshine.

// The whisper/sensevoice/funasr_nano/canary per-family run-params
// factories were removed in the Phase-2 migration:
//   - whisper's knobs are now in transcribe_whisper_run_ext (reached via
//     transcribe_run_params::family); use transcribe_whisper_run_ext_init().
//   - sensevoice/funasr_nano `use_itn` collapsed into the generic
//     transcribe_run_params::itn enum.
//   - canary `pnc` collapsed into the generic transcribe_run_params::pnc enum.

// ---------------------------------------------------------------------------
// Extension helpers
// ---------------------------------------------------------------------------

extern "C" transcribe_status transcribe_ext_check(
    const struct transcribe_ext * ext,
    uint32_t                      expected_kind,
    size_t                        min_size)
{
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

extern "C" bool transcribe_model_accepts_ext_kind(
    const struct transcribe_model * model,
    uint32_t                        kind)
{
    if (model == nullptr || model->arch == nullptr) {
        return false;
    }
    if (model->arch->accepts_ext_kind == nullptr) {
        return false;
    }
    return model->arch->accepts_ext_kind(model, kind);
}

extern "C" bool transcribe_model_supports(
    const struct transcribe_model * model,
    transcribe_feature              feature)
{
    return transcribe::has_feature(model, feature);
}

namespace {

// Minimum struct_size accepted on each caller-owned input/output struct.
// Sized to the prefix the library currently relies on: any field the
// library writes on a given call path must lie inside this prefix. New
// fields appended at the end of the public struct without growing the
// library-side prefix do NOT raise this value.
#define TRANSCRIBE_FIELD_END(type, field) \
    (offsetof(type, field) + sizeof(((type *) 0)->field))

constexpr size_t k_min_model_params_size =
    TRANSCRIBE_FIELD_END(transcribe_model_load_params, gpu_device);
constexpr size_t k_min_context_params_size =
    TRANSCRIBE_FIELD_END(transcribe_session_params, kv_type);
constexpr size_t k_min_run_params_size =
    TRANSCRIBE_FIELD_END(transcribe_run_params, family);
constexpr size_t k_min_stream_params_size =
    TRANSCRIBE_FIELD_END(transcribe_stream_params, family);
constexpr size_t k_min_stream_update_size =
    TRANSCRIBE_FIELD_END(transcribe_stream_update, buffered_ms);
constexpr size_t k_min_capabilities_size =
    TRANSCRIBE_FIELD_END(transcribe_capabilities, streaming_lookahead_ms_min);
constexpr size_t k_min_segment_size =
    TRANSCRIBE_FIELD_END(transcribe_segment, text);
constexpr size_t k_min_word_size =
    TRANSCRIBE_FIELD_END(transcribe_word, text);
constexpr size_t k_min_token_size =
    TRANSCRIBE_FIELD_END(transcribe_token, text);
constexpr size_t k_min_timings_size =
    TRANSCRIBE_FIELD_END(transcribe_timings, decode_ms);
// k_min_whisper_chunk_trace_size lives in arch/whisper/public.cpp with
// the chunk-trace accessor that uses it.

#undef TRANSCRIBE_FIELD_END

// Size-aware ABI helpers (check_struct_size / check_input_struct_size /
// copy_out_prefix) live in transcribe-abi.h so per-family public
// accessors (arch/whisper/public.cpp) share one definition. Pull them
// into this TU's unqualified scope so existing call sites are unchanged.
using transcribe::check_struct_size;
using transcribe::check_input_struct_size;
using transcribe::copy_out_prefix;

// Advisory pnc/itn warning. Emits a WARN log message when a non-DEFAULT
// caller request hits a model that does not support runtime control of
// the corresponding axis, then returns so the dispatcher can proceed.
// "Best effort" semantics: the model produces *something* either way,
// and silently ignoring the request would be a footgun. The reserved
// TRANSCRIBE_ERR_UNSUPPORTED_PNC / _ITN codes are NOT returned today;
// they are placeholders for a future opt-in strict-advisory mode (a
// per-call advisory_strict flag on transcribe_run_params).
//
// The arch + variant strings are included in the message so a grepping
// developer can pinpoint which model dropped the request without having
// to reproduce the call. Emission falls back to stderr when no log
// callback is installed (mirrors transcribe_print_timings).
void warn_unsupported_advisory(
    const struct transcribe_model *  model,
    const struct transcribe_run_params * rp)
{
    if (model == nullptr || rp == nullptr) {
        return;
    }
    const char * arch_name =
        (model->arch != nullptr && model->arch->name != nullptr)
            ? model->arch->name : "(unknown)";
    const char * variant = model->variant.c_str();
    if (variant == nullptr || variant[0] == '\0') {
        variant = "(unknown)";
    }

    char buf[512];
    if (rp->pnc != TRANSCRIBE_PNC_MODE_DEFAULT &&
        !transcribe::has_feature(model, TRANSCRIBE_FEATURE_PNC))
    {
        const char * req = (rp->pnc == TRANSCRIBE_PNC_MODE_ON) ? "ON" : "OFF";
        std::snprintf(buf, sizeof(buf),
            "transcribe_run: caller requested pnc=%s but model '%s' "
            "(variant '%s') does not support pnc control; output will use "
            "the model's default behavior. Use "
            "transcribe_model_supports(model, TRANSCRIBE_FEATURE_PNC) to "
            "pre-check.",
            req, arch_name, variant);
        transcribe_log_emit_or_stderr(TRANSCRIBE_LOG_LEVEL_WARN, buf);
    }
    if (rp->itn != TRANSCRIBE_ITN_MODE_DEFAULT &&
        !transcribe::has_feature(model, TRANSCRIBE_FEATURE_ITN))
    {
        const char * req = (rp->itn == TRANSCRIBE_ITN_MODE_ON) ? "ON" : "OFF";
        std::snprintf(buf, sizeof(buf),
            "transcribe_run: caller requested itn=%s but model '%s' "
            "(variant '%s') does not support itn control; output will use "
            "the model's default behavior. Use "
            "transcribe_model_supports(model, TRANSCRIBE_FEATURE_ITN) to "
            "pre-check.",
            req, arch_name, variant);
        transcribe_log_emit_or_stderr(TRANSCRIBE_LOG_LEVEL_WARN, buf);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle (stubs)
// ---------------------------------------------------------------------------

extern "C" transcribe_status transcribe_model_load_file(
    const char *                           path,
    const struct transcribe_model_load_params * params,
    struct transcribe_model **             out_model)
{
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
    struct transcribe_model_load_params params_defaults; transcribe_model_load_params_init(&params_defaults);
    if (params == nullptr) {
        params = &params_defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_model_params_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Reserved-field validation. gpu_device is documented in the public
    // header as 0 = auto/default in 0.x, reserved for future multi-device
    // selection. 0 is the zero value so a {0} / default-initialized struct
    // passes; reject anything else now so that callers who pass a stale
    // explicit device index (or garbage) get a clean error today rather
    // than a silent success followed by surprise behavior when we actually
    // wire device selection up. A stderr line is included because "invalid
    // argument" alone doesn't tell the caller which field tripped. When
    // multi-device support lands this check relaxes to `< 0 || >= n_devices`.
    if (params->gpu_device != 0) {
        std::fprintf(stderr,
            "transcribe_model_load_file: gpu_device must be 0 (auto) in 0.x "
            "(got %d); multi-device selection is reserved for a "
            "future release\n", params->gpu_device);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Stage 0: format sniff. Two formats are supported today:
    //   GGUF magic 0x46554747 ("GGUF") — the canonical path.
    //   ggml magic 0x67676d6c           — legacy whisper.cpp `.bin`.
    // We detect by reading the first four bytes rather than by file
    // extension: HF distributes whisper.cpp models under the `.bin`
    // suffix even though some are GGUFs internally, and conversely a
    // user could rename a .bin to .gguf.
    //
    // Open semantics preserve the existing public contract: missing
    // path → ERR_FILE_NOT_FOUND; everything else (truncated header,
    // unrecognized magic, structural failure) collapses to ERR_GGUF.
    {
        struct stat st {};
        if (::stat(path, &st) != 0) {
            if (errno == ENOENT || errno == ENOTDIR) {
                return TRANSCRIBE_ERR_FILE_NOT_FOUND;
            }
            // Other stat() failures (EACCES, etc.) are not "not found";
            // the open below will surface them as ERR_GGUF.
        }
    }
    uint32_t magic = 0;
    {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            // stat() said the path is reachable but ifstream couldn't
            // open it — permissions / type issue. Treat as a generic
            // load failure rather than FILE_NOT_FOUND.
            return TRANSCRIBE_ERR_GGUF;
        }
        fin.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (!fin || fin.gcount() != sizeof(magic)) {
            std::fprintf(stderr,
                         "transcribe_model_load_file: short read on file "
                         "magic for %s\n", path);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // 0x67676d6c ("ggml" little-endian): legacy whisper.cpp .bin.
    // Hand off to the whisper-specific .bin adapter, which validates
    // the hparams as whisper-shaped (rejecting Silero VAD and other
    // unrelated `ggml`-magic files with UNSUPPORTED_ARCH).
    if (magic == 0x67676d6cu) {
        return transcribe::whisper::load_from_bin(path, params, out_model);
    }

    // Stage 1: header-only GGUF inspection. The loader returns
    // FILE_NOT_FOUND for a missing path (pre-checked via stat) or
    // ERR_GGUF for any structural failure inside gguf_init_from_file
    // (corrupt magic, truncated header, missing general.architecture).
    // The Loader is stack-allocated; if anything below this point bails
    // out before transferring ownership, its destructor frees the
    // gguf_context on unwind.
    transcribe::Loader loader;
    if (const transcribe_status st = loader.open(path); st != TRANSCRIBE_OK) {
        return st;
    }

    // Stage 2: per-family dispatch. The architecture string came from
    // the GGUF KV section so it is guaranteed non-null and NUL-terminated
    // by the loader.
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
    return arch->load(loader, params, out_model);
}

extern "C" void transcribe_model_free(struct transcribe_model * model) {
    // Polymorphic delete: the base has a virtual destructor (anchored
    // in transcribe-model.cpp), so this dispatches to the per-family
    // subclass destructor and that destructor frees gguf_context,
    // weights, etc. Passing NULL is a no-op per the public contract.
    delete model;
}

extern "C" transcribe_status transcribe_session_init(
    struct transcribe_model *                model,
    const struct transcribe_session_params * params,
    struct transcribe_session **             out_session)
{
    if (out_session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_session = nullptr;
    if (model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // NULL params means "all defaults" (see transcribe_model_load_file).
    struct transcribe_session_params params_defaults; transcribe_session_params_init(&params_defaults);
    if (params == nullptr) {
        params = &params_defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_context_params_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    // n_threads contract from include/transcribe.h: 0 means "library
    // picks a sensible default", positive means "use this many." A
    // negative value is undefined input, not a documented sentinel —
    // reject it here so individual family handlers don't have to
    // re-check the same condition.
    if (params->n_threads < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // The model carries its own arch dispatch pointer. A model that
    // came from a load() call always has arch set; the null check is
    // defensive against a hypothetical malformed model object.
    if (model->arch == nullptr || model->arch->init_context == nullptr) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    return model->arch->init_context(model, params, out_session);
}

extern "C" void transcribe_session_free(struct transcribe_session * session) {
    // Same pattern as transcribe_model_free: virtual destructor on the
    // base lets per-family contexts release their per-run state without
    // a free callback in the Arch trait. Never touches session->model — a
    // session created via transcribe_session_init borrows it.
    delete session;
}

// ---------------------------------------------------------------------------
// Convenience: open / close / get_model
// ---------------------------------------------------------------------------
//
// transcribe_open bundles load + session_init for the common
// single-session case and hands back a session that OWNS its model.
// transcribe_close is the symmetric teardown. These sit entirely on top
// of the public two-step API — no internal shortcuts — so the convenience
// lane and the explicit lane stay interchangeable.

extern "C" transcribe_status transcribe_open(
    const char *                                path,
    const struct transcribe_model_load_params * load_params,
    const struct transcribe_session_params *    session_params,
    struct transcribe_session **                out_session)
{
    if (out_session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_session = nullptr;

    // NULL params are forwarded as-is: the callees already treat NULL as
    // "all defaults", so no substitution is needed here.
    struct transcribe_model * model = nullptr;
    if (const auto st = transcribe_model_load_file(path, load_params, &model);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    struct transcribe_session * session = nullptr;
    if (const auto st = transcribe_session_init(model, session_params, &session);
        st != TRANSCRIBE_OK)
    {
        // Load succeeded but session init failed: own the cleanup so the
        // caller sees the same all-or-nothing contract as the two-step
        // API. No partial state leaks.
        transcribe_model_free(model);
        return st;
    }

    // The convenience session owns the model it loaded; transcribe_close
    // frees both. This is the only place owns_model is set.
    session->owns_model = true;
    *out_session = session;
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_close(struct transcribe_session * session) {
    if (session == nullptr) {
        return;
    }
    // Capture before the session is destroyed; reading session->model
    // after delete would be use-after-free.
    struct transcribe_model * owned =
        session->owns_model ? session->model : nullptr;
    transcribe_session_free(session);
    transcribe_model_free(owned);  // NULL is a no-op
}

extern "C" const struct transcribe_model * transcribe_get_model(
    const struct transcribe_session * session)
{
    if (session == nullptr) {
        return nullptr;
    }
    return session->model;
}

extern "C" void transcribe_set_abort_callback(
    struct transcribe_session * session,
    transcribe_abort_callback   cb,
    void *                      user_data)
{
    if (session == nullptr) {
        return;
    }
    session->abort_cb       = cb;
    session->abort_userdata = user_data;
}

extern "C" bool transcribe_was_aborted(const struct transcribe_session * session) {
    if (session == nullptr) {
        return false;
    }
    return session->was_aborted;
}

// ---------------------------------------------------------------------------
// Streaming dispatcher
// ---------------------------------------------------------------------------
//
// State transitions are managed entirely here; per-family hooks see
// stream_state == ACTIVE on entry to begin/feed/finalize and never
// observe transitions themselves. Hooks own the per-utterance result
// data on the context and may freely mutate tokens/words/segments,
// committed counts, audio cursors, and stream_revision; the
// dispatcher owns lifecycle state and last_status.
//
// The result snapshot and lifecycle state are deliberately separate.
// clear_result() (called by both begin and transcribe_run) wipes the
// snapshot but never touches stream_state, so a future caller that
// wants to "rewind to a fresh result without restarting the stream"
// has a path. The streaming dispatcher is currently the only caller
// that drives stream_state.

extern "C" transcribe_status transcribe_stream_begin(
    struct transcribe_session *              session,
    const struct transcribe_run_params *         run_params,
    const struct transcribe_stream_params *  stream_params)
{
    if (session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // NULL run/stream params each mean "all defaults".
    struct transcribe_run_params    run_params_defaults;    transcribe_run_params_init(&run_params_defaults);
    struct transcribe_stream_params stream_params_defaults; transcribe_stream_params_init(&stream_params_defaults);
    if (run_params == nullptr)    { run_params = &run_params_defaults; }
    if (stream_params == nullptr) { stream_params = &stream_params_defaults; }
    if (const auto st = check_input_struct_size(run_params->struct_size, k_min_run_params_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (const auto st = check_input_struct_size(stream_params->struct_size, k_min_stream_params_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (session->stream_state == TRANSCRIBE_STREAM_ACTIVE) {
        // Caller must finalize or reset before starting a new stream.
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (session->model == nullptr || session->model->arch == nullptr) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    // Capability gate: the model must advertise streaming AND the
    // family must wire the full required hook set. begin / feed /
    // finalize come as a triple — a partially-wired family that
    // accepts begin would otherwise let the caller enter ACTIVE and
    // then get stuck on NOT_IMPLEMENTED at the first feed. stream_reset
    // remains optional; the dispatcher's clear_result + state wipe is
    // sufficient when a family does not need to release per-utterance
    // buffers explicitly.
    if (!session->model->caps.supports_streaming                ||
        session->model->arch->stream_begin    == nullptr        ||
        session->model->arch->stream_feed     == nullptr        ||
        session->model->arch->stream_finalize == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    if (const transcribe_status st = validate_run_params_common(session, run_params);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // v1 rejects TRANSLATE unconditionally for streaming. A future
    // family that supports streaming translate would loosen this in
    // its stream_begin hook, but the central dispatcher refuses
    // upfront so partially-wired callers fail fast.
    if (run_params->task == TRANSCRIBE_TASK_TRANSLATE) {
        return TRANSCRIBE_ERR_UNSUPPORTED_TASK;
    }

    if (stream_params->family != nullptr) {
        if (stream_params->family->size < sizeof(struct transcribe_ext)) {
            return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
        }
        if (!transcribe_model_accepts_ext_kind(session->model, stream_params->family->kind)) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }

    // Advisory warn for pnc/itn requests against models that don't
    // expose the corresponding runtime toggle. Emitted before
    // clear_result so the pre-hook "snapshot preserved on rejection"
    // contract is undisturbed.
    warn_unsupported_advisory(session->model, run_params);

    // All checks pass — clear the previous snapshot and hand off.
    session->clear_result();
    session->t_mel_us    = 0;
    session->t_encode_us = 0;
    session->t_decode_us = 0;
    session->was_aborted = false;
    session->stream_state = TRANSCRIBE_STREAM_ACTIVE;

    const transcribe_status st = session->model->arch->stream_begin(
        session, run_params, stream_params);
    if (st != TRANSCRIBE_OK) {
        // Family hook rejected the begin (config it does not understand,
        // memory allocation failure, etc.). Roll lifecycle back to
        // FAILED so the caller can read last_status; leave the cleared
        // result snapshot in place.
        session->stream_state      = TRANSCRIBE_STREAM_FAILED;
        session->stream_last_status = st;
    }
    return st;
}

// Helper: validate the caller's stream_update struct_size and reset
// every field except struct_size to a clean state before handing the
// update buffer to the family hook. Returns OK when update is NULL
// (the update buffer is optional) or when the size passes validation.
static transcribe_status prepare_stream_update(struct transcribe_stream_update * update) {
    if (update == nullptr) {
        return TRANSCRIBE_OK;
    }
    if (const auto st = check_struct_size(update->struct_size, k_min_stream_update_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    const size_t caller_size = update->struct_size;
    transcribe_stream_update staged{};
    staged.struct_size = caller_size;
    copy_out_prefix(update, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_stream_feed(
    struct transcribe_session *        session,
    const float *                      pcm,
    int                                n_samples,
    struct transcribe_stream_update *  update)
{
    if (const auto st = prepare_stream_update(update); st != TRANSCRIBE_OK) {
        return st;
    }
    if (session == nullptr || pcm == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Zero-length feed is rejected: feed exists to consume audio,
    // and callers that just want to inspect state use the stream
    // accessors. n_samples < 0 is plain garbage.
    if (n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (session->stream_state != TRANSCRIBE_STREAM_ACTIVE) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // begin already confirmed model/arch/hook; we re-check defensively
    // so a malformed context (never happens in practice from a real
    // begin) does not deref a null function pointer.
    if (session->model == nullptr || session->model->arch == nullptr ||
        session->model->arch->stream_feed == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    const transcribe_status st = session->model->arch->stream_feed(
        session, pcm, n_samples, update);
    if (st != TRANSCRIBE_OK) {
        session->stream_state       = TRANSCRIBE_STREAM_FAILED;
        session->stream_last_status = st;
    }
    return st;
}

extern "C" transcribe_status transcribe_stream_finalize(
    struct transcribe_session *        session,
    struct transcribe_stream_update *  update)
{
    if (const auto st = prepare_stream_update(update); st != TRANSCRIBE_OK) {
        return st;
    }
    if (session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (session->stream_state != TRANSCRIBE_STREAM_ACTIVE) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (session->model == nullptr || session->model->arch == nullptr ||
        session->model->arch->stream_finalize == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    const transcribe_status st = session->model->arch->stream_finalize(session, update);
    if (update != nullptr) {
        // Force is_final true regardless of family hook return; the
        // marker describes the call site, not the result. Set after
        // the hook so the family cannot accidentally clear it.
        update->is_final = true;
    }
    if (st != TRANSCRIBE_OK) {
        session->stream_state       = TRANSCRIBE_STREAM_FAILED;
        session->stream_last_status = st;
        return st;
    }
    session->stream_state = TRANSCRIBE_STREAM_FINISHED;
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_stream_reset(struct transcribe_session * session) {
    if (session == nullptr) {
        return;
    }
    // Family hook releases per-utterance state and clears any buffered
    // audio contents while keeping the allocations. A family without
    // streaming just has no hook installed; reset becomes a pure
    // dispatcher state wipe.
    if (session->model != nullptr && session->model->arch != nullptr &&
        session->model->arch->stream_reset != nullptr)
    {
        session->model->arch->stream_reset(session);
    }
    session->clear_result();
    session->stream_state = TRANSCRIBE_STREAM_IDLE;
    // was_aborted is per-stream; reset re-arms it the same way begin
    // does so a caller that resets after an abort starts clean.
    session->was_aborted  = false;
}

extern "C" enum transcribe_stream_state
transcribe_stream_get_state(const struct transcribe_session * session)
{
    if (session == nullptr) {
        return TRANSCRIBE_STREAM_IDLE;
    }
    return session->stream_state;
}

extern "C" int transcribe_stream_revision(const struct transcribe_session * session) {
    if (session == nullptr) {
        return 0;
    }
    return session->stream_revision;
}

extern "C" int transcribe_stream_n_committed_segments(
    const struct transcribe_session * session)
{
    if (session == nullptr) return 0;
    return session->n_committed_segments;
}

extern "C" int transcribe_stream_n_committed_words(
    const struct transcribe_session * session)
{
    if (session == nullptr) return 0;
    return session->n_committed_words;
}

extern "C" int transcribe_stream_n_committed_tokens(
    const struct transcribe_session * session)
{
    if (session == nullptr) return 0;
    return session->n_committed_tokens;
}

extern "C" transcribe_status transcribe_stream_last_status(
    const struct transcribe_session * session)
{
    if (session == nullptr) {
        return TRANSCRIBE_OK;
    }
    return session->stream_last_status;
}

extern "C" transcribe_status transcribe_run(
    struct transcribe_session *      session,
    const float *                    pcm,
    int                              n_samples,
    const struct transcribe_run_params * params)
{
    // Parameter-shape validation runs first and does not touch session
    // state. A caller that passes NULL pointers or a non-positive sample
    // count gets ERR_INVALID_ARG back without any visible side effect
    // — including the session's result fields, which are preserved
    // across a malformed call. This is the narrower half of the
    // "transcribe_run replaces the previous result" contract: the
    // replacement happens when the call is well-formed enough that
    // the dispatcher is willing to dereference session. A call that
    // isn't well-formed isn't considered to have "run".
    if (session == nullptr || pcm == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // n_samples must be strictly positive, matching transcribe_stream_feed.
    // There is no meaningful "transcribe zero samples" operation; a
    // zero-length batch is treated as a caller error, not an empty run.
    if (n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // NULL params means "all defaults" (transcribe vs translate, no
    // timestamps, etc.). A well-formed default run is not a malformed
    // call, so it proceeds and replaces the previous result.
    struct transcribe_run_params params_defaults; transcribe_run_params_init(&params_defaults);
    if (params == nullptr) {
        params = &params_defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_run_params_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    // A run cannot replace an active stream's results — that would
    // strand the in-flight stream's per-family state. Caller must
    // finalize or reset first. FINISHED and FAILED both fall through;
    // the run path below resets stream_state to IDLE.
    if (session->stream_state == TRANSCRIBE_STREAM_ACTIVE) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-clear family-extension validation. A malformed run ext must
    // NOT wipe the previous result snapshot — the snapshot-clearing
    // semantics below are reserved for well-formed calls. Mirrors the
    // transcribe_stream_begin contract for stream_params->family. The
    // kind-accept probe needs a valid model; when model is null we
    // skip the pre-clear check and let the post-clear NOT_IMPLEMENTED
    // path handle it (the existing snapshot-wipe-on-NOT_IMPLEMENTED
    // contract is preserved).
    if (params->family != nullptr &&
        session->model != nullptr && session->model->arch != nullptr)
    {
        if (params->family->size < sizeof(struct transcribe_ext)) {
            return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
        }
        if (!transcribe_model_accepts_ext_kind(session->model, params->family->kind)) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }

    // Advisory warn for pnc/itn requests against models that don't
    // expose the corresponding runtime toggle. Best-effort semantics:
    // log a WARN and proceed with the model's default behavior. Emitted
    // before clear_result so a malformed advisory doesn't disturb the
    // previous snapshot. Skipped when model is null — the post-clear
    // NOT_IMPLEMENTED path will signal that more directly.
    if (session->model != nullptr) {
        warn_unsupported_advisory(session->model, params);
    }

    // Result-replacement contract: everything past this point either
    // succeeds and writes a fresh result, or fails and leaves the
    // context in the documented "no result" sentinel state. Clear
    // eagerly so every downstream rejection path — NOT_IMPLEMENTED on
    // an incomplete arch, INVALID_ARG on an out-of-range enum,
    // UNSUPPORTED_TASK / _TIMESTAMPS / _LANGUAGE on a capability
    // mismatch — inherits the sentinel without each branch having
    // to remember to call clear_result() itself.
    //
    // Per-family run() handlers call clear_result() again after
    // their own front-matter checks succeed; that call is now
    // redundant but idempotent, and removing it is a refactor
    // deferred to a later pass.
    session->clear_result();
    session->t_mel_us    = 0;
    session->t_encode_us = 0;
    session->t_decode_us = 0;
    session->was_aborted = false;
    // Force stream_state to IDLE: clear_result deliberately preserves
    // lifecycle state, but a well-formed transcribe_run subsumes any
    // prior FINISHED/FAILED stream — after a one-shot run the context
    // is no longer meaningfully in a streaming lifecycle.
    session->stream_state = TRANSCRIBE_STREAM_IDLE;

    if (session->model == nullptr || session->model->arch == nullptr ||
        session->model->arch->run == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    // Centralized run-param validation. The per-family run() handlers
    // can assume the params struct has been sanity-checked here, so
    // they don't need to repeat the enum-range, capability, or
    // language-list checks. Anything still family-local — tokenizer-
    // vocabulary constraints, prompt-template quirks, target_language
    // handling for TRANSLATE — continues to live inside the family's
    // run() handler.
    //
    if (const transcribe_status st = validate_run_params_common(session, params);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Reject TRANSLATE for models that don't declare support. The
    // capability is set per-arch in apply_family_invariants and may
    // be overridden by stt.capability.* KV. Models that DO support
    // translate may still validate target_language inside their own
    // run() handler against the model-specific language list. This
    // check runs after enum-range validation in the shared helper so
    // a garbage task value still reports as ERR_INVALID_ARG.
    if (params->task == TRANSCRIBE_TASK_TRANSLATE &&
        !session->model->caps.supports_translate)
    {
        return TRANSCRIBE_ERR_UNSUPPORTED_TASK;
    }

    return session->model->arch->run(session, pcm, n_samples, params);
}

// ---------------------------------------------------------------------------
// Model introspection
// ---------------------------------------------------------------------------
//
// All four accessors read directly from the base struct. There is no
// per-family callback for them: per-family load() is responsible for
// filling caps / variant / backend in the base before returning success.

extern "C" transcribe_status transcribe_model_get_capabilities(
    const struct transcribe_model *  model,
    struct transcribe_capabilities * out_caps)
{
    if (model == nullptr || out_caps == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out_caps->struct_size, k_min_capabilities_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    // Preserve the caller-declared size, then write only the prefix
    // that fits in both the caller's buffer and this library's view;
    // any tail bytes stay as the caller initialized them (zero, by
    // the INIT macro contract).
    //
    // Pointer fields written into the caller buffer (e.g. languages)
    // remain model-owned and valid until transcribe_model_free().
    const size_t caller_size = out_caps->struct_size;
    transcribe_capabilities staged = model->caps;
    staged.struct_size = caller_size;
    copy_out_prefix(out_caps, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

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

extern "C" int transcribe_tokenize(
    const struct transcribe_model * model,
    const char *                    text,
    int32_t *                       tokens,
    size_t                          n_max)
{
    if (model == nullptr || text == nullptr) {
        return INT_MIN;
    }
    const transcribe::Tokenizer * tok = model->tokenizer();
    if (tok == nullptr) {
        return INT_MIN;
    }

    std::vector<int32_t> ids;
    if (tok->encode(std::string(text), ids) != TRANSCRIBE_OK) {
        return INT_MIN;
    }

    const size_t n = ids.size();
    if (n > static_cast<size_t>(INT_MAX)) {
        return INT_MIN;
    }
    if (n > n_max) {
        // Negative-of-N retry contract (mirrors whisper.cpp). Guard
        // against -n overflowing int range for pathologically large
        // vocabularies; fall back to INT_MIN + 1 so the caller still
        // gets a hard-failure code rather than a stale pointer.
        if (n > static_cast<size_t>(INT_MAX)) {
            return INT_MIN;
        }
        return -static_cast<int>(n);
    }
    if (tokens == nullptr && n > 0) {
        return INT_MIN;
    }
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = ids[i];
    }
    return static_cast<int>(n);
}

extern "C" const char * transcribe_model_backend(const struct transcribe_model * model) {
    // Empty string means "no runtime backend bound" — see the public
    // header for the full semantic. In 2B no model is ever bound to a
    // backend, so per-family load() leaves this empty and we just
    // pass it through.
    if (model == nullptr) {
        return "";
    }
    return model->backend.c_str();
}

// ---------------------------------------------------------------------------
// Timings
// ---------------------------------------------------------------------------
//
// All accumulators live on the base classes (transcribe_model::t_load_us
// and transcribe_session::t_{mel,encode,decode}_us). Per-family run()
// drivers populate the context-side fields; per-family load() drivers
// populate t_load_us. The accessors here just convert microseconds to
// milliseconds and return a small value type.

extern "C" transcribe_status transcribe_get_timings(
    const struct transcribe_session * session,
    struct transcribe_timings *       out_timings)
{
    if (session == nullptr || out_timings == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out_timings->struct_size, k_min_timings_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    const size_t caller_size = out_timings->struct_size;
    transcribe_timings staged{};
    staged.struct_size = caller_size;
    if (session->model != nullptr) {
        staged.load_ms = static_cast<float>(session->model->t_load_us) / 1000.0f;
    }
    staged.mel_ms    = static_cast<float>(session->t_mel_us)    / 1000.0f;
    staged.encode_ms = static_cast<float>(session->t_encode_us) / 1000.0f;
    staged.decode_ms = static_cast<float>(session->t_decode_us) / 1000.0f;
    copy_out_prefix(out_timings, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" void
transcribe_print_timings(const struct transcribe_session * session)
{
    if (session == nullptr) {
        return;
    }
    struct transcribe_timings t; transcribe_timings_init(&t);
    (void)transcribe_get_timings(session, &t);
    char buf[256];

    std::snprintf(buf, sizeof(buf),
                  "timings: load=%.2f ms  mel=%.2f ms  "
                  "encode=%.2f ms  decode=%.2f ms",
                  t.load_ms, t.mel_ms, t.encode_ms, t.decode_ms);
    transcribe_log_emit(TRANSCRIBE_LOG_LEVEL_INFO, buf);

    // Fallback: if no callback is registered, the log_emit above is a
    // no-op. Print to stderr so the timings are observable in the
    // common dev case (transcribe-cli without an installed sink).
    if (g_log_cb.load(std::memory_order_acquire) == nullptr) {
        std::fprintf(stderr, "%s\n", buf);
    }
}

extern "C" void
transcribe_reset_timings(struct transcribe_session * session)
{
    if (session == nullptr) {
        return;
    }
    // load_us is intentionally left alone — it's a model-scoped fact,
    // not a per-call accumulator.
    session->t_mel_us    = 0;
    session->t_encode_us = 0;
    session->t_decode_us = 0;
}

// ---------------------------------------------------------------------------
// Result accessors
// ---------------------------------------------------------------------------
//
// All accessors read from the base context's result storage
// (transcribe_session::tokens / words / segments / full_text /
// result_kind / has_result), which is populated by the per-family
// run() driver during decode. Out-of-range indices and pre-run
// access return safe sentinels per the public contract; we never
// reach into invalid memory.

extern "C" const char * transcribe_full_text(const struct transcribe_session * session) {
    if (session == nullptr || !session->has_result) {
        return "";
    }
    return session->full_text.c_str();
}

extern "C" const char * transcribe_detected_language(const struct transcribe_session * session) {
    if (session == nullptr || !session->has_result) {
        return "";
    }
    return session->detected_language.c_str();
}

extern "C" transcribe_timestamp_kind
transcribe_returned_timestamp_kind(const struct transcribe_session * session) {
    if (session == nullptr || !session->has_result) {
        return TRANSCRIBE_TIMESTAMPS_NONE;
    }
    return session->result_kind;
}

extern "C" int transcribe_n_segments(const struct transcribe_session * session) {
    if (session == nullptr || !session->has_result) return 0;
    return static_cast<int>(session->segments.size());
}
extern "C" int transcribe_n_words(const struct transcribe_session * session) {
    if (session == nullptr || !session->has_result) return 0;
    return static_cast<int>(session->words.size());
}
extern "C" int transcribe_n_tokens(const struct transcribe_session * session) {
    if (session == nullptr || !session->has_result) return 0;
    return static_cast<int>(session->tokens.size());
}

// ---------------------------------------------------------------------------
// Result accessors - per-item rows
// ---------------------------------------------------------------------------
//
// 3 copy-out accessors backed by the context's segments / words /
// tokens vectors. Each takes a caller-owned struct initialized via
// TRANSCRIBE_*_INIT and writes only the prefix that fits within the
// caller's struct_size. Out-of-range index or pre-run access leaves
// the caller's struct as zero-init (text==NULL, scalars 0); the
// dispatch path always returns OK in those cases so callers can use
// the row's text!=NULL check as the "row present" signal without
// branching on status.
//
// `text` pointers in the staged structs alias the context's
// std::string storage. The library treats that as session-owned: valid
// until the next transcribe_run / transcribe_stream_begin /
// transcribe_stream_reset / transcribe_session_free on the same
// context. The public header documents the lifetime contract.

extern "C" transcribe_status transcribe_get_segment(
    const struct transcribe_session * session,
    int                               i,
    struct transcribe_segment *       out)
{
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_segment_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    const size_t caller_size = out->struct_size;
    transcribe_segment zero{};
    zero.struct_size = caller_size;
    copy_out_prefix(out, &zero, caller_size, sizeof(zero));
    if (session == nullptr || !session->has_result || i < 0 ||
        static_cast<size_t>(i) >= session->segments.size())
    {
        return TRANSCRIBE_OK;
    }
    const auto & s = session->segments[static_cast<size_t>(i)];
    transcribe_segment staged{};
    staged.struct_size = caller_size;
    staged.t0_ms       = s.t0_ms;
    staged.t1_ms       = s.t1_ms;
    staged.first_word  = s.first_word;
    staged.n_words     = s.n_words;
    staged.first_token = s.first_token;
    staged.n_tokens    = s.n_tokens;
    staged.text        = s.text.c_str();
    copy_out_prefix(out, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_get_word(
    const struct transcribe_session * session,
    int                               i,
    struct transcribe_word *          out)
{
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_word_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    const size_t caller_size = out->struct_size;
    transcribe_word zero{};
    zero.struct_size = caller_size;
    copy_out_prefix(out, &zero, caller_size, sizeof(zero));
    if (session == nullptr || !session->has_result || i < 0 ||
        static_cast<size_t>(i) >= session->words.size())
    {
        return TRANSCRIBE_OK;
    }
    const auto & w = session->words[static_cast<size_t>(i)];
    transcribe_word staged{};
    staged.struct_size = caller_size;
    staged.t0_ms       = w.t0_ms;
    staged.t1_ms       = w.t1_ms;
    staged.seg_index   = w.seg_index;
    staged.first_token = w.first_token;
    staged.n_tokens    = w.n_tokens;
    staged.text        = w.text.c_str();
    copy_out_prefix(out, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_get_token(
    const struct transcribe_session * session,
    int                               i,
    struct transcribe_token *         out)
{
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_token_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    const size_t caller_size = out->struct_size;
    transcribe_token zero{};
    zero.struct_size = caller_size;
    copy_out_prefix(out, &zero, caller_size, sizeof(zero));
    if (session == nullptr || !session->has_result || i < 0 ||
        static_cast<size_t>(i) >= session->tokens.size())
    {
        return TRANSCRIBE_OK;
    }
    const auto & t = session->tokens[static_cast<size_t>(i)];
    transcribe_token staged{};
    staged.struct_size = caller_size;
    staged.id          = t.id;
    staged.p           = t.p;
    staged.t0_ms       = t.t0_ms;
    staged.t1_ms       = t.t1_ms;
    staged.seg_index   = t.seg_index;
    staged.word_index  = t.word_index;
    staged.text        = t.text.c_str();
    copy_out_prefix(out, &staged, caller_size, sizeof(staged));
    return TRANSCRIBE_OK;
}
