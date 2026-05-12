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
// classes in transcribe-model.h / transcribe-context.h hold the data
// the introspection accessors need (caps, variant, backend, arch
// name), so this file reads them directly without going through the
// per-family Arch trait. Per-family code still owns load(),
// init_context(), and run().

#include "transcribe.h"

#include "transcribe-arch.h"
#include "transcribe-context.h"
#include "transcribe-loader.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"

#include "arch/whisper/whisper.h"
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
// any header. Used by future logging code paths (loader, frontend, decode);
// kept here in pass 1 so the load-side memory ordering is locked in
// alongside the store-side. Marked maybe_unused because no caller exists
// yet at pass 1.
[[maybe_unused]] static void transcribe_log_emit(
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

// ---------------------------------------------------------------------------
// Params factories
// ---------------------------------------------------------------------------

extern "C" struct transcribe_model_params transcribe_model_default_params(void) {
    struct transcribe_model_params p = {};
    p.backend    = TRANSCRIBE_BACKEND_AUTO;
    p.gpu_device = -1;
    return p;
}

extern "C" struct transcribe_context_params transcribe_context_default_params(void) {
    struct transcribe_context_params p = {};
    p.n_threads = 0; // 0 = library picks a default
    p.kv_type   = TRANSCRIBE_KV_TYPE_AUTO;
    return p;
}

extern "C" struct transcribe_params transcribe_default_params(void) {
    struct transcribe_params p = {};
    p.task               = TRANSCRIBE_TASK_TRANSCRIBE;
    p.timestamps         = TRANSCRIBE_TIMESTAMPS_NONE;
    p.language           = nullptr;
    p.target_language    = nullptr;
    p.strip_special_tags = true;
    p.whisper            = nullptr;
    p.sensevoice         = nullptr;
    p.funasr_nano        = nullptr;
    p.canary             = nullptr;
    return p;
}

extern "C" struct transcribe_sensevoice_params
transcribe_sensevoice_default_params(void)
{
    struct transcribe_sensevoice_params p = {};
    p.use_itn = false;
    return p;
}

extern "C" struct transcribe_funasr_nano_params
transcribe_funasr_nano_default_params(void)
{
    struct transcribe_funasr_nano_params p = {};
    p.use_itn = false;
    return p;
}

extern "C" struct transcribe_canary_params
transcribe_canary_default_params(void)
{
    struct transcribe_canary_params p = {};
    p.pnc = true;
    return p;
}

extern "C" struct transcribe_whisper_params transcribe_whisper_default_params(void) {
    struct transcribe_whisper_params p = {};
    p.initial_prompt           = nullptr;
    p.prompt_tokens            = nullptr;
    p.n_prompt_tokens          = 0;
    p.prompt_condition         = TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;
    p.condition_on_prev_tokens = false;
    p.max_prev_context_tokens  = 223; // max_target_positions / 2 - 1
    p.temperature              = 0.0f;
    p.temperature_inc          = 0.2f;
    p.compression_ratio_thold  = 2.4f;
    p.logprob_thold            = -1.0f;
    p.no_speech_thold          = 0.6f;
    p.seed                     = 0; // 0 = nondeterministic (whisper.cpp convention)
    p.max_initial_timestamp    = 1.0f;
    return p;
}

// ---------------------------------------------------------------------------
// Lifecycle (stubs)
// ---------------------------------------------------------------------------

extern "C" transcribe_status transcribe_model_load_file(
    const char *                           path,
    const struct transcribe_model_params * params,
    struct transcribe_model **             out_model)
{
    if (out_model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_model = nullptr;
    if (path == nullptr || params == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Reserved-field validation. gpu_device is documented in the
    // public header as MUST-be-(-1) in 0.x, reserved for future
    // multi-device selection. Reject anything else now so that
    // callers who pass a stale "device 0" (or garbage) get a clean
    // error today rather than a silent success followed by surprise
    // behavior when we actually wire device selection up. A stderr
    // line is included because "invalid argument" alone doesn't tell
    // the caller which field tripped. When multi-device support
    // lands this check relaxes to `< -1 || >= n_devices`.
    if (params->gpu_device != -1) {
        std::fprintf(stderr,
            "transcribe_model_load_file: gpu_device must be -1 in 0.x "
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

extern "C" transcribe_status transcribe_context_init(
    struct transcribe_model *                model,
    const struct transcribe_context_params * params,
    struct transcribe_context **             out_ctx)
{
    if (out_ctx == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_ctx = nullptr;
    if (model == nullptr || params == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
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
    return model->arch->init_context(model, params, out_ctx);
}

extern "C" void transcribe_context_free(struct transcribe_context * ctx) {
    // Same pattern as transcribe_model_free: virtual destructor on the
    // base lets per-family contexts release their per-run state without
    // a free callback in the Arch trait.
    delete ctx;
}

extern "C" void transcribe_set_abort_callback(
    struct transcribe_context * ctx,
    transcribe_abort_callback   cb,
    void *                      user_data)
{
    if (ctx == nullptr) {
        return;
    }
    ctx->abort_cb       = cb;
    ctx->abort_userdata = user_data;
}

extern "C" bool transcribe_was_aborted(const struct transcribe_context * ctx) {
    if (ctx == nullptr) {
        return false;
    }
    return ctx->was_aborted;
}

// Whisper-specific decoding trace accessors. The storage lives on
// WhisperContext; we downcast via the model's arch name rather than
// RTTI so non-RTTI builds keep working. Non-Whisper contexts and
// out-of-range indices are defined as zeroed/empty returns.
static const transcribe::whisper::WhisperContext *
maybe_whisper_context(const struct transcribe_context * ctx)
{
    if (ctx == nullptr) return nullptr;
    if (ctx->model == nullptr) return nullptr;
    if (ctx->model->arch == nullptr) return nullptr;
    const char * name = ctx->model->arch->name;
    if (name == nullptr || std::strcmp(name, "whisper") != 0) {
        return nullptr;
    }
    return static_cast<const transcribe::whisper::WhisperContext *>(ctx);
}

extern "C" int transcribe_get_whisper_chunk_count(
    const struct transcribe_context * ctx)
{
    const auto * wc = maybe_whisper_context(ctx);
    if (wc == nullptr) return 0;
    return static_cast<int>(wc->chunk_traces.size());
}

extern "C" struct transcribe_whisper_chunk_trace
transcribe_get_whisper_chunk_trace(
    const struct transcribe_context * ctx, int i)
{
    struct transcribe_whisper_chunk_trace zero = {};
    const auto * wc = maybe_whisper_context(ctx);
    if (wc == nullptr) return zero;
    if (i < 0 || static_cast<size_t>(i) >= wc->chunk_traces.size()) {
        return zero;
    }
    return wc->chunk_traces[static_cast<size_t>(i)];
}

extern "C" transcribe_status transcribe_run(
    struct transcribe_context *      ctx,
    const float *                    pcm,
    int                              n_samples,
    const struct transcribe_params * params)
{
    // Parameter-shape validation runs first and does not touch ctx
    // state. A caller that passes NULL pointers or a negative sample
    // count gets ERR_INVALID_ARG back without any visible side effect
    // — including the context's result fields, which are preserved
    // across a malformed call. This is the narrower half of the
    // "transcribe_run replaces the previous result" contract: the
    // replacement happens when the call is well-formed enough that
    // the dispatcher is willing to dereference ctx. A call that
    // isn't well-formed isn't considered to have "run".
    if (ctx == nullptr || pcm == nullptr || params == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (n_samples < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Result-replacement contract: everything past this point either
    // succeeds and writes a fresh result, or fails and leaves the
    // context in the documented "no result" sentinel state. Clear
    // eagerly so every downstream rejection path — NOT_IMPLEMENTED
    // on an incomplete arch, INVALID_ARG on an out-of-range enum,
    // UNSUPPORTED_TASK / _TIMESTAMPS / _LANGUAGE on a capability
    // mismatch — inherits the sentinel without each branch having
    // to remember to call clear_result() itself.
    //
    // Per-family run() handlers call clear_result() again after
    // their own front-matter checks succeed; that call is now
    // redundant but idempotent, and removing it is a refactor
    // deferred to a later pass.
    ctx->clear_result();
    ctx->t_mel_us    = 0;
    ctx->t_encode_us = 0;
    ctx->t_decode_us = 0;
    ctx->was_aborted = false;

    if (ctx->model == nullptr || ctx->model->arch == nullptr ||
        ctx->model->arch->run == nullptr)
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
    // Out-of-range enum values are handled separately from
    // capability-mediated rejections so the caller can distinguish
    // "you passed garbage" (ERR_INVALID_ARG) from "this model doesn't
    // support that legitimate request" (ERR_UNSUPPORTED_TASK).
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

    // Reject TRANSLATE for models that don't declare support. The
    // capability is set per-arch in apply_family_invariants and may
    // be overridden by stt.capability.* KV. Models that DO support
    // translate may still validate target_language inside their own
    // run() handler against the model-specific language list.
    if (params->task == TRANSCRIBE_TASK_TRANSLATE &&
        !ctx->model->caps.supports_translate)
    {
        return TRANSCRIBE_ERR_UNSUPPORTED_TASK;
    }

    // Reject a timestamp granularity finer than the model can
    // produce. AUTO passes through unconditionally — the per-family
    // run() handler resolves AUTO to its own max_timestamp_kind when
    // it assembles the result. A non-AUTO request is treated as a
    // ceiling: the handler may emit the requested granularity or
    // coarser, never finer.
    //
    // Advertising NONE and then fabricating a zero-timed WordEntry
    // was the bug that motivated the ceiling check — if a family
    // says it can't produce word alignments, the dispatcher holds
    // it to that contract.
    if (params->timestamps != TRANSCRIBE_TIMESTAMPS_AUTO) {
        const int req_rank = timestamp_rank(params->timestamps);
        const int max_rank = timestamp_rank(ctx->model->caps.max_timestamp_kind);
        if (req_rank > max_rank) {
            return TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS;
        }
    }

    // Reject a caller-supplied source language that the model does
    // not declare. A NULL language means "use family default or
    // auto-detect"; that's always accepted at dispatch level. When
    // n_languages == 0 the model's language list is "information
    // gap, not a claim" (see transcribe-meta.cpp read_languages_kv)
    // and we cannot validate, so we defer to the family handler.
    //
    // Per-family run() handlers may still reject a language for
    // tokenizer-vocabulary reasons, but they no longer need to
    // replicate this metadata walk.
    if (params->language != nullptr &&
        ctx->model->caps.n_languages > 0 &&
        ctx->model->caps.languages != nullptr)
    {
        bool found = false;
        for (int i = 0; i < ctx->model->caps.n_languages; ++i) {
            const char * entry = ctx->model->caps.languages[i];
            if (entry != nullptr && std::strcmp(entry, params->language) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
    }

    return ctx->model->arch->run(ctx, pcm, n_samples, params);
}

// ---------------------------------------------------------------------------
// Model introspection
// ---------------------------------------------------------------------------
//
// All four accessors read directly from the base struct. There is no
// per-family callback for them: per-family load() is responsible for
// filling caps / variant / backend in the base before returning success.

extern "C" const struct transcribe_capabilities *
transcribe_model_capabilities(const struct transcribe_model * model) {
    if (model == nullptr) {
        return nullptr;
    }
    return &model->caps;
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
// and transcribe_context::t_{mel,encode,decode}_us). Per-family run()
// drivers populate the context-side fields; per-family load() drivers
// populate t_load_us. The accessors here just convert microseconds to
// milliseconds and return a small value type.

extern "C" struct transcribe_timings
transcribe_get_timings(const struct transcribe_context * ctx)
{
    struct transcribe_timings out = {};
    if (ctx == nullptr) {
        return out;
    }
    if (ctx->model != nullptr) {
        out.load_ms = static_cast<float>(ctx->model->t_load_us) / 1000.0f;
    }
    out.mel_ms    = static_cast<float>(ctx->t_mel_us)    / 1000.0f;
    out.encode_ms = static_cast<float>(ctx->t_encode_us) / 1000.0f;
    out.decode_ms = static_cast<float>(ctx->t_decode_us) / 1000.0f;
    return out;
}

extern "C" void
transcribe_print_timings(const struct transcribe_context * ctx)
{
    if (ctx == nullptr) {
        return;
    }
    const struct transcribe_timings t = transcribe_get_timings(ctx);
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
transcribe_reset_timings(struct transcribe_context * ctx)
{
    if (ctx == nullptr) {
        return;
    }
    // load_us is intentionally left alone — it's a model-scoped fact,
    // not a per-call accumulator.
    ctx->t_mel_us    = 0;
    ctx->t_encode_us = 0;
    ctx->t_decode_us = 0;
}

// ---------------------------------------------------------------------------
// Result accessors
// ---------------------------------------------------------------------------
//
// All accessors read from the base context's result storage
// (transcribe_context::tokens / words / segments / full_text /
// result_kind / has_result), which is populated by the per-family
// run() driver during decode. Out-of-range indices and pre-run
// access return safe sentinels per the public contract; we never
// reach into invalid memory.

extern "C" const char * transcribe_full_text(const struct transcribe_context * ctx) {
    if (ctx == nullptr || !ctx->has_result) {
        return "";
    }
    return ctx->full_text.c_str();
}

extern "C" const char * transcribe_detected_language(const struct transcribe_context * ctx) {
    if (ctx == nullptr || !ctx->has_result) {
        return "";
    }
    return ctx->detected_language.c_str();
}

extern "C" transcribe_timestamp_kind
transcribe_returned_timestamp_kind(const struct transcribe_context * ctx) {
    if (ctx == nullptr || !ctx->has_result) {
        return TRANSCRIBE_TIMESTAMPS_NONE;
    }
    return ctx->result_kind;
}

extern "C" int transcribe_n_segments(const struct transcribe_context * ctx) {
    if (ctx == nullptr || !ctx->has_result) return 0;
    return static_cast<int>(ctx->segments.size());
}
extern "C" int transcribe_n_words(const struct transcribe_context * ctx) {
    if (ctx == nullptr || !ctx->has_result) return 0;
    return static_cast<int>(ctx->words.size());
}
extern "C" int transcribe_n_tokens(const struct transcribe_context * ctx) {
    if (ctx == nullptr || !ctx->has_result) return 0;
    return static_cast<int>(ctx->tokens.size());
}

// ---------------------------------------------------------------------------
// Result accessors - segment level
// ---------------------------------------------------------------------------

namespace {
inline bool seg_in_range(const transcribe_context * ctx, int i) {
    return ctx != nullptr && ctx->has_result && i >= 0 &&
           static_cast<size_t>(i) < ctx->segments.size();
}
inline bool word_in_range(const transcribe_context * ctx, int i) {
    return ctx != nullptr && ctx->has_result && i >= 0 &&
           static_cast<size_t>(i) < ctx->words.size();
}
inline bool token_in_range(const transcribe_context * ctx, int i) {
    return ctx != nullptr && ctx->has_result && i >= 0 &&
           static_cast<size_t>(i) < ctx->tokens.size();
}
} // namespace

extern "C" const char * transcribe_segment_text(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return "";
    return ctx->segments[static_cast<size_t>(i)].text.c_str();
}
extern "C" int64_t transcribe_segment_t0_ms(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return 0;
    return ctx->segments[static_cast<size_t>(i)].t0_ms;
}
extern "C" int64_t transcribe_segment_t1_ms(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return 0;
    return ctx->segments[static_cast<size_t>(i)].t1_ms;
}
extern "C" int transcribe_segment_first_word(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return 0;
    return ctx->segments[static_cast<size_t>(i)].first_word;
}
extern "C" int transcribe_segment_n_words(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return 0;
    return ctx->segments[static_cast<size_t>(i)].n_words;
}
extern "C" int transcribe_segment_first_token(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return 0;
    return ctx->segments[static_cast<size_t>(i)].first_token;
}
extern "C" int transcribe_segment_n_tokens(const struct transcribe_context * ctx, int i) {
    if (!seg_in_range(ctx, i)) return 0;
    return ctx->segments[static_cast<size_t>(i)].n_tokens;
}

// ---------------------------------------------------------------------------
// Result accessors - word level
// ---------------------------------------------------------------------------

extern "C" const char * transcribe_word_text(const struct transcribe_context * ctx, int i) {
    if (!word_in_range(ctx, i)) return "";
    return ctx->words[static_cast<size_t>(i)].text.c_str();
}
extern "C" int64_t transcribe_word_t0_ms(const struct transcribe_context * ctx, int i) {
    if (!word_in_range(ctx, i)) return 0;
    return ctx->words[static_cast<size_t>(i)].t0_ms;
}
extern "C" int64_t transcribe_word_t1_ms(const struct transcribe_context * ctx, int i) {
    if (!word_in_range(ctx, i)) return 0;
    return ctx->words[static_cast<size_t>(i)].t1_ms;
}
extern "C" int transcribe_word_seg_index(const struct transcribe_context * ctx, int i) {
    if (!word_in_range(ctx, i)) return 0;
    return ctx->words[static_cast<size_t>(i)].seg_index;
}
extern "C" int transcribe_word_first_token(const struct transcribe_context * ctx, int i) {
    if (!word_in_range(ctx, i)) return 0;
    return ctx->words[static_cast<size_t>(i)].first_token;
}
extern "C" int transcribe_word_n_tokens(const struct transcribe_context * ctx, int i) {
    if (!word_in_range(ctx, i)) return 0;
    return ctx->words[static_cast<size_t>(i)].n_tokens;
}

// ---------------------------------------------------------------------------
// Result accessors - token level
// ---------------------------------------------------------------------------

extern "C" int transcribe_token_id(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return 0;
    return ctx->tokens[static_cast<size_t>(i)].id;
}
extern "C" const char * transcribe_token_text(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return "";
    return ctx->tokens[static_cast<size_t>(i)].text.c_str();
}
extern "C" float transcribe_token_p(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return NAN;
    return ctx->tokens[static_cast<size_t>(i)].p;
}
extern "C" int64_t transcribe_token_t0_ms(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return 0;
    return ctx->tokens[static_cast<size_t>(i)].t0_ms;
}
extern "C" int64_t transcribe_token_t1_ms(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return 0;
    return ctx->tokens[static_cast<size_t>(i)].t1_ms;
}
extern "C" int transcribe_token_seg_index(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return 0;
    return ctx->tokens[static_cast<size_t>(i)].seg_index;
}
extern "C" int transcribe_token_word_index(const struct transcribe_context * ctx, int i) {
    if (!token_in_range(ctx, i)) return 0;
    return ctx->tokens[static_cast<size_t>(i)].word_index;
}
