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

// Shared run-params validation used by transcribe_run and
// transcribe_stream_begin. Both call sites have already validated
// that ctx, ctx->model, and params are non-null. The translate-task
// rejection differs between the two (run mirrors supports_translate,
// streaming-begin rejects unconditionally in v1), so each caller
// applies its own translate check before reaching this helper.
transcribe_status validate_run_params_common(
    const transcribe_context * ctx,
    const transcribe_params *  params)
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
        const int max_rank = timestamp_rank(ctx->model->caps.max_timestamp_kind);
        if (req_rank > max_rank) {
            return TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS;
        }
    }
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

extern "C" struct transcribe_stream_params
transcribe_stream_default_params(void)
{
    struct transcribe_stream_params p = {};
    p.parakeet = nullptr;
    return p;
}

extern "C" struct transcribe_parakeet_stream_params
transcribe_parakeet_stream_default_params(void)
{
    struct transcribe_parakeet_stream_params p = {};
    p.att_context_right = -1; // use model default
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
    struct transcribe_context *              ctx,
    const struct transcribe_params *         run_params,
    const struct transcribe_stream_params *  stream_params)
{
    if (ctx == nullptr || run_params == nullptr || stream_params == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ctx->stream_state == TRANSCRIBE_STREAM_ACTIVE) {
        // Caller must finalize or reset before starting a new stream.
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ctx->model == nullptr || ctx->model->arch == nullptr) {
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
    if (!ctx->model->caps.supports_streaming                ||
        ctx->model->arch->stream_begin    == nullptr        ||
        ctx->model->arch->stream_feed     == nullptr        ||
        ctx->model->arch->stream_finalize == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    if (const transcribe_status st = validate_run_params_common(ctx, run_params);
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

    // All checks pass — clear the previous snapshot and hand off.
    ctx->clear_result();
    ctx->t_mel_us    = 0;
    ctx->t_encode_us = 0;
    ctx->t_decode_us = 0;
    ctx->was_aborted = false;
    ctx->stream_state = TRANSCRIBE_STREAM_ACTIVE;

    const transcribe_status st = ctx->model->arch->stream_begin(
        ctx, run_params, stream_params);
    if (st != TRANSCRIBE_OK) {
        // Family hook rejected the begin (config it does not understand,
        // memory allocation failure, etc.). Roll lifecycle back to
        // FAILED so the caller can read last_status; leave the cleared
        // result snapshot in place.
        ctx->stream_state      = TRANSCRIBE_STREAM_FAILED;
        ctx->stream_last_status = st;
    }
    return st;
}

extern "C" transcribe_status transcribe_stream_feed(
    struct transcribe_context *        ctx,
    const float *                      pcm,
    int                                n_samples,
    struct transcribe_stream_update *  update)
{
    if (update != nullptr) {
        *update = transcribe_stream_update{};
    }
    if (ctx == nullptr || pcm == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Zero-length feed is rejected: feed exists to consume audio,
    // and callers that just want to inspect state use the stream
    // accessors. n_samples < 0 is plain garbage.
    if (n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ctx->stream_state != TRANSCRIBE_STREAM_ACTIVE) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // begin already confirmed model/arch/hook; we re-check defensively
    // so a malformed context (never happens in practice from a real
    // begin) does not deref a null function pointer.
    if (ctx->model == nullptr || ctx->model->arch == nullptr ||
        ctx->model->arch->stream_feed == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    const transcribe_status st = ctx->model->arch->stream_feed(
        ctx, pcm, n_samples, update);
    if (st != TRANSCRIBE_OK) {
        ctx->stream_state       = TRANSCRIBE_STREAM_FAILED;
        ctx->stream_last_status = st;
    }
    return st;
}

extern "C" transcribe_status transcribe_stream_finalize(
    struct transcribe_context *        ctx,
    struct transcribe_stream_update *  update)
{
    if (update != nullptr) {
        *update = transcribe_stream_update{};
    }
    if (ctx == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ctx->stream_state != TRANSCRIBE_STREAM_ACTIVE) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ctx->model == nullptr || ctx->model->arch == nullptr ||
        ctx->model->arch->stream_finalize == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    const transcribe_status st = ctx->model->arch->stream_finalize(ctx, update);
    if (update != nullptr) {
        // Force is_final true regardless of family hook return; the
        // marker describes the call site, not the result. Set after
        // the hook so the family cannot accidentally clear it.
        update->is_final = true;
    }
    if (st != TRANSCRIBE_OK) {
        ctx->stream_state       = TRANSCRIBE_STREAM_FAILED;
        ctx->stream_last_status = st;
        return st;
    }
    ctx->stream_state = TRANSCRIBE_STREAM_FINISHED;
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_stream_reset(struct transcribe_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    // Family hook releases per-utterance state and clears any buffered
    // audio contents while keeping the allocations. A family without
    // streaming just has no hook installed; reset becomes a pure
    // dispatcher state wipe.
    if (ctx->model != nullptr && ctx->model->arch != nullptr &&
        ctx->model->arch->stream_reset != nullptr)
    {
        ctx->model->arch->stream_reset(ctx);
    }
    ctx->clear_result();
    ctx->stream_state = TRANSCRIBE_STREAM_IDLE;
    // was_aborted is per-stream; reset re-arms it the same way begin
    // does so a caller that resets after an abort starts clean.
    ctx->was_aborted  = false;
}

extern "C" enum transcribe_stream_state
transcribe_stream_get_state(const struct transcribe_context * ctx)
{
    if (ctx == nullptr) {
        return TRANSCRIBE_STREAM_IDLE;
    }
    return ctx->stream_state;
}

extern "C" int transcribe_stream_revision(const struct transcribe_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    return ctx->stream_revision;
}

extern "C" int transcribe_stream_n_committed_segments(
    const struct transcribe_context * ctx)
{
    if (ctx == nullptr) return 0;
    return ctx->n_committed_segments;
}

extern "C" int transcribe_stream_n_committed_words(
    const struct transcribe_context * ctx)
{
    if (ctx == nullptr) return 0;
    return ctx->n_committed_words;
}

extern "C" int transcribe_stream_n_committed_tokens(
    const struct transcribe_context * ctx)
{
    if (ctx == nullptr) return 0;
    return ctx->n_committed_tokens;
}

extern "C" transcribe_status transcribe_stream_last_status(
    const struct transcribe_context * ctx)
{
    if (ctx == nullptr) {
        return TRANSCRIBE_OK;
    }
    return ctx->stream_last_status;
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
    // A run cannot replace an active stream's results — that would
    // strand the in-flight stream's per-family state. Caller must
    // finalize or reset first. FINISHED and FAILED both fall through;
    // the run path below resets stream_state to IDLE.
    if (ctx->stream_state == TRANSCRIBE_STREAM_ACTIVE) {
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
    // Force stream_state to IDLE: clear_result deliberately preserves
    // lifecycle state, but a well-formed transcribe_run subsumes any
    // prior FINISHED/FAILED stream — after a one-shot run the context
    // is no longer meaningfully in a streaming lifecycle.
    ctx->stream_state = TRANSCRIBE_STREAM_IDLE;

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
    if (const transcribe_status st = validate_run_params_common(ctx, params);
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
        !ctx->model->caps.supports_translate)
    {
        return TRANSCRIBE_ERR_UNSUPPORTED_TASK;
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
