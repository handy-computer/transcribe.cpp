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

#include <atomic>
#include <cmath>
#include <cstddef>

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
        default:                                  return "unknown status";
    }
}

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
    p.use_gpu    = true;
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
    p.timestamps         = TRANSCRIBE_TIMESTAMPS_AUTO;
    p.language           = nullptr;
    p.target_language    = nullptr;
    p.strip_special_tags = true;
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

extern "C" transcribe_status transcribe_run(
    struct transcribe_context *      ctx,
    const float *                    pcm,
    int                              n_samples,
    const struct transcribe_params * params)
{
    if (ctx == nullptr || pcm == nullptr || params == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (n_samples < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ctx->model == nullptr || ctx->model->arch == nullptr ||
        ctx->model->arch->run == nullptr)
    {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
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
