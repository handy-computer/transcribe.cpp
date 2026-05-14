// arch/whisper/model.cpp - Whisper ASR family handler.
//
// Stage-4 Whisper runtime. Numerical validation can still inject the
// reference mel tensor with TRANSCRIBE_WHISPER_MEL_FROM_REF to isolate
// encoder/decoder drift, but normal inference uses the C++ mel frontend
// and the autoregressive decoder path below.

#include "whisper.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::whisper {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   WhisperModel>);
static_assert(std::is_base_of_v<transcribe_context, WhisperContext>);

WhisperModel::~WhisperModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin();
         it != plan.scheduler_list.rend(); ++it)
    {
        ggml_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

WhisperContext::~WhisperContext() {
    kv_cache.free();
    enc_out.free();
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    compute_ctx_size = 0;
}

bool enc_out_init(WhisperEncOut & enc_out,
                  ggml_backend_t  backend,
                  int             d_model,
                  int             T_enc)
{
    enc_out.free();

    const size_t ctx_size = ggml_tensor_overhead() + 256;
    ggml_init_params params {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    enc_out.ctx = ggml_init(params);
    if (enc_out.ctx == nullptr) {
        std::fprintf(stderr, "whisper enc_out: ggml_init failed\n");
        return false;
    }

    enc_out.tensor = ggml_new_tensor_2d(enc_out.ctx, GGML_TYPE_F32,
                                        d_model, T_enc);
    ggml_set_name(enc_out.tensor, "enc_out");

    enc_out.buffer = ggml_backend_alloc_ctx_tensors(enc_out.ctx, backend);
    if (enc_out.buffer == nullptr) {
        std::fprintf(stderr, "whisper enc_out: buffer alloc failed\n");
        ggml_free(enc_out.ctx);
        enc_out.ctx    = nullptr;
        enc_out.tensor = nullptr;
        return false;
    }

    enc_out.d_model = d_model;
    enc_out.T_enc   = T_enc;
    return true;
}

int kv_pad_self_attn(transcribe::BackendKind kind, bool use_flash) {
    if (!use_flash) return 1;
    switch (kind) {
        // Match whisper.cpp's whisper_kv_cache_get_padding (Metal+FA).
        case transcribe::BackendKind::Metal: return 32;
        // CUDA uses 256 in whisper.cpp; not exercised here yet.
        default: return 1;
    }
}

bool kv_cache_init(WhisperKvCache & cache,
                   ggml_backend_t   backend,
                   int              n_ctx,
                   int              T_enc,
                   int              d_model,
                   int              n_layer,
                   ggml_type        kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
                     "whisper kv_cache: unsupported kv_type=%d "
                     "(only F16/F32)\n", static_cast<int>(kv_type));
        return false;
    }

    const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;
    ggml_init_params params {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "whisper kv_cache: ggml_init failed\n");
        return false;
    }

    // Cross K/V allocated at GGML_PAD(T_enc, 256) rows per layer so
    // the FA op consumes a sequence dim that is a multiple of the
    // Metal kernel's block size. Only the first T_enc rows are
    // written by build_cross_kv_graph; the trailing rows stay zero
    // (buffer_clear below). With K=V=0 in the padded slots, the
    // unmasked FA cross-attn output picks up a small dilution
    // factor — whisper.cpp ships with this trade-off.
    const int     T_enc_pad     = static_cast<int>(GGML_PAD(T_enc, k_cross_kv_pad));
    const int64_t self_elements = static_cast<int64_t>(d_model) * n_layer * n_ctx;
    const int64_t cross_elements = static_cast<int64_t>(d_model) * n_layer * T_enc_pad;

    cache.self_k  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.self_v  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

    ggml_set_name(cache.self_k,  "kv_self_k");
    ggml_set_name(cache.self_v,  "kv_self_v");
    ggml_set_name(cache.cross_k, "kv_cross_k");
    ggml_set_name(cache.cross_v, "kv_cross_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        std::fprintf(stderr, "whisper kv_cache: buffer alloc failed\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx     = n_ctx;
    cache.T_enc     = T_enc;
    cache.T_enc_pad = T_enc_pad;
    cache.n         = 0;
    cache.head      = 0;
    cache.cross_populated = false;

    return true;
}

namespace {

constexpr const char k_default_variant[] = "whisper";

// Prepare cc->compute_ctx for a graph build of capacity at most
// `mem`. If a context already exists and is at least that large,
// ggml_reset clears the tensor metadata in-place (cheap). Only when
// the existing context is too small do we ggml_free + ggml_init.
//
// Replaces the previous "free + init per graph build" pattern that
// allocated a fresh compute context for the encoder, cross-KV graph,
// every tier's prompt prefill, and every step in the generation
// loop. The free + malloc churn was visible in the per-step build
// timer (~30 us per step on tiny F16) plus allocator pressure on
// very long inputs.
bool ensure_compute_ctx(WhisperContext * cc, size_t mem) {
    if (cc->compute_ctx != nullptr) {
        if (cc->compute_ctx_size >= mem) {
            ggml_reset(cc->compute_ctx);
            return true;
        }
        ggml_free(cc->compute_ctx);
        cc->compute_ctx      = nullptr;
        cc->compute_ctx_size = 0;
    }

    ggml_init_params p {};
    p.mem_size   = mem;
    p.mem_buffer = nullptr;
    p.no_alloc   = true;
    cc->compute_ctx = ggml_init(p);
    if (cc->compute_ctx == nullptr) {
        return false;
    }
    cc->compute_ctx_size = mem;
    return true;
}

// Print the per-stage performance summary collected during the most
// recent whisper_run. Opt-in via TRANSCRIBE_WHISPER_PROFILE=1 — gated
// by the caller, this helper just formats the counters. Output is one
// summary line per stage (encoder / cross / prompt / step) plus a
// per-step average for the steady-state generation loop, which is the
// metric most directly comparable to whisper.cpp's bench numbers.
void print_whisper_perf(const WhisperPerf & p) {
    auto avg_us = [](const WhisperPerfStage & s) -> double {
        return s.count > 0
                   ? static_cast<double>(s.total_us) /
                         static_cast<double>(s.count)
                   : 0.0;
    };
    auto ms = [](int64_t us) -> double {
        return static_cast<double>(us) / 1000.0;
    };
    auto stage_total = [&](const WhisperPerfStage & s) {
        return s.total_us;
    };

    const int64_t enc_total =
        stage_total(p.enc_build) + stage_total(p.enc_alloc) +
        stage_total(p.enc_compute) + stage_total(p.enc_tensor_get);
    const int64_t cross_total =
        stage_total(p.cross_build) + stage_total(p.cross_alloc) +
        stage_total(p.cross_compute);
    const int64_t prompt_total =
        stage_total(p.prompt_build) + stage_total(p.prompt_alloc) +
        stage_total(p.prompt_compute) + stage_total(p.prompt_tensor_get) +
        stage_total(p.prompt_cpu);
    const int64_t step_total =
        stage_total(p.step_build) + stage_total(p.step_alloc) +
        stage_total(p.step_compute) + stage_total(p.step_tensor_get) +
        stage_total(p.step_cpu);

    std::fprintf(stderr,
        "[whisper-perf] chunks=%d  encs=%d  crosses=%d  prompts=%d  steps=%d\n",
        p.chunks, p.enc_compute.count, p.cross_compute.count,
        p.prompt_compute.count, p.step_compute.count);
    std::fprintf(stderr,
        "[whisper-perf] enc    total=%7.2f ms  build=%7.2f  alloc=%7.2f  "
        "compute=%7.2f  tget=%7.2f\n",
        ms(enc_total), ms(p.enc_build.total_us), ms(p.enc_alloc.total_us),
        ms(p.enc_compute.total_us), ms(p.enc_tensor_get.total_us));
    std::fprintf(stderr,
        "[whisper-perf] cross  total=%7.2f ms  build=%7.2f  alloc=%7.2f  "
        "compute=%7.2f\n",
        ms(cross_total), ms(p.cross_build.total_us),
        ms(p.cross_alloc.total_us), ms(p.cross_compute.total_us));
    std::fprintf(stderr,
        "[whisper-perf] prompt total=%7.2f ms  build=%7.2f  alloc=%7.2f  "
        "compute=%7.2f  tget=%7.2f  cpu=%7.2f\n",
        ms(prompt_total), ms(p.prompt_build.total_us),
        ms(p.prompt_alloc.total_us), ms(p.prompt_compute.total_us),
        ms(p.prompt_tensor_get.total_us), ms(p.prompt_cpu.total_us));
    std::fprintf(stderr,
        "[whisper-perf] step   total=%7.2f ms  build=%7.2f  alloc=%7.2f  "
        "compute=%7.2f  tget=%7.2f  cpu=%7.2f\n",
        ms(step_total), ms(p.step_build.total_us),
        ms(p.step_alloc.total_us), ms(p.step_compute.total_us),
        ms(p.step_tensor_get.total_us), ms(p.step_cpu.total_us));
    std::fprintf(stderr,
        "[whisper-perf] step avg us  build=%6.1f  alloc=%6.1f  "
        "compute=%6.1f  tget=%6.1f  cpu=%6.1f\n",
        avg_us(p.step_build), avg_us(p.step_alloc),
        avg_us(p.step_compute), avg_us(p.step_tensor_get),
        avg_us(p.step_cpu));

    // CPU sub-section breakdown — opt-in via TRANSCRIBE_WHISPER_PROFILE
    // values that contain "cpu" or "all". Keeps the default profile
    // output unchanged for existing benchmarks while still surfacing
    // suppress/timestamp/sample/logprob splits when needed.
    const char * profile_env = std::getenv("TRANSCRIBE_WHISPER_PROFILE");
    bool show_cpu_breakdown = false;
    if (profile_env != nullptr) {
        const std::string v = profile_env;
        if (v.find("cpu") != std::string::npos ||
            v.find("all") != std::string::npos) {
            show_cpu_breakdown = true;
        }
    }
    if (show_cpu_breakdown) {
        std::fprintf(stderr,
            "[whisper-perf] prompt cpu  suppress=%7.2f  ts=%7.2f  "
            "sample=%7.2f  logprob=%7.2f  (ms)\n",
            ms(p.prompt_cpu_suppress.total_us),
            ms(p.prompt_cpu_timestamp.total_us),
            ms(p.prompt_cpu_sample.total_us),
            ms(p.prompt_cpu_logprob.total_us));
        std::fprintf(stderr,
            "[whisper-perf] step   cpu  suppress=%7.2f  ts=%7.2f  "
            "sample=%7.2f  logprob=%7.2f  (ms)\n",
            ms(p.step_cpu_suppress.total_us),
            ms(p.step_cpu_timestamp.total_us),
            ms(p.step_cpu_sample.total_us),
            ms(p.step_cpu_logprob.total_us));
        std::fprintf(stderr,
            "[whisper-perf] step avg us  suppress=%6.1f  ts=%6.1f  "
            "sample=%6.1f  logprob=%6.1f\n",
            avg_us(p.step_cpu_suppress), avg_us(p.step_cpu_timestamp),
            avg_us(p.step_cpu_sample),   avg_us(p.step_cpu_logprob));
    }
}

bool whisper_perf_enabled() {
    const char * s = std::getenv("TRANSCRIBE_WHISPER_PROFILE");
    return s != nullptr && s[0] != '\0' && s[0] != '0';
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

transcribe_status whisper_load(
    Loader &                          loader,
    const transcribe_model_params *   params,
    transcribe_model **               out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<WhisperModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    // Hparams first: the frontend / encoder config and the decoder
    // vocab size come from here.
    if (const transcribe_status st = read_whisper_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Tokenizer. Whisper GGUFs carry a "gpt2"-family tokenizer (HF
    // byte-level BPE). Vocab_size stays whatever the GGUF declared —
    // DO NOT overwrite from the tokenizer. Whisper's decoder vocab
    // (51865 for multilingual) is larger than tokenizer.tokens
    // (50258) because language / timestamp / task special tokens are
    // emitted by the model but not listed in the tokens array. That
    // gap is by design for this family.
    if (const transcribe_status st = m->tok.load(loader.gguf());
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    // Whisper uses the canonical GPT-2 pretokenizer regex (digit runs
    // merged, lowercase-only contractions). Force it here so older
    // converter outputs without tokenizer.ggml.pre still tokenize
    // text correctly — the absent-key default "qwen2" would split
    // multi-digit runs and diverge from the HF reference.
    if (m->tok.pretokenizer() != "gpt2") {
        m->tok.set_pretokenizer("gpt2");
    }

    // Mel frontend. Whisper-style 80-bin log-mel at 16 kHz, with
    // per-utterance log10 + clamp(max-8) + (+4)/4 normalization,
    // periodic Hann window, and reflect padding around the centered
    // STFT. Filterbank + window come baked in the GGUF (the converter
    // serializes them verbatim from preprocessor_config.json) so the
    // C++ frontend matches the WhisperFeatureExtractor reference
    // bit-for-bit modulo fp32 reduction-order drift.
    {
        // Filterbank + window from the GGUF (frontend.mel_filterbank,
        // frontend.window). Both are F32 arrays whose shapes match
        // what MelConfig wants (row-major [num_mels, n_fft/2+1] and
        // [n_fft] respectively). When present, MelFrontend uses them
        // verbatim instead of reconstructing from cfg constants.
        using R = transcribe::load_common::ReadF32Result;
        std::vector<float> fb_buf;
        std::vector<float> win_buf;

        const size_t fb_elems =
            static_cast<size_t>(m->hparams.fe_num_mels) *
            static_cast<size_t>(m->hparams.fe_n_fft / 2 + 1);
        const auto fb_rc = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(),
            "frontend.mel_filterbank", fb_elems,
            "whisper", fb_buf);
        if (fb_rc != R::Ok && fb_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t win_elems = static_cast<size_t>(m->hparams.fe_n_fft);
        const auto win_rc = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(),
            "frontend.window", win_elems,
            "whisper", win_buf);
        if (win_rc != R::Ok && win_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }

        if (const transcribe_status st = install_mel_from_buffers(
                m->hparams, std::move(fb_buf), std::move(win_buf), m->mel);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
    }

    // Stage 2: reopen with no_alloc to build the tensor catalog.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_whisper_weights(
            m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // Backend plan.
    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "whisper", m->plan);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    m->backend = ggml_backend_name(m->plan.primary);

    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        std::fprintf(stderr,
                     "whisper: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "whisper");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // Capabilities: apply family invariants, then let the GGUF KV
    // override, then install the language list from general.languages.
    apply_family_invariants(m->caps);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Resolve per-language token ids. The decoder's forced-language
    // step needs <|{code}|> id per supported language. We hydrate the
    // mapping here so a vocab drift surfaces at load, not mid-run.
    //
    // We also keep our own owned copy of the language list so callers
    // that go through m->lang_codes don't depend on the capabilities
    // storage lifetime (set_languages owns the string backing for
    // caps.languages[], but accessing it requires walking the capability
    // chain — a direct owned vector here is simpler for decoder code).
    //
    // Non-multilingual (.en) models advertise a single-language list
    // ["en"] but their vocab does NOT contain a "<|en|>" token: the
    // .en decoder prefix is just <|sot|>, with no language or task
    // tokens. So for .en we keep lang_codes (so the run path can
    // accept params.language == "en") but leave lang_token_ids empty.
    // The run path branches on supports_language_detect to skip the
    // <|lang|>/<|task|> emission for .en.
    m->lang_codes.clear();
    m->lang_token_ids.clear();
    const bool is_multilingual = m->caps.supports_language_detect;
    if (m->caps.languages != nullptr && m->caps.n_languages > 0) {
        m->lang_codes.reserve(static_cast<size_t>(m->caps.n_languages));
        if (is_multilingual) {
            m->lang_token_ids.reserve(static_cast<size_t>(m->caps.n_languages));
        }
        for (int i = 0; i < m->caps.n_languages; ++i) {
            const char * code = m->caps.languages[i];
            if (code == nullptr || code[0] == '\0') continue;
            m->lang_codes.emplace_back(code);
            if (!is_multilingual) {
                continue;
            }
            const std::string piece = std::string("<|") + code + "|>";
            const int id = m->tok.find(piece);
            if (id < 0) {
                std::fprintf(stderr,
                             "whisper: language '%s' has no '%s' token "
                             "in tokenizer vocab\n",
                             code, piece.c_str());
                return TRANSCRIBE_ERR_GGUF;
            }
            m->lang_token_ids.push_back(static_cast<int32_t>(id));
        }
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// init_context
// ---------------------------------------------------------------------------

transcribe_status whisper_init_context(
    transcribe_model *                model,
    const transcribe_context_params * params,
    transcribe_context **             out_ctx)
{
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc = std::make_unique<WhisperContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    // Whisper defaults: both encoder and decoder flash-on. Head dim for
    // whisper-tiny is 64 (d_model=384, 6 heads); supported on every
    // backend we ship. See WhisperContext for the rationale.
    cc->encoder_use_flash = true;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(
        cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// run

namespace {

// Run the encoder on one mel window and leave the output in the
// backend-resident persistent tensor cc->enc_out.tensor (d_enc * T_enc
// floats, F32). Downstream graphs (cross-KV precompute, KV-cached
// decoder) read from that tensor directly via ggml views — no host
// roundtrip. cc->enc_host is only populated on demand for language
// detection's prefill graph, which still needs a host-side
// encoder_out_in input. Also publishes T_enc on the context. The
// compute_ctx is reset inside this helper; the scheduler is created
// lazily on first call.
//
// mel_data is in encoder layout [n_mels, n_mel_frames] with n_mels
// innermost, matching the ggml ne=[n_mels, n_mel_frames] input tensor.
// For the shipped Whisper variants n_mel_frames is 3000; long-form
// windows shorter than 3000 must be zero-padded on the right before
// calling this helper.
//
// allow_dumps controls whether encoder intermediates are emitted via
// transcribe::debug. Set true for validate.py runs (one chunk only in
// short-form) and for the first chunk in long-form; false otherwise
// so the long-form loop does not overwrite dumps from the first chunk.
transcribe_status run_whisper_encoder_on_window(
    WhisperContext * cc,
    WhisperModel *   cm,
    const float *    mel_data,
    int              n_mels,
    int              n_mel_frames,
    bool             allow_dumps,
    int &            out_T_enc)
{
    // Reset per-call compute state. ensure_compute_ctx reuses the
    // context across encoder / cross-KV / prompt / step builds —
    // ggml_reset between calls instead of free + reinit.
    const int64_t t_enc_build_start = ggml_time_us();
    if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        std::fprintf(stderr,
                     "whisper run: ensure_compute_ctx (encoder) failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, cm->hparams, n_mel_frames,
        cc->encoder_use_flash, cm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Backend-resident encoder output. Allocate the persistent
    // F32 tensor once (or re-allocate if T_enc / d_model changed).
    // After build_encoder_graph, append a final ggml_cpy from eb.out
    // into a view of cc->enc_out.tensor — both live on the primary
    // backend, so this is a single intra-backend memcpy that stays
    // off the host. Subsequent graphs (cross-KV, lang-det) read
    // from cc->enc_out.tensor directly.
    {
        const int d_enc_g = static_cast<int>(eb.out->ne[0]);
        const int T_enc_g = static_cast<int>(eb.out->ne[1]);
        if (cc->enc_out.tensor == nullptr ||
            cc->enc_out.d_model != d_enc_g ||
            cc->enc_out.T_enc   != T_enc_g)
        {
            if (!enc_out_init(cc->enc_out, cm->plan.primary,
                              d_enc_g, T_enc_g))
            {
                std::fprintf(stderr,
                             "whisper run: enc_out_init failed\n");
                return TRANSCRIBE_ERR_GGUF;
            }
        }
        ggml_tensor * enc_out_view = ggml_view_2d(
            cc->compute_ctx, cc->enc_out.tensor,
            d_enc_g, T_enc_g,
            cc->enc_out.tensor->nb[1],
            0);
        ggml_build_forward_expand(
            eb.graph, ggml_cpy(cc->compute_ctx, eb.out, enc_out_view));
    }
    cc->perf.enc_build.add(ggml_time_us() - t_enc_build_start);

    // Allocate + compute encoder graph.
    const int64_t t_enc_alloc_start = ggml_time_us();
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr,
                         "whisper run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "whisper run: ggml_backend_sched_alloc_graph failed "
                     "(encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload mel.
    const size_t mel_bytes = static_cast<size_t>(n_mels) *
                             static_cast<size_t>(n_mel_frames) *
                             sizeof(float);
    ggml_backend_tensor_set(eb.mel_in, mel_data, 0, mel_bytes);
    cc->perf.enc_alloc.add(ggml_time_us() - t_enc_alloc_start);

    // Compute.
    const int64_t t_enc_compute_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "whisper run: encoder graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->perf.enc_compute.add(ggml_time_us() - t_enc_compute_start);

    // Dumps. Only the first (or only) chunk emits intermediates so
    // validate.py against a reference dump directory sees stable output.
    if (allow_dumps) {
        auto try_dump = [](const char * name, ggml_tensor * t,
                           const char * stage)
        {
            if (t != nullptr) {
                transcribe::debug::dump_tensor(name, t, stage);
            }
        };
        try_dump("enc.mel.in",    eb.dumps.mel_in,    "encoder.mel");
        try_dump("enc.conv1.out", eb.dumps.conv1_out, "encoder.conv1");
        try_dump("enc.conv2.out", eb.dumps.conv2_out, "encoder.conv2");
        try_dump("enc.pos_emb",   eb.dumps.pos_emb,   "encoder.pos_emb");
        try_dump("enc.embed.out", eb.dumps.embed_out, "encoder.embed");
        for (size_t i = 0; i < eb.dumps.block_outs.size(); ++i) {
            char bname[64], stage[64];
            std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
            std::snprintf(stage, sizeof(stage), "encoder.block%zu.out", i);
            try_dump(bname, eb.dumps.block_outs[i], stage);
        }
        try_dump("enc.final", eb.dumps.final_out, "encoder.final");
    }

    // The persistent enc_out tensor is now populated. Cross-KV
    // reads from it directly — no host materialization needed for
    // the hot path. Lang-detect (one-shot, first chunk) still uses
    // a tensor-set into a graph input; we lazily materialize to
    // enc_host there (see chunk loop below).
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    out_T_enc       = static_cast<int>(eb.out->ne[1]);
    cc->enc_T       = out_T_enc;
    (void)d_enc;
    return TRANSCRIBE_OK;
}

// HF _retrieve_compression_ratio (generation_whisper.py:1948-1954).
// Packs each token id as little-endian with width = floor(log2(V)/8) + 1
// bytes, zlib-compresses, returns len(raw)/len(compressed). For Whisper's
// vocab (≤ 65536) the width is 2. ratio > threshold ⇒ decoder is
// repeating token ids, triggers temperature escalation.
//
// Zero-length input returns 0 (no raw bytes to score).
float compute_compression_ratio_hf(
    const std::vector<int32_t> & tokens,
    int64_t                      vocab_size)
{
    if (tokens.empty()) return 0.0f;
    int bytes_per_token = 1;
    if (vocab_size > 1) {
        const double lg2 = std::log2(static_cast<double>(vocab_size));
        bytes_per_token = static_cast<int>(std::floor(lg2 / 8.0)) + 1;
    }

    std::vector<uint8_t> raw(tokens.size() *
                             static_cast<size_t>(bytes_per_token));
    for (size_t i = 0; i < tokens.size(); ++i) {
        uint64_t v = static_cast<uint64_t>(
            static_cast<uint32_t>(tokens[i]));  // Whisper ids fit in u32
        for (int b = 0; b < bytes_per_token; ++b) {
            raw[i * bytes_per_token + b] =
                static_cast<uint8_t>((v >> (8 * b)) & 0xFF);
        }
    }

    uLongf dest_len = compressBound(static_cast<uLong>(raw.size()));
    std::vector<Bytef> compressed(dest_len);
    const int rc = compress(compressed.data(), &dest_len,
                            raw.data(),
                            static_cast<uLong>(raw.size()));
    if (rc != Z_OK || dest_len == 0) {
        // Degenerate: treat as unratio-able so thresholds pass.
        return 0.0f;
    }
    return static_cast<float>(raw.size()) /
           static_cast<float>(dest_len);
}

// Sample from a distribution defined by last_logits at temperature T.
// T == 0 ⇒ argmax (deterministic). T > 0 ⇒ multinomial sample over
// softmax(logits / T). Returns the sampled token id.
//
// Matches HF's sampling semantics (probability ∝ exp(logit/T)) in the
// numerically stable form with max-subtracted exponentials.
//
// -INFINITY logits (from suppress/timestamp rules) contribute 0 mass.
//
// `scratch` is reused across calls to avoid the per-step double[vocab]
// alloc on the T>0 hot path. Sized lazily; iteration order matches the
// previous code so the multinomial draw is bit-identical.
int sample_from_logits(
    const std::vector<float> & logits,
    float                      temperature,
    std::mt19937 &             rng,
    std::vector<double> *      scratch = nullptr)
{
    const int n = static_cast<int>(logits.size());
    if (temperature <= 0.0f) {
        int best_id = 0;
        float best  = logits[0];
        for (int i = 1; i < n; ++i) {
            if (logits[i] > best) { best = logits[i]; best_id = i; }
        }
        return best_id;
    }

    // Multinomial sampling.
    // Stabilise: subtract max finite value before exp.
    float max_l = -INFINITY;
    for (int i = 0; i < n; ++i) {
        const float l = logits[i];
        if (std::isfinite(l) && l > max_l) max_l = l;
    }
    if (!std::isfinite(max_l)) {
        // All -INF (shouldn't happen); fall back to argmax.
        return 0;
    }

    std::vector<double>   local_probs;
    std::vector<double> & probs = (scratch != nullptr) ? *scratch : local_probs;
    if (probs.size() < static_cast<size_t>(n)) {
        probs.resize(static_cast<size_t>(n));
    }
    // Zero only the prefix we will read in the cumulative pass; the
    // tail past n is irrelevant.
    std::fill_n(probs.begin(), n, 0.0);

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float l = logits[i];
        if (std::isfinite(l)) {
            const double p = std::exp(
                static_cast<double>((l - max_l) / temperature));
            probs[static_cast<size_t>(i)] = p;
            sum += p;
        }
    }
    if (sum <= 0.0) {
        // Defensive: argmax fallback.
        int best_id = 0;
        for (int i = 1; i < n; ++i) {
            if (logits[i] > logits[best_id]) best_id = i;
        }
        return best_id;
    }

    std::uniform_real_distribution<double> u(0.0, sum);
    const double r = u(rng);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += probs[static_cast<size_t>(i)];
        if (r < acc) return i;
    }
    return n - 1;  // numerical edge
}

// Fused argmax + log-softmax for the T == 0 (greedy) path.
//
// Returns the argmax token id; *out_logprob receives logprob_of_token_hf
// of that token at temperature 0 (which uses rescale_T = 1.0). Two
// passes total: pass 1 finds max_l + argmax, pass 2 sums exp.
//
// Numerics: identical to running sample_from_logits(T=0) +
// logprob_of_token_hf(T=0) on the same buffer. -INFINITY logits
// contribute zero mass to the partition; if every entry is -INF
// (shouldn't happen) the function returns id 0 with logprob -inf,
// matching the existing fallback.
int sample_argmax_and_logprob(
    const std::vector<float> & logits,
    float *                    out_logprob)
{
    const int n = static_cast<int>(logits.size());
    int   best_id = 0;
    float best_l  = logits[0];
    float max_l   = std::isfinite(best_l) ? best_l : -INFINITY;
    for (int i = 1; i < n; ++i) {
        const float l = logits[i];
        if (l > best_l) { best_l = l; best_id = i; }
        if (std::isfinite(l) && l > max_l) max_l = l;
    }

    if (out_logprob != nullptr) {
        if (!std::isfinite(max_l)) {
            *out_logprob = -INFINITY;
        } else {
            double sum_exp = 0.0;
            for (int i = 0; i < n; ++i) {
                const float l = logits[i];
                if (std::isfinite(l)) {
                    sum_exp += std::exp(static_cast<double>(l - max_l));
                }
            }
            if (sum_exp <= 0.0) {
                *out_logprob = -INFINITY;
            } else {
                const float log_Z =
                    max_l + static_cast<float>(std::log(sum_exp));
                *out_logprob = best_l - log_Z;
            }
        }
    }
    return best_id;
}

// Compute log_softmax(logits * rescale_T)[token_id] in the numerically
// stable form (subtract max before exp). rescale_T = T if T > 0 else 1
// — matches HF _retrieve_avg_logprobs scaling convention (the logits
// stay the same shape but their relative gaps widen at low T, which is
// the HF semantic choice).
float logprob_of_token_hf(
    const std::vector<float> & logits,
    int                        token_id,
    float                      temperature)
{
    const int n = static_cast<int>(logits.size());
    if (token_id < 0 || token_id >= n) {
        return -INFINITY;
    }
    const float rescale_T = (temperature > 0.0f) ? temperature : 1.0f;
    float max_l = -INFINITY;
    for (int i = 0; i < n; ++i) {
        const float l = logits[i] * rescale_T;
        if (std::isfinite(l) && l > max_l) max_l = l;
    }
    if (!std::isfinite(max_l)) {
        return -INFINITY;
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        const float l = logits[i] * rescale_T;
        if (std::isfinite(l)) {
            sum_exp += std::exp(static_cast<double>(l - max_l));
        }
    }
    if (sum_exp <= 0.0) return -INFINITY;
    const float log_Z = max_l + static_cast<float>(std::log(sum_exp));
    return logits[token_id] * rescale_T - log_Z;
}

// Load 3000 * 80 f32 floats from <dir>/enc.mel.in.f32. Layout on disk
// matches the reference dump: row-major (3000, 80) — exactly what we
// need to upload into the ggml ne=[80, 3000] input tensor via a flat
// memcpy.
transcribe_status load_mel_from_ref(const char *         ref_dir,
                                    int                  n_mels,
                                    int                  n_mel_frames,
                                    std::vector<float> & out)
{
    if (ref_dir == nullptr || ref_dir[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    std::string path = ref_dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "enc.mel.in.f32";

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr,
                     "whisper run: cannot open mel ref '%s'\n", path.c_str());
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }

    const size_t n_elems = static_cast<size_t>(n_mels) *
                           static_cast<size_t>(n_mel_frames);
    const size_t n_bytes = n_elems * sizeof(float);

    out.assign(n_elems, 0.0f);
    f.read(reinterpret_cast<char *>(out.data()),
           static_cast<std::streamsize>(n_bytes));
    if (static_cast<size_t>(f.gcount()) != n_bytes) {
        std::fprintf(stderr,
                     "whisper run: short read on mel ref '%s' "
                     "(got %lld bytes, want %zu)\n",
                     path.c_str(), static_cast<long long>(f.gcount()), n_bytes);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

} // namespace

transcribe_status whisper_run(
    transcribe_context *      ctx,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<WhisperContext *>(ctx);
    auto * cm = static_cast<WhisperModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    transcribe::debug::init();

    // Reset per-stage timing counters; the summary printed at end-of-run
    // is opt-in via TRANSCRIBE_WHISPER_PROFILE=1 but the counters
    // themselves are always populated (the timestamps are negligible).
    cc->perf.reset();

    // ----- Mel frontend ------------------------------------------------
    //
    //   1. TRANSCRIBE_WHISPER_MEL_FROM_REF=<dir>  — debug-only knob.
    //      Reads the reference mel dump from disk so the encoder can be
    //      diffed against the reference WITHOUT introducing C++ mel
    //      drift into the comparison. Use this when a per-tensor
    //      compare regression fires and you want to isolate whether
    //      the drift originates in the C++ mel frontend or downstream
    //      in the encoder/decoder graph. validate.py exposes it via
    //      `--mel-from-ref`; the default validation path exercises the
    //      production C++ MelFrontend below so frontend regressions
    //      (e.g. precision changes that pass tolerance under ref-mel
    //      injection but break WER on borderline utterances) cannot
    //      slip through.
    //
    //   2. C++ MelFrontend (default). Dual behavior:
    //      - Short-form (n_samples <= fe_n_samples): pad PCM to
    //        fe_n_samples (480000) and compute → exactly
    //        fe_nb_max_frames (3000) mel frames. Bit-identical to
    //        pre-Stage-2 behavior; tolerance + smoke tests exercise
    //        this branch.
    //      - Long-form (> fe_n_samples): compute mel on the raw audio
    //        (HF matches this — only short-form pads PCM to 30 s). The
    //        seek loop slices 3000-frame windows from the transposed
    //        buffer and zero-pads the trailing short window.
    const int n_mels                 = cm->hparams.enc_num_mel_bins;
    const int n_mel_frames_per_chunk = cm->hparams.fe_nb_max_frames > 0
                                           ? cm->hparams.fe_nb_max_frames : 3000;
    const int n_samples_per_chunk    = cm->hparams.fe_n_samples > 0
                                           ? cm->hparams.fe_n_samples : 480000;
    // Short-form = fits in a single 30 s window; bypass the dynamic
    // stride + no-speech fallback machinery. This is a numeric AND
    // behavioral parity gate: HF's generate() takes a different code
    // path for is_shortform, and our Stage 1 tolerance tests were all
    // captured in short-form.
    const bool is_short_form = (n_samples <= n_samples_per_chunk);

    const int64_t t_mel_start = ggml_time_us();
    int total_mel_frames = 0;
    if (const char * ref_dir = std::getenv("TRANSCRIBE_WHISPER_MEL_FROM_REF");
        ref_dir != nullptr && ref_dir[0] != '\0')
    {
        if (const transcribe_status st = load_mel_from_ref(
                ref_dir, n_mels, n_mel_frames_per_chunk, cc->mel_buf);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        total_mel_frames = n_mel_frames_per_chunk;
    } else {
        if (!cm->mel.has_value()) {
            std::fprintf(stderr,
                         "whisper run: model has no MelFrontend "
                         "(load skipped?)\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        std::vector<float> pcm_work;
        const float * pcm_in   = pcm;
        size_t        pcm_in_n = static_cast<size_t>(n_samples);
        if (is_short_form) {
            pcm_work.assign(static_cast<size_t>(n_samples_per_chunk), 0.0f);
            const int n_copy = std::min(n_samples, n_samples_per_chunk);
            std::memcpy(pcm_work.data(), pcm,
                        static_cast<size_t>(n_copy) * sizeof(float));
            pcm_in   = pcm_work.data();
            pcm_in_n = pcm_work.size();
        }

        int mel_n_mels = 0, mel_n_frames = 0;
        std::vector<float> mel_mn;  // [n_mels, n_frames] (MelFrontend layout)
        if (const transcribe_status mst = cm->mel->compute(
                pcm_in, pcm_in_n, mel_mn, mel_n_mels, mel_n_frames);
            mst != TRANSCRIBE_OK)
        {
            std::fprintf(stderr,
                         "whisper run: MelFrontend::compute failed (%d)\n",
                         static_cast<int>(mst));
            return mst;
        }
        if (mel_n_mels != n_mels) {
            std::fprintf(stderr,
                         "whisper run: mel n_mels %d != expected %d\n",
                         mel_n_mels, n_mels);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (is_short_form && mel_n_frames != n_mel_frames_per_chunk) {
            std::fprintf(stderr,
                         "whisper run: short-form mel has %d frames, "
                         "expected %d\n",
                         mel_n_frames, n_mel_frames_per_chunk);
            return TRANSCRIBE_ERR_GGUF;
        }

        // Transpose [n_mels, n_frames] → [n_frames, n_mels] so that the
        // slice for chunk k is a contiguous row span
        //   mel_buf[k*3000*n_mels : (k+1)*3000*n_mels).
        cc->mel_buf.resize(static_cast<size_t>(mel_n_mels) *
                           static_cast<size_t>(mel_n_frames));
        for (int t = 0; t < mel_n_frames; ++t) {
            for (int m = 0; m < mel_n_mels; ++m) {
                cc->mel_buf[static_cast<size_t>(t) * mel_n_mels + m] =
                    mel_mn[static_cast<size_t>(m) * mel_n_frames + t];
            }
        }
        total_mel_frames = mel_n_frames;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    if (total_mel_frames <= 0) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Run-scoped constants --------------------------------------
    const int64_t n_ctx_decoder = cm->hparams.dec_max_target_positions;
    const int64_t vocab_size    = cm->hparams.dec_vocab_size;
    const int     eos_id        = cm->tok.eos_id() >= 0 ? cm->tok.eos_id() : 50257;
    const int     timestamp_begin = cm->hparams.no_timestamps_token_id + 1;
    constexpr int k_max_new_tokens = 256;

    const transcribe_timestamp_kind requested_timestamps =
        params != nullptr ? params->timestamps : TRANSCRIBE_TIMESTAMPS_NONE;
    if (requested_timestamps == TRANSCRIBE_TIMESTAMPS_WORD) {
        return TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS;
    }
    const bool want_segment_timestamps =
        requested_timestamps == TRANSCRIBE_TIMESTAMPS_AUTO ||
        requested_timestamps == TRANSCRIBE_TIMESTAMPS_SEGMENT;

    // Multilingual whisper variants emit <|lang|> + <|task|> tokens in
    // the decoder prefix; English-only (.en) variants do not — their
    // prefix is just <|sot|>. The two also differ at runtime: .en
    // models do NOT carry <|translate|> / <|transcribe|> task tokens
    // and have no per-language tokens to detect against. The
    // supports_language_detect capability is the canonical signal; .en
    // load sets it to false.
    const bool is_multilingual = cm->caps.supports_language_detect;

    int32_t task_token = -1;
    if (is_multilingual) {
        task_token = (params != nullptr && params->task == TRANSCRIBE_TASK_TRANSLATE)
                         ? cm->hparams.translate_token_id
                         : cm->hparams.transcribe_token_id;
        if (task_token < 0) {
            return TRANSCRIBE_ERR_GGUF;
        }
    } else if (params != nullptr && params->task == TRANSCRIBE_TASK_TRANSLATE) {
        std::fprintf(stderr,
                     "whisper run: this model does not support translate "
                     "(non-multilingual variant)\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Language hint wins; otherwise we run a SOT-only prefill inside the
    // first chunk's loop iteration (once we have that chunk's enc_host).
    // For non-multilingual (.en) models, the only accepted language is
    // "en" and there is no language token to resolve — the prefix omits
    // the <|lang|> slot entirely.
    int32_t lang_token = -1;
    if (params != nullptr && params->language != nullptr &&
        params->language[0] != '\0')
    {
        const std::string lang_code = params->language;
        if (!is_multilingual) {
            if (lang_code != "en") {
                std::fprintf(stderr,
                             "whisper run: language '%s' is not supported by "
                             "this non-multilingual model (only 'en')\n",
                             lang_code.c_str());
                return TRANSCRIBE_ERR_INVALID_ARG;
            }
            // Accepted; lang_token stays -1 by design.
        } else {
            for (size_t i = 0; i < cm->lang_codes.size(); ++i) {
                if (cm->lang_codes[i] == lang_code) {
                    lang_token = cm->lang_token_ids[i];
                    break;
                }
            }
            if (lang_token < 0) {
                std::fprintf(stderr,
                             "whisper run: language '%s' not in model's language "
                             "table — cannot build decoder prompt\n",
                             lang_code.c_str());
                return TRANSCRIBE_ERR_INVALID_ARG;
            }
        }
    }

    auto new_compute_ctx = [&](size_t mem) -> bool {
        return ensure_compute_ctx(cc, mem);
    };
    auto suppress_in_place = [&](std::vector<float> & logits) {
        for (int32_t id : cm->hparams.suppress_tokens) {
            if (id >= 0 && id < vocab_size) {
                logits[static_cast<size_t>(id)] = -INFINITY;
            }
        }
    };
    auto token_is_timestamp = [&](int id) {
        return id >= timestamp_begin && id < static_cast<int>(vocab_size);
    };

    // ----- Chunk loop ------------------------------------------------
    cc->clear_result();
    cc->chunk_traces.clear();
    std::vector<int32_t> all_text_ids;
    all_text_ids.reserve(256);

    std::vector<float> last_logits(static_cast<size_t>(vocab_size));
    int64_t t_decode_start = 0;
    bool    t_decode_started = false;

    // Finalize the context's result state from whatever accumulated so
    // far. Called at normal completion AND at every mid-run abort exit
    // (honoring the public contract in include/transcribe.h that
    // partial segments/text accumulated before TRANSCRIBE_ERR_ABORTED
    // must be readable via the normal accessors, distinguishable from
    // "no result" by transcribe_was_aborted()).
    //
    // Matches end-of-run finalization: decodes all_text_ids to UTF-8,
    // strips leading/trailing whitespace, pushes the text-only segment
    // for TIMESTAMPS_NONE mode, and flips has_result. Safe to call
    // from any point once the chunk loop has run at least zero times
    // — an empty all_text_ids yields an empty text + an empty NONE-mode
    // segment, which are the accessor sentinels anyway.
    auto commit_result = [&]() {
        cc->t_decode_us = t_decode_started
                              ? ggml_time_us() - t_decode_start
                              : 0;

        std::string text;
        if (!all_text_ids.empty()) {
            std::vector<int> ids_int(all_text_ids.begin(),
                                     all_text_ids.end());
            text = cm->tok.decode(ids_int.data(),
                                  static_cast<int>(ids_int.size()));
            size_t start = 0;
            while (start < text.size() &&
                   (text[start] == ' ' || text[start] == '\t' ||
                    text[start] == '\n' || text[start] == '\r'))
            {
                ++start;
            }
            size_t end = text.size();
            while (end > start &&
                   (text[end - 1] == ' ' || text[end - 1] == '\t' ||
                    text[end - 1] == '\n' || text[end - 1] == '\r'))
            {
                --end;
            }
            text = text.substr(start, end - start);
        }

        if (!want_segment_timestamps) {
            transcribe_context::SegmentEntry seg {};
            seg.text = text;
            cc->segments.push_back(std::move(seg));
        }
        cc->full_text   = std::move(text);
        cc->result_kind = want_segment_timestamps
                              ? TRANSCRIBE_TIMESTAMPS_SEGMENT
                              : TRANSCRIBE_TIMESTAMPS_NONE;
        cc->has_result  = true;

        if (whisper_perf_enabled()) {
            print_whisper_perf(cc->perf);
        }
    };

    // Run-scoped Whisper params pointer + RNG.
    //
    // The params pointer is hoisted here so per-chunk state (temperature
    // tuple, threshold checks, max_initial_timestamp cap) all read from a
    // single source. NULL selects the library defaults; the local
    // `default_wp` must outlive the chunk loop because `wp` aliases it.
    //
    // RNG must be run-scoped, not chunk-scoped. HF does not reset its
    // sampler between chunks (generation_whisper.py threads a single
    // torch.Generator through the whole seek loop), so constructing
    // std::mt19937 per chunk with the caller's seed would replay the
    // same Mersenne-twister prefix at the start of each 30-second
    // window — destroying determinism in exactly the case a fixed seed
    // is meant to achieve (reproducible long-form transcripts). When
    // seed == 0 we draw an OS entropy sample once and then let the
    // single rng advance monotonically across chunks and tiers.
    const transcribe_whisper_params default_wp =
        transcribe_whisper_default_params();
    const transcribe_whisper_params * wp =
        (params != nullptr && params->whisper != nullptr)
            ? params->whisper : &default_wp;
    std::mt19937 rng(wp->seed != 0 ? wp->seed : std::random_device{}());

    // ===== Stage 3: prompt + condition_on_prev_tokens setup =============
    //
    // HF generation_whisper.py:1743-1745 raises ValueError when
    // prompt_condition_type=="all-segments" without
    // condition_on_prev_tokens=True. Mirror as INVALID_ARG before any
    // compute runs.
    if (wp->prompt_condition == TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS &&
        !wp->condition_on_prev_tokens)
    {
        std::fprintf(stderr,
                     "whisper run: prompt_condition=ALL_SEGMENTS requires "
                     "condition_on_prev_tokens=true (HF parity)\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Resolve <|startofprev|> id. The converter writes
    // stt.whisper.prev_sot_token_id into the GGUF; tokenizer fallback
    // covers older artifacts that pre-date that key.
    int32_t prev_sot_id = cm->hparams.prev_sot_token_id;
    if (prev_sot_id < 0) {
        prev_sot_id = cm->tok.find("<|startofprev|>");
    }
    const bool prompt_requested =
        (wp->prompt_tokens != nullptr && wp->n_prompt_tokens > 0) ||
        (wp->initial_prompt != nullptr && wp->initial_prompt[0] != '\0');
    if (prev_sot_id < 0 &&
        (prompt_requested || wp->condition_on_prev_tokens))
    {
        std::fprintf(stderr,
                     "whisper run: model has no <|startofprev|> token; "
                     "initial_prompt / condition_on_prev_tokens unavailable\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Resolve initial prompt → text-only token ids (no <|startofprev|>;
    // the library prepends it). Two input paths:
    //
    //   prompt_tokens: caller-owned bytes used verbatim. Caller must
    //     supply only the text-side tokens — we add <|startofprev|>.
    //
    //   initial_prompt (string): tokenize HF's get_prompt_ids form
    //     ("<|startofprev|>" + " " + initial_prompt.strip()) and reject
    //     any special token (id >= eos_id) that appears in the text
    //     tokens, mirroring tokenization_whisper.py:716. We feed the
    //     leading-space-stripped form through the GPT-2 BPE encoder
    //     (which doesn't itself emit specials), so only special tokens
    //     present literally in user text would surface here.
    const int max_prev_cap = wp->max_prev_context_tokens > 0
                                 ? wp->max_prev_context_tokens
                                 : (cm->hparams.dec_max_target_positions / 2 - 1);
    std::vector<int32_t> prompt_text_ids;
    if (wp->prompt_tokens != nullptr && wp->n_prompt_tokens > 0) {
        // Caller-owned bytes used verbatim. The library prepends
        // <|startofprev|>, so a leading prev_sot id from the caller
        // would double-prepend — surface as INVALID_ARG so the
        // mistake is caught immediately rather than silently producing
        // a malformed prefix. (Header documents this contract;
        // runtime check makes it observable.)
        if (prev_sot_id >= 0 && wp->prompt_tokens[0] == prev_sot_id) {
            std::fprintf(stderr,
                         "whisper run: prompt_tokens must not include "
                         "<|startofprev|> (id %d) at index 0; the library "
                         "prepends it. See transcribe_whisper_params docs.\n",
                         prev_sot_id);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        prompt_text_ids.assign(wp->prompt_tokens,
                               wp->prompt_tokens + wp->n_prompt_tokens);
    } else if (wp->initial_prompt != nullptr &&
               wp->initial_prompt[0] != '\0')
    {
        std::string s(wp->initial_prompt);
        auto issp = [](char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        };
        size_t a = 0, b = s.size();
        while (a < b && issp(s[a])) ++a;
        while (b > a && issp(s[b - 1])) --b;
        if (b > a) {
            std::string text(" ");
            text.append(s.data() + a, b - a);

            // Pre-check for special-token literals (`<|...|>`) before
            // BPE-encoding. HF's get_prompt_ids relies on the
            // tokenizer's added-token recognition to surface specials
            // as single ids and rejects any with id >=
            // all_special_ids[0] (== eos_id). Our gpt-2 BPE encoder
            // doesn't recognize specials, so without this scan a
            // literal "<|en|>" in user text would silently BPE-encode
            // byte-by-byte and slip through. Mirror HF's intent by
            // checking each "<|...|>" substring against the vocab
            // directly.
            for (size_t i = 0; i + 1 < text.size(); ) {
                if (text[i] == '<' && text[i + 1] == '|') {
                    const size_t end = text.find("|>", i + 2);
                    if (end != std::string::npos) {
                        const size_t close = end + 2;
                        std::string piece = text.substr(i, close - i);
                        const int id = cm->tok.find(piece);
                        if (id >= eos_id) {
                            std::fprintf(stderr,
                                         "whisper run: initial_prompt contains "
                                         "disallowed special token \"%s\" (id %d)\n",
                                         piece.c_str(), id);
                            return TRANSCRIBE_ERR_INVALID_ARG;
                        }
                        i = close;
                        continue;
                    }
                }
                ++i;
            }

            if (cm->tok.encode(text, prompt_text_ids) != TRANSCRIBE_OK) {
                std::fprintf(stderr,
                             "whisper run: tokenizer.encode failed on "
                             "initial_prompt\n");
                return TRANSCRIBE_ERR_GGUF;
            }
            for (int32_t id : prompt_text_ids) {
                if (id >= eos_id) {
                    std::fprintf(stderr,
                                 "whisper run: initial_prompt contains "
                                 "disallowed special token id %d\n", id);
                    return TRANSCRIBE_ERR_INVALID_ARG;
                }
            }
        }
    }
    // Cap prompt tokens to max_prev_cap (left-truncate, keep most-recent).
    if (static_cast<int>(prompt_text_ids.size()) > max_prev_cap) {
        prompt_text_ids.erase(prompt_text_ids.begin(),
                              prompt_text_ids.end() - max_prev_cap);
    }

    // HF current_segments[0]. Store history as segment token slices,
    // not one flattened vector, because _pad_to_max_length applies
    // skip_ending_double_timestamps to each segment independently.
    // For "first-segment" the prompt sits at the head of the history;
    // for "all-segments" the history starts empty and the prompt is
    // prepended per-chunk as the BOS.
    std::vector<std::vector<int32_t>> prev_history_segments;
    if (wp->prompt_condition == TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT &&
        !prompt_text_ids.empty())
    {
        prev_history_segments.push_back(prompt_text_ids);
    }

    // Per-chunk decision; HF auto-disables on the previous chunk's hot
    // accepted temperature (>= 0.5) per generation_whisper.py:1090-1093.
    // Initialize to the user knob; flip per accepted tier at chunk end.
    bool do_condition_on_prev_tokens = wp->condition_on_prev_tokens;

    int seek = 0;
    while (seek < total_mel_frames) {
        if (cc->poll_abort()) {
            commit_result();
            return TRANSCRIBE_ERR_ABORTED;
        }
        cc->perf.chunks += 1;

        const bool is_first_chunk = (seek == 0);
        const int64_t time_offset_ms = static_cast<int64_t>(seek) * 10;

        // Slice + zero-pad the mel window to n_mel_frames_per_chunk.
        // Zero-pad is ordered after the real frames, matching HF's
        // F.pad(..., (0, num_segment_frames - cur_frames)). seek_num_frames
        // is the count of real (unpadded) frames in this chunk — also the
        // upper bound on how far `seek` can advance in one iteration.
        std::vector<float> chunk_mel(
            static_cast<size_t>(n_mels) *
                static_cast<size_t>(n_mel_frames_per_chunk),
            0.0f);
        const int seek_num_frames =
            std::min(total_mel_frames - seek, n_mel_frames_per_chunk);
        const int frames_avail = seek_num_frames;
        std::memcpy(chunk_mel.data(),
                    &cc->mel_buf[static_cast<size_t>(seek) *
                                 static_cast<size_t>(n_mels)],
                    static_cast<size_t>(frames_avail) *
                    static_cast<size_t>(n_mels) * sizeof(float));

        // Encoder (per chunk). enc_host is rewritten per call.
        int T_enc_local = 0;
        const int64_t t_enc_start = ggml_time_us();
        if (const transcribe_status st = run_whisper_encoder_on_window(
                cc, cm, chunk_mel.data(), n_mels, n_mel_frames_per_chunk,
                /*allow_dumps=*/is_first_chunk, T_enc_local);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        if (is_first_chunk) {
            cc->t_encode_us = ggml_time_us() - t_enc_start;
            t_decode_start  = ggml_time_us();
            t_decode_started = true;
        }

        // Language detection — once, on first chunk, only if no hint
        // and the model actually has language tokens. Non-multilingual
        // (.en) models have no lang_token_ids so detection is skipped
        // automatically; lang_token stays -1 and the prefix below omits
        // the <|lang|> slot entirely.
        if (is_first_chunk && is_multilingual &&
            lang_token < 0 && !cm->lang_token_ids.empty())
        {
            // Lazily materialize encoder output to host for the
            // prefill graph's encoder_out_in input. The hot path
            // (cross-KV) reads cc->enc_out.tensor directly; this
            // copy is one-shot per run when language detection
            // is needed.
            const int d_enc_lang = cc->enc_out.d_model;
            const int T_enc_lang = cc->enc_out.T_enc;
            cc->enc_host.resize(static_cast<size_t>(d_enc_lang) *
                                static_cast<size_t>(T_enc_lang));
            ggml_backend_tensor_get(cc->enc_out.tensor,
                                    cc->enc_host.data(), 0,
                                    cc->enc_host.size() * sizeof(float));

            if (!new_compute_ctx(16 * 1024 * 1024)) {
                return TRANSCRIBE_ERR_GGUF;
            }
            DecoderBuild det_db = build_decoder_prefill_graph(
                cc->compute_ctx, cm->weights, cm->hparams,
                /*seq_len=*/1, T_enc_local, cc->decoder_use_flash);
            if (det_db.out == nullptr || det_db.graph == nullptr) {
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, det_db.graph)) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const int32_t sot = cm->hparams.decoder_start_token_id;
            ggml_backend_tensor_set(det_db.token_ids_in, &sot,
                                    0, sizeof(int32_t));
            ggml_backend_tensor_set(det_db.encoder_out_in, cc->enc_host.data(),
                                    0, cc->enc_host.size() * sizeof(float));
            if (det_db.causal_mask_in != nullptr) {
                const float zero = 0.0f;
                ggml_backend_tensor_set(det_db.causal_mask_in, &zero,
                                        0, sizeof(float));
            }
            if (ggml_backend_sched_graph_compute(cc->sched, det_db.graph) !=
                GGML_STATUS_SUCCESS)
            {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t row_bytes =
                static_cast<size_t>(vocab_size) * sizeof(float);
            ggml_backend_tensor_get(det_db.dumps.logits_raw, last_logits.data(),
                                    0, row_bytes);

            float best = -INFINITY;
            int best_index = -1;
            for (size_t i = 0; i < cm->lang_token_ids.size(); ++i) {
                const int32_t id = cm->lang_token_ids[i];
                if (id >= 0 && id < static_cast<int>(vocab_size)) {
                    const float v = last_logits[static_cast<size_t>(id)];
                    if (v > best) {
                        best = v;
                        lang_token = id;
                        best_index = static_cast<int>(i);
                    }
                }
            }
            // Publish the detected ISO code only here, not in the
            // user-hint branch above: this field's contract is "what the
            // model told us when we asked it to pick," not "what the
            // caller already knew."
            if (best_index >= 0 &&
                static_cast<size_t>(best_index) < cm->lang_codes.size())
            {
                cc->detected_language = cm->lang_codes[static_cast<size_t>(best_index)];
            }
        }
        if (is_multilingual && lang_token < 0) {
            std::fprintf(stderr,
                         "whisper run: could not resolve a decoder language "
                         "token\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        // ===== Stage 3: per-chunk prefix assembly =====================
        //
        // Mirrors HF generation_whisper.py:_prepare_decoder_input_ids
        // (1853-1918). The decoder input is prev_tokens + init_tokens:
        //
        //   if do_condition_on_prev_tokens AND history non-empty:
        //     prev_tokens = bos + history[-cut_off:]
        //       bos = [<|startofprev|>, prompt_text_ids...] for ALL_SEGMENTS
        //       bos = [<|startofprev|>] for FIRST_SEGMENT
        //     skip_ending_double_timestamps: if history's last two ids are
        //     both timestamps, drop the last one (PR #34537 / #35750).
        //   elif initial prompt is set:
        //     prev_tokens = [<|startofprev|>, prompt_text_ids...] (every chunk)
        //   else:
        //     prev_tokens = empty (Stage 2 behavior)
        std::vector<int32_t> prev_tokens;
        if (do_condition_on_prev_tokens && !prev_history_segments.empty() &&
            prev_sot_id >= 0)
        {
            prev_tokens.push_back(prev_sot_id);
            if (wp->prompt_condition == TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS &&
                !prompt_text_ids.empty())
            {
                prev_tokens.insert(prev_tokens.end(),
                                   prompt_text_ids.begin(),
                                   prompt_text_ids.end());
            }
            std::vector<int32_t> hist;
            for (const auto & seg : prev_history_segments) {
                size_t n = seg.size();
                if (n > 2 && token_is_timestamp(seg[n - 2])) {
                    // HF _pad_to_max_length(...,
                    // skip_ending_double_timestamps=True) drops the
                    // last token from any segment whose penultimate
                    // token is a timestamp.
                    --n;
                }
                hist.insert(hist.end(), seg.begin(), seg.begin() + n);
            }
            const int cap = std::min<int>(static_cast<int>(hist.size()),
                                          max_prev_cap);
            prev_tokens.insert(prev_tokens.end(),
                               hist.end() - cap, hist.end());
        } else if (!prompt_text_ids.empty() && prev_sot_id >= 0) {
            prev_tokens.push_back(prev_sot_id);
            prev_tokens.insert(prev_tokens.end(),
                               prompt_text_ids.begin(),
                               prompt_text_ids.end());
        }

        // Prefix for this chunk:
        //   multilingual: prev_tokens + [SOT, lang, task, notimestamps?]
        //   .en:          prev_tokens + [SOT,             notimestamps?]
        // Non-multilingual models have neither <|lang|> nor <|task|>
        // tokens in their vocab; emitting either would land on a
        // garbage id.
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(prev_tokens.size() + 4);
        prompt_ids.insert(prompt_ids.end(),
                          prev_tokens.begin(), prev_tokens.end());
        prompt_ids.push_back(cm->hparams.decoder_start_token_id);
        if (is_multilingual) {
            prompt_ids.push_back(lang_token);
            prompt_ids.push_back(task_token);
        }
        if (!want_segment_timestamps) {
            prompt_ids.push_back(cm->hparams.no_timestamps_token_id);
        }
        const int seq_len = static_cast<int>(prompt_ids.size());

        // Position of the SOT token within the prefix. Used to read the
        // no-speech logits row from the prompt-pass (HF's
        // begin_index - start_of_trans_offset == len(prev_tokens)).
        const int sot_index = static_cast<int>(prev_tokens.size());

        // Guard: prefix + 1 generated token must fit in the decoder
        // self-attention window. With max_target_positions=448 and a
        // maxed prev (224) + init (4) prefix this is 228 — plenty. A
        // pathological caller passing prompt_tokens > 444 would land
        // here.
        if (seq_len + 1 > static_cast<int>(n_ctx_decoder)) {
            std::fprintf(stderr,
                         "whisper run: prefix length %d exceeds decoder "
                         "context %lld\n",
                         seq_len, static_cast<long long>(n_ctx_decoder));
            return TRANSCRIBE_ERR_INVALID_ARG;
        }

        // Per-chunk generated state.
        std::vector<int32_t> generated_ids;
        std::vector<int32_t> generated_text_ids;
        generated_ids.reserve(128);
        generated_text_ids.reserve(64);

        auto consume_generated_token = [&](int id) {
            generated_ids.push_back(static_cast<int32_t>(id));
            if (!token_is_timestamp(id) && id >= 0 && id < 50257) {
                generated_text_ids.push_back(static_cast<int32_t>(id));
            }
        };

        // max_initial_timestamp_index: HF WhisperTimeStampLogitsProcessor
        // (generation/logits_process.py:2036-2042) masks timestamps above
        // `timestamp_begin + max_initial_timestamp_index` on the first
        // generated token only. The default max_initial_timestamp is
        // 1.0 s, which at Whisper's fixed 20 ms per timestamp token
        // resolves to index 50 — i.e. the first emitted timestamp can
        // mark a cut at most 1.0 s into the chunk. Prevents the decoder
        // from skipping over an entire window's worth of audio on the
        // opening cut.
        //
        // Negative / non-finite max_initial_timestamp disables the cap.
        const int max_initial_timestamp_index =
            (std::isfinite(wp->max_initial_timestamp) &&
             wp->max_initial_timestamp >= 0.0f)
                ? static_cast<int>(std::floor(
                      static_cast<double>(wp->max_initial_timestamp) / 0.02))
                : -1;

        // Mirrors transformers' WhisperTimeStampLogitsProcessor
        // (generation/logits_process.py). Five rules, in order:
        //   1. Always mask <|notimestamps|>.
        //   2. Pairing: timestamps come in pairs. After <text TS> the
        //      next token must close the pair with a TS (or EOS). After
        //      <TS TS> the pair is complete, the next token must be text.
        //   3. Monotonicity: timestamps never decrease.
        //   4. First-generated token must be a timestamp.
        //   5. If log-sum-exp over all timestamps > max single text
        //      logit, force a timestamp.
        //   6. On the first generated token only, cap the timestamp at
        //      timestamp_begin + max_initial_timestamp_index (HF
        //      logits_process.py:2040-2042).
        auto apply_timestamp_rules = [&](std::vector<float> & logits) {
            if (!want_segment_timestamps) {
                return;
            }
            if (cm->hparams.no_timestamps_token_id >= 0 &&
                cm->hparams.no_timestamps_token_id < vocab_size)
            {
                logits[static_cast<size_t>(cm->hparams.no_timestamps_token_id)] =
                    -INFINITY;
            }

            const bool last_ts =
                !generated_ids.empty() &&
                token_is_timestamp(generated_ids.back());
            const bool penult_ts =
                generated_ids.size() < 2 ||
                token_is_timestamp(generated_ids[generated_ids.size() - 2]);

            if (last_ts) {
                if (penult_ts) {
                    for (int i = timestamp_begin;
                         i < static_cast<int>(vocab_size); ++i)
                    {
                        logits[static_cast<size_t>(i)] = -INFINITY;
                    }
                } else {
                    // <text TS> pair opening: next token must either
                    // close the pair with another TS or terminate with
                    // EOS. HF mirrors this at logits_process.py:2018-2023
                    // as `scores_processed[k, :eos_token_id] = -inf` —
                    // upper bound is eos_id (exclusive), NOT
                    // timestamp_begin. Using timestamp_begin here would
                    // mask EOS and also mask the special-token band
                    // [eos_id+1, timestamp_begin) — but EOS is the
                    // correctness-critical one: without it a chunk that
                    // naturally wants to end on a single timestamp
                    // cannot terminate, which hurts single_timestamp_ending
                    // detection in _retrieve_segment.
                    const int mask_hi = std::min(
                        eos_id, static_cast<int>(vocab_size));
                    for (int i = 0; i < mask_hi; ++i) {
                        logits[static_cast<size_t>(i)] = -INFINITY;
                    }
                }
            }

            int last_ts_id = -1;
            for (auto it = generated_ids.rbegin();
                 it != generated_ids.rend(); ++it)
            {
                if (token_is_timestamp(*it)) { last_ts_id = *it; break; }
            }
            if (last_ts_id >= 0) {
                const int ts_low = (last_ts && !penult_ts)
                                       ? last_ts_id : last_ts_id + 1;
                const int clamp_hi = std::min(ts_low,
                                              static_cast<int>(vocab_size));
                for (int i = timestamp_begin; i < clamp_hi; ++i) {
                    logits[static_cast<size_t>(i)] = -INFINITY;
                }
            }

            if (generated_ids.empty()) {
                for (int i = 0; i < timestamp_begin; ++i) {
                    logits[static_cast<size_t>(i)] = -INFINITY;
                }
                // HF logits_process.py:2040-2042: on the first
                // generated token only, after forcing a timestamp, cap
                // the allowed timestamp to timestamp_begin +
                // max_initial_timestamp_index so the initial cut cannot
                // land later than max_initial_timestamp seconds.
                if (max_initial_timestamp_index >= 0) {
                    const int last_allowed =
                        timestamp_begin + max_initial_timestamp_index;
                    for (int i = last_allowed + 1;
                         i < static_cast<int>(vocab_size); ++i)
                    {
                        logits[static_cast<size_t>(i)] = -INFINITY;
                    }
                }
            }

            float max_ts = -INFINITY;
            for (int i = timestamp_begin;
                 i < static_cast<int>(vocab_size); ++i)
            {
                max_ts = std::max(max_ts, logits[static_cast<size_t>(i)]);
            }
            float ts_logsumexp = -INFINITY;
            if (std::isfinite(max_ts)) {
                double sum = 0.0;
                for (int i = timestamp_begin;
                     i < static_cast<int>(vocab_size); ++i)
                {
                    const float v = logits[static_cast<size_t>(i)];
                    if (std::isfinite(v)) {
                        sum += std::exp(static_cast<double>(v - max_ts));
                    }
                }
                if (sum > 0.0) {
                    ts_logsumexp =
                        max_ts + static_cast<float>(std::log(sum));
                }
            }
            float max_text = -INFINITY;
            for (int i = 0; i < timestamp_begin; ++i) {
                max_text = std::max(max_text,
                                    logits[static_cast<size_t>(i)]);
            }
            if (ts_logsumexp > max_text) {
                for (int i = 0; i < timestamp_begin; ++i) {
                    logits[static_cast<size_t>(i)] = -INFINITY;
                }
            }
        };

        int next_id = 0;

        // ===== Stage 2.4: temperature fallback setup ===================
        //
        // Per-chunk temperature tuple = [t0, t0+dt, t0+2*dt, ...] up to
        // and including 1.0. Default [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] —
        // OpenAI whisper's shipping recipe. A tier is accepted when its
        // compression_ratio < compression_ratio_thold AND its
        // avg_logprob > logprob_thold. If all tiers fail, keep the last
        // tier's output (HF does the same to avoid infinite loops).
        //
        // Thresholds use INF sentinels to mean "disabled"; see
        // transcribe_whisper_default_params() and the header's DISABLED
        // constants. `wp` and `rng` are hoisted to run scope above.
        std::vector<float> temperatures;
        temperatures.push_back(wp->temperature);
        if (wp->temperature_inc > 0.0f) {
            for (float T = wp->temperature + wp->temperature_inc;
                 T <= 1.0f + 1e-4f;
                 T += wp->temperature_inc)
            {
                temperatures.push_back(T);
            }
        }

        // Accepted-tier output (commits every tier so the last-fallback
        // is returned when no tier passes thresholds).
        std::vector<int32_t> accepted_generated_ids;
        std::vector<int32_t> accepted_generated_text_ids;
        float accepted_T             = 0.0f;
        float accepted_compression   = 0.0f;
        float accepted_avg_logprob   = 0.0f;
        int   accepted_n_fallbacks   = 0;

        // Stage 2.5 no-speech: probability captured from the SOT-position
        // row of the prompt-pass logits (tier 0 only, matching HF's
        // WhisperNoSpeechDetection which reruns on the full decoder
        // prefix and reads logits[:, begin_index - start_of_trans_offset]
        // — row 0 for our no-prev-context Stage 2 prefix). The token id
        // is positional: no_speech = notimestamps - 1 (multilingual:
        // 50362; .en: 50361).
        //
        // Post-decode, HF's _need_fallback
        // (generation_whisper.py:1275-1286) sets should_skip=True when
        //   avg_logprob < logprob_thold  AND
        //   no_speech_prob > no_speech_thold
        // — both conditions required, not one alone. When should_skip
        // fires, the chunk's output is discarded and seek advances by a
        // full window (no fallback is attempted, which matches setting
        // needs_fallback=False in HF).
        const int no_speech_token_id =
            cm->hparams.no_timestamps_token_id - 1;
        float no_speech_prob = 0.0f;
        bool  no_speech_prob_captured  = false;
        bool  no_speech_fired_this_chunk = false;

        // Port of HF _retrieve_compression_ratio's input convention
        // (generation_whisper.py:1074-1086). The ratio is computed over
        // the full generated tail including timestamp tokens, <|...|>
        // specials, and EOS when present — fallback decides before EOS
        // is stripped from seek_sequence. We mirror that here by
        // assembling the metric input as: generated_ids + (EOS if the
        // step loop terminated because it sampled eos_id). If the loop
        // hit k_max_new_tokens before EOS, no EOS marker is appended —
        // matching HF, whose seek_sequence in that case simply lacks an
        // EOS.
        bool tier_hit_eos = false;

        // KV cache allocation is tier-independent — done once per chunk
        // out here, then self-attention state (n, head) is reset per
        // tier below.
        const int n_layers = cm->hparams.dec_n_layers;
        const bool need_init =
            cc->kv_cache.ctx == nullptr ||
            cc->kv_cache.n_ctx != n_ctx_decoder ||
            cc->kv_cache.T_enc != T_enc_local;
        if (need_init) {
            cc->kv_cache.free();
            // AUTO: match cache dtype to the model's weight dtype. For
            // F32 GGUFs that's F32 (zero-loss cache, no precision loss
            // on the round-trip); for F16 and every quant preset it's
            // F16 (the production default — weight+activation already
            // pay F16 downcast costs, so the cache doesn't add new
            // precision loss). Explicit --kv-type overrides.
            ggml_type kv_type_g;
            if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
                kv_type_g = GGML_TYPE_F32;
            } else if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) {
                kv_type_g = GGML_TYPE_F16;
            } else {
                const ggml_tensor * probe =
                    !cm->weights.dec_blocks.empty()
                        ? cm->weights.dec_blocks[0].self_q_w
                        : nullptr;
                kv_type_g = (probe != nullptr && probe->type == GGML_TYPE_F32)
                                ? GGML_TYPE_F32 : GGML_TYPE_F16;
            }
            if (!kv_cache_init(cc->kv_cache, cm->plan.primary,
                               static_cast<int>(n_ctx_decoder), T_enc_local,
                               cm->hparams.dec_d_model, n_layers, kv_type_g))
            {
                std::fprintf(stderr, "whisper run: KV cache init failed\n");
                return TRANSCRIBE_ERR_BACKEND;
            }
        }

        // ---- Cross-attention K/V precompute (chunk-scoped) ------------
        //
        // Cross-K/V depends only on encoder output and the cross-attn
        // weights — both are tier-invariant. Compute once per chunk and
        // reuse across every fallback tier. Previously this lived inside
        // the tier loop and re-ran every tier, paying n_layers × T_enc ×
        // d_model² of redundant matmul per tier on fallback chunks.
        {
            const int64_t t_cross_build_start = ggml_time_us();
            if (!new_compute_ctx(8 * 1024 * 1024)) {
                std::fprintf(stderr,
                             "whisper run: ggml_init for cross_kv failed\n");
                return TRANSCRIBE_ERR_GGUF;
            }
            DecoderBuild cross_db = build_cross_kv_graph(
                cc->compute_ctx, cm->weights, cm->hparams,
                cc->kv_cache, cc->enc_out.tensor, T_enc_local);
            if (cross_db.graph == nullptr) {
                return TRANSCRIBE_ERR_GGUF;
            }
            cc->perf.cross_build.add(ggml_time_us() - t_cross_build_start);

            const int64_t t_cross_alloc_start = ggml_time_us();
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
                std::fprintf(stderr,
                             "whisper run: alloc_graph failed (cross_kv)\n");
                return TRANSCRIBE_ERR_GGUF;
            }
            // No tensor_set: cross-KV reads cc->enc_out.tensor via
            // a view inside build_cross_kv_graph, populated by the
            // encoder graph's final ggml_cpy.
            cc->perf.cross_alloc.add(ggml_time_us() - t_cross_alloc_start);

            const int64_t t_cross_compute_start = ggml_time_us();
            if (const ggml_status gs = ggml_backend_sched_graph_compute(
                    cc->sched, cross_db.graph);
                gs != GGML_STATUS_SUCCESS)
            {
                std::fprintf(stderr,
                             "whisper run: cross_kv compute failed (%d)\n",
                             static_cast<int>(gs));
                return TRANSCRIBE_ERR_GGUF;
            }
            cc->perf.cross_compute.add(ggml_time_us() - t_cross_compute_start);
            cc->kv_cache.cross_populated = true;
        }

        // ===== Tier loop (temperature fallback) ========================
        for (size_t ti = 0; ti < temperatures.size(); ++ti) {
            const float tier_T = temperatures[ti];

            // Per-tier reset. Self-cache resets every tier (each tier
            // generates its own token sequence); cross-cache stays
            // populated — its contents are tier-invariant.
            cc->kv_cache.n    = 0;
            cc->kv_cache.head = 0;
            generated_ids.clear();
            generated_text_ids.clear();
            tier_hit_eos = false;
            double sum_logprob       = 0.0;
            int    n_logprob_samples = 0;

            // Dump gate. Tolerance references were captured at T=0
            // (tier 0 at defaults); only tier 0 of the first chunk
            // emits intermediates.
            const bool emit_tier_dumps = is_first_chunk && (ti == 0);
            auto tier_try_dump = [&](const char * name, ggml_tensor * t,
                                     const char * stage)
            {
                if (t != nullptr && emit_tier_dumps) {
                    transcribe::debug::dump_tensor(name, t, stage);
                }
            };

            // ---- Prompt pass (emit dumps, n_past=0) -------------------
            {
                const int64_t t_prompt_build_start = ggml_time_us();
                if (!new_compute_ctx(16 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
                const int kv_pad = kv_pad_self_attn(
                    cm->plan.primary_kind, cc->decoder_use_flash);
                DecoderBuild db = build_decoder_graph_kv(
                    cc->compute_ctx, cm->weights, cm->hparams,
                    cc->kv_cache,
                    /*n_tokens=*/seq_len, /*n_past=*/0, T_enc_local,
                    /*kv_pad=*/kv_pad,
                    /*skip_log_softmax=*/false, cc->decoder_use_flash);
                if (db.out == nullptr || db.graph == nullptr) {
                    return TRANSCRIBE_ERR_GGUF;
                }
                cc->perf.prompt_build.add(
                    ggml_time_us() - t_prompt_build_start);

                const int64_t t_prompt_alloc_start = ggml_time_us();
                ggml_backend_sched_reset(cc->sched);
                if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
                    std::fprintf(stderr,
                                 "whisper run: alloc_graph failed (prompt)\n");
                    return TRANSCRIBE_ERR_GGUF;
                }

                ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(),
                                        0, prompt_ids.size() * sizeof(int32_t));
                std::vector<int32_t> pos_ids(seq_len);
                for (int i = 0; i < seq_len; ++i) pos_ids[i] = i;
                ggml_tensor * pos_in = ggml_graph_get_tensor(db.graph, "dec.pos_ids");
                ggml_backend_tensor_set(pos_in, pos_ids.data(),
                                        0, pos_ids.size() * sizeof(int32_t));

                if (db.causal_mask_in != nullptr) {
                    // Mask shape: [n_kv, n_tokens] = [n_kv, seq_len].
                    // db.causal_mask_in carries the runtime n_kv (may
                    // be padded beyond seq_len for FA alignment).
                    const int n_kv_mask =
                        static_cast<int>(db.causal_mask_in->ne[0]);
                    std::vector<float> mask(
                        static_cast<size_t>(n_kv_mask) * seq_len);
                    for (int q = 0; q < seq_len; ++q) {
                        for (int k = 0; k < n_kv_mask; ++k) {
                            // Causal in [0, seq_len), -inf for padded
                            // slots in [seq_len, n_kv_mask).
                            mask[static_cast<size_t>(q) * n_kv_mask + k] =
                                (k < seq_len && k <= q) ? 0.0f : -1e9f;
                        }
                    }
                    ggml_backend_tensor_set(db.causal_mask_in, mask.data(),
                                            0, mask.size() * sizeof(float));
                }

                if (db.cross_mask_in != nullptr) {
                    // Cross mask shape [T_enc_pad, n_tokens]. Zero
                    // for k in [0, T_enc), -inf for trailing padded
                    // slots — same for every query row.
                    const int n_kv_cross =
                        static_cast<int>(db.cross_mask_in->ne[0]);
                    std::vector<float> mask(
                        static_cast<size_t>(n_kv_cross) * seq_len);
                    for (int q = 0; q < seq_len; ++q) {
                        for (int k = 0; k < n_kv_cross; ++k) {
                            mask[static_cast<size_t>(q) * n_kv_cross + k] =
                                (k < T_enc_local) ? 0.0f : -1e9f;
                        }
                    }
                    ggml_backend_tensor_set(db.cross_mask_in, mask.data(),
                                            0, mask.size() * sizeof(float));
                }
                cc->perf.prompt_alloc.add(
                    ggml_time_us() - t_prompt_alloc_start);

                const int64_t t_prompt_compute_start = ggml_time_us();
                if (const ggml_status gs = ggml_backend_sched_graph_compute(
                        cc->sched, db.graph);
                    gs != GGML_STATUS_SUCCESS)
                {
                    return TRANSCRIBE_ERR_GGUF;
                }
                cc->perf.prompt_compute.add(
                    ggml_time_us() - t_prompt_compute_start);
                cc->kv_cache.n    = seq_len;
                cc->kv_cache.head = seq_len;

                tier_try_dump("dec.token_emb",       db.dumps.token_emb,       "decoder.embedding");
                tier_try_dump("dec.pos_emb",         db.dumps.pos_emb,         "decoder.position_embedding");
                tier_try_dump("dec.embed_sum",       db.dumps.embed_sum,       "decoder.embed_sum");
                for (size_t i = 0; i < db.dumps.block_outs.size(); ++i) {
                    char bname[64], stage[64];
                    std::snprintf(bname, sizeof(bname), "dec.block.%zu.out", i);
                    std::snprintf(stage, sizeof(stage), "decoder.block%zu.out", i);
                    tier_try_dump(bname, db.dumps.block_outs[i], stage);
                }
                tier_try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.output_before_head");
                tier_try_dump("dec.logits_raw",      db.dumps.logits_raw,      "decoder.logits_raw");
                tier_try_dump("dec.logits",          db.dumps.logits,          "decoder.logits");

                const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);

                // Capture no_speech_prob from the RAW SOT-position row,
                // matching HF WhisperNoSpeechDetection semantics
                // (transformers generation/logits_process.py:2095-2115):
                // on the first generated-token step, when
                // start_of_trans_offset > 1 — which is always our case
                // (prefix always carries SOT + lang + task [+
                // notimestamps]) — HF reruns the model on the full
                // decoder prefix and indexes
                //   logits[:, begin_index - start_of_trans_offset]
                // which collapses to logits[:, sot_index] where sot_index
                // is the position of <|startoftranscript|> within the
                // prefix. With Stage 3 prev tokens this is len(prev_tokens);
                // with no prev tokens it's 0 (matches the Stage 2 capture).
                //
                // Tier 0 only — matches HF's one-shot capture. Read the
                // raw SOT row into a temporary buffer so suppress /
                // begin_suppress / timestamp rules applied below to
                // last_logits (the seq_len-1 row used to sample the first
                // generated token) do not contaminate the measurement.
                if (ti == 0 && !no_speech_prob_captured &&
                    no_speech_token_id >= 0 &&
                    no_speech_token_id < static_cast<int>(vocab_size))
                {
                    std::vector<float> sot_logits(
                        static_cast<size_t>(vocab_size));
                    const int64_t t_prompt_tget_sot = ggml_time_us();
                    ggml_backend_tensor_get(
                        db.dumps.logits_raw, sot_logits.data(),
                        row_bytes * static_cast<size_t>(sot_index),
                        row_bytes);
                    cc->perf.prompt_tensor_get.add(
                        ggml_time_us() - t_prompt_tget_sot);
                    float max_l = -std::numeric_limits<float>::infinity();
                    for (auto l : sot_logits) {
                        if (std::isfinite(l) && l > max_l) max_l = l;
                    }
                    double sum = 0.0;
                    double ns  = 0.0;
                    if (std::isfinite(max_l)) {
                        for (size_t i = 0; i < sot_logits.size(); ++i) {
                            const float l = sot_logits[i];
                            if (std::isfinite(l)) {
                                const double e = std::exp(
                                    static_cast<double>(l - max_l));
                                sum += e;
                                if (static_cast<int>(i) ==
                                    no_speech_token_id)
                                {
                                    ns = e;
                                }
                            }
                        }
                    }
                    no_speech_prob = (sum > 0.0)
                        ? static_cast<float>(ns / sum) : 0.0f;
                    no_speech_prob_captured = true;
                }

                // Last-row logits → first generated token.
                const int64_t t_prompt_tget_last = ggml_time_us();
                ggml_backend_tensor_get(db.dumps.logits_raw, last_logits.data(),
                                        row_bytes * static_cast<size_t>(seq_len - 1),
                                        row_bytes);
                cc->perf.prompt_tensor_get.add(
                    ggml_time_us() - t_prompt_tget_last);

                const int64_t t_prompt_cpu_start = ggml_time_us();
                const int64_t t_prompt_suppress_start = ggml_time_us();
                suppress_in_place(last_logits);
                for (int32_t id : cm->hparams.begin_suppress_tokens) {
                    if (id >= 0 && id < vocab_size) {
                        last_logits[static_cast<size_t>(id)] = -INFINITY;
                    }
                }
                cc->perf.prompt_cpu_suppress.add(
                    ggml_time_us() - t_prompt_suppress_start);

                const int64_t t_prompt_ts_start = ggml_time_us();
                apply_timestamp_rules(last_logits);
                cc->perf.prompt_cpu_timestamp.add(
                    ggml_time_us() - t_prompt_ts_start);

                if (tier_T <= 0.0f) {
                    // T==0: fused argmax + log-softmax. The whole call
                    // is recorded under sample; logprob bucket stays 0
                    // for this path (it is computed for free here).
                    const int64_t t_prompt_sample_start = ggml_time_us();
                    float lp = 0.0f;
                    next_id = sample_argmax_and_logprob(last_logits, &lp);
                    sum_logprob += lp;
                    cc->perf.prompt_cpu_sample.add(
                        ggml_time_us() - t_prompt_sample_start);
                } else {
                    const int64_t t_prompt_sample_start = ggml_time_us();
                    next_id = sample_from_logits(
                        last_logits, tier_T, rng, &cc->sample_scratch);
                    cc->perf.prompt_cpu_sample.add(
                        ggml_time_us() - t_prompt_sample_start);

                    const int64_t t_prompt_lp_start = ggml_time_us();
                    sum_logprob +=
                        logprob_of_token_hf(last_logits, next_id, tier_T);
                    cc->perf.prompt_cpu_logprob.add(
                        ggml_time_us() - t_prompt_lp_start);
                }
                n_logprob_samples += 1;
                cc->perf.prompt_cpu.add(ggml_time_us() - t_prompt_cpu_start);
            }

            // ---- Step loop (n_tokens=1) -------------------------------
            //
            // Two variants:
            //   GPU (Vulkan/Metal/CUDA/SYCL): build_step_graph — one
            //     static-topology graph per tier. KV writes via
            //     ggml_set_rows at runtime kv_idx; flash-attn reads a
            //     fixed max_n_kv window with a runtime mask. Removes
            //     per-step graph_build + sched_alloc.
            //   CPU (or debug, which needs the gen20 dump path): per-
            //     step build_decoder_graph_kv with a dynamic n_kv.
            int n_past = seq_len;
            const bool primary_is_gpu =
                cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
                cm->plan.primary_kind != transcribe::BackendKind::Accel &&
                cm->plan.primary_kind != transcribe::BackendKind::Unknown;
            const bool use_step_graph = primary_is_gpu &&
                                        !transcribe::debug::enabled();

            // Sized to fit prompt + max generated tail, padded to next
            // pow2. For whisper this is typically 256 or 512 (cap is
            // n_ctx_decoder=448).
            int max_n_kv = 256;
            while (max_n_kv < seq_len + k_max_new_tokens) max_n_kv *= 2;
            if (max_n_kv > static_cast<int>(n_ctx_decoder)) {
                max_n_kv = static_cast<int>(n_ctx_decoder);
            }

            // Step graph + persistent host buffers, only valid on
            // use_step_graph path. Built lazily inside the loop on
            // first use to keep the dynamic-graph fallback unaffected.
            StepBuild sb {};
            std::vector<ggml_fp16_t> step_mask;
            std::vector<float>       step_cross_mask;
            const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);

            if (use_step_graph) {
                if (!new_compute_ctx(8 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
                sb = build_step_graph(
                    cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                    max_n_kv, T_enc_local, cc->decoder_use_flash);
                if (sb.graph == nullptr || sb.logits_out == nullptr) {
                    std::fprintf(stderr,
                                 "whisper run: build_step_graph failed\n");
                    return TRANSCRIBE_ERR_GGUF;
                }
                ggml_backend_sched_reset(cc->sched);
                if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
                    std::fprintf(stderr,
                                 "whisper run: sched_alloc_graph failed (step)\n");
                    return TRANSCRIBE_ERR_GGUF;
                }

                // Self-attn mask: [0, seq_len) populated by prompt pass
                // are attendable; [seq_len, max_n_kv) start as -inf.
                step_mask.assign(max_n_kv, mask_neg_inf);
                for (int p = 0; p < seq_len; ++p) step_mask[p] = mask_zero;

                // Cross mask is invariant across steps within a chunk —
                // upload once and reuse.
                if (sb.cross_mask_in != nullptr) {
                    const int n_kv_cross =
                        static_cast<int>(sb.cross_mask_in->ne[0]);
                    step_cross_mask.assign(
                        static_cast<size_t>(n_kv_cross), 0.0f);
                    for (int k = T_enc_local; k < n_kv_cross; ++k) {
                        step_cross_mask[static_cast<size_t>(k)] = -1e9f;
                    }
                    ggml_backend_tensor_set(
                        sb.cross_mask_in, step_cross_mask.data(),
                        0, step_cross_mask.size() * sizeof(float));
                }
            }

            for (int step = 0; step < k_max_new_tokens; ++step) {
                if (next_id == eos_id) {
                    tier_hit_eos = true;
                    break;
                }
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }
                consume_generated_token(next_id);

                if (n_past + 1 > static_cast<int>(n_ctx_decoder)) break;
                if (use_step_graph && n_past + 1 > max_n_kv) {
                    std::fprintf(stderr,
                                 "whisper run: hit max_n_kv=%d at n_past=%d\n",
                                 max_n_kv, n_past);
                    break;
                }

                const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);

                if (use_step_graph) {
                    const int64_t t_step_alloc_start = ggml_time_us();
                    int32_t tok      = next_id;
                    int32_t pos_val  = n_past;
                    int64_t kv_val   = n_past;
                    ggml_backend_tensor_set(sb.token_id_in, &tok,     0, sizeof(int32_t));
                    ggml_backend_tensor_set(sb.pos_id_in,   &pos_val, 0, sizeof(int32_t));
                    ggml_backend_tensor_set(sb.kv_idx_in,   &kv_val,  0, sizeof(int64_t));

                    step_mask[n_past] = mask_zero;
                    ggml_backend_tensor_set(
                        sb.mask_in, step_mask.data(),
                        0, static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));
                    cc->perf.step_alloc.add(
                        ggml_time_us() - t_step_alloc_start);

                    const int64_t t_step_compute_start = ggml_time_us();
                    if (const ggml_status gs = ggml_backend_sched_graph_compute(
                            cc->sched, sb.graph);
                        gs != GGML_STATUS_SUCCESS)
                    {
                        return TRANSCRIBE_ERR_GGUF;
                    }
                    cc->perf.step_compute.add(
                        ggml_time_us() - t_step_compute_start);

                    n_past += 1;
                    cc->kv_cache.n    = n_past;
                    cc->kv_cache.head = n_past;

                    const int64_t t_step_tget_start = ggml_time_us();
                    ggml_backend_tensor_get(sb.logits_out, last_logits.data(),
                                            0, row_bytes);
                    cc->perf.step_tensor_get.add(
                        ggml_time_us() - t_step_tget_start);
                } else {
                    const int64_t t_step_build_start = ggml_time_us();
                    if (!new_compute_ctx(4 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
                    const int kv_pad = kv_pad_self_attn(
                        cm->plan.primary_kind, cc->decoder_use_flash);
                    DecoderBuild step_db = build_decoder_graph_kv(
                        cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                        /*n_tokens=*/1, /*n_past=*/n_past, T_enc_local,
                        /*kv_pad=*/kv_pad,
                        /*skip_log_softmax=*/true, cc->decoder_use_flash);
                    if (step_db.out == nullptr) return TRANSCRIBE_ERR_GGUF;
                    cc->perf.step_build.add(ggml_time_us() - t_step_build_start);

                    const int64_t t_step_alloc_start = ggml_time_us();
                    ggml_backend_sched_reset(cc->sched);
                    if (!ggml_backend_sched_alloc_graph(cc->sched, step_db.graph)) {
                        return TRANSCRIBE_ERR_GGUF;
                    }

                    int32_t tok = next_id;
                    int32_t pos = n_past;
                    ggml_backend_tensor_set(step_db.token_ids_in, &tok, 0, sizeof(int32_t));
                    ggml_tensor * pos_in = ggml_graph_get_tensor(step_db.graph, "dec.pos_ids");
                    ggml_backend_tensor_set(pos_in, &pos, 0, sizeof(int32_t));

                    // Padded step: build the trailing-padding mask. The
                    // valid range [0, n_past+1) is 0; trailing slots are
                    // -inf so the FA op ignores stale K/V values left in
                    // the cache from previous tiers / chunks.
                    if (step_db.causal_mask_in != nullptr) {
                        const int n_kv_mask =
                            static_cast<int>(step_db.causal_mask_in->ne[0]);
                        std::vector<float> mask(
                            static_cast<size_t>(n_kv_mask), 0.0f);
                        const int n_valid = n_past + 1;
                        for (int k = n_valid; k < n_kv_mask; ++k) {
                            mask[static_cast<size_t>(k)] = -1e9f;
                        }
                        ggml_backend_tensor_set(
                            step_db.causal_mask_in, mask.data(),
                            0, mask.size() * sizeof(float));
                    }

                    if (step_db.cross_mask_in != nullptr) {
                        const int n_kv_cross =
                            static_cast<int>(step_db.cross_mask_in->ne[0]);
                        std::vector<float> mask(
                            static_cast<size_t>(n_kv_cross), 0.0f);
                        for (int k = T_enc_local; k < n_kv_cross; ++k) {
                            mask[static_cast<size_t>(k)] = -1e9f;
                        }
                        ggml_backend_tensor_set(
                            step_db.cross_mask_in, mask.data(),
                            0, mask.size() * sizeof(float));
                    }
                    cc->perf.step_alloc.add(ggml_time_us() - t_step_alloc_start);

                    const int64_t t_step_compute_start = ggml_time_us();
                    if (const ggml_status gs = ggml_backend_sched_graph_compute(
                            cc->sched, step_db.graph);
                        gs != GGML_STATUS_SUCCESS)
                    {
                        return TRANSCRIBE_ERR_GGUF;
                    }
                    cc->perf.step_compute.add(ggml_time_us() - t_step_compute_start);

                    n_past += 1;
                    cc->kv_cache.n    = n_past;
                    cc->kv_cache.head = n_past;

                    // Mid-generation dump: step 19's logits exercise the
                    // n_past>0 KV read/write path the prompt-pass dump
                    // cannot reach. Captured on tier 0 of the first chunk
                    // only (tolerance references were generated at T=0).
                    if (step == 19) {
                        tier_try_dump("dec.logits_raw.gen20", step_db.out,
                                      "decoder.logits_raw.gen20");
                    }

                    const int64_t t_step_tget_start = ggml_time_us();
                    ggml_backend_tensor_get(step_db.out, last_logits.data(), 0, row_bytes);
                    cc->perf.step_tensor_get.add(ggml_time_us() - t_step_tget_start);
                }

                const int64_t t_step_cpu_start = ggml_time_us();
                const int64_t t_step_suppress_start = ggml_time_us();
                suppress_in_place(last_logits);
                cc->perf.step_cpu_suppress.add(
                    ggml_time_us() - t_step_suppress_start);

                const int64_t t_step_ts_start = ggml_time_us();
                apply_timestamp_rules(last_logits);
                cc->perf.step_cpu_timestamp.add(
                    ggml_time_us() - t_step_ts_start);

                if (tier_T <= 0.0f) {
                    const int64_t t_step_sample_start = ggml_time_us();
                    float lp = 0.0f;
                    next_id = sample_argmax_and_logprob(last_logits, &lp);
                    sum_logprob += lp;
                    cc->perf.step_cpu_sample.add(
                        ggml_time_us() - t_step_sample_start);
                } else {
                    const int64_t t_step_sample_start = ggml_time_us();
                    next_id = sample_from_logits(
                        last_logits, tier_T, rng, &cc->sample_scratch);
                    cc->perf.step_cpu_sample.add(
                        ggml_time_us() - t_step_sample_start);

                    const int64_t t_step_lp_start = ggml_time_us();
                    sum_logprob +=
                        logprob_of_token_hf(last_logits, next_id, tier_T);
                    cc->perf.step_cpu_logprob.add(
                        ggml_time_us() - t_step_lp_start);
                }
                n_logprob_samples += 1;
                cc->perf.step_cpu.add(ggml_time_us() - t_step_cpu_start);
            }

            // ---- Compute tier metrics ---------------------------------
            //
            // Compression ratio runs over the full generated tail — all
            // generated ids including timestamps / specials / EOS, with
            // decoder_input_ids (the prompt header) stripped. This
            // matches HF's _retrieve_compression_ratio input
            // (generation_whisper.py:1074-1086 calls
            // _retrieve_compression_ratio(seek_sequence, vocab_size)
            // before EOS is stripped at :1085). An earlier iteration
            // passed generated_text_ids here, which filtered out
            // timestamps and specials and diverged from HF's metric by
            // several percent on typical 20-50-token chunks — enough to
            // flip accept/escalate on the fallback boundary.
            std::vector<int32_t> comp_tail;
            comp_tail.reserve(generated_ids.size() + 1);
            comp_tail.insert(comp_tail.end(),
                             generated_ids.begin(), generated_ids.end());
            if (tier_hit_eos) {
                comp_tail.push_back(static_cast<int32_t>(eos_id));
            }
            const float tier_comp_ratio =
                compute_compression_ratio_hf(comp_tail, vocab_size);
            const float tier_avg_logprob =
                n_logprob_samples > 0
                    ? static_cast<float>(sum_logprob / n_logprob_samples)
                    : -std::numeric_limits<float>::infinity();

            // Thresholds.
            //
            // HF _need_fallback (generation_whisper.py:1243-1287) falls
            // back strictly on `comp_ratio > thold` or `avg_logprob <
            // thold`; equality does NOT trigger fallback. So a tier is
            // accepted (lp_ok/comp_ok) under `<= / >=`. The INF
            // sentinels still work: tier_comp_ratio <= +INF is always
            // true, tier_avg_logprob >= -INF is always true — disabled
            // thresholds trivially satisfy the check.
            const bool comp_ok =
                tier_comp_ratio <= wp->compression_ratio_thold;
            const bool lp_ok =
                tier_avg_logprob >= wp->logprob_thold;

            // HF _need_fallback (generation_whisper.py:1275-1286) sets
            //   should_skip = True, needs_fallback = False
            // when BOTH
            //   avg_logprob < logprob_threshold  AND
            //   no_speech_prob > no_speech_threshold
            // fire at the current tier. The skip halts fallback — HF
            // explicitly flips needs_fallback to False so the outer
            // generate_with_fallback loop terminates without trying
            // hotter temperatures. no_speech_prob is a tier-0 capture
            // (see above) so this is checked against the tier-0 value
            // for every tier, but the logprob is per-tier.
            //
            // Disable sentinels: no_speech_thold = +INF makes the first
            // comparison always false; logprob_thold = -INF makes the
            // second always false. Either one disabled → no skip.
            const bool no_speech_should_skip =
                no_speech_prob > wp->no_speech_thold &&
                tier_avg_logprob < wp->logprob_thold;

            // Commit the tier's output unconditionally. If no tier
            // passes comp_ok/lp_ok AND no_speech_should_skip never
            // fires, the last-tried tier wins (matches HF
            // generate_with_fallback's behavior of keeping the final
            // hypothesis when all tiers missed the thresholds). If
            // no_speech_should_skip fires, we discard the tier output
            // after recording the trace-visible metrics, below.
            accepted_generated_ids      = generated_ids;
            accepted_generated_text_ids = generated_text_ids;
            accepted_T                  = tier_T;
            accepted_compression        = tier_comp_ratio;
            accepted_avg_logprob        = tier_avg_logprob;
            accepted_n_fallbacks        = static_cast<int>(ti);

            if (no_speech_should_skip) {
                no_speech_fired_this_chunk = true;
                break;
            }
            if (comp_ok && lp_ok) {
                break;
            }
        }

        // Tier loop accepted something — hand off to segment emission.
        // When no-speech fired, the tier produced output but we throw
        // it away: seek still advances by a full window (per the
        // _retrieve_segment branch in the seek-advance block below).
        generated_ids      = accepted_generated_ids;
        generated_text_ids = accepted_generated_text_ids;
        if (no_speech_fired_this_chunk) {
            generated_ids.clear();
            generated_text_ids.clear();
        }

        // Stage 2.6 — commit per-chunk trace. Global window is
        // [time_offset_ms, time_offset_ms + seek_num_frames*10ms). The
        // trace captures what the fallback loop decided and what the
        // no-speech gate saw; callers tracing a hallucination can map
        // an output segment back to the tier/metric that produced it.
        {
            transcribe_whisper_chunk_trace trace = {};
            trace.t0_ms               = time_offset_ms;
            trace.t1_ms               = time_offset_ms +
                                        static_cast<int64_t>(seek_num_frames) * 10;
            trace.temperature_used    = accepted_T;
            trace.compression_ratio   = accepted_compression;
            trace.avg_logprob         = accepted_avg_logprob;
            trace.no_speech_prob      = no_speech_prob;
            trace.no_speech_triggered = no_speech_fired_this_chunk;
            trace.n_fallbacks         = accepted_n_fallbacks;
            cc->chunk_traces.push_back(trace);
        }

        // Stage 3 — condition_on_prev_tokens gate for the NEXT chunk.
        //
        // HF generation_whisper.py:1090-1093 sets:
        //   is_low_temperature = temperature is None or temperature < 0.5
        //   do_condition[i] = condition_on_prev_tokens AND is_low_temperature
        // Hot tiers (>= 0.5) are presumed to have hallucinated, so the
        // next chunk forgoes the prev-context window to avoid immediate
        // propagation. The history itself is updated after
        // _retrieve_segment below, because HF carries only the segment
        // tokens that survive that slicing step.
        do_condition_on_prev_tokens =
            wp->condition_on_prev_tokens && (accepted_T < 0.5f);

        // ----- _retrieve_segment ---------------------------------------
        //
        // Port of HF generation_whisper.py:_retrieve_segment
        // (1976-2073). Splits generated_ids on consecutive-timestamp
        // pairs (each pair `<|t_a|><|t_b|>` marks "text between
        // timestamps spans local time a..b"). Emits one SegmentEntry
        // per pair. Returns segment_offset_frames — how far to advance
        // `seek`:
        //
        //   - Chunk ended with a single ts (pair not yet closed): no
        //     speech after last ts → advance past entire chunk
        //     (seek_num_frames). Preserves the tail for the next
        //     chunk's decoder.
        //   - Chunk has at least one closed pair but not single-ended:
        //     discard unfinished tail, advance by the last *closed* ts
        //     position × input_stride(2) mel frames.
        //   - No pairs at all: emit one segment covering the full
        //     chunk; advance by seek_num_frames.
        //
        // For TIMESTAMPS_NONE mode the prompt includes <|notimestamps|>
        // so generated_ids carries no ts tokens → "no pairs" branch,
        // full-chunk advance, no per-chunk segment emission (we gate
        // on want_segment_timestamps).
        auto trim_ascii = [](std::string s) {
            size_t start = 0;
            while (start < s.size() &&
                   (s[start] == ' ' || s[start] == '\t' ||
                    s[start] == '\n' || s[start] == '\r'))
            {
                ++start;
            }
            size_t end = s.size();
            while (end > start &&
                   (s[end - 1] == ' ' || s[end - 1] == '\t' ||
                    s[end - 1] == '\n' || s[end - 1] == '\r'))
            {
                --end;
            }
            return s.substr(start, end - start);
        };
        auto decode_range = [&](int begin, int end) -> std::string {
            if (begin >= end) return std::string();
            std::vector<int> ids_int;
            ids_int.reserve(static_cast<size_t>(end - begin));
            for (int i = begin; i < end; ++i) {
                const int32_t id = generated_ids[i];
                if (id >= 0 && id < 50257 && !token_is_timestamp(id)) {
                    ids_int.push_back(id);
                }
            }
            if (ids_int.empty()) return std::string();
            return trim_ascii(cm->tok.decode(
                ids_int.data(), static_cast<int>(ids_int.size())));
        };

        const int gn = static_cast<int>(generated_ids.size());
        int segment_offset_frames = seek_num_frames;
        std::vector<std::vector<int32_t>> prev_chunk_segments;

        // Collect indices where generated_ids[i] and [i+1] are both
        // timestamps. slices[k] = i+1 (one past the first ts of the
        // pair), mirroring HF's timestamp_segment_indices.add_(1).
        std::vector<int> slices;
        for (int i = 0; i + 1 < gn; ++i) {
            if (token_is_timestamp(generated_ids[i]) &&
                token_is_timestamp(generated_ids[i + 1]))
            {
                slices.push_back(i + 1);
            }
        }
        const bool single_timestamp_ending =
            gn >= 2 &&
            !token_is_timestamp(generated_ids[gn - 2]) &&
             token_is_timestamp(generated_ids[gn - 1]);

        if (!slices.empty()) {
            if (single_timestamp_ending) {
                slices.push_back(gn);
            } else {
                slices.back() += 1;
            }

            int last_slice = 0;
            for (size_t k = 0; k < slices.size(); ++k) {
                const int current_slice = slices[k];
                const bool is_last = (k + 1 == slices.size());
                if (current_slice <= last_slice) {
                    last_slice = current_slice;
                    continue;
                }
                const int32_t first_tok = generated_ids[last_slice];
                if (!token_is_timestamp(first_tok)) {
                    // Segment doesn't open with a ts (shouldn't happen
                    // under the timestamp-rule logits processor, but
                    // guard anyway).
                    last_slice = current_slice;
                    continue;
                }
                const int end_token_idx =
                    (!is_last || single_timestamp_ending)
                        ? current_slice - 1
                        : current_slice - 2;
                if (end_token_idx < last_slice || end_token_idx >= gn) {
                    last_slice = current_slice;
                    continue;
                }
                const int32_t last_tok = generated_ids[end_token_idx];
                if (!token_is_timestamp(last_tok)) {
                    last_slice = current_slice;
                    continue;
                }
                const int64_t start_ts_pos =
                    static_cast<int64_t>(first_tok - timestamp_begin);
                const int64_t end_ts_pos =
                    static_cast<int64_t>(last_tok - timestamp_begin);

                if (want_segment_timestamps) {
                    transcribe_context::SegmentEntry seg {};
                    seg.t0_ms = time_offset_ms + start_ts_pos * 20;
                    seg.t1_ms = time_offset_ms + end_ts_pos   * 20;
                    if (seg.t1_ms < seg.t0_ms) seg.t1_ms = seg.t0_ms;
                    seg.text  = decode_range(last_slice, current_slice);
                    if (!seg.text.empty()) {
                        cc->segments.push_back(std::move(seg));
                    }
                }
                prev_chunk_segments.emplace_back(
                    generated_ids.begin() + last_slice,
                    generated_ids.begin() + current_slice);

                last_slice = current_slice;
            }

            if (single_timestamp_ending) {
                segment_offset_frames = seek_num_frames;
            } else if (last_slice >= 2 && last_slice - 2 < gn &&
                       token_is_timestamp(generated_ids[last_slice - 2]))
            {
                const int last_ts_pos =
                    generated_ids[last_slice - 2] - timestamp_begin;
                // input_stride = 2 (encoder conv2 downsamples by 2).
                segment_offset_frames = last_ts_pos * 2;
            }
        } else {
            // No consecutive pairs → treat the chunk as one segment.
            // Default end time: seek_num_frames of 10ms each. If any
            // non-first timestamp was emitted, prefer its position.
            int64_t last_ts_pos =
                static_cast<int64_t>(seek_num_frames) / 2;  // frames / (ts_unit / frame_unit) = /2
            for (int i = gn - 1; i >= 0; --i) {
                if (token_is_timestamp(generated_ids[i]) &&
                    generated_ids[i] != timestamp_begin)
                {
                    last_ts_pos =
                        static_cast<int64_t>(generated_ids[i] - timestamp_begin);
                    break;
                }
            }
            if (want_segment_timestamps && gn > 0) {
                transcribe_context::SegmentEntry seg {};
                seg.t0_ms = time_offset_ms;
                seg.t1_ms = time_offset_ms + last_ts_pos * 20;
                seg.text  = decode_range(0, gn);
                if (!seg.text.empty()) {
                    cc->segments.push_back(std::move(seg));
                }
            }
            if (gn > 0) {
                prev_chunk_segments.emplace_back(generated_ids.begin(),
                                                 generated_ids.end());
            }
            segment_offset_frames = seek_num_frames;
        }

        // Stage 3 — update prev-context history for later chunks.
        //
        // HF appends the `segments` returned by _retrieve_segment to
        // current_segments, then _prepare_decoder_input_ids builds the
        // prev window from those segment tokens. In the closed-pair
        // branch, _retrieve_segment intentionally discards unfinished
        // tail tokens after the last completed timestamp pair; carrying
        // accepted_generated_ids directly would leak that discarded
        // tail into the next prompt and diverge from HF.
        if (!no_speech_fired_this_chunk && !prev_chunk_segments.empty()) {
            prev_history_segments.insert(prev_history_segments.end(),
                                         prev_chunk_segments.begin(),
                                         prev_chunk_segments.end());
        }

        all_text_ids.insert(all_text_ids.end(),
                            generated_text_ids.begin(),
                            generated_text_ids.end());

        // Seek advance.
        //
        // Short-form: fixed full-chunk advance. Because we padded PCM
        // to exactly n_samples_per_chunk, the chunk's tail after the
        // real audio is clamped-silence; the decoder would hallucinate
        // "you you you" on it if we re-entered. HF's generate() avoids
        // this by taking a separate is_shortform path that doesn't
        // loop at all — we match by single-passing (a second iteration
        // has nothing useful to decode).
        //
        // Long-form: dynamic advance per HF _retrieve_segment. The
        // chunk's real content may not reach the 3000-frame window
        // end; segment_offset_frames stops at the last closed
        // timestamp pair so the next chunk picks up from there.
        // Guarded against 0-advance infinite loops (fall back to full
        // chunk).
        //
        // No-speech override (Stage 2.5): when both thresholds fire,
        // the chunk is declared empty; skip the whole window so the
        // next chunk decodes fresh encoder content.
        if (is_short_form) {
            seek += seek_num_frames;
        } else if (no_speech_fired_this_chunk) {
            seek += seek_num_frames;
        } else {
            if (segment_offset_frames <= 0) {
                segment_offset_frames = seek_num_frames;
            }
            seek += segment_offset_frames;
        }
    }

    // Finalize: decode all_text_ids → UTF-8, strip whitespace, push the
    // single text-only segment for TIMESTAMPS_NONE mode, flip
    // has_result. Shared with the abort exit paths above so partial
    // output is readable via the same accessors after
    // TRANSCRIBE_ERR_ABORTED.
    commit_result();

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name            = */ "whisper",
    /* .load            = */ whisper_load,
    /* .init_context    = */ whisper_init_context,
    /* .run             = */ whisper_run,
    /* .stream_begin    = */ nullptr,
    /* .stream_feed     = */ nullptr,
    /* .stream_finalize = */ nullptr,
    /* .stream_reset    = */ nullptr,
};

} // namespace transcribe::whisper
