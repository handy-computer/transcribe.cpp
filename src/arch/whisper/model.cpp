// arch/whisper/model.cpp - Whisper ASR family handler.
//
// STAGE-4 BRINGUP. Encoder + decoder prefill run through C++; mel
// frontend still uses the reference dump injected via the
// TRANSCRIBE_WHISPER_MEL_FROM_REF env var. The decoder runs a single
// prefill pass over the canonical 4-token multilingual prompt
// `<|sot|> <|en|> <|transcribe|> <|notimestamps|>` so the dec.* tensor
// dumps line up byte-for-byte with the transformers reference. Real
// autoregressive transcription (and a C++ mel frontend) are follow-ups.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
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
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

// ---------------------------------------------------------------------------
// KV cache initialization (decoder). Declared in whisper.h but not
// exercised by the stub run(); the definition lives here so decoder
// bringup (follow-up) can call it directly.
// ---------------------------------------------------------------------------

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

    const int64_t self_elements  = static_cast<int64_t>(d_model) * n_layer * n_ctx;
    const int64_t cross_elements = static_cast<int64_t>(d_model) * n_layer * T_enc;

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

    cache.n_ctx  = n_ctx;
    cache.T_enc  = T_enc;
    cache.n      = 0;
    cache.head   = 0;
    cache.cross_populated = false;

    return true;
}

namespace {

constexpr const char k_default_variant[] = "whisper";

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

    // Mel frontend. Whisper-style 80-bin log-mel at 16 kHz, with
    // per-utterance log10 + clamp(max-8) + (+4)/4 normalization,
    // periodic Hann window, and reflect padding around the centered
    // STFT. Filterbank + window come baked in the GGUF (the converter
    // serializes them verbatim from preprocessor_config.json) so the
    // C++ frontend matches the WhisperFeatureExtractor reference
    // bit-for-bit modulo fp32 reduction-order drift.
    {
        transcribe::MelConfig cfg {};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;        // "reflect"
        cfg.window_type  = m->hparams.fe_window;          // "hann_periodic"
        // Map whisper's "whisper_logmel" tag to MelFrontend's
        // "per_utterance" mode, which implements exactly the
        // log10 → clamp(max-8) → (+4)/4 sequence. Other normalize
        // strings would be a converter bug — guard at load.
        if (m->hparams.fe_normalize == "whisper_logmel" ||
            m->hparams.fe_normalize == "per_utterance")
        {
            cfg.normalize = "per_utterance";
        } else {
            std::fprintf(stderr,
                         "whisper: unsupported fe_normalize='%s'\n",
                         m->hparams.fe_normalize.c_str());
            return TRANSCRIBE_ERR_GGUF;
        }

        // Filterbank + window from the GGUF (frontend.mel_filterbank,
        // frontend.window). Both are F32 arrays whose shapes match
        // what MelConfig wants (row-major [num_mels, n_fft/2+1] and
        // [n_fft] respectively). When present, MelFrontend uses them
        // verbatim instead of reconstructing from cfg constants.
        using R = transcribe::load_common::ReadF32Result;
        const size_t fb_elems =
            static_cast<size_t>(cfg.num_mels) *
            static_cast<size_t>(cfg.n_fft / 2 + 1);
        const auto fb_rc = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(),
            "frontend.mel_filterbank", fb_elems,
            "whisper", cfg.filterbank);
        if (fb_rc != R::Ok && fb_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t win_elems = static_cast<size_t>(cfg.n_fft);
        const auto win_rc = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(),
            "frontend.window", win_elems,
            "whisper", cfg.window);
        if (win_rc != R::Ok && win_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }

        m->mel.emplace(cfg);
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
            gguf_data, m->ctx_meta, m->hparams, m->weights);
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
    m->lang_codes.clear();
    m->lang_token_ids.clear();
    if (m->caps.languages != nullptr && m->caps.n_languages > 0) {
        m->lang_codes.reserve(static_cast<size_t>(m->caps.n_languages));
        m->lang_token_ids.reserve(static_cast<size_t>(m->caps.n_languages));
        for (int i = 0; i < m->caps.n_languages; ++i) {
            const char * code = m->caps.languages[i];
            if (code == nullptr || code[0] == '\0') continue;
            const std::string piece = std::string("<|") + code + "|>";
            const int id = m->tok.find(piece);
            if (id < 0) {
                std::fprintf(stderr,
                             "whisper: language '%s' has no '%s' token "
                             "in tokenizer vocab\n",
                             code, piece.c_str());
                return TRANSCRIBE_ERR_GGUF;
            }
            m->lang_codes.emplace_back(code);
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
// run  (STUB — encoder-only)
// ---------------------------------------------------------------------------
//
// Path for the stub:
//   1. Obtain the mel spectrogram. The C++ mel frontend is not yet
//      written; for bringup we read the reference dump from a
//      directory the caller names in TRANSCRIBE_WHISPER_MEL_FROM_REF.
//      If the env var is unset we refuse to run and return
//      NOT_IMPLEMENTED so the failure mode is unambiguous.
//   2. Build + compute the encoder graph.
//   3. Dump every named encoder tensor so validate.py can diff
//      against the reference dumps.
//   4. Return an empty transcript. Decoder is a follow-up.

namespace {

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
    (void)params;
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<WhisperContext *>(ctx);
    auto * cm = static_cast<WhisperModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    transcribe::debug::init();

    // ----- Mel frontend ------------------------------------------------
    //
    // Two paths:
    //
    //   1. TRANSCRIBE_WHISPER_MEL_FROM_REF=<dir>  — read the reference
    //      mel dump from disk. Used by validate.py so the encoder can
    //      be diffed against the reference WITHOUT introducing C++ mel
    //      drift into the comparison. Requires <dir>/enc.mel.in.f32 in
    //      [n_mels, n_mel_frames] row-major layout.
    //
    //   2. C++ MelFrontend (default) — pad-or-trim PCM to exactly
    //      30 s = fe_n_samples (480000), then call MelFrontend::compute
    //      which produces n_mels × fe_nb_max_frames (= 80 × 3000) for
    //      whisper-tiny, matching the WhisperFeatureExtractor contract.
    const int n_mels       = cm->hparams.enc_num_mel_bins;
    const int n_mel_frames = cm->hparams.fe_nb_max_frames > 0
                                ? cm->hparams.fe_nb_max_frames : 3000;
    const int64_t t_mel_start = ggml_time_us();
    if (const char * ref_dir = std::getenv("TRANSCRIBE_WHISPER_MEL_FROM_REF");
        ref_dir != nullptr && ref_dir[0] != '\0')
    {
        if (const transcribe_status st = load_mel_from_ref(
                ref_dir, n_mels, n_mel_frames, cc->mel_buf);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
    } else {
        if (!cm->mel.has_value()) {
            std::fprintf(stderr,
                         "whisper run: model has no MelFrontend "
                         "(load skipped?)\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        // Pad-or-trim to exactly 30 s = fe_n_samples (480000 for the
        // shipped whisper variants). Long-form chunked decoding is a
        // follow-up; first port handles single-chunk audio only.
        const int n_target = cm->hparams.fe_n_samples > 0
                                 ? cm->hparams.fe_n_samples : 480000;
        std::vector<float> pcm_padded(static_cast<size_t>(n_target), 0.0f);
        const int n_copy = std::min(n_samples, n_target);
        std::memcpy(pcm_padded.data(), pcm,
                    static_cast<size_t>(n_copy) * sizeof(float));

        int mel_n_mels = 0, mel_n_frames = 0;
        std::vector<float> mel_mn;  // [n_mels, n_frames] (MelFrontend layout)
        if (const transcribe_status mst = cm->mel->compute(
                pcm_padded.data(), pcm_padded.size(),
                mel_mn, mel_n_mels, mel_n_frames);
            mst != TRANSCRIBE_OK)
        {
            std::fprintf(stderr,
                         "whisper run: MelFrontend::compute failed (%d)\n",
                         static_cast<int>(mst));
            return mst;
        }
        if (mel_n_mels != n_mels || mel_n_frames != n_mel_frames) {
            std::fprintf(stderr,
                         "whisper run: mel shape %dx%d != expected %dx%d\n",
                         mel_n_mels, mel_n_frames, n_mels, n_mel_frames);
            return TRANSCRIBE_ERR_GGUF;
        }

        // Transpose to encoder/reference layout [n_frames, n_mels]
        // (n_mels innermost, matching the WhisperFeatureExtractor dump
        // and the encoder's ne=[n_mels, n_mel_frames] tensor).
        cc->mel_buf.resize(static_cast<size_t>(mel_n_mels) *
                           static_cast<size_t>(mel_n_frames));
        for (int t = 0; t < mel_n_frames; ++t) {
            for (int m = 0; m < mel_n_mels; ++m) {
                cc->mel_buf[static_cast<size_t>(t) * mel_n_mels + m] =
                    mel_mn[static_cast<size_t>(m) * mel_n_frames + t];
            }
        }
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // ----- Reset per-call compute state -------------------------------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }

    // ----- Build encoder graph ----------------------------------------
    {
        ggml_init_params init_params {};
        init_params.mem_size   = 8 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx = ggml_init(init_params);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr,
                         "whisper run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, cm->hparams, n_mel_frames,
        cc->encoder_use_flash, cm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Allocate + compute encoder graph ---------------------------
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
                     "whisper run: ggml_backend_sched_alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload mel. Host layout: row-major (T, n_mels) = same bytes as
    // ggml ne=[n_mels, T] with n_mels innermost. Size check: assign
    // above already sized exactly n_mels * n_mel_frames.
    const size_t mel_bytes = static_cast<size_t>(n_mels) *
                             static_cast<size_t>(n_mel_frames) *
                             sizeof(float);
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(), 0, mel_bytes);

    // Compute.
    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "whisper run: encoder graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // ----- Dump encoder intermediates ---------------------------------
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

    // ----- Stash encoder output to host -------------------------------
    //
    // Materialize to host so the decoder pass below can operate from a
    // fresh compute_ctx (we ggml_free the encoder ctx before building
    // the decoder graph, which would dangle eb.out otherwise).
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc_local = static_cast<int>(eb.out->ne[1]);
    cc->enc_T = T_enc_local;
    cc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc_local));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ----- Decoder prefill --------------------------------------------
    //
    // Build the canonical multilingual prompt that the Stage-2 reference
    // dump used:  <|sot|> <|en|> <|transcribe|> <|notimestamps|>. This
    // keeps the dec.* dumps byte-comparable to the reference for Stage-4
    // numerical validation. Real transcription (variable language, real
    // audio language detection, autoregressive loop) is a follow-up that
    // will reuse the same decoder graph.
    // Resolve the language code: caller-provided params->language wins;
    // fall back to "en" so existing tests that don't pass a hint keep
    // their behavior. Whisper's automatic language-detection path
    // (run the decoder for one step over the SOT-only prompt and pick
    // argmax over the language tokens) is a follow-up.
    std::string lang_code = "en";
    if (params != nullptr && params->language != nullptr &&
        params->language[0] != '\0')
    {
        lang_code = params->language;
    }
    int32_t lang_token = -1;
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

    // Task: transcribe by default; translate if caller asked for it.
    const int32_t task_token =
        (params != nullptr && params->task == TRANSCRIBE_TASK_TRANSLATE)
            ? cm->hparams.translate_token_id
            : cm->hparams.transcribe_token_id;

    std::vector<int32_t> prompt_ids;
    prompt_ids.reserve(4);
    prompt_ids.push_back(cm->hparams.decoder_start_token_id);
    prompt_ids.push_back(lang_token);
    prompt_ids.push_back(task_token);
    prompt_ids.push_back(cm->hparams.no_timestamps_token_id);
    const int seq_len = static_cast<int>(prompt_ids.size());

    // Pick decode strategy. Default: KV-cached (prompt pass writes cache,
    // step pass rewrites only the last slot). Escape hatch: set
    // TRANSCRIBE_WHISPER_NO_KV=1 to force the non-cached prefill loop
    // (keeps every step's graph shape identical to the validate.py dump
    // path; useful when debugging KV-specific bugs).
    const bool use_kv = [] {
        const char * s = std::getenv("TRANSCRIBE_WHISPER_NO_KV");
        return !(s != nullptr && s[0] != '\0' && s[0] != '0');
    }();

    const int64_t n_ctx_decoder = cm->hparams.dec_max_target_positions;
    const int64_t vocab_size    = cm->hparams.dec_vocab_size;
    const int     eos_id        = cm->tok.eos_id() >= 0 ? cm->tok.eos_id() : 50257;
    constexpr int k_max_new_tokens = 256;

    std::vector<float>   last_logits(static_cast<size_t>(vocab_size));
    std::vector<int32_t> generated_text_ids;
    generated_text_ids.reserve(64);

    auto suppress_in_place = [&](std::vector<float> & logits) {
        for (int32_t id : cm->hparams.suppress_tokens) {
            if (id >= 0 && id < vocab_size) {
                logits[static_cast<size_t>(id)] = -INFINITY;
            }
        }
    };
    auto argmax_v = [&](const std::vector<float> & logits) {
        int best_id = 0;
        float best  = logits[0];
        for (int i = 1; i < static_cast<int>(vocab_size); ++i) {
            if (logits[i] > best) {
                best = logits[i];
                best_id = i;
            }
        }
        return best_id;
    };
    auto new_compute_ctx = [&](size_t mem) -> bool {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        ggml_init_params p {};
        p.mem_size   = mem;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        cc->compute_ctx = ggml_init(p);
        return cc->compute_ctx != nullptr;
    };

    const int64_t t_dec_start = ggml_time_us();

    int next_id = 0;

    if (use_kv) {
        // -------------------------------------------------------------
        // KV-cached path.
        // -------------------------------------------------------------

        // Allocate / reset cache tensors. Held on the same backend as
        // the model's primary scheduler — the cache tensors are views
        // into a backend buffer, not host memory.
        const int n_layers = cm->hparams.dec_n_layers;
        const bool need_init =
            cc->kv_cache.ctx == nullptr ||
            cc->kv_cache.n_ctx != n_ctx_decoder ||
            cc->kv_cache.T_enc != T_enc_local;
        if (need_init) {
            cc->kv_cache.free();
            // AUTO: match cache dtype to the model's weight dtype. For
            // F32 GGUFs that's F32 (zero-loss cache, matches the
            // non-cached numerical story); for F16 and every quant
            // preset it's F16 (the production default — weight+activation
            // already pay F16 downcast costs, so the cache doesn't add
            // new precision loss). Explicit --kv-type overrides.
            //
            // Detection uses a representative Linear tensor (decoder
            // self-attention q_w); every Linear weight in a given preset
            // shares the same ggml_type, so this is a reliable signal.
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
        } else {
            cc->kv_cache.n = 0;
            cc->kv_cache.head = 0;
            cc->kv_cache.cross_populated = false;
        }

        // ---- Cross-attention K/V precompute ---------------------------
        {
            if (!new_compute_ctx(8 * 1024 * 1024)) {
                std::fprintf(stderr,
                             "whisper run: ggml_init for cross_kv failed\n");
                return TRANSCRIBE_ERR_GGUF;
            }
            DecoderBuild cross_db = build_cross_kv_graph(
                cc->compute_ctx, cm->weights, cm->hparams,
                cc->kv_cache, T_enc_local);
            if (cross_db.graph == nullptr) {
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
                std::fprintf(stderr,
                             "whisper run: alloc_graph failed (cross_kv)\n");
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_tensor_set(cross_db.encoder_out_in,
                                    cc->enc_host.data(), 0,
                                    cc->enc_host.size() * sizeof(float));
            if (const ggml_status gs = ggml_backend_sched_graph_compute(
                    cc->sched, cross_db.graph);
                gs != GGML_STATUS_SUCCESS)
            {
                std::fprintf(stderr,
                             "whisper run: cross_kv compute failed (%d)\n",
                             static_cast<int>(gs));
                return TRANSCRIBE_ERR_GGUF;
            }
            cc->kv_cache.cross_populated = true;
        }

        // ---- Prompt pass (emit dumps, n_past=0) -----------------------
        {
            if (!new_compute_ctx(16 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
            DecoderBuild db = build_decoder_graph_kv(
                cc->compute_ctx, cm->weights, cm->hparams,
                cc->kv_cache,
                /*n_tokens=*/seq_len, /*n_past=*/0, T_enc_local,
                /*skip_log_softmax=*/false, cc->decoder_use_flash);
            if (db.out == nullptr || db.graph == nullptr) {
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
                std::fprintf(stderr,
                             "whisper run: alloc_graph failed (prompt)\n");
                return TRANSCRIBE_ERR_GGUF;
            }

            // Upload prompt + position ids [0, 1, ..., seq_len-1].
            ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(),
                                    0, prompt_ids.size() * sizeof(int32_t));
            std::vector<int32_t> pos_ids(seq_len);
            for (int i = 0; i < seq_len; ++i) pos_ids[i] = i;
            ggml_tensor * pos_in = ggml_graph_get_tensor(db.graph, "dec.pos_ids");
            ggml_backend_tensor_set(pos_in, pos_ids.data(),
                                    0, pos_ids.size() * sizeof(int32_t));

            if (db.causal_mask_in != nullptr) {
                std::vector<float> mask(static_cast<size_t>(seq_len) * seq_len);
                for (int q = 0; q < seq_len; ++q) {
                    for (int k = 0; k < seq_len; ++k) {
                        mask[static_cast<size_t>(q) * seq_len + k] =
                            (k <= q) ? 0.0f : -1e9f;
                    }
                }
                ggml_backend_tensor_set(db.causal_mask_in, mask.data(),
                                        0, mask.size() * sizeof(float));
            }

            if (const ggml_status gs = ggml_backend_sched_graph_compute(
                    cc->sched, db.graph);
                gs != GGML_STATUS_SUCCESS)
            {
                return TRANSCRIBE_ERR_GGUF;
            }
            cc->kv_cache.n    = seq_len;
            cc->kv_cache.head = seq_len;

            try_dump("dec.token_emb",       db.dumps.token_emb,       "decoder.embedding");
            try_dump("dec.pos_emb",         db.dumps.pos_emb,         "decoder.position_embedding");
            try_dump("dec.embed_sum",       db.dumps.embed_sum,       "decoder.embed_sum");
            for (size_t i = 0; i < db.dumps.block_outs.size(); ++i) {
                char bname[64], stage[64];
                std::snprintf(bname, sizeof(bname), "dec.block.%zu.out", i);
                std::snprintf(stage, sizeof(stage), "decoder.block%zu.out", i);
                try_dump(bname, db.dumps.block_outs[i], stage);
            }
            try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.output_before_head");
            try_dump("dec.logits_raw",      db.dumps.logits_raw,      "decoder.logits_raw");
            try_dump("dec.logits",          db.dumps.logits,          "decoder.logits");

            // Last-row logits → first generated token.
            const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
            ggml_backend_tensor_get(db.dumps.logits_raw, last_logits.data(),
                                    row_bytes * static_cast<size_t>(seq_len - 1),
                                    row_bytes);
            suppress_in_place(last_logits);
            for (int32_t id : cm->hparams.begin_suppress_tokens) {
                if (id >= 0 && id < vocab_size) {
                    last_logits[static_cast<size_t>(id)] = -INFINITY;
                }
            }
            next_id = argmax_v(last_logits);
        }

        // ---- Step loop (n_tokens=1) -----------------------------------
        int n_past = seq_len;
        for (int step = 0; step < k_max_new_tokens; ++step) {
            if (next_id == eos_id) break;
            if (next_id < 50257) generated_text_ids.push_back(next_id);

            if (n_past + 1 > static_cast<int>(n_ctx_decoder)) break;

            if (!new_compute_ctx(4 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
            DecoderBuild step_db = build_decoder_graph_kv(
                cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                /*n_tokens=*/1, /*n_past=*/n_past, T_enc_local,
                /*skip_log_softmax=*/true, cc->decoder_use_flash);
            if (step_db.out == nullptr) return TRANSCRIBE_ERR_GGUF;

            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, step_db.graph)) {
                return TRANSCRIBE_ERR_GGUF;
            }

            int32_t tok = next_id;
            int32_t pos = n_past;
            ggml_backend_tensor_set(step_db.token_ids_in, &tok, 0, sizeof(int32_t));
            ggml_tensor * pos_in = ggml_graph_get_tensor(step_db.graph, "dec.pos_ids");
            ggml_backend_tensor_set(pos_in, &pos, 0, sizeof(int32_t));

            if (const ggml_status gs = ggml_backend_sched_graph_compute(
                    cc->sched, step_db.graph);
                gs != GGML_STATUS_SUCCESS)
            {
                return TRANSCRIBE_ERR_GGUF;
            }

            n_past += 1;
            cc->kv_cache.n    = n_past;
            cc->kv_cache.head = n_past;

            // Mid-generation dump: capture the logits from the 20th
            // autoregressive step (0-indexed: step==19). This exercises
            // the n_past>0 KV-cache read/write path which the prompt-
            // pass dump cannot reach. Matches a reference dump named
            // dec.logits_raw.gen20 (input = prompt + generated[0..19],
            // extracts logits at the last position).
            if (step == 19) {
                try_dump("dec.logits_raw.gen20", step_db.out,
                         "decoder.logits_raw.gen20");
            }

            // step graph emits pre-softmax logits (argmax-invariant).
            const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
            ggml_backend_tensor_get(step_db.out, last_logits.data(), 0, row_bytes);
            suppress_in_place(last_logits);
            next_id = argmax_v(last_logits);
        }
    } else {
        // -------------------------------------------------------------
        // Non-cached path (escape hatch, TRANSCRIBE_WHISPER_NO_KV=1).
        // Preserved as a correctness oracle: its graph shape per step
        // matches validate.py's prefill dumps byte-for-byte.
        // -------------------------------------------------------------
        if (!new_compute_ctx(16 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
        DecoderBuild db = build_decoder_prefill_graph(
            cc->compute_ctx, cm->weights, cm->hparams,
            seq_len, T_enc_local, cc->decoder_use_flash);
        if (db.out == nullptr || db.graph == nullptr) return TRANSCRIBE_ERR_GGUF;

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) return TRANSCRIBE_ERR_GGUF;

        ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(),
                                0, prompt_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(db.encoder_out_in, cc->enc_host.data(),
                                0, cc->enc_host.size() * sizeof(float));
        if (db.causal_mask_in != nullptr) {
            std::vector<float> mask(static_cast<size_t>(seq_len) * seq_len);
            for (int q = 0; q < seq_len; ++q) {
                for (int k = 0; k < seq_len; ++k) {
                    mask[static_cast<size_t>(q) * seq_len + k] =
                        (k <= q) ? 0.0f : -1e9f;
                }
            }
            ggml_backend_tensor_set(db.causal_mask_in, mask.data(),
                                    0, mask.size() * sizeof(float));
        }
        if (ggml_backend_sched_graph_compute(cc->sched, db.graph)
            != GGML_STATUS_SUCCESS) return TRANSCRIBE_ERR_GGUF;

        try_dump("dec.token_emb",       db.dumps.token_emb,       "decoder.embedding");
        try_dump("dec.pos_emb",         db.dumps.pos_emb,         "decoder.position_embedding");
        try_dump("dec.embed_sum",       db.dumps.embed_sum,       "decoder.embed_sum");
        for (size_t i = 0; i < db.dumps.block_outs.size(); ++i) {
            char bname[64], stage[64];
            std::snprintf(bname, sizeof(bname), "dec.block.%zu.out", i);
            std::snprintf(stage, sizeof(stage), "decoder.block%zu.out", i);
            try_dump(bname, db.dumps.block_outs[i], stage);
        }
        try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.output_before_head");
        try_dump("dec.logits_raw",      db.dumps.logits_raw,      "decoder.logits_raw");
        try_dump("dec.logits",          db.dumps.logits,          "decoder.logits");

        auto read_last = [&](DecoderBuild & build, int s_len) {
            const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
            ggml_backend_tensor_get(build.dumps.logits_raw, last_logits.data(),
                                    row_bytes * static_cast<size_t>(s_len - 1),
                                    row_bytes);
        };
        read_last(db, seq_len);
        suppress_in_place(last_logits);
        for (int32_t id : cm->hparams.begin_suppress_tokens) {
            if (id >= 0 && id < vocab_size) {
                last_logits[static_cast<size_t>(id)] = -INFINITY;
            }
        }
        next_id = argmax_v(last_logits);

        for (int step = 0; step < k_max_new_tokens; ++step) {
            if (next_id == eos_id) break;
            if (next_id < 50257) generated_text_ids.push_back(next_id);
            prompt_ids.push_back(next_id);

            const int s_len = static_cast<int>(prompt_ids.size());
            if (s_len > static_cast<int>(n_ctx_decoder)) break;

            if (!new_compute_ctx(16 * 1024 * 1024)) return TRANSCRIBE_ERR_GGUF;
            db = build_decoder_prefill_graph(
                cc->compute_ctx, cm->weights, cm->hparams,
                s_len, T_enc_local, cc->decoder_use_flash);
            if (db.out == nullptr) return TRANSCRIBE_ERR_GGUF;

            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) return TRANSCRIBE_ERR_GGUF;

            ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(),
                                    0, prompt_ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(db.encoder_out_in, cc->enc_host.data(),
                                    0, cc->enc_host.size() * sizeof(float));
            if (db.causal_mask_in != nullptr) {
                std::vector<float> mask(static_cast<size_t>(s_len) * s_len);
                for (int q = 0; q < s_len; ++q) {
                    for (int k = 0; k < s_len; ++k) {
                        mask[static_cast<size_t>(q) * s_len + k] =
                            (k <= q) ? 0.0f : -1e9f;
                    }
                }
                ggml_backend_tensor_set(db.causal_mask_in, mask.data(),
                                        0, mask.size() * sizeof(float));
            }
            if (ggml_backend_sched_graph_compute(cc->sched, db.graph)
                != GGML_STATUS_SUCCESS) return TRANSCRIBE_ERR_GGUF;

            read_last(db, s_len);
            suppress_in_place(last_logits);
            next_id = argmax_v(last_logits);
        }
    }

    cc->t_decode_us = ggml_time_us() - t_dec_start;

    // Decode generated text tokens to UTF-8 (gpt2 byte-level path) and
    // strip leading/trailing whitespace to match the reference's
    // tokenizer.decode(skip_special_tokens=True).strip().
    std::string text;
    if (!generated_text_ids.empty()) {
        std::vector<int> ids_int(generated_text_ids.begin(),
                                 generated_text_ids.end());
        text = cm->tok.decode(ids_int.data(),
                              static_cast<int>(ids_int.size()));
        // Trim ASCII whitespace.
        size_t start = 0;
        while (start < text.size() &&
               (text[start] == ' ' || text[start] == '\t' ||
                text[start] == '\n' || text[start] == '\r')) {
            ++start;
        }
        size_t end = text.size();
        while (end > start &&
               (text[end - 1] == ' ' || text[end - 1] == '\t' ||
                text[end - 1] == '\n' || text[end - 1] == '\r')) {
            --end;
        }
        text = text.substr(start, end - start);
    }

    cc->clear_result();
    transcribe_context::SegmentEntry seg;
    seg.t0_ms       = 0;
    seg.t1_ms       = 0;
    seg.first_token = 0;
    seg.n_tokens    = 0;
    seg.first_word  = 0;
    seg.n_words     = 0;
    seg.text        = text;
    cc->segments.push_back(std::move(seg));
    cc->full_text   = std::move(text);
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name         = */ "whisper",
    /* .load         = */ whisper_load,
    /* .init_context = */ whisper_init_context,
    /* .run          = */ whisper_run,
};

} // namespace transcribe::whisper
