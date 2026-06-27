// arch/vibevoice/model.cpp - VibeVoice-ASR family handler.
//
// Phase A: load() is real (reads GGUF KV, tokenizer, hparams, builds the
// tensor catalog, streams weights) so conversion round-trips are verifiable
// before inference lands. init_context() allocates a bare session; run()
// returns TRANSCRIBE_ERR_NOT_IMPLEMENTED until the VAE encoder (Phase B) and
// Qwen2.5 LM (Phase C) graph builders exist.

#include "vibevoice.h"

#include "transcribe-arch.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace transcribe::vibevoice {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   VibeVoiceModel>);
static_assert(std::is_base_of_v<transcribe_session, VibeVoiceSession>);

namespace {
constexpr const char * k_default_variant = "vibevoice-asr";
}

VibeVoiceSession::~VibeVoiceSession() {
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

VibeVoiceModel::~VibeVoiceModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
}

transcribe_status load(
    Loader &                              loader,
    const transcribe_model_load_params *  params,
    transcribe_model **                   out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<VibeVoiceModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps);
        st != TRANSCRIBE_OK) return st;
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m);
        st != TRANSCRIBE_OK) return st;

    // Tokenizer (byte-level BPE; loader handles the "gpt2" model tag).
    if (const transcribe_status st = m->tok.load(loader.gguf());
        st != TRANSCRIBE_OK) return st;

    // Chat template (stored now; required at decode time).
    (void)read_optional_string_kv(
        loader.gguf(), "tokenizer.chat_template", "vibevoice", "", m->chat_template);

    // Hparams.
    if (const transcribe_status st = read_vibevoice_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK) return st;

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "vibevoice: tokenizer vocab (%d) != decoder vocab_size (%d)",
                m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "vibevoice: GGUF tokenizer has no eos_token_id");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Reopen with no_alloc to build the tensor catalog.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_vibevoice_weights(
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
            backend_req, (params != nullptr) ? params->gpu_device : 0,
            "vibevoice", m->plan);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "vibevoice: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "vibevoice");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(
    transcribe_model *                model,
    const transcribe_session_params * params,
    transcribe_session **             out_ctx)
{
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc = std::make_unique<VibeVoiceSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;
    cc->n_ctx     = transcribe_session_params_n_ctx(params);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

transcribe_status run(
    transcribe_session *          ctx,
    const float *                 pcm,
    int                           n_samples,
    const transcribe_run_params * params)
{
    (void)ctx;
    (void)pcm;
    (void)n_samples;
    (void)params;
    // Phase B (VAE encoder) and Phase C (Qwen2.5 LM) land here.
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}

extern const Arch arch = {
    /* .name             = */ "vibevoice",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::vibevoice
