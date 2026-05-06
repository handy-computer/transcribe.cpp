// arch/sensevoice/model.cpp - SenseVoice family handler.
//
// load() validates every weight and binds the backend buffer; run()
// drives the kaldi fbank frontend host-side, builds + computes the
// encoder + CTC graph, and runs greedy CTC decode (argmax →
// unique_consecutive → drop-blank → SP detokenize) to populate the
// public result hierarchy.

#include "sensevoice.h"

#include "encoder.h"
#include "weights.h"

#include "sanm/sanm.h"
#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-kaldi-fbank.h"
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
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

namespace transcribe::sensevoice {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   SenseVoiceModel>);
static_assert(std::is_base_of_v<transcribe_context, SenseVoiceContext>);

SenseVoiceContext::~SenseVoiceContext() {
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

SenseVoiceModel::~SenseVoiceModel() {
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

namespace {

constexpr const char k_default_variant[] = "sensevoice-small";

extern transcribe_status load        (Loader &, const transcribe_model_params *,
                                      transcribe_model **);
extern transcribe_status init_context(transcribe_model *, const transcribe_context_params *,
                                      transcribe_context **);
extern transcribe_status run         (transcribe_context *, const float *, int,
                                      const transcribe_params *);

transcribe_status load(
    Loader &                          loader,
    const transcribe_model_params *   params,
    transcribe_model **               out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<SenseVoiceModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    m->variant = loader.variant().empty() ? k_default_variant
                                          : loader.variant();
    m->backend.clear();

    apply_family_invariants(m->caps);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) return st;
    if (auto st = read_languages_kv (loader.gguf(), *m);       st != TRANSCRIBE_OK) return st;

    if (auto st = m->tok.load(loader.gguf());                  st != TRANSCRIBE_OK) return st;
    if (auto st = read_sensevoice_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) return st;
    m->hparams.vocab_size = static_cast<int32_t>(m->tok.n_tokens());

    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (auto st = build_sensevoice_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(
            backend_req, "sensevoice", m->plan); st != TRANSCRIBE_OK)
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
                     "sensevoice: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "sensevoice");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // Construct the kaldi fbank frontend now that the CMVN tensors
    // are bound to backend memory. SenseVoice uses the shared
    // host-side frontend with apply_cmvn=true; pull the CMVN tensors
    // off backend storage into host buffers first.
    {
        const auto & hp_ = m->hparams;
        transcribe::KaldiFbankParams fe_params;
        fe_params.n_mels          = hp_.fe_num_mels;
        fe_params.sample_rate     = hp_.fe_sample_rate;
        fe_params.win_length      = hp_.fe_win_length;
        fe_params.hop_length      = hp_.fe_hop_length;
        fe_params.lfr_m           = hp_.fe_lfr_m;
        fe_params.lfr_n           = hp_.fe_lfr_n;
        fe_params.d_input         = hp_.enc_d_input;
        fe_params.upscale_samples = hp_.fe_upscale_samples;
        fe_params.apply_cmvn      = true;
        fe_params.cmvn_shift.resize(static_cast<size_t>(hp_.enc_d_input));
        fe_params.cmvn_scale.resize(static_cast<size_t>(hp_.enc_d_input));
        const size_t cmvn_bytes =
            static_cast<size_t>(hp_.enc_d_input) * sizeof(float);
        ggml_backend_tensor_get(m->weights.cmvn_shift,
                                fe_params.cmvn_shift.data(), 0, cmvn_bytes);
        ggml_backend_tensor_get(m->weights.cmvn_scale,
                                fe_params.cmvn_scale.data(), 0, cmvn_bytes);
        m->frontend = std::make_unique<transcribe::KaldiFbankFrontend>(
            std::move(fe_params));
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(
    transcribe_model *                model,
    const transcribe_context_params * params,
    transcribe_context **             out_ctx)
{
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto cc = std::make_unique<SenseVoiceContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Resolve the requested language (or "auto") to the row index in the
// 16-row prefix-embedding table (= lid_dict[language]).
int resolve_lid_idx(const SenseVoiceHParams & hp, const char * lang_or_null) {
    if (lang_or_null == nullptr || lang_or_null[0] == '\0') {
        return hp.prefix_lang_auto;
    }
    if (std::strcmp(lang_or_null, "auto") == 0) return hp.prefix_lang_auto;
    if (std::strcmp(lang_or_null, "zh")   == 0) return hp.prefix_lang_zh;
    if (std::strcmp(lang_or_null, "en")   == 0) return hp.prefix_lang_en;
    if (std::strcmp(lang_or_null, "yue")  == 0) return hp.prefix_lang_yue;
    if (std::strcmp(lang_or_null, "ja")   == 0) return hp.prefix_lang_ja;
    if (std::strcmp(lang_or_null, "ko")   == 0) return hp.prefix_lang_ko;
    if (std::strcmp(lang_or_null, "nospeech") == 0) return hp.prefix_lang_nospeech;
    return hp.prefix_lang_auto;
}

void apply_thread_policy(SenseVoiceContext * cc) {
    int n_threads = cc->n_threads;
    if (n_threads <= 0) {
        n_threads = std::min(8, std::max(1, static_cast<int>(
            std::thread::hardware_concurrency())));
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(cc->sched); ++i) {
        ggml_backend_t be       = ggml_backend_sched_get_backend(cc->sched, i);
        ggml_backend_dev_t dev  = ggml_backend_get_device(be);
        ggml_backend_reg_t reg  = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) continue;
        auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (fn != nullptr) fn(be, n_threads);
    }
}

transcribe_status run(
    transcribe_context *      ctx,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<SenseVoiceContext *>(ctx);
    auto * cm = static_cast<SenseVoiceModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() ||
        cm->frontend == nullptr)
    {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();
    cc->clear_result();

    const auto & hp = cm->hparams;

    cc->t_mel_us    = 0;
    cc->t_encode_us = 0;
    cc->t_decode_us = 0;

    // ---------- Frontend (host-side) ----------------------------------
    const int64_t t_mel_start = ggml_time_us();
    const int T_lfr = cm->frontend->compute(
        pcm, static_cast<size_t>(n_samples), cc->frontend_buf);
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    if (T_lfr <= 0) {
        std::fprintf(stderr,
                     "sensevoice run: input too short for kaldi fbank "
                     "(n_samples=%d → T_lfr=0)\n", n_samples);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int T_full = T_lfr + 4;  // 4 prepended prefix tokens

    // ---------- Reset per-call compute state --------------------------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params init_params {};
        // 70 SAN-M blocks * ~30 ops + a few hundred frontend / PE / CTC
        // ops easily fits in 8 MB of metadata arena.
        init_params.mem_size   = 8 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx = ggml_init(init_params);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr, "sensevoice run: ggml_init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ---------- Build the encoder graph -------------------------------
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx,
                                          cm->weights, hp, T_lfr);
    if (eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---------- Allocate compute tensors via scheduler ---------------
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr,
                         "sensevoice run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "sensevoice run: ggml_backend_sched_alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---------- Upload inputs ----------------------------------------
    // Frontend output tensor: row-major [T_lfr, d_input], byte-identical
    // to ggml ne=[d_input, T_lfr] (d_input is innermost).
    ggml_backend_tensor_set(eb.frontend_in, cc->frontend_buf.data(),
                            0, cc->frontend_buf.size() * sizeof(float));

    // Prefix indices.
    const char * lang = (params != nullptr) ? params->language : nullptr;
    const int32_t lid_idx = resolve_lid_idx(hp, lang);
    const int32_t event_emo[2] = { 1, 2 };  // literal indices in the embed table

    // ITN slot. Library callers flip it via params->sensevoice->use_itn.
    // CLI doesn't expose a flag yet, but the public API does, so any
    // library consumer can drive the textnorm prefix programmatically.
    bool use_itn = false;
    if (params != nullptr && params->sensevoice != nullptr) {
        use_itn = params->sensevoice->use_itn;
    }
    const int32_t textnorm_idx = use_itn ? hp.prefix_withitn : hp.prefix_woitn;

    ggml_backend_tensor_set(eb.lid_idx,       &lid_idx,       0, sizeof(int32_t));
    ggml_backend_tensor_set(eb.event_emo_idx, event_emo,      0, 2 * sizeof(int32_t));
    ggml_backend_tensor_set(eb.textnorm_idx,  &textnorm_idx,  0, sizeof(int32_t));

    // Sinusoidal PE: depth = current width = d_input (NOT d_model).
    transcribe::sanm::build_sinusoidal_pe(cc->pe_buf, hp.enc_d_input, T_full);
    ggml_tensor * pe_in = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(cc->compute_ctx);
         t != nullptr; t = ggml_get_next_tensor(cc->compute_ctx, t))
    {
        if (std::strcmp(t->name, "pe.in") == 0) {
            pe_in = t;
            break;
        }
    }
    if (pe_in == nullptr) {
        std::fprintf(stderr, "sensevoice run: pe.in not found in graph\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_tensor_set(pe_in, cc->pe_buf.data(),
                            0, cc->pe_buf.size() * sizeof(float));

    // Optional dump of the input (matches reference's frontend dump).
    transcribe::debug::dump_tensor("frontend.in", eb.frontend_in,
                                   "frontend.in");

    apply_thread_policy(cc);

    // ---------- Compute -----------------------------------------------
    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "sensevoice run: graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // ---------- Dump intermediates -----------------------------------
    auto try_dump = [](const char * name, ggml_tensor * t,
                       const char * stage)
    {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("frontend.fbank.lfr.cmvn.out", eb.dumps.frontend_out,    "frontend.lfr.cmvn");
    try_dump("enc.prefix.lid_emb",          eb.dumps.prefix_lid,      "encoder.prefix.lid");
    try_dump("enc.prefix.event_emo_emb",    eb.dumps.prefix_event_emo, "encoder.prefix.event_emo");
    try_dump("enc.prefix.textnorm_emb",     eb.dumps.prefix_textnorm, "encoder.prefix.textnorm");
    try_dump("enc.input.with_prefix",       eb.dumps.input_with_prefix, "encoder.input.with_prefix");
    try_dump("enc.embed.out",               eb.dumps.embed_out,       "encoder.embed.pos_added");
    try_dump("enc.encoders0.0.out",         eb.dumps.encoders0_0_out, "encoder.encoders0.0");
    if (eb.dumps.encoders_first != nullptr) {
        try_dump("enc.encoders.0.out",      eb.dumps.encoders_first,  "encoder.encoders.0");
    }
    if (eb.dumps.encoders_mid != nullptr) {
        const char * nm = eb.dumps.encoders_mid->name;
        try_dump(nm, eb.dumps.encoders_mid, "encoder.encoders.mid");
    }
    if (eb.dumps.encoders_last != nullptr) {
        const char * nm = eb.dumps.encoders_last->name;
        try_dump(nm, eb.dumps.encoders_last, "encoder.encoders.last");
    }
    try_dump("enc.after_norm.out",          eb.dumps.after_norm_out,  "encoder.after_norm");
    if (eb.dumps.tp_encoders_first != nullptr) {
        try_dump("enc.tp_encoders.0.out",   eb.dumps.tp_encoders_first, "encoder.tp_encoders.0");
    }
    if (eb.dumps.tp_encoders_mid != nullptr) {
        const char * nm = eb.dumps.tp_encoders_mid->name;
        try_dump(nm, eb.dumps.tp_encoders_mid, "encoder.tp_encoders.mid");
    }
    if (eb.dumps.tp_encoders_last != nullptr) {
        const char * nm = eb.dumps.tp_encoders_last->name;
        try_dump(nm, eb.dumps.tp_encoders_last, "encoder.tp_encoders.last");
    }
    try_dump("enc.tp_norm.out",             eb.dumps.tp_norm_out,     "encoder.tp_norm");
    try_dump("ctc.logits.raw",              eb.dumps.ctc_logits,      "ctc.logits.raw");
    try_dump("ctc.log_probs",               eb.dumps.ctc_log_probs,   "ctc.log_probs");

    // ---------- Read CTC log-probs to host ---------------------------
    const int vocab = hp.vocab_size;
    cc->logits_buf.resize(static_cast<size_t>(T_full) * vocab);
    ggml_backend_tensor_get(eb.out, cc->logits_buf.data(),
                            0, cc->logits_buf.size() * sizeof(float));

    // ---------- Greedy CTC decode -------------------------------------
    const int blank_id = 0;
    cc->token_ids.clear();
    cc->token_ids.reserve(static_cast<size_t>(T_full));
    int prev_id = -1;
    for (int t = 0; t < T_full; ++t) {
        const float * row = cc->logits_buf.data() +
                            static_cast<size_t>(t) * vocab;
        int   best_id = 0;
        float best    = row[0];
        for (int v = 1; v < vocab; ++v) {
            if (row[v] > best) {
                best    = row[v];
                best_id = v;
            }
        }
        if (best_id != prev_id) {
            if (best_id != blank_id) {
                cc->token_ids.push_back(best_id);
            }
            prev_id = best_id;
        }
    }

    // ---------- Build the public result ------------------------------
    if (!cc->token_ids.empty()) {
        const transcribe::Tokenizer & tok = cm->tok;

        // Per-token text fragments (for the public token table). These
        // ALWAYS carry the raw piece — the per-token accessors expose
        // the unfiltered token stream so library callers can observe
        // language/event/emotion/itn control tokens regardless of
        // strip_special_tags.
        cc->tokens.reserve(cc->token_ids.size());
        for (int id : cc->token_ids) {
            transcribe_context::TokenEntry te;
            te.id    = id;
            te.text  = tok.decode(&id, 1);
            te.t0_ms = 0;
            te.t1_ms = 0;
            cc->tokens.push_back(std::move(te));
        }

        // Decode the full sequence into the segment / full_text fields.
        // The decode here may filter out CONTROL-typed tokens depending
        // on strip_special_tags so the user-facing text is clean by
        // default but the raw stream stays accessible above.
        const bool strip = (params == nullptr) ? true : params->strip_special_tags;
        std::vector<int> ids_for_text;
        if (strip) {
            ids_for_text.reserve(cc->token_ids.size());
            for (int id : cc->token_ids) {
                if (!tok.is_control(id)) {
                    ids_for_text.push_back(id);
                }
            }
        } else {
            ids_for_text.assign(cc->token_ids.begin(), cc->token_ids.end());
        }
        std::string full = ids_for_text.empty()
            ? std::string()
            : tok.decode(ids_for_text.data(),
                         static_cast<int>(ids_for_text.size()));
        if (!full.empty() && full.front() == ' ') {
            full.erase(full.begin());
        }

        transcribe_context::SegmentEntry seg;
        seg.t0_ms = 0;
        seg.t1_ms = static_cast<int64_t>(
            std::llround(1000.0 * static_cast<double>(n_samples) /
                         static_cast<double>(hp.fe_sample_rate)));
        seg.first_token = 0;
        seg.n_tokens    = static_cast<int>(cc->tokens.size());
        seg.first_word  = 0;
        seg.n_words     = 0;
        seg.text        = full;

        cc->full_text  = std::move(full);
        cc->segments.push_back(std::move(seg));

        // Family default: NONE. Caller-requested kinds finer than NONE
        // are clamped per the dispatcher's contract.
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        cc->has_result  = true;
    }

    cc->t_decode_us = 0;
    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name         = */ "sensevoice",
    /* .load         = */ load,
    /* .init_context = */ init_context,
    /* .run          = */ run,
};

} // namespace transcribe::sensevoice
