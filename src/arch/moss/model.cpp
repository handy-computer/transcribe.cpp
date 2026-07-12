// arch/moss/model.cpp - MOSS-Transcribe-Diarize family handler.
//
// audio-LLM: Whisper-Medium encoder + 4x time merge + VQAdaptor + Qwen3-0.6B
// causal LM with non-contiguous audio-token injection. Encoder is per-chunk
// (30s Whisper chunks concatenated before the merge); decode reuses the shared
// causal_lm Qwen3 driver (see arch/qwen3_asr for the sibling design).

#include "causal_lm/causal_lm.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "moss.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace transcribe::moss {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, MossModel>);
static_assert(std::is_base_of_v<transcribe_session, MossSession>);

MossSession::~MossSession() {
    kv_cache.free();
    kv_cache_batch.free();
    if (sched != nullptr) {
        safe_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

MossModel::~MossModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    packed_gate_up.free();
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

// ---------------------------------------------------------------------------
// Prompt construction (declared in moss.h)
// ---------------------------------------------------------------------------

void build_audio_span(const MossHParams &    hp,
                      int                    audio_seq_len,
                      std::vector<int32_t> & out_span_ids,
                      std::vector<int32_t> & out_audio_offsets) {
    out_span_ids.clear();
    out_audio_offsets.clear();
    const int32_t audio_id = hp.audio_token_id;

    auto push_audio = [&](int count) {
        for (int i = 0; i < count; ++i) {
            out_audio_offsets.push_back(static_cast<int32_t>(out_span_ids.size()));
            out_span_ids.push_back(audio_id);
        }
    };

    if (!hp.enable_time_marker || audio_seq_len <= 0 || hp.time_marker_every_seconds <= 0) {
        push_audio(std::max(audio_seq_len, 0));
        return;
    }
    const int tokens_per_marker = static_cast<int>(hp.audio_tokens_per_second * hp.time_marker_every_seconds);
    if (tokens_per_marker <= 0) {
        push_audio(audio_seq_len);
        return;
    }
    const int tme      = hp.time_marker_every_seconds;
    const int duration = static_cast<int>(audio_seq_len / hp.audio_tokens_per_second);
    int       consumed = 0;
    for (int sec = tme; sec <= duration; sec += tme) {
        const int pos = (sec / tme) * tokens_per_marker;
        const int seg = pos - consumed;
        if (seg > 0) {
            push_audio(seg);
            consumed += seg;
        }
        // Marker: decimal digits of `sec`, each a single baked digit token.
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", sec);
        for (const char * p = buf; *p != '\0'; ++p) {
            out_span_ids.push_back(hp.digit_tokens[static_cast<size_t>(*p - '0')]);
        }
    }
    const int remainder = audio_seq_len - consumed;
    if (remainder > 0) {
        push_audio(remainder);
    }
}

void build_prompt_tokens(const MossHParams &    hp,
                         int                    audio_seq_len,
                         std::vector<int32_t> & out_ids,
                         std::vector<int32_t> & out_audio_positions) {
    out_ids.clear();
    out_audio_positions.clear();

    out_ids.insert(out_ids.end(), hp.prompt_prefix_tokens.begin(), hp.prompt_prefix_tokens.end());
    const int prefix_len = static_cast<int>(out_ids.size());

    std::vector<int32_t> span_ids;
    std::vector<int32_t> span_offsets;
    build_audio_span(hp, audio_seq_len, span_ids, span_offsets);
    out_ids.insert(out_ids.end(), span_ids.begin(), span_ids.end());
    for (int32_t off : span_offsets) {
        out_audio_positions.push_back(prefix_len + off);
    }

    out_ids.insert(out_ids.end(), hp.prompt_suffix_tokens.begin(), hp.prompt_suffix_tokens.end());
}

namespace {

constexpr const char k_default_variant[] = "moss-transcribe-diarize";
constexpr int        k_max_new           = 256;

int moss_context_ceiling(int32_t n_ctx_knob, const MossHParams & hp) {
    int ceiling = hp.dec_max_position_embeddings;
    if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
        ceiling = n_ctx_knob;
    }
    return ceiling;
}

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<MossModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }
    (void) read_optional_string_kv(loader.gguf(), "tokenizer.chat_template", "moss", "", m->chat_template);

    if (const transcribe_status st = read_moss_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();
    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: tokenizer vocab (%d) != decoder vocab_size (%d)",
                m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: GGUF tokenizer has no eos_token_id");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Publish an advisory input bound (decoder context / audio-token rate).
    if (m->hparams.dec_max_position_embeddings > 0 && m->hparams.fe_hop_length > 0 && m->hparams.fe_sample_rate > 0) {
        m->limits.has_context_cap = true;
        m->limits.model_max_ctx   = m->hparams.dec_max_position_embeddings;
        m->limits.prompt_overhead =
            static_cast<int>(m->hparams.prompt_prefix_tokens.size() + m->hparams.prompt_suffix_tokens.size());
        m->limits.gen_reserve = k_max_new;
        // audio_tokens ~= samples / (hop*2*merge) ; ms = tokens * hop*2*merge*1000/sr
        m->limits.ms_per_audio_token = static_cast<double>(m->hparams.fe_hop_length) * 2.0 *
                                       m->hparams.audio_merge_size * 1000.0 / m->hparams.fe_sample_rate;
        m->limits.kv_elems_per_ctx_token =
            static_cast<int64_t>(m->hparams.dec_n_kv_heads) * m->hparams.dec_head_dim * m->hparams.dec_n_layers * 2;
    }

    // Mel frontend (Whisper feature extractor; per-utterance log norm).
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;
        cfg.window_type  = m->hparams.fe_window;
        cfg.normalize    = m->hparams.fe_normalize;
        {
            using R               = transcribe::load_common::ReadF32Result;
            const size_t fb_elems = static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto   fb_rc    = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.mel_filterbank", fb_elems, "moss", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto   win_rc    = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.window", win_elems, "moss", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }
        m->mel.emplace(cfg);
    }

    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (const transcribe_status st = build_moss_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->gpu_device : 0, "moss", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "moss");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    {
        std::vector<transcribe::causal_lm::GateUpEntry> entries;
        entries.reserve(m->weights.dec_blocks.size());
        for (auto & b : m->weights.dec_blocks) {
            entries.push_back({ b.ffn_gate_w, b.ffn_up_w, &b.ffn_gate_up_w });
        }
        if (!transcribe::causal_lm::pack_gate_up(m->plan.primary, m->hparams.dec_hidden, m->hparams.dec_intermediate,
                                                 entries, m->packed_gate_up, "moss")) {
            m->packed_gate_up.free();
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto cc       = std::make_unique<MossSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;
    cc->n_ctx     = transcribe_session_params_n_ctx(params);

    cc->encoder_use_flash = false;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    auto * cm = static_cast<MossModel *>(model);
    {
        ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary, /*n_ctx=*/2048, cm->hparams.dec_n_kv_heads,
                                            cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss init_context: KV cache allocation failed");
            return TRANSCRIBE_ERR_OOM;
        }
    }
    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Encoder: chunk audio -> whisper encoder per chunk -> trim -> concat ->
// 4x merge + adaptor -> enc_out [dec_hidden, T_enc].
// ---------------------------------------------------------------------------

transcribe_status ensure_sched(MossSession * cc, MossModel * cm) {
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status reset_compute_ctx(MossSession * cc, int mb) {
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size     = static_cast<size_t>(mb) * 1024 * 1024;
    ip.mem_buffer   = nullptr;
    ip.no_alloc     = true;
    cc->compute_ctx = ggml_init(ip);
    return cc->compute_ctx != nullptr ? TRANSCRIBE_OK : TRANSCRIBE_ERR_GGUF;
}

// Fills enc_out [dec_hidden, T_enc]; returns T_enc via out_T_enc. `dumps` marks
// the single-chunk validation case (dumps enc.mel.in + encoder/adaptor stages).
transcribe_status encode_one(MossSession *        cc,
                             MossModel *          cm,
                             const float *        pcm,
                             int                  n_samples,
                             bool                 dumps,
                             std::vector<float> & enc_out,
                             int &                out_T_enc,
                             int64_t &            mel_us,
                             int64_t &            enc_us) {
    const auto & hp        = cm->hparams;
    const int    n_mels    = hp.enc_num_mel_bins;
    const int    n_chunk   = hp.fe_n_samples > 0 ? hp.fe_n_samples : 30 * hp.fe_sample_rate;
    const int    d_model   = hp.enc_d_model;
    const int    merge     = hp.audio_merge_size;

    if (const transcribe_status st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) {
        return st;
    }

    std::vector<float> concat_trim;  // [d_model, T_trim] column-major (d innermost)
    int                T_trim_total = 0;

    for (int start = 0; start < n_samples; start += n_chunk) {
        const int real_len   = std::min(n_chunk, n_samples - start);
        const int token_len  = audio_token_length(real_len, hp);
        if (token_len <= 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encode: degenerate token_len for chunk (real_len=%d)", real_len);
            return TRANSCRIBE_ERR_GGUF;
        }

        // Pad chunk to n_chunk samples (zeros) — matches processing._pad_or_trim_audio.
        std::vector<float> chunk(static_cast<size_t>(n_chunk), 0.0f);
        std::memcpy(chunk.data(), pcm + start, static_cast<size_t>(real_len) * sizeof(float));

        int           nm = 0, nf = 0;
        const int64_t t_mel0 = ggml_time_us();
        if (const transcribe_status mst = cm->mel->compute(chunk.data(), static_cast<size_t>(n_chunk), cc->mel_buf, nm,
                                                           nf, cc->n_threads);
            mst != TRANSCRIBE_OK) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encode: mel compute failed (%s)", transcribe_status_string(mst));
            return mst;
        }
        mel_us += ggml_time_us() - t_mel0;

        if (dumps && start == 0 && transcribe::debug::enabled()) {
            const long long shape[2] = { nm, nf };
            transcribe::debug::dump_host_f32("enc.mel.in", cc->mel_buf.data(),
                                             static_cast<long long>(cc->mel_buf.size()), shape, 2, "frontend.mel.norm");
        }

        // Whisper encoder graph on this chunk's mel [n_mels, nf].
        if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
            return st;
        }
        EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights, hp, nf, cc->encoder_use_flash);
        if (eb.graph == nullptr || eb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
            return TRANSCRIBE_ERR_OOM;
        }
        ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(), 0, cc->mel_buf.size() * sizeof(float));
        transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);

        const int64_t t_enc0 = ggml_time_us();
        if (ggml_backend_sched_graph_compute(cc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encode: encoder graph compute failed");
            return TRANSCRIBE_ERR_GGUF;
        }
        enc_us += ggml_time_us() - t_enc0;

        if (dumps && start == 0) {
            auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
                if (t != nullptr) {
                    transcribe::debug::dump_tensor(name, t, stage);
                }
            };
            try_dump("enc.pos_add.out", eb.dumps.pos_add_out, "enc.pos_add");
            try_dump("enc.block.0.out", eb.dumps.block_0_out, "enc.block.0");
            {
                char nm2[64];
                std::snprintf(nm2, sizeof(nm2), "enc.block.%d.out", hp.enc_n_layers - 1);
                try_dump(nm2, eb.dumps.block_last_out, "enc.block.last");
            }
            try_dump("enc.ln_post.out", eb.dumps.ln_post_out, "enc.ln_post");
        }

        // Read [d_model, T_whisper], trim to first token_len*merge columns.
        const int T_whisper = static_cast<int>(eb.out->ne[1]);
        const int keep_cols = token_len * merge;
        if (keep_cols > T_whisper) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encode: keep_cols=%d > T_whisper=%d", keep_cols, T_whisper);
            return TRANSCRIBE_ERR_GGUF;
        }
        std::vector<float> chunk_out(static_cast<size_t>(d_model) * T_whisper);
        ggml_backend_tensor_get(eb.out, chunk_out.data(), 0, chunk_out.size() * sizeof(float));
        // Columns are contiguous (d innermost); append the first keep_cols columns.
        const size_t old = concat_trim.size();
        concat_trim.resize(old + static_cast<size_t>(d_model) * keep_cols);
        std::memcpy(concat_trim.data() + old, chunk_out.data(),
                    static_cast<size_t>(d_model) * keep_cols * sizeof(float));
        T_trim_total += keep_cols;
    }

    if (T_trim_total <= 0 || T_trim_total % merge != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encode: bad T_trim_total=%d", T_trim_total);
        return TRANSCRIBE_ERR_GGUF;
    }

    // Adaptor graph on the concatenated, trimmed encoder output.
    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    AdaptorBuild ab = build_adaptor_graph(cc->compute_ctx, cm->weights, hp, T_trim_total);
    if (ab.graph == nullptr || ab.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, ab.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_tensor_set(ab.in, concat_trim.data(), 0, concat_trim.size() * sizeof(float));
    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
    const int64_t t_enc1 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(cc->sched, ab.graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss encode: adaptor graph compute failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    enc_us += ggml_time_us() - t_enc1;

    if (dumps) {
        auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
            if (t != nullptr) {
                transcribe::debug::dump_tensor(name, t, stage);
            }
        };
        try_dump("enc.merge.out", ab.dumps.merge_out, "enc.merge");
        try_dump("enc.adaptor.out", ab.dumps.adaptor_out, "enc.adaptor");
    }

    out_T_enc = ab.T_enc;
    enc_out.resize(static_cast<size_t>(hp.dec_hidden) * out_T_enc);
    ggml_backend_tensor_get(ab.out, enc_out.data(), 0, enc_out.size() * sizeof(float));
    return TRANSCRIBE_OK;
}

// Build audio_dense [hidden, T_prompt] + keep_mask [1, T_prompt] host-side by
// scattering enc_out columns into the audio-pad prompt positions.
void build_injection(int                        hidden,
                     int                        T_prompt,
                     const std::vector<float> & enc_out,
                     const std::vector<int32_t> & audio_positions,
                     std::vector<float> &       audio_dense,
                     std::vector<float> &       keep_mask) {
    audio_dense.assign(static_cast<size_t>(hidden) * T_prompt, 0.0f);
    keep_mask.assign(static_cast<size_t>(T_prompt), 1.0f);
    for (size_t j = 0; j < audio_positions.size(); ++j) {
        const int pos = audio_positions[j];
        std::memcpy(audio_dense.data() + static_cast<size_t>(pos) * hidden,
                    enc_out.data() + j * hidden, static_cast<size_t>(hidden) * sizeof(float));
        keep_mask[static_cast<size_t>(pos)] = 0.0f;
    }
}

int32_t argmax_vec(const std::vector<float> & v) {
    int32_t best   = 0;
    float   best_v = v.empty() ? 0.0f : v[0];
    for (int32_t i = 1; i < static_cast<int32_t>(v.size()); ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best   = i;
        }
    }
    return best;
}

transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    (void) params;
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<MossSession *>(session);
    auto * cm = static_cast<MossModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    transcribe::debug::init();
    const bool dumps_on = transcribe::debug::enabled();

    // Encoder.
    int T_enc = 0;
    if (const transcribe_status st =
            encode_one(cc, cm, pcm, n_samples, /*dumps=*/true, cc->enc_host, T_enc, cc->t_mel_us, cc->t_encode_us);
        st != TRANSCRIBE_OK) {
        return st;
    }

    const int64_t t_dec_start = ggml_time_us();

    // Prompt.
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> audio_positions;
    build_prompt_tokens(cm->hparams, T_enc, prompt_ids, audio_positions);
    const int T_prompt = static_cast<int>(prompt_ids.size());
    if (static_cast<int>(audio_positions.size()) != T_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss run: audio_positions(%zu) != T_enc(%d)", audio_positions.size(),
                T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }

    const int ceiling = moss_context_ceiling(cc->n_ctx, cm->hparams);
    if (T_prompt + k_max_new > ceiling) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "moss run: input too long — %d prompt + %d generation exceed the %d-token context.", T_prompt,
                k_max_new, ceiling);
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }

    // Generation budget scales with audio length: the emergent transcript
    // (text + [start]/[Sxx]/[end] markers) tracks the audio-token count, which
    // for long-form far exceeds the k_max_new floor. Clamp to the context.
    const int gen_budget = std::min(ceiling - T_prompt, std::max(k_max_new, 2 * T_enc + 128));

    // KV cache (grow-to-fit, pow2, clamped to ceiling).
    int want_n_ctx = 1024;
    while (want_n_ctx < T_prompt + gen_budget) {
        want_n_ctx *= 2;
    }
    if (want_n_ctx > ceiling) {
        want_n_ctx = ceiling;
    }
    if (cc->kv_cache.ctx != nullptr && cc->kv_cache.n_ctx < want_n_ctx) {
        cc->kv_cache.free();
    }
    if (cc->kv_cache.ctx == nullptr) {
        ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary, want_n_ctx, cm->hparams.dec_n_kv_heads,
                                            cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss run: KV cache allocation failed (n_ctx=%d)", want_n_ctx);
            return TRANSCRIBE_ERR_OOM;
        }
    } else {
        if (cc->kv_cache.buffer != nullptr) {
            ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
        }
        cc->kv_cache.n    = 0;
        cc->kv_cache.head = 0;
    }

    // Prefill.
    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    const bool   slice_last = !dumps_on;
    PrefillBuild pb = build_prefill_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, T_prompt,
                                          cc->decoder_use_flash, slice_last);
    if (pb.graph == nullptr || pb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
    {
        std::vector<float> audio_dense, keep_mask;
        build_injection(cm->hparams.dec_hidden, T_prompt, cc->enc_host, audio_positions, audio_dense, keep_mask);
        ggml_backend_tensor_set(pb.audio_dense_in, audio_dense.data(), 0, audio_dense.size() * sizeof(float));
        ggml_backend_tensor_set(pb.keep_mask_in, keep_mask.data(), 0, keep_mask.size() * sizeof(float));
    }
    {
        std::vector<int32_t> positions(T_prompt);
        for (int i = 0; i < T_prompt; ++i) {
            positions[i] = i;
        }
        ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    }
    {
        const ggml_fp16_t        mz = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        mn = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * T_prompt, mn);
        for (int r = 0; r < T_prompt; ++r) {
            for (int c = 0; c <= r; ++c) {
                mask[static_cast<size_t>(r) * T_prompt + c] = mz;
            }
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(cc->sched, pb.graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss run: prefill compute failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->kv_cache.n    = T_prompt;
    cc->kv_cache.head = T_prompt;

    if (dumps_on) {
        auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
            if (t != nullptr) {
                transcribe::debug::dump_tensor(name, t, stage);
            }
        };
        try_dump("dec.token_emb", pb.dumps.token_emb, "dec.token_emb");
        try_dump("dec.audio_injected", pb.dumps.audio_injected, "dec.audio_injected");
        try_dump("dec.block.0.out", pb.dumps.block_0_out, "dec.block.0");
        {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "dec.block.%d.out", cm->hparams.dec_n_layers - 1);
            try_dump(nm, pb.dumps.block_last_out, "dec.block.last");
        }
        try_dump("dec.out_before_head", pb.dumps.out_before_head, "dec.out_before_head");
        try_dump("dec.logits_raw", pb.dumps.logits_raw, "dec.logits_raw");
    }

    const int          vocab = cm->hparams.dec_vocab_size;
    std::vector<float> logits(vocab);
    ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));

    std::vector<int32_t> generated_ids;
    int32_t              next_tok = argmax_vec(logits);
    generated_ids.push_back(next_tok);

    // Step loop.
    const int32_t eos_id  = cm->hparams.eos_token_id;
    int           cur_past = T_prompt;
    int           max_n_kv = 1024;
    while (max_n_kv < T_prompt + gen_budget) {
        max_n_kv *= 2;
    }
    if (max_n_kv > cc->kv_cache.n_ctx) {
        max_n_kv = cc->kv_cache.n_ctx;
    }

    if (const transcribe_status st = reset_compute_ctx(cc, 8); st != TRANSCRIBE_OK) {
        return st;
    }
    StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, max_n_kv,
                                    cc->decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    const ggml_fp16_t        mz = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t        mn = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(max_n_kv, mn);

    int gen_dump_step = -1;
    if (dumps_on) {
        gen_dump_step = 8;  // dec.logits_raw.gen8 (mid-generation, n_past>0 coverage)
    }

    while (next_tok != eos_id && static_cast<int32_t>(generated_ids.size()) < gen_budget && cur_past + 1 <= max_n_kv) {
        ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
        const int32_t pos_val = cur_past;
        ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
        const int64_t kv_idx_val = cur_past;
        ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));
        if (cur_past == T_prompt) {
            std::fill(step_mask.begin(), step_mask.begin() + cur_past + 1, mz);
        } else {
            step_mask[cur_past] = mz;
        }
        ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0, static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));

        if (ggml_backend_sched_graph_compute(cc->sched, sb.graph) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss step: graph compute failed");
            return TRANSCRIBE_ERR_GGUF;
        }

        // Mid-generation logits dump (n_past>0 coverage) at a fixed step.
        const int steps_done = static_cast<int>(generated_ids.size());  // == n produced before this step's token
        if (dumps_on && steps_done == gen_dump_step && sb.logits != nullptr) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "dec.logits_raw.gen%d", gen_dump_step);
            transcribe::debug::dump_tensor(nm, sb.logits, "dec.logits_raw.gen");
        }

        int32_t argmax_tok = 0;
        ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
        next_tok = argmax_tok;
        generated_ids.push_back(next_tok);
        cur_past += 1;
        cc->kv_cache.n    = cur_past + 1;
        cc->kv_cache.head = cur_past + 1;
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
    }

    if (next_tok != eos_id) {
        cc->was_truncated = true;
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "moss run: output truncated at %d tokens",
                static_cast<int>(generated_ids.size()));
    }
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    // Raw transcript: [start][Sxx]text[end] emergent text. De-diarization is a
    // scoring-time concern (WER harness), not the runtime's.
    std::string raw_text = cm->tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));

    cc->t_decode_us = ggml_time_us() - t_dec_start;
    cc->full_text   = raw_text;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;
    transcribe_session::SegmentEntry seg{};
    seg.text  = raw_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    return cc->was_truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Offline batched decode: per-utterance encoder + prefill (serial into KV
// slabs) or a batched prefill, then a batched step loop.
// ---------------------------------------------------------------------------

transcribe_session::ResultSet finalize_utterance(MossModel * cm, std::vector<int32_t> generated_ids, int n_samples) {
    transcribe_session::ResultSet rs;
    const int32_t                 eos_id = cm->hparams.eos_token_id;
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }
    std::string raw_text = cm->tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));
    rs.full_text   = raw_text;
    rs.result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    rs.has_result  = true;
    rs.status      = TRANSCRIBE_OK;
    transcribe_session::SegmentEntry seg{};
    seg.text  = raw_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    rs.segments.push_back(std::move(seg));
    return rs;
}

transcribe_status run_batch_serial(MossSession *                 cc,
                                   const float * const *         pcm,
                                   const int *                   n_samples,
                                   int                           n,
                                   const transcribe_run_params * params) {
    for (int i = 0; i < n; ++i) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const transcribe_status st = (pcm[i] == nullptr || n_samples[i] <= 0) ? TRANSCRIBE_ERR_INVALID_ARG
                                                                              : run(cc, pcm[i], n_samples[i], params);
        if (st == TRANSCRIBE_OK) {
            cc->batch_results.push_back(cc->capture_result(st));
        } else {
            transcribe_session::ResultSet rs;
            rs.status = st;
            cc->batch_results.push_back(std::move(rs));
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status run_batch(transcribe_session *          session,
                            const float * const *         pcm,
                            const int *                   n_samples,
                            int                           n,
                            const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<MossSession *>(session);
    auto * cm = static_cast<MossModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Batched decode requires the flash step path and dump-free operation.
    if (!cc->decoder_use_flash || transcribe::debug::enabled() || n == 1) {
        return run_batch_serial(cc, pcm, n_samples, n, params);
    }
    (void) params;
    transcribe::debug::init();

    const int hidden = cm->hparams.dec_hidden;

    // Pass 1: per-utterance encoder + prompt build.
    std::vector<std::vector<float>>   enc_hosts(n);
    std::vector<int>                  T_enc(n, 0);
    std::vector<std::vector<int32_t>> prompt_ids(n);
    std::vector<std::vector<int32_t>> audio_positions(n);
    std::vector<int>                  T_prompt(n, 0);
    std::vector<char>                 valid(n, 0);
    std::vector<transcribe_status>    fail_status(n, TRANSCRIBE_ERR_INVALID_ARG);
    int64_t                           mel_us = 0, enc_us = 0;

    const int ceiling      = moss_context_ceiling(cc->n_ctx, cm->hparams);
    int       max_T_prompt = 0;
    int       max_T_enc    = 0;
    for (int b = 0; b < n; ++b) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (pcm[b] == nullptr || n_samples[b] <= 0) {
            continue;
        }
        int te = 0;
        if (encode_one(cc, cm, pcm[b], n_samples[b], /*dumps=*/false, enc_hosts[b], te, mel_us, enc_us) !=
            TRANSCRIBE_OK) {
            continue;
        }
        T_enc[b] = te;
        build_prompt_tokens(cm->hparams, te, prompt_ids[b], audio_positions[b]);
        T_prompt[b] = static_cast<int>(prompt_ids[b].size());
        if (T_prompt[b] + k_max_new > ceiling) {
            fail_status[b] = TRANSCRIBE_ERR_INPUT_TOO_LONG;
            continue;
        }
        valid[b]     = 1;
        max_T_prompt = std::max(max_T_prompt, T_prompt[b]);
        max_T_enc    = std::max(max_T_enc, T_enc[b]);
    }
    if (max_T_prompt == 0) {
        for (int b = 0; b < n; ++b) {
            transcribe_session::ResultSet rs;
            rs.status = fail_status[b];
            cc->batch_results.push_back(std::move(rs));
        }
        return TRANSCRIBE_OK;
    }

    // Batch-wide generation budget: covers the longest utterance's transcript
    // (scales with its audio tokens), clamped to the context.
    const int batch_budget = std::min(ceiling - max_T_prompt, std::max(k_max_new, 2 * max_T_enc + 128));

    int max_n_kv = 1024;
    while (max_n_kv < max_T_prompt + batch_budget) {
        max_n_kv *= 2;
    }
    if (max_n_kv > ceiling) {
        max_n_kv = ceiling;
    }

    ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    if (cc->kv_cache_batch.self_k == nullptr || cc->kv_batch_cap != n || cc->kv_batch_n_ctx != max_n_kv) {
        cc->kv_cache_batch.free();
        if (!transcribe::causal_lm::kv_init_batched(cc->kv_cache_batch, cm->plan.primary, max_n_kv,
                                                    cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                                    cm->hparams.dec_n_layers, n, kv_type)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss run_batch: batched KV allocation failed");
            return TRANSCRIBE_ERR_OOM;
        }
        cc->kv_batch_cap   = n;
        cc->kv_batch_n_ctx = max_n_kv;
    } else if (cc->kv_cache_batch.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache_batch.buffer, 0);
    }

    // Pass 2: batched prefill.
    std::vector<std::vector<int32_t>> generated(n);
    std::vector<int32_t>              next_tok(n, 0);
    std::vector<int>                  n_past(n, 0);
    {
        int T_prompt_max = max_T_prompt;
        if (const transcribe_status st = reset_compute_ctx(cc, 32); st != TRANSCRIBE_OK) {
            return st;
        }
        PrefillBuildBatched pb = build_prefill_graph_batched(cc->compute_ctx, cm->weights, cm->hparams,
                                                             cc->kv_cache_batch, T_prompt_max, n, cc->decoder_use_flash);
        if (pb.graph == nullptr || pb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (const transcribe_status st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) {
            return st;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
            return TRANSCRIBE_ERR_OOM;
        }

        // input_ids [T_prompt_max, n].
        {
            std::vector<int32_t> ids(static_cast<size_t>(T_prompt_max) * n, 0);
            for (int b = 0; b < n; ++b) {
                if (valid[b]) {
                    std::memcpy(ids.data() + static_cast<size_t>(b) * T_prompt_max, prompt_ids[b].data(),
                                static_cast<size_t>(T_prompt[b]) * sizeof(int32_t));
                }
            }
            ggml_backend_tensor_set(pb.input_ids_in, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        // audio_dense + keep (per-utterance scatter at audio_positions).
        {
            std::vector<float> audio_dense(static_cast<size_t>(hidden) * T_prompt_max * n, 0.0f);
            std::vector<float> keep(static_cast<size_t>(T_prompt_max) * n, 1.0f);
            for (int b = 0; b < n; ++b) {
                if (!valid[b]) {
                    continue;
                }
                for (size_t j = 0; j < audio_positions[b].size(); ++j) {
                    const size_t dst_col = static_cast<size_t>(b) * T_prompt_max + audio_positions[b][j];
                    std::memcpy(audio_dense.data() + dst_col * hidden, enc_hosts[b].data() + j * hidden,
                                static_cast<size_t>(hidden) * sizeof(float));
                    keep[dst_col] = 0.0f;
                }
            }
            ggml_backend_tensor_set(pb.audio_dense_in, audio_dense.data(), 0, audio_dense.size() * sizeof(float));
            ggml_backend_tensor_set(pb.keep_mask_in, keep.data(), 0, keep.size() * sizeof(float));
        }
        {
            std::vector<int32_t> pos(T_prompt_max);
            for (int t = 0; t < T_prompt_max; ++t) {
                pos[t] = t;
            }
            ggml_backend_tensor_set(pb.positions_in, pos.data(), 0, pos.size() * sizeof(int32_t));
        }
        {
            const ggml_fp16_t        mz = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t        mn = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt_max) * T_prompt_max, mn);
            for (int q = 0; q < T_prompt_max; ++q) {
                std::fill(mask.begin() + static_cast<size_t>(q) * T_prompt_max,
                          mask.begin() + static_cast<size_t>(q) * T_prompt_max + q + 1, mz);
            }
            ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        {
            std::vector<int64_t> kidx(static_cast<size_t>(T_prompt_max) * n);
            for (int b = 0; b < n; ++b) {
                for (int t = 0; t < T_prompt_max; ++t) {
                    kidx[static_cast<size_t>(b) * T_prompt_max + t] = t;
                }
            }
            ggml_backend_tensor_set(pb.kv_idx_in, kidx.data(), 0, kidx.size() * sizeof(int64_t));
        }
        {
            std::vector<int32_t> lidx(n, 0);
            for (int b = 0; b < n; ++b) {
                lidx[b] = valid[b] ? (T_prompt[b] - 1) : 0;
            }
            ggml_backend_tensor_set(pb.last_idx_in, lidx.data(), 0, lidx.size() * sizeof(int32_t));
        }

        transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
        if (ggml_backend_sched_graph_compute(cc->sched, pb.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_GGUF;
        }
        std::vector<int32_t> amax(n, 0);
        ggml_backend_tensor_get(pb.out, amax.data(), 0, amax.size() * sizeof(int32_t));
        for (int b = 0; b < n; ++b) {
            if (valid[b]) {
                n_past[b]   = T_prompt[b];
                next_tok[b] = amax[b];
                generated[b].push_back(amax[b]);
            }
        }
    }

    // Pass 3: batched step loop (shared driver).
    const int32_t eos_id = cm->hparams.eos_token_id;
    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    StepBuildBatched sb = build_step_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache_batch,
                                                   max_n_kv, n, cc->decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    transcribe::causal_lm::StepBatchedIO io{};
    io.input_ids = sb.input_ids_in;
    io.positions = sb.position_in;
    io.kv_idx    = sb.kv_idx_in;
    io.mask      = sb.mask_in;
    io.argmax    = sb.out;
    io.graph     = sb.graph;

    transcribe::causal_lm::StepBatchedState step_state;
    step_state.valid    = valid;
    step_state.next_tok = next_tok;
    step_state.n_past   = n_past;

    std::vector<char> truncated;
    if (const transcribe_status st = transcribe::causal_lm::run_batched_step_loop(
            cc, cc->sched, io, n, max_n_kv, eos_id, batch_budget, step_state, generated, nullptr, &truncated);
        st != TRANSCRIBE_OK) {
        return st;
    }

    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            transcribe_session::ResultSet rs;
            rs.status = fail_status[b];
            cc->batch_results.push_back(std::move(rs));
            continue;
        }
        transcribe_session::ResultSet rs = finalize_utterance(cm, generated[b], n_samples[b]);
        if (b < static_cast<int>(truncated.size()) && truncated[b]) {
            cc->was_truncated = true;
            rs.status         = TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
        }
        cc->batch_results.push_back(std::move(rs));
    }
    return TRANSCRIBE_OK;
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "moss",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ run_batch,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

}  // namespace transcribe::moss
