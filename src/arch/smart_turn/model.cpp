// arch/smart_turn/model.cpp - Smart Turn v3 family handler (TURN role): load,
// the clip forward pass, the turn ops table and the validation dumps.
//
// Reference (pipecat-ai/smart-turn inference.py, train.py SmartTurnV3Model):
//   audio: the last max_audio_samples, left zero padded to that length;
//   WhisperFeatureExtractor(do_normalize=True): zero mean, unit variance of
//   the whole window (padding included), then Whisper's log-mel;
//   WhisperEncoder; attention pooling softmax_t(Linear-Tanh-Linear) . h;
//   classifier Linear-LayerNorm-GELU-Linear-GELU-Linear; sigmoid.

#include "../whisper/encoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "smart_turn.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace transcribe::smart_turn {

namespace {

constexpr const char * kTag = "smart_turn";

constexpr const char k_default_variant[] = "smart-turn-v3.2";

constexpr size_t k_compute_ctx_bytes = 8u * 1024u * 1024u;
constexpr int    k_graph_size        = 4096;

struct GgufGuard {
    gguf_context * ctx = nullptr;

    GgufGuard() = default;

    ~GgufGuard() {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }

    GgufGuard(const GgufGuard &)             = delete;
    GgufGuard & operator=(const GgufGuard &) = delete;
};

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp) {
    whisper::WhisperHParams & w = hp.whisper;

    struct U32Key {
        const char * key;
        int32_t *    out;
    };

    const U32Key u32_keys[] = {
        { "stt.smart_turn.encoder.n_layers",             &w.enc_n_layers             },
        { "stt.smart_turn.encoder.d_model",              &w.enc_d_model              },
        { "stt.smart_turn.encoder.n_heads",              &w.enc_n_heads              },
        { "stt.smart_turn.encoder.ffn_dim",              &w.enc_ffn_dim              },
        { "stt.smart_turn.encoder.num_mel_bins",         &w.enc_num_mel_bins         },
        { "stt.smart_turn.encoder.max_source_positions", &w.enc_max_source_positions },
        { "stt.smart_turn.max_audio_samples",            &hp.max_audio_samples       },
    };
    for (const U32Key & k : u32_keys) {
        if (auto st = read_required_u32_kv(gguf, k.key, kTag, *k.out); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (auto st = read_optional_string_kv(gguf, "stt.smart_turn.encoder.activation", kTag, "gelu", w.enc_activation);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.smart_turn.threshold", kTag, hp.threshold); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_bool_kv(gguf, "stt.smart_turn.waveform_normalize", kTag, true, hp.waveform_normalize);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.smart_turn.waveform_normalize_eps", kTag, hp.waveform_normalize_eps);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = whisper::read_whisper_frontend_hparams(gguf, w); st != TRANSCRIBE_OK) {
        // The shared reader logs under "whisper:"; name the family too.
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: Whisper front-end metadata rejected", kTag);
        return st;
    }

    if (w.enc_n_layers <= 0 || w.enc_d_model <= 0 || w.enc_n_heads <= 0 || w.enc_ffn_dim <= 0 ||
        w.enc_max_source_positions <= 0 || w.enc_d_model % w.enc_n_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid encoder hparams", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (w.enc_activation != "gelu") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: unsupported encoder activation \"%s\"", kTag,
                w.enc_activation.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (w.fe_num_mels != w.enc_num_mel_bins || w.fe_sample_rate != 16000) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: front end does not match the encoder (%d mels, %d Hz)", kTag,
                w.fe_num_mels, w.fe_sample_rate);
        return TRANSCRIBE_ERR_GGUF;
    }
    // The window maps to exactly 2 * max_source_positions mel frames.
    if (hp.max_audio_samples <= 0 || hp.max_audio_samples % w.fe_hop_length != 0 ||
        hp.n_mel_frames() != 2 * w.enc_max_source_positions) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: max_audio_samples %d does not give %d mel frames", kTag,
                hp.max_audio_samples, 2 * w.enc_max_source_positions);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!(hp.threshold >= 0.0f && hp.threshold <= 1.0f) || !(hp.waveform_normalize_eps > 0.0f)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid threshold or normalisation eps", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

#define GET_F32(slot, name, ...)                                                                          \
    do {                                                                                                  \
        ggml_tensor * _t =                                                                                \
            transcribe::weights::find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr)                                                                                \
            return TRANSCRIBE_ERR_GGUF;                                                                   \
        (slot) = _t;                                                                                      \
    } while (0)

#define GET_LIN(slot, name, ...)                                                                                 \
    do {                                                                                                         \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                              \
        if (_t == nullptr)                                                                                       \
            return TRANSCRIBE_ERR_GGUF;                                                                          \
        (slot) = _t;                                                                                             \
    } while (0)

// Head widths come from the tensors; bind them after reading them.
transcribe_status bind_head(ggml_context * ctx_meta, HParams & hp, HeadWeights & h) {
    const int64_t d  = hp.whisper.enc_d_model;
    ggml_tensor * p0 = ggml_get_tensor(ctx_meta, "head.pool.0.weight");
    ggml_tensor * c0 = ggml_get_tensor(ctx_meta, "head.cls.0.weight");
    ggml_tensor * c4 = ggml_get_tensor(ctx_meta, "head.cls.4.weight");
    if (p0 == nullptr || c0 == nullptr || c4 == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: missing head tensors", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    hp.pool_hidden = static_cast<int32_t>(p0->ne[1]);
    hp.cls_hidden0 = static_cast<int32_t>(c0->ne[1]);
    hp.cls_hidden1 = static_cast<int32_t>(c4->ne[1]);

    GET_LIN(h.pool0_w, "head.pool.0.weight", d, hp.pool_hidden);
    GET_F32(h.pool0_b, "head.pool.0.bias", hp.pool_hidden);
    GET_LIN(h.pool2_w, "head.pool.2.weight", hp.pool_hidden, 1);
    GET_F32(h.pool2_b, "head.pool.2.bias", 1);
    GET_LIN(h.cls0_w, "head.cls.0.weight", d, hp.cls_hidden0);
    GET_F32(h.cls0_b, "head.cls.0.bias", hp.cls_hidden0);
    GET_F32(h.cls1_w, "head.cls.1.weight", hp.cls_hidden0);
    GET_F32(h.cls1_b, "head.cls.1.bias", hp.cls_hidden0);
    GET_LIN(h.cls4_w, "head.cls.4.weight", hp.cls_hidden0, hp.cls_hidden1);
    GET_F32(h.cls4_b, "head.cls.4.bias", hp.cls_hidden1);
    GET_LIN(h.cls6_w, "head.cls.6.weight", hp.cls_hidden1, 1);
    GET_F32(h.cls6_b, "head.cls.6.bias", 1);
    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_LIN

transcribe_status install_mel(Loader & loader, Model & m) {
    using R                           = load_common::ReadF32Result;
    const whisper::WhisperHParams & w = m.hparams.whisper;
    std::vector<float>              fb;
    std::vector<float>              win;
    const size_t fb_elems = static_cast<size_t>(w.fe_num_mels) * static_cast<size_t>(w.fe_n_fft / 2 + 1);
    if (load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.mel_filterbank", fb_elems, kTag,
                                             fb) != R::Ok ||
        load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.window",
                                             static_cast<size_t>(w.fe_n_fft), kTag, win) != R::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: frontend.mel_filterbank / frontend.window missing or unreadable",
                kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    return whisper::install_mel_from_buffers(w, std::move(fb), std::move(win), m.mel);
}

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m     = std::make_unique<Model>();
    m->arch    = &arch;
    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();

    if (auto st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = install_mel(loader, *m); st != TRANSCRIBE_OK) {
        return st;
    }

    GgufGuard guard;
    {
        gguf_init_params init_params{};
        init_params.no_alloc = true;
        init_params.ctx      = &m->ctx_meta;
        guard.ctx            = gguf_init_from_file(loader.path().c_str(), init_params);
        if (guard.ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to reopen \"%s\" for tensor data", kTag,
                    loader.path().c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (auto st = whisper::build_whisper_encoder_weights(m->ctx_meta, m->hparams.whisper, m->encoder);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = bind_head(m->ctx_meta, m->hparams, m->head); st != TRANSCRIBE_OK) {
        return st;
    }

    // AUTO resolves to the CPU backend: the model is small, runs once per
    // pause, and the CPU path is the one validated against the reference. A
    // specific backend or device is still honored.
    transcribe_device_t        device      = params != nullptr ? params->device : nullptr;
    transcribe_backend_request backend_req = params != nullptr ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    int                        req_raw     = TRANSCRIBE_BACKEND_AUTO;
    std::memcpy(&req_raw, &backend_req, sizeof(req_raw));
    if (req_raw == TRANSCRIBE_BACKEND_AUTO && device == nullptr) {
        backend_req = TRANSCRIBE_BACKEND_CPU;
    }
    if (auto st = load_common::init_backends(backend_req, device, kTag, m->plan); st != TRANSCRIBE_OK) {
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    if (auto st =
            load_common::load_weights(loader.path(), guard.ctx, m->ctx_meta, m->plan, params, kTag, m->weight_storage);
        st != TRANSCRIBE_OK) {
        return st;
    }

    bool dec_flash_unused = false;
    flash::apply_env_overrides(m->use_flash, dec_flash_unused);

    m->turn_info.max_audio_samples = m->hparams.max_audio_samples;  // a clip model: no frames, no speech track
    m->turn_info.threshold         = m->hparams.threshold;

    m->roles = TRANSCRIBE_ROLE_TURN;
    set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

struct HeadDumps {
    ggml_tensor * pool_weights = nullptr;  // [T]
    ggml_tensor * pooled       = nullptr;  // [d_model]
    ggml_tensor * logit        = nullptr;  // [1]
};

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul_mat(ctx, w, x), b);
}

// enc ne [d_model, T] -> probability ne [1].
ggml_tensor * build_head(ggml_context * ctx, const Model & m, ggml_tensor * enc, HeadDumps & d) {
    const HeadWeights & h = m.head;
    const int64_t       T = enc->ne[1];

    // Attention pooling: softmax over time of Linear-Tanh-Linear.
    ggml_tensor * a      = ggml_tanh(ctx, linear(ctx, enc, h.pool0_w, h.pool0_b));          // [pool_hidden, T]
    a                    = linear(ctx, a, h.pool2_w, h.pool2_b);                            // [1, T]
    a                    = ggml_soft_max(ctx, ggml_reshape_2d(ctx, a, T, 1));               // [T, 1]
    d.pool_weights       = a;
    ggml_tensor * pooled = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, enc)), a);  // [d_model, 1]
    d.pooled             = pooled;

    // Classifier (Dropout is identity at inference).
    ggml_tensor * x = linear(ctx, pooled, h.cls0_w, h.cls0_b);
    x               = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, m.hparams.cls_norm_eps), h.cls1_w), h.cls1_b);
    x               = ggml_gelu_erf(ctx, x);
    x               = ggml_gelu_erf(ctx, linear(ctx, x, h.cls4_w, h.cls4_b));
    x               = linear(ctx, x, h.cls6_w, h.cls6_b);
    d.logit         = x;
    return ggml_sigmoid(ctx, x);
}

// s.window = pcm left zero padded to max_audio_samples, then (HF's
// zero_mean_unit_var_norm) zero mean, unit variance over the whole window.
// Requires 0 < n <= max_audio_samples.
void prepare_window(Session & s, const HParams & hp, const float * pcm, int64_t n) {
    const size_t N = static_cast<size_t>(hp.max_audio_samples);
    s.window.assign(N, 0.0f);
    std::memcpy(s.window.data() + (N - static_cast<size_t>(n)), pcm, static_cast<size_t>(n) * sizeof(float));
    if (!hp.waveform_normalize) {
        return;
    }
    double sum = 0.0;
    for (float v : s.window) {
        sum += v;
    }
    const double mean = sum / static_cast<double>(N);
    double       var  = 0.0;
    for (float v : s.window) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(N);
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(hp.waveform_normalize_eps));
    for (float & v : s.window) {
        v = static_cast<float>((v - mean) * inv);
    }
}

transcribe_status score_clip(transcribe_turn_session * session, const float * pcm, int64_t n, float * p) {
    auto &          s  = static_cast<Session &>(*session);
    const Model &   m  = static_cast<const Model &>(*session->model);
    const HParams & hp = m.hparams;
    debug::init();
    const int n_threads = s.n_threads > 0 ? s.n_threads : default_n_threads();

    // The dispatcher crops; prepare_window relies on it.
    if (n <= 0 || n > hp.max_audio_samples) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: clip of %lld samples outside 1..%d", kTag, static_cast<long long>(n),
                hp.max_audio_samples);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // ---- front end (host) ----
    const int64_t t_mel_start = ggml_time_us();
    prepare_window(s, hp, pcm, n);
    int n_mels = 0;
    int T      = 0;
    if (auto st = m.mel->compute(s.window.data(), s.window.size(), s.mel_raw, n_mels, T, n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (T != hp.n_mel_frames() || n_mels != hp.whisper.enc_num_mel_bins) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: front end gave [%d, %d], expected [%d, %d]", kTag, n_mels, T,
                hp.whisper.enc_num_mel_bins, hp.n_mel_frames());
        return TRANSCRIBE_ERR_BACKEND;
    }
    s.mel_in.resize(s.mel_raw.size());
    for (int mi = 0; mi < n_mels; ++mi) {
        for (int t = 0; t < T; ++t) {
            s.mel_in[static_cast<size_t>(t) * n_mels + mi] = s.mel_raw[static_cast<size_t>(mi) * T + t];
        }
    }
    s.t_mel_us = ggml_time_us() - t_mel_start;
    if (s.poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    // ---- graph ----
    if (s.compute_ctx != nullptr) {
        ggml_free(s.compute_ctx);
        s.compute_ctx = nullptr;
    }
    {
        ggml_init_params init_params{};
        init_params.mem_size = k_compute_ctx_bytes;
        init_params.no_alloc = true;
        s.compute_ctx        = ggml_init(init_params);
        if (s.compute_ctx == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
    }
    whisper::EncoderBuild eb =
        whisper::build_encoder_graph(s.compute_ctx, m.encoder, hp.whisper, T, m.use_flash, m.backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    HeadDumps     hd;
    ggml_tensor * prob = build_head(s.compute_ctx, m, eb.out, hd);
    ggml_set_output(prob);
    if (debug::enabled()) {
        for (ggml_tensor * t : { hd.pool_weights, hd.pooled, hd.logit }) {
            ggml_set_output(t);
            ggml_build_forward_expand(eb.graph, t);
        }
    }
    ggml_build_forward_expand(eb.graph, prob);

    if (s.sched == nullptr) {
        auto & backends = const_cast<std::vector<ggml_backend_t> &>(m.plan.scheduler_list);
        s.sched = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()), k_graph_size,
                                         /*parallel=*/false, /*op_offload=*/true);
        if (s.sched == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
    }
    configure_sched_n_threads(s.sched, n_threads);
    ggml_backend_sched_reset(s.sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(s.sched, eb.graph, kTag)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: graph allocation failed", kTag);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_tensor_set(eb.mel_in, s.mel_in.data(), 0, s.mel_in.size() * sizeof(float));

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs = ggml_backend_sched_graph_compute(s.sched, eb.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: graph compute failed (%d)", kTag, static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    s.t_encode_us = ggml_time_us() - t_enc_start;
    ggml_backend_tensor_get(prob, p, 0, sizeof(float));

    if (debug::enabled()) {
        const long long shape_w[1] = { static_cast<long long>(s.window.size()) };
        debug::dump_host_f32("fe.window", s.window.data(), static_cast<long long>(s.window.size()), shape_w, 1,
                             "frontend");
        auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
            if (t != nullptr) {
                debug::dump_tensor(name, t, stage);
            }
        };
        try_dump("enc.mel.in", eb.dumps.mel_in, "encoder.mel");
        try_dump("enc.conv1.out", eb.dumps.conv1_out, "encoder.conv1");
        try_dump("enc.conv2.out", eb.dumps.conv2_out, "encoder.conv2");
        try_dump("enc.embed.out", eb.dumps.embed_out, "encoder.embed");
        for (size_t i = 0; i < eb.dumps.block_outs.size(); ++i) {
            char name[64];
            std::snprintf(name, sizeof(name), "enc.block.%zu.out", i);
            try_dump(name, eb.dumps.block_outs[i], "encoder.block");
        }
        try_dump("enc.final", eb.dumps.final_out, "encoder.final");
        try_dump("head.pool_weights", hd.pool_weights, "head");
        try_dump("head.pooled", hd.pooled, "head");
        try_dump("head.logit", hd.logit, "head");
        try_dump("head.prob", prob, "head");
    }
    return TRANSCRIBE_OK;
}

const TurnModelInfo & turn_info(const transcribe_model * model) {
    return static_cast<const Model *>(model)->turn_info;
}

transcribe_turn_session * turn_new_session() {
    return new Session();
}

const TurnOps k_turn_ops = { turn_info, turn_new_session, score_clip };

}  // namespace

Model::~Model() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    weight_storage.reset();
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

const Arch arch = {
    /* .name             = */ "smart_turn",
    /* .load             = */ load,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ nullptr,
    /* .vad              = */ nullptr,
    /* .turn             = */ &k_turn_ops,
};

}  // namespace transcribe::smart_turn
