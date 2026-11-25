// arch/sava/model.cpp - SAVA family handler (TURN and VAD roles): load, the
// turn and VAD ops tables, and the validation dumps.
//
// Both roles step the same stream (graph.cpp). A TURN session reports the
// horizon-0 turn probability and the emitted speech probability of every
// step; a VAD session reports the speech probability only. Each session owns
// its own stream state.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "sava.h"
#include "transcribe-abi.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe/sava.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace transcribe::sava {

namespace {

constexpr const char * kTag = "sava";

constexpr const char k_default_variant[] = "sava";

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

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m     = std::make_unique<Model>();
    m->arch    = &arch;
    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();

    if (auto st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    const size_t half = static_cast<size_t>(m->hparams.attn.head_dim / 2);
    if (load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "rope.freqs", half, kTag, m->rope_freqs) !=
        load_common::ReadF32Result::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: rope.freqs missing or not [%zu] F32", kTag, half);
        return TRANSCRIBE_ERR_GGUF;
    }

    // Host front end: the mel filterbank as stored ([n_mels][n_bins]), the
    // periodic Hann window and the FFT twiddles, all fp64.
    const HParams &    hp = m->hparams;
    std::vector<float> bank;
    if (load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.mel_filterbank",
                                             static_cast<size_t>(hp.n_mels) * static_cast<size_t>(hp.n_bins), kTag,
                                             bank) != load_common::ReadF32Result::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: frontend.mel_filterbank missing or not [%d, %d] F32", kTag, hp.n_mels,
                hp.n_bins);
        return TRANSCRIBE_ERR_GGUF;
    }
    m->mel_bank.assign(bank.begin(), bank.end());
    const double two_pi = 2.0 * 3.14159265358979323846;
    m->window.resize(static_cast<size_t>(hp.n_fft));
    m->twiddle_re.resize(static_cast<size_t>(hp.n_fft));
    m->twiddle_im.resize(static_cast<size_t>(hp.n_fft));
    for (int i = 0; i < hp.n_fft; ++i) {
        const double a                        = two_pi * i / hp.n_fft;
        m->window[static_cast<size_t>(i)]     = 0.5 - 0.5 * std::cos(a);
        m->twiddle_re[static_cast<size_t>(i)] = std::cos(a);
        m->twiddle_im[static_cast<size_t>(i)] = -std::sin(a);
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
    if (auto st = bind_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        return st;
    }

    // AUTO resolves to the CPU backend: the CPU path is the one validated
    // against the reference and the one where a stream is bit-identical to a
    // whole run. A specific backend or device is still honored.
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

    TurnModelInfo & ti         = m->turn_info;
    ti.max_audio_samples       = 0;
    ti.frame_samples           = m->hparams.frame_samples;
    ti.has_speech              = true;
    ti.speech_lookahead_frames = m->hparams.lookahead_frames;
    ti.defaults                = m->hparams.policy;

    m->roles = TRANSCRIBE_ROLE_VAD | TRANSCRIBE_ROLE_TURN;
    set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Validation dumps
// ---------------------------------------------------------------------------

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void dump(const char * name, const std::vector<float> & v, std::initializer_list<long long> shape, const char * stage) {
    std::vector<long long> s(shape);
    debug::dump_host_f32(name, v.data(), static_cast<long long>(v.size()), s.data(), static_cast<int>(s.size()), stage);
}

}  // namespace

// The stream's per-step tensors under the reference dumper's names
// (scripts/dump_reference_sava_author.py), and the run's products assuming
// the stream was flushed (its last lookahead_frames steps are padding).
void stream_dump(Stream & st, const Model & m) {
    DumpTrace & t = st.trace;
    if (!debug::enabled() || t.n_steps == 0) {
        t.clear();
        return;
    }
    const HParams & hp = m.hparams;
    const long long S  = t.n_steps;
    const long long H  = hp.n_horizons;
    dump("fe.mel", t.mel, { S, hp.mel_width }, "frontend");
    dump("fe.log_energy", t.log_energy, { S }, "frontend");
    dump("mixer.out", t.mixer, { S, hp.hidden }, "frontend");
    char name[64];
    for (size_t i = 0; i < t.trunk_mix.size(); ++i) {
        std::snprintf(name, sizeof(name), "trunk.%zu.mix", i);
        dump(name, t.trunk_mix[i], { S, hp.hidden }, "trunk");
        std::snprintf(name, sizeof(name), "trunk.%zu.out", i);
        dump(name, t.trunk_out[i], { S, hp.hidden }, "trunk");
        if (hp.layer_is_ssm[i]) {
            std::snprintf(name, sizeof(name), "trunk.%zu.ssm_y", i);
            dump(name, t.ssm_y[i], { S, hp.ssm_inner }, "trunk");
            const ggml_tensor * s = st.ssm[i].ssm;
            std::vector<float>  state(static_cast<size_t>(ggml_nelements(s)));
            ggml_backend_tensor_get(s, state.data(), 0, ggml_nbytes(s));
            std::snprintf(name, sizeof(name), "trunk.%zu.ssm_state", i);
            dump(name, state, { hp.ssm_heads, hp.ssm_head_dim, hp.ssm_state }, "trunk");
        }
    }
    dump("trunk.norm", t.trunk_norm, { S, hp.hidden }, "trunk");
    dump("head.vad.hidden", t.vad_hidden, { S, hp.vad_hidden }, "heads");
    dump("head.vad.logit", t.logit, { S }, "heads");
    dump("head.vad.logit_la", t.logit_la, { S }, "heads");
    dump("head.turn.speech_in", t.speech_in, { S }, "heads");
    dump("head.turn.input", t.turn_in, { S, hp.hidden + hp.vad_embed }, "heads");
    dump("head.turn.embed", t.turn_emb, { S, hp.vad_embed }, "heads");
    dump("head.turn.hidden", t.turn_hidden, { S, hp.turn_hidden }, "heads");
    dump("head.turn.logits", t.turn, { S, H }, "heads");

    std::vector<float> speech(static_cast<size_t>(S)), turn(t.turn.size()), on_time(static_cast<size_t>(S));
    for (size_t i = 0; i < speech.size(); ++i) {
        speech[i]  = sigmoid(t.logit_la[i]);
        on_time[i] = sigmoid(t.logit[i]);
    }
    for (size_t i = 0; i < turn.size(); ++i) {
        turn[i] = sigmoid(t.turn[i]);
    }
    dump("stream.speech", speech, { S }, "stream");
    dump("stream.turn", turn, { S, H }, "stream");
    const long long lag = hp.lookahead_frames;
    if (S > lag) {
        const long long    n = S - lag;
        std::vector<float> released(speech.begin() + lag, speech.end());
        std::vector<float> on(on_time.begin(), on_time.begin() + n);
        std::vector<float> real(turn.begin(), turn.begin() + n * H);
        std::vector<float> flush(turn.begin() + n * H, turn.end());
        dump("sava.speech", released, { n }, "output");
        dump("sava.speech_on_time", on, { n }, "output");
        dump("sava.turn", real, { n, H }, "output");
        dump("sava.flush_turns", flush, { lag, H }, "output");
    }
    t.clear();
}

namespace {

// ---------------------------------------------------------------------------
// TURN ops
// ---------------------------------------------------------------------------

const TurnModelInfo & turn_info(const transcribe_model * model) {
    return static_cast<const Model *>(model)->turn_info;
}

transcribe_turn_session * turn_new_session() {
    return new TurnSession();
}

void turn_reset(transcribe_turn_session * session) {
    auto &        s = static_cast<TurnSession &>(*session);
    const Model & m = static_cast<const Model &>(*session->model);
    stream_dump(s.st, m);
    stream_reset(s.st, m);
}

transcribe_status turn_step(transcribe_turn_session * session,
                            const float *             pcm,
                            int64_t                   n_frames,
                            float *                   turn,
                            float *                   speech) {
    auto & s = static_cast<TurnSession &>(*session);
    return stream_step(s, s.st, static_cast<const Model &>(*session->model), pcm, n_frames, turn, speech);
}

// One frame for each of count sessions in one graph, owned by sessions[0].
// With TRANSCRIBE_DUMP_DIR set the sessions are stepped one by one instead
// (the same values), so each stream's validation trace stays complete.
transcribe_status turn_step_batch(transcribe_turn_session * const * sessions,
                                  int                               count,
                                  const float * const *             frame_pcm,
                                  float *                           turn,
                                  float *                           speech) {
    const Model & m = static_cast<const Model &>(*sessions[0]->model);
    if (debug::init() && debug::enabled()) {
        for (int i = 0; i < count; ++i) {
            auto & s = static_cast<TurnSession &>(*sessions[i]);
            if (auto st = stream_step(s, s.st, m, frame_pcm[i], 1, turn + i, speech != nullptr ? speech + i : nullptr);
                st != TRANSCRIBE_OK) {
                return st;
            }
        }
        return TRANSCRIBE_OK;
    }
    std::vector<Stream *>      streams(static_cast<size_t>(count));
    std::vector<SessionCore *> cores(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        auto & s                        = static_cast<TurnSession &>(*sessions[i]);
        streams[static_cast<size_t>(i)] = &s.st;
        cores[static_cast<size_t>(i)]   = &s;
    }
    auto & first = static_cast<TurnSession &>(*sessions[0]);
    return stream_step_batch(first.batch, m, streams.data(), cores.data(), count, first.n_threads, frame_pcm, turn,
                             speech);
}

// The TURN_RUN extension: Inworld's segment rule, field by field; negative
// keeps the model's default.
transcribe_status turn_apply_run_ext(const transcribe_model *, const transcribe_ext * ext, TurnPolicy & policy) {
    constexpr uint64_t k_min = TRANSCRIBE_FIELD_END(transcribe_sava_turn_ext, min_confident_speech_ms);
    if (const auto st = transcribe_ext_check(ext, TRANSCRIBE_EXT_KIND_SAVA_TURN, k_min); st != TRANSCRIBE_OK) {
        return st;
    }
    const auto & e    = *reinterpret_cast<const transcribe_sava_turn_ext *>(ext);
    TurnPolicy   r    = policy;
    bool         ok   = true;
    auto         prob = [&ok](double v, double & dst) {
        if (std::isnan(v) || v > 1.0) {
            ok = false;
        } else if (v >= 0.0) {
            dst = v;
        }
    };
    auto ms = [](int32_t v, int32_t & dst) {
        if (v >= 0) {
            dst = v;
        }
    };
    prob(e.speech_threshold, r.speech_threshold);
    prob(e.confident_speech_threshold, r.confident_speech_threshold);
    ms(e.pre_roll_ms, r.pre_roll_ms);
    ms(e.min_silence_before_check_ms, r.min_silence_before_check_ms);
    ms(e.max_silence_ms, r.max_silence_ms);
    ms(e.min_speech_ms, r.min_speech_ms);
    ms(e.min_confident_speech_ms, r.min_confident_speech_ms);
    if (!ok) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    policy = r;
    return TRANSCRIBE_OK;
}

const TurnOps k_turn_ops = { turn_info, turn_new_session, nullptr,           turn_reset,
                             turn_step, turn_step_batch,  turn_apply_run_ext };

// ---------------------------------------------------------------------------
// VAD ops
// ---------------------------------------------------------------------------

int32_t vad_frame_samples(const transcribe_model * model) {
    return static_cast<const Model *>(model)->hparams.frame_samples;
}

transcribe_vad_session * vad_new_session() {
    return new VadSession();
}

void vad_reset(transcribe_vad_session * session) {
    auto &        s = static_cast<VadSession &>(*session);
    const Model & m = static_cast<const Model &>(*session->model);
    stream_dump(s.st, m);
    stream_reset(s.st, m);
}

transcribe_status vad_score(transcribe_vad_session * session, const float * pcm, int64_t n_frames, float * probs) {
    auto & s = static_cast<VadSession &>(*session);
    return stream_step(s, s.st, static_cast<const Model &>(*session->model), pcm, n_frames, nullptr, probs);
}

int32_t vad_lookahead_frames(const transcribe_model * model) {
    return static_cast<const Model *>(model)->hparams.lookahead_frames;
}

const VadOps k_vad_ops = { vad_frame_samples, vad_new_session, vad_reset, vad_score, vad_lookahead_frames };

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

static bool accepts_ext_kind(const transcribe_model *, transcribe_ext_slot slot, uint32_t kind) {
    return slot == TRANSCRIBE_EXT_SLOT_TURN_RUN && kind == TRANSCRIBE_EXT_KIND_SAVA_TURN;
}

const Arch arch = {
    /* .name             = */ "sava",
    /* .load             = */ load,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ accepts_ext_kind,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ nullptr,
    /* .vad              = */ &k_vad_ops,
    /* .turn             = */ &k_turn_ops,
};

}  // namespace transcribe::sava

extern "C" void transcribe_sava_turn_ext_init(struct transcribe_sava_turn_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size                    = sizeof(*p);
    p->ext.kind                    = TRANSCRIBE_EXT_KIND_SAVA_TURN;
    p->speech_threshold            = -1.0;
    p->confident_speech_threshold  = -1.0;
    p->pre_roll_ms                 = -1;
    p->min_silence_before_check_ms = -1;
    p->max_silence_ms              = -1;
    p->min_speech_ms               = -1;
    p->min_confident_speech_ms     = -1;
}
