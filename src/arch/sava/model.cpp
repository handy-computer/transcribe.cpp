// arch/sava/model.cpp - SAVA family handler (TURN and VAD roles): load, the
// TURN and VAD ops tables, and the validation dumps. Both roles step the same
// stream (graph.cpp): TURN reports the horizon-0 turn and the speech
// probability of every step, VAD the speech probability only.

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

#include <algorithm>
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
    // against the reference and bit-identical across chunkings. A specific
    // backend or device is still honored.
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
    ti.threshold               = m->hparams.threshold;

    m->roles = TRANSCRIBE_ROLE_VAD | TRANSCRIBE_ROLE_TURN;
    set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Validation dumps
// ---------------------------------------------------------------------------

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

// TurnOps::reset / VadOps::reset return nothing: a failed reset (the state
// buffer could not be allocated) is logged here, and the stream's next step
// retries the allocation and returns its status (stream_step).
void reset_logged(Stream & st, const Model & m) {
    stream_dump(st, m);
    if (const transcribe_status rc = stream_reset(st, m); rc != TRANSCRIBE_OK) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: stream reset failed (status %d); the next step retries it", kTag,
                static_cast<int>(rc));
    }
}

void turn_reset(transcribe_turn_session * session) {
    reset_logged(static_cast<TurnSession &>(*session).st, static_cast<const Model &>(*session->model));
}

transcribe_status turn_step(transcribe_turn_session * session,
                            const float *             pcm,
                            int64_t                   n_frames,
                            float *                   turn,
                            float *                   speech) {
    auto & s = static_cast<TurnSession &>(*session);
    return stream_step(s, s.st, static_cast<const Model &>(*session->model), pcm, n_frames, turn, speech);
}

// Streams per batched compute: 64, fewer past 6 layers (trunk and the two
// head attentions). ggml-alloc's free blocks run out at about 120 streams of
// 6 layers (the real model) and 1000 stream-layers of the toy fixture.
// Every column's values are independent of the column count, so chunks give
// the same bits as one graph.
int max_batch_streams(const Model & m) {
    const int n_layers = static_cast<int>(m.hparams.layer_is_ssm.size()) + 2;
    return std::clamp(384 / n_layers, 1, 64);
}

// One frame for each of count sessions, in chunks of max_batch_streams;
// each chunk's graph is cached by its first session. With TRANSCRIBE_DUMP_DIR
// set the sessions step one by one (the same values, the batch's thread
// count) so each stream's trace stays complete.
transcribe_status turn_step_batch(transcribe_turn_session * const * sessions,
                                  int                               count,
                                  int                               n_threads,
                                  const float * const *             frame_pcm,
                                  float *                           turn,
                                  float *                           speech) {
    const Model & m = static_cast<const Model &>(*sessions[0]->model);
    if (debug::init() && debug::enabled()) {
        for (int i = 0; i < count; ++i) {
            auto &    s     = static_cast<TurnSession &>(*sessions[i]);
            const int saved = s.n_threads;
            s.n_threads     = n_threads;
            const auto st =
                stream_step(s, s.st, m, frame_pcm[i], 1, turn + i, speech != nullptr ? speech + i : nullptr);
            s.n_threads = saved;
            if (st != TRANSCRIBE_OK) {
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
        // Every check before the first chunk steps, so a failed or aborted
        // batch leaves no chunk one frame ahead.
        if (auto st = ensure_state(s.st, m); st != TRANSCRIBE_OK) {
            return st;
        }
        if (s.poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
    }
    const int chunk = max_batch_streams(m);
    for (int c0 = 0; c0 < count; c0 += chunk) {
        const int    nc    = std::min(chunk, count - c0);
        auto &       first = static_cast<TurnSession &>(*sessions[c0]);
        const size_t o     = static_cast<size_t>(c0);
        if (auto st = step(first.batch, m, streams.data() + o, cores.data() + o, nc, 1, n_threads, frame_pcm + o,
                           turn + o, speech != nullptr ? speech + o : nullptr);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    return TRANSCRIBE_OK;
}

const TurnOps k_turn_ops = { turn_info, turn_new_session, nullptr, turn_reset, turn_step, turn_step_batch };

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
    reset_logged(static_cast<VadSession &>(*session).st, static_cast<const Model &>(*session->model));
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
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ nullptr,
    /* .vad              = */ &k_vad_ops,
    /* .turn             = */ &k_turn_ops,
};

}  // namespace transcribe::sava
