// Turn segmentation adapted from Inworld SAVA (inworld.sava.segmenter).
// Copyright 2026 Inworld AI. Apache License 2.0.

// transcribe-turn.cpp - TURN role C ABI (include/transcribe/turn.h): input
// validation, cropping for clip models, frame buffering and the lagged speech
// track for streaming models, flushing, and the whole-run turn segmenter.

#include "transcribe-turn.h"

#include "transcribe-abi.h"
#include "transcribe-api-guard.h"
#include "transcribe-arch.h"
#include "transcribe-model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>

using transcribe::api_guard_status;
using transcribe::api_guard_void;
using transcribe::check_input_struct_size;
using transcribe::check_struct_size;
using transcribe::copy_out_prefix;
using transcribe::TurnModelInfo;
using transcribe::TurnPolicy;
using transcribe::TurnSegmentEntry;

namespace {

constexpr size_t k_min_info_size           = TRANSCRIBE_FIELD_END(transcribe_turn_info, threshold);
constexpr size_t k_min_session_params_size = TRANSCRIBE_FIELD_END(transcribe_turn_session_params, n_threads);
constexpr size_t k_min_params_size         = TRANSCRIBE_FIELD_END(transcribe_turn_params, family);
constexpr size_t k_min_result_size         = TRANSCRIBE_FIELD_END(transcribe_turn_result, n_segments);
constexpr size_t k_min_segment_size        = TRANSCRIBE_FIELD_END(transcribe_turn_segment, end_reason);
constexpr size_t k_min_timings_size        = TRANSCRIBE_FIELD_END(transcribe_timings, decode_ms);

constexpr int64_t k_sample_rate = 16000;

const transcribe::TurnOps * turn_ops(const transcribe_model * model) {
    return model != nullptr && (model->roles & TRANSCRIBE_ROLE_TURN) != 0 ? model->arch->turn : nullptr;
}

bool streams(const TurnModelInfo & info) {
    return info.frame_samples > 0;
}

// Speech lag the dispatcher aligns; 0 when the model scores no speech.
int64_t speech_lag(const TurnModelInfo & info) {
    return info.has_speech ? info.speech_lookahead_frames : 0;
}

void clear_result(transcribe_turn_session * s) {
    s->p     = 0.0f;
    s->ended = false;
    s->probs.clear();
    s->first_frame = 0;
    s->speech_probs.clear();
    s->speech_first_frame = 0;
    s->segments.clear();
    s->t_mel_us    = 0;
    s->t_encode_us = 0;
    s->t_decode_us = 0;
}

void reset_stream(transcribe_turn_session * s, const transcribe::TurnOps * ops) {
    s->pending.clear();
    s->frames_done = 0;
    if (ops->reset != nullptr) {
        s->needs_reset = true;
        ops->reset(s);
    }
    s->needs_reset = false;
}

// After validation, every failed call (including an exception) discards its
// partial result and stream state. Cleanup itself must not escape the ABI.
struct TurnFailureGuard {
    transcribe_turn_session *   session;
    const transcribe::TurnOps * ops;
    bool                        complete = false;

    ~TurnFailureGuard() noexcept {
        if (!complete) {
            clear_result(session);
            api_guard_void("transcribe_turn_cleanup", [&] { reset_stream(session, ops); });
        }
    }
};

// Step n_frames whole frames as the continuation of the stream. Appends every
// step's turn score to turn_out and the speech scores of judged frames (frame
// index >= 0) to speech_out.
transcribe_status step_frames(transcribe_turn_session *   s,
                              const transcribe::TurnOps * ops,
                              const TurnModelInfo &       info,
                              const float *               pcm,
                              int64_t                     n_frames,
                              std::vector<float> &        turn_out,
                              std::vector<float> &        speech_out) {
    if (n_frames <= 0) {
        return TRANSCRIBE_OK;
    }
    const size_t       n = static_cast<size_t>(n_frames);
    std::vector<float> turn(n, 0.0f);
    std::vector<float> speech(info.has_speech ? n : 0, 0.0f);
    if (const auto st = ops->step(s, pcm, n_frames, turn.data(), info.has_speech ? speech.data() : nullptr);
        st != TRANSCRIBE_OK) {
        return st;
    }
    turn_out.insert(turn_out.end(), turn.begin(), turn.end());
    if (info.has_speech) {
        const int64_t lag = speech_lag(info);
        for (int64_t i = 0; i < n_frames; ++i) {
            if (s->frames_done + i - lag >= 0) {
                speech_out.push_back(speech[static_cast<size_t>(i)]);
            }
        }
    }
    s->frames_done += n_frames;
    return TRANSCRIBE_OK;
}

// Python's round() on a nonnegative double: half to even.
int64_t round_half_even(double x) {
    return static_cast<int64_t>(std::nearbyint(x));
}

}  // namespace

transcribe_status transcribe::resolve_turn_params(const transcribe_model *       model,
                                                  const transcribe_turn_params * params,
                                                  TurnPolicy &                   out) {
    transcribe_turn_params defaults;
    transcribe_turn_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::TurnOps * ops = turn_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    TurnPolicy   r   = ops->info(model).defaults;
    const double thr = params->threshold;
    if (std::isnan(thr) || thr > 1.0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (thr >= 0.0) {
        r.threshold = thr;
    }
    if (params->family != nullptr) {
        if (params->family->size < sizeof(struct transcribe_ext)) {
            return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
        }
        if (ops->apply_run_ext == nullptr ||
            !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_TURN_RUN, params->family->kind)) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        if (const auto st = ops->apply_run_ext(model, params->family, r); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    out = r;
    return TRANSCRIBE_OK;
}

void transcribe::turn_frames_to_segments(const float *                   speech,
                                         const float *                   turn_at_step,
                                         int64_t                         n_frames,
                                         int64_t                         n_samples,
                                         int32_t                         frame_samples,
                                         int32_t                         lag,
                                         const TurnPolicy &              policy,
                                         std::vector<TurnSegmentEntry> & out) {
    out.clear();
    const double  frame_seconds = static_cast<double>(frame_samples) / static_cast<double>(k_sample_rate);
    const int64_t frame_ms      = round_half_even(1000.0 * frame_seconds);
    const int64_t pre_roll      = round_half_even(policy.pre_roll_ms / (1000.0 * frame_seconds));

    bool    open         = false;
    int64_t start        = 0;
    int64_t speech_ms    = 0;
    int64_t confident_ms = 0;
    int64_t pause_frames = 0;
    int64_t last_judged  = 0;

    auto close = [&](int64_t end_frame, int32_t reason) {
        const bool kept = speech_ms >= policy.min_speech_ms || confident_ms >= policy.min_confident_speech_ms;
        if (kept) {
            TurnSegmentEntry e;
            e.start_sample = std::min(start * frame_samples, n_samples);
            e.end_sample   = std::min(end_frame * frame_samples, n_samples);
            e.end_reason   = reason;
            out.push_back(e);
        }
        open = false;
    };

    for (int64_t judged = 0; judged < n_frames; ++judged) {
        const int64_t now    = judged + lag;
        const double  sp     = static_cast<double>(speech[judged]);
        const double  tp     = static_cast<double>(turn_at_step[now]);
        last_judged          = judged;
        const bool is_speech = sp >= policy.speech_threshold;
        if (!open) {
            if (!is_speech) {
                continue;
            }
            open         = true;
            start        = std::max<int64_t>(0, judged - pre_roll);
            speech_ms    = 0;
            confident_ms = 0;
            pause_frames = 0;
        }
        if (is_speech) {
            pause_frames = 0;
            speech_ms += frame_ms;
            if (sp >= policy.confident_speech_threshold) {
                confident_ms += frame_ms;
            }
            continue;
        }
        pause_frames += 1;
        const int64_t pause_ms = pause_frames * frame_ms;
        if (pause_ms >= policy.max_silence_ms) {
            close(now + 1, TRANSCRIBE_TURN_END_SILENCE);
            continue;
        }
        if (pause_ms >= policy.min_silence_before_check_ms && tp >= policy.threshold) {
            close(now + 1, TRANSCRIBE_TURN_END_MODEL);
        }
    }
    if (open) {
        close(last_judged + 1, TRANSCRIBE_TURN_END_AUDIO);
    }
}

extern "C" void transcribe_turn_info_init(struct transcribe_turn_info * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_turn_session_params_init(struct transcribe_turn_session_params * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_turn_params_init(struct transcribe_turn_params * p) {
    transcribe::init_sized(p);
    if (p != nullptr) {
        p->threshold = -1.0;
        p->family    = nullptr;
    }
}

extern "C" void transcribe_turn_result_init(struct transcribe_turn_result * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_turn_segment_init(struct transcribe_turn_segment * p) {
    transcribe::init_sized(p);
}

static transcribe_status turn_get_info_impl(const transcribe_model * model, transcribe_turn_info * out) {
    if (model == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_info_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::TurnOps * ops = turn_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    const TurnModelInfo & info = ops->info(model);
    transcribe_turn_info  staged{};
    staged.struct_size   = out->struct_size;
    staged.sample_rate   = static_cast<int32_t>(k_sample_rate);
    staged.max_audio_ms  = static_cast<int32_t>(static_cast<int64_t>(info.max_audio_samples) * 1000 / k_sample_rate);
    staged.frame_samples = info.frame_samples;
    staged.has_speech    = info.has_speech;
    staged.speech_lookahead_frames = static_cast<int32_t>(speech_lag(info));
    staged.threshold               = info.defaults.threshold;
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

static transcribe_status turn_session_init_impl(transcribe_model *                     model,
                                                const transcribe_turn_session_params * params,
                                                transcribe_turn_session **             out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out = nullptr;
    if (model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::TurnOps * ops = turn_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_turn_session_params defaults;
    transcribe_turn_session_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_session_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    if (params->n_threads < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    std::unique_ptr<transcribe_turn_session> s(ops->new_session());
    s->model     = model;
    s->n_threads = params->n_threads;
    if (ops->reset != nullptr) {
        ops->reset(s.get());
    }
    *out = s.release();
    return TRANSCRIBE_OK;
}

static transcribe_status turn_run_impl(transcribe_turn_session *      session,
                                       const float *                  pcm,
                                       int                            n_samples,
                                       const transcribe_turn_params * params) {
    // Everything up to the commit point leaves the stream and previous result intact.
    if (session == nullptr || pcm == nullptr || n_samples <= 0 || !transcribe::pcm_is_finite(pcm, n_samples)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::TurnOps * ops  = turn_ops(session->model);
    const TurnModelInfo &       info = ops->info(session->model);
    TurnPolicy                  policy;
    if (const auto st = transcribe::resolve_turn_params(session->model, params, policy); st != TRANSCRIBE_OK) {
        return st;
    }

    TurnFailureGuard failure{ session, ops };
    clear_result(session);
    reset_stream(session, ops);
    transcribe::ScratchReleaseGuard release{ session, true };

    // Crop to the last max_audio_samples.
    int64_t n = n_samples;
    if (info.max_audio_samples > 0 && n > info.max_audio_samples) {
        pcm += n - info.max_audio_samples;
        n = info.max_audio_samples;
    }

    if (!streams(info)) {
        float p = 0.0f;
        if (const auto st = ops->score_clip(session, pcm, n, &p); st != TRANSCRIBE_OK) {
            return st;
        }
        if (!std::isfinite(p)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        session->p       = p;
        session->ended   = static_cast<double>(p) >= policy.threshold;
        failure.complete = true;
        return TRANSCRIBE_OK;
    }

    // A streaming model: the clip is one stream, fed whole and flushed.
    const int64_t frame   = info.frame_samples;
    const int64_t lag     = speech_lag(info);
    const int64_t n_full  = n / frame;
    const int64_t n_tail  = n - n_full * frame;
    const int64_t n_total = n_full + (n_tail > 0 ? 1 : 0);

    std::vector<float> turn_all;
    std::vector<float> speech;
    turn_all.reserve(static_cast<size_t>(n_total + lag));
    speech.reserve(static_cast<size_t>(n_total));
    transcribe_status st = step_frames(session, ops, info, pcm, n_full, turn_all, speech);
    if (st == TRANSCRIBE_OK && n_tail > 0) {
        std::vector<float> last(static_cast<size_t>(frame), 0.0f);
        std::memcpy(last.data(), pcm + n_full * frame, static_cast<size_t>(n_tail) * sizeof(float));
        st = step_frames(session, ops, info, last.data(), 1, turn_all, speech);
    }
    if (st == TRANSCRIBE_OK && lag > 0) {
        const std::vector<float> silence(static_cast<size_t>(lag * frame), 0.0f);
        st = step_frames(session, ops, info, silence.data(), lag, turn_all, speech);
    }
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    reset_stream(session, ops);

    session->probs.assign(turn_all.begin(), turn_all.begin() + n_total);
    session->p     = session->probs.back();
    session->ended = static_cast<double>(session->p) >= policy.threshold;
    if (info.has_speech) {
        session->speech_probs = std::move(speech);
        transcribe::turn_frames_to_segments(session->speech_probs.data(), turn_all.data(), n_total, n,
                                            info.frame_samples, static_cast<int32_t>(lag), policy, session->segments);
    }
    failure.complete = true;
    return TRANSCRIBE_OK;
}

static transcribe_status turn_stream_feed_impl(transcribe_turn_session * session, const float * pcm, int n_samples) {
    if (session == nullptr || n_samples < 0 || (n_samples > 0 && pcm == nullptr) ||
        (n_samples > 0 && !transcribe::pcm_is_finite(pcm, n_samples))) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::TurnOps * ops  = turn_ops(session->model);
    const TurnModelInfo &       info = ops->info(session->model);
    if (!streams(info)) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    const size_t frame = static_cast<size_t>(info.frame_samples);

    TurnFailureGuard failure{ session, ops };
    clear_result(session);
    if (session->needs_reset) {
        reset_stream(session, ops);
    }
    session->first_frame        = session->frames_done;
    session->speech_first_frame = std::max<int64_t>(0, session->frames_done - speech_lag(info));
    if (n_samples == 0) {
        failure.complete = true;
        return TRANSCRIBE_OK;
    }

    // Whole frames of pending + pcm; the remainder stays pending.
    const float *      src   = pcm;
    size_t             avail = static_cast<size_t>(n_samples);
    std::vector<float> joined;
    if (!session->pending.empty()) {
        joined.reserve(session->pending.size() + avail);
        joined.assign(session->pending.begin(), session->pending.end());
        joined.insert(joined.end(), pcm, pcm + avail);
        src   = joined.data();
        avail = joined.size();
    }
    const size_t n_frames = avail / frame;
    if (const auto st =
            step_frames(session, ops, info, src, static_cast<int64_t>(n_frames), session->probs, session->speech_probs);
        st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<float> rest(src + n_frames * frame, src + avail);
    session->pending.swap(rest);
    failure.complete = true;
    return TRANSCRIBE_OK;
}

static transcribe_status turn_stream_feed_batch_impl(transcribe_turn_session * const * sessions,
                                                     const float * const *             pcm,
                                                     const int *                       n_samples,
                                                     int                               n) {
    // Validate every session and input before touching any of them.
    if (sessions == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0 || sessions[0] == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe_model * model = sessions[0]->model;
    for (int i = 0; i < n; ++i) {
        const transcribe_turn_session * s = sessions[i];
        if (s == nullptr || s->model != model || n_samples[i] < 0 || (n_samples[i] > 0 && pcm[i] == nullptr) ||
            (n_samples[i] > 0 && !transcribe::pcm_is_finite(pcm[i], n_samples[i]))) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        for (int j = 0; j < i; ++j) {
            if (sessions[j] == s) {
                return TRANSCRIBE_ERR_INVALID_ARG;
            }
        }
    }
    const transcribe::TurnOps * ops  = turn_ops(model);
    const TurnModelInfo &       info = ops->info(model);
    if (!streams(info)) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    const size_t  frame = static_cast<size_t>(info.frame_samples);
    const int64_t lag   = speech_lag(info);
    const size_t  count = static_cast<size_t>(n);

    // One guard per session: a failure anywhere resets every listed stream.
    std::vector<std::unique_ptr<TurnFailureGuard>> guards;
    guards.reserve(count);
    std::vector<std::vector<float>> joined(count);
    std::vector<const float *>      src(count);
    std::vector<size_t>             avail(count);
    std::vector<int64_t>            n_frames(count);
    int64_t                         max_frames = 0;
    for (size_t i = 0; i < count; ++i) {
        transcribe_turn_session * s = sessions[i];
        guards.emplace_back(new TurnFailureGuard{ s, ops });
        clear_result(s);
        if (s->needs_reset) {
            reset_stream(s, ops);
        }
        s->first_frame        = s->frames_done;
        s->speech_first_frame = std::max<int64_t>(0, s->frames_done - lag);
        src[i]                = pcm[i];
        avail[i]              = static_cast<size_t>(n_samples[i]);
        if (!s->pending.empty()) {
            joined[i].assign(s->pending.begin(), s->pending.end());
            if (avail[i] > 0) {
                joined[i].insert(joined[i].end(), pcm[i], pcm[i] + avail[i]);
            }
            src[i]   = joined[i].data();
            avail[i] = joined[i].size();
        }
        n_frames[i] = static_cast<int64_t>(avail[i] / frame);
        max_frames  = std::max(max_frames, n_frames[i]);
    }

    if (ops->step_batch == nullptr) {
        // No batched path: step each session's frames in turn.
        for (size_t i = 0; i < count; ++i) {
            if (const auto st = step_frames(sessions[i], ops, info, src[i], n_frames[i], sessions[i]->probs,
                                            sessions[i]->speech_probs);
                st != TRANSCRIBE_OK) {
                return st;
            }
        }
    } else {
        // Frame k of every session that has one, in one compute.
        std::vector<transcribe_turn_session *> active;
        std::vector<const float *>             frames;
        std::vector<float>                     turn;
        std::vector<float>                     speech;
        for (int64_t k = 0; k < max_frames; ++k) {
            active.clear();
            frames.clear();
            for (size_t i = 0; i < count; ++i) {
                if (n_frames[i] > k) {
                    active.push_back(sessions[i]);
                    frames.push_back(src[i] + static_cast<size_t>(k) * frame);
                }
            }
            turn.assign(active.size(), 0.0f);
            speech.assign(active.size(), 0.0f);
            if (const auto st = ops->step_batch(active.data(), static_cast<int>(active.size()), frames.data(),
                                                turn.data(), info.has_speech ? speech.data() : nullptr);
                st != TRANSCRIBE_OK) {
                return st;
            }
            for (size_t j = 0; j < active.size(); ++j) {
                transcribe_turn_session * s = active[j];
                s->probs.push_back(turn[j]);
                if (info.has_speech && s->frames_done - lag >= 0) {
                    s->speech_probs.push_back(speech[j]);
                }
                s->frames_done += 1;
            }
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const size_t       used = static_cast<size_t>(n_frames[i]) * frame;
        std::vector<float> rest(src[i] + used, src[i] + avail[i]);
        sessions[i]->pending.swap(rest);
    }
    for (auto & g : guards) {
        g->complete = true;
    }
    return TRANSCRIBE_OK;
}

static transcribe_status turn_stream_flush_impl(transcribe_turn_session * session) {
    if (session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::TurnOps * ops  = turn_ops(session->model);
    const TurnModelInfo &       info = ops->info(session->model);
    if (!streams(info)) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    const size_t  frame = static_cast<size_t>(info.frame_samples);
    const int64_t lag   = speech_lag(info);

    TurnFailureGuard failure{ session, ops };
    clear_result(session);
    if (session->needs_reset) {
        reset_stream(session, ops);
    }
    session->first_frame        = session->frames_done;
    session->speech_first_frame = std::max<int64_t>(0, session->frames_done - lag);
    transcribe_status st        = TRANSCRIBE_OK;
    if (!session->pending.empty()) {
        std::vector<float> last(frame, 0.0f);
        std::copy(session->pending.begin(), session->pending.end(), last.begin());
        st = step_frames(session, ops, info, last.data(), 1, session->probs, session->speech_probs);
    }
    if (st == TRANSCRIBE_OK && lag > 0 && session->frames_done > 0) {
        // Silent steps so the last frames get their speech verdicts; their
        // own turn scores score padding and are not returned.
        const std::vector<float> silence(static_cast<size_t>(lag) * frame, 0.0f);
        std::vector<float>       padding_turn;
        st = step_frames(session, ops, info, silence.data(), lag, padding_turn, session->speech_probs);
    }
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    reset_stream(session, ops);
    failure.complete = true;
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_turn_set_abort_callback(struct transcribe_turn_session * session,
                                                   transcribe_abort_callback        cb,
                                                   void *                           user_data) {
    if (session != nullptr) {
        session->abort_cb       = cb;
        session->abort_userdata = user_data;
    }
}

extern "C" transcribe_status transcribe_turn_get_result(const struct transcribe_turn_session * session,
                                                        struct transcribe_turn_result *        out) {
    if (session == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_result_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe_turn_result staged{};
    staged.struct_size        = out->struct_size;
    staged.p                  = session->p;
    staged.ended              = session->ended;
    staged.n_probs            = static_cast<int32_t>(session->probs.size());
    staged.first_frame        = session->first_frame;
    staged.n_speech_probs     = static_cast<int32_t>(session->speech_probs.size());
    staged.speech_first_frame = session->speech_first_frame;
    staged.n_segments         = static_cast<int32_t>(session->segments.size());
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" const float * transcribe_turn_probs(const struct transcribe_turn_session * session) {
    return session != nullptr && !session->probs.empty() ? session->probs.data() : nullptr;
}

extern "C" const float * transcribe_turn_speech_probs(const struct transcribe_turn_session * session) {
    return session != nullptr && !session->speech_probs.empty() ? session->speech_probs.data() : nullptr;
}

extern "C" transcribe_status transcribe_turn_get_segment(const struct transcribe_turn_session * session,
                                                         int                                    i,
                                                         struct transcribe_turn_segment *       out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_segment_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe_turn_segment staged{};
    staged.struct_size = out->struct_size;
    if (session != nullptr && i >= 0 && static_cast<size_t>(i) < session->segments.size()) {
        const TurnSegmentEntry & e = session->segments[static_cast<size_t>(i)];
        staged.start_sample        = e.start_sample;
        staged.end_sample          = e.end_sample;
        staged.end_reason          = e.end_reason;
    }
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_turn_get_timings(const struct transcribe_turn_session * session,
                                                         struct transcribe_timings *            out) {
    if (session == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_timings_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe::copy_out_timings(session->model->t_load_us, session->t_mel_us, session->t_encode_us,
                                 session->t_decode_us, out);
    return TRANSCRIBE_OK;
}

// C ABI forwarders for the entry points that allocate, compute, or transfer
// ownership; the ones above are nothrow by construction.

extern "C" transcribe_status transcribe_turn_get_info(const struct transcribe_model * model,
                                                      struct transcribe_turn_info *   out) {
    return api_guard_status("transcribe_turn_get_info", [&] { return turn_get_info_impl(model, out); });
}

extern "C" transcribe_status transcribe_turn_session_init(struct transcribe_model *                     model,
                                                          const struct transcribe_turn_session_params * params,
                                                          struct transcribe_turn_session **             out) {
    return api_guard_status("transcribe_turn_session_init", [&] { return turn_session_init_impl(model, params, out); });
}

extern "C" void transcribe_turn_session_free(struct transcribe_turn_session * session) {
    api_guard_void("transcribe_turn_session_free", [&] { delete session; });
}

extern "C" transcribe_status transcribe_turn_run(struct transcribe_turn_session *      session,
                                                 const float *                         pcm,
                                                 int                                   n_samples,
                                                 const struct transcribe_turn_params * params) {
    return api_guard_status("transcribe_turn_run", [&] { return turn_run_impl(session, pcm, n_samples, params); });
}

extern "C" transcribe_status transcribe_turn_stream_feed(struct transcribe_turn_session * session,
                                                         const float *                    pcm,
                                                         int                              n_samples) {
    return api_guard_status("transcribe_turn_stream_feed",
                            [&] { return turn_stream_feed_impl(session, pcm, n_samples); });
}

extern "C" transcribe_status transcribe_turn_stream_feed_batch(struct transcribe_turn_session * const * sessions,
                                                               const float * const *                    pcm,
                                                               const int *                              n_samples,
                                                               int                                      n) {
    return api_guard_status("transcribe_turn_stream_feed_batch",
                            [&] { return turn_stream_feed_batch_impl(sessions, pcm, n_samples, n); });
}

extern "C" transcribe_status transcribe_turn_stream_flush(struct transcribe_turn_session * session) {
    return api_guard_status("transcribe_turn_stream_flush", [&] { return turn_stream_flush_impl(session); });
}

extern "C" void transcribe_turn_stream_reset(struct transcribe_turn_session * session) {
    api_guard_void("transcribe_turn_stream_reset", [&] {
        if (session != nullptr) {
            reset_stream(session, turn_ops(session->model));
        }
    });
}
