// Probability segmentation adapted from Silero VAD 6.2.3.
// Copyright (c) 2020-present Silero Team
// MIT license: src/third_party/silero_vad/LICENSE (distributed with artifacts).

// transcribe-vad.cpp - VAD role C ABI (include/transcribe/vad.h): input
// validation, frame buffering for streaming, the zero-padded last frame of
// an offline run, and probabilities -> speech segments.

#include "transcribe-vad.h"

#include "transcribe-abi.h"
#include "transcribe-api-guard.h"
#include "transcribe-arch.h"
#include "transcribe-model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>

using transcribe::api_guard_status;
using transcribe::api_guard_void;
using transcribe::check_input_struct_size;
using transcribe::check_struct_size;
using transcribe::copy_out_prefix;
using transcribe::VadSegmentEntry;
using transcribe::VadSegmentParams;

namespace {

constexpr size_t k_min_info_size           = TRANSCRIBE_FIELD_END(transcribe_vad_info, frame_samples);
constexpr size_t k_min_session_params_size = TRANSCRIBE_FIELD_END(transcribe_vad_session_params, n_threads);
constexpr size_t k_min_params_size         = TRANSCRIBE_FIELD_END(transcribe_vad_params, max_speech_ms);
constexpr size_t k_min_result_size         = TRANSCRIBE_FIELD_END(transcribe_vad_result, n_segments);
constexpr size_t k_min_segment_size        = TRANSCRIBE_FIELD_END(transcribe_vad_segment, end_sample);
constexpr size_t k_min_timings_size        = TRANSCRIBE_FIELD_END(transcribe_timings, decode_ms);

constexpr double k_sample_rate = 16000.0;

const transcribe::VadOps * vad_ops(const transcribe_model * model) {
    return model != nullptr && (model->roles & TRANSCRIBE_ROLE_VAD) != 0 ? model->arch->vad : nullptr;
}

// Python's // on ints (floor, not truncation).
int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

void clear_result(transcribe_vad_session * s) {
    s->probs.clear();
    s->segments.clear();
    s->first_frame = 0;
    s->t_mel_us    = 0;
    s->t_encode_us = 0;
    s->t_decode_us = 0;
}

void reset_stream(transcribe_vad_session * s, const transcribe::VadOps * ops) {
    s->pending.clear();
    s->frames_done = 0;
    s->needs_reset = true;
    ops->reset(s);
    s->needs_reset = false;
}

// After validation, every failed call (including an exception) discards its
// partial result and recurrent state. Cleanup itself must not escape the ABI.
struct VadFailureGuard {
    transcribe_vad_session *   session;
    const transcribe::VadOps * ops;
    bool                       complete = false;

    ~VadFailureGuard() noexcept {
        if (!complete) {
            clear_result(session);
            api_guard_void("transcribe_vad_cleanup", [&] { reset_stream(session, ops); });
        }
    }
};

}  // namespace

transcribe_status transcribe::resolve_vad_params(const transcribe_vad_params * params, VadSegmentParams & out) {
    transcribe_vad_params defaults;
    transcribe_vad_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const double thr = params->threshold;
    const double neg = params->neg_threshold;
    if (!std::isfinite(thr) || !std::isfinite(neg) || thr < 0.0 || thr > 1.0 || neg > thr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (params->min_speech_ms < 0 || params->min_silence_ms < 0 || params->speech_pad_ms < 0 ||
        params->max_speech_ms < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    VadSegmentParams r;
    r.threshold      = thr;
    // The reference's default: max(threshold - 0.15, 0.01).
    r.neg_threshold  = neg < 0.0 ? std::max(thr - 0.15, 0.01) : neg;
    r.min_speech_ms  = params->min_speech_ms;
    r.min_silence_ms = params->min_silence_ms;
    r.speech_pad_ms  = params->speech_pad_ms;
    r.max_speech_ms  = params->max_speech_ms;
    out              = r;
    return TRANSCRIBE_OK;
}

// A line-by-line port of silero_vad.utils_vad.get_speech_timestamps_from_probs
// (silero-vad 6.2.3) with return_seconds=False and step=1. Sample positions
// are integers held in int64; the Python float thresholds (sample counts
// derived from ms) are doubles. Python truthiness of a position (`if
// temp_end:`) is a != 0 test, kept as is.
void transcribe::vad_probs_to_segments(const float *                  probs,
                                       int64_t                        n_probs,
                                       int64_t                        n_samples,
                                       int32_t                        frame_samples,
                                       const VadSegmentParams &       p,
                                       std::vector<VadSegmentEntry> & out) {
    out.clear();
    const int64_t window = frame_samples;

    const double min_speech_samples = k_sample_rate * p.min_speech_ms / 1000.0;
    const double speech_pad_samples = k_sample_rate * p.speech_pad_ms / 1000.0;
    // The reference takes max_speech_duration_s in seconds: sr * (ms / 1000).
    const double max_speech_samples = p.max_speech_ms > 0 ? k_sample_rate * (p.max_speech_ms / 1000.0) -
                                                                static_cast<double>(window) - 2.0 * speech_pad_samples :
                                                            std::numeric_limits<double>::infinity();
    const double min_silence_samples               = k_sample_rate * p.min_silence_ms / 1000.0;
    const double min_silence_samples_at_max_speech = k_sample_rate * p.min_silence_at_max_speech_ms / 1000.0;

    struct PossibleEnd {
        int64_t end;
        int64_t dur;
    };

    bool                     triggered = false;
    VadSegmentEntry          cur;                 // current_speech
    bool                     cur_has    = false;  // current_speech is non-empty
    int64_t                  temp_end   = 0;
    int64_t                  prev_end   = 0;
    int64_t                  next_start = 0;
    std::vector<PossibleEnd> possible_ends;

    auto drop_current = [&]() {
        cur     = VadSegmentEntry{};
        cur_has = false;
    };

    for (int64_t i = 0; i < n_probs; ++i) {
        const double  prob       = static_cast<double>(probs[i]);
        const int64_t cur_sample = window * i;

        // Speech returns after a temp_end: record the silence if long enough.
        if (prob >= p.threshold && temp_end != 0) {
            const int64_t sil_dur = cur_sample - temp_end;
            if (static_cast<double>(sil_dur) > min_silence_samples_at_max_speech) {
                possible_ends.push_back({ temp_end, sil_dur });
            }
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cur_sample;
            }
        }

        // Start of speech.
        if (prob >= p.threshold && !triggered) {
            triggered        = true;
            cur.start_sample = cur_sample;
            cur_has          = true;
            continue;
        }

        // Max speech length reached: decide where to cut.
        if (triggered && static_cast<double>(cur_sample - cur.start_sample) > max_speech_samples) {
            if (p.use_max_possible_silence && !possible_ends.empty()) {
                // max(possible_ends, key=dur): the first of equal maxima.
                size_t best = 0;
                for (size_t k = 1; k < possible_ends.size(); ++k) {
                    if (possible_ends[k].dur > possible_ends[best].dur) {
                        best = k;
                    }
                }
                prev_end          = possible_ends[best].end;
                const int64_t dur = possible_ends[best].dur;
                cur.end_sample    = prev_end;
                out.push_back(cur);
                drop_current();
                next_start = prev_end + dur;

                if (next_start < prev_end + cur_sample) {
                    cur.start_sample = next_start;
                    cur_has          = true;
                } else {
                    triggered = false;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else if (prev_end != 0) {
                cur.end_sample = prev_end;
                out.push_back(cur);
                drop_current();
                if (next_start < prev_end) {
                    triggered = false;
                } else {
                    cur.start_sample = next_start;
                    cur_has          = true;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else {
                cur.end_sample = cur_sample;
                out.push_back(cur);
                drop_current();
                prev_end = next_start = temp_end = 0;
                triggered                        = false;
                possible_ends.clear();
                continue;
            }
        }

        // Silence while in speech.
        if (prob < p.neg_threshold && triggered) {
            if (temp_end == 0) {
                temp_end = cur_sample;
            }
            const int64_t sil_dur_now = cur_sample - temp_end;

            if (!p.use_max_possible_silence && static_cast<double>(sil_dur_now) > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }

            if (static_cast<double>(sil_dur_now) < min_silence_samples) {
                continue;
            }
            cur.end_sample = temp_end;
            if (static_cast<double>(cur.end_sample - cur.start_sample) > min_speech_samples) {
                out.push_back(cur);
            }
            drop_current();
            prev_end = next_start = temp_end = 0;
            triggered                        = false;
            possible_ends.clear();
            continue;
        }
    }

    if (cur_has && static_cast<double>(n_samples - cur.start_sample) > min_speech_samples) {
        cur.end_sample = n_samples;
        out.push_back(cur);
    }

    // Padding. speech_pad_samples is integral (16 * ms), so the int() casts
    // of the reference are exact.
    const int64_t pad = static_cast<int64_t>(speech_pad_samples);
    for (size_t i = 0; i < out.size(); ++i) {
        VadSegmentEntry & s = out[i];
        if (i == 0) {
            s.start_sample = std::max<int64_t>(0, s.start_sample - pad);
        }
        if (i + 1 != out.size()) {
            const int64_t silence = out[i + 1].start_sample - s.end_sample;
            if (silence < 2 * pad) {
                s.end_sample += floor_div(silence, 2);
                out[i + 1].start_sample = std::max<int64_t>(0, out[i + 1].start_sample - floor_div(silence, 2));
            } else {
                s.end_sample            = std::min<int64_t>(n_samples, s.end_sample + pad);
                out[i + 1].start_sample = std::max<int64_t>(0, out[i + 1].start_sample - pad);
            }
        } else {
            s.end_sample = std::min<int64_t>(n_samples, s.end_sample + pad);
        }
    }
}

extern "C" void transcribe_vad_info_init(struct transcribe_vad_info * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_vad_session_params_init(struct transcribe_vad_session_params * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_vad_params_init(struct transcribe_vad_params * p) {
    transcribe::init_sized(p);
    if (p != nullptr) {
        p->threshold      = 0.5;
        p->neg_threshold  = -1.0;
        p->min_speech_ms  = 250;
        p->min_silence_ms = 100;
        p->speech_pad_ms  = 30;
        p->max_speech_ms  = 0;
    }
}

extern "C" void transcribe_vad_result_init(struct transcribe_vad_result * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_vad_segment_init(struct transcribe_vad_segment * p) {
    transcribe::init_sized(p);
}

static transcribe_status vad_get_info_impl(const transcribe_model * model, transcribe_vad_info * out) {
    if (model == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_info_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::VadOps * ops = vad_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_vad_info staged{};
    staged.struct_size   = out->struct_size;
    staged.sample_rate   = 16000;
    staged.frame_samples = ops->frame_samples(model);
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

static transcribe_status vad_session_init_impl(transcribe_model *                    model,
                                               const transcribe_vad_session_params * params,
                                               transcribe_vad_session **             out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out = nullptr;
    if (model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::VadOps * ops = vad_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_vad_session_params defaults;
    transcribe_vad_session_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_session_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    if (params->n_threads < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    std::unique_ptr<transcribe_vad_session> s(ops->new_session());
    s->model     = model;
    s->n_threads = params->n_threads;
    ops->reset(s.get());
    *out = s.release();
    return TRANSCRIBE_OK;
}

static transcribe_status vad_run_impl(transcribe_vad_session *      session,
                                      const float *                 pcm,
                                      int                           n_samples,
                                      const transcribe_vad_params * params) {
    // Everything up to the commit point leaves the previous result intact.
    if (session == nullptr || pcm == nullptr || n_samples <= 0 || !transcribe::pcm_is_finite(pcm, n_samples)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    VadSegmentParams sp;
    if (const auto st = transcribe::resolve_vad_params(params, sp); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::VadOps * ops   = vad_ops(session->model);
    const int64_t              frame = ops->frame_samples(session->model);

    VadFailureGuard failure{ session, ops };
    clear_result(session);
    reset_stream(session, ops);
    transcribe::ScratchReleaseGuard release{ session, true };

    const int64_t n       = n_samples;
    const int64_t n_full  = n / frame;
    const int64_t n_tail  = n - n_full * frame;
    const int64_t n_total = n_full + (n_tail > 0 ? 1 : 0);
    session->probs.resize(static_cast<size_t>(n_total));

    transcribe_status st = TRANSCRIBE_OK;
    if (n_full > 0) {
        st = ops->score(session, pcm, n_full, session->probs.data());
    }
    if (st == TRANSCRIBE_OK && n_tail > 0) {
        // get_speech_timestamps zero pads the last frame.
        std::vector<float> last(static_cast<size_t>(frame), 0.0f);
        std::memcpy(last.data(), pcm + n_full * frame, static_cast<size_t>(n_tail) * sizeof(float));
        st = ops->score(session, last.data(), 1, session->probs.data() + n_full);
    }
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    reset_stream(session, ops);
    transcribe::vad_probs_to_segments(session->probs.data(), n_total, n, static_cast<int32_t>(frame), sp,
                                      session->segments);
    failure.complete = true;
    return TRANSCRIBE_OK;
}

static transcribe_status vad_stream_feed_impl(transcribe_vad_session * session, const float * pcm, int n_samples) {
    if (session == nullptr || n_samples < 0 || (n_samples > 0 && pcm == nullptr) ||
        (n_samples > 0 && !transcribe::pcm_is_finite(pcm, n_samples))) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::VadOps * ops   = vad_ops(session->model);
    const size_t               frame = static_cast<size_t>(ops->frame_samples(session->model));

    VadFailureGuard failure{ session, ops };
    clear_result(session);
    if (session->needs_reset) {
        reset_stream(session, ops);
    }
    session->first_frame = session->frames_done;
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
    if (n_frames > 0) {
        session->probs.resize(n_frames);
        const transcribe_status st = ops->score(session, src, static_cast<int64_t>(n_frames), session->probs.data());
        if (st != TRANSCRIBE_OK) {
            return st;
        }
    }
    std::vector<float> rest(src + n_frames * frame, src + avail);
    session->pending.swap(rest);
    session->frames_done += static_cast<int64_t>(n_frames);
    failure.complete = true;
    return TRANSCRIBE_OK;
}

static transcribe_status vad_stream_flush_impl(transcribe_vad_session * session) {
    if (session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::VadOps * ops   = vad_ops(session->model);
    const size_t               frame = static_cast<size_t>(ops->frame_samples(session->model));

    VadFailureGuard failure{ session, ops };
    clear_result(session);
    if (session->needs_reset) {
        reset_stream(session, ops);
    }
    session->first_frame = session->frames_done;
    transcribe_status st = TRANSCRIBE_OK;
    if (!session->pending.empty()) {
        std::vector<float> last(frame, 0.0f);
        std::copy(session->pending.begin(), session->pending.end(), last.begin());
        session->probs.resize(1);
        st = ops->score(session, last.data(), 1, session->probs.data());
    }
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    reset_stream(session, ops);
    failure.complete = true;
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_vad_set_abort_callback(struct transcribe_vad_session * session,
                                                  transcribe_abort_callback       cb,
                                                  void *                          user_data) {
    if (session != nullptr) {
        session->abort_cb       = cb;
        session->abort_userdata = user_data;
    }
}

extern "C" transcribe_status transcribe_vad_get_result(const struct transcribe_vad_session * session,
                                                       struct transcribe_vad_result *        out) {
    if (session == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_result_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe_vad_result staged{};
    staged.struct_size = out->struct_size;
    staged.n_probs     = static_cast<int32_t>(session->probs.size());
    staged.first_frame = session->first_frame;
    staged.n_segments  = static_cast<int32_t>(session->segments.size());
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" const float * transcribe_vad_probs(const struct transcribe_vad_session * session) {
    return session != nullptr && !session->probs.empty() ? session->probs.data() : nullptr;
}

extern "C" transcribe_status transcribe_vad_get_segment(const struct transcribe_vad_session * session,
                                                        int                                   i,
                                                        struct transcribe_vad_segment *       out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_segment_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe_vad_segment staged{};
    staged.struct_size = out->struct_size;
    if (session != nullptr && i >= 0 && static_cast<size_t>(i) < session->segments.size()) {
        staged.start_sample = session->segments[static_cast<size_t>(i)].start_sample;
        staged.end_sample   = session->segments[static_cast<size_t>(i)].end_sample;
    }
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_vad_get_timings(const struct transcribe_vad_session * session,
                                                        struct transcribe_timings *           out) {
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

extern "C" transcribe_status transcribe_vad_get_info(const struct transcribe_model * model,
                                                     struct transcribe_vad_info *    out) {
    return api_guard_status("transcribe_vad_get_info", [&] { return vad_get_info_impl(model, out); });
}

extern "C" transcribe_status transcribe_vad_session_init(struct transcribe_model *                    model,
                                                         const struct transcribe_vad_session_params * params,
                                                         struct transcribe_vad_session **             out) {
    return api_guard_status("transcribe_vad_session_init", [&] { return vad_session_init_impl(model, params, out); });
}

extern "C" void transcribe_vad_session_free(struct transcribe_vad_session * session) {
    api_guard_void("transcribe_vad_session_free", [&] { delete session; });
}

extern "C" transcribe_status transcribe_vad_run(struct transcribe_vad_session *      session,
                                                const float *                        pcm,
                                                int                                  n_samples,
                                                const struct transcribe_vad_params * params) {
    return api_guard_status("transcribe_vad_run", [&] { return vad_run_impl(session, pcm, n_samples, params); });
}

extern "C" transcribe_status transcribe_vad_stream_feed(struct transcribe_vad_session * session,
                                                        const float *                   pcm,
                                                        int                             n_samples) {
    return api_guard_status("transcribe_vad_stream_feed",
                            [&] { return vad_stream_feed_impl(session, pcm, n_samples); });
}

extern "C" transcribe_status transcribe_vad_stream_flush(struct transcribe_vad_session * session) {
    return api_guard_status("transcribe_vad_stream_flush", [&] { return vad_stream_flush_impl(session); });
}

extern "C" void transcribe_vad_stream_reset(struct transcribe_vad_session * session) {
    api_guard_void("transcribe_vad_stream_reset", [&] {
        if (session != nullptr) {
            reset_stream(session, vad_ops(session->model));
        }
    });
}
