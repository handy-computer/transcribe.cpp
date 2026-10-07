// transcribe-diarize.cpp - DIARIZE role C ABI (include/transcribe/diarize.h)
// and the shared probs -> speaker segments step.

#include "transcribe-diarize.h"

#include "transcribe-abi.h"
#include "transcribe-api-guard.h"
#include "transcribe-arch.h"
#include "transcribe-model.h"

#include <cmath>
#include <cstddef>

using transcribe::api_guard_status;
using transcribe::api_guard_void;
using transcribe::check_input_struct_size;
using transcribe::check_struct_size;
using transcribe::copy_out_prefix;

namespace {

constexpr size_t k_min_info_size           = TRANSCRIBE_FIELD_END(transcribe_diarize_info, max_speakers);
constexpr size_t k_min_session_params_size = TRANSCRIBE_FIELD_END(transcribe_diarize_session_params, n_threads);
constexpr size_t k_min_params_size         = TRANSCRIBE_FIELD_END(transcribe_diarize_params, family);
constexpr size_t k_min_timings_size        = TRANSCRIBE_FIELD_END(transcribe_timings, decode_ms);

const transcribe::DiarizeOps * diarize_ops(const transcribe_model * model) {
    return (model->roles & TRANSCRIBE_ROLE_DIARIZE) != 0 ? model->arch->diarize : nullptr;
}

}  // namespace

void transcribe::probs_to_segments(const float *                      probs,
                                   int                                n_frames,
                                   int                                n_speakers,
                                   double                             frame_ms,
                                   std::vector<SpeakerSegmentEntry> & out) {
    for (int s = 0; s < n_speakers; ++s) {
        int run_start = -1;
        for (int t = 0; t <= n_frames; ++t) {
            const bool active = t < n_frames && probs[static_cast<size_t>(t) * n_speakers + s] > 0.5f;
            if (active && run_start < 0) {
                run_start = t;
            } else if (!active && run_start >= 0) {
                SpeakerSegmentEntry row;
                row.t0_ms      = static_cast<int64_t>(std::llround(run_start * frame_ms));
                row.t1_ms      = static_cast<int64_t>(std::llround(t * frame_ms));
                row.speaker_id = s + 1;
                out.push_back(row);
                run_start = -1;
            }
        }
    }
}

extern "C" void transcribe_diarize_info_init(struct transcribe_diarize_info * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_diarize_session_params_init(struct transcribe_diarize_session_params * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_diarize_params_init(struct transcribe_diarize_params * p) {
    transcribe::init_sized(p);
}

static transcribe_status diarize_get_info_impl(const transcribe_model * model, transcribe_diarize_info * out) {
    if (model == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_info_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::DiarizeOps * ops = diarize_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_diarize_info staged{};
    staged.struct_size  = out->struct_size;
    staged.sample_rate  = 16000;
    staged.max_speakers = ops->max_speakers(model);
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

static transcribe_status diarize_session_init_impl(transcribe_model *                        model,
                                                   const transcribe_diarize_session_params * params,
                                                   transcribe_diarize_session **             out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out = nullptr;
    if (model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::DiarizeOps * ops = diarize_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_diarize_session_params defaults;
    transcribe_diarize_session_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_session_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    if (params->n_threads < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out              = ops->new_session();
    (*out)->model     = model;
    (*out)->n_threads = params->n_threads;
    return TRANSCRIBE_OK;
}

static transcribe_status diarize_run_impl(transcribe_diarize_session *      session,
                                          const float *                     pcm,
                                          int                               n_samples,
                                          const transcribe_diarize_params * params) {
    // Everything up to the commit point leaves the previous result intact.
    if (session == nullptr || pcm == nullptr || n_samples <= 0 || !transcribe::pcm_is_finite(pcm, n_samples)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe_diarize_params defaults;
    transcribe_diarize_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::DiarizeOps * ops = diarize_ops(session->model);
    if (params->family != nullptr) {
        if (params->family->size < sizeof(struct transcribe_ext)) {
            return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
        }
        if (!transcribe_model_accepts_ext_kind(session->model, TRANSCRIBE_EXT_SLOT_DIARIZE_RUN, params->family->kind)) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        if (ops->run_validate != nullptr) {
            if (const auto st = ops->run_validate(params); st != TRANSCRIBE_OK) {
                return st;
            }
        }
    }

    session->segments.clear();
    session->t_mel_us    = 0;
    session->t_encode_us = 0;
    session->t_decode_us = 0;
    transcribe::ScratchReleaseGuard release{ session, true };

    transcribe::DiarizeProbs out;
    if (const auto st = ops->run(session, pcm, n_samples, params, out); st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<transcribe::SpeakerSegmentEntry> segments;
    transcribe::probs_to_segments(out.probs.data(), out.n_frames, out.n_speakers, out.frame_ms, segments);
    session->segments.swap(segments);
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_diarize_set_abort_callback(struct transcribe_diarize_session * session,
                                                      transcribe_abort_callback           cb,
                                                      void *                              user_data) {
    if (session != nullptr) {
        session->abort_cb       = cb;
        session->abort_userdata = user_data;
    }
}

extern "C" int transcribe_diarize_n_segments(const struct transcribe_diarize_session * session) {
    return session != nullptr ? static_cast<int>(session->segments.size()) : 0;
}

extern "C" transcribe_status transcribe_diarize_get_segment(const struct transcribe_diarize_session * session,
                                                            int                                       i,
                                                            struct transcribe_speaker_segment *       out) {
    const bool in_range = i >= 0 && i < transcribe_diarize_n_segments(session);
    return transcribe::copy_out_speaker_segment(in_range ? &session->segments[static_cast<size_t>(i)] : nullptr, out);
}

extern "C" transcribe_status transcribe_diarize_get_timings(const struct transcribe_diarize_session * session,
                                                            struct transcribe_timings *               out) {
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

extern "C" transcribe_status transcribe_diarize_get_info(const struct transcribe_model *  model,
                                                         struct transcribe_diarize_info * out) {
    return api_guard_status("transcribe_diarize_get_info", [&] { return diarize_get_info_impl(model, out); });
}

extern "C" transcribe_status transcribe_diarize_session_init(struct transcribe_model *                        model,
                                                             const struct transcribe_diarize_session_params * params,
                                                             struct transcribe_diarize_session **             out) {
    return api_guard_status("transcribe_diarize_session_init",
                            [&] { return diarize_session_init_impl(model, params, out); });
}

extern "C" void transcribe_diarize_session_free(struct transcribe_diarize_session * session) {
    api_guard_void("transcribe_diarize_session_free", [&] { delete session; });
}

extern "C" transcribe_status transcribe_diarize_run(struct transcribe_diarize_session *      session,
                                                    const float *                            pcm,
                                                    int                                      n_samples,
                                                    const struct transcribe_diarize_params * params) {
    return api_guard_status("transcribe_diarize_run",
                            [&] { return diarize_run_impl(session, pcm, n_samples, params); });
}
