// transcribe-langid.cpp - LANGID role C ABI (include/transcribe/langid.h):
// label table, allowed set, crop, softmax and ranking.

#include "transcribe-langid.h"

#include "transcribe-abi.h"
#include "transcribe-api-guard.h"
#include "transcribe-arch.h"
#include "transcribe-log.h"
#include "transcribe-model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <utility>

using transcribe::api_guard_status;
using transcribe::api_guard_void;
using transcribe::check_input_struct_size;
using transcribe::check_struct_size;
using transcribe::copy_out_prefix;
using transcribe::LangidCandidateEntry;
using transcribe::LangidLabels;

namespace {

constexpr size_t k_min_info_size           = TRANSCRIBE_FIELD_END(transcribe_langid_info, max_audio_ms);
constexpr size_t k_min_session_params_size = TRANSCRIBE_FIELD_END(transcribe_langid_session_params, n_threads);
constexpr size_t k_min_params_size         = TRANSCRIBE_FIELD_END(transcribe_langid_params, n_allowed);
constexpr size_t k_min_result_size         = TRANSCRIBE_FIELD_END(transcribe_langid_result, allowed_mass);
constexpr size_t k_min_candidate_size      = TRANSCRIBE_FIELD_END(transcribe_langid_candidate, logit);
constexpr size_t k_min_timings_size        = TRANSCRIBE_FIELD_END(transcribe_timings, decode_ms);

constexpr int64_t k_samples_per_ms = 16;  // 16 kHz

const transcribe::LangidOps * langid_ops(const transcribe_model * model) {
    return model != nullptr && (model->roles & TRANSCRIBE_ROLE_LANGID) != 0 ? model->arch->langid : nullptr;
}

const LangidLabels * langid_labels(const transcribe_model * model) {
    const transcribe::LangidOps * ops = langid_ops(model);
    return ops != nullptr ? &ops->labels(model) : nullptr;
}

int32_t label_index(const LangidLabels & labels, const char * code) {
    if (code == nullptr) {
        return -1;
    }
    const auto it = labels.index.find(std::string_view(code));
    return it != labels.index.end() ? it->second : -1;
}

// Resolve params->allowed into a per-label mask. Only NULL with
// n_allowed == 0 means "all".
transcribe_status build_allowed_mask(const LangidLabels &             labels,
                                     const transcribe_langid_params * params,
                                     std::vector<uint8_t> &           mask,
                                     int32_t &                        n_allowed) {
    const size_t n_labels = labels.codes.size();
    if (params->allowed == nullptr) {
        if (params->n_allowed != 0) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        mask.assign(n_labels, 1);
        n_allowed = static_cast<int32_t>(n_labels);
        return TRANSCRIBE_OK;
    }
    if (params->n_allowed <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    mask.assign(n_labels, 0);
    for (int32_t i = 0; i < params->n_allowed; ++i) {
        if (params->allowed[i] == nullptr) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    n_allowed = 0;
    for (int32_t i = 0; i < params->n_allowed; ++i) {
        const int32_t idx = label_index(labels, params->allowed[i]);
        if (idx < 0) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "transcribe_langid_run: unknown language '%s'",
                                params->allowed[i]);
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
        if (mask[static_cast<size_t>(idx)] == 0) {
            mask[static_cast<size_t>(idx)] = 1;
            ++n_allowed;
        }
    }
    return TRANSCRIBE_OK;
}

// Softmax over the entries with mask[i] != 0 (every entry when mask is
// NULL); masked-out entries get 0. Sums in double.
void masked_softmax(const std::vector<float> & logits, const uint8_t * mask, std::vector<float> & out) {
    const size_t n  = logits.size();
    float        mx = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        if (mask == nullptr || mask[i] != 0) {
            mx = std::max(mx, logits[i]);
        }
    }
    out.assign(n, 0.0f);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (mask == nullptr || mask[i] != 0) {
            const double e = std::exp(static_cast<double>(logits[i]) - static_cast<double>(mx));
            out[i]         = static_cast<float>(e);
            sum += e;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(static_cast<double>(out[i]) / sum);
    }
}

}  // namespace

transcribe_status transcribe::build_langid_labels(std::vector<std::string>         codes,
                                                  std::vector<std::string>         names,
                                                  const std::vector<std::string> & alias_specs,
                                                  const char *                     tag,
                                                  LangidLabels &                   out) {
    auto fail = [tag](const char * what, const std::string & item) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: label table: %s '%s'", tag, what, item.c_str());
        return TRANSCRIBE_ERR_GGUF;
    };
    if (codes.empty() || codes.size() != names.size()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: label table: %zu codes vs %zu names", tag, codes.size(), names.size());
        return TRANSCRIBE_ERR_GGUF;
    }
    LangidLabels staged;
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i].empty()) {
            return fail("empty code at index", std::to_string(i));
        }
        if (!staged.index.emplace(codes[i], static_cast<int>(i)).second) {
            return fail("duplicate code", codes[i]);
        }
    }
    for (const std::string & spec : alias_specs) {
        const size_t eq = spec.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == spec.size()) {
            return fail("malformed alias (want alias=code)", spec);
        }
        std::string alias  = spec.substr(0, eq);
        std::string target = spec.substr(eq + 1);
        const auto  it     = staged.index.find(target);
        if (it == staged.index.end() || codes[static_cast<size_t>(it->second)] != target) {
            return fail("alias names an unknown code", spec);
        }
        if (!staged.index.emplace(std::move(alias), it->second).second) {
            return fail("alias collides with a code or another alias", spec);
        }
    }
    staged.codes = std::move(codes);
    staged.names = std::move(names);
    out          = std::move(staged);
    return TRANSCRIBE_OK;
}

extern "C" void transcribe_langid_info_init(struct transcribe_langid_info * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_langid_session_params_init(struct transcribe_langid_session_params * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_langid_params_init(struct transcribe_langid_params * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_langid_result_init(struct transcribe_langid_result * p) {
    transcribe::init_sized(p);
}

extern "C" void transcribe_langid_candidate_init(struct transcribe_langid_candidate * p) {
    transcribe::init_sized(p);
}

static transcribe_status langid_get_info_impl(const transcribe_model * model, transcribe_langid_info * out) {
    if (model == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_info_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const LangidLabels * labels = langid_labels(model);
    if (labels == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_langid_info staged{};
    staged.struct_size  = out->struct_size;
    staged.sample_rate  = 16000;
    staged.n_labels     = static_cast<int32_t>(labels->codes.size());
    staged.min_audio_ms = transcribe::k_langid_min_audio_ms;
    staged.max_audio_ms = transcribe::k_langid_max_audio_ms;
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

static transcribe_status langid_session_init_impl(transcribe_model *                       model,
                                                  const transcribe_langid_session_params * params,
                                                  transcribe_langid_session **             out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out = nullptr;
    if (model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const transcribe::LangidOps * ops = langid_ops(model);
    if (ops == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ROLE;
    }
    transcribe_langid_session_params defaults;
    transcribe_langid_session_params_init(&defaults);
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

static transcribe_status langid_run_impl(transcribe_langid_session *      session,
                                         const float *                    pcm,
                                         int                              n_samples,
                                         const transcribe_langid_params * params) {
    // Everything up to the commit point leaves the previous result intact.
    if (session == nullptr || pcm == nullptr || n_samples <= 0 || !transcribe::pcm_is_finite(pcm, n_samples)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe_langid_params defaults;
    transcribe_langid_params_init(&defaults);
    if (params == nullptr) {
        params = &defaults;
    }
    if (const auto st = check_input_struct_size(params->struct_size, k_min_params_size); st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe::LangidOps * ops    = langid_ops(session->model);
    const LangidLabels &          labels = ops->labels(session->model);
    std::vector<uint8_t>          mask;
    int32_t                       n_allowed = 0;
    if (const auto st = build_allowed_mask(labels, params, mask, n_allowed); st != TRANSCRIBE_OK) {
        return st;
    }
    // Score the first k_langid_max_audio_ms; the minimum applies to what is
    // scored.
    const int64_t max_samples = static_cast<int64_t>(transcribe::k_langid_max_audio_ms) * k_samples_per_ms;
    const int64_t n_used      = std::min<int64_t>(n_samples, max_samples);
    if (n_used < static_cast<int64_t>(transcribe::k_langid_min_audio_ms) * k_samples_per_ms) {
        return TRANSCRIBE_ERR_INPUT_TOO_SHORT;
    }
    const float * pcm_used = pcm;

    session->candidates.clear();
    session->allowed_mass = 0.0f;
    session->t_mel_us     = 0;
    session->t_encode_us  = 0;
    session->t_decode_us  = 0;
    transcribe::ScratchReleaseGuard release{ session, true };

    std::vector<float> logits;
    if (const auto st = ops->run(session, pcm_used, static_cast<int>(n_used), logits); st != TRANSCRIBE_OK) {
        return st;
    }
    if (logits.size() != labels.codes.size()) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "transcribe_langid_run: %zu logits for %zu labels",
                            logits.size(), labels.codes.size());
        return TRANSCRIBE_ERR_BACKEND;
    }
    for (const float v : logits) {
        if (!std::isfinite(v)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "transcribe_langid_run: non-finite logit");
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    const bool         restricted = n_allowed != static_cast<int32_t>(labels.codes.size());
    std::vector<float> p_open;
    std::vector<float> p_allowed;
    masked_softmax(logits, nullptr, p_open);
    masked_softmax(logits, mask.data(), p_allowed);
    double mass = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (mask[i] != 0) {
            mass += p_open[i];
        }
    }

    std::vector<LangidCandidateEntry> ranked;
    ranked.reserve(static_cast<size_t>(n_allowed));
    for (size_t i = 0; i < logits.size(); ++i) {
        if (mask[i] != 0) {
            ranked.push_back({ static_cast<int32_t>(i), p_allowed[i], logits[i] });
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const LangidCandidateEntry & a, const LangidCandidateEntry & b) { return a.p > b.p; });

    session->candidates.swap(ranked);
    session->allowed_mass = restricted ? static_cast<float>(mass) : 1.0f;
    return TRANSCRIBE_OK;
}

extern "C" const char * transcribe_langid_label_code(const struct transcribe_model * model, int32_t i) {
    const LangidLabels * labels = langid_labels(model);
    if (labels == nullptr || i < 0 || static_cast<size_t>(i) >= labels->codes.size()) {
        return nullptr;
    }
    return labels->codes[static_cast<size_t>(i)].c_str();
}

extern "C" const char * transcribe_langid_label_name(const struct transcribe_model * model, int32_t i) {
    const LangidLabels * labels = langid_labels(model);
    if (labels == nullptr || i < 0 || static_cast<size_t>(i) >= labels->names.size()) {
        return nullptr;
    }
    return labels->names[static_cast<size_t>(i)].c_str();
}

extern "C" void transcribe_langid_set_abort_callback(struct transcribe_langid_session * session,
                                                     transcribe_abort_callback          cb,
                                                     void *                             user_data) {
    if (session != nullptr) {
        session->abort_cb       = cb;
        session->abort_userdata = user_data;
    }
}

extern "C" transcribe_status transcribe_langid_get_result(const struct transcribe_langid_session * session,
                                                          struct transcribe_langid_result *        out) {
    if (session == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_result_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe_langid_result staged{};
    staged.struct_size  = out->struct_size;
    staged.n_candidates = static_cast<int32_t>(session->candidates.size());
    staged.allowed_mass = session->allowed_mass;
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_langid_get_candidate(const struct transcribe_langid_session * session,
                                                             int                                      i,
                                                             struct transcribe_langid_candidate *     out) {
    if (out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = check_struct_size(out->struct_size, k_min_candidate_size); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe_langid_candidate staged{};
    staged.struct_size = out->struct_size;
    if (session != nullptr && i >= 0 && static_cast<size_t>(i) < session->candidates.size()) {
        const LangidCandidateEntry & c = session->candidates[static_cast<size_t>(i)];
        staged.index                   = c.index;
        staged.code                    = transcribe_langid_label_code(session->model, c.index);
        staged.name                    = transcribe_langid_label_name(session->model, c.index);
        staged.p                       = c.p;
        staged.logit                   = c.logit;
    }
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
    return TRANSCRIBE_OK;
}

extern "C" transcribe_status transcribe_langid_get_timings(const struct transcribe_langid_session * session,
                                                           struct transcribe_timings *              out) {
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

// C ABI forwarders for the entry points that allocate, compute, transfer
// ownership, or build a lookup key; the ones above are nothrow by
// construction.

extern "C" transcribe_status transcribe_langid_get_info(const struct transcribe_model * model,
                                                        struct transcribe_langid_info * out) {
    return api_guard_status("transcribe_langid_get_info", [&] { return langid_get_info_impl(model, out); });
}

extern "C" int32_t transcribe_langid_label_index(const struct transcribe_model * model, const char * code_or_alias) {
    return transcribe::api_guard_value("transcribe_langid_label_index", int32_t{ -1 }, [&] {
        const LangidLabels * labels = langid_labels(model);
        return labels != nullptr ? label_index(*labels, code_or_alias) : int32_t{ -1 };
    });
}

extern "C" transcribe_status transcribe_langid_session_init(struct transcribe_model *                       model,
                                                            const struct transcribe_langid_session_params * params,
                                                            struct transcribe_langid_session **             out) {
    return api_guard_status("transcribe_langid_session_init",
                            [&] { return langid_session_init_impl(model, params, out); });
}

extern "C" void transcribe_langid_session_free(struct transcribe_langid_session * session) {
    api_guard_void("transcribe_langid_session_free", [&] { delete session; });
}

extern "C" transcribe_status transcribe_langid_run(struct transcribe_langid_session *      session,
                                                   const float *                           pcm,
                                                   int                                     n_samples,
                                                   const struct transcribe_langid_params * params) {
    return api_guard_status("transcribe_langid_run", [&] { return langid_run_impl(session, pcm, n_samples, params); });
}
