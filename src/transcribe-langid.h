// transcribe-langid.h - internal LANGID role surface: the session base, the
// per-arch ops table, and the shared label table.
//
// Families compute logits over their label table; the role dispatcher
// (transcribe-langid.cpp) crops the input, applies the allowed set, and
// owns the softmax and ranking.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/langid.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace transcribe {

struct LangidCandidateEntry {
    int32_t index = 0;
    float   p     = 0.0f;
    float   logit = 0.0f;
};

// A model's labels. Built once at load by build_langid_labels, immutable after.
struct LangidLabels {
    std::vector<std::string>                codes;
    std::vector<std::string>                names;
    std::map<std::string, int, std::less<>> index;  // every code and alias -> label index
};

// Build a label table from parallel code / name lists and "alias=code"
// specs. Returns TRANSCRIBE_ERR_GGUF (and logs `tag`) when the lists are
// empty or differ in length, a code is empty or repeated, or an alias is
// malformed, repeated, collides with a code, or names an unknown code.
transcribe_status build_langid_labels(std::vector<std::string>         codes,
                                      std::vector<std::string>         names,
                                      const std::vector<std::string> & alias_specs,
                                      const char *                     tag,
                                      LangidLabels &                   out);

// Scored-audio bounds every LANGID model shares. Longer input is scored on
// its first k_langid_max_audio_ms.
constexpr int32_t k_langid_min_audio_ms = 500;
constexpr int32_t k_langid_max_audio_ms = 30000;

struct LangidOps {
    const LangidLabels & (*labels)(const transcribe_model * model);
    transcribe_langid_session * (*new_session)();
    // Score already validated, already cropped PCM; fill one logit per label.
    // Poll session->poll_abort() and return TRANSCRIBE_ERR_ABORTED when it
    // fires. Set session->t_mel_us / t_encode_us.
    transcribe_status (*run)(transcribe_langid_session * session,
                             const float *               pcm,
                             int                         n_samples,
                             std::vector<float> &        logits);
};

}  // namespace transcribe

struct transcribe_langid_session : transcribe::SessionCore {
    // Last successful run's result; empty / zero otherwise.
    std::vector<transcribe::LangidCandidateEntry> candidates;
    float                                         allowed_mass = 0.0f;
};
