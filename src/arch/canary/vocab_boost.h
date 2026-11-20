// arch/canary/vocab_boost.h - vocabulary biasing for canary's greedy decode.
//
// INTERNAL. Canary has no prompt slot for custom terms: the decoder-context
// slot makes the output loop and over-insert (docs/models/canary-1b-v2.md).
// So transcribe_run_params::vocabulary biases the decode instead, by shallow
// fusion on the raw logits: a term's first token gets +strength at any step,
// and each following token gets +strength while the transcript so far ends
// with the term's previous tokens. Measured on canary-1b-v2 over 312 phone
// turns answering "which area?" (8 kHz, es/de): strength 3 read 21 more area
// names, about 5 of them inserted into unrelated speech, against 3 lost to
// answers that named two areas; no added empty output, no added latency.
// Strength 5 started to loop and inserted more.

#pragma once

#include "transcribe-env.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace transcribe::canary {

inline constexpr float k_vocab_boost_default = 3.0f;

// TRANSCRIBE_CANARY_VOCAB_BOOST, or the default when unset or not a number.
// A value <= 0 turns the bias off.
inline float vocab_boost_strength() {
    const char * raw = transcribe::env::str("TRANSCRIBE_CANARY_VOCAB_BOOST");
    if (raw == nullptr) {
        return k_vocab_boost_default;
    }
    char *      end   = nullptr;
    const float value = std::strtof(raw, &end);
    return end == raw ? k_vocab_boost_default : value;
}

struct VocabBoost {
    std::vector<std::vector<int32_t>> terms;  // token ids, one entry per spelling
    float                             strength = 0.0f;

    bool active() const { return strength > 0.0f && !terms.empty(); }

    // The next token: the argmax of `logits` after the bias. `generated` is
    // the transcript so far (prompt excluded).
    int pick(const float * logits, int64_t vocab, const std::vector<int> & generated) const {
        int   best_id = 0;
        float best    = logits[0];
        for (int64_t j = 1; j < vocab; ++j) {
            if (logits[j] > best) {
                best    = logits[j];
                best_id = static_cast<int>(j);
            }
        }
        if (!active()) {
            return best_id;
        }
        for (const std::vector<int32_t> & term : terms) {
            for (size_t k = 0; k < term.size() && k <= generated.size(); ++k) {
                if (!ends_with(generated, term, k) || term[k] < 0 || term[k] >= vocab) {
                    continue;
                }
                const float score = logits[term[k]] + strength;
                if (score > best) {
                    best    = score;
                    best_id = term[k];
                }
            }
        }
        return best_id;
    }

  private:
    // Whether `generated` ends with the first `k` tokens of `term`.
    static bool ends_with(const std::vector<int> & generated, const std::vector<int32_t> & term, size_t k) {
        const size_t offset = generated.size() - k;
        for (size_t i = 0; i < k; ++i) {
            if (generated[offset + i] != term[i]) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace transcribe::canary
