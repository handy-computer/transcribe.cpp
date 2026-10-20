// Phrase boosting automaton (pure host, no model). The scores are the whole
// contract: a wrong one either invents terms or never finds them, and the
// decoder has no other check.

#include "transcribe-phrase-boost.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void expect(const char * what, float got, float expected) {
    if (std::fabs(got - expected) > 1e-6f) {
        std::fprintf(stderr, "FAIL %s: got %.3f, expected %.3f\n", what, got, expected);
        ++g_failures;
    }
}

// Emits `tokens` from the root and returns the summed score.
float walk(const transcribe::PhraseBoost & b, const std::vector<int32_t> & tokens, int * end_state = nullptr) {
    int   state = transcribe::PhraseBoost::root;
    float total = 0.0f;
    for (const int32_t t : tokens) {
        total += b.delta(state, t);
        state = b.next(state, t);
    }
    if (end_state != nullptr) {
        *end_state = state;
    }
    return total;
}

}  // namespace

int main(void) {
    transcribe::PhraseBoost b;
    b.build({ { 1, 2, 3 }, { 2, 4 }, {}, { 1, 2, 3 } }, 1.0f);

    // A term's first piece earns nothing; each later matched piece earns the
    // score; an unrelated token earns nothing.
    expect("start a term", b.delta(transcribe::PhraseBoost::root, 1), 0.0f);
    expect("continue a term", b.delta(b.next(transcribe::PhraseBoost::root, 1), 2), 1.0f);
    expect("unrelated from root", b.delta(transcribe::PhraseBoost::root, 9), 0.0f);
    expect("complete term", walk(b, { 1, 2, 3 }), 2.0f);

    // A completed term keeps its bonus whatever follows.
    expect("completed then unrelated", walk(b, { 1, 2, 3, 9, 9 }), 2.0f);

    // A broken match gives back exactly what it had earned.
    expect("broken match", walk(b, { 1, 2, 9 }), 0.0f);
    expect("broken at once", walk(b, { 1, 9 }), 0.0f);

    // Falling back keeps the suffix that still matches another term:
    // 1 2 then 4 completes {2, 4} through the failure link.
    int state = 0;
    expect("suffix completes another term", walk(b, { 1, 2, 4 }, &state), 1.0f);
    expect("and keeps it", walk(b, { 1, 2, 4, 9 }), 1.0f);

    // apply() adds exactly delta() to each non-blank token and leaves blank.
    int s12 = 0;
    walk(b, { 1, 2 }, &s12);
    std::vector<float> logits(6, 0.0f);
    b.apply(s12, logits.data(), static_cast<int>(logits.size()), /*blank=*/5);
    for (int t = 0; t < 5; ++t) {
        expect("apply matches delta", logits[static_cast<size_t>(t)], b.delta(s12, t));
    }
    expect("blank untouched", logits[5], 0.0f);

    // No phrases: empty, and every score is zero.
    transcribe::PhraseBoost none;
    none.build({}, 2.0f);
    if (!none.empty()) {
        std::fprintf(stderr, "FAIL empty automaton reports terms\n");
        ++g_failures;
    }
    expect("empty automaton", none.delta(transcribe::PhraseBoost::root, 1), 0.0f);

    if (g_failures == 0) {
        std::printf("phrase_boost_unit: ok\n");
    }
    return g_failures == 0 ? 0 : 1;
}
