// Canary vocabulary bias (pure host, no model). Too strong and a listed term
// replaces unrelated speech; too weak and a term the audio does carry loses to
// a near miss. See src/arch/canary/vocab_boost.h.

#include "arch/canary/vocab_boost.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

void expect(const char * what, bool ok) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

void unset_env(const char * key) {
#if defined(_WIN32)
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

constexpr int k_vocab = 8;

std::vector<float> logits(int top, float top_score, int other, float other_score) {
    std::vector<float> out(k_vocab, -10.0f);
    out[top]   = top_score;
    out[other] = other_score;
    return out;
}

}  // namespace

int main(void) {
    using transcribe::canary::VocabBoost;

    VocabBoost boost;
    boost.terms = {
        { 5, 6 }
    };  // a two-token term
    boost.strength = 3.0f;

    // A term's first token wins a near miss, not a clear winner.
    expect("first token beats a near miss", boost.pick(logits(1, 0.0f, 5, -2.0f).data(), k_vocab, {}) == 5);
    expect("first token loses to a clear winner", boost.pick(logits(1, 0.0f, 5, -4.0f).data(), k_vocab, {}) == 1);

    // The next token is boosted only right after the term's previous tokens.
    expect("continuation after its prefix", boost.pick(logits(1, 0.0f, 6, -2.0f).data(), k_vocab, { 3, 5 }) == 6);
    expect("no continuation elsewhere", boost.pick(logits(1, 0.0f, 6, -2.0f).data(), k_vocab, { 3, 4 }) == 1);

    // Off: plain argmax.
    boost.strength = 0.0f;
    expect("strength 0 is argmax", boost.pick(logits(1, 0.0f, 5, -0.5f).data(), k_vocab, {}) == 1);
    expect("strength 0 is inactive", !boost.active());

    // Out-of-range term ids are skipped, not read.
    VocabBoost bad;
    bad.terms    = { { k_vocab + 3 } };
    bad.strength = 3.0f;
    expect("out-of-range id skipped", bad.pick(logits(2, 0.0f, 4, -1.0f).data(), k_vocab, {}) == 2);

    // Strength from the environment, default when unset or not a number.
    unset_env("TRANSCRIBE_CANARY_VOCAB_BOOST");
    expect("default strength", transcribe::canary::vocab_boost_strength() == transcribe::canary::k_vocab_boost_default);
    set_env("TRANSCRIBE_CANARY_VOCAB_BOOST", "1.5");
    expect("env strength", transcribe::canary::vocab_boost_strength() == 1.5f);
    set_env("TRANSCRIBE_CANARY_VOCAB_BOOST", "0");
    expect("env turns it off", transcribe::canary::vocab_boost_strength() == 0.0f);
    set_env("TRANSCRIBE_CANARY_VOCAB_BOOST", "high");
    expect("garbage keeps the default",
           transcribe::canary::vocab_boost_strength() == transcribe::canary::k_vocab_boost_default);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("canary_vocab_boost_unit: ok\n");
    return 0;
}
