// wav_loader_smoke.cpp - unit smoke for the example WAV loader.
//
// The WAV loader lives under examples/common/ (not the core library)
// because the core never decodes WAV. We still want it under ctest so a
// failure at the example boundary is caught by the same gate as the
// public ABI tests.
//
// Two cases:
//   1. samples/jfk.wav loads cleanly: 16 kHz mono, non-zero samples.
//   2. A bogus path produces a non-empty error message and clears the
//      output (loosely - the contract leaves out_pcm "unspecified" on
//      failure but the error string must be populated).
//
// Both samples used here are committed to the repo, so the test always
// has fixtures and never needs the SKIP_RETURN_CODE dance.

#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_SAMPLES_DIR
#    error "TRANSCRIBE_TEST_SAMPLES_DIR must be defined by the build system"
#endif

namespace {

const std::string g_samples_dir = TRANSCRIBE_TEST_SAMPLES_DIR;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void test_load_jfk() {
    const std::string  path = g_samples_dir + "/jfk.wav";
    std::vector<float> pcm;
    std::string        err;

    const bool ok = transcribe_cli::load_wav_mono_16k(path, pcm, err);
    if (!ok) {
        std::fprintf(stderr, "FAIL load_jfk: %s\n", err.c_str());
        ++g_failures;
        return;
    }
    CHECK(err.empty());
    CHECK(!pcm.empty());

    // jfk.wav is the canonical 11-second JFK clip used by whisper.cpp /
    // ggml. We don't pin an exact sample count (so we can swap the
    // sample without breaking the test) but we do require it to be
    // bigger than half a second of audio.
    CHECK(pcm.size() > 16000u / 2u);

    // Sample range sanity: dr_wav normalizes to [-1, 1] for float reads.
    // We don't require that every sample is non-zero, just that at least
    // one sample escapes the silence floor; otherwise the file would be
    // a quiet 11-second nothing.
    bool any_nonzero = false;
    for (float s : pcm) {
        if (s != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    CHECK(any_nonzero);
}

void test_load_missing() {
    std::vector<float> pcm = { 1.0f, 2.0f, 3.0f };
    std::string        err = "stale";
    const bool         ok  = transcribe_cli::load_wav_mono_16k("/__wav_loader_smoke_missing__.wav", pcm, err);
    CHECK(!ok);
    CHECK(!err.empty());
    // out_pcm.clear() is the loader's first line; relying on it pins the
    // documented "leaves out_pcm in an unspecified state" loosely toward
    // "cleared" without making it part of the public contract.
    CHECK(pcm.empty());
}

}  // namespace

int main() {
    test_load_jfk();
    test_load_missing();

    if (g_failures > 0) {
        std::fprintf(stderr, "wav_loader_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "wav_loader_smoke: ok\n");
    return EXIT_SUCCESS;
}
