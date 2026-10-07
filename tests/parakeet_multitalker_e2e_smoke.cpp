// parakeet_multitalker_e2e_smoke.cpp - real-model gated end-to-end test
// for the multitalker bundle (parakeet + embedded sortformer).
//
// Gating (same pattern as parakeet_real_smoke):
//   - Built only under TRANSCRIBE_BUILD_REAL_MODEL_TESTS.
//   - The bundle path comes from TRANSCRIBE_MULTITALKER_BUNDLE_GGUF; if
//     unset, exits 77 (CTest "skipped").
//
// What we assert, on samples/multitalker-2spk-mix.wav (the deterministic
// 2-speaker fixture whose NeMo reference lives under
// build/validate/parakeet/.../multitalker/):
//
//   1. The bundle loads and reports TRANSCRIBE_FEATURE_DIARIZATION.
//   2. diarize=OFF: single-speaker semantics — a result with every
//      segment's speaker_id == 0 and no speaker_segment rows.
//   3. diarize=ON with TRANSCRIBE_MULTITALKER_MODE=masked: >= 2 segments
//      carrying at least two distinct 1-based speaker_ids, speaker_segment
//      rows present, and each fixture voice's anchor phrase attributed to a
//      single speaker ("fellow Americans" for the JFK track,
//      "publication" for the Whole Earth track).
//   4. Same invariants in the default kernel mode.

#include "transcribe.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// speaker_id owning the segment whose text contains `needle`; 0 if absent
// or ambiguous (present under more than one speaker).
int32_t phrase_speaker(struct transcribe_session * ctx, const char * needle) {
    std::set<int32_t> owners;
    const int         n = transcribe_n_segments(ctx);
    for (int i = 0; i < n; ++i) {
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        if (transcribe_get_segment(ctx, i, &seg) != TRANSCRIBE_OK || seg.text == nullptr) {
            continue;
        }
        if (std::strstr(seg.text, needle) != nullptr) {
            owners.insert(seg.speaker_id);
        }
    }
    return owners.size() == 1 ? *owners.begin() : 0;
}

void run_multitalker_checks(struct transcribe_session * ctx, const std::vector<float> & pcm, const char * label) {
    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.diarize = TRANSCRIBE_DIARIZE_MODE_ON;

    const transcribe_status st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
    std::fprintf(stderr, "[%s] run: %s\n", label, transcribe_status_string(st));
    check(st == TRANSCRIBE_OK, "diarize=ON run returns OK");
    if (st != TRANSCRIBE_OK) {
        return;
    }

    const int n_segments = transcribe_n_segments(ctx);
    check(n_segments >= 2, "diarize=ON yields >= 2 segments");

    std::set<int32_t> speakers;
    for (int i = 0; i < n_segments; ++i) {
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        if (transcribe_get_segment(ctx, i, &seg) == TRANSCRIBE_OK) {
            std::fprintf(stderr, "[%s]   S%d [%lld, %lld] %s\n", label, seg.speaker_id,
                         static_cast<long long>(seg.t0_ms), static_cast<long long>(seg.t1_ms),
                         seg.text != nullptr ? seg.text : "<null>");
            speakers.insert(seg.speaker_id);
        }
    }
    check(speakers.size() >= 2, "segments carry >= 2 distinct speaker_ids");
    check(speakers.count(0) == 0, "every segment is speaker-attributed (no speaker_id 0)");

    check(transcribe_n_speaker_segments(ctx) > 0, "speaker_segment rows populated");

    const int32_t spk_a = phrase_speaker(ctx, "fellow Americans");
    const int32_t spk_b = phrase_speaker(ctx, "publication");
    check(spk_a > 0, "JFK anchor phrase attributed to exactly one speaker");
    check(spk_b > 0, "Whole Earth anchor phrase attributed to exactly one speaker");
    check(spk_a == 0 || spk_b == 0 || spk_a != spk_b, "the two voices land on different speakers");
}

}  // namespace

int main() {
    const char * gguf = std::getenv("TRANSCRIBE_MULTITALKER_BUNDLE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::fprintf(stderr,
                     "SKIP: set TRANSCRIBE_MULTITALKER_BUNDLE_GGUF to a multitalker bundle GGUF "
                     "(scripts/compose-multitalker-bundle.py output)\n");
        return 77;
    }
    const std::string wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/multitalker-2spk-mix.wav";

    struct transcribe_model * model = nullptr;
    {
        const transcribe_status st = transcribe_model_load_file(gguf, nullptr, &model);
        if (st != TRANSCRIBE_OK || model == nullptr) {
            std::fprintf(stderr, "FAIL: model load: %s\n", transcribe_status_string(st));
            return EXIT_FAILURE;
        }
    }

    check(transcribe_model_supports(model, TRANSCRIBE_FEATURE_DIARIZATION),
          "bundle reports TRANSCRIBE_FEATURE_DIARIZATION");

    std::vector<float> pcm;
    std::string        load_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path.c_str(), pcm, load_err)) {
        std::fprintf(stderr, "FAIL: wav load: %s\n", load_err.c_str());
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    struct transcribe_session * ctx = nullptr;
    if (transcribe_session_init(model, nullptr, &ctx) != TRANSCRIBE_OK || ctx == nullptr) {
        std::fprintf(stderr, "FAIL: session init\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // diarize=OFF: shipped single-speaker semantics.
    {
        const transcribe_status st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), nullptr);
        check(st == TRANSCRIBE_OK, "diarize=OFF run returns OK");
        if (st == TRANSCRIBE_OK) {
            check(transcribe_n_segments(ctx) >= 1, "diarize=OFF yields a result");
            check(transcribe_n_speaker_segments(ctx) == 0, "diarize=OFF has no speaker_segment rows");
            struct transcribe_segment seg;
            transcribe_segment_init(&seg);
            if (transcribe_get_segment(ctx, 0, &seg) == TRANSCRIBE_OK) {
                check(seg.speaker_id == 0, "diarize=OFF segments are unattributed");
            }
        }
    }

    // diarize=ON in both supervision modes. Select each mode explicitly so
    // a change to the runtime default cannot silently reduce test coverage.
    setenv("TRANSCRIBE_MULTITALKER_MODE", "masked", /*overwrite=*/1);
    run_multitalker_checks(ctx, pcm, "masked");
    setenv("TRANSCRIBE_MULTITALKER_MODE", "kernel", /*overwrite=*/1);
    run_multitalker_checks(ctx, pcm, "kernel");
    unsetenv("TRANSCRIBE_MULTITALKER_MODE");

    transcribe_session_free(ctx);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "parakeet_multitalker_e2e_smoke: all checks passed\n");
    return EXIT_SUCCESS;
}
