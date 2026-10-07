// Real-model regression test: the dispatcher releases the per-run compute
// scratch after every offline run / batch (transcribe_session::release_scratch),
// so a long utterance cannot pin its workspace for the session's lifetime, and
// the scheduler rebuild does not change numerics.

#include "transcribe-session.h"  // base-owned sched (internal header)
#include "transcribe.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int    k_rate = 16000;
constexpr double k_pi   = 3.14159265358979323846;  // M_PI is not portable (MSVC)

// Deterministic input whose length controls the workspace size.
std::vector<float> make_pcm(double seconds) {
    const size_t       n = static_cast<size_t>(seconds * k_rate);
    std::vector<float> pcm(n);
    for (size_t i = 0; i < n; ++i) {
        const double t   = static_cast<double>(i) / k_rate;
        const double env = 0.5 + 0.5 * std::sin(2.0 * k_pi * 0.7 * t);
        pcm[i]           = static_cast<float>(0.2 * env * std::sin(2.0 * k_pi * (220.0 + 60.0 * std::sin(t)) * t));
    }
    return pcm;
}

bool sched_live(const transcribe_session * s) {
    return s->sched != nullptr;  // owned by the base session, family-agnostic
}

int fail(const char * what) {
    std::fprintf(stderr, "FAIL %s\n", what);
    return EXIT_FAILURE;
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_GIGAAM_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "gigaam_workspace_release_smoke: TRANSCRIBE_GIGAAM_GGUF not set; skipping.\n"
                     "Re-run with TRANSCRIBE_GIGAAM_GGUF=/path/to/gigaam-v3-e2e-rnnt-Q8_0.gguf\n");
        return 77;
    }

    transcribe_model_load_params lp;
    transcribe_model_load_params_init(&lp);
    lp.backend = TRANSCRIBE_BACKEND_CPU;

    transcribe_session * s = nullptr;
    if (const transcribe_status st = transcribe_open(env, &lp, nullptr, &s); st != TRANSCRIBE_OK || s == nullptr) {
        std::fprintf(stderr, "FAIL transcribe_open(%s): %s\n", env, transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    const std::vector<float> short_pcm = make_pcm(5.0);
    const std::vector<float> long_pcm  = make_pcm(45.0);

    if (transcribe_run(s, short_pcm.data(), static_cast<int>(short_pcm.size()), nullptr) != TRANSCRIBE_OK) {
        return fail("short run #1");
    }
    const std::string text_before = transcribe_full_text(s) ? transcribe_full_text(s) : "";
    if (sched_live(s)) {
        return fail("short run #1: expected the scheduler to be released");
    }
    std::printf("short run #1: scheduler released\n");

    if (transcribe_run(s, long_pcm.data(), static_cast<int>(long_pcm.size()), nullptr) != TRANSCRIBE_OK) {
        return fail("long run");
    }
    if (sched_live(s)) {
        return fail("long run: expected the scheduler to be released");
    }
    std::printf("long run: scheduler released\n");

    if (transcribe_run(s, short_pcm.data(), static_cast<int>(short_pcm.size()), nullptr) != TRANSCRIBE_OK) {
        return fail("short run #2");
    }
    const std::string text_after = transcribe_full_text(s) ? transcribe_full_text(s) : "";
    if (sched_live(s)) {
        return fail("short run #2: expected the scheduler to be released");
    }
    std::printf("short run #2: scheduler released\n");
    if (text_after != text_before) {
        std::fprintf(stderr, "before: '%s'\nafter:  '%s'\n", text_before.c_str(), text_after.c_str());
        return fail("short run #2: transcript changed after the scheduler rebuild");
    }

    const float * pcms[2] = { long_pcm.data(), long_pcm.data() };
    const int     lens[2] = { static_cast<int>(long_pcm.size()), static_cast<int>(long_pcm.size()) };
    if (transcribe_run_batch(s, pcms, lens, 2, nullptr) != TRANSCRIBE_OK) {
        return fail("batch run");
    }
    if (sched_live(s)) {
        return fail("batch run: expected the scheduler to be released");
    }
    std::printf("batch run: scheduler released\n");

    if (transcribe_run(s, short_pcm.data(), static_cast<int>(short_pcm.size()), nullptr) != TRANSCRIBE_OK) {
        return fail("short run #3");
    }
    if (sched_live(s)) {
        return fail("short run #3: expected the scheduler to be released");
    }
    std::printf("short run #3: scheduler released\n");

    transcribe_close(s);
    std::printf("gigaam_workspace_release_smoke: OK\n");
    return EXIT_SUCCESS;
}
