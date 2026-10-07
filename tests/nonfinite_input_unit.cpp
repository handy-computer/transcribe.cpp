// nonfinite_input_unit.cpp - every ASR PCM entry point rejects NaN / Inf
// samples with INVALID_ARG before any result or stream state is modified
// and before the family hook runs (D16a).

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

int g_run_calls       = 0;
int g_run_batch_calls = 0;
int g_feed_calls      = 0;

transcribe_status fake_run(transcribe_session * s, const float *, int, const transcribe_run_params *) {
    ++g_run_calls;
    s->full_text  = "fresh";
    s->has_result = true;
    return TRANSCRIBE_OK;
}

transcribe_status fake_run_batch(transcribe_session * s,
                                 const float * const *,
                                 const int *,
                                 int n,
                                 const transcribe_run_params *) {
    ++g_run_batch_calls;
    for (int i = 0; i < n; ++i) {
        s->batch_results.push_back(s->capture_result(TRANSCRIBE_OK));
    }
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_begin(transcribe_session *,
                                    const transcribe_run_params *,
                                    const transcribe_stream_params *) {
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_feed(transcribe_session *, const float *, int, transcribe_stream_update *) {
    ++g_feed_calls;
    return TRANSCRIBE_OK;
}

transcribe_status fake_stream_finalize(transcribe_session *, transcribe_stream_update *) {
    return TRANSCRIBE_OK;
}

const transcribe::Arch k_arch = {
    /* .name             = */ "fake-nonfinite",
    /* .load             = */ nullptr,
    /* .init_context     = */ nullptr,
    /* .run              = */ fake_run,
    /* .run_batch        = */ fake_run_batch,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ fake_stream_begin,
    /* .stream_feed      = */ fake_stream_feed,
    /* .stream_finalize  = */ fake_stream_finalize,
};

const float k_nan    = std::numeric_limits<float>::quiet_NaN();
const float k_inf    = std::numeric_limits<float>::infinity();
const float k_bads[] = { k_nan, k_inf, -k_inf };

// A session holding a prior single result and a prior batch result.
void seed(transcribe_session & s, transcribe_model & m) {
    s.model      = &m;
    s.full_text  = "previous";
    s.has_result = true;
    s.batch_results.clear();
    s.batch_results.push_back(s.capture_result(TRANSCRIBE_OK));
}

bool prior_intact(const transcribe_session & s) {
    return s.full_text == "previous" && s.has_result && s.batch_results.size() == 1;
}

void test_run_rejects_nonfinite() {
    transcribe_model model;
    model.arch = &k_arch;
    for (float bad : k_bads) {
        transcribe_session s;
        seed(s, model);
        std::vector<float> pcm(1600, 0.1f);
        pcm[799]    = bad;
        g_run_calls = 0;
        CHECK(transcribe_run(&s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(g_run_calls == 0);
        CHECK(prior_intact(s));
    }
}

void test_batch_rejects_nonfinite_whole_batch() {
    transcribe_model model;
    model.arch = &k_arch;
    for (float bad : k_bads) {
        transcribe_session s;
        seed(s, model);
        std::vector<float> a(800, 0.1f), b(800, 0.1f), c(800, 0.1f);
        b[400]                    = bad;
        const float * pcm[]       = { a.data(), b.data(), c.data() };
        const int     n_samples[] = { 800, 800, 800 };
        g_run_calls               = 0;
        g_run_batch_calls         = 0;
        CHECK(transcribe_run_batch(&s, pcm, n_samples, 3, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(g_run_calls == 0);
        CHECK(g_run_batch_calls == 0);
        CHECK(prior_intact(s));
    }
}

void test_stream_feed_rejects_nonfinite() {
    transcribe_model model;
    model.arch                    = &k_arch;
    model.caps.supports_streaming = true;
    transcribe_session s;
    s.model = &model;
    CHECK(transcribe_stream_begin(&s, nullptr, nullptr) == TRANSCRIBE_OK);
    const int revision = transcribe_stream_revision(&s);

    for (float bad : k_bads) {
        std::vector<float> pcm(1600, 0.1f);
        pcm[1000]    = bad;
        g_feed_calls = 0;
        transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        CHECK(transcribe_stream_feed(&s, pcm.data(), static_cast<int>(pcm.size()), &upd) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(g_feed_calls == 0);
        // The stream stays usable: rejected before the hook, so no FAILED.
        CHECK(transcribe_stream_get_state(&s) == TRANSCRIBE_STREAM_ACTIVE);
        CHECK(transcribe_stream_revision(&s) == revision);
    }

    std::vector<float> ok(1600, 0.0f);
    g_feed_calls = 0;
    CHECK(transcribe_stream_feed(&s, ok.data(), static_cast<int>(ok.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(g_feed_calls == 1);
    CHECK(transcribe_stream_finalize(&s, nullptr) == TRANSCRIBE_OK);
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);

    test_run_rejects_nonfinite();
    test_batch_rejects_nonfinite_whole_batch();
    test_stream_feed_rejects_nonfinite();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("nonfinite_input_unit: ok\n");
    return EXIT_SUCCESS;
}
