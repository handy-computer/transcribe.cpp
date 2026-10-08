// ecapa_tdnn_smoke.cpp - end-to-end smoke of the ecapa_tdnn family (LANGID
// role) against tiny synthetic GGUFs, through the public C API.
//
// arch_ecapa_tdnn_minimal.gguf (tests/fixtures/make_gguf_fixtures.py) is a
// 1/32-width model with the exact metadata and tensor contract of the real
// VoxLingua107 file, so this drives the production loader, front end, ggml
// graph and role dispatcher over the same code path the real checkpoint
// uses. It asserts structure and invariants (statuses, orderings,
// determinism), never specific values, because the weights are random.
//
// Everything runs on the CPU backend: the thread-count comparison below is
// only meaningful there, and it keeps the test hermetic on GPU-equipped CI.

#include "transcribe.h"
#include "transcribe/langid.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Deterministic pseudo-noise in [-0.5, 0.5); a hand-rolled LCG so the
// samples are identical on every platform and standard library.
std::vector<float> noise(size_t n, uint32_t seed) {
    std::vector<float> out(n);
    uint32_t           s = seed | 1u;
    for (size_t i = 0; i < n; ++i) {
        s      = s * 1664525u + 1013904223u;
        out[i] = static_cast<float>((s >> 8) & 0xFFFFFF) / 16777216.0f - 0.5f;
    }
    return out;
}

std::string fixture(const char * name) {
    return std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/" + name;
}

transcribe_model * load_cpu(const char * name, transcribe_status * st_out = nullptr) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                 = TRANSCRIBE_BACKEND_CPU;
    transcribe_model *      m  = nullptr;
    const transcribe_status st = transcribe_model_load_file(fixture(name).c_str(), &mp, &m);
    if (st_out != nullptr) {
        *st_out = st;
    }
    return m;
}

transcribe_langid_session * open_session(transcribe_model * m, int n_threads, int max_audio_ms = 0) {
    transcribe_langid_session_params sp;
    transcribe_langid_session_params_init(&sp);
    sp.n_threads                  = n_threads;
    sp.max_audio_ms               = max_audio_ms;
    transcribe_langid_session * s = nullptr;
    CHECK(transcribe_langid_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    return s;
}

transcribe_langid_result result_of(const transcribe_langid_session * s) {
    transcribe_langid_result r;
    transcribe_langid_result_init(&r);
    CHECK(transcribe_langid_get_result(s, &r) == TRANSCRIBE_OK);
    return r;
}

std::vector<transcribe_langid_candidate> candidates_of(const transcribe_langid_session * s) {
    const transcribe_langid_result           r = result_of(s);
    std::vector<transcribe_langid_candidate> out(static_cast<size_t>(r.n_candidates));
    for (int i = 0; i < r.n_candidates; ++i) {
        transcribe_langid_candidate_init(&out[static_cast<size_t>(i)]);
        CHECK(transcribe_langid_get_candidate(s, i, &out[static_cast<size_t>(i)]) == TRANSCRIBE_OK);
    }
    return out;
}

void test_model_surface(transcribe_model * m) {
    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_LANGID);
    CHECK(std::strcmp(transcribe_model_arch_string(m), "ecapa_tdnn") == 0);
    CHECK(std::strcmp(transcribe_model_variant_string(m), "ecapa-tdnn-toy") == 0);
    CHECK(transcribe_model_supports(m, TRANSCRIBE_FEATURE_CANCELLATION));

    transcribe_langid_info info;
    transcribe_langid_info_init(&info);
    CHECK(transcribe_langid_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.n_labels == 5 && info.min_audio_ms == 500);
    CHECK(std::strcmp(transcribe_langid_label_code(m, 2), "cc") == 0);
    CHECK(std::strcmp(transcribe_langid_label_name(m, 2), "Charlie") == 0);
    CHECK(transcribe_langid_label_index(m, "xx") == 0);  // alias from the GGUF

    // Not an ASR model.
    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_model_get_capabilities(m, &caps) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    transcribe_session * asr = nullptr;
    CHECK(transcribe_session_init(m, nullptr, &asr) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(asr == nullptr);
}

void test_run(transcribe_model * m) {
    transcribe_langid_session * s   = open_session(m, 1);
    const std::vector<float>    pcm = noise(16000, 7);  // 1 s

    CHECK(transcribe_langid_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    const transcribe_langid_result r = result_of(s);
    CHECK(r.n_candidates == 5 && r.n_allowed == 5 && r.allowed_mass == 1.0f && r.audio_ms == 1000);
    const auto first = candidates_of(s);
    double     sum   = 0.0;
    for (size_t i = 0; i < first.size(); ++i) {
        CHECK(std::isfinite(first[i].p) && std::isfinite(first[i].logit));
        CHECK(first[i].code != nullptr && first[i].name != nullptr);
        if (i > 0) {
            CHECK(first[i - 1].p >= first[i].p);
        }
        sum += first[i].p;
    }
    CHECK(std::fabs(sum - 1.0) < 1e-5);

    // Determinism: a second run is bit-identical.
    CHECK(transcribe_langid_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    const auto again = candidates_of(s);
    CHECK(again.size() == first.size());
    for (size_t i = 0; i < first.size() && i < again.size(); ++i) {
        CHECK(again[i].index == first[i].index && again[i].logit == first[i].logit);
    }

    // Silence is valid input.
    const std::vector<float> zeros(16000, 0.0f);
    CHECK(transcribe_langid_run(s, zeros.data(), 16000, nullptr) == TRANSCRIBE_OK);

    // Abort before compute.
    transcribe_langid_set_abort_callback(s, [](void *) { return true; }, nullptr);
    CHECK(transcribe_langid_run(s, pcm.data(), 16000, nullptr) == TRANSCRIBE_ERR_ABORTED);
    CHECK(result_of(s).n_candidates == 0);
    transcribe_langid_set_abort_callback(s, nullptr, nullptr);

    transcribe_langid_session_free(s);
}

// The front end and graph give the same logits for any thread count (the
// mel frames and the ggml CPU rows are split without changing any sum).
void test_thread_invariance(transcribe_model * m) {
    const std::vector<float>    pcm = noise(16000 * 3, 3);
    transcribe_langid_session * s1  = open_session(m, 1);
    transcribe_langid_session * s4  = open_session(m, 4);
    CHECK(transcribe_langid_run(s1, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_langid_run(s4, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    const auto a = candidates_of(s1);
    const auto b = candidates_of(s4);
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        CHECK(a[i].index == b[i].index && std::fabs(a[i].logit - b[i].logit) < 1e-5f);
    }
    transcribe_langid_session_free(s1);
    transcribe_langid_session_free(s4);
}

// Q8_0 weights are widened to F16 at load: a Q8_0 model must give exactly
// the logits of the same model stored as F16 holding the dequantized values.
void test_q8_0_widened_to_f16() {
    transcribe_model * q8  = load_cpu("arch_ecapa_tdnn_q8_0.gguf");
    transcribe_model * ref = load_cpu("arch_ecapa_tdnn_q8_0_as_f16.gguf");
    CHECK(q8 != nullptr && ref != nullptr);
    if (q8 == nullptr || ref == nullptr) {
        transcribe_model_free(q8);
        transcribe_model_free(ref);
        return;
    }
    const std::vector<float>    pcm = noise(16000 * 2, 5);
    transcribe_langid_session * sq  = open_session(q8, 2);
    transcribe_langid_session * sr  = open_session(ref, 2);
    CHECK(transcribe_langid_run(sq, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_langid_run(sr, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    const auto a = candidates_of(sq);
    const auto b = candidates_of(sr);
    CHECK(a.size() == 5 && a.size() == b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        CHECK(a[i].index == b[i].index && a[i].logit == b[i].logit);
    }
    transcribe_langid_session_free(sq);
    transcribe_langid_session_free(sr);
    transcribe_model_free(q8);
    transcribe_model_free(ref);
}

}  // namespace

int main() {
    transcribe_log_set(nullptr, nullptr);

    transcribe_status  st = TRANSCRIBE_OK;
    transcribe_model * m  = load_cpu("arch_ecapa_tdnn_minimal.gguf", &st);
    if (st != TRANSCRIBE_OK || m == nullptr) {
        std::fprintf(stderr, "FAIL: load arch_ecapa_tdnn_minimal.gguf: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }
    test_model_surface(m);
    test_run(m);
    test_thread_invariance(m);
    transcribe_model_free(m);

    test_q8_0_widened_to_f16();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("ecapa_tdnn_smoke: ok\n");
    return EXIT_SUCCESS;
}
