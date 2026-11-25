// smart_turn_smoke.cpp - the toy smart_turn fixture through the public TURN
// API: load, roles, info, clip scores, the crop to the last window, left
// padding, no streaming, and a rejected bad geometry.

#include "transcribe.h"
#include "transcribe/turn.h"
#include "transcribe/vad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

std::string fixture(const char * name) {
    return std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/" + name;
}

transcribe_status load(const char * name, transcribe_model ** m) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    return transcribe_model_load_file(fixture(name).c_str(), &mp, m);
}

std::vector<float> signal(int n, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        seed                      = seed * 1664525u + 1013904223u;
        const float noise         = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.05f;
        v[static_cast<size_t>(i)] = noise + 0.3f * std::sin(0.05f * static_cast<float>(i));
    }
    return v;
}

float score(transcribe_turn_session * s, const float * pcm, int n) {
    CHECK(transcribe_turn_run(s, pcm, n, nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.n_probs == 0 && r.n_speech_probs == 0);
    CHECK(r.ended == (r.p >= 0.5f));
    return r.p;
}

void test_toy() {
    transcribe_model * m = nullptr;
    CHECK(load("arch_smart_turn.gguf", &m) == TRANSCRIBE_OK);
    if (m == nullptr) {
        return;
    }
    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_TURN);

    transcribe_turn_info info;
    transcribe_turn_info_init(&info);
    CHECK(transcribe_turn_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.max_audio_ms == 200 && info.frame_samples == 0);
    CHECK(!info.has_speech && info.threshold == 0.5);

    // Not a VAD model.
    transcribe_vad_info vinfo;
    transcribe_vad_info_init(&vinfo);
    CHECK(transcribe_vad_get_info(m, &vinfo) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(m, nullptr, &s) == TRANSCRIBE_OK);

    const std::vector<float> a = signal(3200, 1);
    const float              p = score(s, a.data(), 3200);
    CHECK(std::isfinite(p) && p > 0.0f && p < 1.0f);

    // Audio before the last window does not change the score.
    std::vector<float> longer = signal(5000, 2);
    longer.insert(longer.end(), a.begin(), a.end());
    CHECK(score(s, longer.data(), static_cast<int>(longer.size())) == p);

    // A short clip is left padded with zeros: the same as passing the zeros.
    std::vector<float> padded(3200, 0.0f);
    std::copy(a.begin(), a.begin() + 1000, padded.begin() + 2200);
    CHECK(score(s, a.data(), 1000) == score(s, padded.data(), 3200));

    // Silence works (the normalisation guards a zero variance).
    const std::vector<float> zeros(800, 0.0f);
    CHECK(std::isfinite(score(s, zeros.data(), 800)));

    // A clip model does not stream.
    CHECK(transcribe_turn_stream_feed(s, a.data(), 640) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);

    transcribe_turn_session_free(s);
    transcribe_model_free(m);
}

void test_bad_window() {
    transcribe_model * m = nullptr;
    CHECK(load("arch_smart_turn_bad_window.gguf", &m) == TRANSCRIBE_ERR_GGUF);
    CHECK(m == nullptr);
}

}  // namespace

int main() {
    test_toy();
    test_bad_window();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("smart_turn_smoke: OK\n");
    return EXIT_SUCCESS;
}
