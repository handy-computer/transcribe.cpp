// silero_vad_smoke.cpp - the toy silero_vad fixture through the public VAD
// API: load, info, a run on synthetic audio, and streaming parity with the
// offline run (bit for bit) across ragged chunkings.

#include "transcribe.h"
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

transcribe_model * load(const char * name) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend           = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * m = nullptr;
    transcribe_model_load_file(fixture(name).c_str(), &mp, &m);
    return m;
}

// Deterministic test signal: bursts of a chirp over low noise.
std::vector<float> signal(int n) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t           seed = 12345;
    for (int i = 0; i < n; ++i) {
        seed                      = seed * 1664525u + 1013904223u;
        const float noise         = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.01f;
        const float t             = static_cast<float>(i) / 16000.0f;
        const bool  burst         = (i / 8000) % 3 != 0;
        v[static_cast<size_t>(i)] = noise + (burst ? 0.3f * std::sin(6.2831853f * (200.0f + 400.0f * t) * t) : 0.0f);
    }
    return v;
}

std::vector<float> offline_probs(transcribe_vad_session * s, const std::vector<float> & pcm) {
    CHECK(transcribe_vad_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    const float * p = transcribe_vad_probs(s);
    return p != nullptr ? std::vector<float>(p, p + r.n_probs) : std::vector<float>();
}

std::vector<float> stream_probs(transcribe_vad_session * s, const std::vector<float> & pcm, int chunk) {
    std::vector<float>    out;
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    for (size_t off = 0; off < pcm.size(); off += static_cast<size_t>(chunk)) {
        const int n = static_cast<int>(std::min(static_cast<size_t>(chunk), pcm.size() - off));
        CHECK(transcribe_vad_stream_feed(s, pcm.data() + off, n) == TRANSCRIBE_OK);
        CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(out.size()));
        const float * p = transcribe_vad_probs(s);
        if (p != nullptr) {
            out.insert(out.end(), p, p + r.n_probs);
        }
    }
    CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    const float * p = transcribe_vad_probs(s);
    if (p != nullptr) {
        out.insert(out.end(), p, p + r.n_probs);
    }
    return out;
}

void test_model() {
    transcribe_model * m = load("arch_silero_vad.gguf");
    CHECK(m != nullptr);
    if (m == nullptr) {
        return;
    }
    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_VAD);
    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    CHECK(transcribe_vad_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.frame_samples == 512);

    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(m, nullptr, &s) == TRANSCRIBE_OK);

    // Crosses the encoder's 256-frame block boundary with a partial tail.
    const std::vector<float> pcm = signal(256 * 512 + 123);
    const std::vector<float> off = offline_probs(s, pcm);
    CHECK(off.size() == (pcm.size() + 511) / 512);
    for (const float p : off) {
        CHECK(std::isfinite(p) && p >= 0.0f && p <= 1.0f);
    }

    // Streaming in any chunking equals the offline run, bit for bit.
    for (const int c : { 1, 511, 513, 131073 }) {
        CHECK(stream_probs(s, pcm, c) == off);
    }

    transcribe_vad_session_free(s);
    transcribe_model_free(m);
}

}  // namespace

int main() {
    test_model();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("silero_vad_smoke: ok\n");
    return EXIT_SUCCESS;
}
