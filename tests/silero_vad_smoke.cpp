// silero_vad_smoke.cpp - the toy silero_vad fixture through the public VAD
// API: load, roles and info, a run on synthetic audio, streaming parity with
// the offline run for every chunking (bit for bit, at 1 and 4 threads),
// session reuse, and the rejected-geometry fixture.

#include "transcribe.h"
#include "transcribe/vad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

transcribe_model * load(const char * name, transcribe_status * st_out = nullptr) {
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

void test_model(int n_threads) {
    transcribe_model * m = load("arch_silero_vad.gguf");
    CHECK(m != nullptr);
    if (m == nullptr) {
        return;
    }
    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_VAD);
    CHECK(std::strcmp(transcribe_model_arch_string(m), "silero_vad") == 0);
    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    CHECK(transcribe_vad_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.frame_samples == 512);

    // Not an ASR model.
    transcribe_session * asr = nullptr;
    CHECK(transcribe_session_init(m, nullptr, &asr) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    sp.n_threads               = n_threads;
    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(m, &sp, &s) == TRANSCRIBE_OK);

    // 2.5 s, not a whole number of frames.
    const std::vector<float> pcm = signal(40000 + 123);
    const std::vector<float> off = offline_probs(s, pcm);
    CHECK(off.size() == (pcm.size() + 511) / 512);
    for (const float p : off) {
        CHECK(std::isfinite(p) && p >= 0.0f && p <= 1.0f);
    }
    transcribe_timings tm;
    transcribe_timings_init(&tm);
    CHECK(transcribe_vad_get_timings(s, &tm) == TRANSCRIBE_OK);
    CHECK(tm.encode_ms > 0.0f);

    // A second run from a fresh state gives the same result (state is reset).
    CHECK(offline_probs(s, pcm) == off);

    // Streaming in any chunking equals the offline run, bit for bit: one
    // sample, ragged either side of a frame, and the whole input at once.
    const int chunks[] = { 1, 511, 513, 200000 };
    for (const int c : chunks) {
        const std::vector<float> st = stream_probs(s, pcm, c);
        CHECK(st == off);
        if (st != off) {
            std::fprintf(stderr, "  chunk %d: stream differs (%zu vs %zu probs)\n", c, st.size(), off.size());
        }
    }

    // Cross the encoder's 256-frame block boundary with a partial tail.
    const std::vector<float> long_pcm = signal(256 * 512 + 1);
    const std::vector<float> long_off = offline_probs(s, long_pcm);
    CHECK(long_off.size() == 257u);
    for (const int c : { 511, 513, 131072, 131073 }) {
        CHECK(stream_probs(s, long_pcm, c) == long_off);
    }

    // A run in the middle of a stream resets it.
    CHECK(transcribe_vad_stream_feed(s, pcm.data(), 1000) == TRANSCRIBE_OK);
    CHECK(offline_probs(s, pcm) == off);
    CHECK(stream_probs(s, pcm, 777) == off);

    transcribe_vad_session_free(s);
    transcribe_model_free(m);
}

void test_bad_geometry() {
    transcribe_status  st = TRANSCRIBE_OK;
    transcribe_model * m  = load("arch_silero_vad_bad_frame.gguf", &st);
    CHECK(m == nullptr);
    CHECK(st == TRANSCRIBE_ERR_GGUF);
    transcribe_model_free(m);
}

}  // namespace

int main() {
    test_model(1);
    test_model(4);
    test_bad_geometry();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("silero_vad_smoke: ok\n");
    return EXIT_SUCCESS;
}
