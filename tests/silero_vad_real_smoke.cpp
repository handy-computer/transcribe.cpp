// silero_vad_real_smoke.cpp - real Silero VAD GGUF / v5.1.2 / v6.2.0 bin through the
// public VAD API, against numbers taken from the reference (upstream JIT, stored
// bin weights imported for bin models; silero-vad 6.2.3 get_speech_timestamps
// with default parameters): the speech segments of samples/jfk.wav sample for
// sample, no speech in samples/noise.wav, and 20 ms stream / offline parity on
// the real weights, on the CPU. Gated by TRANSCRIBE_SILERO_VAD_GGUF / _V5_BIN /
// _V6_BIN (RC 77 skip).

#include "transcribe.h"
#include "transcribe/vad.h"
#include "wav.h"

#include <algorithm>
#include <cstdint>
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

bool load_sample(const char * name, std::vector<float> & pcm) {
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name, pcm, err)) {
        std::fprintf(stderr, "FAIL: %s: %s\n", name, err.c_str());
        ++g_failures;
        return false;
    }
    return true;
}

struct Run {
    std::vector<float>   probs;
    std::vector<int64_t> segments;  // start, end pairs
};

Run run(transcribe_vad_session * s, const std::vector<float> & pcm) {
    Run r;
    CHECK(transcribe_vad_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result res;
    transcribe_vad_result_init(&res);
    transcribe_vad_get_result(s, &res);
    const float * p = transcribe_vad_probs(s);
    if (p != nullptr) {
        r.probs.assign(p, p + res.n_probs);
    }
    for (int i = 0; i < res.n_segments; ++i) {
        transcribe_vad_segment seg;
        transcribe_vad_segment_init(&seg);
        transcribe_vad_get_segment(s, i, &seg);
        r.segments.push_back(seg.start_sample);
        r.segments.push_back(seg.end_sample);
    }
    return r;
}

transcribe_model * load(const char * path) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend           = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * m = nullptr;
    if (transcribe_model_load_file(path, &mp, &m) != TRANSCRIBE_OK) {
        return nullptr;
    }
    return m;
}

}  // namespace

int test_model(const char * path, bool v5, int threads) {
    transcribe_model * m = load(path);
    if (m == nullptr) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", path);
        return EXIT_FAILURE;
    }
    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_VAD);
    CHECK(std::string(transcribe_model_arch_string(m)) == "silero_vad");
    CHECK(transcribe_model_supports(m, TRANSCRIBE_FEATURE_CANCELLATION));
    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    CHECK(transcribe_vad_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.frame_samples == 512);
    CHECK(transcribe_model_device(m) != nullptr);
    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    sp.n_threads = threads;
    CHECK(transcribe_vad_session_init(m, &sp, &s) == TRANSCRIBE_OK);

    std::vector<float> jfk;
    std::vector<float> noise;
    Run                jfk_run;
    if (load_sample("jfk.wav", jfk)) {
        jfk_run = run(s, jfk);
        // get_speech_timestamps(read_audio("samples/jfk.wav"), load_silero_vad())
        const std::vector<int64_t> want =
            v5 ? std::vector<int64_t>{ 4640, 35808, 53280, 60384, 64032, 69600, 86048, 122336, 130592, 169952 } :
                 std::vector<int64_t>{ 5152, 36320, 52256, 71136, 86048, 122848, 130592, 169952 };
        CHECK(jfk_run.probs.size() == 344u);
        CHECK(jfk_run.segments == want);

        // 20 ms chunks, the shape of a live microphone feed.
        std::vector<float>    streamed;
        transcribe_vad_result res;
        transcribe_vad_result_init(&res);
        for (size_t off = 0; off < jfk.size(); off += 320) {
            const int n = static_cast<int>(std::min<size_t>(320, jfk.size() - off));
            CHECK(transcribe_vad_stream_feed(s, jfk.data() + off, n) == TRANSCRIBE_OK);
            transcribe_vad_get_result(s, &res);
            const float * p = transcribe_vad_probs(s);
            if (p != nullptr) {
                streamed.insert(streamed.end(), p, p + res.n_probs);
            }
        }
        CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
        transcribe_vad_get_result(s, &res);
        const float * p = transcribe_vad_probs(s);
        if (p != nullptr) {
            streamed.insert(streamed.end(), p, p + res.n_probs);
        }
        CHECK(streamed == jfk_run.probs);
    }
    if (load_sample("noise.wav", noise)) {
        const Run r = run(s, noise);
        CHECK(r.segments.empty());
        float mx = 0.0f;
        for (const float v : r.probs) {
            mx = std::max(mx, v);
        }
        CHECK(mx < 0.1f);
    }
    transcribe_vad_session_free(s);
    transcribe_model_free(m);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("silero_vad_real_smoke: ok\n");
    return EXIT_SUCCESS;
}

int main() {
    bool         tested = false;
    const char * envs[] = { "TRANSCRIBE_SILERO_VAD_GGUF", "TRANSCRIBE_SILERO_VAD_V5_BIN",
                            "TRANSCRIBE_SILERO_VAD_V6_BIN" };
    for (int i = 0; i < 3; ++i) {
        const char * path = std::getenv(envs[i]);
        if (path == nullptr || path[0] == '\0') {
            continue;
        }
        tested = true;
        for (int threads : { 1, 4 }) {
            std::printf("%s: %d threads\n", envs[i], threads);
            if (test_model(path, i == 1, threads) != EXIT_SUCCESS) {
                ++g_failures;
            }
        }
    }
    if (!tested) {
        std::fprintf(stderr, "SKIP: set TRANSCRIBE_SILERO_VAD_GGUF / _V5_BIN / _V6_BIN\n");
        return 77;
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
