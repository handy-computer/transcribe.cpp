// silero_vad_real_smoke.cpp - real Silero VAD GGUF / v5.1.2 / v6.2.0 bin through the
// public VAD API, against numbers taken from the reference (upstream JIT, stored
// bin weights imported for bin models; silero-vad 6.2.3 probability policies,
// get_speech_timestamps with default parameters): the speech segments of
// samples/jfk.wav sample for sample, no speech in samples/noise.wav, and
// stream / offline parity on the real weights. AUTO must pick the CPU. When
// an explicitly requested GPU backend loads, its probabilities must stay
// within 5e-3 of the CPU's (ggml-metal's F32 mul_mm stages operands as half;
// observed 2e-3) with the same segments (GGUF smoke only). Gated by
// TRANSCRIBE_SILERO_VAD_GGUF / _V5_BIN / _V6_BIN (RC 77 skip).

#include "transcribe.h"
#include "transcribe/vad.h"
#include "wav.h"

#include <cmath>
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

transcribe_model * load(const char * path, transcribe_backend_request backend) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend           = backend;
    transcribe_model * m = nullptr;
    if (transcribe_model_load_file(path, &mp, &m) != TRANSCRIBE_OK) {
        return nullptr;
    }
    return m;
}

}  // namespace

int test_model(const char * path, bool v5, int threads, bool gpu_smoke) {
    transcribe_model * m = load(path, TRANSCRIBE_BACKEND_CPU);
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
    Run                cpu_jfk;
    if (load_sample("jfk.wav", jfk)) {
        cpu_jfk = run(s, jfk);
        // get_speech_timestamps(read_audio("samples/jfk.wav"), load_silero_vad())
        const std::vector<int64_t> want =
            v5 ? std::vector<int64_t>{ 4640, 35808, 53280, 60384, 64032, 69600, 86048, 122336, 130592, 169952 } :
                 std::vector<int64_t>{ 5152, 36320, 52256, 71136, 86048, 122848, 130592, 169952 };
        CHECK(cpu_jfk.probs.size() == 344u);
        CHECK(cpu_jfk.segments == want);

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
        CHECK(streamed == cpu_jfk.probs);
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

    // AUTO resolves to the CPU.
    transcribe_model * a = load(path, TRANSCRIBE_BACKEND_AUTO);
    CHECK(a != nullptr && std::string(transcribe_model_backend(a)) == "CPU");
    transcribe_model_free(a);

    // An explicitly requested GPU backend against the CPU's probabilities.
    const transcribe_backend_request gpus[] = { TRANSCRIBE_BACKEND_METAL, TRANSCRIBE_BACKEND_VULKAN,
                                                TRANSCRIBE_BACKEND_CUDA };
    if (!gpu_smoke) {
        return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    for (const transcribe_backend_request b : gpus) {
        transcribe_model * g = cpu_jfk.probs.empty() ? nullptr : load(path, b);
        if (g == nullptr) {
            continue;
        }
        transcribe_vad_session * gs = nullptr;
        CHECK(transcribe_vad_session_init(g, nullptr, &gs) == TRANSCRIBE_OK);
        const Run r  = run(gs, jfk);
        float     md = 0.0f;
        for (size_t i = 0; i < r.probs.size() && i < cpu_jfk.probs.size(); ++i) {
            md = std::max(md, std::fabs(r.probs[i] - cpu_jfk.probs[i]));
        }
        std::printf("  %s vs CPU: max |dp| %.3g\n", transcribe_model_backend(g), static_cast<double>(md));
        CHECK(r.probs.size() == cpu_jfk.probs.size());
        CHECK(md < 5e-3f);
        CHECK(r.segments == cpu_jfk.segments);
        transcribe_vad_session_free(gs);
        transcribe_model_free(g);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("silero_vad_real_smoke: ok\n");
    return EXIT_SUCCESS;
}

// Numerical tooling uses the same public streaming API, not an internal graph.
int dump_stream(const char * path, const char * audio, const char * output, int threads) {
    transcribe_model * m = load(path, TRANSCRIBE_BACKEND_CPU);
    if (m == nullptr) {
        return EXIT_FAILURE;
    }
    std::vector<float> pcm;
    std::string        err;
    if (!transcribe_cli::load_wav_mono_16k(audio, pcm, err)) {
        transcribe_model_free(m);
        return EXIT_FAILURE;
    }
    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    sp.n_threads               = threads;
    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    if (s == nullptr) {
        transcribe_model_free(m);
        return EXIT_FAILURE;
    }
    std::vector<float> probs;
    auto               collect = [&]() {
        transcribe_vad_result r;
        transcribe_vad_result_init(&r);
        CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(probs.size()));
        const float * p = transcribe_vad_probs(s);
        if (p != nullptr) {
            probs.insert(probs.end(), p, p + r.n_probs);
        }
    };
    for (size_t off = 0; off < pcm.size(); off += 320) {
        CHECK(transcribe_vad_stream_feed(s, pcm.data() + off,
                                         static_cast<int>(std::min<size_t>(320, pcm.size() - off))) == TRANSCRIBE_OK);
        collect();
    }
    CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
    collect();
    FILE * f = std::fopen(output, "wb");
    CHECK(f != nullptr);
    if (f != nullptr) {
        CHECK(std::fwrite(probs.data(), sizeof(float), probs.size(), f) == probs.size());
        CHECK(std::fclose(f) == 0);
    }
    transcribe_vad_session_free(s);
    transcribe_model_free(m);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char ** argv) {
    if (argc == 6 && std::string(argv[1]) == "--stream-probs") {
        return dump_stream(argv[2], argv[3], argv[4], std::atoi(argv[5]));
    }
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
            if (test_model(path, i == 1, threads, i == 0 && threads == 1) != EXIT_SUCCESS) {
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
