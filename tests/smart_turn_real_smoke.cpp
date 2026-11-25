// smart_turn_real_smoke.cpp - the real Smart Turn v3.2 F32 GGUF through the
// public TURN API, against onnxruntime on upstream's fp32 ONNX (numbers from
// scripts/dump_reference_smart_turn_onnx.py, ref/turn.json): the clip score
// of committed samples to 1e-4, the decision, the 8 s crop, and a threshold
// override. CPU. TRANSCRIBE_SMART_TURN_GGUF, else the default F32 GGUF
// under models/ (RC 77 skip when neither exists).

#include "transcribe.h"
#include "transcribe/turn.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

transcribe_turn_result run(transcribe_turn_session *      s,
                           const std::vector<float> &     pcm,
                           const transcribe_turn_params * params = nullptr) {
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), params) == TRANSCRIBE_OK);
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    return r;
}

struct Case {
    const char * wav;
    double       p_ref;  // onnxruntime, fp32 ONNX
    bool         ended;
};

// The reference decision is p > 0.5; none of these sit on the cut-off.
const Case k_cases[] = {
    { "jfk.wav",                0.9486198425292969,   true  }, // 11 s: cropped to the last 8 s
    { "turn-complete-en.wav",   0.9906327724456787,   true  },
    { "turn-incomplete-en.wav", 0.005582541227340698, false },
    { "turn-uncertain-en.wav",  0.27423417568206787,  false },
    { "silence.wav",            0.4615057110786438,   false },
};

}  // namespace

int main() {
    const char *      env  = std::getenv("TRANSCRIBE_SMART_TURN_GGUF");
    const std::string file = env != nullptr && env[0] != '\0' ? env : TRANSCRIBE_SMART_TURN_DEFAULT_GGUF;
    if (!std::ifstream(file).good()) {
        std::fprintf(stderr, "skip: no Smart Turn GGUF at %s (set TRANSCRIBE_SMART_TURN_GGUF)\n", file.c_str());
        return 77;
    }
    const char *                 path = file.c_str();
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend           = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * m = nullptr;
    if (transcribe_model_load_file(path, &mp, &m) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", path);
        return EXIT_FAILURE;
    }
    transcribe_turn_info info;
    transcribe_turn_info_init(&info);
    CHECK(transcribe_turn_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.max_audio_ms == 8000 && info.frame_samples == 0 && !info.has_speech && info.threshold == 0.5);

    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(m, nullptr, &s) == TRANSCRIBE_OK);

    for (const Case & c : k_cases) {
        std::vector<float> pcm;
        if (!load_sample(c.wav, pcm)) {
            continue;
        }
        const transcribe_turn_result r = run(s, pcm);
        std::printf("%-24s p %.6f (ref %.6f, |dp| %.1e) ended %d\n", c.wav, static_cast<double>(r.p), c.p_ref,
                    std::fabs(r.p - c.p_ref), r.ended ? 1 : 0);
        CHECK(std::fabs(r.p - c.p_ref) <= 1e-4);
        CHECK(r.ended == c.ended);
    }

    // The 8 s crop: jfk's last 128000 samples score the same as the whole clip.
    std::vector<float> jfk;
    if (load_sample("jfk.wav", jfk) && jfk.size() > 128000) {
        const float              whole = run(s, jfk).p;
        const std::vector<float> tail(jfk.end() - 128000, jfk.end());
        CHECK(run(s, tail).p == whole);
    }

    // A threshold override changes the decision, not the score.
    std::vector<float> unc;
    if (load_sample("turn-uncertain-en.wav", unc)) {
        transcribe_turn_params tp;
        transcribe_turn_params_init(&tp);
        tp.threshold                   = 0.2;
        const transcribe_turn_result r = run(s, unc, &tp);
        CHECK(r.ended && std::fabs(r.p - 0.27423417568206787) <= 1e-4);
    }

    transcribe_turn_session_free(s);
    transcribe_model_free(m);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("smart_turn_real_smoke: OK\n");
    return EXIT_SUCCESS;
}
