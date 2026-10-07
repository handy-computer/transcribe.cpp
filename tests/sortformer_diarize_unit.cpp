// sortformer_diarize_unit.cpp - Sortformer on the DIARIZE role, against a
// real GGUF (TRANSCRIBE_SORTFORMER_GGUF; RC 77 skip when unset).

#include "transcribe.h"
#include "transcribe/diarize.h"
#include "transcribe/sortformer.h"
#include "wav.h"

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

struct Row {
    int64_t t0, t1;
    int32_t spk;

    bool operator==(const Row & o) const { return t0 == o.t0 && t1 == o.t1 && spk == o.spk; }
};

std::vector<Row> diarize_rows(const transcribe_diarize_session * s) {
    std::vector<Row> rows;
    for (int i = 0; i < transcribe_diarize_n_segments(s); ++i) {
        transcribe_speaker_segment r;
        transcribe_speaker_segment_init(&r);
        transcribe_diarize_get_segment(s, i, &r);
        rows.push_back({ r.t0_ms, r.t1_ms, r.speaker_id });
    }
    return rows;
}

transcribe_model * load_cpu(const char * path) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend               = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    return transcribe_model_load_file(path, &mp, &model) == TRANSCRIBE_OK ? model : nullptr;
}

std::vector<Row> diarize(transcribe_diarize_session * s,
                         const std::vector<float> &   pcm,
                         transcribe_sortformer_preset preset) {
    transcribe_sortformer_diarize_ext ext;
    transcribe_sortformer_diarize_ext_init(&ext);
    ext.preset = preset;
    transcribe_diarize_params dp;
    transcribe_diarize_params_init(&dp);
    dp.family = &ext.ext;
    CHECK(transcribe_diarize_run(s, pcm.data(), static_cast<int>(pcm.size()), &dp) == TRANSCRIBE_OK);
    return diarize_rows(s);
}

}  // namespace

int main() {
    const char * path = std::getenv("TRANSCRIBE_SORTFORMER_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "sortformer_diarize_unit: TRANSCRIBE_SORTFORMER_GGUF not set; skipping.\n");
        return 77;
    }
    std::vector<float> pcm;
    std::string        err;
    if (!transcribe_cli::load_wav_mono_16k(std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/sortformer-2spk-mix.wav", pcm,
                                           err)) {
        std::fprintf(stderr, "sortformer_diarize_unit: wav: %s\n", err.c_str());
        return 77;
    }
    ::unsetenv("TRANSCRIBE_SORTFORMER_STREAM_PRESET");

    transcribe_model * model = load_cpu(path);
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }

    // 1. Roles, info, extension probe.
    CHECK(transcribe_model_roles(model) == TRANSCRIBE_ROLE_DIARIZE);
    CHECK(!transcribe_model_supports(model, TRANSCRIBE_FEATURE_DIARIZATION));
    CHECK(transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION));
    transcribe_diarize_info info;
    transcribe_diarize_info_init(&info);
    CHECK(transcribe_diarize_get_info(model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.max_speakers == 4);
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_DIARIZE_RUN,
                                            TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE));

    transcribe_diarize_session * dia = nullptr;
    CHECK(transcribe_diarize_session_init(model, nullptr, &dia) == TRANSCRIBE_OK);

    // 2. Golden segments (ms, speaker) recorded on CPU, and ext == env preset.
    const std::vector<Row> golden_default = {
        { 320,   2400,  1 },
        { 7360,  9360,  1 },
        { 10240, 10640, 1 },
        { 4240,  6640,  2 },
        { 9760,  12000, 2 },
    };
    const std::vector<Row> golden_low_latency = {
        { 320,   2480,  1 },
        { 7360,  9360,  1 },
        { 10240, 10640, 1 },
        { 4160,  6640,  2 },
        { 9760,  12000, 2 },
    };
    CHECK(diarize(dia, pcm, TRANSCRIBE_SORTFORMER_PRESET_DEFAULT) == golden_default);
    CHECK(diarize(dia, pcm, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY) == golden_low_latency);
    ::setenv("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "low_latency", 1);
    CHECK(transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(diarize_rows(dia) == golden_low_latency);
    ::unsetenv("TRANSCRIBE_SORTFORMER_STREAM_PRESET");

    // 3. Pre-clear rejection of a bad preset keeps the previous result.
    const int                         n_before = transcribe_diarize_n_segments(dia);
    transcribe_sortformer_diarize_ext bad;
    transcribe_sortformer_diarize_ext_init(&bad);
    bad.preset = static_cast<transcribe_sortformer_preset>(99);
    transcribe_diarize_params dp;
    transcribe_diarize_params_init(&dp);
    dp.family = &bad.ext;
    CHECK(transcribe_diarize_run(dia, pcm.data(), static_cast<int>(pcm.size()), &dp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_diarize_n_segments(dia) == n_before);

    transcribe_diarize_session_free(dia);
    transcribe_model_free(model);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("sortformer_diarize_unit: ok\n");
    return EXIT_SUCCESS;
}
