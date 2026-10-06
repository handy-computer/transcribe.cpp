// ecapa_tdnn_real_smoke.cpp - the real VoxLingua107 ECAPA-TDNN GGUF through
// the public LANGID API: label table, aliases, top-1 on committed FLEURS
// clips, the crop on a long clip, and a short clip just over the minimum.
// Gated by TRANSCRIBE_ECAPA_TDNN_GGUF (RC 77 skip). CPU backend.

#include "transcribe.h"
#include "transcribe/langid.h"
#include "wav.h"

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

bool load_sample(const char * name, std::vector<float> & pcm) {
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name, pcm, err)) {
        std::fprintf(stderr, "FAIL: %s: %s\n", name, err.c_str());
        ++g_failures;
        return false;
    }
    return true;
}

// Top-1 code and p for one clip, or "" when the run fails.
std::string top1(transcribe_langid_session * s, const char * wav, float * p_out, int64_t * audio_ms = nullptr) {
    std::vector<float> pcm;
    if (!load_sample(wav, pcm)) {
        return "";
    }
    const transcribe_status st = transcribe_langid_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: %s: run: %s\n", wav, transcribe_status_string(st));
        ++g_failures;
        return "";
    }
    transcribe_langid_result r;
    transcribe_langid_result_init(&r);
    transcribe_langid_get_result(s, &r);
    if (audio_ms != nullptr) {
        *audio_ms = r.audio_ms;
    }
    transcribe_langid_candidate c;
    transcribe_langid_candidate_init(&c);
    transcribe_langid_get_candidate(s, 0, &c);
    *p_out = c.p;
    return c.code != nullptr ? c.code : "";
}

}  // namespace

int main() {
    const char * path = std::getenv("TRANSCRIBE_ECAPA_TDNN_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "ecapa_tdnn_real_smoke: TRANSCRIBE_ECAPA_TDNN_GGUF not set; skipping.\n");
        return 77;
    }
    transcribe_log_set(nullptr, nullptr);

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                  = TRANSCRIBE_BACKEND_CPU;
    transcribe_model *      m   = nullptr;
    const transcribe_status lst = transcribe_model_load_file(path, &mp, &m);
    if (lst != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: load %s: %s\n", path, transcribe_status_string(lst));
        return EXIT_FAILURE;
    }

    CHECK(transcribe_model_roles(m) == TRANSCRIBE_ROLE_LANGID);
    transcribe_langid_info info;
    transcribe_langid_info_init(&info);
    CHECK(transcribe_langid_get_info(m, &info) == TRANSCRIBE_OK);
    CHECK(info.n_labels == 107 && info.sample_rate == 16000 && info.min_audio_ms == 500);
    // Modern ISO codes alias the VoxLingua107 legacy labels.
    CHECK(transcribe_langid_label_index(m, "he") == transcribe_langid_label_index(m, "iw"));
    CHECK(transcribe_langid_label_index(m, "nb") == transcribe_langid_label_index(m, "no"));
    CHECK(transcribe_langid_label_index(m, "he") >= 0);
    const int en = transcribe_langid_label_index(m, "en");
    CHECK(en >= 0 && std::strcmp(transcribe_langid_label_code(m, en), "en") == 0);

    transcribe_langid_session_params sp;
    transcribe_langid_session_params_init(&sp);
    transcribe_langid_session * s = nullptr;
    CHECK(transcribe_langid_session_init(m, &sp, &s) == TRANSCRIBE_OK);

    const char * cases[][2] = {
        { "fleurs-en.wav", "en" },
        { "fleurs-de.wav", "de" },
        { "fleurs-fr.wav", "fr" },
        { "fleurs-es.wav", "es" },
        { "fleurs-ja.wav", "ja" },
        { "fleurs-zh.wav", "zh" },
        { "fleurs-ru.wav", "ru" },
        { "fleurs-id.wav", "id" },
    };
    for (const auto & c : cases) {
        float             p    = 0.0f;
        const std::string code = top1(s, c[0], &p);
        if (code != c[1] || !(p >= 0.5f)) {
            std::fprintf(stderr, "FAIL: %s: top-1 %s (p=%.3f), want %s\n", c[0], code.c_str(), p, c[1]);
            ++g_failures;
        }
    }

    // Long input is scored on its last 30 s; a clip just over the minimum
    // still runs.
    float   p        = 0.0f;
    int64_t audio_ms = 0;
    top1(s, "ru-long.wav", &p, &audio_ms);
    CHECK(audio_ms == 30000);
    top1(s, "short-800ms.wav", &p, &audio_ms);
    CHECK(audio_ms == 800 && std::isfinite(p));

    transcribe_langid_session_free(s);
    transcribe_model_free(m);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("ecapa_tdnn_real_smoke: ok\n");
    return EXIT_SUCCESS;
}
