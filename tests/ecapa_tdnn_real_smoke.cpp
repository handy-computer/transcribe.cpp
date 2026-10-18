// ecapa_tdnn_real_smoke.cpp - the real VoxLingua107 ECAPA-TDNN GGUF through
// the public LANGID API: label table, aliases, top-1 on committed FLEURS
// clips, the crop to the first 30 s of a long clip, and a clip cut to 800 ms,
// all on the CPU
// backend; then, when a GPU backend loads, the stock-op GPU graph against the
// CPU graph's logits. Gated by TRANSCRIBE_ECAPA_TDNN_GGUF (RC 77 skip).

#include "gguf.h"
#include "transcribe.h"
#include "transcribe/langid.h"
#include "wav.h"

#include <cmath>
#include <cstddef>
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
std::string top1(transcribe_langid_session * s, const char * wav, float * p_out) {
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
    transcribe_langid_candidate c;
    transcribe_langid_candidate_init(&c);
    transcribe_langid_get_candidate(s, 0, &c);
    *p_out = c.p;
    return c.code != nullptr ? c.code : "";
}

// Every label's logit for PCM, by label index; empty when the run fails.
std::vector<float> all_logits(transcribe_langid_session * s, const std::vector<float> & pcm) {
    if (pcm.empty() || transcribe_langid_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) != TRANSCRIBE_OK) {
        return {};
    }
    transcribe_langid_result r;
    transcribe_langid_result_init(&r);
    transcribe_langid_get_result(s, &r);
    std::vector<float> out(static_cast<size_t>(r.n_candidates), 0.0f);
    for (int i = 0; i < r.n_candidates; ++i) {
        transcribe_langid_candidate c;
        transcribe_langid_candidate_init(&c);
        transcribe_langid_get_candidate(s, i, &c);
        if (c.index >= 0 && c.index < r.n_candidates) {
            out[static_cast<size_t>(c.index)] = c.logit;
        }
    }
    return out;
}

std::vector<float> all_logits(transcribe_langid_session * s, const char * wav) {
    std::vector<float> pcm;
    return load_sample(wav, pcm) ? all_logits(s, pcm) : std::vector<float>{};
}

// general.file_type of the GGUF (0 = all F32, 1 = F16, 7 = Q8_0), or -1.
int gguf_file_type(const char * path) {
    gguf_init_params gp{};
    gp.no_alloc      = true;
    gp.ctx           = nullptr;
    gguf_context * g = gguf_init_from_file(path, gp);
    if (g == nullptr) {
        return -1;
    }
    const int64_t key = gguf_find_key(g, "general.file_type");
    const int     ft =
        key >= 0 && gguf_get_kv_type(g, key) == GGUF_TYPE_UINT32 ? static_cast<int>(gguf_get_val_u32(g, key)) : -1;
    gguf_free(g);
    return ft;
}

// The GPU backends run the stock-op graph, which no other test reaches.
// Their matmuls are not bit-exact F32 (Vulkan on an AMD iGPU lands ~1e-2 off
// the CPU logits, Metal ~7e-3), so this bounds the drift rather than
// demanding parity. F16 / Q8_0 files drift further because the CPU and GPU
// round F16 operands differently (Metal: up to 0.057 on these clips), so they
// get twice that.
void check_gpu_matches_cpu(const char * path, transcribe_model * cpu_model) {
    const float                      bound  = gguf_file_type(path) == 0 ? 0.05f : 0.1f;
    const transcribe_backend_request gpus[] = { TRANSCRIBE_BACKEND_VULKAN, TRANSCRIBE_BACKEND_METAL };
    transcribe_model *               gm     = nullptr;
    for (const transcribe_backend_request b : gpus) {
        transcribe_model_load_params mp;
        transcribe_model_load_params_init(&mp);
        mp.backend = b;
        if (transcribe_model_load_file(path, &mp, &gm) == TRANSCRIBE_OK) {
            break;
        }
        gm = nullptr;
    }
    if (gm == nullptr) {
        std::fprintf(stderr, "ecapa_tdnn_real_smoke: no GPU backend; skipping the GPU graph check.\n");
        return;
    }

    transcribe_langid_session * cs = nullptr;
    transcribe_langid_session * gs = nullptr;
    CHECK(transcribe_langid_session_init(cpu_model, nullptr, &cs) == TRANSCRIBE_OK);
    CHECK(transcribe_langid_session_init(gm, nullptr, &gs) == TRANSCRIBE_OK);
    for (const char * wav : { "fleurs-en.wav", "fleurs-zh.wav", "ru-long.wav" }) {
        const std::vector<float> a = all_logits(cs, wav);
        const std::vector<float> b = all_logits(gs, wav);
        if (a.empty() || a.size() != b.size()) {
            std::fprintf(stderr, "FAIL: %s: GPU run (%zu logits) vs CPU (%zu)\n", wav, b.size(), a.size());
            ++g_failures;
            continue;
        }
        float max_diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            max_diff = std::fmax(max_diff, std::fabs(a[i] - b[i]));
        }
        std::fprintf(stderr, "ecapa_tdnn_real_smoke: %s: GPU vs CPU max |logit diff| %.4f (bound %.2f)\n", wav,
                     max_diff, bound);
        if (!(max_diff < bound)) {
            std::fprintf(stderr, "FAIL: %s: GPU logits differ from CPU by %.4f\n", wav, max_diff);
            ++g_failures;
        }
    }
    transcribe_langid_session_free(gs);
    transcribe_langid_session_free(cs);
    transcribe_model_free(gm);
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
    CHECK(info.n_labels == 107 && info.sample_rate == 16000 && info.min_audio_ms == 500 && info.max_audio_ms == 30000);
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

    // Long input (ru-long.wav, ~33.8 s) is scored on its first max_audio_ms:
    // the whole clip and its first-30 s slice give identical logits, the
    // last-30 s slice does not.
    std::vector<float> pcm;
    const size_t       max_n = static_cast<size_t>(info.max_audio_ms) * static_cast<size_t>(info.sample_rate) / 1000;
    if (load_sample("ru-long.wav", pcm)) {
        CHECK(pcm.size() > max_n);
        if (pcm.size() > max_n) {
            const std::vector<float> last(pcm.end() - static_cast<std::ptrdiff_t>(max_n), pcm.end());
            const std::vector<float> first(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(max_n));
            const std::vector<float> l_full  = all_logits(s, pcm);
            const std::vector<float> l_last  = all_logits(s, last);
            const std::vector<float> l_first = all_logits(s, first);
            CHECK(l_full.size() == 107 && l_full == l_first);
            CHECK(l_last.size() == 107 && l_last != l_full);
        }
    }
    // A clip just over the minimum still runs.
    if (load_sample("fleurs-en.wav", pcm)) {
        pcm.resize(12800);  // 800 ms
        CHECK(transcribe_langid_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
        transcribe_langid_result r;
        transcribe_langid_result_init(&r);
        transcribe_langid_get_result(s, &r);
        transcribe_langid_candidate c;
        transcribe_langid_candidate_init(&c);
        transcribe_langid_get_candidate(s, 0, &c);
        CHECK(r.n_candidates == 107 && std::isfinite(c.p));
    }

    transcribe_langid_session_free(s);

    check_gpu_matches_cpu(path, m);
    transcribe_model_free(m);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("ecapa_tdnn_real_smoke: ok\n");
    return EXIT_SUCCESS;
}
