// qwen3_asr_e2e_smoke.cpp - real-model gated end-to-end transcription
// test through the public C ABI.
//
// Loads a real Qwen3-ASR GGUF and runs two cases through transcribe_run:
//   - jfk.wav  (11s, single-chunk, no ragged tail) — the common path.
//   - dots.wav (35s, multi-chunk, ragged tail)     — the chunk-trim
//     path at src/arch/qwen3_asr/model.cpp:~789. Golden tensor
//     validation is jfk-only (graph-level enc.* dumps stay at the
//     padded T_enc while the reference is ragged), so this case is
//     the automated gate that the ragged-tail trim logic produces a
//     coherent transcript end-to-end.
//
// For each case we verify:
//   1. transcribe_run completes OK.
//   2. transcribe_full_text is non-empty and matches a checked-in
//      reference transcript within a small Levenshtein edit distance.
//      (Slightly looser than Cohere's 3 because we use bf16 LM on CPU
//      without the reference's flash attention — identical argmaxes at
//      prefill, but accumulated step-decoder drift at the 10–20 token
//      mark can change a single word occasionally. Dots tolerance is
//      proportionally larger to cover its 577-char length.)
//   3. Load + all runs complete under 180 seconds on CPU.
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake) controls whether this
//     binary is built.
//   - TRANSCRIBE_QWEN3_ASR_GGUF picks the GGUF at runtime. Unset → 77.
//   - TRANSCRIBE_TEST_AUDIO overrides the first-case audio path (for
//     ad-hoc single-file runs); default jfk.wav. Dots is always the
//     second case when the samples/dots.wav file is present.

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_SAMPLES_DIR
#  define TRANSCRIBE_TEST_SAMPLES_DIR "samples"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                  \
    do {                                                                    \
        const std::string _av = (a);                                        \
        const std::string _bv = (b);                                        \
        if (_av != _bv) {                                                   \
            std::fprintf(stderr,                                            \
                         "FAIL %s:%d: \"%s\" != \"%s\"\n",                  \
                         __FILE__, __LINE__,                                \
                         _av.c_str(), _bv.c_str());                         \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

int edit_distance(const std::string & a, const std::string & b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    std::vector<int> prev(static_cast<size_t>(m + 1));
    std::vector<int> curr(static_cast<size_t>(m + 1));
    for (int j = 0; j <= m; ++j) prev[static_cast<size_t>(j)] = j;
    for (int i = 1; i <= n; ++i) {
        curr[0] = i;
        for (int j = 1; j <= m; ++j) {
            const int cost = (a[static_cast<size_t>(i - 1)] ==
                              b[static_cast<size_t>(j - 1)]) ? 0 : 1;
            const int del = prev[static_cast<size_t>(j)]     + 1;
            const int ins = curr[static_cast<size_t>(j - 1)] + 1;
            const int sub = prev[static_cast<size_t>(j - 1)] + cost;
            int best = del < ins ? del : ins;
            if (sub < best) best = sub;
            curr[static_cast<size_t>(j)] = best;
        }
        prev.swap(curr);
    }
    return prev[static_cast<size_t>(m)];
}

// Reference transcript for samples/jfk.wav (Qwen3-ASR 0.6B greedy
// decode stripped of the "language English<asr_text>" prefix).
const char * const k_jfk_reference_text =
    "And so, my fellow Americans, ask not what your country can do for "
    "you; ask what you can do for your country.";

constexpr int k_jfk_max_edit_distance = 5;

// Reference transcript for samples/dots.wav (35.3s Steve Jobs
// commencement excerpt, multi-chunk ragged tail). This matches the
// current C++ CPU output; the Python reference differs by ~3 bytes
// (an em-dash where the reference shows ": "), which the tolerance
// below accommodates. Any larger drift indicates a regression in
// the ragged-tail trim, chunk boundary, or step-decoder paths.
const char * const k_dots_reference_text =
    "Of course, it was impossible to connect the dots looking forward "
    "when I was in college, but it was very, very clear looking "
    "backwards ten years later. Again, you can't connect the dots "
    "looking forward; you can only connect them looking backwards. "
    "So you have to trust that the dots will somehow connect in your "
    "future. You have to trust in something\xe2\x80\x94your gut, destiny, "
    "life, karma, whatever. Because believing that the dots will "
    "connect down the road, will give you the confidence to follow "
    "your heart, even when it leads you off the well-worn path, and "
    "that will make all the difference.";

constexpr int k_dots_max_edit_distance = 10;

// Runs one (audio, reference) case against `model`: creates a fresh
// context, loads the wav, runs transcribe, and checks that the
// resulting transcript is within `max_edit_distance` of `reference`.
// Increments g_failures on any failure; the caller is responsible for
// aborting early if the context fails to init.
void run_case(transcribe_model *  model,
              const char *        case_name,
              const std::string & wav_path,
              const char *        reference,
              int                 max_edit_distance)
{
    if (!file_exists(wav_path)) {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke[%s]: wav not found: %s\n",
                     case_name, wav_path.c_str());
        ++g_failures;
        return;
    }

    std::vector<float> pcm;
    std::string load_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, load_err)) {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke[%s]: wav load: %s\n",
                     case_name, load_err.c_str());
        ++g_failures;
        return;
    }

    transcribe_context_params cp = transcribe_context_default_params();
    struct transcribe_context * ctx = nullptr;
    {
        const transcribe_status st =
            transcribe_context_init(model, &cp, &ctx);
        if (st != TRANSCRIBE_OK || ctx == nullptr) {
            std::fprintf(stderr,
                         "qwen3_asr_e2e_smoke[%s]: FAIL ctx_init: %s\n",
                         case_name, transcribe_status_string(st));
            ++g_failures;
            return;
        }
    }

    transcribe_params rp = transcribe_default_params();
    {
        const transcribe_status st =
            transcribe_run(ctx, pcm.data(),
                           static_cast<int>(pcm.size()), &rp);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr,
                         "qwen3_asr_e2e_smoke[%s]: FAIL run: %s\n",
                         case_name, transcribe_status_string(st));
            ++g_failures;
            transcribe_context_free(ctx);
            return;
        }
    }

    const char * full = transcribe_full_text(ctx);
    const std::string actual = full ? full : "";
    std::fprintf(stderr,
                 "qwen3_asr_e2e_smoke[%s]: text=\"%s\"\n",
                 case_name, actual.c_str());
    if (actual.empty()) {
        std::fprintf(stderr,
                     "FAIL[%s]: transcript is empty\n", case_name);
        ++g_failures;
        transcribe_context_free(ctx);
        return;
    }

    const int dist = edit_distance(actual, reference);
    std::fprintf(stderr,
                 "qwen3_asr_e2e_smoke[%s]: edit_distance=%d "
                 "(tolerance=%d)\n",
                 case_name, dist, max_edit_distance);
    if (dist > max_edit_distance) {
        std::fprintf(stderr,
                     "FAIL[%s]: text edit distance %d exceeds %d\n",
                     case_name, dist, max_edit_distance);
        std::fprintf(stderr, "  reference: %s\n", reference);
        std::fprintf(stderr, "  actual:    %s\n", actual.c_str());
        ++g_failures;
    }

    transcribe_context_free(ctx);
}

} // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_QWEN3_ASR_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke: TRANSCRIBE_QWEN3_ASR_GGUF not set; "
                     "skipping.\n");
        return 77;
    }
    const std::string model_path = env;
    if (!file_exists(model_path)) {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke: model not found: %s\n",
                     model_path.c_str());
        return 77;
    }

    std::string jfk_wav;
    if (const char * w = std::getenv("TRANSCRIBE_TEST_AUDIO"); w && w[0]) {
        jfk_wav = w;
    } else {
        jfk_wav = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    }
    const std::string dots_wav =
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/dots.wav";

    const auto t_start = std::chrono::steady_clock::now();

    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    {
        const transcribe_status st =
            transcribe_model_load_file(model_path.c_str(), &mp, &model);
        if (st != TRANSCRIBE_OK || model == nullptr) {
            std::fprintf(stderr, "FAIL load: %s\n",
                         transcribe_status_string(st));
            return EXIT_FAILURE;
        }
    }

    CHECK_STR_EQ(transcribe_model_arch_string(model), "qwen3_asr");

    // Language hinting contract:
    //   - A BCP-47 code advertised in caps.languages must be accepted
    //     and produce a transcript. For English audio we expect the
    //     forced-"en" transcript to be within edit-distance tolerance
    //     of the auto-detect transcript (same tolerance as the main
    //     jfk case below).
    //   - A code not in caps.languages must be rejected by the
    //     dispatcher with UNSUPPORTED_LANGUAGE before we ever reach
    //     the family handler. We use a made-up code ("xx-fake") to
    //     exercise that path.
    //
    // Exercised before the full test cases so a regression here
    // can't mask a later bug.
    {
        std::vector<float> pcm;
        std::string load_err;
        if (!file_exists(jfk_wav) ||
            !transcribe_cli::load_wav_mono_16k(jfk_wav, pcm, load_err))
        {
            std::fprintf(stderr,
                         "qwen3_asr_e2e_smoke: jfk wav load for language "
                         "hint check: %s\n",
                         load_err.empty() ? "missing" : load_err.c_str());
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        transcribe_context_params cp = transcribe_context_default_params();
        struct transcribe_context * ctx = nullptr;
        if (transcribe_context_init(model, &cp, &ctx) != TRANSCRIBE_OK ||
            ctx == nullptr)
        {
            std::fprintf(stderr,
                         "qwen3_asr_e2e_smoke: ctx_init for language "
                         "hint check failed\n");
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        // Accept path: force "en" on English audio and expect a
        // transcript within the same edit-distance tolerance as the
        // auto-detect jfk case.
        transcribe_params rp_lang = transcribe_default_params();
        rp_lang.language = "en";
        const transcribe_status st =
            transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()),
                           &rp_lang);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr,
                         "FAIL: language=\"en\" returned %s (expected OK)\n",
                         transcribe_status_string(st));
            ++g_failures;
        } else {
            const char * forced = transcribe_full_text(ctx);
            const std::string forced_text = forced ? forced : "";
            const int dist = edit_distance(forced_text, k_jfk_reference_text);
            std::fprintf(stderr,
                         "qwen3_asr_e2e_smoke[lang=en]: text=\"%s\" "
                         "edit_distance=%d\n",
                         forced_text.c_str(), dist);
            if (dist > k_jfk_max_edit_distance) {
                std::fprintf(stderr,
                             "FAIL: forced-en edit distance %d > %d\n",
                             dist, k_jfk_max_edit_distance);
                ++g_failures;
            }
        }

        // Reject path: a BCP-47 code that is NOT in caps.languages
        // must be rejected with UNSUPPORTED_LANGUAGE by the
        // dispatcher. Use an unambiguously bogus code.
        rp_lang.language = "xx-fake";
        const transcribe_status st_bad =
            transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()),
                           &rp_lang);
        if (st_bad != TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE) {
            std::fprintf(stderr,
                         "FAIL: expected UNSUPPORTED_LANGUAGE for "
                         "language=\"xx-fake\", got %s\n",
                         transcribe_status_string(st_bad));
            ++g_failures;
        }

        transcribe_context_free(ctx);
    }

    // Case 1: jfk.wav (single-chunk).
    run_case(model, "jfk", jfk_wav,
             k_jfk_reference_text, k_jfk_max_edit_distance);

    // Case 2: dots.wav (multi-chunk ragged tail). This is the
    // automated gate for the ragged-tail trim path — golden tensor
    // validation is jfk-only because graph-level enc.* dumps stay at
    // T_enc_padded while the reference is ragged. Skip (not fail)
    // when the sample file is absent so ad-hoc runs that only care
    // about jfk still pass.
    if (file_exists(dots_wav)) {
        run_case(model, "dots", dots_wav,
                 k_dots_reference_text, k_dots_max_edit_distance);
    } else {
        std::fprintf(stderr,
                     "qwen3_asr_e2e_smoke[dots]: %s not present; "
                     "skipping multi-chunk case.\n", dots_wav.c_str());
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::fprintf(stderr, "qwen3_asr_e2e_smoke: wall=%.2f ms\n", wall_ms);
    if (wall_ms > 180000.0) {
        std::fprintf(stderr,
                     "FAIL: wall time %.2f ms exceeds 180000 ms\n", wall_ms);
        ++g_failures;
    }

    transcribe_model_free(model);

    if (g_failures > 0) return EXIT_FAILURE;
    std::fprintf(stdout, "qwen3_asr_e2e_smoke: ok\n");
    return EXIT_SUCCESS;
}
