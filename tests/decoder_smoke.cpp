// decoder_smoke.cpp - real-model gated end-to-end decoder accuracy test.
//
// Phase 5 + phase 6 acceptance gate. Loads a real Parakeet v2 GGUF,
// runs the full pipeline (load → mel → encoder → predictor + joint
// + TDT decode → result accessor population) on samples/jfk.wav, and
// asserts:
//
//   1. transcribe_run completes OK on the canonical sample.
//   2. transcribe_full_text matches the parakeet-mlx reference text
//      to within edit distance 3 (PLAN.md's WER acceptance budget).
//   3. transcribe_n_tokens > 0 and the per-token accessors return
//      non-sentinel values: ids in range, text non-empty, t0_ms ≤
//      t1_ms, p in [0, 1].
//   4. transcribe_n_words > 0 and at least one word has the leading
//      space stripped (sanity check on the SentencePiece word
//      boundary handling in the result builder).
//   5. transcribe_n_segments == 1 (v1 produces a single segment per
//      run).
//   6. transcribe_returned_timestamp_kind == TRANSCRIBE_TIMESTAMPS_TOKEN
//      (Parakeet TDT produces token-level timestamps from encoder
//      frame indices).
//   7. transcribe_get_timings reports non-zero mel + encode + decode.
//
// The reference text comes from running parakeet-mlx (greedy decode)
// against the same v2 mlx model directory; the constant below is the
// verbatim output. If parakeet-mlx ever changes its decode behavior
// the constant will need updating, but the WER tolerance gives the
// test 3 character-edits of slack so minor upstream tokenizer
// tweaks won't break the gate immediately.
//
// Gating: same two-knob pattern as parakeet_real_smoke /
// encoder_smoke. Built only when TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON;
// at run time TRANSCRIBE_REAL_PARAKEET_GGUF must point at a v2 GGUF.
// The test exits 77 (CTest "skipped") when either knob is unset.
//
// The golden was generated against v2; v3's encoder weights are
// different and the resulting text would differ, so the test hard-
// fails on v3 rather than silently mis-validating.

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

#define CHECK_EQ_INT(actual, expected)                                      \
    do {                                                                    \
        const long long _a = static_cast<long long>(actual);                \
        const long long _e = static_cast<long long>(expected);              \
        if (_a != _e) {                                                     \
            std::fprintf(stderr,                                            \
                         "FAIL %s:%d: %s = %lld, expected %lld\n",          \
                         __FILE__, __LINE__, #actual, _a, _e);              \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

// Levenshtein edit distance between two strings. Used for the WER-ish
// gate ("edit distance ≤ 3" per PLAN.md). Implementation is the
// standard rolling-row DP; quadratic in time, linear in memory.
// jfk.wav's reference text is ~110 characters so this is trivially
// fast at the test gate.
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

// SentencePiece word-boundary marker U+2581 in UTF-8.
constexpr const char k_sp_marker[] = "\xE2\x96\x81";

// Reference text from parakeet-mlx greedy decode against the v2
// model on samples/jfk.wav. Captured at phase 5 bring-up; matches
// our C++ output exactly under greedy decode.
const char * const k_jfk_reference_text =
    "And so, my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country.";

constexpr int k_max_edit_distance = 3;

} // namespace

int main() {
    // ---- Resolve env -----------------------------------------------
    const char * gguf_env = std::getenv("TRANSCRIBE_REAL_PARAKEET_GGUF");
    if (gguf_env == nullptr || gguf_env[0] == '\0') {
        std::fprintf(stderr,
                     "decoder_smoke: TRANSCRIBE_REAL_PARAKEET_GGUF not set; "
                     "skipping\n");
        return 77;
    }
    if (!file_exists(gguf_env)) {
        std::fprintf(stderr,
                     "decoder_smoke: gguf file not found: %s\n", gguf_env);
        return 77;
    }

    const std::string wav_path =
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    if (!file_exists(wav_path)) {
        std::fprintf(stderr, "decoder_smoke: wav not found: %s\n",
                     wav_path.c_str());
        return EXIT_FAILURE;
    }

    // ---- Load model ------------------------------------------------
    transcribe_model_params mp = transcribe_model_default_params();
    struct transcribe_model * model = nullptr;
    {
        const transcribe_status st =
            transcribe_model_load_file(gguf_env, &mp, &model);
        if (st != TRANSCRIBE_OK || model == nullptr) {
            std::fprintf(stderr, "decoder_smoke: load failed: %s\n",
                         transcribe_status_string(st));
            return EXIT_FAILURE;
        }
    }

    // The reference golden was generated against v2; v3 has different
    // weights and produces different text. Fail loudly rather than
    // silently mis-validating.
    {
        const std::string variant = transcribe_model_variant_string(model);
        if (variant != "tdt-0.6b-v2") {
            std::fprintf(stderr,
                         "decoder_smoke: golden is for tdt-0.6b-v2 only, "
                         "got \"%s\"\n", variant.c_str());
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    const std::string backend = transcribe_model_backend(model);
    std::fprintf(stdout, "decoder_smoke: backend=%s\n", backend.c_str());

    // ---- Load wav --------------------------------------------------
    std::vector<float> pcm;
    std::string load_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, load_err)) {
        std::fprintf(stderr, "decoder_smoke: wav load: %s\n",
                     load_err.c_str());
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // ---- Init context + run ----------------------------------------
    transcribe_context_params cp = transcribe_context_default_params();
    struct transcribe_context * ctx = nullptr;
    {
        const transcribe_status st =
            transcribe_context_init(model, &cp, &ctx);
        if (st != TRANSCRIBE_OK || ctx == nullptr) {
            std::fprintf(stderr, "decoder_smoke: ctx init: %s\n",
                         transcribe_status_string(st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    // Per the public contract, accessors are safe to call before
    // transcribe_run and return safe sentinels. Belt-and-braces
    // check this so a regression that pre-populates ghost results
    // before the first run is caught.
    CHECK_EQ_INT(transcribe_n_segments(ctx), 0);
    CHECK_EQ_INT(transcribe_n_words(ctx),    0);
    CHECK_EQ_INT(transcribe_n_tokens(ctx),   0);
    CHECK(std::strcmp(transcribe_full_text(ctx), "") == 0);
    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_NONE);

    transcribe_params rp = transcribe_default_params();
    {
        const transcribe_status st =
            transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "decoder_smoke: run: %s\n",
                         transcribe_status_string(st));
            transcribe_context_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    // ---- Top-level result ------------------------------------------
    const char * full = transcribe_full_text(ctx);
    CHECK(full != nullptr);
    const std::string actual = full ? full : "";
    std::fprintf(stdout, "decoder_smoke: text=\"%s\"\n", actual.c_str());

    const int dist = edit_distance(actual, k_jfk_reference_text);
    std::fprintf(stdout,
                 "decoder_smoke: edit_distance=%d (tolerance=%d)\n",
                 dist, k_max_edit_distance);
    if (dist > k_max_edit_distance) {
        std::fprintf(stderr,
                     "FAIL: text edit distance %d exceeds %d\n",
                     dist, k_max_edit_distance);
        std::fprintf(stderr, "  reference: %s\n", k_jfk_reference_text);
        std::fprintf(stderr, "  actual:    %s\n", actual.c_str());
        ++g_failures;
    }

    CHECK(transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_TOKEN);

    // ---- Per-token sanity ------------------------------------------
    const int n_tokens = transcribe_n_tokens(ctx);
    CHECK(n_tokens > 0);

    int   prev_t0  = -1;
    int   n_with_p = 0;
    for (int i = 0; i < n_tokens; ++i) {
        const int    id    = transcribe_token_id(ctx, i);
        const char * text  = transcribe_token_text(ctx, i);
        const float  p     = transcribe_token_p(ctx, i);
        const int64_t t0   = transcribe_token_t0_ms(ctx, i);
        const int64_t t1   = transcribe_token_t1_ms(ctx, i);
        const int    sidx  = transcribe_token_seg_index(ctx, i);
        const int    widx  = transcribe_token_word_index(ctx, i);

        CHECK(id >= 0);
        CHECK(text != nullptr);
        CHECK(t0 >= 0);
        CHECK(t1 >= t0);
        CHECK(t0 >= prev_t0);  // monotone
        CHECK_EQ_INT(sidx, 0);
        CHECK(widx >= 0);
        CHECK(p >= 0.0f && p <= 1.0001f && !std::isnan(p));
        if (p > 0.0f) ++n_with_p;
        prev_t0 = static_cast<int>(t0);
    }
    // At least most tokens should carry a positive confidence (the
    // entropy-based formula gives ~0 for a uniform distribution and
    // ~1 for a confident decode; on jfk.wav nearly every token is
    // emitted with very high confidence).
    CHECK(n_with_p > n_tokens / 2);

    // ---- Per-word sanity -------------------------------------------
    const int n_words = transcribe_n_words(ctx);
    CHECK(n_words > 0);
    // The reference text has 22 word-tokens; the result builder
    // splits on the SentencePiece marker. We don't pin the exact
    // count because punctuation tokens get attached to whichever
    // word they follow, but a 11-second jfk should produce on the
    // order of ~20 words.
    CHECK(n_words >= 15 && n_words <= 30);

    int last_word_t1 = -1;
    int n_words_with_text = 0;
    for (int i = 0; i < n_words; ++i) {
        const char *  text = transcribe_word_text(ctx, i);
        const int64_t t0   = transcribe_word_t0_ms(ctx, i);
        const int64_t t1   = transcribe_word_t1_ms(ctx, i);
        const int     sidx = transcribe_word_seg_index(ctx, i);
        const int     ft   = transcribe_word_first_token(ctx, i);
        const int     nt   = transcribe_word_n_tokens(ctx, i);

        CHECK(text != nullptr);
        if (text != nullptr && text[0] != '\0') ++n_words_with_text;
        CHECK_EQ_INT(sidx, 0);
        CHECK(ft >= 0 && ft < n_tokens);
        CHECK(nt > 0 && ft + nt <= n_tokens);
        CHECK(t0 >= 0);
        CHECK(t1 >= t0);
        CHECK(t0 >= last_word_t1 - 200); // allow some inter-word slop
        // No leading space on a word — the result builder strips it.
        if (text != nullptr && text[0] == ' ') {
            std::fprintf(stderr,
                         "FAIL: word %d has leading space: \"%s\"\n",
                         i, text);
            ++g_failures;
        }
        // No SentencePiece marker should leak into a word's text.
        if (text != nullptr &&
            std::strstr(text, k_sp_marker) != nullptr)
        {
            std::fprintf(stderr,
                         "FAIL: word %d contains raw SP marker: \"%s\"\n",
                         i, text);
            ++g_failures;
        }
        last_word_t1 = static_cast<int>(t1);
    }
    CHECK(n_words_with_text == n_words);

    // ---- Per-segment sanity ----------------------------------------
    const int n_segments = transcribe_n_segments(ctx);
    CHECK_EQ_INT(n_segments, 1);

    const char *  seg_text = transcribe_segment_text(ctx, 0);
    const int64_t seg_t0   = transcribe_segment_t0_ms(ctx, 0);
    const int64_t seg_t1   = transcribe_segment_t1_ms(ctx, 0);
    const int     seg_fw   = transcribe_segment_first_word(ctx, 0);
    const int     seg_nw   = transcribe_segment_n_words(ctx, 0);
    const int     seg_ft   = transcribe_segment_first_token(ctx, 0);
    const int     seg_nt   = transcribe_segment_n_tokens(ctx, 0);

    CHECK(seg_text != nullptr);
    CHECK(seg_text != nullptr && seg_text[0] != '\0');
    CHECK(seg_t0 >= 0);
    CHECK(seg_t1 > seg_t0);
    CHECK_EQ_INT(seg_fw, 0);
    CHECK_EQ_INT(seg_nw, n_words);
    CHECK_EQ_INT(seg_ft, 0);
    CHECK_EQ_INT(seg_nt, n_tokens);
    // Segment text should equal full_text.
    CHECK(seg_text != nullptr && std::strcmp(seg_text, full) == 0);
    // Segment time should span the audio.
    CHECK(seg_t1 <= 12000); // jfk.wav is ~11 s; allow 1 s slack

    // ---- Out-of-bounds accessors return safe sentinels --------------
    CHECK(transcribe_token_id(ctx, n_tokens)    == 0);
    CHECK(transcribe_token_id(ctx, -1)          == 0);
    CHECK(std::strcmp(transcribe_token_text(ctx, n_tokens), "") == 0);
    CHECK(std::isnan(transcribe_token_p(ctx, n_tokens)));
    CHECK(transcribe_word_n_tokens(ctx, n_words) == 0);
    CHECK(std::strcmp(transcribe_segment_text(ctx, 5), "") == 0);

    // ---- Timings ---------------------------------------------------
    const transcribe_timings t = transcribe_get_timings(ctx);
    std::fprintf(stdout,
                 "decoder_smoke: timings load=%.2f mel=%.2f encode=%.2f decode=%.2f\n",
                 t.load_ms, t.mel_ms, t.encode_ms, t.decode_ms);
    CHECK(t.load_ms   > 0.0f);
    CHECK(t.mel_ms    > 0.0f);
    CHECK(t.encode_ms > 0.0f);
    CHECK(t.decode_ms > 0.0f);

    // ---- Teardown --------------------------------------------------
    transcribe_context_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "decoder_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "decoder_smoke: ok\n");
    return EXIT_SUCCESS;
}
