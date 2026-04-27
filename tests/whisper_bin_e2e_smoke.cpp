// whisper_bin_e2e_smoke.cpp - end-to-end smoke for the legacy
// whisper.cpp `.bin` adapter, covering both multilingual and `.en`
// variants across the public ABI.
//
// Multilingual path (TRANSCRIBE_WHISPER_BIN_TINY_Q8_0):
//   - load + JFK transcribe via magic-byte dispatch path
//   - variant detected as "whisper-tiny" (cosmetic) with multilingual
//     capability flags + non-empty caps.languages list of 99 entries
//   - explicit --language en transcribes JFK correctly
//   - explicit --language de transcribes a German sample correctly
//     (lang_token_ids derived by ID arithmetic, not vocab lookup)
//   - text initial_prompt is accepted (tiktoken-style encoder works)
//   - hand-written prompt_tokens is accepted
//   - transcribe_tokenize() returns non-empty for representative text
//
// English-only path (TRANSCRIBE_WHISPER_BIN_TINY_EN):
//   - variant detected as "whisper-tiny.en"
//   - capabilities advertise lang_detect=false, translate=false
//   - caps.languages == ["en"] and caps.n_languages == 1
//   - JFK transcribes without language hint (auto-detect short-
//     circuits to "en" since the model has no language tokens)
//   - --language en accepted; --language de rejected with a non-OK
//     status
//   - translate task rejected (no <|translate|> token)
//
// Each subsection skips (RC 77) when its env var is unset; both
// subsections may be exercised independently.

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_SAMPLES_DIR
#  error "TRANSCRIBE_TEST_SAMPLES_DIR must be defined by the build system"
#endif

namespace {

int g_failures = 0;
int g_subtests_run = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const char * path) {
    struct stat st {};
    return ::stat(path, &st) == 0;
}

bool contains(const char * haystack, const char * needle) {
    if (haystack == nullptr) return false;
    return std::strstr(haystack, needle) != nullptr;
}

bool load_wav(const char * fname, std::vector<float> & out) {
    std::string err;
    return transcribe_cli::load_wav_mono_16k(
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + fname, out, err);
}

const char * env_or_null(const char * key) {
    const char * v = std::getenv(key);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

void test_multilingual(const char * model_path) {
    std::vector<float> jfk;
    std::vector<float> german;
    if (!load_wav("jfk.wav", jfk) || !load_wav("german.wav", german)) {
        std::fprintf(stderr, "FAIL: could not load sample wavs\n");
        ++g_failures;
        return;
    }

    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    transcribe_status st =
        transcribe_model_load_file(model_path, &mp, &model);
    CHECK(st == TRANSCRIBE_OK);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load (multi): %s\n",
                     transcribe_status_string(st));
        return;
    }

    // Identity / capabilities.
    CHECK(std::strcmp(transcribe_model_arch_string(model), "whisper") == 0);
    CHECK(contains(transcribe_model_variant_string(model), "whisper-tiny"));
    const transcribe_capabilities * caps = transcribe_model_capabilities(model);
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(caps->native_sample_rate == 16000);
        CHECK(caps->supports_language_detect);
        CHECK(caps->supports_translate);
        CHECK(caps->supports_long_form);
        CHECK(caps->supports_initial_prompt);
        // Multilingual whisper has 99 languages in canonical builds.
        CHECK(caps->n_languages == 99);
        CHECK(caps->languages != nullptr);
        if (caps->languages != nullptr && caps->n_languages > 0) {
            // First entry must be "en" (whisper language id 0).
            CHECK(std::strcmp(caps->languages[0], "en") == 0);
        }
    }

    // Tokenizer: encode is now available via tiktoken-style merge loop.
    {
        int32_t buf[64];
        const int n = transcribe_tokenize(model, " hello world", buf, 64);
        CHECK(n > 0);
    }

    transcribe_context_params cp = transcribe_context_default_params();
    transcribe_context * ctx = nullptr;
    st = transcribe_context_init(model, &cp, &ctx);
    CHECK(st == TRANSCRIBE_OK);
    if (st != TRANSCRIBE_OK || ctx == nullptr) {
        transcribe_model_free(model);
        return;
    }

    // English JFK with explicit hint.
    {
        transcribe_params rp = transcribe_default_params();
        rp.language = "en";
        st = transcribe_run(ctx, jfk.data(),
                            static_cast<int>(jfk.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        const char * text = transcribe_full_text(ctx);
        CHECK(text != nullptr);
        if (text) {
            CHECK(contains(text, "fellow Americans"));
            CHECK(contains(text, "country"));
        }
    }

    // German with explicit --language de — exercises the
    // arithmetic-derived lang_token_ids on the .bin path.
    {
        transcribe_params rp = transcribe_default_params();
        rp.language = "de";
        st = transcribe_run(ctx, german.data(),
                            static_cast<int>(german.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        const char * text = transcribe_full_text(ctx);
        CHECK(text != nullptr);
        if (text) {
            // German "Strand" or "Sonne" both appear in the canonical
            // transcript — assert at least one to keep the bar
            // robust against minor decode drift.
            CHECK(contains(text, "Strand") || contains(text, "Sonne"));
        }
    }

    // Special-token literals embedded in initial_prompt must be
    // rejected with INVALID_ARG, matching the GGUF path. The .bin
    // vocab does not store "<|en|>" / "<|notimestamps|>" / "<|0.00|>"
    // strings directly; the bin adapter synthesizes them into the
    // tokenizer's special-piece map so find() can resolve them. A
    // gap here would let users smuggle special bytes into the
    // decoder context.
    {
        const char * literals[] = {
            "transcribe <|en|> address",
            "use <|notimestamps|> please",
            "<|0.00|> beginning",
            "<|30.00|> end",
            "<|translate|> task",
        };
        for (const char * t : literals) {
            transcribe_params rp = transcribe_default_params();
            rp.language = "en";
            transcribe_whisper_params wp = transcribe_whisper_default_params();
            wp.initial_prompt = t;
            rp.whisper = &wp;
            st = transcribe_run(ctx, jfk.data(),
                                static_cast<int>(jfk.size()), &rp);
            CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
        }
    }

    // Text initial_prompt — now ACCEPTED (tiktoken encoder works).
    {
        transcribe_params rp = transcribe_default_params();
        rp.language = "en";
        transcribe_whisper_params wp = transcribe_whisper_default_params();
        wp.initial_prompt = "Inaugural address";
        rp.whisper = &wp;
        st = transcribe_run(ctx, jfk.data(),
                            static_cast<int>(jfk.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(contains(transcribe_full_text(ctx), "country"));
    }

    // Meaningful prompt biasing: product-names.wav contains awkward
    // dashed/camel-cased product names. With no prompt, whisper-tiny
    // produces "P3 Quattro" / "O3 Omnay" / "W3, RAP, Z" / etc. With
    // the glossary in initial_prompt, the model lands on the exact
    // canonical forms ("P3-Quattro", "O3-Omni", "W3-WrapZ", ...).
    //
    // This is the real proof that the tiktoken-style encoder is
    // producing the same token ids the model was trained against:
    // a wrong encoding would either land on rare-token noise or
    // fail to steer at all.
    {
        std::vector<float> product_pcm;
        if (!load_wav("product-names.wav", product_pcm)) {
            std::fprintf(stderr,
                         "FAIL: could not load product-names.wav\n");
            ++g_failures;
        } else {
            auto product_hits = [](const char * text) {
                const char * terms[] = {
                    "QuirkQuid",
                    "P3-Quattro",
                    "O3-Omni",
                    "B3-BondX",
                    "E3-Equity",
                    "W3-WrapZ",
                    "O2-Outlier",
                    "U3-UniFund",
                    "M3-Mover",
                };
                int n = 0;
                for (const char * term : terms) {
                    if (text != nullptr && std::strstr(text, term) != nullptr) {
                        ++n;
                    }
                }
                return n;
            };

            transcribe_params rp = transcribe_default_params();
            rp.language = "en";
            st = transcribe_run(ctx, product_pcm.data(),
                                static_cast<int>(product_pcm.size()), &rp);
            CHECK(st == TRANSCRIBE_OK);
            const int unprompted_hits =
                product_hits(transcribe_full_text(ctx));

            transcribe_whisper_params wp =
                transcribe_whisper_default_params();
            wp.initial_prompt =
                "QuirkQuid Quill Inc, P3-Quattro, O3-Omni, "
                "B3-BondX, E3-Equity, W3-WrapZ, O2-Outlier, "
                "U3-UniFund, M3-Mover";
            rp.whisper = &wp;
            st = transcribe_run(ctx, product_pcm.data(),
                                static_cast<int>(product_pcm.size()), &rp);
            CHECK(st == TRANSCRIBE_OK);
            const int prompted_hits =
                product_hits(transcribe_full_text(ctx));
            CHECK(prompted_hits >= 5);
            CHECK(prompted_hits > unprompted_hits + 3);
        }
    }

    // prompt_tokens (pre-encoded ids) — round-trip through
    // transcribe_tokenize.
    {
        int32_t tok_buf[32];
        const int n = transcribe_tokenize(model, " inaugural address",
                                          tok_buf, 32);
        CHECK(n > 0);
        if (n > 0) {
            transcribe_params rp = transcribe_default_params();
            rp.language = "en";
            transcribe_whisper_params wp = transcribe_whisper_default_params();
            wp.prompt_tokens   = tok_buf;
            wp.n_prompt_tokens = static_cast<size_t>(n);
            rp.whisper = &wp;
            st = transcribe_run(ctx, jfk.data(),
                                static_cast<int>(jfk.size()), &rp);
            CHECK(st == TRANSCRIBE_OK);
            CHECK(contains(transcribe_full_text(ctx), "country"));
        }
    }

    transcribe_context_free(ctx);
    transcribe_model_free(model);
    ++g_subtests_run;
}

void test_english_only(const char * model_path) {
    std::vector<float> jfk;
    if (!load_wav("jfk.wav", jfk)) {
        std::fprintf(stderr, "FAIL: could not load jfk.wav\n");
        ++g_failures;
        return;
    }

    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    transcribe_status st =
        transcribe_model_load_file(model_path, &mp, &model);
    CHECK(st == TRANSCRIBE_OK);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load (.en): %s\n",
                     transcribe_status_string(st));
        return;
    }

    CHECK(contains(transcribe_model_variant_string(model), "whisper-tiny.en"));
    const transcribe_capabilities * caps = transcribe_model_capabilities(model);
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(!caps->supports_language_detect);
        CHECK(!caps->supports_translate);
        CHECK(caps->n_languages == 1);
        CHECK(caps->languages != nullptr);
        if (caps->languages != nullptr && caps->n_languages == 1) {
            CHECK(std::strcmp(caps->languages[0], "en") == 0);
        }
    }

    transcribe_context_params cp = transcribe_context_default_params();
    transcribe_context * ctx = nullptr;
    st = transcribe_context_init(model, &cp, &ctx);
    CHECK(st == TRANSCRIBE_OK);
    if (st != TRANSCRIBE_OK || ctx == nullptr) {
        transcribe_model_free(model);
        return;
    }

    // No language hint — lang detection short-circuits to "en"
    // because the model only advertises one language.
    {
        transcribe_params rp = transcribe_default_params();
        st = transcribe_run(ctx, jfk.data(),
                            static_cast<int>(jfk.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(contains(transcribe_full_text(ctx), "country"));
    }

    // Explicit --language en accepted (no <|en|> token to resolve).
    {
        transcribe_params rp = transcribe_default_params();
        rp.language = "en";
        st = transcribe_run(ctx, jfk.data(),
                            static_cast<int>(jfk.size()), &rp);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(contains(transcribe_full_text(ctx), "country"));
    }

    // Non-English language — rejected with a non-OK status.
    {
        transcribe_params rp = transcribe_default_params();
        rp.language = "de";
        st = transcribe_run(ctx, jfk.data(),
                            static_cast<int>(jfk.size()), &rp);
        CHECK(st != TRANSCRIBE_OK);
    }

    // Translate task — rejected (no <|translate|> token).
    {
        transcribe_params rp = transcribe_default_params();
        rp.task     = TRANSCRIBE_TASK_TRANSLATE;
        rp.language = "en";
        st = transcribe_run(ctx, jfk.data(),
                            static_cast<int>(jfk.size()), &rp);
        CHECK(st != TRANSCRIBE_OK);
    }

    transcribe_context_free(ctx);
    transcribe_model_free(model);
    ++g_subtests_run;
}

} // namespace

int main() {
    const char * multi_path = env_or_null("TRANSCRIBE_WHISPER_BIN_TINY_Q8_0");
    const char * en_path    = env_or_null("TRANSCRIBE_WHISPER_BIN_TINY_EN");

    if ((multi_path == nullptr || !file_exists(multi_path)) &&
        (en_path    == nullptr || !file_exists(en_path)))
    {
        std::fprintf(stderr,
                     "SKIP: neither TRANSCRIBE_WHISPER_BIN_TINY_Q8_0 nor "
                     "TRANSCRIBE_WHISPER_BIN_TINY_EN is set\n");
        return 77;
    }

    if (multi_path != nullptr && file_exists(multi_path)) {
        test_multilingual(multi_path);
    }
    if (en_path != nullptr && file_exists(en_path)) {
        test_english_only(en_path);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d check(s) (%d subtest(s) ran)\n",
                     g_failures, g_subtests_run);
        return 1;
    }
    std::fprintf(stderr, "OK (%d subtest(s) ran)\n", g_subtests_run);
    return 0;
}
