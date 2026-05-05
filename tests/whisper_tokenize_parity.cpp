// whisper_tokenize_parity.cpp - real-model gated test that
// transcribe_tokenize() produces HF-identical token sequences on
// Whisper's GPT-2 byte-level BPE, including the digit / contraction
// cases that forced the GPT-2 pretokenizer branch in Stage 1.
//
// The expected token ids were computed with HuggingFace's
// openai/whisper-tiny tokenizer, add_special_tokens=False. That
// tokenizer is vocab-compatible across whisper-tiny / base / small
// / medium / large-* multilingual variants, so the same expected
// sequences hold for any multilingual tiny-family GGUF. The .en
// variants carry a 50257-token vocab (one fewer language token)
// but the text-only BPE pieces tested here sit in the shared prefix
// of both vocabs, so the expected ids match there too.
//
// Gated by TRANSCRIBE_WHISPER_MODEL (same env var used by
// whisper_e2e_smoke). Exits 77 (cmake SKIP_RETURN_CODE) when unset.

#include "transcribe.h"

#include <sys/stat.h>

#include <climits>
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

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

struct Case {
    const char *         text;
    std::vector<int32_t> expected;
};

void check_case(const struct transcribe_model * model, const Case & c) {
    int32_t tokens[64];
    const int n = transcribe_tokenize(model, c.text, tokens, 64);
    if (n < 0 || static_cast<size_t>(n) != c.expected.size()) {
        std::fprintf(stderr,
                     "FAIL tokenize(%s): n=%d, expected size %zu\n",
                     c.text, n, c.expected.size());
        ++g_failures;
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (tokens[i] != c.expected[i]) {
            std::fprintf(stderr,
                         "FAIL tokenize(%s): tokens[%d]=%d, expected %d\n",
                         c.text, i, tokens[i], c.expected[i]);
            ++g_failures;
            return;
        }
    }
}

} // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_WHISPER_MODEL");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "whisper_tokenize_parity: TRANSCRIBE_WHISPER_MODEL "
                     "not set; skipping.\n");
        return 77;
    }
    const std::string model_path = env;
    if (!file_exists(model_path)) {
        std::fprintf(stderr,
                     "whisper_tokenize_parity: model not found: %s\n",
                     model_path.c_str());
        return 77;
    }

    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    const transcribe_status st = transcribe_model_load_file(
        model_path.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    // Parity cases. Expected sequences copied from the HF reference
    // (openai/whisper-tiny, add_special_tokens=False). The digit cases
    // are the load-bearing ones — they exercise the GPT-2 `\p{N}+`
    // path that the Qwen2 pretok (single-digit) would get wrong.
    const std::vector<Case> cases = {
        { "hello 123 world",    { 675, 1913, 34466, 1002 } },
        { "The year is 2024.",  { 2278, 1064, 307, 45237, 13 } },
        { "cost: $99.99",       { 27718, 25, 1848, 8494, 13, 8494 } },
        { "  multi  spaces",    { 220, 4825, 220, 7673 } },
        { "I'm going",          { 40, 478, 516 } },
        { "she isn't here",     { 9611, 1943, 380, 510 } },
        { "hello world",        { 675, 1913, 1002 } },
        { "hello  world",       { 675, 1913, 220, 1002 } },
        { "   triple   space",  { 220, 220, 15508, 220, 220, 1901 } },
        { "42",                 { 15628 } },
        { "",                   {} },
    };
    for (const Case & c : cases) {
        check_case(model, c);
    }

    // Buffer-too-small contract: n_max smaller than needed must
    // return negative-of-N (mirrors whisper.cpp), leaving the caller
    // free to reallocate.
    {
        const char * text = "hello 123 world"; // needs 4 tokens
        int32_t tokens[2];
        const int n = transcribe_tokenize(model, text, tokens, 2);
        CHECK(n == -4);
    }

    // Empty text with any n_max returns 0 (success, nothing written).
    {
        int32_t tokens[1];
        const int n = transcribe_tokenize(model, "", tokens, 1);
        CHECK(n == 0);
    }

    // NULL tokens with zero-length input is fine (nothing to write).
    {
        const int n = transcribe_tokenize(model, "", nullptr, 0);
        CHECK(n == 0);
    }

    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr,
                     "whisper_tokenize_parity: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "whisper_tokenize_parity: ok\n");
    return EXIT_SUCCESS;
}
