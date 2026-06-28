// whisper_bin_tokenize_parity.cpp - tokenizer parity between the
// legacy whisper.cpp `.bin` adapter (tiktoken-style raw-bytes encoder)
// and the canonical GGUF whisper-tiny tokenizer (HF GPT-2 byte-level
// BPE with merge ranks).
//
// Both surfaces are reachable through the public transcribe_tokenize
// ABI. We load each model, encode the same probe strings, and assert
// the resulting token id sequences are identical. If they diverge,
// something is wrong with the tiktoken encoder, the byte-unicode
// inversion, or one of the rename / load steps that fed the .bin
// vocab.
//
// Probe set covers:
//   - simple ASCII words (matches GPT-2 base merges)
//   - leading-space variants (whisper's most common pattern)
//   - digits (whitespace-merged digit runs)
//   - English contractions ('s 't 're 've 'm 'll 'd)
//   - mixed punctuation
//   - multibyte UTF-8 (German umlaut, Japanese kana, emoji)
//
// Gated by both:
//   TRANSCRIBE_WHISPER_GGUF_TINY     - whisper-tiny GGUF
//   TRANSCRIBE_WHISPER_BIN_TINY_Q8_0 - any whisper-tiny .bin (q8_0
//                                      or q5_1; F16 also fine — same
//                                      vocab across quantizations).

#include "transcribe.h"

#include <sys/stat.h>

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

bool file_exists(const char * path) {
    struct stat st{};
    return ::stat(path, &st) == 0;
}

const char * env_or_null(const char * key) {
    const char * v = std::getenv(key);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

std::vector<int32_t> tokenize(const transcribe_model * model, const char * text) {
    int32_t   buf[256];
    const int n = transcribe_tokenize(model, text, buf, 256);
    if (n < 0) {
        return {};
    }
    return std::vector<int32_t>(buf, buf + n);
}

std::string fmt_ids(const std::vector<int32_t> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) {
            s += ", ";
        }
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

void compare(const transcribe_model * gguf, const transcribe_model * bin, const char * text) {
    const auto a = tokenize(gguf, text);
    const auto b = tokenize(bin, text);
    if (a == b && !a.empty()) {
        return;
    }
    std::fprintf(stderr, "FAIL parity for %s\n  gguf: %s\n  bin:  %s\n", text, fmt_ids(a).c_str(), fmt_ids(b).c_str());
    ++g_failures;
}

}  // namespace

int main() {
    const char * gguf_path = env_or_null("TRANSCRIBE_WHISPER_GGUF_TINY");
    const char * bin_path  = env_or_null("TRANSCRIBE_WHISPER_BIN_TINY_Q8_0");
    if (gguf_path == nullptr || bin_path == nullptr || !file_exists(gguf_path) || !file_exists(bin_path)) {
        std::fprintf(stderr,
                     "SKIP: TRANSCRIBE_WHISPER_GGUF_TINY and "
                     "TRANSCRIBE_WHISPER_BIN_TINY_Q8_0 must both be set\n");
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CPU;

    transcribe_model * gguf = nullptr;
    if (transcribe_model_load_file(gguf_path, &mp, &gguf) != TRANSCRIBE_OK || gguf == nullptr) {
        std::fprintf(stderr, "FAIL: could not load GGUF %s\n", gguf_path);
        return 1;
    }
    transcribe_model * bin = nullptr;
    if (transcribe_model_load_file(bin_path, &mp, &bin) != TRANSCRIBE_OK || bin == nullptr) {
        std::fprintf(stderr, "FAIL: could not load .bin %s\n", bin_path);
        transcribe_model_free(gguf);
        return 1;
    }

    const char * probes[] = {
        // Simple ASCII.
        "hello",
        " hello",
        " hello world",
        // Whisper-typical leading-space patterns.
        " inaugural address",
        " country",
        " fellow Americans",
        // Digits.
        "1989",
        " 1989",
        "100 200 300",
        // Contractions (case-sensitive in GPT-2).
        "it's",
        " I've",
        " they're",
        // Punctuation runs.
        "hello, world!",
        " ... and so on?",
        // Multibyte UTF-8.
        "café",
        " München",
        // CJK + emoji.
        " 日本語",
        " 🚀 launch",
    };

    for (const char * p : probes) {
        compare(gguf, bin, p);
    }

    transcribe_model_free(bin);
    transcribe_model_free(gguf);

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d parity check(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}
