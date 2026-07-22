// whisper_bin_parser_unit.cpp - unit smoke for the whisper.cpp `.bin`
// parser and the magic-byte dispatch.
//
// Three behaviors covered:
//
//   1. Silero VAD `.bin` (also uses 'ggml' magic) is rejected with
//      TRANSCRIBE_ERR_UNSUPPORTED_ARCH at the hparams gate.
//   2. The truncated `for-tests-*.bin` fixtures from the upstream
//      whisper.cpp repo (header-only, no tensor payload) are rejected
//      with TRANSCRIBE_ERR_GGUF after the parser realizes there are
//      no tensors in the manifest.
//   3. A real whisper.cpp `.bin` (e.g. ggml-tiny-q8_0.bin) parses
//      cleanly and the parsed hparams match expected values.
//
// All three are gated by env vars rather than local fixtures because
// the upstream artifacts are not part of this repo. Each path skips
// (RC 77) when the corresponding env var is unset or the file is
// missing.
//
//   TRANSCRIBE_WHISPER_BIN_SILERO       - any .bin file with `ggml`
//                                         magic that is NOT a whisper
//                                         model. The upstream
//                                         models/for-tests-silero-v6.2.0-ggml.bin
//                                         is the canonical test
//                                         artifact.
//   TRANSCRIBE_WHISPER_BIN_TRUNCATED    - a stripped `for-tests-*.bin`
//                                         from upstream models/.
//                                         Whisper-shaped hparams,
//                                         no tensor payload.
//   TRANSCRIBE_WHISPER_BIN_TINY_Q8_0    - a real whisper.cpp .bin
//                                         (recommended:
//                                         ggml-tiny-q8_0.bin from
//                                         huggingface.co/ggerganov/
//                                         whisper.cpp).

#include "transcribe-bin-loader.h"

#include <sys/stat.h>
#ifndef _WIN32
#    include <unistd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_skipped  = 0;

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

void test_silero_rejected() {
    const char * path = env_or_null("TRANSCRIBE_WHISPER_BIN_SILERO");
    if (path == nullptr || !file_exists(path)) {
        ++g_skipped;
        return;
    }
    transcribe::bin_loader::WhisperBinModel m;
    const auto                              rc = transcribe::bin_loader::parse_whisper_bin(path, m);
    CHECK(rc == TRANSCRIBE_ERR_UNSUPPORTED_ARCH);
}

void test_truncated_rejected() {
    const char * path = env_or_null("TRANSCRIBE_WHISPER_BIN_TRUNCATED");
    if (path == nullptr || !file_exists(path)) {
        ++g_skipped;
        return;
    }
    transcribe::bin_loader::WhisperBinModel m;
    const auto                              rc = transcribe::bin_loader::parse_whisper_bin(path, m);
    // for-tests fixtures pass the hparams gate (real geometry) but
    // fail at the tensor-manifest pass with no tensors declared.
    CHECK(rc == TRANSCRIBE_ERR_GGUF);
}

void test_tiny_q8_parsed() {
    const char * path = env_or_null("TRANSCRIBE_WHISPER_BIN_TINY_Q8_0");
    if (path == nullptr || !file_exists(path)) {
        ++g_skipped;
        return;
    }
    transcribe::bin_loader::WhisperBinModel m;
    const auto                              rc = transcribe::bin_loader::parse_whisper_bin(path, m);
    CHECK(rc == TRANSCRIBE_OK);

    // Tiny multilingual whisper geometry.
    CHECK(m.hp.n_audio_layer == 4);
    CHECK(m.hp.n_text_layer == 4);
    CHECK(m.hp.n_audio_state == 384);
    CHECK(m.hp.n_text_state == 384);
    CHECK(m.hp.n_audio_head == 6);
    CHECK(m.hp.n_text_head == 6);
    CHECK(m.hp.n_mels == 80);
    CHECK(m.is_multilingual);
    CHECK(m.num_languages == 99);

    // Mel filterbank dims for whisper: 80 mels × 201 freq bins.
    CHECK(m.n_mel_filters == 80);
    CHECK(m.n_fft_filters == 201);
    CHECK(m.mel_filterbank.size() == 80u * 201u);

    // Tensor manifest: tiny multilingual has 167 tensors (10 input +
    // 15 + 15*4 + 24*4). Allow a small range to permit minor variants.
    CHECK(m.tensors.size() >= 160 && m.tensors.size() <= 200);

    // Spot-check a couple of canonical legacy names.
    bool saw_conv1     = false;
    bool saw_token_emb = false;
    for (const auto & t : m.tensors) {
        if (t.name == "encoder.conv1.weight") {
            saw_conv1 = true;
        }
        if (t.name == "decoder.token_embedding.weight") {
            saw_token_emb = true;
        }
    }
    CHECK(saw_conv1);
    CHECK(saw_token_emb);
}

void test_missing_path() {
    transcribe::bin_loader::WhisperBinModel m;
    const auto rc = transcribe::bin_loader::parse_whisper_bin("/nonexistent/path/that/does/not/exist.bin", m);
    CHECK(rc == TRANSCRIBE_ERR_FILE_NOT_FOUND);
}

// Write a minimal header-only .bin with the given hparams + mel
// filter dims to `path`. Vocab and tensors are intentionally absent
// so the parser will fail at the manifest stage if it gets past the
// header checks — but for these tests we only care about the
// hparams + mel-filter-dim gates.
bool write_synthetic_bin(const std::string & path,
                         int32_t             n_vocab,
                         int32_t             n_audio_layer,
                         int32_t             n_text_layer,
                         int32_t             n_mels,
                         int32_t             n_mel_filters,
                         int32_t             n_fft_filters) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    auto wi = [&](int32_t v) {
        f.write(reinterpret_cast<const char *>(&v), sizeof(v));
    };
    const uint32_t magic = 0x67676d6cu;
    f.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    // 11 × int32 hparams.
    wi(n_vocab);
    wi(1500);  // n_audio_ctx
    wi(384);   // n_audio_state
    wi(6);     // n_audio_head
    wi(n_audio_layer);
    wi(448);   // n_text_ctx
    wi(384);   // n_text_state
    wi(6);     // n_text_head
    wi(n_text_layer);
    wi(n_mels);
    wi(0);  // ftype
    // mel filter dims, then n_mel_filters * n_fft_filters floats.
    wi(n_mel_filters);
    wi(n_fft_filters);
    const size_t fb_bytes = static_cast<size_t>(n_mel_filters) * static_cast<size_t>(n_fft_filters) * sizeof(float);
    std::vector<char> zeros(fb_bytes, 0);
    f.write(zeros.data(), zeros.size());
    return static_cast<bool>(f);
}

// Portable stand-in for mkstemps(): create a unique empty file named
// transcribe_bin_test_<rand>.bin in the system temp dir. Returns the
// path, or an empty string on failure. Collision-then-overwrite is
// acceptable for these synthetic-fixture tests.
std::string make_temp_bin_path() {
    std::error_code             ec;
    const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        return {};
    }
    std::random_device          rd;
    const std::filesystem::path p = dir / ("transcribe_bin_test_" + std::to_string(rd()) + ".bin");
    std::ofstream               f(p, std::ios::binary);
    if (!f) {
        return {};
    }
    return p.string();
}

void test_bad_n_fft() {
    const std::string path = make_temp_bin_path();
    if (path.empty()) {
        std::fprintf(stderr, "SKIP: could not create tempfile for synthetic bin\n");
        ++g_skipped;
        return;
    }

    // Whisper-shaped hparams but non-canonical n_fft (200 instead of
    // 201). Parser must reject before we even get to the vocab phase.
    if (!write_synthetic_bin(path, 51865, 4, 4, 80, 80, 200)) {
        std::fprintf(stderr, "SKIP: failed to write synthetic .bin\n");
        ++g_skipped;
        std::remove(path.c_str());
        return;
    }
    transcribe::bin_loader::WhisperBinModel m;
    const auto                              rc = transcribe::bin_loader::parse_whisper_bin(path.c_str(), m);
    CHECK(rc == TRANSCRIBE_ERR_GGUF);
    std::remove(path.c_str());
}

void test_distil_layer_count_accepted() {
    // Distil-style asymmetric layers: n_text_layer=2 with otherwise
    // whisper-shaped hparams should pass the hparams gate. We don't
    // care that the parser later fails at "no tensors" — the hparams
    // check should not be the gating step.
    const std::string path = make_temp_bin_path();
    if (path.empty()) {
        ++g_skipped;
        return;
    }

    if (!write_synthetic_bin(path, 51865, 24, 2, 80, 80, 201)) {
        ++g_skipped;
        std::remove(path.c_str());
        return;
    }
    transcribe::bin_loader::WhisperBinModel m;
    const auto                              rc = transcribe::bin_loader::parse_whisper_bin(path.c_str(), m);
    // Header + mel filters parse; we then run out of bytes for the
    // vocab/tensor sections. The expected status is ERR_GGUF (with a
    // "no tensors" / truncated diagnostic), NOT UNSUPPORTED_ARCH —
    // that's the proof that the geometry gate accepts distil layers.
    CHECK(rc == TRANSCRIBE_ERR_GGUF);
    std::remove(path.c_str());
}

}  // namespace

int main() {
    test_missing_path();
    test_bad_n_fft();
    test_distil_layer_count_accepted();
    test_silero_rejected();
    test_truncated_rejected();
    test_tiny_q8_parsed();

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d check(s); %d sub-test(s) skipped\n", g_failures, g_skipped);
        return 1;
    }
    std::fprintf(stderr, "OK (%d sub-test(s) skipped)\n", g_skipped);
    return 0;
}
