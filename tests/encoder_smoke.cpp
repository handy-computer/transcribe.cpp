// encoder_smoke.cpp - real-model gated encoder accuracy test.
//
// Phase 4 step 4. Loads a real Parakeet GGUF, runs the C++ encoder
// forward on samples/jfk.wav, and compares the resulting encoder
// output element-wise to the parakeet-mlx-derived golden at
// tests/golden/parakeet/enc.final.f32.
//
// Backend-aware tolerance budget. The CPU backend computes
// everything in fp32 and matches the parakeet-mlx fp32 reference
// to ~3.84e-06 max abs. The Metal backend uses simdgroup f16
// internally for its matmul kernels, so its error is ~1000x
// looser at ~3.55e-03 max abs. Both are well within "good
// enough for STT" — the joint network's argmax is robust to
// ~1e-2 perturbations in encoder activations — but the strict
// CPU tolerance won't pass on Metal. The test detects the
// backend that the loader picked and applies the right budget:
//
//   CPU:    max abs < 1e-4    (~26x headroom over 3.84e-06)
//   Metal:  max abs < 1e-2    (~3x  headroom over 3.55e-03)
//
// Gating:
//   - CMake option TRANSCRIBE_BUILD_REAL_MODEL_TESTS controls
//     whether the binary is built (default OFF).
//   - At run time, TRANSCRIBE_REAL_PARAKEET_GGUF must point at a
//     converted v2 GGUF (the golden was generated against v2; v3
//     produces a different encoder output and would need its own
//     golden — see RESUME.md). If unset, the test exits 77
//     (CTest "skipped").
//
// What we assert:
//   1. The full pipeline (load -> mel -> encoder) completes OK.
//   2. The encoder output tensor has the expected shape:
//      ne=[d_model=1024, T_enc, 1, 1] where T_enc matches the
//      golden's row count (138 for jfk.wav at 16 kHz).
//   3. Element-wise comparison against the golden is within
//      tolerance.
//   4. Lifecycle teardown (context_free, model_free) is clean.

#include "transcribe.h"

#include "arch/parakeet/parakeet.h"
#include "wav.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>  // for setenv

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

std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const std::streampos sz = f.tellg();
    if (sz < 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<float> out(static_cast<size_t>(sz) / sizeof(float));
    f.read(reinterpret_cast<char *>(out.data()), sz);
    return out;
}

const transcribe::parakeet::ParakeetContext *
parakeet_ctx_view(const struct transcribe_context * ctx) {
    return static_cast<const transcribe::parakeet::ParakeetContext *>(ctx);
}

} // namespace

int main() {
    // ---- Resolve env -----------------------------------------------
    const char * gguf_env = std::getenv("TRANSCRIBE_REAL_PARAKEET_GGUF");
    if (gguf_env == nullptr || gguf_env[0] == '\0') {
        std::fprintf(stderr,
                     "encoder_smoke: TRANSCRIBE_REAL_PARAKEET_GGUF not set; "
                     "skipping\n");
        return 77;
    }
    if (!file_exists(gguf_env)) {
        std::fprintf(stderr,
                     "encoder_smoke: gguf file not found: %s\n", gguf_env);
        return 77;
    }

    const std::string golden_path =
        std::string(TRANSCRIBE_TEST_GOLDEN_DIR) + "/enc.final.f32";
    if (!file_exists(golden_path)) {
        std::fprintf(stderr,
                     "encoder_smoke: golden not found: %s\n",
                     golden_path.c_str());
        return EXIT_FAILURE;
    }
    const std::string wav_path =
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    if (!file_exists(wav_path)) {
        std::fprintf(stderr, "encoder_smoke: wav not found: %s\n",
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
            std::fprintf(stderr, "encoder_smoke: load failed: %s\n",
                         transcribe_status_string(st));
            return EXIT_FAILURE;
        }
    }

    // The golden was generated against v2; v3 has identical encoder
    // dims but different weights, so its encoder output diverges.
    // Hard-fail if the test was pointed at v3 — better than silently
    // mis-validating.
    {
        const std::string variant = transcribe_model_variant_string(model);
        if (variant != "tdt-0.6b-v2") {
            std::fprintf(stderr,
                         "encoder_smoke: golden is for tdt-0.6b-v2 only, "
                         "got \"%s\"\n", variant.c_str());
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    // Pick tolerances based on the runtime backend the loader
    // chose. Metal uses simdgroup f16 internally for its matmul
    // kernels and lands ~1000x looser than the CPU's all-fp32
    // path; both are within STT-acceptable range.
    //
    // Quantized GGUFs accumulate additional rounding error in the
    // encoder matmuls. The defaults below are sized for an F32 (or
    // F16, which lands bit-identical at the encoder output) build.
    // To run encoder_smoke against a Q8_0 / Q5_K_M / Q4_K_M model,
    // override via the env vars below — typical bands measured
    // empirically against the parakeet-mlx F32 reference dump:
    //
    //   q8_0:    max=1e-1  mean=1e-3
    //   q5_k_m:  max=1     mean=1e-2
    //   q4_k_m:  max=1     mean=2e-2
    //
    // The harness in scripts/quant_accuracy.py is the canonical
    // per-quant gate; this env-override exists so encoder_smoke can
    // be re-used as a CI knob without standing up the full harness.
    const std::string backend = transcribe_model_backend(model);
    double max_abs_tol  = 1e-4;
    double mean_abs_tol = 1e-6;
    // GPU backends (Metal, Vulkan, CUDA) use lower-precision internal
    // matmul kernels and need wider tolerances. The device registry
    // names vary (e.g. "MTL0", "Vulkan0", "CUDA0") so we treat
    // anything that isn't "cpu"/"CPU" as a GPU backend.
    if (backend != "cpu" && backend != "CPU") {
        max_abs_tol  = 1e-2;
        mean_abs_tol = 1e-3;
    }
    if (const char * env = std::getenv("TRANSCRIBE_ENCODER_MAX_ABS_TOL")) {
        max_abs_tol = std::strtod(env, nullptr);
    }
    if (const char * env = std::getenv("TRANSCRIBE_ENCODER_MEAN_ABS_TOL")) {
        mean_abs_tol = std::strtod(env, nullptr);
    }
    std::fprintf(stdout, "encoder_smoke: backend=%s\n", backend.c_str());

    // ---- Load wav --------------------------------------------------
    std::vector<float> pcm;
    std::string load_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, load_err)) {
        std::fprintf(stderr, "encoder_smoke: wav load: %s\n",
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
            std::fprintf(stderr, "encoder_smoke: ctx init: %s\n",
                         transcribe_status_string(st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    transcribe_params rp = transcribe_default_params();
    {
        const transcribe_status st =
            transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "encoder_smoke: run: %s\n",
                         transcribe_status_string(st));
            transcribe_context_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    // ---- Read encoder output via internal view ---------------------
    //
    // The public API doesn't expose the encoder output yet (that's
    // phase 5's decoder consumer). For the accuracy test we cast to
    // ParakeetContext and read encoder_out via ggml_backend_tensor_get.
    const auto * pc = parakeet_ctx_view(ctx);
    CHECK(pc != nullptr);
    if (pc == nullptr || pc->encoder_out == nullptr) {
        std::fprintf(stderr,
                     "encoder_smoke: encoder_out is null after run\n");
        transcribe_context_free(ctx);
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    const ggml_tensor * enc_out = pc->encoder_out;
    // Expected ne = [d_model=1024, T_enc=138, 1, 1] for jfk.wav.
    CHECK_EQ_INT(enc_out->ne[0], 1024);
    CHECK_EQ_INT(enc_out->ne[1], 138);
    CHECK_EQ_INT(enc_out->ne[2], 1);
    CHECK_EQ_INT(enc_out->ne[3], 1);

    const size_t n_elem = static_cast<size_t>(enc_out->ne[0]) *
                          static_cast<size_t>(enc_out->ne[1]);
    std::vector<float> cpp_buf(n_elem);
    ggml_backend_tensor_get(enc_out, cpp_buf.data(), 0,
                            n_elem * sizeof(float));

    // ---- Load golden -----------------------------------------------
    const std::vector<float> ref_buf = read_f32_file(golden_path);
    if (ref_buf.size() != n_elem) {
        std::fprintf(stderr,
                     "encoder_smoke: golden size mismatch: %zu floats, "
                     "expected %zu\n",
                     ref_buf.size(), n_elem);
        transcribe_context_free(ctx);
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // ---- Element-wise comparison -----------------------------------
    double max_abs  = 0.0;
    double sum_abs  = 0.0;
    size_t first_diff_idx = n_elem;  // sentinel = "no diff"
    for (size_t i = 0; i < n_elem; ++i) {
        const double diff = std::fabs(static_cast<double>(cpp_buf[i]) -
                                       static_cast<double>(ref_buf[i]));
        if (diff > max_abs) {
            max_abs = diff;
        }
        if (diff > 0.0 && first_diff_idx == n_elem) {
            first_diff_idx = i;
        }
        sum_abs += diff;
    }
    const double mean_abs = sum_abs / static_cast<double>(n_elem);

    std::fprintf(stdout,
                 "encoder_smoke: max_abs=%.4e mean_abs=%.4e first_diff=%zu "
                 "(tolerances max<%g mean<%g, backend=%s)\n",
                 max_abs, mean_abs,
                 first_diff_idx == n_elem ? static_cast<size_t>(0) : first_diff_idx,
                 max_abs_tol, mean_abs_tol, backend.c_str());

    if (max_abs > max_abs_tol) {
        std::fprintf(stderr,
                     "FAIL: max_abs %.4e exceeds tolerance %g\n",
                     max_abs, max_abs_tol);
        ++g_failures;
    }
    if (mean_abs > mean_abs_tol) {
        std::fprintf(stderr,
                     "FAIL: mean_abs %.4e exceeds tolerance %g\n",
                     mean_abs, mean_abs_tol);
        ++g_failures;
    }

    // ---- Teardown --------------------------------------------------
    transcribe_context_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "encoder_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "encoder_smoke: ok\n");
    return EXIT_SUCCESS;
}
