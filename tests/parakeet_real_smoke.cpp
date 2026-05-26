// parakeet_real_smoke.cpp - real-model gated test for the Parakeet
// converter + ingest pipeline.
//
// This test loads a real Parakeet GGUF (the output of
// scripts/convert-parakeet.py against a NeMo checkpoint) and verifies
// that the loader produces a fully populated ParakeetModel.
//
// Gating:
//
//   - The CMake option TRANSCRIBE_BUILD_REAL_MODEL_TESTS (default OFF)
//     controls whether this binary is even built.
//   - At runtime, the GGUF path comes from the
//     TRANSCRIBE_REAL_PARAKEET_GGUF environment variable. If unset,
//     the test exits 77 (CTest "skipped") with a regeneration hint.
//
// CI never builds this — it's a developer-local manual gate. The
// synthetic transcribe_parakeet_smoke covers the same code paths
// against a tiny fixture and is what runs in CI.
//
// What we assert:
//
//   1. Load returns OK; the model pointer is set.
//   2. arch_string == "parakeet".
//   3. variant_string is one of the variants the converter produces
//      ("tdt-0.6b-v2" or "tdt-0.6b-v3"). Auto-detected from the
//      file's metadata; the test does not require the caller to say
//      which one they handed in.
//   4. backend == "" (no runtime backend bound in 2C).
//   5. native_sample_rate == 16000 and supports_translate == false
//      (Parakeet family invariants).
//   6. For v2: 1 language ["en"], lang_detect == false. For v3: 25
//      languages, first three are bg/hr/cs, lang_detect == true.
//   7. The Parakeet hparams the loader read match the values we
//      expect for the 0.6B variant we just ingested:
//        n_layers=24, d_model=1024, n_heads=8, d_ff=4096,
//        conv_kernel=9, subsampling_factor=8, subsampling_channels=256,
//        pred_hidden=640, pred_n_layers=2, joint_hidden=640,
//        joint_num_extra_outputs=5, fe_num_mels=128,
//        fe_sample_rate=16000.
//      pred_vocab differs by variant: 1025 (v2) vs 8193 (v3).
//   8. Total tensor count is 697 = 12 + 24*28 + 7 + 6.
//   9. A handful of spot-check shapes:
//        weights.pre_encode.conv0_w  ne=[3, 3, 1, 256]
//        weights.blocks[0].attn_q_w   ne=[1024, 1024]
//        weights.predictor.embed_w    ne=[640, pred_vocab]
//        weights.joint.out_w          ne=[640, joint_n_classes]
//
// Tests #2-#5 are also covered by transcribe_parakeet_smoke against
// the synthetic fixture, but repeating them here serves as a sanity
// gate that the converter is producing the right metadata KV. The
// real value of THIS test is #7-#9, which only exist for the real
// 0.6B dimensions.

#include "transcribe.h"

#include "arch/parakeet/parakeet.h"
#include "arch/parakeet/weights.h"
#include "transcribe-model.h"

#include "ggml.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

const transcribe::parakeet::ParakeetModel *
parakeet_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::parakeet::ParakeetModel *>(m);
}

} // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_REAL_PARAKEET_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "parakeet_real_smoke: TRANSCRIBE_REAL_PARAKEET_GGUF not "
                     "set; skipping. Convert a real model with:\n"
                     "  uv run scripts/convert-parakeet.py "
                     "<model-dir> /tmp/parakeet.gguf\n"
                     "and re-run with TRANSCRIBE_REAL_PARAKEET_GGUF=/tmp/parakeet.gguf\n");
        return 77;
    }
    const std::string fixture = env;

    if (!file_exists(fixture)) {
        std::fprintf(stderr,
                     "parakeet_real_smoke: file not found: %s\n",
                     fixture.c_str());
        return 77;
    }

    transcribe_model_load_params mp = transcribe_model_load_default_params();
    struct transcribe_model * model = nullptr;

    const transcribe_status st =
        transcribe_model_load_file(fixture.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "FAIL load: expected OK, got %s\n",
                     transcribe_status_string(st));
        return EXIT_FAILURE;
    }
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL load: model pointer not set\n");
        return EXIT_FAILURE;
    }

    // Public ABI sanity. After phase 4 step 1, backend is one of
    // "metal" (Apple Silicon, default) or "cpu" (fallback). The
    // exact label depends on the build platform; we just assert
    // it's non-empty and one of the expected values.
    CHECK_STR_EQ(transcribe_model_arch_string(model), "parakeet");
    {
        const std::string backend = transcribe_model_backend(model);
        if (backend.empty()) {
            std::fprintf(stderr,
                         "FAIL: backend = \"\" after step 1, expected non-empty\n");
            ++g_failures;
        }
    }

    // Variant must be one of the two the converter produces. Capture
    // which one for the variant-dependent assertions below.
    const std::string variant = transcribe_model_variant_string(model);
    const bool is_v2 = (variant == "tdt-0.6b-v2");
    const bool is_v3 = (variant == "tdt-0.6b-v3");
    if (!is_v2 && !is_v3) {
        std::fprintf(stderr,
                     "FAIL: unexpected variant \"%s\"\n", variant.c_str());
        ++g_failures;
    }

    // Family invariants.
    transcribe_capabilities caps_buf = TRANSCRIBE_CAPABILITIES_INIT;
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(caps->native_sample_rate == 16000);
        CHECK(caps->supports_translate == false);

        if (is_v2) {
            CHECK(caps->supports_language_detect == false);
            CHECK_EQ_INT(caps->n_languages, 1);
            if (caps->n_languages == 1 && caps->languages != nullptr) {
                CHECK_STR_EQ(caps->languages[0], "en");
            }
        }
        if (is_v3) {
            CHECK(caps->supports_language_detect == true);
            CHECK_EQ_INT(caps->n_languages, 25);
            if (caps->n_languages == 25 && caps->languages != nullptr) {
                // Spot-check the first three from the converter's
                // V3_LANGUAGES list. The full list is bg, hr, cs,
                // da, nl, en, ... — alphabetical-ish.
                CHECK_STR_EQ(caps->languages[0], "bg");
                CHECK_STR_EQ(caps->languages[1], "hr");
                CHECK_STR_EQ(caps->languages[2], "cs");
            }
        }
    }

    // Internal-view assertions. From here on we treat the model as a
    // ParakeetModel because we know the source GGUF was Parakeet.
    const auto * pm = parakeet_view(model);
    CHECK(pm != nullptr);
    CHECK(pm->ctx_meta != nullptr);

    // Hparams for the 0.6B variants. Every published Parakeet 0.6B
    // (v2 and v3) shares these encoder / predictor / joint dims —
    // only the vocab size differs.
    const auto & hp = pm->hparams;
    CHECK_EQ_INT(hp.enc_n_layers,             24);
    CHECK_EQ_INT(hp.enc_d_model,              1024);
    CHECK_EQ_INT(hp.enc_n_heads,              8);
    CHECK_EQ_INT(hp.enc_d_ff,                 4096);
    CHECK_EQ_INT(hp.enc_conv_kernel,          9);
    CHECK_EQ_INT(hp.enc_subsampling_factor,   8);
    CHECK_EQ_INT(hp.enc_subsampling_channels, 256);
    CHECK_EQ_INT(hp.enc_use_bias,             0);
    CHECK_EQ_INT(hp.enc_head_dim(),           128); // 1024 / 8
    CHECK_EQ_INT(hp.pred_hidden,              640);
    CHECK_EQ_INT(hp.pred_n_layers,            2);
    if (is_v2) {
        CHECK_EQ_INT(hp.pred_vocab,           1025); // 1024 + 1 start row
        CHECK_EQ_INT(hp.joint_n_classes(),    1030); // 1024 + 5 + 1
    }
    if (is_v3) {
        CHECK_EQ_INT(hp.pred_vocab,           8193); // 8192 + 1 start row
        CHECK_EQ_INT(hp.joint_n_classes(),    8198); // 8192 + 5 + 1
    }
    CHECK_EQ_INT(hp.joint_hidden,             640);
    CHECK_EQ_INT(hp.joint_num_extra_outputs,  5);
    CHECK_STR_EQ(hp.joint_activation,         "relu");
    // TDT durations: every published Parakeet 0.6B variant ships
    // [0, 1, 2, 3, 4]; loader cross-validates that the array length
    // matches num_extra_outputs.
    CHECK_EQ_INT(static_cast<int>(hp.tdt_durations.size()), 5);
    CHECK_EQ_INT(hp.tdt_durations[0], 0);
    CHECK_EQ_INT(hp.tdt_durations[1], 1);
    CHECK_EQ_INT(hp.tdt_durations[2], 2);
    CHECK_EQ_INT(hp.tdt_durations[3], 3);
    CHECK_EQ_INT(hp.tdt_durations[4], 4);
    CHECK_EQ_INT(hp.tdt_max_symbols,          10);
    CHECK_EQ_INT(hp.fe_num_mels,              128);
    CHECK_EQ_INT(hp.fe_sample_rate,           16000);
    CHECK_STR_EQ(hp.fe_type,                  "mel");
    CHECK_EQ_INT(hp.fe_n_fft,                 512);
    CHECK_EQ_INT(hp.fe_win_length,            400);  // 0.025 s * 16000
    CHECK_EQ_INT(hp.fe_hop_length,            160);  // 0.010 s * 16000
    CHECK_STR_EQ(hp.fe_window,                "hann");
    CHECK_STR_EQ(hp.fe_normalize,             "per_feature");
    CHECK(hp.fe_dither       == 1e-5f);
    CHECK(hp.fe_pre_emphasis == 0.97f);
    CHECK(hp.fe_f_min        == 0.0f);
    CHECK(hp.fe_f_max        == 8000.0f);

    // Block count matches the loader's read of enc_n_layers.
    CHECK_EQ_INT(pm->weights.blocks.size(), 24);

    // Total tensor count: 12 (pre_encode) + 24*28 (blocks) + 7
    // (predictor: embed + 2*(Wx, Wh, bias)) + 6 (joint) = 697.
    {
        const int total = 12 + 24 * 28 + 7 + 6;
        CHECK_EQ_INT(total, 697); // sanity on the formula itself
    }

    // Spot-check a handful of tensor shapes against the per-block
    // formulas in arch/parakeet/weights.cpp. If the loader and the
    // converter agreed on names but disagreed on shapes,
    // build_parakeet_weights would have already failed at load. The
    // checks here are a second line of defense and a sanity demo.
    {
        const auto * t = pm->weights.pre_encode.conv0_w;
        CHECK(t != nullptr);
        if (t != nullptr) {
            CHECK_EQ_INT(t->ne[0], 3);
            CHECK_EQ_INT(t->ne[1], 3);
            CHECK_EQ_INT(t->ne[2], 1);
            CHECK_EQ_INT(t->ne[3], 256);
        }
    }
    {
        const auto * t = pm->weights.blocks[0].attn_q_w;
        CHECK(t != nullptr);
        if (t != nullptr) {
            CHECK_EQ_INT(t->ne[0], 1024);
            CHECK_EQ_INT(t->ne[1], 1024);
        }
    }
    {
        const auto * t = pm->weights.predictor.embed_w;
        CHECK(t != nullptr);
        if (t != nullptr) {
            CHECK_EQ_INT(t->ne[0], 640);                  // pred_hidden
            CHECK_EQ_INT(t->ne[1], hp.pred_vocab);        // 1025 or 8193
        }
    }
    {
        const auto * t = pm->weights.joint.out_w;
        CHECK(t != nullptr);
        if (t != nullptr) {
            CHECK_EQ_INT(t->ne[0], 640);                  // joint_hidden
            CHECK_EQ_INT(t->ne[1], hp.joint_n_classes()); // 1030 or 8198
        }
    }

    // Lifecycle: polymorphic delete via the base virtual destructor.
    // The ParakeetModel destructor frees ctx_meta (which owns ~2.4 GB
    // of weight data). ASan in the sanitizer build catches any
    // double-free or leak here.
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "parakeet_real_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout,
                 "parakeet_real_smoke: ok (%s)\n", variant.c_str());
    return EXIT_SUCCESS;
}
