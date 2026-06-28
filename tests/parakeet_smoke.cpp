// parakeet_smoke.cpp - end-to-end test for the Parakeet tensor ingest
// path through the public C ABI.
//
// Loads tokenizer_minimal.gguf (which is now a structurally complete
// minimal Parakeet model with 78 weight tensors filled with
// deterministic toy float32 data) and asserts:
//
//   1. The Parakeet hparams the loader read out of the GGUF KV match
//      the hparams the fixture wrote.
//   2. Every named slot in ParakeetWeights is non-null after load.
//   3. The tensor data was actually copied from the GGUF data section
//      into the model's ggml_context — verified by reading the first
//      element of two specific tensors and comparing against the
//      deterministic value the fixture wrote (idx + 0 * 0.0001 = idx).
//      A loader bug that left tensor data zeroed or garbage would fail
//      this assertion.
//   4. ggml_tensor::ne for a couple of representative tensors matches
//      the per-block shape formulas baked into weights.cpp's GET()
//      sequence — paranoia in case the canonical shape formulas drift
//      out of sync with the fixture.
//   5. transcribe_model_backend() is non-empty after load
//      (Metal on Apple Silicon, CPU elsewhere or on Metal init
//      failure). The synthetic fixture uses the same load path as
//      the real model so this is the same code being exercised.
//
// The test reaches into internal headers via the test target's
// PRIVATE include of src/. Same pattern as tokenizer_smoke. The public
// C ABI does not (and should not) expose ggml_tensor pointers
// directly.

#include "transcribe.h"

#include "arch/parakeet/parakeet.h"
#include "arch/parakeet/weights.h"
#include "transcribe-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#  error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

const std::string g_fixtures_dir = TRANSCRIBE_TEST_FIXTURES_DIR;

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

// Pull the parakeet.h ParakeetModel out of a public transcribe_model
// pointer. This is the test-only downcast — we know the load came
// from the parakeet handler because we just loaded a parakeet GGUF.
// In real code the family handler keeps the downcast inside its own
// init_context / run.
const transcribe::parakeet::ParakeetModel *
parakeet_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::parakeet::ParakeetModel *>(m);
}

} // namespace

int main() {
    const std::string fixture = g_fixtures_dir + "/tokenizer_minimal.gguf";

    if (!file_exists(fixture)) {
        std::fprintf(stderr,
                     "parakeet_smoke: fixture not found: %s\n"
                     "regenerate with:\n"
                     "  cmake --build build --target fixtures\n",
                     fixture.c_str());
        return 77; // CTest "skipped"
    }

    transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
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

    // Public ABI sanity. After load, backend is one of
    // "metal" or "cpu" (Metal-first on Apple Silicon, CPU fallback).
    // The exact label is platform-dependent so we just assert
    // non-empty here; the real-model test does the same.
    if (std::strcmp(transcribe_model_arch_string(model), "parakeet") != 0) {
        std::fprintf(stderr, "FAIL: arch_string != parakeet\n");
        ++g_failures;
    }
    {
        const char * backend = transcribe_model_backend(model);
        if (backend == nullptr || backend[0] == '\0') {
            std::fprintf(stderr,
                         "FAIL: backend = \"\" after step 1, expected non-empty\n");
            ++g_failures;
        }
    }

    // Internal-view assertions. From here on we treat the model as a
    // ParakeetModel because we know the source GGUF was Parakeet.
    const auto * pm = parakeet_view(model);
    CHECK(pm != nullptr);
    CHECK(pm->ctx_meta != nullptr);

    // ----- Hparams -----
    // These match the values baked into make_gguf_fixtures.py
    // PARAKEET_HP. If either side drifts the test fails loudly.
    const auto & hp = pm->hparams;
    CHECK_EQ_INT(hp.enc_n_layers,             2);
    CHECK_EQ_INT(hp.enc_d_model,              8);
    CHECK_EQ_INT(hp.enc_n_heads,              2);
    CHECK_EQ_INT(hp.enc_d_ff,                 16);
    CHECK_EQ_INT(hp.enc_conv_kernel,          3);
    CHECK_EQ_INT(hp.enc_subsampling_factor,   2);
    CHECK_EQ_INT(hp.enc_subsampling_channels, 4);
    CHECK_EQ_INT(hp.enc_pos_emb_max_len,      32);
    CHECK_EQ_INT(hp.enc_use_bias,             0); // default false
    CHECK_EQ_INT(hp.enc_head_dim(),           4); // 8 / 2
    CHECK_EQ_INT(hp.pred_hidden,              4);
    CHECK_EQ_INT(hp.pred_n_layers,            1);
    CHECK_EQ_INT(hp.pred_vocab,               17);
    CHECK_EQ_INT(hp.joint_hidden,             4);
    CHECK_EQ_INT(hp.joint_num_extra_outputs,  2);
    CHECK_EQ_INT(hp.joint_n_classes(),        19); // (17-1) + 2 + 1
    CHECK_STR_EQ(hp.joint_activation,         "relu");
    // TDT durations: 2 entries to match joint_num_extra_outputs.
    CHECK_EQ_INT(static_cast<int>(hp.tdt_durations.size()), 2);
    CHECK_EQ_INT(hp.tdt_durations[0], 0);
    CHECK_EQ_INT(hp.tdt_durations[1], 1);
    CHECK_EQ_INT(hp.tdt_max_symbols,          10);
    // Frontend hparams: full stt.frontend.* block. The synthetic
    // fixture's PARAKEET_HP and the loader's read_parakeet_hparams
    // both have to know about every field, and this assertion is the
    // gate that catches a future contributor who adds a key on one
    // side but forgets the other.
    CHECK_STR_EQ(hp.fe_type,                  "mel");
    CHECK_EQ_INT(hp.fe_num_mels,              4);
    CHECK_EQ_INT(hp.fe_sample_rate,           16000);
    CHECK_EQ_INT(hp.fe_n_fft,                 16);
    CHECK_EQ_INT(hp.fe_win_length,            8);
    CHECK_EQ_INT(hp.fe_hop_length,            4);
    CHECK_STR_EQ(hp.fe_window,                "hann");
    CHECK_STR_EQ(hp.fe_normalize,             "per_feature");
    // f32 fields are checked exactly because the fixture writes
    // exact values and the round-trip is bit-perfect for these
    // sizes. If a future fixture starts using less round-friendly
    // values, switch to a fabs(diff) < epsilon check.
    CHECK(hp.fe_dither       == 1e-5f);
    CHECK(hp.fe_pre_emphasis == 0.97f);
    CHECK(hp.fe_f_min        == 0.0f);
    CHECK(hp.fe_f_max        == 8000.0f);

    // ----- Pre-encode slots populated -----
    const auto & pe = pm->weights.pre_encode;
    CHECK(pe.conv0_w != nullptr);
    CHECK(pe.conv0_b != nullptr);
    CHECK(pe.conv2_w != nullptr);
    CHECK(pe.conv2_b != nullptr);
    CHECK(pe.conv3_w != nullptr);
    CHECK(pe.conv3_b != nullptr);
    CHECK(pe.conv5_w != nullptr);
    CHECK(pe.conv5_b != nullptr);
    CHECK(pe.conv6_w != nullptr);
    CHECK(pe.conv6_b != nullptr);
    CHECK(pe.out_w   != nullptr);
    CHECK(pe.out_b   != nullptr);

    // ----- Encoder block slots populated for every layer -----
    CHECK_EQ_INT(pm->weights.blocks.size(), 2);
    for (size_t i = 0; i < pm->weights.blocks.size(); ++i) {
        const auto & b = pm->weights.blocks[i];
        // Spot-check one slot per logical sub-component to confirm
        // every category got populated. Not exhaustive — the goal is
        // "the build_parakeet_weights walk visited every code path",
        // not "I am about to forget the names of all 28 tensors".
        CHECK(b.norm_ff1_w  != nullptr);
        CHECK(b.ff1_lin1_w  != nullptr);
        CHECK(b.norm_attn_w != nullptr);
        CHECK(b.attn_q_w    != nullptr);
        CHECK(b.attn_pos_u  != nullptr);
        CHECK(b.attn_pos_v  != nullptr);
        CHECK(b.norm_conv_w != nullptr);
        CHECK(b.conv_pw1_w  != nullptr);
        CHECK(b.conv_dw_w   != nullptr);
        CHECK(b.conv_pw2_w  != nullptr);
        CHECK(b.conv_bn_w   != nullptr);
        CHECK(b.conv_bn_rm  != nullptr);
        CHECK(b.conv_bn_rv  != nullptr);
        CHECK(b.norm_ff2_w  != nullptr);
        CHECK(b.ff2_lin1_w  != nullptr);
        CHECK(b.norm_out_w  != nullptr);
    }

    // ----- Predictor slots populated -----
    CHECK(pm->weights.predictor.embed_w != nullptr);
    CHECK_EQ_INT(pm->weights.predictor.lstm.size(), 1);
    if (!pm->weights.predictor.lstm.empty()) {
        const auto & l = pm->weights.predictor.lstm[0];
        CHECK(l.Wx != nullptr);
        CHECK(l.Wh != nullptr);
        CHECK(l.b  != nullptr);
    }

    // ----- Joint slots populated -----
    CHECK(pm->weights.joint.enc_w  != nullptr);
    CHECK(pm->weights.joint.enc_b  != nullptr);
    CHECK(pm->weights.joint.pred_w != nullptr);
    CHECK(pm->weights.joint.pred_b != nullptr);
    CHECK(pm->weights.joint.out_w  != nullptr);
    CHECK(pm->weights.joint.out_b  != nullptr);

    // ----- Spot-check tensor shapes against the catalog -----
    // The first pre_encode conv: ne should be [3, 3, 1, channels].
    if (pe.conv0_w != nullptr) {
        CHECK_EQ_INT(pe.conv0_w->ne[0], 3);
        CHECK_EQ_INT(pe.conv0_w->ne[1], 3);
        CHECK_EQ_INT(pe.conv0_w->ne[2], 1);
        CHECK_EQ_INT(pe.conv0_w->ne[3], 4);
    }
    // The block-0 q projection should be square [d_model, d_model].
    if (!pm->weights.blocks.empty() &&
        pm->weights.blocks[0].attn_q_w != nullptr)
    {
        const auto * t = pm->weights.blocks[0].attn_q_w;
        CHECK_EQ_INT(t->ne[0], 8);
        CHECK_EQ_INT(t->ne[1], 8);
        CHECK_EQ_INT(t->ne[2], 1);
        CHECK_EQ_INT(t->ne[3], 1);
    }
    // The joint output head: ne[0] = joint_hidden, ne[1] = n_classes.
    if (pm->weights.joint.out_w != nullptr) {
        CHECK_EQ_INT(pm->weights.joint.out_w->ne[0], hp.joint_hidden);
        CHECK_EQ_INT(pm->weights.joint.out_w->ne[1], hp.joint_n_classes());
    }

    // ----- Spot-check that data was actually copied -----
    // The fixture writes deterministic data: tensor at index i has
    // first element exactly float(i). Pre-encode is the first group,
    // and conv.0.weight is the very first tensor (index 0), so its
    // first element should be 0.0f. Pre-encode.out.weight is at index
    // 10 (12 pre_encode tensors, out_w is the 11th), so its first
    // element should be 10.0f.
    //
    // The two checks together prove (a) the data section got
    // streamed into the backend buffer (zeros would fail both), (b)
    // the per-tensor offsets in the GGUF tensor info section match
    // what the loader reads (otherwise the second check would land
    // on the wrong tensor).
    //
    // Reads go through ggml_backend_tensor_get rather than direct
    // pointer access so the test is correct on every backend
    // (host buffers + discrete GPUs alike).
    if (pe.conv0_w != nullptr) {
        float v = 0.0f;
        ggml_backend_tensor_get(pe.conv0_w, &v, 0, sizeof(float));
        if (v != 0.0f) {
            std::fprintf(stderr,
                         "FAIL pre_encode.conv0_w[0]: got %f, expected 0.0\n", v);
            ++g_failures;
        }
    } else {
        std::fprintf(stderr, "FAIL pre_encode.conv0_w is null\n");
        ++g_failures;
    }
    if (pe.out_w != nullptr) {
        float v = 0.0f;
        ggml_backend_tensor_get(pe.out_w, &v, 0, sizeof(float));
        if (v != 10.0f) {
            std::fprintf(stderr,
                         "FAIL pre_encode.out_w[0]: got %f, expected 10.0\n", v);
            ++g_failures;
        }
    } else {
        std::fprintf(stderr, "FAIL pre_encode.out_w is null\n");
        ++g_failures;
    }

    // ----- Lifecycle -----
    // Polymorphic delete via the base virtual destructor. The
    // ParakeetModel destructor frees ctx_meta (which owns every
    // tensor's data buffer). ASan in the sanitizer build catches any
    // double-free, leak, or use-after-free here.
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "parakeet_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "parakeet_smoke: ok\n");
    return EXIT_SUCCESS;
}
