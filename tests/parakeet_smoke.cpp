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

#include "arch/parakeet/parakeet.h"
#include "arch/parakeet/weights.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

const std::string g_fixtures_dir = TRANSCRIBE_TEST_FIXTURES_DIR;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_EQ_INT(actual, expected)                                                                           \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                                                        \
    do {                                                                                                          \
        const std::string _av = (a);                                                                              \
        const std::string _bv = (b);                                                                              \
        if (_av != _bv) {                                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: \"%s\" != \"%s\"\n", __FILE__, __LINE__, _av.c_str(), _bv.c_str()); \
            ++g_failures;                                                                                         \
        }                                                                                                         \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// Pull the parakeet.h ParakeetModel out of a public transcribe_model
// pointer. This is the test-only downcast — we know the load came
// from the parakeet handler because we just loaded a parakeet GGUF.
// In real code the family handler keeps the downcast inside its own
// init_context / run.
const transcribe::parakeet::ParakeetModel * parakeet_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::parakeet::ParakeetModel *>(m);
}

// Fluid Speech Pro's pre-Conformer block. pre_block_kernel announces it;
// the KV pair must be consistent, only the offline (regular) attention
// style may carry it, and a file that announces it without the eight
// enc.pre_block.* tensors must fail to load rather than silently run the
// plain encoder.
void check_pre_block(const std::string & fixture) {
    using transcribe::parakeet::ParakeetHParams;
    using transcribe::parakeet::read_parakeet_hparams;

    gguf_init_params gp{};
    gp.no_alloc       = true;
    gguf_context * kv = gguf_init_from_file(fixture.c_str(), gp);
    CHECK(kv != nullptr);
    if (kv == nullptr) {
        return;
    }
    {
        ParakeetHParams hp{};
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_OK);
        CHECK_EQ_INT(hp.enc_pre_block_kernel, 0);
    }
    gguf_set_val_i32(kv, "stt.parakeet.encoder.pre_block_kernel", 3);
    {
        ParakeetHParams hp{};  // kernel without dim
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    gguf_set_val_i32(kv, "stt.parakeet.encoder.pre_block_dim", 4);
    {
        ParakeetHParams hp{};
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_OK);
        CHECK_EQ_INT(hp.enc_pre_block_kernel, 3);
        CHECK_EQ_INT(hp.enc_pre_block_dim, 4);
    }
    gguf_set_val_i32(kv, "stt.parakeet.encoder.pre_block_kernel", 4);
    {
        ParakeetHParams hp{};  // even kernel
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    gguf_set_val_i32(kv, "stt.parakeet.encoder.pre_block_kernel", 3);
    gguf_set_val_str(kv, "stt.parakeet.encoder.att_context_style", "chunked_limited");
    {
        ParakeetHParams hp{};  // streaming encoder style
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    gguf_free(kv);

    // Rewrite the fixture with the KVs, first without and then with the
    // block's tensors, and load each through the public ABI.
    // gguf_write_to_file reads tensor bytes through the ggml tensors it was
    // given, so copy the KVs into a fresh context and re-add the fixture's
    // tensors from the data context the reader filled.
    ggml_context *   data_ctx = nullptr;
    gguf_init_params dp{};
    dp.no_alloc        = false;
    dp.ctx             = &data_ctx;
    gguf_context * src = gguf_init_from_file(fixture.c_str(), dp);
    CHECK(src != nullptr);
    if (src == nullptr) {
        return;
    }
    gguf_context * gf = gguf_init_empty();
    gguf_set_kv(gf, src);
    for (ggml_tensor * t = ggml_get_first_tensor(data_ctx); t != nullptr; t = ggml_get_next_tensor(data_ctx, t)) {
        gguf_add_tensor(gf, t);
    }
    gguf_set_val_i32(gf, "stt.parakeet.encoder.pre_block_kernel", 3);
    gguf_set_val_i32(gf, "stt.parakeet.encoder.pre_block_dim", 4);

    std::error_code             ec;
    const std::filesystem::path dir     = std::filesystem::temp_directory_path(ec);
    const std::string           missing = (dir / "transcribe_parakeet_pre_block_missing.gguf").string();
    const std::string           present = (dir / "transcribe_parakeet_pre_block_present.gguf").string();

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);

    CHECK(gguf_write_to_file(gf, missing.c_str(), false));
    {
        transcribe_model * m  = nullptr;
        transcribe_status  st = transcribe_model_load_file(missing.c_str(), &mp, &m);
        CHECK(st != TRANSCRIBE_OK);
        CHECK(m == nullptr);
        if (m != nullptr) {
            transcribe_model_free(m);
        }
    }

    // d_model=8 in the fixture; dim=4, kernel=3.
    ggml_init_params ip{};
    ip.mem_size          = 64 * 1024;
    ip.no_alloc          = false;
    ggml_context * extra = ggml_init(ip);
    CHECK(extra != nullptr);
    if (extra != nullptr) {
        struct spec {
            const char * name;
            int64_t      ne0, ne1, ne2;
        };

        const spec specs[] = {
            { "enc.pre_block.norm.weight", 8, 1, 1 },
            { "enc.pre_block.norm.bias",   8, 1, 1 },
            { "enc.pre_block.down.weight", 8, 4, 1 },
            { "enc.pre_block.down.bias",   4, 1, 1 },
            { "enc.pre_block.mix.weight",  3, 1, 4 },
            { "enc.pre_block.mix.bias",    4, 1, 1 },
            { "enc.pre_block.up.weight",   4, 8, 1 },
            { "enc.pre_block.up.bias",     8, 1, 1 },
        };
        for (const spec & sp : specs) {
            ggml_tensor * t = nullptr;
            if (sp.ne2 > 1) {
                t = ggml_new_tensor_3d(extra, GGML_TYPE_F32, sp.ne0, sp.ne1, sp.ne2);
            } else if (sp.ne1 > 1) {
                t = ggml_new_tensor_2d(extra, GGML_TYPE_F32, sp.ne0, sp.ne1);
            } else {
                t = ggml_new_tensor_1d(extra, GGML_TYPE_F32, sp.ne0);
            }
            ggml_set_name(t, sp.name);
            float * d = static_cast<float *>(t->data);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                d[i] = 0.01f * static_cast<float>(i % 7);
            }
            gguf_add_tensor(gf, t);
        }
        CHECK(gguf_write_to_file(gf, present.c_str(), false));

        transcribe_model * m  = nullptr;
        transcribe_status  st = transcribe_model_load_file(present.c_str(), &mp, &m);
        CHECK(st == TRANSCRIBE_OK);
        if (m != nullptr) {
            const auto * pm = parakeet_view(m);
            CHECK_EQ_INT(pm->hparams.enc_pre_block_kernel, 3);
            CHECK_EQ_INT(pm->hparams.enc_pre_block_dim, 4);
            CHECK(pm->weights.pre_block.norm_w != nullptr);
            CHECK(pm->weights.pre_block.up_b != nullptr);
            CHECK(pm->weights.pre_block.mix_w != nullptr);
            if (pm->weights.pre_block.mix_w != nullptr) {
                CHECK_EQ_INT(pm->weights.pre_block.mix_w->ne[0], 3);
                CHECK_EQ_INT(pm->weights.pre_block.mix_w->ne[2], 4);
            }
            CHECK(!pm->caps.supports_streaming);
            transcribe_model_free(m);
        }
        ggml_free(extra);
    }

    gguf_free(gf);
    gguf_free(src);
    ggml_free(data_ctx);
    std::filesystem::remove(missing, ec);
    std::filesystem::remove(present, ec);
}

// Fluid Speech Pro's clean-mode adapter is carried in the GGUF but not
// applied: the loader must accept a complete, well-shaped adapter and
// refuse a partial or mis-shaped one, or adapter tensors without the KVs.
void set_clean_mode_kvs(gguf_context * kv, const char * const * modules, size_t n_modules) {
    gguf_set_val_u32(kv, "stt.parakeet.clean_mode.rank", 2);
    gguf_set_val_f32(kv, "stt.parakeet.clean_mode.alpha", 4.0f);
    gguf_set_arr_str(kv, "stt.parakeet.clean_mode.modules", const_cast<const char **>(modules), n_modules);
}

void check_clean_mode(const std::string & fixture) {
    using transcribe::parakeet::ParakeetHParams;
    using transcribe::parakeet::read_parakeet_hparams;

    const char * const modules[] = { "attn.linear_q", "conv.pointwise1" };

    gguf_init_params gp{};
    gp.no_alloc       = true;
    gguf_context * kv = gguf_init_from_file(fixture.c_str(), gp);
    CHECK(kv != nullptr);
    if (kv == nullptr) {
        return;
    }
    {
        ParakeetHParams hp{};
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_OK);
        CHECK_EQ_INT(hp.clean_mode_rank, 0);
    }
    gguf_set_val_u32(kv, "stt.parakeet.clean_mode.rank", 2);
    {
        ParakeetHParams hp{};  // rank without alpha/modules
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    set_clean_mode_kvs(kv, modules, 2);
    {
        ParakeetHParams hp{};
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_OK);
        CHECK_EQ_INT(hp.clean_mode_rank, 2);
        CHECK(hp.clean_mode_alpha == 4.0f);
        CHECK_EQ_INT(hp.clean_mode_modules.size(), 2);
    }
    {
        const char * const bad[] = { "attn.linear_q", "conv.depthwise" };
        gguf_set_arr_str(kv, "stt.parakeet.clean_mode.modules", const_cast<const char **>(bad), 2);
        ParakeetHParams hp{};  // a module the adapter cannot name
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    {
        const char * const dup[] = { "attn.linear_q", "attn.linear_q" };
        gguf_set_arr_str(kv, "stt.parakeet.clean_mode.modules", const_cast<const char **>(dup), 2);
        ParakeetHParams hp{};
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    set_clean_mode_kvs(kv, modules, 2);
    gguf_set_val_f32(kv, "stt.parakeet.clean_mode.alpha", 0.0f);
    {
        ParakeetHParams hp{};
        CHECK(read_parakeet_hparams(kv, hp) == TRANSCRIBE_ERR_GGUF);
    }
    gguf_free(kv);

    ggml_context *   data_ctx = nullptr;
    gguf_init_params dp{};
    dp.no_alloc        = false;
    dp.ctx             = &data_ctx;
    gguf_context * src = gguf_init_from_file(fixture.c_str(), dp);
    CHECK(src != nullptr);
    if (src == nullptr) {
        return;
    }

    // Fixture: d_model=8, d_ff=16, 2 layers, pred_hidden=4, joint_hidden=4.
    // rank=2: attn.linear_q is 8->8, conv.pointwise1 is 8->16 (stored F16
    // here to cover the F16 adapter the quant presets write).
    struct spec {
        std::string name;
        int64_t     ne0, ne1;
        ggml_type   type;
    };

    std::vector<spec> full;
    for (int i = 0; i < 2; ++i) {
        const std::string l = "enc.blocks." + std::to_string(i) + ".";
        full.push_back({ l + "attn.linear_q.clean_mode.a", 8, 2, GGML_TYPE_F32 });
        full.push_back({ l + "attn.linear_q.clean_mode.b", 2, 8, GGML_TYPE_F32 });
        full.push_back({ l + "conv.pointwise1.clean_mode.a", 8, 2, GGML_TYPE_F16 });
        full.push_back({ l + "conv.pointwise1.clean_mode.b", 2, 16, GGML_TYPE_F16 });
    }
    full.push_back({ "pred.lstm.0.clean_mode.bias", 16, 1, GGML_TYPE_F32 });
    full.push_back({ "joint.pred.clean_mode.bias", 4, 1, GGML_TYPE_F32 });

    ggml_init_params ip{};
    ip.mem_size          = 256 * 1024;
    ip.no_alloc          = false;
    ggml_context * extra = ggml_init(ip);
    CHECK(extra != nullptr);
    if (extra == nullptr) {
        gguf_free(src);
        ggml_free(data_ctx);
        return;
    }

    std::error_code             ec;
    const std::filesystem::path dir  = std::filesystem::temp_directory_path(ec);
    const std::string           path = (dir / "transcribe_parakeet_clean_mode.gguf").string();

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);

    // Write the fixture plus `specs` (with or without the KVs) and load it.
    auto load_with = [&](bool with_kvs, const std::vector<spec> & specs, transcribe_model ** out) {
        gguf_context * gf = gguf_init_empty();
        gguf_set_kv(gf, src);
        for (ggml_tensor * t = ggml_get_first_tensor(data_ctx); t != nullptr; t = ggml_get_next_tensor(data_ctx, t)) {
            gguf_add_tensor(gf, t);
        }
        if (with_kvs) {
            set_clean_mode_kvs(gf, modules, 2);
        }
        for (const spec & sp : specs) {
            ggml_tensor * t = sp.ne1 > 1 ? ggml_new_tensor_2d(extra, sp.type, sp.ne0, sp.ne1) :
                                           ggml_new_tensor_1d(extra, sp.type, sp.ne0);
            ggml_set_name(t, sp.name.c_str());
            std::memset(t->data, 0, ggml_nbytes(t));
            gguf_add_tensor(gf, t);
        }
        const bool wrote = gguf_write_to_file(gf, path.c_str(), false);
        CHECK(wrote);
        gguf_free(gf);
        *out = nullptr;
        return wrote ? transcribe_model_load_file(path.c_str(), &mp, out) : TRANSCRIBE_ERR_GGUF;
    };
    auto expect_refused = [&](bool with_kvs, const std::vector<spec> & specs) {
        transcribe_model * m = nullptr;
        CHECK(load_with(with_kvs, specs, &m) != TRANSCRIBE_OK);
        CHECK(m == nullptr);
        if (m != nullptr) {
            transcribe_model_free(m);
        }
    };

    {
        transcribe_model * m = nullptr;
        CHECK(load_with(true, full, &m) == TRANSCRIBE_OK);
        if (m != nullptr) {
            const auto * pm = parakeet_view(m);
            CHECK_EQ_INT(pm->hparams.clean_mode_rank, 2);
            CHECK_EQ_INT(pm->hparams.clean_mode_modules.size(), 2);
            transcribe_model_free(m);
        }
    }
    expect_refused(true, {});                                     // KVs, no adapter tensors
    expect_refused(false, full);                                  // adapter tensors, no KVs
    {
        std::vector<spec> partial(full.begin(), full.end() - 1);  // no joint bias delta
        expect_refused(true, partial);
    }
    {
        std::vector<spec> partial(full.begin() + 1, full.end());  // layer 0 attn.linear_q A missing
        expect_refused(true, partial);
    }
    {
        std::vector<spec> bad = full;
        bad[0].ne0            = 2;  // A stored [rank, in] instead of [in, rank]
        bad[0].ne1            = 8;
        expect_refused(true, bad);
    }
    {
        std::vector<spec> bad = full;
        bad[2].type           = GGML_TYPE_Q8_0;  // quantized factor
        bad[2].ne0            = 32;
        expect_refused(true, bad);
    }
    {
        std::vector<spec> stray = full;  // a module the KVs do not list
        stray.push_back({ "enc.blocks.0.ff1.linear1.clean_mode.a", 8, 2, GGML_TYPE_F32 });
        expect_refused(true, stray);
    }

    ggml_free(extra);
    gguf_free(src);
    ggml_free(data_ctx);
    std::filesystem::remove(path, ec);
}

}  // namespace

int main() {
    const std::string fixture = g_fixtures_dir + "/tokenizer_minimal.gguf";

    if (!file_exists(fixture)) {
        std::fprintf(stderr,
                     "parakeet_smoke: fixture not found: %s\n"
                     "regenerate with:\n"
                     "  cmake --build build --target fixtures\n",
                     fixture.c_str());
        return 77;  // CTest "skipped"
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    struct transcribe_model * model = nullptr;

    const transcribe_status st = transcribe_model_load_file(fixture.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL load: expected OK, got %s\n", transcribe_status_string(st));
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
            std::fprintf(stderr, "FAIL: backend = \"\" after step 1, expected non-empty\n");
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
    CHECK_EQ_INT(hp.enc_n_layers, 2);
    CHECK_EQ_INT(hp.enc_d_model, 8);
    CHECK_EQ_INT(hp.enc_n_heads, 2);
    CHECK_EQ_INT(hp.enc_d_ff, 16);
    CHECK_EQ_INT(hp.enc_conv_kernel, 3);
    CHECK_EQ_INT(hp.enc_subsampling_factor, 2);
    CHECK_EQ_INT(hp.enc_subsampling_channels, 4);
    CHECK_EQ_INT(hp.enc_pos_emb_max_len, 32);
    CHECK_EQ_INT(hp.enc_use_bias, 0);    // default false
    CHECK_EQ_INT(hp.enc_head_dim(), 4);  // 8 / 2
    CHECK_EQ_INT(hp.pred_hidden, 4);
    CHECK_EQ_INT(hp.pred_n_layers, 1);
    CHECK_EQ_INT(hp.pred_vocab, 17);
    CHECK_EQ_INT(hp.joint_hidden, 4);
    CHECK_EQ_INT(hp.joint_num_extra_outputs, 2);
    CHECK_EQ_INT(hp.joint_n_classes(), 19);  // (17-1) + 2 + 1
    CHECK_STR_EQ(hp.joint_activation, "relu");
    // TDT durations: 2 entries to match joint_num_extra_outputs.
    CHECK_EQ_INT(static_cast<int>(hp.tdt_durations.size()), 2);
    CHECK_EQ_INT(hp.tdt_durations[0], 0);
    CHECK_EQ_INT(hp.tdt_durations[1], 1);
    CHECK_EQ_INT(hp.tdt_max_symbols, 10);
    // Frontend hparams: full stt.frontend.* block. The synthetic
    // fixture's PARAKEET_HP and the loader's read_parakeet_hparams
    // both have to know about every field, and this assertion is the
    // gate that catches a future contributor who adds a key on one
    // side but forgets the other.
    CHECK_STR_EQ(hp.fe_type, "mel");
    CHECK_EQ_INT(hp.fe_num_mels, 4);
    CHECK_EQ_INT(hp.fe_sample_rate, 16000);
    CHECK_EQ_INT(hp.fe_n_fft, 16);
    CHECK_EQ_INT(hp.fe_win_length, 8);
    CHECK_EQ_INT(hp.fe_hop_length, 4);
    CHECK_STR_EQ(hp.fe_window, "hann");
    CHECK_STR_EQ(hp.fe_normalize, "per_feature");
    // f32 fields are checked exactly because the fixture writes
    // exact values and the round-trip is bit-perfect for these
    // sizes. If a future fixture starts using less round-friendly
    // values, switch to a fabs(diff) < epsilon check.
    CHECK(hp.fe_dither == 1e-5f);
    CHECK(hp.fe_pre_emphasis == 0.97f);
    CHECK(hp.fe_f_min == 0.0f);
    CHECK(hp.fe_f_max == 8000.0f);

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
    CHECK(pe.out_w != nullptr);
    CHECK(pe.out_b != nullptr);

    // ----- Encoder block slots populated for every layer -----
    CHECK_EQ_INT(pm->weights.blocks.size(), 2);
    for (size_t i = 0; i < pm->weights.blocks.size(); ++i) {
        const auto & b = pm->weights.blocks[i];
        // Spot-check one slot per logical sub-component to confirm
        // every category got populated. Not exhaustive — the goal is
        // "the build_parakeet_weights walk visited every code path",
        // not "I am about to forget the names of all 28 tensors".
        CHECK(b.norm_ff1_w != nullptr);
        CHECK(b.ff1_lin1_w != nullptr);
        CHECK(b.norm_attn_w != nullptr);
        CHECK(b.attn_q_w != nullptr);
        CHECK(b.attn_pos_u != nullptr);
        CHECK(b.attn_pos_v != nullptr);
        CHECK(b.norm_conv_w != nullptr);
        CHECK(b.conv_pw1_w != nullptr);
        CHECK(b.conv_dw_w != nullptr);
        CHECK(b.conv_pw2_w != nullptr);
        CHECK(b.conv_bn_w != nullptr);
        CHECK(b.conv_bn_rm != nullptr);
        CHECK(b.conv_bn_rv != nullptr);
        CHECK(b.norm_ff2_w != nullptr);
        CHECK(b.ff2_lin1_w != nullptr);
        CHECK(b.norm_out_w != nullptr);
    }

    // ----- Predictor slots populated -----
    CHECK(pm->weights.predictor.embed_w != nullptr);
    CHECK_EQ_INT(pm->weights.predictor.lstm.size(), 1);
    if (!pm->weights.predictor.lstm.empty()) {
        const auto & l = pm->weights.predictor.lstm[0];
        CHECK(l.Wx != nullptr);
        CHECK(l.Wh != nullptr);
        CHECK(l.b != nullptr);
    }

    // ----- Joint slots populated -----
    CHECK(pm->weights.joint.enc_w != nullptr);
    CHECK(pm->weights.joint.enc_b != nullptr);
    CHECK(pm->weights.joint.pred_w != nullptr);
    CHECK(pm->weights.joint.pred_b != nullptr);
    CHECK(pm->weights.joint.out_w != nullptr);
    CHECK(pm->weights.joint.out_b != nullptr);

    // ----- Spot-check tensor shapes against the catalog -----
    // The first pre_encode conv: ne should be [3, 3, 1, channels].
    if (pe.conv0_w != nullptr) {
        CHECK_EQ_INT(pe.conv0_w->ne[0], 3);
        CHECK_EQ_INT(pe.conv0_w->ne[1], 3);
        CHECK_EQ_INT(pe.conv0_w->ne[2], 1);
        CHECK_EQ_INT(pe.conv0_w->ne[3], 4);
    }
    // The block-0 q projection should be square [d_model, d_model].
    if (!pm->weights.blocks.empty() && pm->weights.blocks[0].attn_q_w != nullptr) {
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
            std::fprintf(stderr, "FAIL pre_encode.conv0_w[0]: got %f, expected 0.0\n", v);
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
            std::fprintf(stderr, "FAIL pre_encode.out_w[0]: got %f, expected 10.0\n", v);
            ++g_failures;
        }
    } else {
        std::fprintf(stderr, "FAIL pre_encode.out_w is null\n");
        ++g_failures;
    }

    // Optional mask modes must preserve the legacy default and reject
    // misspellings rather than silently using another reference's math.
    gguf_init_params gp{};
    gp.no_alloc             = true;
    gguf_context * metadata = gguf_init_from_file(fixture.c_str(), gp);
    CHECK(metadata != nullptr);
    if (metadata != nullptr) {
        for (const char * mode : { "none", "kestrel", "transformers" }) {
            gguf_set_val_str(metadata, "stt.parakeet.encoder.length_masking", mode);
            transcribe::parakeet::ParakeetHParams parsed{};
            CHECK(transcribe::parakeet::read_parakeet_hparams(metadata, parsed) == TRANSCRIBE_OK);
            CHECK(parsed.kestrel_length_masking == (std::strcmp(mode, "none") != 0));
            CHECK(parsed.mask_after_pointwise == (std::strcmp(mode, "transformers") == 0));
        }
        gguf_set_val_str(metadata, "stt.parakeet.encoder.length_masking", "typo");
        transcribe::parakeet::ParakeetHParams parsed{};
        CHECK(transcribe::parakeet::read_parakeet_hparams(metadata, parsed) == TRANSCRIBE_ERR_GGUF);
        gguf_free(metadata);
    }

    check_pre_block(fixture);
    check_clean_mode(fixture);

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
