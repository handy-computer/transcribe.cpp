// granite5_ctc_real_smoke.cpp - real-model gated test for the Granite
// Speech 5.0 TurboCTC converter + ingest pipeline.
//
// Loads a real granite5_ctc GGUF (the output of
// scripts/convert-granite5_ctc.py against the HF checkpoint) and verifies
// that the loader produces a fully populated Granite5CtcModel.
//
// Gating:
//
//   - The CMake option TRANSCRIBE_BUILD_REAL_MODEL_TESTS (default OFF)
//     controls whether this binary is even built.
//   - At runtime, the GGUF path comes from the
//     TRANSCRIBE_GRANITE5_CTC_GGUF environment variable. If unset, the
//     test exits 77 (CTest "skipped") with a regeneration hint.
//
// CI never builds this — it's a developer-local manual gate.
//
// What we assert:
//
//   1. Load returns OK; the model pointer is set.
//   2. arch_string == "granite_speech5_ctc", variant string matches the
//      one variant the converter produces.
//   3. backend is non-empty after load.
//   4. native_sample_rate == 16000, supports_translate == false,
//      lang_detect == false, 1 language ["en"] (family invariants: this
//      is a monolingual English CTC encoder).
//   5. The hparams the loader read match the 470M variant:
//        enc_n_layers=16, enc_hidden=1024, enc_n_heads=8,
//        enc_head_dim=128, enc_intermediate=4096, enc_input_dim=320,
//        enc_output_dim=16384, enc_conv_kernel_size=7,
//        enc_conv_expansion=2, enc_context_size=128,
//        enc_max_pos_emb=512, enc_self_cond_layer=8,
//        fe_num_mels=80, fe_sample_rate=16000, fe_n_fft=512,
//        fe_win_length=400, fe_hop_length=160.
//   6. enc_subsample_layers == [0, 1] — the two in-block stride-2 blocks.
//      This is the structural distinctive of this family; getting it
//      wrong silently changes the output frame rate.
//   7. Total tensor count is 520 = 8 + 16*32.
//   8. Spot-check shapes, chosen to cover the pieces that are NOT shared
//      with granite 4.x:
//        input_linear_w  ne=[320, 1024]   (stacked mel+delta frontend)
//        ctc_proj_w      ne=[1024, 16384] (tied CTC head)
//        ctc_bypass_w    ne=[16384, 1024] (self-conditioning injection)
//        enc_blocks[0].attn_kv_w        ne=[1024, 2048]  (fused K|V)
//        enc_blocks[0].attn_rel_pos_emb ne=[128, 1025]   (Shaw, 2*512+1)
//        enc_blocks[0].conv_depthwise_w ne=[7, 1, 2048]
//        enc_blocks[0].conv_pointwise1_w ne=[1024, 4096] (2-D nn.Linear here,
//          NOT granite 4.x's 3-D [1, in, out] Conv1d layout — this shape
//          is what lets the Stage 5 quantizer treat it as a Linear)

#include "arch/granite5_ctc/granite5_ctc.h"
#include "ggml.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
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

const transcribe::granite5_ctc::Granite5CtcModel * g5_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::granite5_ctc::Granite5CtcModel *>(m);
}

// Shape check against an explicit ne[] list. Trailing dims must be 1.
void check_shape(const char * label, const ggml_tensor * t, const std::vector<int64_t> & want) {
    if (t == nullptr) {
        std::fprintf(stderr, "FAIL shape %s: tensor is null\n", label);
        ++g_failures;
        return;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t w = (i < static_cast<int>(want.size())) ? want[static_cast<size_t>(i)] : 1;
        if (t->ne[i] != w) {
            std::fprintf(stderr, "FAIL shape %s: ne[%d] = %lld, expected %lld\n", label, i,
                         static_cast<long long>(t->ne[i]), static_cast<long long>(w));
            ++g_failures;
        }
    }
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_GRANITE5_CTC_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "granite5_ctc_real_smoke: TRANSCRIBE_GRANITE5_CTC_GGUF "
                     "not set; skipping. Convert a real model with:\n"
                     "  uv run --project scripts/envs/granite5_ctc \\\n"
                     "    scripts/convert-granite5_ctc.py "
                     "ibm-granite/granite-speech-5.0-470m-turboctc \\\n"
                     "    --repo-id ibm-granite/granite-speech-5.0-470m-turboctc\n"
                     "and re-run with TRANSCRIBE_GRANITE5_CTC_GGUF=models/"
                     "granite-speech-5.0-470m-turboctc/"
                     "granite-speech-5.0-470m-turboctc-BF16.gguf\n");
        return 77;
    }
    const std::string fixture = env;

    if (!file_exists(fixture)) {
        std::fprintf(stderr, "granite5_ctc_real_smoke: file not found: %s\n", fixture.c_str());
        return 77;
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

    // --- 2/3: public ABI sanity ---------------------------------------
    CHECK_STR_EQ(transcribe_model_arch_string(model), "granite_speech5_ctc");

    // Two variants share this arch and this loader. They are identical in
    // config and tensor names; only the tokenizer, licence and training data
    // differ, so every structural assertion below applies to both.
    const std::string variant = transcribe_model_variant_string(model);
    const bool        is_nc   = (variant == "granite-speech-5.0-470m-turboctc-nc");
    if (variant != "granite-speech-5.0-470m-turboctc" && !is_nc) {
        std::fprintf(stderr, "FAIL: unexpected variant string \"%s\"\n", variant.c_str());
        ++g_failures;
    }
    {
        const std::string backend = transcribe_model_backend(model);
        if (backend.empty()) {
            std::fprintf(stderr, "FAIL: backend = \"\" after load, expected non-empty\n");
            ++g_failures;
        }
    }

    // --- 4: family invariants -----------------------------------------
    transcribe_capabilities caps_buf;
    transcribe_capabilities_init(&caps_buf);
    const bool                      caps_ok = transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps    = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(caps->native_sample_rate == 16000);
        CHECK(caps->supports_translate == false);
        CHECK(caps->supports_language_detect == false);
        CHECK_EQ_INT(caps->n_languages, 1);
        if (caps->n_languages == 1 && caps->languages != nullptr) {
            CHECK_STR_EQ(caps->languages[0], "en");
        }
    }

    // --- 5/6: hparams the loader read ---------------------------------
    const auto * g5 = g5_view(model);
    const auto & hp = g5->hparams;

    CHECK_EQ_INT(hp.enc_n_layers, 16);
    CHECK_EQ_INT(hp.enc_hidden, 1024);
    CHECK_EQ_INT(hp.enc_n_heads, 8);
    CHECK_EQ_INT(hp.enc_head_dim, 128);
    CHECK_EQ_INT(hp.enc_intermediate, 4096);
    CHECK_EQ_INT(hp.enc_input_dim, 320);
    CHECK_EQ_INT(hp.enc_output_dim, 16384);
    CHECK_EQ_INT(hp.enc_conv_kernel_size, 7);
    CHECK_EQ_INT(hp.enc_conv_expansion, 2);
    CHECK_EQ_INT(hp.enc_context_size, 128);
    CHECK_EQ_INT(hp.enc_max_pos_emb, 512);
    CHECK_EQ_INT(hp.enc_self_cond_layer, 8);

    CHECK_EQ_INT(hp.fe_num_mels, 80);
    CHECK_EQ_INT(hp.fe_sample_rate, 16000);
    CHECK_EQ_INT(hp.fe_n_fft, 512);
    CHECK_EQ_INT(hp.fe_win_length, 400);
    CHECK_EQ_INT(hp.fe_hop_length, 160);

    // The structural distinctive of this family: blocks 0 and 1 halve the
    // time axis in-block. Wrong here means the wrong output frame rate.
    CHECK_EQ_INT(hp.enc_subsample_layers.size(), 2);
    if (hp.enc_subsample_layers.size() == 2) {
        CHECK_EQ_INT(hp.enc_subsample_layers[0], 0);
        CHECK_EQ_INT(hp.enc_subsample_layers[1], 1);
    }

    // --- 7: tensor count ----------------------------------------------
    // 520 = 8 non-block + 16 blocks * 32 per block.
    {
        int n_tensors = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(g5->ctx_meta); t != nullptr;
             t               = ggml_get_next_tensor(g5->ctx_meta, t)) {
            ++n_tensors;
        }
        CHECK_EQ_INT(n_tensors, 520);
    }

    // --- 8: spot-check shapes -----------------------------------------
    const auto & w = g5->weights;
    check_shape("enc_top.input_linear_w", w.enc_top.input_linear_w, { 320, 1024 });
    check_shape("enc_top.ctc_proj_w", w.enc_top.ctc_proj_w, { 1024, 16384 });
    check_shape("enc_top.ctc_bypass_w", w.enc_top.ctc_bypass_w, { 16384, 1024 });

    CHECK_EQ_INT(w.enc_blocks.size(), 16);
    if (!w.enc_blocks.empty()) {
        const auto & b = w.enc_blocks[0];
        check_shape("enc_blocks[0].attn_kv_w", b.attn_kv_w, { 1024, 2048 });
        // Shaw table is 2 * max_pos_emb + 1 rows.
        check_shape("enc_blocks[0].attn_rel_pos_emb", b.attn_rel_pos_emb, { 128, 1025 });
        check_shape("enc_blocks[0].conv_depthwise_w", b.conv_depthwise_w, { 7, 1, 2048 });
        // 2-D here, unlike granite 4.x's [1, in, out] Conv1d layout. This
        // shape is load-bearing: policy.cpp::classify_tensor uses ne0 to
        // decide the pointwise bucket, so a regression to the 3-D layout
        // would silently pin 201 MB at F16 in every quant tier.
        check_shape("enc_blocks[0].conv_pointwise1_w", b.conv_pointwise1_w, { 1024, 4096 });
    }

    // --- 9: tokenizer decode contract ---------------------------------
    //
    // The two variants ship different tokenizer FAMILIES behind identical
    // configs: byte-level BPE (Apache) vs SentencePiece-derived BPE with byte
    // fallback (-nc). The -nc side has two decode paths that the byte-level
    // side cannot reach at all, and both were signed MUST PASS at Stage 1:
    // byte-fallback reassembly, and the 254 reserved <|tokN|> placeholders.
    //
    // Expected strings below are the reference ParakeetTokenizer's own output,
    // captured at Stage 2 in
    //   build/validate/granite5_ctc/<variant>/tokenizer_decode_oracle.json
    // via tokenizer.decode(ids, skip_special_tokens=True).
    //
    // Every sequence is free of ADJACENT DUPLICATE ids on purpose:
    // ParakeetTokenizer._decode performs the CTC run-length collapse itself,
    // so an already-collapsed sequence containing [A, A] (legal, from a
    // [A, blank, A] input) would collapse twice in the reference and record a
    // wrong expectation. With no adjacent duplicates the collapse is a no-op
    // and the oracle is pure detokenization -- exactly what Tokenizer::decode
    // must reproduce.
    {
        const transcribe::Tokenizer * tok = g5->tokenizer();
        CHECK(tok != nullptr);
        if (tok != nullptr) {
            auto decode = [&](const std::vector<int> & ids) {
                return tok->decode(ids.data(), static_cast<int>(ids.size()));
            };
            if (is_nc) {
                // byte fallback: <0xNN> ids are 255 + byte value.
                // "naïve" -> 0xC3 0xAF for the ï.
                CHECK_STR_EQ(decode({ 7942, 255 + 0xC3, 255 + 0xAF, 561 }), " naïve");
                // 3-byte UTF-8 through byte fallback: U+6F22 -> E6 BC A2.
                CHECK_STR_EQ(decode({ 515, 1114, 255 + 0xE6, 255 + 0xBC, 255 + 0xA2 }), " the word漢");
                // 4-byte UTF-8: U+1F600 -> F0 9F 98 80.
                CHECK_STR_EQ(decode({ 1515, 255 + 0xF0, 255 + 0x9F, 255 + 0x98, 255 + 0x80 }), " ok😀");
                // Reserved placeholders decode to their LITERAL text. The
                // reference suppresses nothing, so neither may we.
                CHECK_STR_EQ(decode({ 515, 1, 2288 }), " the<|tok0|> dog");
                CHECK_STR_EQ(decode({ 1, 2, 3 }), "<|tok0|><|tok1|><|tok2|>");
                // Plain words: U+2581 becomes a leading ASCII space. The
                // reference strips exactly one; build_result's
                // collapse_whitespace does that for full_text, so the raw
                // decode legitimately keeps it here.
                CHECK_STR_EQ(decode({ 515, 1773, 2977, 5384 }), " the quick brown fox");
            } else {
                CHECK_STR_EQ(decode({ 406, 1731, 2855, 5723 }), "the quick brown fox");
            }
        }
    }

    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "granite5_ctc_real_smoke: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("granite5_ctc_real_smoke: OK\n");
    return EXIT_SUCCESS;
}
