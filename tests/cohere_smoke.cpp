// cohere_smoke.cpp - end-to-end test for the Cohere ASR tensor ingest
// path through the public C ABI.
//
// Loads arch_cohere_minimal.gguf (a structurally complete minimal
// Cohere ASR model with the full encoder + enc_dec_proj + decoder +
// final norm + head bias tensor catalog at toy dims, filled with
// deterministic float32 data) and asserts:
//
//   1. The CohereHParams the loader read out of the GGUF KV match
//      the hparams the fixture wrote (every field in stt.cohere.* /
//      stt.frontend.*).
//   2. Every named slot in CohereWeights is non-null after load —
//      pre_encode + every encoder block sub-component + enc_dec_proj
//      + dec_embed + every decoder block sub-component + dec_final +
//      head.bias. Catches any drift between the converter's output
//      tensor names and build_cohere_weights' GET_* sequence.
//   3. The fixture data was actually copied from the GGUF data section
//      into the model's ggml_context — verified by reading the first
//      element of two known-index tensors. The fixture's _f32_seq
//      writes float(idx) as the first element of tensor i, so a wrong
//      index or a zeroed/garbage buffer trips this assertion.
//   4. ggml_tensor::ne for a couple of representative tensors matches
//      the canonical shape formulas in build_cohere_weights — same
//      drift-catching pattern as parakeet_smoke.cpp.
//   5. transcribe_model_backend() is non-empty after load (Metal on
//      Apple Silicon, CPU elsewhere or on Metal init failure).
//   6. The post-load BN fuse step ran (each block's
//      conv_bn_fused_scale and conv_bn_fused_bias are populated).
//
// The test reaches into internal headers via the test target's
// PRIVATE include of src/. Same pattern as parakeet_smoke. The public
// C ABI does not expose ggml_tensor pointers directly.
//
// Companion tests:
//   - cohere_real_smoke.cpp:  real-model gated structural test
//                             (asserts hparams match the published
//                             cohere-transcribe-03-2026 numbers).
//   - cohere_e2e_smoke.cpp:   real-model gated end-to-end transcription
//                             test (loads jfk.wav, checks transcript).

#include "transcribe.h"

#include "arch/cohere/cohere.h"
#include "arch/cohere/weights.h"
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

const transcribe::cohere::CohereModel *
cohere_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::cohere::CohereModel *>(m);
}

} // namespace

int main() {
    const std::string fixture = g_fixtures_dir + "/arch_cohere_minimal.gguf";

    if (!file_exists(fixture)) {
        std::fprintf(stderr,
                     "cohere_smoke: fixture not found: %s\n"
                     "regenerate with:\n"
                     "  cmake --build build --target fixtures\n",
                     fixture.c_str());
        return 77; // CTest "skipped"
    }

    // Force CPU backend: the fuse_batch_norm + promote_conv_pw_to_f32
    // post-load steps depend on backend selection. CPU is the only
    // backend guaranteed to be present everywhere we run this test;
    // it also exercises promote_conv_pw_to_f32_on_cpu, which is a
    // no-op on GPU backends.
    transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = TRANSCRIBE_BACKEND_CPU;
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

    // Public ABI sanity.
    if (std::strcmp(transcribe_model_arch_string(model), "cohere_asr") != 0) {
        std::fprintf(stderr, "FAIL: arch_string != cohere_asr\n");
        ++g_failures;
    }
    {
        const char * backend = transcribe_model_backend(model);
        if (backend == nullptr || backend[0] == '\0') {
            std::fprintf(stderr,
                         "FAIL: backend = \"\" after load, expected non-empty\n");
            ++g_failures;
        }
    }

    // Internal-view assertions. From here on we treat the model as a
    // CohereModel because we know the source GGUF was Cohere.
    const auto * cm = cohere_view(model);
    CHECK(cm != nullptr);
    CHECK(cm->ctx_meta != nullptr);

    // ----- Hparams -----
    // These match the values baked into make_gguf_fixtures.py
    // COHERE_HP. If either side drifts the test fails loudly.
    const auto & hp = cm->hparams;
    // Encoder.
    CHECK_EQ_INT(hp.enc_n_layers,             2);
    CHECK_EQ_INT(hp.enc_d_model,              8);
    CHECK_EQ_INT(hp.enc_n_heads,              2);
    CHECK_EQ_INT(hp.enc_d_ff,                 16);
    CHECK_EQ_INT(hp.enc_conv_kernel,          3);
    CHECK_EQ_INT(hp.enc_subsampling_factor,   2);
    CHECK_EQ_INT(hp.enc_subsampling_channels, 4);
    CHECK_EQ_INT(hp.enc_pos_emb_max_len,      32);
    CHECK_EQ_INT(hp.enc_use_bias,             1); // fixture sets true
    CHECK_EQ_INT(hp.enc_head_dim(),           4); // 8 / 2

    // Decoder.
    CHECK_EQ_INT(hp.dec_n_layers, 2);
    CHECK_EQ_INT(hp.dec_hidden,   8);
    CHECK_EQ_INT(hp.dec_n_heads,  2);
    CHECK_EQ_INT(hp.dec_inner,    16);
    CHECK_EQ_INT(hp.dec_max_seq,  32);
    CHECK_EQ_INT(hp.dec_head_dim(), 4); // 8 / 2
    CHECK_STR_EQ(hp.dec_activation, "relu");

    // Token IDs / vocab. vocab_size is derived from the tokenizer
    // (set in load() after read_cohere_hparams), so we check it
    // post-load against TOY_VOCAB length.
    CHECK_EQ_INT(hp.vocab_size,              16);
    CHECK_EQ_INT(hp.decoder_start_token_id,  1);
    CHECK_EQ_INT(hp.bos_token_id,            1);
    CHECK_EQ_INT(hp.eos_token_id,            2);

    // Head defaults (fixture sets both true).
    CHECK_EQ_INT(hp.head_log_softmax,  1);
    CHECK_EQ_INT(hp.head_tied_weights, 1);

    // Frontend.
    CHECK_STR_EQ(hp.fe_type,        "mel");
    CHECK_EQ_INT(hp.fe_num_mels,    4);
    CHECK_EQ_INT(hp.fe_sample_rate, 16000);
    CHECK_EQ_INT(hp.fe_n_fft,       16);
    CHECK_EQ_INT(hp.fe_win_length,  8);
    CHECK_EQ_INT(hp.fe_hop_length,  4);
    CHECK_STR_EQ(hp.fe_window,      "hann");
    CHECK_STR_EQ(hp.fe_normalize,   "per_feature");
    CHECK_STR_EQ(hp.fe_pad_mode,    "reflect");
    CHECK(hp.fe_dither       == 1e-5f);
    CHECK(hp.fe_pre_emphasis == 0.97f);
    CHECK(hp.fe_f_min        == 0.0f);
    CHECK(hp.fe_f_max        == 8000.0f);

    // ----- Pre-encode slots populated -----
    const auto & pe = cm->weights.pre_encode;
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
    CHECK_EQ_INT(cm->weights.blocks.size(), 2);
    for (size_t i = 0; i < cm->weights.blocks.size(); ++i) {
        const auto & b = cm->weights.blocks[i];
        // Spot-check one slot per logical sub-component to confirm
        // every category got populated. Cohere differs from Parakeet
        // by carrying FFN bias on every linear, so we explicitly
        // check those biases here.
        CHECK(b.norm_ff1_w  != nullptr);
        CHECK(b.ff1_lin1_w  != nullptr);
        CHECK(b.ff1_lin1_b  != nullptr);
        CHECK(b.ff1_lin2_w  != nullptr);
        CHECK(b.ff1_lin2_b  != nullptr);
        CHECK(b.norm_attn_w != nullptr);
        CHECK(b.attn_q_w    != nullptr);
        // attn_q_b is fused into pos_bias_u/v at load time
        // (fuse_encoder_q_bias); after fusion it MUST be null so the
        // graph builder skips the redundant add.
        CHECK(b.attn_q_b    == nullptr);
        CHECK(b.attn_k_w    != nullptr);
        CHECK(b.attn_k_b    != nullptr);
        CHECK(b.attn_v_w    != nullptr);
        CHECK(b.attn_v_b    != nullptr);
        CHECK(b.attn_out_w  != nullptr);
        CHECK(b.attn_out_b  != nullptr);
        CHECK(b.attn_pos_w  != nullptr);
        CHECK(b.attn_pos_u  != nullptr);
        CHECK(b.attn_pos_v  != nullptr);
        CHECK(b.norm_conv_w != nullptr);
        CHECK(b.conv_pw1_w  != nullptr);
        CHECK(b.conv_pw1_b  != nullptr);
        CHECK(b.conv_dw_w   != nullptr);
        CHECK(b.conv_dw_b   != nullptr);
        CHECK(b.conv_pw2_w  != nullptr);
        CHECK(b.conv_pw2_b  != nullptr);
        CHECK(b.conv_bn_w   != nullptr);
        CHECK(b.conv_bn_rm  != nullptr);
        CHECK(b.conv_bn_rv  != nullptr);
        // BN fuse runs at load: both fused tensors must be populated.
        CHECK(b.conv_bn_fused_scale != nullptr);
        CHECK(b.conv_bn_fused_bias  != nullptr);
        CHECK(b.norm_ff2_w  != nullptr);
        CHECK(b.ff2_lin1_w  != nullptr);
        CHECK(b.ff2_lin1_b  != nullptr);
        CHECK(b.ff2_lin2_w  != nullptr);
        CHECK(b.ff2_lin2_b  != nullptr);
        CHECK(b.norm_out_w  != nullptr);
    }

    // ----- Encoder-decoder projection -----
    CHECK(cm->weights.enc_dec_proj.weight != nullptr);
    CHECK(cm->weights.enc_dec_proj.bias   != nullptr);

    // ----- Decoder embedding -----
    CHECK(cm->weights.dec_embed.token_w != nullptr);
    CHECK(cm->weights.dec_embed.pos_enc != nullptr);
    CHECK(cm->weights.dec_embed.norm_w  != nullptr);
    CHECK(cm->weights.dec_embed.norm_b  != nullptr);

    // ----- Decoder block slots populated for every layer -----
    CHECK_EQ_INT(cm->weights.dec_blocks.size(), 2);
    for (size_t i = 0; i < cm->weights.dec_blocks.size(); ++i) {
        const auto & db = cm->weights.dec_blocks[i];
        // Self-attention.
        CHECK(db.norm_self_w != nullptr);
        CHECK(db.self_q_w    != nullptr);
        CHECK(db.self_q_b    != nullptr);
        CHECK(db.self_k_w    != nullptr);
        CHECK(db.self_k_b    != nullptr);
        CHECK(db.self_v_w    != nullptr);
        CHECK(db.self_v_b    != nullptr);
        CHECK(db.self_out_w  != nullptr);
        CHECK(db.self_out_b  != nullptr);
        // Cross-attention.
        CHECK(db.norm_cross_w != nullptr);
        CHECK(db.cross_q_w    != nullptr);
        CHECK(db.cross_q_b    != nullptr);
        CHECK(db.cross_k_w    != nullptr);
        CHECK(db.cross_k_b    != nullptr);
        CHECK(db.cross_v_w    != nullptr);
        CHECK(db.cross_v_b    != nullptr);
        CHECK(db.cross_out_w  != nullptr);
        CHECK(db.cross_out_b  != nullptr);
        // FFN.
        CHECK(db.norm_ff_w != nullptr);
        CHECK(db.ff_in_w   != nullptr);
        CHECK(db.ff_in_b   != nullptr);
        CHECK(db.ff_out_w  != nullptr);
        CHECK(db.ff_out_b  != nullptr);
    }

    // ----- Decoder final norm + head -----
    CHECK(cm->weights.dec_final.norm_w != nullptr);
    CHECK(cm->weights.dec_final.norm_b != nullptr);
    // head.bias is the only head tensor in the catalog when
    // head_tied_weights is true (the head weight shares storage with
    // dec_embed.token_w).
    CHECK(cm->weights.head.bias != nullptr);

    // ----- Spot-check tensor shapes against the catalog -----
    if (pe.conv0_w != nullptr) {
        CHECK_EQ_INT(pe.conv0_w->ne[0], 3);
        CHECK_EQ_INT(pe.conv0_w->ne[1], 3);
        CHECK_EQ_INT(pe.conv0_w->ne[2], 1);
        CHECK_EQ_INT(pe.conv0_w->ne[3], 4); // channels
    }
    // First encoder block's q projection: square [d_model, d_model].
    if (!cm->weights.blocks.empty() &&
        cm->weights.blocks[0].attn_q_w != nullptr)
    {
        const auto * t = cm->weights.blocks[0].attn_q_w;
        CHECK_EQ_INT(t->ne[0], 8);
        CHECK_EQ_INT(t->ne[1], 8);
        CHECK_EQ_INT(t->ne[2], 1);
        CHECK_EQ_INT(t->ne[3], 1);
    }
    // Decoder token embedding: [dec_hidden, vocab_size] in ggml.
    if (cm->weights.dec_embed.token_w != nullptr) {
        const auto * t = cm->weights.dec_embed.token_w;
        CHECK_EQ_INT(t->ne[0], hp.dec_hidden);
        CHECK_EQ_INT(t->ne[1], hp.vocab_size);
    }
    // Encoder-decoder projection: [enc_d_model, dec_hidden].
    if (cm->weights.enc_dec_proj.weight != nullptr) {
        const auto * t = cm->weights.enc_dec_proj.weight;
        CHECK_EQ_INT(t->ne[0], hp.enc_d_model);
        CHECK_EQ_INT(t->ne[1], hp.dec_hidden);
    }
    // Head bias: [vocab_size].
    if (cm->weights.head.bias != nullptr) {
        CHECK_EQ_INT(cm->weights.head.bias->ne[0], hp.vocab_size);
    }

    // ----- Spot-check that data was actually copied -----
    // The fixture writes deterministic data: tensor at descriptor
    // index i has first element exactly float(i). Pre-encode is the
    // first group, so conv.0.weight is at index 0 (first element
    // 0.0f), and conv.0.bias is at index 1 (first element 1.0f).
    //
    // The two checks together prove (a) the data section got
    // streamed into the backend buffer (zeros would fail both), (b)
    // the per-tensor offsets in the GGUF tensor info section match
    // what the loader reads (otherwise the second check would land
    // on the wrong tensor).
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
    if (pe.conv0_b != nullptr) {
        float v = 0.0f;
        ggml_backend_tensor_get(pe.conv0_b, &v, 0, sizeof(float));
        if (v != 1.0f) {
            std::fprintf(stderr,
                         "FAIL pre_encode.conv0_b[0]: got %f, expected 1.0\n", v);
            ++g_failures;
        }
    } else {
        std::fprintf(stderr, "FAIL pre_encode.conv0_b is null\n");
        ++g_failures;
    }

    // ----- Lifecycle -----
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "cohere_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "cohere_smoke: ok\n");
    return EXIT_SUCCESS;
}
