// qwen3_asr_smoke.cpp - default-ctest loader smoke for the Qwen3-ASR
// family.
//
// Loads arch_qwen3_asr_minimal.gguf (a structurally complete minimal
// Qwen3-ASR audio-LLM: 3x Conv2d subsampler + 2 bidirectional encoder
// blocks + encoder head (LN + proj1 + proj2) + 2 Qwen3 decoder blocks
// (GQA 2/1 + per-head Q/K-RMSNorm + SwiGLU) + final RMSNorm, tied
// lm_head, gpt2-style BPE tokenizer, chat-template string, 30-lang
// capability) and asserts:
//
//   1. QwenAsrHParams read out of stt.qwen3_asr.* / stt.frontend.*
//      match the fixture's pinned numeric values.
//   2. Every named slot in QwenAsrWeights is non-null after load —
//      encoder subsample + every encoder block + encoder head +
//      decoder embed + every decoder block + final norm. Catches
//      drift between the converter's tensor names and the loader's
//      GET_* sequence.
//   3. The fixture data was actually copied from the GGUF data
//      section: the first element of tensor index i is float(i)
//      (see _f32_seq in make_gguf_fixtures.py).
//   4. ggml_tensor::ne for representative tensors matches the
//      catalog shape formulas in build_qwen3_asr_weights.
//   5. The Phase 1.6 resolve_chat_tokens() call succeeded at load —
//      every chat-template piece ended up in the resolved
//      ChatTokens struct with a valid id.
//   6. The ffn_gate_up packed weight was populated at load (model.cpp
//      does a CPU round-trip to combine ffn_gate + ffn_up into one
//      matmul).
//   7. Capabilities read from the fixture: supports_language_detect
//      true, 4-language list round-trips, supports_translate false
//      from apply_family_invariants.
//   8. transcribe_model_backend() is non-empty after load.
//
// This test does NOT call transcribe_run — the toy weights are
// deterministic padding, not a valid model, and the decode graph
// would produce nonsense even if it worked structurally. Run-path
// coverage lives in qwen3_asr_real_smoke.cpp (env-gated, needs the
// actual HF checkpoint converted to GGUF).
//
// Companion tests:
//   - qwen3_asr_real_smoke_{0_6b,1_7b}.cpp: env-gated structural tests
//     against the actual converted GGUFs.
//   - qwen3_asr_e2e_smoke.cpp:              env-gated public-ABI
//                                           transcription round-trip.

#include "transcribe.h"

#include "arch/qwen3_asr/qwen3_asr.h"
#include "arch/qwen3_asr/weights.h"
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

const transcribe::qwen3_asr::QwenAsrModel *
qwen_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::qwen3_asr::QwenAsrModel *>(m);
}

} // namespace

int main() {
    const std::string fixture = g_fixtures_dir + "/arch_qwen3_asr_minimal.gguf";

    if (!file_exists(fixture)) {
        std::fprintf(stderr,
                     "qwen3_asr_smoke: fixture not found: %s\n"
                     "regenerate with:\n"
                     "  cmake --build build --target fixtures\n",
                     fixture.c_str());
        return 77; // CTest "skipped"
    }

    // Force CPU backend: ffn_gate_up packing at load does a CPU
    // round-trip (ggml_backend_tensor_get + _set). Keeping the test on
    // CPU keeps the dependency narrow — no Metal/Vulkan runtime
    // required for default ctest.
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
    if (std::strcmp(transcribe_model_arch_string(model), "qwen3_asr") != 0) {
        std::fprintf(stderr, "FAIL: arch_string != qwen3_asr\n");
        ++g_failures;
    }
    if (std::strcmp(transcribe_model_variant_string(model),
                    "qwen3-asr-toy") != 0)
    {
        std::fprintf(stderr, "FAIL: variant_string != qwen3-asr-toy\n");
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

    // Capabilities: advertised language list + detection bool round-
    // trip; translate stays false (family invariant); max_timestamp
    // stays NONE (first port doesn't emit alignment).
    {
        const auto * caps = transcribe_model_capabilities(model);
        CHECK(caps != nullptr);
        CHECK_EQ_INT(caps->native_sample_rate, 16000);
        CHECK(caps->supports_translate == false);
        CHECK(caps->supports_language_detect == true);
        CHECK_EQ_INT(caps->n_languages, 4);
        CHECK(caps->languages != nullptr);
        CHECK_EQ_INT(caps->max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_NONE);
    }

    // Internal-view assertions.
    const auto * qm = qwen_view(model);
    CHECK(qm != nullptr);
    CHECK(qm->ctx_meta != nullptr);
    CHECK(qm->mel.has_value());

    // ----- Hparams match the fixture -----
    const auto & hp = qm->hparams;
    CHECK_EQ_INT(hp.enc_n_layers,              2);
    CHECK_EQ_INT(hp.enc_d_model,               16);
    CHECK_EQ_INT(hp.enc_n_heads,               2);
    CHECK_EQ_INT(hp.enc_ffn_dim,               32);
    CHECK_EQ_INT(hp.enc_num_mel_bins,          8);
    CHECK_EQ_INT(hp.enc_downsample_hidden,     16);
    CHECK_EQ_INT(hp.enc_output_dim,            16);
    CHECK_EQ_INT(hp.enc_max_source_positions,  64);
    CHECK_EQ_INT(hp.enc_n_window,              2);
    CHECK_EQ_INT(hp.enc_n_window_infer,        4);
    CHECK_EQ_INT(hp.enc_conv_chunksize,        8);
    CHECK_STR_EQ(hp.enc_activation,            "gelu");

    CHECK_EQ_INT(hp.dec_n_layers,                2);
    CHECK_EQ_INT(hp.dec_hidden,                  16);
    CHECK_EQ_INT(hp.dec_intermediate,            32);
    CHECK_EQ_INT(hp.dec_n_heads,                 2);
    CHECK_EQ_INT(hp.dec_n_kv_heads,              1);
    CHECK_EQ_INT(hp.dec_head_dim,                8);
    CHECK_STR_EQ(hp.dec_hidden_act,              "silu");
    CHECK_EQ_INT(hp.dec_rope_mrope_section_t,    2);
    CHECK_EQ_INT(hp.dec_rope_mrope_section_h,    1);
    CHECK_EQ_INT(hp.dec_rope_mrope_section_w,    1);
    CHECK(hp.dec_rope_mrope_interleaved == true);
    CHECK_EQ_INT(hp.dec_max_position_embeddings, 128);
    CHECK(hp.dec_tie_word_embeddings == true);
    CHECK_EQ_INT(hp.dec_vocab_size,              32);

    CHECK_EQ_INT(hp.audio_start_token_id, 16);
    CHECK_EQ_INT(hp.audio_end_token_id,   17);
    CHECK_EQ_INT(hp.audio_token_id,       18);

    // Frontend: a subset of the full block; the full read is exercised
    // by the MelConfig passthrough inside load(). Just confirm the
    // hparams reached this struct.
    CHECK_STR_EQ(hp.fe_type,      "mel");
    CHECK_EQ_INT(hp.fe_num_mels,  8);
    CHECK_EQ_INT(hp.fe_sample_rate, 16000);
    CHECK_STR_EQ(hp.fe_window,    "hann_periodic");
    CHECK_STR_EQ(hp.fe_normalize, "per_utterance");

    // Tokenizer vocab size drives hp.vocab_size + the dec_vocab_size
    // cross-check. All must agree.
    CHECK_EQ_INT(hp.vocab_size, 32);
    CHECK_EQ_INT(hp.eos_token_id, 3);

    // ----- Chat template string round-tripped -----
    CHECK(!qm->chat_template.empty());

    // ----- Phase 1.6: chat-template token ids resolved at load -----
    {
        const auto & ct = qm->chat_tokens;
        CHECK_EQ_INT(ct.im_start,        2);
        CHECK_EQ_INT(ct.im_end,          3);
        CHECK_EQ_INT(ct.newline,         4);
        CHECK_EQ_INT(ct.role_system,     5);
        CHECK_EQ_INT(ct.role_user,       6);
        CHECK_EQ_INT(ct.role_assistant,  7);
    }

    // ----- Encoder subsample slots populated -----
    const auto & es = qm->weights.enc_subsample;
    CHECK(es.conv0_w  != nullptr);
    CHECK(es.conv0_b  != nullptr);
    CHECK(es.conv1_w  != nullptr);
    CHECK(es.conv1_b  != nullptr);
    CHECK(es.conv2_w  != nullptr);
    CHECK(es.conv2_b  != nullptr);
    CHECK(es.conv_out != nullptr);

    // ----- Encoder block slots populated for every layer -----
    CHECK_EQ_INT(qm->weights.enc_blocks.size(), 2);
    for (size_t i = 0; i < qm->weights.enc_blocks.size(); ++i) {
        const auto & b = qm->weights.enc_blocks[i];
        CHECK(b.norm_attn_w != nullptr);
        CHECK(b.norm_attn_b != nullptr);
        CHECK(b.attn_q_w    != nullptr);
        CHECK(b.attn_q_b    != nullptr);
        CHECK(b.attn_k_w    != nullptr);
        CHECK(b.attn_k_b    != nullptr);
        CHECK(b.attn_v_w    != nullptr);
        CHECK(b.attn_v_b    != nullptr);
        CHECK(b.attn_out_w  != nullptr);
        CHECK(b.attn_out_b  != nullptr);
        CHECK(b.norm_ffn_w  != nullptr);
        CHECK(b.norm_ffn_b  != nullptr);
        CHECK(b.fc1_w       != nullptr);
        CHECK(b.fc1_b       != nullptr);
        CHECK(b.fc2_w       != nullptr);
        CHECK(b.fc2_b       != nullptr);
    }

    // ----- Encoder head slots populated -----
    const auto & eh = qm->weights.enc_head;
    CHECK(eh.ln_post_w != nullptr);
    CHECK(eh.ln_post_b != nullptr);
    CHECK(eh.proj1_w   != nullptr);
    CHECK(eh.proj1_b   != nullptr);
    CHECK(eh.proj2_w   != nullptr);
    CHECK(eh.proj2_b   != nullptr);

    // ----- Decoder embed (tied to lm_head) + final norm -----
    CHECK(qm->weights.dec_embed.token_w != nullptr);
    CHECK(qm->weights.dec_final.norm_w  != nullptr);

    // ----- Decoder block slots populated for every layer -----
    CHECK_EQ_INT(qm->weights.dec_blocks.size(), 2);
    for (size_t i = 0; i < qm->weights.dec_blocks.size(); ++i) {
        const auto & b = qm->weights.dec_blocks[i];
        CHECK(b.norm_attn_w != nullptr);
        CHECK(b.norm_ffn_w  != nullptr);
        CHECK(b.attn_q_w    != nullptr);
        CHECK(b.attn_k_w    != nullptr);
        CHECK(b.attn_v_w    != nullptr);
        CHECK(b.attn_o_w    != nullptr);
        CHECK(b.attn_q_norm != nullptr);
        CHECK(b.attn_k_norm != nullptr);
        CHECK(b.ffn_gate_w  != nullptr);
        CHECK(b.ffn_up_w    != nullptr);
        CHECK(b.ffn_down_w  != nullptr);
        // Packed gate+up: filled by the post-load packing step in
        // model.cpp. Must be non-null if the step ran.
        CHECK(b.ffn_gate_up_w != nullptr);
    }

    // ----- Spot-check tensor shapes against the catalog -----
    if (es.conv0_w != nullptr) {
        CHECK_EQ_INT(es.conv0_w->ne[0], 3);
        CHECK_EQ_INT(es.conv0_w->ne[1], 3);
        CHECK_EQ_INT(es.conv0_w->ne[2], 1);
        CHECK_EQ_INT(es.conv0_w->ne[3], hp.enc_downsample_hidden);
    }
    if (es.conv_out != nullptr) {
        // ne = [conv_out_in, d_model] = [ds_h * mel_ds3, d_model].
        // mel_ds chain for num_mels=8: 4 -> 2 -> 1; conv_out_in = 16.
        CHECK_EQ_INT(es.conv_out->ne[0], 16);
        CHECK_EQ_INT(es.conv_out->ne[1], hp.enc_d_model);
    }
    if (qm->weights.dec_embed.token_w != nullptr) {
        const auto * t = qm->weights.dec_embed.token_w;
        CHECK_EQ_INT(t->ne[0], hp.dec_hidden);
        CHECK_EQ_INT(t->ne[1], hp.dec_vocab_size);
    }
    if (!qm->weights.dec_blocks.empty()) {
        const auto & b0 = qm->weights.dec_blocks[0];
        if (b0.attn_q_w != nullptr) {
            // GQA: q is [hidden, n_heads * head_dim] = [16, 16].
            CHECK_EQ_INT(b0.attn_q_w->ne[0], hp.dec_hidden);
            CHECK_EQ_INT(b0.attn_q_w->ne[1], hp.dec_n_heads * hp.dec_head_dim);
        }
        if (b0.attn_k_w != nullptr) {
            // GQA: k is [hidden, n_kv_heads * head_dim] = [16, 8].
            CHECK_EQ_INT(b0.attn_k_w->ne[0], hp.dec_hidden);
            CHECK_EQ_INT(b0.attn_k_w->ne[1], hp.dec_n_kv_heads * hp.dec_head_dim);
        }
        if (b0.ffn_gate_up_w != nullptr) {
            // Packed: [hidden, 2 * intermediate].
            CHECK_EQ_INT(b0.ffn_gate_up_w->ne[0], hp.dec_hidden);
            CHECK_EQ_INT(b0.ffn_gate_up_w->ne[1], 2 * hp.dec_intermediate);
        }
    }

    // ----- Spot-check that data was actually copied -----
    // Descriptor order: the first 7 slots are encoder subsample, so
    // enc.conv.0.weight is at index 0 (first element 0.0f) and
    // enc.conv.0.bias is at index 1 (first element 1.0f).
    if (es.conv0_w != nullptr) {
        float v = 0.0f;
        ggml_backend_tensor_get(es.conv0_w, &v, 0, sizeof(float));
        if (v != 0.0f) {
            std::fprintf(stderr,
                         "FAIL es.conv0_w[0]: got %f, expected 0.0\n", v);
            ++g_failures;
        }
    }
    if (es.conv0_b != nullptr) {
        float v = 0.0f;
        ggml_backend_tensor_get(es.conv0_b, &v, 0, sizeof(float));
        if (v != 1.0f) {
            std::fprintf(stderr,
                         "FAIL es.conv0_b[0]: got %f, expected 1.0\n", v);
            ++g_failures;
        }
    }

    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "qwen3_asr_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "qwen3_asr_smoke: ok\n");
    return EXIT_SUCCESS;
}
