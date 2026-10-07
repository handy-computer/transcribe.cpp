// granite5_ctc_smoke.cpp - hermetic Granite Speech 5.0 loader smoke.
//
// Loads a complete two-block toy TurboCTC GGUF through the public C ABI,
// then checks the hparameter contract, every weight category, representative
// shapes and bytes, post-load BatchNorm fusion, backend selection, and the
// no-timestamps capability ceiling.

#include "arch/granite5_ctc/granite5_ctc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_EQ(actual, expected)                                                                               \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

const transcribe::granite5_ctc::Granite5CtcModel * g5_view(const transcribe_model * model) {
    return static_cast<const transcribe::granite5_ctc::Granite5CtcModel *>(model);
}

}  // namespace

int main() {
    const std::string fixture = std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/arch_granite5_ctc_minimal.gguf";
    if (!file_exists(fixture)) {
        std::fprintf(stderr, "granite5_ctc_smoke: fixture not found: %s\n", fixture.c_str());
        return 77;
    }

    transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    load_params.backend           = TRANSCRIBE_BACKEND_CPU;
    transcribe_model *      model = nullptr;
    const transcribe_status st    = transcribe_model_load_file(fixture.c_str(), &load_params, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    CHECK(std::strcmp(transcribe_model_arch_string(model), "granite_speech5_ctc") == 0);
    CHECK(std::strcmp(transcribe_model_variant_string(model), "granite5-ctc-toy") == 0);
    CHECK(transcribe_model_backend(model) != nullptr);
    CHECK(transcribe_model_backend(model)[0] != '\0');

    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
    CHECK_EQ(caps.native_sample_rate, 16000);
    CHECK_EQ(caps.max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_NONE);
    CHECK(!caps.supports_translate);
    CHECK(!caps.supports_language_detect);
    CHECK_EQ(caps.n_languages, 1);
    if (caps.n_languages == 1 && caps.languages != nullptr) {
        CHECK(std::strcmp(caps.languages[0], "en") == 0);
    }

    const auto * g5 = g5_view(model);
    CHECK(g5->ctx_meta != nullptr);
    const auto & hp = g5->hparams;
    CHECK_EQ(hp.enc_n_layers, 2);
    CHECK_EQ(hp.enc_hidden, 8);
    CHECK_EQ(hp.enc_n_heads, 2);
    CHECK_EQ(hp.enc_head_dim, 4);
    CHECK_EQ(hp.enc_intermediate, 16);
    CHECK_EQ(hp.enc_input_dim, 8);
    CHECK_EQ(hp.enc_output_dim, 16);
    CHECK_EQ(hp.enc_conv_kernel_size, 3);
    CHECK_EQ(hp.enc_conv_expansion, 2);
    CHECK_EQ(hp.enc_context_size, 4);
    CHECK_EQ(hp.enc_max_pos_emb, 8);
    CHECK_EQ(hp.enc_self_cond_layer, 1);
    CHECK_EQ(hp.enc_subsample_layers.size(), 1);
    if (hp.enc_subsample_layers.size() == 1) {
        CHECK_EQ(hp.enc_subsample_layers[0], 0);
    }
    CHECK_EQ(hp.blank_id, 15);
    CHECK_EQ(hp.fe_num_mels, 2);
    CHECK_EQ(hp.fe_n_fft, 16);
    CHECK_EQ(hp.fe_win_length, 8);
    CHECK_EQ(hp.fe_hop_length, 4);
    CHECK_EQ(hp.fe_delta_win_length, 3);
    CHECK_EQ(hp.fe_stack_factor, 2);
    CHECK(hp.fe_deltas);
    CHECK(hp.fe_normalize == "per_utterance");
    CHECK(hp.fe_window == "hann_periodic");
    CHECK(hp.fe_mel_norm == "htk");

    const auto & top = g5->weights.enc_top;
    CHECK(top.input_linear_w != nullptr);
    CHECK(top.input_linear_b != nullptr);
    CHECK(top.ctc_proj_w != nullptr);
    CHECK(top.ctc_proj_b != nullptr);
    CHECK(top.ctc_bypass_w != nullptr);
    CHECK(top.ctc_bypass_b != nullptr);

    CHECK_EQ(g5->weights.enc_blocks.size(), 2);
    for (const auto & block : g5->weights.enc_blocks) {
        CHECK(block.norm_ff1_w != nullptr);
        CHECK(block.norm_ff1_b != nullptr);
        CHECK(block.ff1_lin1_w != nullptr);
        CHECK(block.ff1_lin1_b != nullptr);
        CHECK(block.ff1_lin2_w != nullptr);
        CHECK(block.ff1_lin2_b != nullptr);
        CHECK(block.norm_attn_w != nullptr);
        CHECK(block.norm_attn_b != nullptr);
        CHECK(block.attn_q_w != nullptr);
        CHECK(block.attn_kv_w != nullptr);
        CHECK(block.attn_out_w != nullptr);
        CHECK(block.attn_out_b != nullptr);
        CHECK(block.attn_rel_pos_emb != nullptr);
        CHECK(block.norm_conv_w != nullptr);
        CHECK(block.norm_conv_b != nullptr);
        CHECK(block.conv_pointwise1_w != nullptr);
        CHECK(block.conv_pointwise1_b != nullptr);
        CHECK(block.conv_depthwise_w != nullptr);
        CHECK(block.conv_bn_w != nullptr);
        CHECK(block.conv_bn_b != nullptr);
        CHECK(block.conv_bn_mean != nullptr);
        CHECK(block.conv_bn_var != nullptr);
        CHECK(block.conv_pointwise2_w != nullptr);
        CHECK(block.conv_pointwise2_b != nullptr);
        CHECK(block.conv_bn_fused_scale != nullptr);
        CHECK(block.conv_bn_fused_bias != nullptr);
        CHECK(block.norm_ff2_w != nullptr);
        CHECK(block.norm_ff2_b != nullptr);
        CHECK(block.ff2_lin1_w != nullptr);
        CHECK(block.ff2_lin1_b != nullptr);
        CHECK(block.ff2_lin2_w != nullptr);
        CHECK(block.ff2_lin2_b != nullptr);
        CHECK(block.norm_out_w != nullptr);
        CHECK(block.norm_out_b != nullptr);
    }
    CHECK(g5->weights.frontend_mel_filterbank != nullptr);
    CHECK(g5->weights.frontend_window != nullptr);

    if (top.input_linear_w != nullptr) {
        CHECK_EQ(top.input_linear_w->ne[0], 8);
        CHECK_EQ(top.input_linear_w->ne[1], 8);
        float first = -1.0f;
        ggml_backend_tensor_get(top.input_linear_w, &first, 0, sizeof(first));
        CHECK(first == 0.0f);
    }
    if (!g5->weights.enc_blocks.empty()) {
        const auto * rel = g5->weights.enc_blocks[0].attn_rel_pos_emb;
        CHECK_EQ(rel->ne[0], 4);
        CHECK_EQ(rel->ne[1], 17);
        float first = -1.0f;
        ggml_backend_tensor_get(g5->weights.enc_blocks[0].norm_ff1_b, &first, 0, sizeof(first));
        CHECK(first == 7.0f);
    }
    CHECK_EQ(g5->weights.frontend_mel_filterbank->ne[0], 9);
    CHECK_EQ(g5->weights.frontend_mel_filterbank->ne[1], 2);

    transcribe_model_free(model);
    if (g_failures != 0) {
        std::fprintf(stderr, "granite5_ctc_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "granite5_ctc_smoke: ok\n");
    return EXIT_SUCCESS;
}
