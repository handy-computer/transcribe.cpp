#pragma once

#include "transcribe.h"

#include <cstdint>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::arkasr {

struct ArkAsrHParams {
    int32_t dec_hidden       = 0;
    int32_t dec_n_layers     = 0;
    int32_t dec_n_heads      = 0;
    int32_t dec_n_kv_heads   = 0;
    int32_t dec_head_dim     = 0;
    int32_t dec_intermediate = 0;
    int32_t dec_vocab_size   = 0;
    int32_t dec_max_position = 0;
    float   dec_rope_theta   = 0.0f;
    float   dec_rms_eps      = 0.0f;

    int32_t enc_d_model      = 0;
    int32_t enc_n_layers     = 0;
    int32_t enc_n_heads      = 0;
    int32_t enc_head_dim     = 0;
    int32_t enc_ffn_dim      = 0;
    int32_t enc_num_mels     = 0;
    int32_t enc_max_position = 0;
    int32_t enc_rot_dim      = 0;
    float   enc_rope_theta   = 0.0f;
    float   enc_ln_eps       = 0.0f;

    int32_t merge_factor   = 0;
    int32_t audio_token_id = -1;
    int32_t bos_token_id   = -1;
    int32_t eos_token_id   = -1;
    int32_t pad_token_id   = -1;
    int32_t n_fft          = 0;
    int32_t hop_length     = 0;
    int32_t sample_rate    = 0;
};

struct ArkAsrEncBlock {
    ggml_tensor * attn_ln_w = nullptr;
    ggml_tensor * attn_ln_b = nullptr;
    ggml_tensor * q_w       = nullptr;
    ggml_tensor * q_b       = nullptr;
    ggml_tensor * k_w       = nullptr;
    ggml_tensor * v_w       = nullptr;
    ggml_tensor * v_b       = nullptr;
    ggml_tensor * o_w       = nullptr;
    ggml_tensor * o_b       = nullptr;
    ggml_tensor * ffn_ln_w  = nullptr;
    ggml_tensor * ffn_ln_b  = nullptr;
    ggml_tensor * fc1_w     = nullptr;
    ggml_tensor * fc1_b     = nullptr;
    ggml_tensor * fc2_w     = nullptr;
    ggml_tensor * fc2_b     = nullptr;
};

struct ArkAsrDecBlock {
    ggml_tensor * attn_norm_w   = nullptr;
    ggml_tensor * q_w           = nullptr;
    ggml_tensor * q_b           = nullptr;
    ggml_tensor * k_w           = nullptr;
    ggml_tensor * k_b           = nullptr;
    ggml_tensor * v_w           = nullptr;
    ggml_tensor * v_b           = nullptr;
    ggml_tensor * o_w           = nullptr;
    ggml_tensor * ffn_norm_w    = nullptr;
    ggml_tensor * ffn_gate_w    = nullptr;
    ggml_tensor * ffn_up_w      = nullptr;
    ggml_tensor * ffn_gate_up_w = nullptr;
    ggml_tensor * ffn_down_w    = nullptr;
};

struct ArkAsrWeights {
    ggml_tensor *               conv1_w = nullptr;
    ggml_tensor *               conv1_b = nullptr;
    ggml_tensor *               conv2_w = nullptr;
    ggml_tensor *               conv2_b = nullptr;
    std::vector<ArkAsrEncBlock> enc_blocks;

    ggml_tensor * adapter_ln_w  = nullptr;
    ggml_tensor * adapter_ln_b  = nullptr;
    ggml_tensor * adapter_fc1_w = nullptr;
    ggml_tensor * adapter_fc1_b = nullptr;
    ggml_tensor * adapter_fc2_w = nullptr;
    ggml_tensor * adapter_fc2_b = nullptr;

    ggml_tensor *               dec_embed_w = nullptr;
    ggml_tensor *               dec_norm_w  = nullptr;
    std::vector<ArkAsrDecBlock> dec_blocks;
};

transcribe_status read_arkasr_hparams(const gguf_context * gguf, ArkAsrHParams & hp);
transcribe_status build_arkasr_weights(ggml_context * ctx, const ArkAsrHParams & hp, ArkAsrWeights & weights);

}  // namespace transcribe::arkasr
