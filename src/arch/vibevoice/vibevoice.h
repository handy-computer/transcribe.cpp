// arch/vibevoice/vibevoice.h - VibeVoice-ASR model/context types, hparams,
// and tensor catalog.
//
// INTERNAL to src/arch/vibevoice/. Audio-LLM: raw 24 kHz waveform -> two
// parallel causal-conv VAE encoders (acoustic vae_dim 64, semantic vae_dim
// 128) -> SpeechConnector (-> 3584) each -> element-wise SUM -> scattered
// into a Qwen2.5-7B causal LM at the speech_pad positions -> lm_head.
//
// Tensor naming mirrors scripts/convert-vibevoice.py:
//   enc.{acoustic,semantic}.*   VAE encoder (causal SConv1d stem + 6 strided
//                               downsample convs, 7 Block1D stages, head)
//   conn.{acoustic,semantic}.*  SpeechConnector (fc1 -> RMSNorm -> fc2)
//   dec.*                       Qwen2.5 LM (28 layers, GQA 28/4, q/k/v biases,
//                               plain rotate_half RoPE, untied lm_head)

#pragma once

#include "transcribe-backend.h"
#include "transcribe-session.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::vibevoice {

void apply_family_invariants(transcribe_model & model);

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

// Per-stream VAE encoder config (acoustic and semantic share structure).
struct VaeHParams {
    int32_t vae_dim   = 0;   // 64 acoustic / 128 semantic
    int32_t n_filters = 0;   // stem output channels (32); stage i has 32*2^i
    std::vector<int32_t> depths;   // parsed "3-3-3-3-3-3-8" -> blocks per stage
    float   layernorm_eps    = 1e-5f;
    float   layer_scale_init = 1e-6f;
    float   fix_std          = 0.0f;   // acoustic 0.5 (unused: we take the mean)
    std::string std_dist_type;          // "gaussian" / "none"
    std::string mixer;                  // "depthwise_conv"
    std::string layernorm;              // "RMSNorm"
    std::string pad_mode;               // "constant"
    bool    causal            = true;
    bool    disable_last_norm = true;   // final encoder norm is Identity
    bool    conv_bias         = true;
};

struct VibeVoiceHParams {
    // Qwen2.5 LM.
    int32_t dec_n_layers     = 0;
    int32_t dec_hidden       = 0;
    int32_t dec_intermediate = 0;
    int32_t dec_n_heads      = 0;
    int32_t dec_n_kv_heads   = 0;
    int32_t dec_head_dim     = 0;
    std::string dec_hidden_act;        // "silu"
    float   dec_rms_norm_eps = 0.0f;
    float   dec_rope_theta   = 0.0f;
    int32_t dec_max_position_embeddings = 0;
    bool    dec_tie_word_embeddings     = false;
    int32_t dec_vocab_size   = 0;

    // VAE encoders.
    VaeHParams acoustic;
    VaeHParams semantic;

    // Speech fusion + prompt metadata.
    int32_t speech_start_token_id    = -1;
    int32_t speech_end_token_id      = -1;
    int32_t speech_pad_token_id      = -1;
    int32_t im_start_token_id        = -1;
    int32_t im_end_token_id          = -1;
    int32_t speech_tok_compress_ratio = 0;

    // Frontend (raw 24 kHz waveform; no STFT/mel).
    int32_t frontend_sample_rate = 0;

    // Resolved from tokenizer KV at load time.
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t vocab_size   = 0;
};

transcribe_status read_vibevoice_hparams(const gguf_context * gguf,
                                         VibeVoiceHParams &   hp);

// ---------------------------------------------------------------------------
// Weight slots - VAE encoders (acoustic + semantic)
// ---------------------------------------------------------------------------

// A causal SConv1d (kernel ne = [K, in_ch, out_ch]); strided downsample convs
// derive stride = K/2 at graph-build time.
struct VaeConv {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

// One Block1D: ConvRMSNorm -> depthwise-conv mixer -> gamma -> residual,
// then ConvRMSNorm -> FFN(linear1 -> GELU -> linear2) -> ffn_gamma -> residual.
struct VaeBlock {
    ggml_tensor * norm_w     = nullptr;  // ConvRMSNorm (mixer path)
    VaeConv       mixer;                  // depthwise conv (groups=C, k=7)
    ggml_tensor * gamma      = nullptr;   // per-channel layer scale [C]
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_lin1_w = nullptr;   // [C, 4C]
    ggml_tensor * ffn_lin1_b = nullptr;
    ggml_tensor * ffn_lin2_w = nullptr;   // [4C, C]
    ggml_tensor * ffn_lin2_b = nullptr;
    ggml_tensor * ffn_gamma  = nullptr;   // [C]
};

struct VaeEncoderWeights {
    std::vector<VaeConv>                downsample;  // [0]=stem, [1..]=strided
    std::vector<std::vector<VaeBlock>>  stages;      // per stage, per block
    VaeConv                             head;        // SConv1d -> vae_dim
    // final encoder norm is Identity (disable_last_norm) -> no slot
};

// SpeechConnector: fc1 (vae_dim -> hidden) -> RMSNorm(hidden) -> fc2.
struct ConnectorWeights {
    ggml_tensor * fc1_w  = nullptr;
    ggml_tensor * fc1_b  = nullptr;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * fc2_w  = nullptr;
    ggml_tensor * fc2_b  = nullptr;
};

// ---------------------------------------------------------------------------
// Weight slots - Qwen2.5 LM
// ---------------------------------------------------------------------------

struct VibeVoiceDecBlock {
    ggml_tensor * norm_attn_w = nullptr;  // input_layernorm (RMSNorm)
    // Qwen2: q/k/v carry biases; o_proj has no bias.
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * attn_o_w = nullptr;
    ggml_tensor * norm_ffn_w = nullptr;   // post_attention_layernorm
    ggml_tensor * ffn_gate_w = nullptr;
    ggml_tensor * ffn_up_w   = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
};

struct VibeVoiceWeights {
    VaeEncoderWeights enc_acoustic;
    VaeEncoderWeights enc_semantic;
    ConnectorWeights  conn_acoustic;
    ConnectorWeights  conn_semantic;

    ggml_tensor * dec_token_embd = nullptr;   // [hidden, vocab]
    std::vector<VibeVoiceDecBlock> dec_blocks;
    ggml_tensor * dec_output_norm = nullptr;  // RMSNorm before lm_head
    ggml_tensor * dec_output      = nullptr;  // untied lm_head [hidden, vocab]
};

transcribe_status build_vibevoice_weights(ggml_context *           ctx_meta,
                                          const VibeVoiceHParams & hp,
                                          VibeVoiceWeights &       weights);

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

struct VibeVoiceModel final : public transcribe_model {
    Tokenizer        tok;
    VibeVoiceHParams hparams;
    VibeVoiceWeights weights;
    ggml_context *   ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    std::string chat_template;

    VibeVoiceModel() = default;
    ~VibeVoiceModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct VibeVoiceSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    std::vector<float> enc_host;  // VAE-combined features, host-side

    VibeVoiceSession() = default;
    ~VibeVoiceSession() override;
};

} // namespace transcribe::vibevoice
