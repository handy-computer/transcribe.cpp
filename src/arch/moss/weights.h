// arch/moss/weights.h - MOSS-Transcribe-Diarize tensor catalog and hparams.
//
// INTERNAL to src/arch/moss/. Two-sided audio-LLM layout:
//
//   enc.*      Whisper-Medium encoder (2x Conv1d stem + learned positional
//              embedding + 24 pre-LN LayerNorm transformer blocks, gelu FFN;
//              q/v/out have bias, k does not). Matches the `whisper` family.
//   adaptor.*  4x-time-merge bridge: Linear(4096->1024) + SiLU +
//              Linear(1024->1024) + LayerNorm(bias). Merge itself is a reshape.
//   dec.*      Qwen3-0.6B causal LM (28 layers, GQA 16/8, head_dim 128, per-head
//              q/k RMSNorm, SwiGLU, NeoX RoPE theta 1e6, tied lm_head). Identical
//              tensor names + block math to the `qwen3_asr` text LM.
//
// Linear weights use ggml [in, out] order; Conv1d kernels are [K, IC, OC].

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::moss {

struct MossHParams {
    // Whisper encoder.
    int32_t     enc_n_layers             = 0;
    int32_t     enc_d_model              = 0;
    int32_t     enc_n_heads              = 0;
    int32_t     enc_ffn_dim              = 0;
    int32_t     enc_num_mel_bins         = 0;
    int32_t     enc_max_source_positions = 0;
    std::string enc_activation;  // "gelu"

    // VQAdaptor.
    int32_t adaptor_input_dim = 0;  // 4096 (= dec_hidden * audio_merge_size)
    int32_t audio_merge_size  = 0;  // 4

    // Qwen3 text LM.
    int32_t     dec_n_layers     = 0;
    int32_t     dec_hidden       = 0;
    int32_t     dec_intermediate = 0;
    int32_t     dec_n_heads      = 0;
    int32_t     dec_n_kv_heads   = 0;
    int32_t     dec_head_dim     = 0;
    std::string dec_hidden_act;  // "silu"
    float       dec_rms_norm_eps            = 0.0f;
    float       dec_rope_theta              = 0.0f;
    int32_t     dec_max_position_embeddings = 0;
    bool        dec_tie_word_embeddings     = true;
    int32_t     dec_vocab_size              = 0;

    // Audio-token injection + time-marker span construction (processor).
    int32_t audio_token_id            = 0;     // <|audio_pad|> (151671)
    float   audio_tokens_per_second   = 0.0f;  // 12.5
    int32_t time_marker_every_seconds = 0;     // 5
    bool    enable_time_marker        = true;

    // Baked fixed prompt (see convert-moss.py::compute_prompt_tokens).
    std::vector<int32_t> prompt_prefix_tokens;
    std::vector<int32_t> prompt_suffix_tokens;
    std::vector<int32_t> digit_tokens;  // ids for '0'..'9'

    // Token ids (resolved from tokenizer KV at load).
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t pad_token_id = -1;
    int32_t vocab_size   = 0;

    // Frontend (Whisper feature extractor).
    std::string fe_type;
    int32_t     fe_num_mels    = 0;
    int32_t     fe_sample_rate = 0;
    int32_t     fe_n_fft       = 0;
    int32_t     fe_win_length  = 0;
    int32_t     fe_hop_length  = 0;
    std::string fe_window;
    std::string fe_normalize;
    float       fe_dither       = 0.0f;
    float       fe_pre_emphasis = 0.0f;
    float       fe_f_min        = 0.0f;
    float       fe_f_max        = 0.0f;
    std::string fe_pad_mode;
    bool        fe_center = true;
    std::string fe_mel_norm;
    int32_t     fe_chunk_length  = 0;
    int32_t     fe_n_samples     = 0;
    int32_t     fe_nb_max_frames = 0;
};

transcribe_status read_moss_hparams(const gguf_context * gguf, MossHParams & hp);

// ---------------------------------------------------------------------------
// Weight slots
// ---------------------------------------------------------------------------

// Whisper 2-layer Conv1d stem.
struct MossEncStem {
    ggml_tensor * conv0_w = nullptr;  // [3, num_mel_bins, d_model]
    ggml_tensor * conv0_b = nullptr;  // [d_model]
    ggml_tensor * conv1_w = nullptr;  // [3, d_model, d_model]
    ggml_tensor * conv1_b = nullptr;  // [d_model]
};

struct MossEncTop {
    ggml_tensor * pos_emb_w    = nullptr;  // [d_model, max_source_positions]
    ggml_tensor * final_norm_w = nullptr;  // [d_model]
    ggml_tensor * final_norm_b = nullptr;  // [d_model]
};

// One Whisper pre-LN encoder block (q/v/out have bias, k does not).
struct MossEncBlock {
    ggml_tensor * norm_attn_w = nullptr;
    ggml_tensor * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr;
    ggml_tensor * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr;  // no bias
    ggml_tensor * attn_v_w    = nullptr;
    ggml_tensor * attn_v_b    = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_out_b  = nullptr;
    ggml_tensor * norm_ffn_w  = nullptr;
    ggml_tensor * norm_ffn_b  = nullptr;
    ggml_tensor * ffn_fc1_w   = nullptr;  // [d_model, ffn_dim]
    ggml_tensor * ffn_fc1_b   = nullptr;
    ggml_tensor * ffn_fc2_w   = nullptr;  // [ffn_dim, d_model]
    ggml_tensor * ffn_fc2_b   = nullptr;
};

// VQAdaptor: Linear(4096->1024) + SiLU + Linear(1024->1024) + LayerNorm.
struct MossAdaptor {
    ggml_tensor * fc1_w      = nullptr;  // [adaptor_input_dim, dec_hidden]
    ggml_tensor * fc1_b      = nullptr;  // [dec_hidden]
    ggml_tensor * fc2_w      = nullptr;  // [dec_hidden, dec_hidden]
    ggml_tensor * fc2_b      = nullptr;
    ggml_tensor * norm_out_w = nullptr;  // [dec_hidden]
    ggml_tensor * norm_out_b = nullptr;
};

struct MossDecEmbed {
    ggml_tensor * token_w = nullptr;  // [hidden, vocab_size] — tied to lm_head
};

// Qwen3 block (mirrors qwen3_asr QwenAsrDecBlock for causal_lm::BlockView).
struct MossDecBlock {
    ggml_tensor * norm_attn_w   = nullptr;
    ggml_tensor * norm_ffn_w    = nullptr;
    ggml_tensor * attn_q_w      = nullptr;
    ggml_tensor * attn_k_w      = nullptr;
    ggml_tensor * attn_v_w      = nullptr;
    ggml_tensor * attn_o_w      = nullptr;
    ggml_tensor * attn_q_norm   = nullptr;  // [head_dim]
    ggml_tensor * attn_k_norm   = nullptr;  // [head_dim]
    ggml_tensor * ffn_gate_w    = nullptr;
    ggml_tensor * ffn_up_w      = nullptr;
    ggml_tensor * ffn_down_w    = nullptr;
    ggml_tensor * ffn_gate_up_w = nullptr;  // packed at load
};

struct MossDecFinal {
    ggml_tensor * norm_w = nullptr;
};

struct MossWeights {
    MossEncStem               enc_stem;
    MossEncTop                enc_top;
    std::vector<MossEncBlock> enc_blocks;
    MossAdaptor               adaptor;

    MossDecEmbed              dec_embed;
    std::vector<MossDecBlock> dec_blocks;
    MossDecFinal              dec_final;
};

transcribe_status build_moss_weights(ggml_context * ctx_meta, const MossHParams & hp, MossWeights & weights);

}  // namespace transcribe::moss
