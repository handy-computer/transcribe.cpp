// arch/sortformer/weights.h - Sortformer hparams + the Sortformer-specific
// weight slots (encoder_proj + the 18-layer post-LN Transformer encoder +
// the diarization sigmoid head). The 17-layer NEST/FastConformer encoder is
// NOT here: it reuses parakeet::ParakeetWeights (pre_encode + blocks),
// loaded separately in the sortformer loader.
//
// Tensor layout conventions (matched by scripts/convert-sortformer.py):
//   - Linear weights: PyTorch [out, in] -> ggml ne [in, out].
//   - LayerNorm: separate weight + bias, shape [d_model].

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::sortformer {

struct SortformerHParams {
    // Diarization.
    int32_t max_speakers = 4;   // hard architectural cap
    int32_t frame_hop    = 0;   // samples per diar frame (hop * subsampling)

    // Conformer encoder (mirrors the parakeet fields; see convert KVs).
    int32_t enc_n_layers             = 0;
    int32_t enc_d_model              = 0;
    int32_t enc_n_heads              = 0;
    int32_t enc_d_ff                 = 0;
    int32_t enc_conv_kernel          = 0;
    int32_t enc_subsampling_factor   = 0;
    int32_t enc_subsampling_channels = 0;
    int32_t enc_feat_in              = 0;
    int32_t enc_pos_emb_max_len      = 0;
    std::string enc_conv_norm_type;  // "batch_norm"

    // Transformer encoder (post-LN, no positional encoding, no final norm).
    int32_t     tf_n_layers = 0;
    int32_t     tf_d_model  = 0;
    int32_t     tf_n_heads  = 0;
    int32_t     tf_d_ff     = 0;
    std::string tf_activation;  // "relu"
    bool        tf_pre_ln = false;

    // Frontend (mel feature extractor).
    int32_t     fe_num_mels    = 0;
    int32_t     fe_sample_rate = 0;
    int32_t     fe_n_fft       = 0;
    int32_t     fe_win_length  = 0;
    int32_t     fe_hop_length  = 0;
    std::string fe_window;     // "hann"
    std::string fe_normalize;  // "none"
    float       fe_dither       = 0.0f;
    float       fe_pre_emphasis = 0.0f;

    // Shipped streaming defaults (deferred streaming path; read for KV
    // completeness and later use).
    int32_t stream_chunk_len             = 0;
    int32_t stream_spkcache_len          = 0;
    int32_t stream_fifo_len              = 0;
    int32_t stream_spkcache_update_period = 0;

    int32_t tf_head_dim() const { return tf_n_heads > 0 ? tf_d_model / tf_n_heads : 0; }
};

// One post-LN Transformer encoder block. All linears carry bias.
struct SortformerTfBlock {
    ggml_tensor * norm1_w = nullptr;  // [d]
    ggml_tensor * norm1_b = nullptr;  // [d]
    ggml_tensor * attn_q_w = nullptr; // [d, d]
    ggml_tensor * attn_q_b = nullptr; // [d]
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * attn_o_w = nullptr;
    ggml_tensor * attn_o_b = nullptr;
    ggml_tensor * norm2_w = nullptr;  // [d]
    ggml_tensor * norm2_b = nullptr;  // [d]
    ggml_tensor * ff_in_w  = nullptr; // [d, d_ff]
    ggml_tensor * ff_in_b  = nullptr; // [d_ff]
    ggml_tensor * ff_out_w = nullptr; // [d_ff, d]
    ggml_tensor * ff_out_b = nullptr; // [d]
};

struct SortformerWeights {
    // encoder_proj: Linear 512 -> 192.
    ggml_tensor * enc_proj_w = nullptr;  // [enc_d_model, tf_d_model]
    ggml_tensor * enc_proj_b = nullptr;  // [tf_d_model]

    std::vector<SortformerTfBlock> tf_blocks;  // tf_n_layers entries

    // Diar head (forward_speaker_sigmoids offline path).
    ggml_tensor * fc1_w = nullptr;         // [tf_d_model, tf_d_model]
    ggml_tensor * fc1_b = nullptr;         // [tf_d_model]
    ggml_tensor * single_spk_head_w = nullptr;  // [tf_d_model, max_speakers]
    ggml_tensor * single_spk_head_b = nullptr;  // [max_speakers]
    // Streaming 2*hidden head (spkcache concat path); loaded but unused offline.
    ggml_tensor * spk_head_w = nullptr;    // [2*tf_d_model, max_speakers]
    ggml_tensor * spk_head_b = nullptr;    // [max_speakers]
};

// Read every required stt.sortformer.* / stt.frontend.* KV into hp.
transcribe_status read_sortformer_hparams(const gguf_context * gguf, SortformerHParams & hp);

// Look up the Sortformer-specific tensors (encoder_proj + transformer +
// head) by name in ctx_meta, validate shapes against hp, store borrowed
// pointers. Returns TRANSCRIBE_ERR_GGUF (naming the tensor) on any
// missing / mis-shaped tensor.
transcribe_status build_sortformer_weights(ggml_context *          ctx_meta,
                                           const SortformerHParams & hp,
                                           SortformerWeights &       weights);

}  // namespace transcribe::sortformer
