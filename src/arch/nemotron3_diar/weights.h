// arch/nemotron3_diar/weights.h - Nemotron-3-Diarization hparams + weight
// slots: feature-stacking pre_encode, 31-layer pre-LN RoPE Transformer
// encoder, encoder_proj, subpixel upsampler, sigmoid speaker head, and the
// learned AOSC silence embedding.
//
// Tensor layout conventions (matched by scripts/convert-nemotron3_diar.py):
//   - Linear weights: PyTorch [out, in] -> ggml ne [in, out].
//   - attn.qkv: [d, 3d] (ne), output rows [q | k | v], each head-major.
//   - diar.upsample.conv.weight: PyTorch Conv1d [out=1536, in=192, k=3]
//     -> ggml ne [3, 192, 1536].
//   - LayerNorm: separate weight + bias, shape [d].

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::nemotron3_diar {

struct Nemotron3DiarHParams {
    // Diarization.
    int32_t max_speakers    = 8;  // hard architectural cap
    int32_t frame_hop       = 0;  // samples per encoder / AOSC frame (80 ms)
    int32_t output_hop      = 0;  // samples per output frame (10 ms)
    int32_t upsample_factor = 0;  // output frames per encoder frame

    // Encoder (pre-LN Transformer, RoPE).
    int32_t     enc_n_layers           = 0;
    int32_t     enc_d_model            = 0;
    int32_t     enc_n_heads            = 0;
    int32_t     enc_d_ff               = 0;
    int32_t     enc_feat_in            = 0;
    int32_t     enc_subsampling_factor = 0;
    float       enc_rope_base          = 10000.0f;
    float       enc_rotary_fraction    = 1.0f;
    float       enc_ln_eps             = 1e-5f;
    std::string enc_subsampling;  // "feature_stacking"
    std::string enc_activation;   // "gelu"

    // Head.
    int32_t head_d_model = 0;

    // AOSC compression constants.
    int32_t spkcache_sil_frames_per_spk = 1;
    float   pred_score_threshold        = 0.25f;
    float   scores_boost_latest         = 0.05f;
    float   sil_threshold               = 0.2f;
    float   strong_boost_rate           = 0.75f;
    float   weak_boost_rate             = 1.5f;
    float   min_pos_scores_rate         = 0.5f;
    int32_t max_index                   = 99999;

    // Shipped default streaming preset (card very_high_latency).
    int32_t stream_spkcache_len           = 0;
    int32_t stream_fifo_len               = 0;
    int32_t stream_chunk_len              = 0;
    int32_t stream_chunk_right_context    = 0;
    int32_t stream_spkcache_update_period = 0;

    // Frontend.
    int32_t     fe_num_mels    = 0;
    int32_t     fe_sample_rate = 0;
    int32_t     fe_n_fft       = 0;
    int32_t     fe_win_length  = 0;
    int32_t     fe_hop_length  = 0;
    std::string fe_window;
    std::string fe_normalize;
    float       fe_dither       = 0.0f;
    float       fe_pre_emphasis = 0.0f;

    int32_t head_dim() const { return enc_n_heads > 0 ? enc_d_model / enc_n_heads : 0; }
};

// One pre-LN Transformer block.
struct Nemotron3DiarBlock {
    ggml_tensor * norm1_w    = nullptr;  // [d]
    ggml_tensor * norm1_b    = nullptr;  // [d]
    ggml_tensor * attn_qkv_w = nullptr;  // [d, 3d] (no bias)
    ggml_tensor * attn_out_w = nullptr;  // [d, d]
    ggml_tensor * attn_out_b = nullptr;  // [d]
    ggml_tensor * norm2_w    = nullptr;  // [d]
    ggml_tensor * norm2_b    = nullptr;  // [d]
    ggml_tensor * ff_in_w    = nullptr;  // [d, d_ff]
    ggml_tensor * ff_in_b    = nullptr;  // [d_ff]
    ggml_tensor * ff_out_w   = nullptr;  // [d_ff, d]
    ggml_tensor * ff_out_b   = nullptr;  // [d]
};

struct Nemotron3DiarWeights {
    // Frontend buffers (F32, BF16-exact checkpoint values).
    ggml_tensor * fe_window     = nullptr;      // [win_length]
    ggml_tensor * fe_filterbank = nullptr;      // [n_fft/2+1, n_mels]

    ggml_tensor * pre_encode_proj_w = nullptr;  // [feat_in*sub, d] (no bias)
    ggml_tensor * embed_norm_w      = nullptr;  // [d]
    ggml_tensor * embed_norm_b      = nullptr;  // [d]

    std::vector<Nemotron3DiarBlock> blocks;     // enc_n_layers

    ggml_tensor * final_norm_w = nullptr;       // [d]
    ggml_tensor * final_norm_b = nullptr;       // [d]

    ggml_tensor * enc_proj_w = nullptr;         // [d, head_d]
    ggml_tensor * enc_proj_b = nullptr;         // [head_d]
    ggml_tensor * upsample_w = nullptr;         // [3, head_d, head_d*upsample]
    ggml_tensor * upsample_b = nullptr;         // [head_d*upsample]
    ggml_tensor * fc1_w      = nullptr;         // [head_d, head_d]
    ggml_tensor * fc1_b      = nullptr;         // [head_d]
    ggml_tensor * spk_head_w = nullptr;         // [head_d, max_speakers]
    ggml_tensor * spk_head_b = nullptr;         // [max_speakers]
    ggml_tensor * sil_emb    = nullptr;         // [d]
};

// Read every required stt.nemotron3_diar.* / stt.frontend.* KV into hp.
transcribe_status read_hparams(const gguf_context * gguf, Nemotron3DiarHParams & hp);

// Look up every tensor by name in ctx_meta, validate shapes against hp and
// store borrowed pointers. Returns TRANSCRIBE_ERR_GGUF (naming the tensor)
// on any missing / mis-shaped tensor.
transcribe_status build_weights(ggml_context * ctx_meta, const Nemotron3DiarHParams & hp, Nemotron3DiarWeights & w);

}  // namespace transcribe::nemotron3_diar
