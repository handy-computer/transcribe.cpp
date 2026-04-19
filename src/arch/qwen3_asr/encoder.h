// arch/qwen3_asr/encoder.h - Qwen3-ASR audio encoder graph builder.
//
// The encoder is a bidirectional 18-layer transformer on top of a 3x
// Conv2d subsampler, with chunked block-diagonal attention driven by
// `cu_seqlens`. Modeled directly off
// qwen_asr.core.transformers_backend.modeling_qwen3_asr
// .Qwen3ASRAudioEncoder.
//
// Shape conventions (ggml fast-to-slow ne[]):
//
//   mel_in      : [mel_per_chunk,          n_mels,       1, n_chunks]
//                 per-chunk batched input. Chunks with fewer real mel
//                 frames than mel_per_chunk are zero-padded by the caller.
//   after 3x conv: [F_ds=n_mels/8, T_ds=mel_per_chunk/8, downsample_hidden, n_chunks]
//                 (ggml conv_2d convention swaps PyTorch's H/W, which is
//                 a no-op for the square kernel + symmetric stride.)
//   after conv_out + PE: [d_model, per_chunk_aftercnn, n_chunks]
//   flattened   : [d_model, T_enc_padded]  where T_enc_padded = n_chunks
//                 * per_chunk_aftercnn. The reference's ragged
//                 `padded_mask_after_cnn` select matches this when every
//                 chunk is a full `mel_per_chunk`; the first port
//                 enforces that precondition and the general short-tail
//                 case is tracked as a TODO.
//   attn mask   : [T_enc_padded, T_enc_padded] additive. Block-diagonal
//                 from cu_seqlens derived off the original (pre-chunk)
//                 `aftercnn_lens`; positions past cu_seqlens.back() have
//                 all-min rows (matches the reference quirk — see
//                 build_cu_seqlens_mask docstring).
//   output      : [output_dim, T_enc_padded]

#pragma once

#include "weights.h"

#include "ggml.h"

#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::qwen3_asr {

// Per-utterance timing metadata. Computed host-side from the mel length
// and encoder hparams; drives graph shape and mask construction.
struct EncoderTiming {
    int32_t n_mel_frames        = 0;  // input mel frame count
    int32_t mel_per_chunk       = 0;  // n_window * 2
    int32_t n_chunks            = 0;  // ceil(n_mel_frames / mel_per_chunk)
    int32_t last_chunk_real_mel = 0;  // real (unpadded) mel count in the last chunk
    int32_t per_chunk_aftercnn  = 0;  // aftercnn(mel_per_chunk) — same for every full chunk
    int32_t last_chunk_aftercnn = 0;  // aftercnn(last_chunk_real_mel)
    int32_t T_enc_padded        = 0;  // n_chunks * per_chunk_aftercnn (graph sequence length)
    int32_t T_enc               = 0;  // (n_chunks-1)*per + last_after (real, ragged)
    int32_t aftercnn_lens_total = 0;  // aftercnn(n_mel_frames) — drives cu_seqlens
};

// Apply the 3x stride-2 pad-1 kernel-3 downsampling formula three times.
int32_t aftercnn_len(int32_t mel_len);

EncoderTiming compute_encoder_timing(int32_t n_mel_frames,
                                     const QwenAsrHParams & hp);

// Sinusoidal position table, row-major "[length, d_model]" layout.
// Matches SinusoidsPositionEmbedding: the first d_model/2 channels are
// sin(p * inv_ts[k]), the second d_model/2 are cos(p * inv_ts[k]).
std::vector<float> build_sinusoid_pe(int32_t d_model, int32_t length,
                                     double max_timescale = 10000.0);

// Additive attention bias [T_enc_padded, T_enc_padded]. Returns a flat
// row-major row*T + col buffer — ggml consumes it as ne=[T, T, 1, 1]
// with ne[0] fastest. Reference quirk: cu_seqlens is built from the
// *pre-chunk* aftercnn length, so for inputs where T_enc exceeds
// cu_seqlens.back() the tail positions get all-min rows. We preserve
// that behavior bit-for-bit — the positions feed into the LM via audio
// injection and the model was trained with the quirk in place.
std::vector<float> build_cu_seqlens_mask(const EncoderTiming & t,
                                         const QwenAsrHParams & hp);

struct EncoderDumps {
    ggml_tensor * mel_in         = nullptr;  // graph input
    ggml_tensor * subsample_out  = nullptr;  // post conv_out linear, pre-PE
    ggml_tensor * pos_add_out    = nullptr;  // post-PE, flattened to [d_model, T_enc_padded]
    ggml_tensor * block_0_out    = nullptr;
    ggml_tensor * block_last_out = nullptr;
    ggml_tensor * ln_post_out    = nullptr;
    ggml_tensor * proj_out       = nullptr;
};

struct EncoderBuild {
    ggml_tensor * mel_in     = nullptr;  // [mel_per_chunk, n_mels, 1, n_chunks]
    ggml_tensor * pos_emb_in = nullptr;  // [d_model, per_chunk_aftercnn]
    ggml_tensor * mask_in    = nullptr;  // [T_enc_padded, T_enc_padded]
    ggml_tensor * out        = nullptr;  // [output_dim, T_enc_padded]
    EncoderDumps  dumps {};
    ggml_cgraph * graph      = nullptr;
    EncoderTiming timing {};
};

EncoderBuild build_encoder_graph(ggml_context *         ctx,
                                 const QwenAsrWeights & weights,
                                 const QwenAsrHParams & hp,
                                 const EncoderTiming &  timing,
                                 bool                   use_flash = false);

} // namespace transcribe::qwen3_asr
