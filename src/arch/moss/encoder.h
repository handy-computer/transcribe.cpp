// arch/moss/encoder.h - MOSS encoder (Whisper) + adaptor graph builders.
//
// Two graphs, run per-chunk then once:
//
//   build_encoder_graph: log-mel [n_mels, n_mel_frames] -> Whisper 2-conv stem
//     (+GELU) -> + learned positional embedding -> 24 pre-LN blocks -> final
//     LayerNorm -> [d_model, T_whisper] (T_whisper = n_mel_frames / 2). One
//     chunk per call; the reference pads every 30s chunk to n_samples so
//     n_mel_frames is always nb_max_frames (3000) for a full/padded chunk.
//
//   build_adaptor_graph: concatenated + per-chunk-trimmed encoder output
//     [d_model, T_trim] (T_trim divisible by audio_merge_size) -> 4x time merge
//     (reshape to [d_model*merge, T_trim/merge]) -> VQAdaptor (fc1+SiLU+fc2+LN)
//     -> [dec_hidden, T_enc]  (T_enc = T_trim / merge).
//
// ggml ne (fast innermost): [n_mels, T] for mel, [d_model, T] after the stem.

#pragma once

#include "ggml.h"
#include "weights.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::moss {

struct EncoderDumps {
    ggml_tensor * pos_add_out    = nullptr;  // post learned-PE add
    ggml_tensor * block_0_out    = nullptr;
    ggml_tensor * block_last_out = nullptr;
    ggml_tensor * ln_post_out    = nullptr;  // after final LayerNorm
};

struct EncoderBuild {
    ggml_tensor * mel_in = nullptr;  // [n_mels, n_mel_frames]
    ggml_tensor * out    = nullptr;  // [d_model, n_mel_frames/2]
    EncoderDumps  dumps{};
    ggml_cgraph * graph = nullptr;
};

EncoderBuild build_encoder_graph(ggml_context *      ctx,
                                 const MossWeights & weights,
                                 const MossHParams & hp,
                                 int                 n_mel_frames,
                                 bool                use_flash);

struct AdaptorDumps {
    ggml_tensor * merge_out   = nullptr;  // [adaptor_input_dim, T_enc]
    ggml_tensor * adaptor_out = nullptr;  // [dec_hidden, T_enc]
};

struct AdaptorBuild {
    ggml_tensor * in  = nullptr;  // [d_model, T_trim]
    ggml_tensor * out = nullptr;  // [dec_hidden, T_enc]
    AdaptorDumps  dumps{};
    ggml_cgraph * graph = nullptr;
    int           T_enc = 0;
};

// T_trim must be > 0 and divisible by hp.audio_merge_size.
AdaptorBuild build_adaptor_graph(ggml_context * ctx, const MossWeights & weights, const MossHParams & hp, int T_trim);

// Whisper conv-stem length reduction: n_mel_frames -> n_mel_frames/2 (conv2
// stride 2). Exposed for host-side sizing.
int whisper_enc_len(int n_mel_frames);

// Per-chunk audio token length (processing._compute_audio_token_length):
// (num_samples - 1) / (hop_length * 2 * merge_size) + 1.
int audio_token_length(int num_samples, const MossHParams & hp);

}  // namespace transcribe::moss
