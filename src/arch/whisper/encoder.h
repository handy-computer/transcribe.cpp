// arch/whisper/encoder.h - Whisper encoder graph builder.
//
// Forward declarations for the encoder graph construction entry point.
// The encoder is a 2-layer Conv1d stem followed by a stack of pre-LN
// transformer blocks and a final LayerNorm; no relative position
// encoding, no convolution module inside the blocks, and no
// enc-dec projection (the encoder output is already in the decoder's
// hidden dim since upstream whisper ties enc_d_model == dec_d_model).
//
// Dumped tensor names follow the Stage-2 reference script
// (dump_reference_whisper_transformers.py): enc.mel.in, enc.conv1.out,
// enc.conv2.out, enc.pos_emb, enc.embed.out, enc.block.{0..N-1}.out
// (emitted for index 0 and the last block only — the first and last
// are the two spot-checks the reference dumps), enc.final.
//
// Layout: all intermediate activations are kept in [d_model, T] ggml
// layout (d_model innermost, T second), matching the
// channel-innermost convention the reference dumps use. The conv stem
// transposes into [T, d_model] for the ggml_conv_1d calls and back
// out afterwards.

#pragma once

#include "ggml.h"

#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::whisper {

struct WhisperHParams;
struct WhisperWeights;

struct EncoderDumps {
    ggml_tensor * mel_in     = nullptr;
    ggml_tensor * conv1_out  = nullptr;
    ggml_tensor * conv2_out  = nullptr;
    ggml_tensor * pos_emb    = nullptr;
    ggml_tensor * embed_out  = nullptr;
    ggml_tensor * block0_out = nullptr;
    ggml_tensor * block_last_out = nullptr;
    ggml_tensor * final_out  = nullptr;

    // Every post-block residual stream output, in block order. Populated
    // alongside block0_out / block_last_out so the model driver can dump
    // every block for validate.py parity against the reference dumps,
    // which emit enc.block.{0..N-1}.out for all N layers.
    std::vector<ggml_tensor *> block_outs;
};

struct EncoderBuild {
    // Input tensor: [n_mel_bins, n_mel_frames] host-side layout. The
    // graph builder declares ggml_set_input on this tensor; the caller
    // uploads the mel spectrogram with ggml_backend_tensor_set before
    // the graph is computed.
    ggml_tensor * mel_in = nullptr;

    // Output tensor: [d_model, T_enc] where T_enc = n_mel_frames / 2
    // after the stride-2 second conv.
    ggml_tensor * out = nullptr;

    EncoderDumps  dumps {};
    ggml_cgraph * graph = nullptr;
};

// Build the encoder forward graph in compute_ctx.
//
// n_mel_frames is the number of frames in the mel input. Upstream
// whisper always pads-or-trims to 3000; the builder accepts any
// positive even value so unit tests can exercise smaller fixtures.
//
// use_flash selects ggml_flash_attn_ext vs. manual mul_mat + soft_max
// for each transformer block. The caller is responsible for gating
// this against backend support (see TRANSCRIBE_NO_FLASH policy).
//
// backend_name is informational only; no per-backend conv policy is
// applied on whisper (the stem is small, vanilla conv1d is fine on
// every backend).
EncoderBuild build_encoder_graph(ggml_context *          compute_ctx,
                                 const WhisperWeights &  weights,
                                 const WhisperHParams &  hp,
                                 int                     n_mel_frames,
                                 bool                    use_flash = true,
                                 const char *            backend_name = "");

} // namespace transcribe::whisper
