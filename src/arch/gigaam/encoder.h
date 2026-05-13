// arch/gigaam/encoder.h - GigaAM Conformer encoder graph builder.
//
// M1 stub: build_encoder_graph is not yet implemented. M2 lands the
// pre-encode + rotary PE bank + all 16 conformer blocks. M3 wires the
// per-call run() against this.

#pragma once

#include "ggml.h"

#include <string>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::gigaam {

struct GigaamHParams;
struct GigaamWeights;

struct EncoderDumps {
    ggml_tensor * pre_encode_out = nullptr;
    ggml_tensor * pos_emb        = nullptr;
    ggml_tensor * block0_out     = nullptr;
    ggml_tensor * mid_block_out  = nullptr;
    ggml_tensor * last_block_out = nullptr;
    ggml_tensor * final_out      = nullptr;
    // Pre-final-transpose tensor, ne=[d_model, T_enc, 1]. PyTorch
    // equivalent shape (1, T_enc, d_model). This is what the RNN-T /
    // CTC head consumes for decode and what the reference dumps as
    // `rnnt.encoded`. final_out (post-transpose) is what gets compared
    // against the reference `enc.out` dump.
    ggml_tensor * rnnt_encoded   = nullptr;
    int           mid_block_idx  = -1;
    int           last_block_idx = -1;
    std::vector<ggml_tensor *> all_block_outs;
};

struct EncoderBuild {
    ggml_tensor * mel_in     = nullptr;
    ggml_tensor * out        = nullptr;
    EncoderDumps  dumps      {};
    ggml_cgraph * graph      = nullptr;
};

// Build a fresh encoder forward graph. Not yet implemented (M2).
EncoderBuild build_encoder_graph(ggml_context *        compute_ctx,
                                 const GigaamWeights & weights,
                                 const GigaamHParams & hp,
                                 int                   n_mel_frames,
                                 ggml_type             kv_type = GGML_TYPE_COUNT,
                                 const char *          backend_name = "");

} // namespace transcribe::gigaam
