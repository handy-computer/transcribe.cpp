#pragma once

#include "weights.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::arkasr {

struct EncoderBuild {
    ggml_tensor * mel       = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * output    = nullptr;
    ggml_cgraph * graph     = nullptr;
    int           n_audio   = 0;
};

EncoderBuild build_encoder_graph(ggml_context *        ctx,
                                 const ArkAsrWeights & weights,
                                 const ArkAsrHParams & hp,
                                 int                   n_mel_frames);

}  // namespace transcribe::arkasr
