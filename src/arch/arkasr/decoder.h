#pragma once

#include "causal_lm/causal_lm.h"
#include "weights.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::arkasr {

struct PrefillBuild {
    ggml_tensor * input_ids = nullptr;
    ggml_tensor * audio     = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask      = nullptr;
    ggml_tensor * logits    = nullptr;
    ggml_cgraph * graph     = nullptr;
};

PrefillBuild build_prefill_graph(ggml_context *                   ctx,
                                 const ArkAsrWeights &            weights,
                                 const ArkAsrHParams &            hp,
                                 transcribe::causal_lm::KvCache & kv_cache,
                                 int                              prompt_len,
                                 int                              audio_len,
                                 int                              audio_start,
                                 bool                             use_flash);

struct StepBuild {
    ggml_tensor * input_id   = nullptr;
    ggml_tensor * position   = nullptr;
    ggml_tensor * kv_index   = nullptr;
    ggml_tensor * mask       = nullptr;
    ggml_tensor * logit_mask = nullptr;
    ggml_tensor * token      = nullptr;
    ggml_cgraph * graph      = nullptr;
};

StepBuild build_step_graph(ggml_context *                   ctx,
                           const ArkAsrWeights &            weights,
                           const ArkAsrHParams &            hp,
                           transcribe::causal_lm::KvCache & kv_cache,
                           int                              max_kv,
                           bool                             use_flash);

}  // namespace transcribe::arkasr
