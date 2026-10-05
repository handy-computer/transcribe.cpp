#pragma once

#include "causal_lm/causal_lm.h"
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <optional>
#include <vector>

namespace transcribe::arkasr {

void apply_family_invariants(transcribe_model & model);

struct SpecialTokens {
    int32_t user        = -1;
    int32_t begin_audio = -1;
    int32_t audio       = -1;
    int32_t end_audio   = -1;
    int32_t assistant   = -1;
    int32_t eos         = -1;
};

struct ArkAsrModel final : public transcribe_model {
    Tokenizer            tok;
    ArkAsrHParams        hparams;
    ArkAsrWeights        weights;
    SpecialTokens        special;
    std::vector<int32_t> instruction_tokens;
    ggml_context *       ctx_meta = nullptr;

    BackendPlan                            plan;
    ggml_backend_buffer_t                  backend_buffer = nullptr;
    causal_lm::PackedGateUpHandles         packed_gate_up;
    std::optional<transcribe::MelFrontend> mel;

    ~ArkAsrModel() override;

    const Tokenizer * tokenizer() const override { return &tok; }
};

struct ArkAsrSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    causal_lm::KvCache   kv_cache;
    std::vector<float>   mel_buf;
    std::vector<float>   audio_buf;
    bool                 decoder_use_flash = true;

    ~ArkAsrSession() override;
};

}  // namespace transcribe::arkasr
