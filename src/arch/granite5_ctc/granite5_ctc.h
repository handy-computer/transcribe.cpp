// arch/granite5_ctc/granite5_ctc.h - IBM Granite Speech 5.0 TurboCTC.
//
// Family `granite5_ctc` covers granite-speech-5.0-470m-turboctc: a bare
// Granite Conformer encoder with a tied 16384-way BPE CTC head. No
// projector, no LM, no autoregressive loop — one encoder dispatch, one
// linear, then argmax + CTC collapse on the host.
//
// Kept separate from `granite` (audio-llm: encoder + Q-Former + Granite-4
// LM) and `granite_nar` (non-autoregressive editor), both of which carry
// a text model this variant does not have. The shared piece is
// src/granite_conformer/shaw_attn.h.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::granite5_ctc {

void apply_family_invariants(transcribe_model & model);

struct Granite5CtcModel final : public transcribe_model {
    Tokenizer          tok;
    Granite5CtcHParams hparams;
    Granite5CtcWeights weights;
    ggml_context *     ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Per-block fused BatchNorm scale/bias, precomputed at load
    // (model.cpp fuse_batch_norm). Lives in its own context/buffer so
    // the weight buffer stays exactly what the GGUF shipped.
    ggml_context *        bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t bn_fused_buffer = nullptr;

    // Shared mel frontend, configured from the GGUF frontend KV and the
    // baked filterbank/window tensors.
    std::optional<transcribe::MelFrontend> mel;

    Granite5CtcModel() = default;
    ~Granite5CtcModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct Granite5CtcSession final : public transcribe_session {
    // Host scratch, reused across runs.
    std::vector<float> feats_buf;   // [T_enc, input_dim] stacked frontend output
    std::vector<float> logits_buf;  // [T_out, vocab] CTC logits

    Granite5CtcSession() = default;
    ~Granite5CtcSession() override;
};

}  // namespace transcribe::granite5_ctc
