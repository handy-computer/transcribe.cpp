// arch/ecapa_tdnn/ecapa_tdnn.h - ECAPA-TDNN family model and session types
// (SpeechBrain ECAPA-TDNN + VoxLingua107 classifier, LANGID role).
//
// INTERNAL to src/arch/ecapa_tdnn/.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-langid.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "weights.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace transcribe {
struct Arch;
}

namespace transcribe::ecapa_tdnn {

struct Model final : public transcribe_model {
    HParams hparams;
    Weights weights;

    // GGUF metadata context: owns the ggml_tensor structs the weight slots
    // above point at. Their data lives in backend_buffer.
    ggml_context *        ctx_meta       = nullptr;
    ggml_backend_buffer_t backend_buffer = nullptr;
    BackendPlan           plan;

    // Load-time derived weights (Weights::blk0_w_im2col): their own metadata
    // context and backend buffer, since ctx_meta is sized for the file.
    ggml_context *        ctx_derived    = nullptr;
    ggml_backend_buffer_t derived_buffer = nullptr;

    // Built at load from stt.langid.labels.*; immutable after.
    LangidLabels labels;

    // Built at load from stt.frontend.* and frontend.mel_filterbank; shared
    // by every session.
    std::unique_ptr<MelFrontend> mel;

    Model() = default;
    ~Model() override;
};

struct Session final : public transcribe_langid_session {
    // Host scratch, reused across calls.
    std::vector<float>   mel_raw;                // [n_mels * T], mel-major (MelFrontend)
    std::vector<float>   mel_buf;                // [T * n_mels], frame-major
    std::vector<float>   im2col_buf;             // [T * blk0_cols], frame-major
    std::vector<int32_t> idx_buf[kNumSeBlocks];  // reflect indices, [T + 2p]
};

extern const Arch arch;

}  // namespace transcribe::ecapa_tdnn
