// arch/ecapa_tdnn/ecapa_tdnn.h - ECAPA-TDNN family model and session types.
//
// INTERNAL to src/arch/ecapa_tdnn/. Defines the concrete classes deriving
// from transcribe_model / transcribe_langid_session, plus the family's Arch
// instance.
//
// The family is SpeechBrain's ECAPA-TDNN embedder with the VoxLingua107
// classifier head on top, serving the LANGID role. One model owns the
// weights, the GGUF metadata context, the backend buffer, the label table,
// and a shared (const-after-construction) log-mel front end; one session
// owns per-call host scratch plus the ggml scheduler inherited from
// SessionCore.

#pragma once

#include "mel.h"
#include "transcribe-backend.h"
#include "transcribe-langid.h"
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

    // Built at load from stt.langid.labels.*; immutable after.
    LangidLabels labels;

    // Front end, built once at load from stt.frontend.* plus the
    // `frontend.mel_filterbank` tensor. const after construction, so every
    // session shares it.
    std::unique_ptr<MelFrontend> mel;

    Model() = default;
    ~Model() override;
};

struct Session final : public transcribe_langid_session {
    // Host scratch, reused across calls.
    std::vector<float>   mel_buf;                // [T * n_mels], frame-major
    std::vector<int32_t> idx_buf[kNumSeBlocks];  // reflect indices, [T + 2p]

    // Metadata arena for the per-call graph build, handed to ggml_init as
    // mem_buffer so a per-call ggml_init does not malloc / free it on every
    // run. Outlives compute_ctx's use: ggml_free never touches a borrowed
    // mem_buffer.
    std::vector<uint8_t> graph_arena;

    // One-shot TRANSCRIBE_ECAPA_GRAPH_STATS reporting latch (see model.cpp).
    bool logged_graph_stats = false;
};

extern const Arch arch;

}  // namespace transcribe::ecapa_tdnn
