// arch/gigaam/gigaam.h - GigaAM-family internal model and context types.
//
// INTERNAL to src/arch/gigaam/. The public C ABI in include/transcribe.h
// knows nothing about Arch; dispatch happens through find_arch() in
// transcribe-arch.cpp.
//
// GigaAM-v3 ships four ported variants (e2e_rnnt, e2e_ctc, rnnt, ctc).
// All four share a 16-layer Conformer encoder with rotary attention.
// The e2e variants use a SentencePiece tokenizer with punctuation +
// Cyrillic casing; the non-e2e (rnnt, ctc) variants use a charwise
// 33-entry vocab. The upstream `v3_ssl` (HuBERT-CTC pretraining
// checkpoint) is intentionally out of scope: encoder-only, no head, no
// transcribe path, and transcribe.cpp has no encoder-output emission
// CLI.

#pragma once

#include "decoder.h"
#include "mel.h"
#include "transcribe-backend.h"
#include "transcribe-context.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::gigaam {

// Family defaults — applied before read_capability_kv runs. Defined in
// capabilities.cpp. KV-driven overrides take precedence per the
// project-wide convention.
void apply_family_invariants(transcribe_capabilities & caps);

// Concrete model. Owns the ggml_context that holds every weight tensor's
// data buffer, and any per-family precomputed buffers we may add later
// (Stage 4 milestones will introduce a host decoder mirror analogous to
// parakeet's `host_decoder` for the RNN-T greedy decode loop).
struct GigaamModel final : public transcribe_model {
    Tokenizer       tok;
    GigaamHParams   hparams;
    GigaamWeights   weights;
    ggml_context *  ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Family-specific mel frontend (HTK mel, power=2, log-clamp,
    // center=False, periodic Hann). See arch/gigaam/mel.{h,cpp}.
    GigaamMelFrontend mel;

    // Host mirror of predictor + joint (RNN-T variants) or CTC head
    // (CTC variants). Populated in M3/M4. Empty for now.
    HostDecoderWeights host_decoder;

    GigaamModel() = default;
    ~GigaamModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

// Concrete context. One scheduler + compute_ctx lifecycle per context,
// mirroring parakeet.
struct GigaamContext final : public transcribe_context {
    ggml_context *        compute_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    ggml_tensor * encoder_out = nullptr;

    std::vector<float>   mel_buf;
    std::vector<float>   pos_buf;          // cos/sin rotary PE bank, host scratch
    std::vector<float>   enc_host;
    // RNN-T greedy decoder scratch goes here when M3 lands; CTC greedy
    // scratch lands at M4.

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    GigaamContext() = default;
    ~GigaamContext() override;
};

} // namespace transcribe::gigaam
