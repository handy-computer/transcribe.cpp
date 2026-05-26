// arch/qwen3_asr/qwen3_asr.h - Qwen3-ASR model and context types.
//
// INTERNAL to src/arch/qwen3_asr/. Defines the concrete classes that
// derive from transcribe_model / transcribe_session for the Qwen3-ASR
// family (audio-LLM: audio encoder + Qwen3 causal LM with audio-token
// injection).
//
// First port targets non-streaming ASR transcription. The sibling
// forced-aligner variant (Qwen3-ForcedAligner-0.6B) and the streaming
// stream_transcribe path are tracked separately.

#pragma once

#include "qwen3_lm/qwen3_lm.h"
#include "transcribe-backend.h"
#include "transcribe-session.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <optional>
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

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_model & model);

// Encode "language {Name}<asr_text>" for the given BCP-47 code. The
// output is the token-id sequence the Qwen3-ASR chat template seeds
// the assistant turn with when a caller forces a language hint.
// Declared here (rather than hiding in the model.cpp anonymous
// namespace) so the BPE parity test can verify the full prefix
// matches the HF reference including the <asr_text> special token.
//
// Returns:
//   TRANSCRIBE_OK                     on success, out_ids populated.
//   TRANSCRIBE_ERR_INVALID_ARG        bcp47 null or empty.
//   TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE  bcp47 not in the publisher's
//                                     support list (the static map
//                                     in model.cpp is the source of
//                                     truth; drift vs. caps.languages
//                                     surfaces here).
//   TRANSCRIBE_ERR_GGUF               tokenizer has no encoder
//                                     (missing merges) or vocab
//                                     missing <asr_text> special.
transcribe_status encode_language_prefix(const transcribe::Tokenizer & tok,
                                         const char *                  bcp47,
                                         std::vector<int32_t> &        out_ids);

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

// Qwen3 chat-template special-token ids resolved through the loaded
// tokenizer at load time. The prompt renderer is narrow (no Jinja)
// but it still tokenizes to exact ids; resolving at load lets a
// future checkpoint that reorders the vocab fail loudly instead of
// silently corrupting the prompt.
//
// All fields are the *resolved* ids; the reference ids on the 0.6B
// and 1.7B vocab are stable (im_start=151644, im_end=151645,
// newline=198, system=8948, user=872, assistant=77091) but we do
// not hardcode them at runtime.
struct ChatTokens {
    int32_t im_start       = -1;
    int32_t im_end         = -1;
    int32_t newline        = -1;
    int32_t role_system    = -1;
    int32_t role_user      = -1;
    int32_t role_assistant = -1;
};

struct QwenAsrModel final : public transcribe_model {
    Tokenizer       tok;
    QwenAsrHParams  hparams;
    QwenAsrWeights  weights;
    ggml_context *  ctx_meta = nullptr;

    transcribe::BackendPlan       plan;
    ggml_backend_buffer_t         backend_buffer = nullptr;
    transcribe::qwen3_lm::PackedGateUpHandles packed_gate_up;

    std::optional<transcribe::MelFrontend> mel;

    // Jinja chat template. The prompt is rebuilt at run time with the
    // caller's language / context; we only store the template string
    // from the GGUF KV. Empty string means "no template" (the first
    // port will refuse to run without one).
    std::string chat_template;

    // Resolved chat-template token ids (filled at load time, see
    // resolve_chat_tokens in model.cpp). Used by build_prompt_tokens.
    ChatTokens chat_tokens;

    QwenAsrModel() = default;
    ~QwenAsrModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct QwenAsrSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    transcribe::qwen3_lm::KvCache kv_cache;

    std::vector<float> mel_buf;
    std::vector<float> enc_host;  // audio encoder output, pre-injection

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    QwenAsrSession() = default;
    ~QwenAsrSession() override;
};

} // namespace transcribe::qwen3_asr
