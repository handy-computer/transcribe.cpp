// arch/granite/granite.h - IBM Granite Speech model and context types.
//
// INTERNAL to src/arch/granite/. Defines the concrete classes that
// derive from transcribe_model / transcribe_context for the Granite
// Speech family (audio-LLM: Conformer encoder + BLIP-2 Q-Former
// projector + Granite-4 causal LM with audio-token injection).
//
// First port targets non-streaming ASR transcription across 3 AR
// variants:
//   - granite-4.0-1b-speech
//   - granite-speech-4.1-2b
//   - granite-speech-4.1-2b-plus
//
// The plus variant adds:
//   - encoder.cat_hidden_layers = [3]  (concat layer-3 hidden with
//     final hidden, doubling the projector's cross-attention K/V dim)
//   - text.tie_word_embeddings = true  (lm_head reuses token_embd)
//   - a full Granite-4 chat template (the 1b/2b variants use a bare
//     USER:/ASSISTANT: template)
//
// The NAR variant (granite-speech-4.1-2b-nar) is a distinct
// architecture (encoder-ctc with bidirectional LLM editor) and is
// handled by the `granite_nar` family.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-context.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "qwen3_lm/qwen3_lm.h"
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

namespace transcribe::granite {

void apply_family_invariants(transcribe_model & model);

// Chat-template token ids resolved through the loaded tokenizer at
// load time. Granite's bare-USER:/ASSISTANT: template (1b/2b) and the
// Granite-4 system-role template (plus) overlap on the role-marker
// tokens; we resolve every piece we might need ahead of time so a
// future vocab drift fails loudly at load instead of mid-decode.
//
// All fields are the *resolved* ids. The reference IDs on the granite
// vocab (100256..100352 are added tokens) are stable across the 3 AR
// variants, but we look them up rather than hardcode.
struct ChatTokens {
    int32_t audio          = -1;   // <|audio|>
    int32_t end_of_text    = -1;   // <|end_of_text|>  (also EOS / BOS)
    int32_t pad            = -1;   // <|pad|>
    int32_t start_of_role  = -1;   // <|start_of_role|>  (granite-4 chat only)
    int32_t end_of_role    = -1;   // <|end_of_role|>    (granite-4 chat only)
};

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

struct GraniteModel final : public transcribe_model {
    Tokenizer        tok;
    GraniteHParams   hparams;
    GraniteWeights   weights;
    ggml_context *   ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Pre-fused BatchNorm scale + bias for every encoder block. Lives
    // in a separate ggml context + backend buffer so we can write the
    // fused values once at load time (ctx_meta is no_alloc and the
    // weight buffer is read-only). The per-block fused tensors are
    // referenced from GraniteEncBlock::conv_bn_fused_*.
    ggml_context *        bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t bn_fused_buffer = nullptr;

    // Packed FFN gate+up: one mul_mat per block instead of two. Owned by
    // qwen3_lm::pack_gate_up; freed in ~GraniteModel().
    transcribe::qwen3_lm::PackedGateUpHandles packed_gate_up;

    std::optional<transcribe::MelFrontend> mel;

    // Jinja chat template from the GGUF KV. Stored verbatim — we don't
    // render Jinja at runtime; the family-specific prompt builder
    // (Phase 5) emits the equivalent token sequence directly. Empty
    // string means "no template" (the runtime will refuse to decode
    // without one).
    std::string chat_template;

    // Resolved chat-template token ids (filled at load time).
    ChatTokens chat_tokens;

    GraniteModel() = default;
    ~GraniteModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct GraniteContext final : public transcribe_context {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    // Audio encoder output (post-projector) buffered between encode and
    // decode. Each row is one audio token in LM hidden space (2048).
    std::vector<float> mel_buf;
    std::vector<float> audio_tokens_host;
    int32_t            n_audio_tokens = 0;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    // Self-attention KV cache (40 layers, 4 kv heads × 128 head_dim).
    // Allocated lazily on the first run() call so we can size n_ctx to
    // the actual prompt+gen budget.
    transcribe::qwen3_lm::KvCache kv;

    GraniteContext() = default;
    ~GraniteContext() override;
};

} // namespace transcribe::granite
