// arch/moss/moss.h - MOSS-Transcribe-Diarize model and context types.
//
// INTERNAL to src/arch/moss/. Concrete transcribe_model / transcribe_session
// subclasses for the MOSS family (audio-LLM: Whisper-Medium encoder + 4x time
// merge + VQAdaptor + Qwen3-0.6B causal LM with audio-token injection).
//
// Reuses src/causal_lm/ for the Qwen3 decoder block math (identical to
// arch/qwen3_asr) and a whisper-style encoder graph (arch/whisper). MOSS-
// specific pieces: the 4x temporal merge + VQAdaptor bridge, the non-contiguous
// audio-token injection (time-marker digits interleave the audio span), and the
// fixed baked prompt.

#pragma once

#include "causal_lm/causal_lm.h"
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
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::moss {

void apply_family_invariants(transcribe_model & model);

// Reproduce processing_moss_transcribe_diarize._audio_span_ids: the audio
// placeholder span with time-marker digit tokens interleaved. Deterministic
// integer math (no BPE); the digit ids come from the baked digit_tokens KV.
// out_audio_positions receives the prompt-relative indices (0-based within the
// span) of the audio_pad tokens, in order — the j-th audio feature lands at
// span position out_audio_positions[j].
void build_audio_span(const MossHParams &    hp,
                      int                    audio_seq_len,
                      std::vector<int32_t> & out_span_ids,
                      std::vector<int32_t> & out_audio_offsets);

// Assemble the full prompt: prefix_tokens + audio_span + suffix_tokens.
// out_audio_positions holds the absolute prompt positions of the audio_pad
// tokens (in order), so the b-th audio feature is scattered to
// input_ids[out_audio_positions[b]].
void build_prompt_tokens(const MossHParams &    hp,
                         int                    audio_seq_len,
                         std::vector<int32_t> & out_ids,
                         std::vector<int32_t> & out_audio_positions);

struct MossModel final : public transcribe_model {
    Tokenizer      tok;
    MossHParams    hparams;
    MossWeights    weights;
    ggml_context * ctx_meta = nullptr;

    transcribe::BackendPlan                    plan;
    ggml_backend_buffer_t                      backend_buffer = nullptr;
    transcribe::causal_lm::PackedGateUpHandles packed_gate_up;

    std::optional<transcribe::MelFrontend> mel;

    std::string chat_template;  // informational; not used at inference

    MossModel() = default;
    ~MossModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct MossSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    transcribe::causal_lm::KvCache kv_cache;

    // Batched KV cache for offline transcribe_run_batch (n_batch slabs).
    transcribe::causal_lm::KvCache kv_cache_batch;
    int                            kv_batch_cap   = 0;
    int                            kv_batch_n_ctx = 0;

    std::vector<float> mel_buf;
    std::vector<float> enc_host;  // adaptor output [dec_hidden, T_enc], pre-injection

    bool encoder_use_flash = false;
    bool decoder_use_flash = true;

    MossSession() = default;
    ~MossSession() override;
};

}  // namespace transcribe::moss
