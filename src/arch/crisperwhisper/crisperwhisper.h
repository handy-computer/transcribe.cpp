// arch/crisperwhisper/crisperwhisper.h - CrisperWhisper 2.0 model and context
// types. INTERNAL to src/arch/crisperwhisper/.
//
// Defines the concrete transcribe_model / transcribe_session subclasses. The
// graph state (KV cache, encoder output) comes from src/whisper_graph/; what
// is added here is the state the decode harness needs and whisper has no use
// for: the 10 ms mel kept for the word-timing blank signal, and the captured
// cross-attention rows.

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

namespace transcribe::crisperwhisper {

void apply_family_invariants(transcribe_model & model);

struct CwModel final : public transcribe_model {
    Tokenizer        tok;
    CwHParams        hparams;
    CwDecodeContract contract;
    CwWeights        weights;
    ggml_context *   ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Language token ids keyed by BCP-47 short code (from general.languages).
    std::vector<std::string> lang_codes;
    std::vector<int32_t>     lang_token_ids;

    // C++ mel frontend (per_utterance / hann_periodic / reflect / Slaney).
    std::optional<transcribe::MelFrontend> mel;

    CwModel() = default;
    ~CwModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

// One aligned word in chunk-local seconds. `start`/`end` are negative when the
// Viterbi could not place the word (the reference's None placeholders), which
// the long-form drop logic tolerates and the public API filters out.
struct CwWord {
    std::string text;
    float       start = -1.0f;
    float       end   = -1.0f;

    bool placed() const { return start >= 0.0f && end >= 0.0f; }
};

struct CwSession final : public transcribe_session {
    ggml_context *       compute_ctx      = nullptr;
    size_t               compute_ctx_size = 0;
    ggml_backend_sched_t sched            = nullptr;

    CwEncOut           enc_out;
    std::vector<float> enc_host;
    int                enc_T = 0;

    // Host buffer for the mel fed to the encoder ([n_mels, 3000], padded).
    std::vector<float> mel_buf;

    // The unpadded 10 ms-hop log-mel for the CURRENT window, kept because the
    // Viterbi aligner derives its per-frame blank (silence) probability from
    // mel energy — see word_timing.blank_logp_from_mel_energy. The encoder
    // consumes mel_buf; this is the same data before right-padding, and
    // mel_frames records how many frames are real audio.
    std::vector<float> mel_10ms;
    int                mel_10ms_frames = 0;

    CwKvCache kv_cache;

    // Head-averaged cross-attention, one row per generated token, [n_gen,
    // T_enc]. Filled only when word timing is requested.
    std::vector<float> cross_attn;
    int                cross_attn_rows   = 0;
    int                cross_attn_frames = 0;

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    CwSession() = default;
    ~CwSession() override;
};

}  // namespace transcribe::crisperwhisper
