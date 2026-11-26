// arch/smart_turn/smart_turn.h - Smart Turn v3 (Pipecat) family: a Whisper
// encoder over the last 8 s of a turn, attention pooling and an MLP head
// giving one end-of-turn probability (TURN role, clip model).
//
// The encoder and the log-mel front end are Whisper's (arch/whisper/encoder.h,
// transcribe-mel.h); the GGUF stores them under the Whisper tensor names.

#pragma once

#include "../whisper/weights.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-turn.h"

#include <cstdint>
#include <optional>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace transcribe {
struct Arch;
}

namespace transcribe::smart_turn {

struct HParams {
    // Encoder (enc_*) and front end (fe_*); the decoder fields are unused.
    whisper::WhisperHParams whisper;

    int32_t max_audio_samples      = 0;     // the scored window: shorter input is left zero padded
    float   threshold              = 0.5f;  // upstream cut-off
    bool    waveform_normalize     = true;  // zero mean, unit variance over the whole window
    float   waveform_normalize_eps = 1e-7f;
    int32_t pool_hidden            = 0;     // head.pool.0 width (256)
    int32_t cls_hidden0            = 0;     // head.cls.0 width (256)
    int32_t cls_hidden1            = 0;     // head.cls.4 width (64)
    float   cls_norm_eps           = 1e-5f;

    int32_t n_mel_frames() const { return max_audio_samples / whisper.fe_hop_length; }
};

struct HeadWeights {
    ggml_tensor * pool0_w = nullptr;  // ne [d_model, pool_hidden]
    ggml_tensor * pool0_b = nullptr;  // ne [pool_hidden]
    ggml_tensor * pool2_w = nullptr;  // ne [pool_hidden, 1]
    ggml_tensor * pool2_b = nullptr;  // ne [1]
    ggml_tensor * cls0_w  = nullptr;  // ne [d_model, cls_hidden0]
    ggml_tensor * cls0_b  = nullptr;
    ggml_tensor * cls1_w  = nullptr;  // LayerNorm, ne [cls_hidden0]
    ggml_tensor * cls1_b  = nullptr;
    ggml_tensor * cls4_w  = nullptr;  // ne [cls_hidden0, cls_hidden1]
    ggml_tensor * cls4_b  = nullptr;
    ggml_tensor * cls6_w  = nullptr;  // ne [cls_hidden1, 1]
    ggml_tensor * cls6_b  = nullptr;  // ne [1]
};

struct Model final : public transcribe_model {
    HParams                 hparams;
    whisper::WhisperWeights encoder;  // frontend + enc_* only
    HeadWeights             head;
    TurnModelInfo           turn_info;

    ggml_context *             ctx_meta = nullptr;
    load_common::WeightStorage weight_storage;
    BackendPlan                plan;

    std::optional<MelFrontend> mel;

    // Manual attention by default (closest to the reference on CPU);
    // TRANSCRIBE_FLASH_ATTN overrides.
    bool use_flash = false;

    Model() = default;
    ~Model() override;
};

struct Session final : public transcribe_turn_session {
    std::vector<float> window;   // [max_audio_samples], normalised
    std::vector<float> mel_raw;  // [n_mels, T] mel-major (MelFrontend)
    std::vector<float> mel_in;   // [T, n_mels] frame-major (encoder input)
};

extern const Arch arch;

}  // namespace transcribe::smart_turn
