// arch/nemotron3_diar/capabilities.cpp - Nemotron-3-Diarization capability
// defaults.
//
// Applied before transcribe::read_capability_kv (KV present overrides, KV
// absent keeps the default). A pure frame-level diarizer: no transcript, no
// translation, no transcript timestamps. Output is speaker segments via the
// transcript-independent speaker_segment surface, from transcribe_run and
// from push-audio streaming.

#include "nemotron3_diar.h"

namespace transcribe::nemotron3_diar {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    // Fixed 16 kHz mel bank (NeMo AudioToMelSpectrogramPreprocessor).
    caps.native_sample_rate = 16000;

    // Not a transcription model.
    caps.supports_translate = false;
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Push-audio live diarization (transcribe_stream_*).
    caps.supports_streaming = true;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_DIARIZATION, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::nemotron3_diar
