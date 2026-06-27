// arch/vibevoice/capabilities.cpp - family invariants.

#include "vibevoice.h"

namespace transcribe::vibevoice {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    // VibeVoice-ASR ingests raw 24 kHz waveform (no mel frontend). The
    // library's run() contract still hands us 16 kHz mono PCM; the family
    // resamples to 24 kHz internally before the VAE tokenizers.
    caps.native_sample_rate = 24000;

    // The model emits structured Who/When/What JSON with segment-level
    // Start/End times. Segment timestamps are surfaced once the structured
    // output parser lands (Stage 4); advertise segment granularity.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;

    // Transcribe (incl. implicit language detection + diarization); no
    // speech-to-text translation is advertised.
    caps.supports_translate = false;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

} // namespace transcribe::vibevoice
