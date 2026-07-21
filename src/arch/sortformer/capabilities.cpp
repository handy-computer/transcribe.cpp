// arch/sortformer/capabilities.cpp - Sortformer capability defaults.
//
// Applied before transcribe::read_capability_kv (KV present overrides,
// KV absent keeps the default). Sortformer is a pure frame-level diarizer:
// no transcript, no translation, no transcript-timestamp capability. Its
// output is speaker segments via the transcript-independent speaker_segment
// surface, so DIARIZATION is a family invariant.

#include "sortformer.h"

namespace transcribe::sortformer {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    // Fixed 16 kHz mel bank (NeMo AudioToMelSpectrogramPreprocessor).
    caps.native_sample_rate = 16000;

    // Not a transcription model.
    caps.supports_translate = false;

    // Diarization segment times are intrinsic; there is no transcript, so
    // there is no transcript-timestamp capability to advertise.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // The model emits speaker-attributed output (T x 4 activity -> "who
    // spoke when" segments). run() populates the transcript-independent
    // speaker_segment surface directly from the probs.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_DIARIZATION, true);

    // Abort callback honored at the top of each run.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::sortformer
