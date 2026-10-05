// arch/arkasr/capabilities.cpp - family invariants.

#include "arkasr.h"

namespace transcribe::arkasr {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Transcript-only chat response; no timestamps (the sibling
    // Qwen3-ForcedAligner would be its own family).
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // The fixed transcription prompt lets the model infer language, but this
    // port does not yet consume a caller-supplied language hint. Translation
    // is not advertised.
    caps.supports_translate = false;

    // Cancellation is wired at the per-run level. No PNC/ITN toggle; the
    // Whisper-specific features do not apply here.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::arkasr
