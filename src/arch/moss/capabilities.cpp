// arch/moss/capabilities.cpp - family invariants.

#include "moss.h"

namespace transcribe::moss {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Segment timestamps and speaker tags are emergent plain text inside the
    // transcript ([start][Sxx]text[end]); the runtime returns the raw string
    // without parsing them into structured segments/timestamps.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Transcription only; language is steered by the fixed prompt (no token).
    caps.supports_translate = false;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::moss
