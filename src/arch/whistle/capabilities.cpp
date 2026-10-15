// arch/whistle/capabilities.cpp - Whistle capability defaults.

#include "whistle.h"

namespace transcribe::whistle {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate       = 16000;
    caps.supports_translate       = false;
    caps.supports_language_detect = true;
    // Word timings come from DTW over cross-attention (80 ms frames).
    caps.max_timestamp_kind       = TRANSCRIBE_TIMESTAMPS_WORD;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::whistle
