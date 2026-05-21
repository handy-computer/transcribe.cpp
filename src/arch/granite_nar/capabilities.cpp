// arch/granite_nar/capabilities.cpp - family invariants.

#include "granite_nar.h"

namespace transcribe::granite_nar {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // NAR variant is ASR only. The non-autoregressive editor does not
    // support translation or speaker diarization or word timestamps.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    caps.supports_translate = false;

    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::granite_nar
