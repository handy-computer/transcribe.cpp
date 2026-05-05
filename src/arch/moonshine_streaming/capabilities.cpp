// arch/moonshine_streaming/capabilities.cpp - Moonshine-Streaming
// capability defaults.

#include "moonshine_streaming.h"

namespace transcribe::moonshine_streaming {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // English-only model. Translation and language detection are off; the
    // family doc's Capability Validation table marks both SKIP — not
    // exposed by runtime.
    caps.supports_translate = false;

    // No timestamp tokens in the vocab; runtime returns a single
    // text-only segment with zeroed timings (same policy as moonshine).
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Streaming session API is post-port (Stage 4 ships one-shot).
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::moonshine_streaming
