// arch/moonshine/capabilities.cpp - Moonshine ASR capability defaults.

#include "moonshine.h"

namespace transcribe::moonshine {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // English-only model (per the model card and tokenizer vocab — no
    // language tokens, no <|translate|>). Translation and language
    // detection are off; the family doc's Capability Validation table
    // marks both SKIP — not exposed by runtime.
    caps.supports_translate = false;

    // Moonshine emits no timestamp tokens and the model card does not
    // advertise alignment. Mirror cohere's NONE policy: the runtime
    // returns a single text-only segment with zeroed timings.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Whisper-specific features that moonshine does not participate in.
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::moonshine
