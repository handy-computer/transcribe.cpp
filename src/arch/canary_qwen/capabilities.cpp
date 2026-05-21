// arch/canary_qwen/capabilities.cpp - family invariants.

#include "canary_qwen.h"

namespace transcribe::canary_qwen {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // SALM emits a chat-format Qwen3 response. No timestamp head; no
    // language conditioning (English-only training); no separate
    // initial-prompt path beyond the chat-template prefix.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    caps.supports_translate            = false;
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::canary_qwen
