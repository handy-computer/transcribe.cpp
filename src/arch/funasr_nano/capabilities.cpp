// arch/funasr_nano/capabilities.cpp - family invariants.

#include "funasr_nano.h"

namespace transcribe::funasr_nano {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // Fun-ASR-Nano produces transcript text via a Qwen3 chat-format
    // response; no timestamp head is exposed. The CTC head shipped in
    // the checkpoint is auxiliary (and partial — last 3 blocks missing
    // from the .pt file), so it is intentionally skipped at convert.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    caps.supports_translate            = false;
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::funasr_nano
