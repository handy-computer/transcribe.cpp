// arch/canary/capabilities.cpp - Canary multitask AED capability defaults.

#include "canary.h"

namespace transcribe::canary {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // Canary is a multitask AED — every variant supports en->de/es/fr
    // translation plus en/de/es/fr ASR, except canary-1b which is
    // English-ASR-only. The runtime advertises translate=true at the
    // family default; the GGUF's stt.capability.translate value (read
    // by read_capability_kv after this call) overrides per-variant.
    caps.supports_translate = true;

    // V1 port: timestamps are explicitly experimental upstream and
    // out of scope (per intake known_risks). Advertise NONE; the GGUF
    // KV stt.capability.timestamps overrides this when a future port
    // wires up the timestamp decoder path.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // No initial-prompt / temperature-fallback / long-form support in
    // v1. Cancellation is wired at the run level; flag accordingly.
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::canary
