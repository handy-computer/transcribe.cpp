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

    // Stream-of-whole streaming wired in Phase 4a: stream_begin
    // buffers PCM, stream_finalize drains the buffer through the
    // one-shot inference path. Real incremental encoder/decoder
    // (Phase 4b+) keeps the same capability flag.
    caps.supports_streaming            = true;
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
    // Stream-of-whole has no fixed lookahead window or chunk hint:
    // the family accepts arbitrary feed sizes and only does work at
    // finalize. The two timing hints stay at the documented
    // "unsupported or unknown" sentinel (0) until real incremental
    // streaming provides concrete values.
}

} // namespace transcribe::moonshine_streaming
