// arch/sensevoice/capabilities.cpp - SenseVoice family-default
// capability values applied before read_capability_kv overlays the
// converter-emitted KV.

#include "sensevoice.h"

namespace transcribe::sensevoice {

void apply_family_invariants(transcribe_capabilities & caps) {
    // FunASR WavFrontend hard-codes 16 kHz; the upstream model card
    // rejects anything else. The CLI rejects non-16 kHz WAVs at load
    // time so this number is the contract.
    caps.native_sample_rate = 16000;

    // SenseVoice-Small does not advertise translation. The upstream
    // task heads emit language / event / emotion / ITN labels — not a
    // separate translation stream.
    caps.supports_translate = false;

    // The non-AR CTC head has no segment- or word-timestamp head in
    // the published runtime. forced-CTC alignment IS available via
    // ctc_forced_align in the upstream demo, but the C++ runtime does
    // not expose it in v1; mark NONE and revisit in a follow-up if a
    // user asks for word timing.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // SenseVoice has no autoregressive decoder, no temperature loop,
    // no long-form chunker. Direct inference is hard-capped at 30 s
    // per call by the upstream model card; long-form is delegated to
    // an external fsmn-vad chunker out of scope for this port.
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::sensevoice
