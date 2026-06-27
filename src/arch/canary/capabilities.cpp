// arch/canary/capabilities.cpp - Canary multitask AED capability defaults.

#include "canary.h"

namespace transcribe::canary {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Canary is a multitask AED. The runtime advertises translate=true at the
    // family default; the GGUF's stt.capability.translate value (read by
    // read_capability_kv after this call) overrides per-variant. Exact
    // directions are an optional GGUF contract in stt.translation.pairs.
    caps.supports_translate = true;

    // V1 port: timestamps are explicitly experimental upstream and
    // out of scope (per intake known_risks). Advertise NONE; the GGUF
    // KV stt.capability.timestamps overrides this when a future port
    // wires up the timestamp decoder path.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Feature bits: cancellation is wired at the run level. Canary
    // exposes a runtime PNC toggle via the multitask prompt's pnc /
    // nopnc token slot; the generic transcribe_run_params::pnc enum routes
    // there. No runtime ITN control on canary, so FEATURE_ITN stays
    // unset and a non-DEFAULT itn against canary triggers the
    // advisory WARN.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_PNC,          true);
}

} // namespace transcribe::canary
