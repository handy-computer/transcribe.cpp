// arch/cohere/capabilities.cpp - Cohere ASR capability defaults.

#include "cohere.h"

namespace transcribe::cohere {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;
    caps.supports_translate = false;

    // Cohere ASR uses an autoregressive decoder with the
    // <|notimestamp|> prompt token hard-wired on every run, so the
    // generated token stream carries no alignment information. The
    // family emits a full-text transcript inside a single segment
    // with zeroed t0/t1 — there is no honest token, word, or
    // segment timing to return. Advertise NONE and let the
    // dispatcher reject any request for finer granularity.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Cancellation is wired at the per-run level. The other unallied
    // features (initial prompt, temperature fallback, long-form, PNC,
    // ITN) are Whisper-specific or family-specific elsewhere; Cohere
    // ASR does not participate.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

} // namespace transcribe::cohere
