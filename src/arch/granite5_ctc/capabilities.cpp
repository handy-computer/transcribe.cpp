// arch/granite5_ctc/capabilities.cpp - family invariants.

#include "granite5_ctc.h"

namespace transcribe::granite5_ctc {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // CTC frame alignment gives a per-frame emission index for free, so
    // the family publishes word-level timestamps. They are OUR
    // construction, not a reference output: upstream neither advertises
    // nor emits timings.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_WORD;

    // English-only ASR. No translation, no language identification, no
    // diarizer, and the frontend is non-causal (centered STFT plus a
    // per-utterance log-mel floor), so no streaming either.
    caps.supports_translate = false;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::granite5_ctc
