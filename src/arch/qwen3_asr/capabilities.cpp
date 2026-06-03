// arch/qwen3_asr/capabilities.cpp - family invariants.

#include "qwen3_asr.h"

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Qwen3-ASR emits transcript text inside a Qwen-style chat response.
    // It does not surface timestamps of any kind for the transcript-only
    // task. The sibling Qwen3-ForcedAligner produces word timestamps,
    // but that is a separate checkpoint with a different head contract
    // and would register as its own family (qwen3_forced_aligner) when
    // ported.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // The publisher documents language detection (the LM prefixes its
    // transcript with "language X") and also accepts a caller-supplied
    // language hint — the chat template seeds the assistant turn with
    // "language {Name}<asr_text>" and the LM continues with pure
    // transcript text, skipping its own detection preamble. Either
    // path is supported; caps.languages is the list of codes both
    // support, with the same BCP-47 canonicalization used by every
    // other family in this library.
    //
    // Translation is not an advertised capability for Qwen3-ASR;
    // callers get transcript-only output in the audio's source
    // language.
    caps.supports_translate = false;

    // Cancellation is wired at the per-run level. Whisper-specific
    // long-form / temperature-fallback / initial-prompt features do
    // not apply here. No runtime PNC/ITN toggle.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

} // namespace transcribe::qwen3_asr
