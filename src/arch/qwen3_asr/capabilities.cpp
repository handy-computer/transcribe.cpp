// arch/qwen3_asr/capabilities.cpp - family invariants.

#include "qwen3_asr.h"

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Transcript-only chat response; no timestamps (the sibling
    // Qwen3-ForcedAligner would be its own family).
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Supports language auto-detect and a caller-supplied hint (caps.languages
    // lists the BCP-47 codes). Translation is not advertised.
    caps.supports_translate = false;

    // Cancellation is wired at the per-run level. No PNC/ITN toggle; the
    // Whisper-specific features do not apply here.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
    // Generic vocabulary and context prompt both go in the system message,
    // the model's only context slot (prompting A/B,
    // notes/prompting-ab-results.md).
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_VOCABULARY, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CONTEXT_PROMPT, true);
}

}  // namespace transcribe::qwen3_asr
