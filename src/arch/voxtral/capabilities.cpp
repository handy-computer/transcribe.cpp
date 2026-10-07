// arch/voxtral/capabilities.cpp - family invariants.

#include "voxtral.h"

namespace transcribe::voxtral {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Voxtral 2507 emits transcript / answer text only; no timestamp
    // tokens of any kind.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Speech translation IS an advertised capability for 2507. It is not
    // a task token: translation, Q&A and summarization all go through the
    // mistral-common instruct template (audio + free-text instruction).
    // The runtime exposes it via --translate / --target-language (which
    // synthesize a translate instruction).
    // The GGUF's stt.capability.translate=true is read into supports_translate
    // by read_capability_kv at load; default it true here as a fallback.
    caps.supports_translate = true;

    // Per-run cancellation. No INITIAL_PROMPT: Voxtral has no run extension
    // that accepts a prompt.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
    // TRANSCRIBE_TASK_INSTRUCT: the caller's prompt through the same chat
    // path as translation. Both 2507 sizes follow free-text instructions
    // (prompting A/B, notes/prompting-ab-results.md).
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_INSTRUCT, true);
}

}  // namespace transcribe::voxtral
