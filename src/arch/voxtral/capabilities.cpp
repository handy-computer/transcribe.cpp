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
    // synthesize a translate instruction) and a general free-text prompt.
    // The GGUF's stt.capability.translate=true is read into supports_translate
    // by read_capability_kv at load; default it true here as a fallback.
    caps.supports_translate = true;

    // Per-run cancellation only. INITIAL_PROMPT stays false: Voxtral's
    // transcription template has no recognition-biasing slot for
    // transcribe_run_params::context — the only place free text can go is
    // instruct mode, which replaces the [TRANSCRIBE] task (a different
    // output contract). If free-text instruct is ever exposed, it belongs
    // on a voxtral run extension, not the core context field.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::voxtral
