// arch/granite/capabilities.cpp - family invariants.

#include "granite.h"

namespace transcribe::granite {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // Word timestamps are advertised on the -plus variant (the
    // timestamps prompt template emits inline `<|n.nn|> word` markers).
    // The 1b / 2b variants only emit transcript text. The runtime
    // distinguishes via a per-variant capability KV (the converter sets
    // stt.capability.word_timestamps appropriately); apply_family_invariants
    // sets the default upper bound here so the loader's
    // read_capability_kv step can lower it for variants that do not
    // expose timestamps.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_WORD;

    // Translation is advertised on the 1b / 2b variants via a separate
    // chat-template prompt. The -plus variant drops the translation
    // capability (its model card lists ASR + speaker diarization, not
    // translation). The default here is true so the loader can lower
    // it per-variant from the GGUF KV stt.capability.translation.
    caps.supports_translate = true;

    // Cancellation is wired at the per-run level. Whisper-specific
    // long-form / temperature-fallback / initial-prompt features do
    // not apply to granite (the LLM produces a single greedy
    // transcript per utterance).
    caps.supports_initial_prompt       = false;
    caps.supports_temperature_fallback = false;
    caps.supports_long_form            = false;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::granite
