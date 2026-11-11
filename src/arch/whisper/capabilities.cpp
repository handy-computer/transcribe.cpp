// arch/whisper/capabilities.cpp - Whisper ASR capability defaults.

#include "whisper.h"

namespace transcribe::whisper {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Multilingual defaults; read_capability_kv() in the load path overrides
    // these from GGUF (.en variants carry lang_detect/translate=false).
    caps.supports_language_detect = true;
    caps.supports_translate       = true;
    // supports_streaming left at its zero-init default (false).

    // Segment timestamps are available for all variants. The GGUF loader
    // advertises word timestamps when it can identify an alignment preset.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;

    // Run knobs are reached through transcribe_whisper_run_ext. No PNC/ITN
    // runtime toggle — whisper emits whatever its training distribution does.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_INITIAL_PROMPT, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_LONG_FORM, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
    // Generic vocabulary (", "-joined terms) and context prompt, both in the
    // <|startofprev|> slot (prompting A/B, notes/prompting-ab-results.md).
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_VOCABULARY, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CONTEXT_PROMPT, true);
    // Transcript prefix after the SOT sequence (openai DecodingOptions.prefix),
    // first window only.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_TRANSCRIPT_PREFIX, true);
}

}  // namespace transcribe::whisper
