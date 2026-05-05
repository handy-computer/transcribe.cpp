// arch/whisper/capabilities.cpp - Whisper ASR capability defaults.

#include "whisper.h"

namespace transcribe::whisper {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // Multilingual whisper variants support both lang-detect and the
    // <|translate|> task token. The shared read_capability_kv() call
    // in the load path will override these from GGUF if the converter
    // emitted stt.capability.{lang_detect,translate}; the defaults
    // here are the correct answer for the multilingual variants we
    // ship first. The .en variants will carry
    // stt.capability.{lang_detect,translate}=false and overwrite.
    caps.supports_language_detect = true;
    caps.supports_translate       = true;
    caps.supports_streaming       = false;

    // Segment timestamps via the timestamp-token stream (<|t=0.00|> …
    // <|t=30.00|>, ids 50364+). Word-level timestamps via DTW over
    // cross-attention alignment heads are not yet implemented — the
    // model is capable, but the DTW path is a Stage-7 follow-up. Cap
    // at SEGMENT so the dispatcher rejects premature requests for
    // WORD-grain timing rather than silently falling back.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;

    // Stage 2 lights up temperature fallback + long-form decoding +
    // per-chunk decoding trace. Stage 1 already shipped cancellation
    // (abort callback between chunks and between decode steps). Stage 3
    // wires initial_prompt (text or pre-tokenized) and
    // condition_on_prev_tokens (cross-chunk coherence under
    // <|startofprev|>) — both surfaces are reachable through
    // transcribe_whisper_params.
    caps.supports_initial_prompt       = true;
    caps.supports_temperature_fallback = true;
    caps.supports_long_form            = true;
    caps.supports_cancellation         = true;
}

} // namespace transcribe::whisper
