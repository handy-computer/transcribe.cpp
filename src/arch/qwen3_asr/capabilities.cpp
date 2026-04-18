// arch/qwen3_asr/capabilities.cpp - family invariants.

#include "qwen3_asr.h"

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;

    // Qwen3-ASR emits transcript text inside a Qwen-style chat response.
    // It does not surface timestamps of any kind for the transcript-only
    // task. The sibling Qwen3-ForcedAligner produces word timestamps,
    // but that is a separate checkpoint with a different head contract
    // and would register as its own family (qwen3_forced_aligner) when
    // ported.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // The publisher documents language detection (the LM prefixes its
    // transcript with "language X"). Translation is not an advertised
    // capability for Qwen3-ASR; callers get transcript-only output in
    // the audio's source language.
    caps.supports_translate = false;
}

} // namespace transcribe::qwen3_asr
