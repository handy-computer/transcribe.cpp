// arch/crisperwhisper/capabilities.cpp - CrisperWhisper capability defaults.

#include "crisperwhisper.h"

namespace transcribe::crisperwhisper {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Multilingual by inheritance: the checkpoint carries all 99 <|lang|>
    // tokens. read_capability_kv() in the load path overrides from GGUF.
    caps.supports_language_detect = true;

    // Translation is NOT advertised. <|translate|> survives in the vocab and
    // generation_config.task_to_id, but the author package exposes no
    // translate path and neither the model card nor its DOCS claims one, so
    // there is no reference to validate against. See the family doc's
    // Capability Validation table (Translate — OUT OF SCOPE).
    caps.supports_translate = false;

    // supports_streaming stays false: long-form is offline overlapped-window
    // continuation, not a streaming contract.

    // WORD, not SEGMENT. The model is always decoded with <|notimestamps|>
    // and never emits timestamp tokens, so there is no segment-timestamp
    // stream to parse; timing comes from the Viterbi aligner over supervised
    // cross-attention. Capping here makes the dispatcher reject a
    // SEGMENT-grain request rather than silently returning whole-chunk spans.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_WORD;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_LONG_FORM, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);

    // Deliberately NOT set, each with a reason recorded in the family doc:
    //   INITIAL_PROMPT        — the prompt slot is owned by the mode tags and
    //                           the <ctx> continuation block; there is no
    //                           free-text conditioning surface upstream.
    //   TEMPERATURE_FALLBACK  — the reference samples (do_sample, escalating
    //                           ladder, per-attempt seed), so parity needs an
    //                           RNG-stream contract this repo does not have.
}

}  // namespace transcribe::crisperwhisper
