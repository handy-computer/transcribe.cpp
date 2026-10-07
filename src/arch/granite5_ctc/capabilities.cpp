// arch/granite5_ctc/capabilities.cpp - family invariants.

#include "granite5_ctc.h"

namespace transcribe::granite5_ctc {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // No timestamps. max_timestamp_kind stays at the zero-initialized
    // TRANSCRIBE_TIMESTAMPS_NONE, so transcribe_run() rejects any finer
    // request with TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS.
    //
    // CTC frame alignment does hand us an emission index per token, and
    // an earlier revision published word timestamps built from it. It was
    // withdrawn: CTC is peaky, so a token occupies exactly the one frame
    // where it wins, and a word's end time came out as "emitting frame +
    // one frame". Nearly every word therefore reported exactly 80 ms of
    // duration regardless of how long it actually took to say --
    // "americans", "impossible" and "backwards" all measured one frame.
    // Onsets were sound; ends were not, and upstream neither advertises
    // nor emits timings, so there is no reference alignment to validate
    // against. Publishing a duration we know to be wrong is worse than
    // publishing none. Reinstating this needs a real word-end rule (the
    // next token's emitting frame, or the last frame before the next
    // word's onset) plus something to check it against.

    // English-only ASR. No translation, no language identification, no
    // diarizer, and the frontend is non-causal (centered STFT plus a
    // per-utterance log-mel floor), so no streaming either.
    caps.supports_translate = false;

    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

}  // namespace transcribe::granite5_ctc
