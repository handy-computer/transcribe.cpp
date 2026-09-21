// transcribe-decode-budget.h - shared per-run autoregressive decode budget.
//
// INTERNAL. Header-only, like the ABI helpers in transcribe-abi.h.
//
// Autoregressive families used to cap generation at a constant (256 / 512)
// that ignored the audio, while the input gate accepted clips far longer than
// that many tokens could describe — so a long clip came back
// TRANSCRIBE_ERR_OUTPUT_TRUNCATED with the context still mostly free. The
// budget has to track the input instead. See docs/input-limits.md.
//
// This is deliberately NOT a public run parameter. The only caller-facing knob
// is transcribe_session_params::n_ctx, which lowers `ceiling` and the budget
// with it.

#pragma once

#include <algorithm>

namespace transcribe {

// Speech-rate bound on transcript length, in text tokens per second of audio.
// Generous on purpose: English BPE measures ~3.4/sec and CJK is denser. 12 also
// matches what the 80 ms-per-token encoders already got from their raw
// audio-token count (12.5/sec), so the duration form leaves them where they were.
constexpr int k_transcript_tokens_per_sec = 12;

// Predicted transcript length, in tokens, for an utterance the encoder turned
// into `audio_tokens` outputs at `ms_per_audio_token` each.
//
// The raw audio-token count is NOT a portable proxy: encoder rates differ ~6x.
// Most emit one token per 80 ms (12.5/sec, safely above any speech rate), but
// funasr_nano's LFR frontend emits one per ~480 ms (2.08/sec) — below the
// text-token rate, so its count under-predicts and the budget truncates a
// transcript the context had room for. Going via seconds removes the encoder
// rate from the estimate.
//
// A non-positive `ms_per_audio_token` means the family published no rate; fall
// back to the audio-token count, the pre-rate-aware behavior.
inline int predict_transcript_tokens(int audio_tokens, double ms_per_audio_token) {
    if (audio_tokens <= 0) {
        return 0;
    }
    if (!(ms_per_audio_token > 0.0)) {
        return audio_tokens;
    }
    const double seconds   = static_cast<double>(audio_tokens) * ms_per_audio_token / 1000.0;
    const double predicted = seconds * k_transcript_tokens_per_sec;
    // Clamp into int range; callers clamp again to the context actually left.
    if (predicted <= 0.0) {
        return 0;
    }
    constexpr double k_int_max = 2147483647.0;
    return predicted >= k_int_max ? 2147483647 : static_cast<int>(predicted);
}

// Per-run decode budget, in transcript tokens.
//
//   predicted     upper-bound estimate of transcript length, normally from
//                 predict_transcript_tokens() above. A family whose output
//                 carries more than the transcript (moss emits speaker
//                 markers) scales it up.
//   floor_tokens  never plan for fewer. Each family passes its historical
//                 fixed budget, so a clip that fits today stays
//                 byte-identical and the reserve max_audio_ms subtracts (via
//                 transcribe_model::LimitsBasis::gen_reserve) stays exact.
//   t_prompt      prompt tokens already in the decoder context. Includes the
//                 audio embeddings for in-context families; 0 for
//                 encoder-decoder families whose audio lives in a separate
//                 cross-attention cache.
//   ceiling       decoder context ceiling, already lowered (never raised) by
//                 transcribe_session_params::n_ctx.
//
// Clamped to the context actually left, never negative. Zero means the prompt
// already fills the ceiling; callers gate that up front (INPUT_TOO_LONG).
inline int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
    int       budget = std::max(floor_tokens, predicted);
    const int room   = ceiling - t_prompt;
    if (budget > room) {
        budget = room;
    }
    return budget > 0 ? budget : 0;
}

}  // namespace transcribe
