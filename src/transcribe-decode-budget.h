// transcribe-decode-budget.h - shared per-run autoregressive decode budget.
//
// The budget has to track the input: a flat per-family constant returns
// TRANSCRIBE_ERR_OUTPUT_TRUNCATED on a long clip with the decoder context still
// mostly free. Deliberately not a public run parameter — the only caller-facing
// knob is transcribe_session_params::n_ctx. See docs/input-limits.md.

#pragma once

#include <algorithm>

namespace transcribe {

// Speech-rate bound on transcript length, in text tokens per second of audio.
// Generous on purpose: English BPE measures ~3.4/sec and CJK is denser.
constexpr int k_transcript_tokens_per_sec = 12;

// Predicted transcript length for `audio_tokens` encoder outputs at
// `ms_per_audio_token` each. Going via seconds is required, not cosmetic:
// encoder rates differ ~6x, and funasr_nano's LFR frontend emits one token per
// ~480 ms — below the text-token rate, so its raw count under-predicts.
//
// A non-positive rate means the family published none; fall back to the count.
inline int predict_transcript_tokens(int audio_tokens, double ms_per_audio_token) {
    if (audio_tokens <= 0) {
        return 0;
    }
    if (!(ms_per_audio_token > 0.0)) {
        return audio_tokens;
    }
    const double seconds   = static_cast<double>(audio_tokens) * ms_per_audio_token / 1000.0;
    const double predicted = seconds * k_transcript_tokens_per_sec;
    if (predicted <= 0.0) {
        return 0;
    }
    constexpr double k_int_max = 2147483647.0;
    return predicted >= k_int_max ? 2147483647 : static_cast<int>(predicted);
}

// Per-run decode budget: `predicted` raised to `floor_tokens` and clamped to the
// context left under `ceiling`. `floor_tokens` is the family's historical fixed
// budget, so short clips decode byte-identically and the reserve max_audio_ms
// subtracts (transcribe_model::LimitsBasis::gen_reserve) stays exact.
inline int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
    int       budget = std::max(floor_tokens, predicted);
    const int room   = ceiling - t_prompt;
    if (budget > room) {
        budget = room;
    }
    return budget > 0 ? budget : 0;
}

}  // namespace transcribe
