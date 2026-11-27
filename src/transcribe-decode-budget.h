// Shared helpers for sizing autoregressive decode budgets.

#pragma once

#include <algorithm>

namespace transcribe {

// Default transcript estimate, in tokens per second. Calibrated on
// Latin-script languages: English realizes ~1.2 tokens per word (~2.6 tok/s
// of speech), so 12 leaves generous headroom. Dense-script languages
// fragment far more in multilingual tokenizers (measured on Qwen3-ASR's
// tokenizer: Bengali 6.0 tokens/word, Hindi 4.6, vs English 1.2), so
// families that transcribe them pass a higher rate.
constexpr int k_transcript_tokens_per_sec = 12;

// Fall back to the encoder-token count when its duration is unknown.
inline int predict_transcript_tokens(int audio_tokens, double ms_per_audio_token,
                                     int tokens_per_sec = k_transcript_tokens_per_sec) {
    if (audio_tokens <= 0) {
        return 0;
    }
    if (!(ms_per_audio_token > 0.0)) {
        return audio_tokens;
    }
    const double seconds   = static_cast<double>(audio_tokens) * ms_per_audio_token / 1000.0;
    const double predicted = seconds * tokens_per_sec;
    if (predicted <= 0.0) {
        return 0;
    }
    constexpr double k_int_max = 2147483647.0;
    return predicted >= k_int_max ? 2147483647 : static_cast<int>(predicted);
}

// Apply the family floor without exceeding the available context.
inline int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
    int       budget = std::max(floor_tokens, predicted);
    const int room   = ceiling - t_prompt;
    if (budget > room) {
        budget = room;
    }
    return budget > 0 ? budget : 0;
}

}  // namespace transcribe
