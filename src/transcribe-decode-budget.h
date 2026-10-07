// Shared helpers for sizing autoregressive decode budgets.

#pragma once

#include <algorithm>

namespace transcribe {

// Conservative multilingual transcript estimate, in tokens per second.
constexpr int k_transcript_tokens_per_sec = 12;

// Fall back to the encoder-token count when its duration is unknown.
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
