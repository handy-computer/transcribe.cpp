// transcribe-decode-budget.h - shared per-run autoregressive decode budget.
//
// INTERNAL. Header-only, like the ABI helpers in transcribe-abi.h.
//
// Autoregressive families used to hardcode their generation budget as a
// constant (256 / 512 tokens) that did not depend on the audio at all, while
// the up-front input gate accepted clips orders of magnitude longer than that
// many tokens could describe. Any clip whose natural transcript outran the
// constant came back as TRANSCRIBE_ERR_OUTPUT_TRUNCATED with a partial
// transcript, even though the decoder context had ample room left. See
// docs/input-limits.md.
//
// The budget must track the input instead. Speech yields fewer text tokens
// than the encoder yields audio tokens, so the audio-token count is a safe
// upper bound on the transcript length — the estimate voxtral has shipped
// with since its introduction, generalized here so every family shares it.
//
// This is deliberately NOT a public run parameter. The only caller-facing
// knob is transcribe_session_params::n_ctx, which lowers `ceiling` and so
// lowers the budget with it.

#pragma once

#include <algorithm>

namespace transcribe {

// Speech-rate bound on transcript length, in text tokens per second of audio.
//
// Deliberately generous: measured English BPE runs ~3.4 tokens/sec, and CJK is
// denser, so this keeps a wide margin. It also matches what the 80 ms-per-token
// encoders were already getting from their raw audio-token count (12.5/sec), so
// adopting the duration form below leaves those families where they were.
constexpr int k_transcript_tokens_per_sec = 12;

// Predicted transcript length, in tokens, for an utterance the encoder turned
// into `audio_tokens` outputs at `ms_per_audio_token` each.
//
// The raw audio-token count is NOT a portable proxy for transcript length:
// encoders differ ~6x in rate. Most emit one token per 80 ms (12.5/sec, safely
// above any speech rate), but funasr_nano's LFR frontend stacks frames and
// emits one per ~480 ms (2.08/sec) — below the text-token rate, so using its
// audio-token count under-predicts and the budget truncates a transcript the
// context had room for. Converting to seconds first removes the encoder rate
// from the estimate entirely.
//
// A non-positive `ms_per_audio_token` means the family did not publish its
// rate; fall back to the audio-token count, which is what every family used
// before this became rate-aware.
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
//   predicted     the family's upper-bound estimate of transcript length,
//                 in tokens. For most families this is the audio-token
//                 count; a family whose output carries more than the
//                 transcript (moss emits speaker markers) scales it up.
//   floor_tokens  never plan for fewer than this. Each family passes its
//                 historical fixed budget, so a clip that fits today keeps
//                 byte-identical behavior, and the generation reserve that
//                 transcribe_capabilities::max_audio_ms subtracts (via
//                 transcribe_model::LimitsBasis::gen_reserve) stays exact.
//   t_prompt      prompt tokens already committed to the decoder context.
//                 Includes the audio embeddings for families that put audio
//                 in-context; 0 for encoder-decoder families whose audio
//                 lives in a separate cross-attention cache.
//   ceiling       decoder context ceiling in tokens, already lowered (never
//                 raised) by transcribe_session_params::n_ctx.
//
// Returns the budget clamped to the context actually left, never negative.
// A zero return means the prompt already fills the ceiling; callers gate
// that case up front (INPUT_TOO_LONG) rather than entering the step loop.
inline int pick_decode_budget(int predicted, int floor_tokens, int t_prompt, int ceiling) {
    int       budget = std::max(floor_tokens, predicted);
    const int room   = ceiling - t_prompt;
    if (budget > room) {
        budget = room;
    }
    return budget > 0 ? budget : 0;
}

}  // namespace transcribe
