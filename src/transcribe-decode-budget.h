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
