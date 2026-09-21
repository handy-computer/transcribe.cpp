// Per-run decode budget rule (pure host, no model). Getting it wrong is silent
// both ways: too low and long clips truncate with context to spare, too high
// and the KV allocation balloons every run. See docs/input-limits.md.

#include "transcribe-decode-budget.h"

#include <cstdio>

namespace {

int g_failures = 0;

void check_predict(const char * what, int audio_tokens, double ms_per_audio_token, int expected) {
    const int got = transcribe::predict_transcript_tokens(audio_tokens, ms_per_audio_token);
    if (got != expected) {
        std::fprintf(stderr, "FAIL %s: predict_transcript_tokens(%d, %.3f) = %d, expected %d\n", what, audio_tokens,
                     ms_per_audio_token, got, expected);
        ++g_failures;
    }
}

void check_budget(const char * what, int predicted, int floor_tokens, int t_prompt, int ceiling, int expected) {
    const int got = transcribe::pick_decode_budget(predicted, floor_tokens, t_prompt, ceiling);
    if (got != expected) {
        std::fprintf(stderr, "FAIL %s: pick_decode_budget(%d, %d, %d, %d) = %d, expected %d\n", what, predicted,
                     floor_tokens, t_prompt, ceiling, got, expected);
        ++g_failures;
    }
}

}  // namespace

int main(void) {
    // Duration-based: two encoders that heard the same 197 s must agree.
    check_predict("80 ms encoder, 197 s", /*audio_tokens=*/2463, /*ms_per_audio_token=*/80.0,
                  /*expected=*/2364);
    check_predict("480 ms LFR encoder, same 197 s", /*audio_tokens=*/410, /*ms_per_audio_token=*/480.0,
                  /*expected=*/2361);

    // An unpublished rate falls back to the audio-token count.
    check_predict("unknown rate falls back", 2463, 0.0, 2463);
    check_predict("negative rate falls back", 2463, -1.0, 2463);
    check_predict("no audio", 0, 80.0, 0);

    // The floor holds for short audio, so those decodes are unchanged.
    check_budget("short clip keeps the floor", /*predicted=*/10, /*floor=*/256, /*t_prompt=*/64,
                 /*ceiling=*/65536, /*expected=*/256);
    check_budget("floor applies at zero audio", 0, 256, 64, 65536, 256);
    check_budget("canary floor", 300, 512, 6, 1024, 512);

    // Past the floor the budget scales with the audio. This is the fix.
    check_budget("5 min qwen3-asr scales", /*predicted=*/3824, /*floor=*/256, /*t_prompt=*/3872,
                 /*ceiling=*/65536, /*expected=*/3824);
    check_budget("20 min qwen3-asr scales", 15000, 256, 15048, 65536, 15000);

    // The ceiling always wins: canary sees ~5000 encoder frames into a 1024 self-KV.
    check_budget("canary clamps to dec ctx", /*predicted=*/5000, /*floor=*/512, /*t_prompt=*/6,
                 /*ceiling=*/1024, /*expected=*/1018);
    check_budget("clamp beats the floor too", 10, 512, 900, 1024, 124);

    // n_ctx is the knob: lowering `ceiling` lowers the budget with it.
    check_budget("full n_ctx leaves the audio-sized budget intact", 3824, 256, 3872, 8192, 3824);
    check_budget("lowered n_ctx lowers the budget", 3824, 256, 3872, 6000, 2128);
    check_budget("n_ctx below the floor still clamps", 3824, 256, 3872, 4000, 128);

    // Never a negative step count.
    check_budget("prompt exactly fills ceiling", 3824, 256, 1024, 1024, 0);
    check_budget("prompt overruns ceiling", 3824, 256, 2048, 1024, 0);

    if (g_failures > 0) {
        std::fprintf(stderr, "decode_budget_unit: %d failures\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "decode_budget_unit: ok\n");
    return 0;
}
