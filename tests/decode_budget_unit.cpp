// Per-run decode budget rule (pure host, no model).
//
// Every autoregressive family used to hardcode its generation budget as a
// constant that ignored the audio entirely (qwen3_asr 256, canary 512, ...).
// The up-front input gate meanwhile accepted clips orders of magnitude longer
// than that many tokens could describe — qwen3_asr advertises 87 minutes of
// audio against a 256-token output cap — so any clip past roughly a minute of
// speech came back TRANSCRIBE_ERR_OUTPUT_TRUNCATED with context to spare.
//
// transcribe::pick_decode_budget replaces those constants. This test pins the
// two properties the families depend on, because getting either wrong is
// silent: too low and long clips truncate again; too high and the KV
// allocation (112 KiB per token on qwen3-asr) balloons on every run.
//
//   1. Never below the family's floor  -> a clip that fits today is unchanged,
//      and transcribe_capabilities::max_audio_ms (which subtracts that same
//      floor via LimitsBasis::gen_reserve) stays exact.
//   2. Never past the context left     -> prompt + budget always fits the
//      ceiling, which is what transcribe_session_params::n_ctx lowers.

#include "transcribe-decode-budget.h"

#include <cstdio>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

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
    // ---- Property 1: the floor holds for short audio. ----
    // A clip whose audio-token count is below the family's historical fixed
    // budget must still get that budget, so its decode is byte-identical to
    // what shipped before.
    check_budget("short clip keeps the floor", /*predicted=*/10, /*floor=*/256, /*t_prompt=*/64,
                 /*ceiling=*/65536, /*expected=*/256);
    check_budget("floor applies at zero audio", 0, 256, 64, 65536, 256);
    check_budget("canary floor", 300, 512, 6, 1024, 512);

    // ---- Property 2: the budget scales past the floor. ----
    // This is the fix. qwen3-asr at 80 ms per audio token: a 5-minute clip is
    // 3824 audio tokens, which used to decode under a flat 256-token cap.
    check_budget("5 min qwen3-asr scales", /*predicted=*/3824, /*floor=*/256, /*t_prompt=*/3872,
                 /*ceiling=*/65536, /*expected=*/3824);
    check_budget("20 min qwen3-asr scales", 15000, 256, 15048, 65536, 15000);

    // ---- Property 3: the context ceiling always wins. ----
    // canary is the tight case: a 400 s clip is ~5000 encoder frames but the
    // decoder self-KV is only 1024, so the budget clamps to what is left.
    check_budget("canary clamps to dec ctx", /*predicted=*/5000, /*floor=*/512, /*t_prompt=*/6,
                 /*ceiling=*/1024, /*expected=*/1018);
    check_budget("clamp beats the floor too", 10, 512, 900, 1024, 124);

    // ---- Property 4: n_ctx is the knob. ----
    // Lowering transcribe_session_params::n_ctx lowers `ceiling`, and the
    // budget must follow it down. Same inputs as the 5-minute case above.
    check_budget("full n_ctx leaves the audio-sized budget intact", 3824, 256, 3872, 8192, 3824);
    check_budget("lowered n_ctx lowers the budget", 3824, 256, 3872, 6000, 2128);
    check_budget("n_ctx below the floor still clamps", 3824, 256, 3872, 4000, 128);

    // ---- Property 5: never negative. ----
    // A prompt that already fills the ceiling yields 0, not a negative step
    // count. Families gate this case up front with INPUT_TOO_LONG; the helper
    // must not hand a negative loop bound to a step loop regardless.
    check_budget("prompt exactly fills ceiling", 3824, 256, 1024, 1024, 0);
    check_budget("prompt overruns ceiling", 3824, 256, 2048, 1024, 0);

    if (g_failures > 0) {
        std::fprintf(stderr, "decode_budget_unit: %d failures\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "decode_budget_unit: ok\n");
    return 0;
}
