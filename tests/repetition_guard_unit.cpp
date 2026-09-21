// Repetition guard stop rule (pure host, no model). Too eager and it cuts real
// speech mid-utterance; too lax and a looping decode runs to the budget. See
// src/transcribe-repetition-guard.h.

#include "transcribe-repetition-guard.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void expect(const char * what, bool ok) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

std::vector<int32_t> seq(int first, int count) {
    std::vector<int32_t> out;
    for (int i = 0; i < count; ++i) {
        out.push_back(first + i);
    }
    return out;
}

std::vector<int32_t> repeat(const std::vector<int32_t> & block, int copies) {
    std::vector<int32_t> out;
    for (int c = 0; c < copies; ++c) {
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t> & b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

struct Decoded {
    std::vector<int32_t> ids;
    int                  stopped_at = -1;  // 1-based token count when the guard fired, or -1
};

// Feed `stream` one token at a time the way a decode loop does, calling the
// guard after every append.
Decoded decode(const std::vector<int32_t> & stream) {
    Decoded d;
    for (size_t i = 0; i < stream.size(); ++i) {
        d.ids.push_back(stream[i]);
        if (transcribe::stop_on_repetition(d.ids, "repetition_guard_unit")) {
            d.stopped_at = static_cast<int>(i + 1);
            return d;
        }
    }
    return d;
}

int tail_block(const std::vector<int32_t> & ids) {
    return transcribe::repeating_tail_block(ids.data(), static_cast<int>(ids.size()));
}

int budget_tail_block(const std::vector<int32_t> & ids) {
    return transcribe::repeating_tail_block(ids.data(), static_cast<int>(ids.size()),
                                            transcribe::k_budget_trim_min_copies, transcribe::k_budget_trim_min_tokens);
}

std::vector<int32_t> budget_trimmed(std::vector<int32_t> ids) {
    transcribe::trim_repetition_at_budget_stop(ids, "repetition_guard_unit");
    return ids;
}

}  // namespace

int main(void) {
    const std::vector<int32_t> prefix = seq(1000, 10);

    // Stop thresholds: 8 copies covering at least 64 tokens.
    expect("empty is not a loop", tail_block({}) == 0);
    expect("distinct tokens are not a loop", tail_block(seq(1, 200)) == 0);
    expect("1-token block x63 is not a loop", tail_block(repeat({ 7 }, 63)) == 0);
    expect("1-token block x64 is a loop", tail_block(repeat({ 7 }, 64)) == 1);
    expect("2-token block x31 is not a loop", tail_block(repeat({ 7, 8 }, 31)) == 0);
    expect("2-token block x32 is a loop", tail_block(repeat({ 7, 8 }, 32)) == 2);
    expect("4-token block x15 is not a loop", tail_block(repeat(seq(1, 4), 15)) == 0);
    expect("4-token block x16 is a loop", tail_block(repeat(seq(1, 4), 16)) == 4);
    expect("8-token block x7 is not a loop", tail_block(repeat(seq(1, 8), 7)) == 0);
    expect("8-token block x8 is a loop", tail_block(repeat(seq(1, 8), 8)) == 8);
    expect("20-token block x4 is not a loop", tail_block(repeat(seq(1, 20), 4)) == 0);
    expect("20-token block x7 is not a loop", tail_block(repeat(seq(1, 20), 7)) == 0);
    expect("20-token block x8 is a loop", tail_block(repeat(seq(1, 20), 8)) == 20);
    expect("64-token block x8 is a loop", tail_block(repeat(seq(1, 64), 8)) == 64);
    expect("65-token block is past the limit", tail_block(repeat(seq(1, 65), 10)) == 0);
    expect("smallest period wins", tail_block(repeat({ 7, 8 }, 64)) == 2);
    expect("loop after a prefix", tail_block(cat(prefix, repeat(seq(1, 10), 8))) == 10);
    expect("a broken final copy is not a loop", tail_block(cat(repeat(seq(1, 10), 9), { 99 })) == 0);

    // Budget-stop thresholds: 3 copies covering at least 32 tokens.
    expect("budget: 20-token block x2 is not a loop", budget_tail_block(repeat(seq(1, 20), 2)) == 0);
    expect("budget: 20-token block x3 is a loop", budget_tail_block(repeat(seq(1, 20), 3)) == 20);
    expect("budget: 8-token block x3 is not a loop", budget_tail_block(repeat(seq(1, 8), 3)) == 0);
    expect("budget: 8-token block x4 is a loop", budget_tail_block(repeat(seq(1, 8), 4)) == 8);
    expect("budget: 1-token block x31 is not a loop", budget_tail_block(repeat({ 7 }, 31)) == 0);
    expect("budget: 1-token block x32 is a loop", budget_tail_block(repeat({ 7 }, 32)) == 1);

    // Trimming keeps the prefix and one copy.
    {
        const std::vector<int32_t> ids = cat(prefix, repeat(seq(1, 10), 6));
        const int                  n   = transcribe::trim_repeating_tail(ids.data(), static_cast<int>(ids.size()), 10);
        expect("trim keeps prefix + one copy", n == 20);
        expect("trim of block 0 is a no-op", transcribe::trim_repeating_tail(ids.data(), 70, 0) == 70);
    }

    // Stop reasons reported by the batched step loops.
    expect("eos row is OK", transcribe::decode_stop_status(transcribe::k_stop_eos) == TRANSCRIBE_OK);
    expect("budget row is OUTPUT_TRUNCATED",
           transcribe::decode_stop_status(transcribe::k_stop_budget) == TRANSCRIBE_ERR_OUTPUT_TRUNCATED);
    expect("repetition row is OUTPUT_REPETITION",
           transcribe::decode_stop_status(transcribe::k_stop_repetition) == TRANSCRIBE_ERR_OUTPUT_REPETITION);

    if (!transcribe::repetition_guard_enabled()) {
        std::fprintf(stdout, "repetition_guard_unit: guard disabled by env, skipping decode-loop cases\n");
        return g_failures > 0 ? 1 : 0;
    }

    // Emphatic repetition mid-utterance must not stop the decode.
    {
        const std::vector<int32_t> stream = cat(cat(prefix, repeat(seq(1, 5), 3)), seq(2000, 40));
        const Decoded              d      = decode(stream);
        expect("5-token phrase x3 mid-utterance keeps decoding", d.stopped_at == -1 && d.ids == stream);
    }
    {
        const std::vector<int32_t> stream = cat(repeat(seq(1, 4), 3), seq(2000, 40));
        const Decoded              d      = decode(stream);
        expect("4-token phrase x3 keeps decoding", d.stopped_at == -1 && d.ids == stream);
    }
    {
        const std::vector<int32_t> stream = cat(repeat({ 42, 43 }, 6), seq(2000, 20));
        const Decoded              d      = decode(stream);
        expect("'no, no, no, no, no, no' keeps decoding", d.stopped_at == -1 && d.ids == stream);
    }

    {
        // A sung line or chant repeated a handful of times, then more speech.
        const std::vector<int32_t> stream = cat(cat(prefix, repeat(seq(1, 12), 7)), seq(2000, 40));
        const Decoded              d      = decode(stream);
        expect("12-token line x7 keeps decoding", d.stopped_at == -1 && d.ids == stream);
    }

    // A runaway loop stops as soon as it qualifies, keeping prefix + one copy.
    {
        const std::vector<int32_t> loop = seq(1, 10);
        const Decoded              d    = decode(cat(prefix, repeat(loop, 40)));
        expect("10-token loop stops after 8 copies", d.stopped_at == 10 + 8 * 10);
        expect("10-token loop keeps prefix + one copy", d.ids == cat(prefix, loop));
    }
    {
        const std::vector<int32_t> loop = seq(1, 30);
        const Decoded              d    = decode(cat(prefix, repeat(loop, 10)));
        expect("30-token sentence loop stops", d.stopped_at == 10 + 8 * 30);
        expect("30-token sentence loop keeps prefix + one copy", d.ids == cat(prefix, loop));
    }
    {
        const Decoded d = decode(repeat({ 5 }, 100));
        expect("1-token loop stops at 64", d.stopped_at == 64);
        expect("1-token loop keeps one token", d.ids == std::vector<int32_t>{ 5 });
    }

    // At a budget stop, a shorter repeating tail is dropped; anything below
    // the budget bar is left alone.
    {
        const std::vector<int32_t> loop = seq(1, 20);
        expect("budget stop drops a 3-copy tail", budget_trimmed(cat(prefix, repeat(loop, 3))) == cat(prefix, loop));
        const std::vector<int32_t> twice = cat(prefix, repeat(loop, 2));
        expect("budget stop keeps a 2-copy tail", budget_trimmed(twice) == twice);
        const std::vector<int32_t> no_no = cat(prefix, repeat({ 42, 43 }, 6));
        expect("budget stop keeps 'no, no, no, no, no, no'", budget_trimmed(no_no) == no_no);
        const std::vector<int32_t> clean = cat(prefix, seq(2000, 40));
        expect("budget stop leaves non-repeating output alone", budget_trimmed(clean) == clean);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "repetition_guard_unit: %d failures\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "repetition_guard_unit: ok\n");
    return 0;
}
