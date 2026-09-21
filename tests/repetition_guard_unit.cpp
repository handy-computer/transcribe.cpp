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

}  // namespace

int main(void) {
    const std::vector<int32_t> prefix = seq(1000, 10);

    // Detection thresholds.
    expect("empty is not a loop", tail_block({}) == 0);
    expect("distinct tokens are not a loop", tail_block(seq(1, 200)) == 0);
    expect("1-token block x31 is not a loop", tail_block(repeat({7}, 31)) == 0);
    expect("1-token block x32 is a loop", tail_block(repeat({7}, 32)) == 1);
    expect("2-token block x16 is a loop", tail_block(repeat({7, 8}, 16)) == 2);
    expect("4-token block x7 is not a loop", tail_block(repeat(seq(1, 4), 7)) == 0);
    expect("4-token block x8 is a loop", tail_block(repeat(seq(1, 4), 8)) == 4);
    expect("8-token block x3 is not a loop", tail_block(repeat(seq(1, 8), 3)) == 0);
    expect("8-token block x4 is a loop", tail_block(repeat(seq(1, 8), 4)) == 8);
    expect("20-token block x4 is a loop", tail_block(repeat(seq(1, 20), 4)) == 20);
    expect("64-token block x4 is a loop", tail_block(repeat(seq(1, 64), 4)) == 64);
    expect("65-token block is past the limit", tail_block(repeat(seq(1, 65), 6)) == 0);
    expect("smallest period wins", tail_block(repeat({7, 8}, 32)) == 2);
    expect("loop after a prefix", tail_block(cat(prefix, repeat(seq(1, 10), 4))) == 10);
    expect("a broken final copy is not a loop",
           tail_block(cat(repeat(seq(1, 10), 5), {99})) == 0);

    // Trimming keeps the prefix and one copy.
    {
        const std::vector<int32_t> ids = cat(prefix, repeat(seq(1, 10), 6));
        const int n = transcribe::trim_repeating_tail(ids.data(), static_cast<int>(ids.size()), 10);
        expect("trim keeps prefix + one copy", n == 20);
        expect("trim of block 0 is a no-op", transcribe::trim_repeating_tail(ids.data(), 70, 0) == 70);
    }

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
        const std::vector<int32_t> stream = cat(repeat({42, 43}, 6), seq(2000, 20));
        const Decoded              d      = decode(stream);
        expect("'no, no, no, no, no, no' keeps decoding", d.stopped_at == -1 && d.ids == stream);
    }

    // A runaway loop stops as soon as it qualifies, keeping prefix + one copy.
    {
        const std::vector<int32_t> loop   = seq(1, 10);
        const Decoded              d      = decode(cat(prefix, repeat(loop, 40)));
        expect("10-token loop stops after 4 copies", d.stopped_at == 10 + 4 * 10);
        expect("10-token loop keeps prefix + one copy", d.ids == cat(prefix, loop));
    }
    {
        const std::vector<int32_t> loop = seq(1, 30);
        const Decoded              d    = decode(cat(prefix, repeat(loop, 10)));
        expect("30-token sentence loop stops", d.stopped_at == 10 + 4 * 30);
        expect("30-token sentence loop keeps prefix + one copy", d.ids == cat(prefix, loop));
    }
    {
        const Decoded d = decode(repeat({5}, 100));
        expect("1-token loop stops at 32", d.stopped_at == 32);
        expect("1-token loop keeps one token", d.ids == std::vector<int32_t>{5});
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "repetition_guard_unit: %d failures\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "repetition_guard_unit: ok\n");
    return 0;
}
