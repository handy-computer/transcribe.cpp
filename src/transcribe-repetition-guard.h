// Stop rule for greedy autoregressive decoders stuck repeating themselves.
//
// Greedy argmax has no way out of a self-reinforcing state: once the likeliest
// continuation of a phrase is the phrase itself, it repeats until the decode
// budget runs out. The guard works on token ids only, so it fits decoders that
// read back a device-side argmax, and leaves output that never loops unchanged.
//
// Two bars. Stopping early cuts off whatever the audio said after the loop, so
// it needs strong evidence: many copies. A decode that already hit its budget
// has failed anyway, so dropping a repeating tail needs less.

#pragma once

#include "transcribe-env.h"
#include "transcribe-log.h"
#include "transcribe.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace transcribe {

// When a block repeating at the tail counts as a loop. The evidence is how
// many tokens repeat verbatim, so a block needs `copies` copies spanning at
// least `min_tokens`, but once that span passes `max_tokens` it only has to
// cover `max_tokens`, with at least `min_copies` copies. Short blocks need many
// more copies (a 1-token block needs 64 to stop), so emphatic and sung
// repetition survives; a paragraph-length loop stops before the budget.
struct RepeatBar {
    int max_block;   // longest block looked for
    int copies;      // copies a block needs...
    int min_tokens;  // ...spanning at least this many tokens...
    int max_tokens;  // ...or at most this many...
    int min_copies;  // ...in no fewer than this many copies
};

// Stop mid-decode: 8 copies of a block up to 27 tokens, tapering to 4 copies
// of a 48-128-token block (192-512 tokens), which fits the decode budgets.
constexpr RepeatBar k_stop_bar        = { 128, 8, 64, 192, 4 };
// Trim at a budget stop: 3 copies spanning at least 32 tokens (max_tokens
// never binds). The trim runs once per decode, so it looks for longer blocks.
constexpr RepeatBar k_budget_trim_bar = { 256, 3, 32, 256 * 3, 3 };

// Copies of a `block`-token block that make a loop under `bar`.
constexpr int repeat_copies_needed(const RepeatBar & bar, int block) {
    const int span = std::min(std::max(bar.copies * block, bar.min_tokens), bar.max_tokens);
    return std::max(bar.min_copies, (span + block - 1) / block);
}

// Length of the block repeating at the end of ids[0, n), or 0.
inline int repeating_tail_block(const int32_t * ids, int n, const RepeatBar & bar = k_stop_bar) {
    for (int block = 1; block <= bar.max_block; ++block) {
        const int copies = repeat_copies_needed(bar, block);
        const int span   = block * copies;
        if (span > n) {
            continue;
        }
        bool periodic = true;
        for (int i = n - 1; i >= n - span + block; --i) {
            if (ids[i] != ids[i - block]) {
                periodic = false;
                break;
            }
        }
        if (periodic) {
            return block;
        }
    }
    return 0;
}

// Length of ids[0, n) with every copy of the tail block but the first dropped.
inline int trim_repeating_tail(const int32_t * ids, int n, int block) {
    while (block > 0 && n >= 2 * block && std::equal(ids + n - block, ids + n, ids + n - 2 * block)) {
        n -= block;
    }
    return n;
}

// TRANSCRIBE_NO_REPETITION_GUARD=1 turns the guard off (reference parity).
inline bool repetition_guard_enabled() {
    static const bool enabled = !env::flag("TRANSCRIBE_NO_REPETITION_GUARD");
    return enabled;
}

// Call after appending a token. On a loop, trims `ids` to one copy, logs at
// `level` (a WARN unless the decode is an interim one) tagged `who`, and
// returns true: the caller stops decoding and reports
// TRANSCRIBE_ERR_OUTPUT_REPETITION, since whatever the audio said after the
// loop was never decoded.
inline bool stop_on_repetition(std::vector<int32_t> & ids,
                               const char *           who,
                               transcribe_log_level   level = TRANSCRIBE_LOG_LEVEL_WARN) {
    if (!repetition_guard_enabled()) {
        return false;
    }
    const int n     = static_cast<int>(ids.size());
    const int block = repeating_tail_block(ids.data(), n);
    if (block == 0) {
        return false;
    }
    ids.resize(static_cast<size_t>(trim_repeating_tail(ids.data(), n, block)));
    log_msg(level,
            "%s: output began repeating a %d-token block; decode stopped with the repeats dropped (%d tokens "
            "kept). The transcript may be incomplete.",
            who, block, static_cast<int>(ids.size()));
    return true;
}

// Call once when a decode stopped at its budget or context window before eos.
// Drops the repeats of a block repeating at the tail, at the lower budget-stop
// bar, and logs what it dropped at `level`.
inline void trim_repetition_at_budget_stop(std::vector<int32_t> & ids,
                                           const char *           who,
                                           transcribe_log_level   level = TRANSCRIBE_LOG_LEVEL_WARN) {
    if (!repetition_guard_enabled()) {
        return;
    }
    const int n     = static_cast<int>(ids.size());
    const int block = repeating_tail_block(ids.data(), n, k_budget_trim_bar);
    if (block == 0) {
        return;
    }
    ids.resize(static_cast<size_t>(trim_repeating_tail(ids.data(), n, block)));
    log_msg(level, "%s: dropped %d tokens of a repeating %d-token block at the budget stop", who,
            n - static_cast<int>(ids.size()), block);
}

// Why a batched decode row stopped, as reported through a shared step loop's
// stop_out. Non-zero means the row never reached eos.
enum DecodeStop : char {
    k_stop_eos        = 0,
    k_stop_budget     = 1,  // generation budget or context window
    k_stop_repetition = 2,  // stop_on_repetition
};

inline transcribe_status decode_stop_status(char stop) {
    switch (stop) {
        case k_stop_eos:
            return TRANSCRIBE_OK;
        case k_stop_repetition:
            return TRANSCRIBE_ERR_OUTPUT_REPETITION;
        default:
            return TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
    }
}

}  // namespace transcribe
