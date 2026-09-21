// Stop rule for greedy autoregressive decoders stuck repeating themselves.
//
// Greedy argmax has no way out of a self-reinforcing state: once the likeliest
// continuation of a phrase is the phrase itself, it repeats until the decode
// budget runs out. The guard works on token ids only, so it fits decoders that
// read back a device-side argmax, and leaves output that never loops unchanged.

#pragma once

#include "transcribe-env.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace transcribe {

// A block repeating at the tail is a loop once it has k_repeat_min_copies
// copies covering at least k_repeat_min_tokens. Short blocks need many more
// copies (a 1-token block needs 32), so emphatic human repetition survives.
constexpr int k_repeat_max_block  = 64;
constexpr int k_repeat_min_copies = 4;
constexpr int k_repeat_min_tokens = 32;

// Length of the block repeating at the end of ids[0, n), or 0.
inline int repeating_tail_block(const int32_t * ids, int n) {
    for (int block = 1; block <= k_repeat_max_block; ++block) {
        const int copies = std::max(k_repeat_min_copies, (k_repeat_min_tokens + block - 1) / block);
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

// Call after appending a token. On a loop, trims `ids` to one copy, logs a WARN
// tagged `who`, and returns true: the caller stops decoding and flags the run
// truncated, since whatever the audio said after the loop was never decoded.
inline bool stop_on_repetition(std::vector<int32_t> & ids, const char * who) {
    if (!repetition_guard_enabled()) {
        return false;
    }
    const int n     = static_cast<int>(ids.size());
    const int block = repeating_tail_block(ids.data(), n);
    if (block == 0) {
        return false;
    }
    ids.resize(static_cast<size_t>(trim_repeating_tail(ids.data(), n, block)));
    log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
            "%s: output began repeating a %d-token block; decode stopped with the repeats dropped (%d tokens "
            "kept). The transcript may be incomplete.",
            who, block, static_cast<int>(ids.size()));
    return true;
}

}  // namespace transcribe
