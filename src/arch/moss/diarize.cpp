// arch/moss/diarize.cpp - see diarize.h for the grammar contract.

#include "diarize.h"

#include "transcribe.h"

#include <cctype>
#include <cstdint>

namespace transcribe::moss {

namespace {

constexpr int64_t k_unknown_ms = -1;

bool is_ascii_digit(char c) {
    return c >= '0' && c <= '9';
}

// Parse TIME := '[' digits ('.' digits)? ']' at raw[i]. On success writes
// the value in milliseconds (rounded from the 4th fractional digit) and
// the index one past the closing ']'.
bool try_time(const std::string & raw, size_t i, int64_t & out_ms, size_t & out_next) {
    const size_t n = raw.size();
    if (i >= n || raw[i] != '[') {
        return false;
    }
    size_t j = i + 1;
    if (j >= n || !is_ascii_digit(raw[j])) {
        return false;
    }
    int64_t int_part = 0;
    while (j < n && is_ascii_digit(raw[j])) {
        if (int_part > (INT64_MAX - 9) / 10) {
            return false;  // absurd magnitude: not a timestamp
        }
        int_part = int_part * 10 + (raw[j] - '0');
        ++j;
    }
    int64_t frac_ms   = 0;
    int     n_frac    = 0;
    bool    round_up  = false;
    if (j < n && raw[j] == '.') {
        ++j;
        if (j >= n || !is_ascii_digit(raw[j])) {
            return false;  // '.' must be followed by at least one digit
        }
        while (j < n && is_ascii_digit(raw[j])) {
            if (n_frac < 3) {
                frac_ms = frac_ms * 10 + (raw[j] - '0');
                ++n_frac;
            } else if (n_frac == 3) {
                round_up = raw[j] >= '5';
                ++n_frac;
            }
            ++j;
        }
    }
    if (j >= n || raw[j] != ']') {
        return false;
    }
    while (n_frac < 3) {
        frac_ms *= 10;
        ++n_frac;
    }
    if (int_part > (INT64_MAX - frac_ms - 1) / 1000) {
        return false;
    }
    out_ms   = int_part * 1000 + frac_ms + (round_up ? 1 : 0);
    out_next = j + 1;
    return true;
}

// Parse SPK := '[S' digits ']' at raw[i]. Speaker ids are 1-based in the
// model's output (S01, S02, ...); an id of 0 is rejected as unrecognized
// so it degrades to verbatim text instead of colliding with the
// "unattributed" sentinel.
bool try_spk(const std::string & raw, size_t i, int32_t & out_id, size_t & out_next) {
    const size_t n = raw.size();
    if (i + 2 >= n || raw[i] != '[' || raw[i + 1] != 'S') {
        return false;
    }
    size_t j = i + 2;
    if (!is_ascii_digit(raw[j])) {
        return false;
    }
    int64_t id = 0;
    while (j < n && is_ascii_digit(raw[j])) {
        id = id * 10 + (raw[j] - '0');
        if (id > INT32_MAX) {
            return false;
        }
        ++j;
    }
    if (j >= n || raw[j] != ']' || id <= 0) {
        return false;
    }
    out_id   = static_cast<int32_t>(id);
    out_next = j + 1;
    return true;
}

std::string trimmed(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) {
        --e;
    }
    return s.substr(b, e - b);
}

struct Turn {
    int64_t     t0 = k_unknown_ms;
    int64_t     t1 = k_unknown_ms;
    int32_t     spk = 0;
    std::string text;
};

}  // namespace

bool diarize_resolves_on(const transcribe_run_params * params) {
    // moss family default is ON (the model always emits speaker markers
    // and the WER gates scored dediarized text). Only an explicit OFF
    // selects the raw passthrough.
    if (params == nullptr) {
        return true;
    }
    return params->diarize != TRANSCRIBE_DIARIZE_MODE_OFF;
}

bool parse_diarized_transcript(const std::string &                                     raw,
                               int64_t                                                 audio_ms,
                               std::vector<transcribe_session::SegmentEntry> &         out_segments,
                               std::vector<transcribe_session::SpeakerSegmentEntry> & out_speaker_segments,
                               std::string &                                           out_full_text) {
    std::vector<Turn> turns;
    std::string       pre_text;  // verbatim text before the first turn

    const size_t n        = raw.size();
    bool         has_open = false;
    Turn         open;

    auto text_sink = [&]() -> std::string & { return has_open ? open.text : pre_text; };

    auto close_open = [&](int64_t t1) {
        open.t1 = t1;
        turns.push_back(std::move(open));
        open     = Turn{};
        has_open = false;
    };

    size_t i = 0;
    while (i < n) {
        if (raw[i] != '[') {
            text_sink().push_back(raw[i]);
            ++i;
            continue;
        }
        int64_t t_ms = 0;
        int32_t spk  = 0;
        size_t  j    = 0;
        if (try_time(raw, i, t_ms, j)) {
            int64_t t2_ms = 0;
            int32_t spk2  = 0;
            size_t  k     = 0;
            if (try_spk(raw, j, spk2, k)) {
                // TIME SPK: shared boundary (or the first turn's start).
                if (has_open) {
                    close_open(t_ms);
                }
                open.t0  = t_ms;
                open.spk = spk2;
                has_open = true;
                i        = k;
                continue;
            }
            size_t m = 0;
            if (try_time(raw, j, t2_ms, m) && try_spk(raw, m, spk2, k)) {
                // TIME TIME SPK: end of the open turn, start of the next.
                // With no open turn the first TIME is stray -> verbatim.
                if (has_open) {
                    close_open(t_ms);
                    open.t0  = t2_ms;
                    open.spk = spk2;
                    has_open = true;
                    i        = k;
                    continue;
                }
                text_sink().append(raw, i, j - i);
                i = j;
                continue;
            }
            if (j >= n && has_open) {
                // TIME at end of input closes the open turn.
                close_open(t_ms);
                i = j;
                continue;
            }
            // TIME followed by plain text: no boundary -> verbatim.
            text_sink().append(raw, i, j - i);
            i = j;
            continue;
        }
        if (try_spk(raw, i, spk, j)) {
            // Bare SPK: turn boundary with unknown times.
            if (has_open) {
                close_open(k_unknown_ms);
            }
            open.t0  = k_unknown_ms;
            open.spk = spk;
            has_open = true;
            i        = j;
            continue;
        }
        // Unrecognized '[' span: literal text.
        text_sink().push_back(raw[i]);
        ++i;
    }
    if (has_open) {
        close_open(k_unknown_ms);
    }

    if (turns.empty()) {
        return false;  // caller keeps the raw passthrough behavior
    }

    // Patch unknown boundaries. Forward pass: an unknown t0 inherits the
    // previous turn's end (or start as a last resort), the first turn
    // falls back to 0. Backward pass: an unknown t1 inherits the next
    // turn's (patched) start, the last falls back to audio_ms. Known
    // values from the model are never reordered; only per-turn t1 >= t0
    // is enforced, so out-of-order model output stays visible.
    for (size_t t = 0; t < turns.size(); ++t) {
        if (turns[t].t0 == k_unknown_ms) {
            if (t == 0) {
                turns[t].t0 = 0;
            } else if (turns[t - 1].t1 != k_unknown_ms) {
                turns[t].t0 = turns[t - 1].t1;
            } else {
                turns[t].t0 = turns[t - 1].t0;
            }
        }
        if (turns[t].t0 < 0) {
            turns[t].t0 = 0;
        }
    }
    for (size_t t = turns.size(); t-- > 0;) {
        if (turns[t].t1 == k_unknown_ms) {
            turns[t].t1 = (t + 1 < turns.size()) ? turns[t + 1].t0 : audio_ms;
        }
        if (turns[t].t1 < turns[t].t0) {
            turns[t].t1 = turns[t].t0;
        }
    }

    out_segments.clear();
    out_speaker_segments.clear();
    out_full_text.clear();

    auto append_full_text = [&](const std::string & piece) {
        if (piece.empty()) {
            return;
        }
        if (!out_full_text.empty()) {
            out_full_text.push_back(' ');
        }
        out_full_text += piece;
    };

    const std::string lead = trimmed(pre_text);
    if (!lead.empty()) {
        transcribe_session::SegmentEntry seg{};
        seg.text  = lead;
        seg.t0_ms = 0;
        seg.t1_ms = turns.front().t0;
        out_segments.push_back(std::move(seg));
        append_full_text(lead);
    }

    for (auto & turn : turns) {
        transcribe_session::SegmentEntry seg{};
        seg.text       = trimmed(turn.text);
        seg.t0_ms      = turn.t0;
        seg.t1_ms      = turn.t1;
        seg.speaker_id = turn.spk;
        append_full_text(seg.text);
        out_segments.push_back(std::move(seg));

        transcribe_session::SpeakerSegmentEntry row{};
        row.t0_ms      = turn.t0;
        row.t1_ms      = turn.t1;
        row.speaker_id = turn.spk;
        out_speaker_segments.push_back(row);
    }
    return true;
}

}  // namespace transcribe::moss
