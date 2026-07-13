// arch/granite/diarize.cpp - see diarize.h for the task contract.

#include "diarize.h"

#include "transcribe-model.h"
#include "transcribe.h"

#include <cstdint>

namespace transcribe::granite {

const char k_saa_instruction[] =
    " Speaker attribution: Transcribe and denote who is speaking by adding "
    "[Speaker 1]: and [Speaker 2]: tags before speaker turns.";

namespace {

bool is_ascii_digit(char c) {
    return c >= '0' && c <= '9';
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

// Parse "[Speaker <digits>]" at raw[i], consuming an optional ':' and one
// optional following space. N must be >= 1 (1-based upstream numbering).
bool try_speaker_marker(const std::string & raw, size_t i, int32_t & out_id, size_t & out_next) {
    static const char k_prefix[]  = "[Speaker ";
    constexpr size_t  k_prefix_len = sizeof(k_prefix) - 1;
    const size_t      n            = raw.size();
    if (raw.compare(i, k_prefix_len, k_prefix) != 0) {
        return false;
    }
    size_t j = i + k_prefix_len;
    if (j >= n || !is_ascii_digit(raw[j])) {
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
    ++j;  // ']'
    if (j < n && raw[j] == ':') {
        ++j;
    }
    if (j < n && raw[j] == ' ') {
        ++j;
    }
    out_id   = static_cast<int32_t>(id);
    out_next = j;
    return true;
}

}  // namespace

bool diarize_requested(const transcribe_model * model, const transcribe_run_params * params) {
    if (model == nullptr || params == nullptr) {
        return false;
    }
    if (!transcribe::has_feature(model, TRANSCRIBE_FEATURE_DIARIZATION)) {
        return false;
    }
    return params->diarize == TRANSCRIBE_DIARIZE_MODE_ON;
}

bool split_speaker_turns(const std::string &                                     raw,
                         std::vector<transcribe_session::SegmentEntry> &         out_segments,
                         std::vector<transcribe_session::SpeakerSegmentEntry> & out_speaker_segments,
                         std::string &                                           out_full_text) {
    struct Turn {
        int32_t     spk = 0;  // 0 = leading unattributed text
        std::string text;
    };

    std::vector<Turn> turns;
    turns.push_back(Turn{});  // slot for text before the first marker

    const size_t n = raw.size();
    size_t       i = 0;
    bool         any_marker = false;
    while (i < n) {
        int32_t id   = 0;
        size_t  next = 0;
        if (raw[i] == '[' && try_speaker_marker(raw, i, id, next)) {
            turns.push_back(Turn{ id, std::string() });
            any_marker = true;
            i          = next;
            continue;
        }
        turns.back().text.push_back(raw[i]);
        ++i;
    }

    if (!any_marker) {
        return false;
    }

    out_segments.clear();
    out_speaker_segments.clear();
    out_full_text.clear();

    for (const Turn & turn : turns) {
        const std::string text = trimmed(turn.text);
        if (turn.spk == 0 && text.empty()) {
            continue;  // empty leading slot
        }
        transcribe_session::SegmentEntry seg{};
        seg.text       = text;
        seg.speaker_id = turn.spk;
        // The speaker-attribution task carries no timing information;
        // t0 == t1 == 0 is the documented "absent" sentinel.
        if (!out_full_text.empty() && !text.empty()) {
            out_full_text.push_back(' ');
        }
        out_full_text += text;
        out_segments.push_back(std::move(seg));

        if (turn.spk > 0) {
            transcribe_session::SpeakerSegmentEntry row{};
            row.speaker_id = turn.spk;
            out_speaker_segments.push_back(row);
        }
    }
    return true;
}

}  // namespace transcribe::granite
