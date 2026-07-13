// granite_diarize_parser_unit.cpp - grammar tests for the granite
// speaker-attribution splitter (arch/granite/diarize.h). Pure host-side:
// no model, no GGUF. Invariant: splitting never drops transcript bytes —
// anything outside a recognized "[Speaker N]" marker survives verbatim.

#include "arch/granite/diarize.h"
#include "transcribe-session.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

struct Parsed {
    bool                                                  ok = false;
    std::vector<transcribe_session::SegmentEntry>         segments;
    std::vector<transcribe_session::SpeakerSegmentEntry> speakers;
    std::string                                           full_text;
};

Parsed split(const std::string & raw) {
    Parsed p;
    p.ok = transcribe::granite::split_speaker_turns(raw, p.segments, p.speakers, p.full_text);
    return p;
}

void test_model_card_example() {
    // Verbatim example from the -plus model card. Numbering is 1-based.
    const Parsed p = split(
        "[Speaker 1]: Hello how are you [Speaker 2]: I'm fine and how are you feeling [Speaker 1]: I feel wonderful");
    CHECK(p.ok);
    CHECK(p.segments.size() == 3);
    CHECK(p.segments[0].text == "Hello how are you");
    CHECK(p.segments[0].speaker_id == 1);
    CHECK(p.segments[1].text == "I'm fine and how are you feeling");
    CHECK(p.segments[1].speaker_id == 2);
    CHECK(p.segments[2].text == "I feel wonderful");
    CHECK(p.segments[2].speaker_id == 1);
    CHECK(p.full_text == "Hello how are you I'm fine and how are you feeling I feel wonderful");
    CHECK(p.speakers.size() == 3);
    CHECK(p.speakers[0].speaker_id == 1);
    CHECK(p.speakers[1].speaker_id == 2);
    CHECK(p.speakers[2].speaker_id == 1);
    // The SAA task carries no timing: absent sentinel on every row.
    CHECK(p.segments[0].t0_ms == 0 && p.segments[0].t1_ms == 0);
    CHECK(p.speakers[0].t0_ms == 0 && p.speakers[0].t1_ms == 0);
    CHECK(std::isnan(p.speakers[0].p));
}

void test_no_marker_returns_false() {
    // Single-speaker output has no tags: caller keeps the plain path
    // byte-identical (this protects the jfk exact-compare golden).
    Parsed p = split("The quick brown fox jumps over the lazy dog.");
    CHECK(!p.ok);
    CHECK(p.segments.empty());
    CHECK(p.full_text.empty());
    // Bracket noise that is not a speaker marker also does not split.
    p = split("math notation [Speaker] and [Speaker x] and [T:45] stay text");
    CHECK(!p.ok);
}

void test_leading_text_before_first_marker() {
    const Parsed p = split("uh huh [Speaker 1]: right");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "uh huh");
    CHECK(p.segments[0].speaker_id == 0);  // unattributed
    CHECK(p.segments[1].text == "right");
    CHECK(p.segments[1].speaker_id == 1);
    CHECK(p.speakers.size() == 1);  // no row for the unattributed lead
    CHECK(p.full_text == "uh huh right");
}

void test_marker_without_colon() {
    // ':' and the following space are optional in the lexer.
    const Parsed p = split("[Speaker 1] hello [Speaker 2]hi");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "hello");
    CHECK(p.segments[1].text == "hi");
}

void test_speaker_zero_is_not_a_marker() {
    // [Speaker 0] would collide with the unattributed sentinel: verbatim.
    const Parsed p = split("[Speaker 1]: a [Speaker 0]: b");
    CHECK(p.ok);
    CHECK(p.segments.size() == 1);
    CHECK(p.segments[0].text == "a [Speaker 0]: b");
    CHECK(p.segments[0].speaker_id == 1);
}

void test_multidigit_speaker() {
    const Parsed p = split("[Speaker 12]: many voices");
    CHECK(p.ok);
    CHECK(p.segments.size() == 1);
    CHECK(p.segments[0].speaker_id == 12);
}

void test_no_byte_loss_on_malformed() {
    const Parsed p = split("[Speaker 1]: a [ b [Speaker two]: c [Speaker 2]: d");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "a [ b [Speaker two]: c");
    CHECK(p.segments[1].text == "d");
}

}  // namespace

int main() {
    test_model_card_example();
    test_no_marker_returns_false();
    test_leading_text_before_first_marker();
    test_marker_without_colon();
    test_speaker_zero_is_not_a_marker();
    test_multidigit_speaker();
    test_no_byte_loss_on_malformed();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
