// moss_diarize_parser_unit.cpp - grammar tests for the MOSS emergent
// diarized-transcript parser (arch/moss/diarize.h). Pure host-side: no
// model, no GGUF. The invariant under test throughout: parsing never
// drops transcript bytes — anything outside a recognized [time]/[Sxx]
// span survives verbatim in some segment's text.

#include "arch/moss/diarize.h"
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

Parsed parse(const std::string & raw, int64_t audio_ms = 10000) {
    Parsed p;
    p.ok = transcribe::moss::parse_diarized_transcript(raw, audio_ms, p.segments, p.speakers, p.full_text);
    return p;
}

void test_canonical_two_speakers() {
    const Parsed p = parse("[0.48][S01]Welcome.[1.66][1.70][S02]Thanks for having me.[3.20]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "Welcome.");
    CHECK(p.segments[0].t0_ms == 480);
    CHECK(p.segments[0].t1_ms == 1660);
    CHECK(p.segments[0].speaker_id == 1);
    CHECK(p.segments[1].text == "Thanks for having me.");
    CHECK(p.segments[1].t0_ms == 1700);
    CHECK(p.segments[1].t1_ms == 3200);
    CHECK(p.segments[1].speaker_id == 2);
    CHECK(p.full_text == "Welcome. Thanks for having me.");
    CHECK(p.speakers.size() == 2);
    CHECK(p.speakers[0].speaker_id == 1);
    CHECK(p.speakers[1].speaker_id == 2);
    CHECK(std::isnan(p.speakers[0].p));
}

void test_shared_boundary() {
    // Single TIME between turns: end of turn 1 == start of turn 2.
    const Parsed p = parse("[0.00][S01]a[1.00][S02]b[2.00]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].t1_ms == 1000);
    CHECK(p.segments[1].t0_ms == 1000);
    CHECK(p.segments[1].t1_ms == 2000);
}

void test_missing_end_timestamp() {
    const Parsed p = parse("[0.50][S01]Trailing words", 8000);
    CHECK(p.ok);
    CHECK(p.segments.size() == 1);
    CHECK(p.segments[0].text == "Trailing words");
    CHECK(p.segments[0].t0_ms == 500);
    CHECK(p.segments[0].t1_ms == 8000);  // patched to audio end
}

void test_bare_speaker_tags() {
    // Speaker changes without any timestamps still split turns.
    const Parsed p = parse("[S01]alpha[S02]beta", 4000);
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "alpha");
    CHECK(p.segments[0].speaker_id == 1);
    CHECK(p.segments[1].text == "beta");
    CHECK(p.segments[1].speaker_id == 2);
    CHECK(p.segments[0].t0_ms == 0);
    CHECK(p.segments[1].t1_ms == 4000);
    CHECK(p.full_text == "alpha beta");
}

void test_leading_text_before_first_tag() {
    const Parsed p = parse("hmm [0.90][S01]hi[2.00]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "hmm");
    CHECK(p.segments[0].speaker_id == 0);  // unattributed
    CHECK(p.segments[0].t0_ms == 0);
    CHECK(p.segments[0].t1_ms == 900);
    CHECK(p.segments[1].speaker_id == 1);
    CHECK(p.full_text == "hmm hi");
    // Unattributed lead does NOT get a speaker row.
    CHECK(p.speakers.size() == 1);
}

void test_out_of_order_timestamps() {
    // t1 < t0 clamps t1 up to t0; no crash, no reordering of t0.
    const Parsed p = parse("[5.00][S01]x[2.00][2.00][S02]y[3.00]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].t0_ms == 5000);
    CHECK(p.segments[0].t1_ms == 5000);  // clamped from 2000
}

void test_malformed_spans_pass_through() {
    // None of these match the grammar; every byte must survive.
    const Parsed p = parse("[0.48][S01]a [0.4x] b [S] c [ d[9.99]", 10000);
    CHECK(p.ok);
    CHECK(p.segments.size() == 1);
    CHECK(p.segments[0].text == "a [0.4x] b [S] c [ d");
    CHECK(p.segments[0].t1_ms == 9990);  // trailing TIME at EOS closes the turn
}

void test_time_in_running_text_is_verbatim() {
    // A TIME not followed by SPK/TIME-SPK/EOS is transcript text.
    const Parsed p = parse("[0.00][S01]wait [3.00] more words[5.00]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 1);
    CHECK(p.segments[0].text == "wait [3.00] more words");
    CHECK(p.segments[0].t1_ms == 5000);
}

void test_s00_is_not_a_speaker_tag() {
    // S00 would collide with the "unattributed" sentinel: verbatim text.
    const Parsed p = parse("[0.00][S01]a [S00] b[2.00]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 1);
    CHECK(p.segments[0].text == "a [S00] b");
}

void test_marker_free_input_returns_false() {
    Parsed p = parse("just a plain transcript with no markers");
    CHECK(!p.ok);
    CHECK(p.segments.empty());
    CHECK(p.full_text.empty());
    // Bracketed but unrecognized spans alone also do not constitute a turn.
    p = parse("weird [brackets] only");
    CHECK(!p.ok);
}

void test_cjk_text() {
    const Parsed p = parse("[0.00][S01]\xE4\xBD\xA0\xE5\xA5\xBD[1.00][1.00][S02]\xE4\xB8\x96\xE7\x95\x8C[2.00]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].text == "\xE4\xBD\xA0\xE5\xA5\xBD");
    CHECK(p.segments[1].text == "\xE4\xB8\x96\xE7\x95\x8C");
    CHECK(p.full_text == "\xE4\xBD\xA0\xE5\xA5\xBD \xE4\xB8\x96\xE7\x95\x8C");
}

void test_fractional_rounding() {
    // 4th fractional digit rounds; 3 digits pass through exactly.
    const Parsed p = parse("[0.1234][S01]x[0.5678][0.6][S02]y[1.5]");
    CHECK(p.ok);
    CHECK(p.segments.size() == 2);
    CHECK(p.segments[0].t0_ms == 123);  // 0.1234 -> 123 (round down)
    CHECK(p.segments[0].t1_ms == 568);  // 0.5678 -> 568 (round up)
    CHECK(p.segments[1].t0_ms == 600);
    CHECK(p.segments[1].t1_ms == 1500);
}

void test_dediarize_equivalence() {
    // full_text must equal the raw text with recognized spans removed and
    // whitespace collapsed — the same normalization the WER gates score.
    const Parsed p = parse("[0.48][S01]  Welcome home.  [1.66][1.70][S02]  Thanks!  [3.20]");
    CHECK(p.ok);
    CHECK(p.full_text == "Welcome home. Thanks!");
}

}  // namespace

int main() {
    test_canonical_two_speakers();
    test_shared_boundary();
    test_missing_end_timestamp();
    test_bare_speaker_tags();
    test_leading_text_before_first_tag();
    test_out_of_order_timestamps();
    test_malformed_spans_pass_through();
    test_time_in_running_text_is_verbatim();
    test_s00_is_not_a_speaker_tag();
    test_marker_free_input_returns_false();
    test_cjk_text();
    test_fractional_rounding();
    test_dediarize_equivalence();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
