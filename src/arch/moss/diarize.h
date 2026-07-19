// arch/moss/diarize.h - host-side parser for MOSS's emergent diarized
// transcript format.
//
// The model generates speaker turns as plain text:
//
//     [0.48][S01]Welcome.[1.66][1.70][S02]Thanks for having me.[3.20]
//
// where [t] is a start/end timestamp in seconds and [Sxx] a 1-based
// speaker tag. Both are ordinary generated text, not special tokens, so
// the parser applies a strict grammar and treats anything that does not
// match a recognized bracket span as verbatim transcript text: parsing
// never drops bytes.
//
// Non-static (and free of MossModel/MossSession dependencies) so the
// unit test can exercise the grammar without a model.

#pragma once

#include "transcribe-session.h"

#include <string>
#include <vector>

struct transcribe_run_params;

namespace transcribe::moss {

// True only for explicit diarize=ON. DEFAULT/OFF follow the library-wide
// no-attribution default; MOSS still parses its inline metadata for clean text.
bool diarize_resolves_on(const transcribe_run_params * params);

// Apply the independent diarization/timestamp run axes to a successfully
// parsed MOSS result. DEFAULT/OFF erase attribution. timestamps=NONE erases
// timing but retains the clean per-turn text segmentation.
void apply_result_policy(const transcribe_run_params *                          params,
                         std::vector<transcribe_session::SegmentEntry> &        segments,
                         std::vector<transcribe_session::SpeakerSegmentEntry> & speaker_segments);

transcribe_timestamp_kind returned_timestamp_kind(const transcribe_run_params * params);

// Parse the raw generated transcript into per-turn segments.
//
// Returns true when at least one timestamped speaker turn was
// recognized; the caller then installs the parsed segments (result kind
// SEGMENT), speaker rows, and the dediarized full text. Returns false
// when no turn was recognized; outputs are untouched and the caller
// keeps today's raw single-segment behavior.
//
// Grammar (applied at each '['):
//   TIME := '[' digits ('.' digits)? ']'   value in seconds
//   SPK  := '[S' digits ']'                1-based speaker id
// A TIME closes the open turn only when followed by SPK, TIME SPK, or
// end of input; otherwise it is verbatim text. A bare SPK is a turn
// boundary with unknown times, patched from the neighboring turns
// (first turn t0 -> 0, last turn t1 -> audio_ms). Text before the first
// turn becomes a leading speaker_id == 0 segment.
//
// out_full_text is the trimmed turn texts joined with single spaces —
// the same shape scripts/wer/score.py's dediarize() produces, so the
// default full_text matches what the WER gates scored.
bool parse_diarized_transcript(const std::string &                                    raw,
                               int64_t                                                audio_ms,
                               std::vector<transcribe_session::SegmentEntry> &        out_segments,
                               std::vector<transcribe_session::SpeakerSegmentEntry> & out_speaker_segments,
                               std::string &                                          out_full_text);

}  // namespace transcribe::moss
