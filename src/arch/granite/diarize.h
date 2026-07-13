// arch/granite/diarize.h - speaker-attribution (SAA) support for
// granite-speech-4.1-2b-plus.
//
// Unlike moss (whose speaker markers are always present), granite's
// diarization is a prompt-selected task: the user turn must carry IBM's
// verbatim speaker-attribution instruction, and the model then emits
// "[Speaker N]:" tags before speaker turns (N is 1-based, numbered by
// order of appearance). Diarization and word timestamps are separate
// tasks upstream and do not compose; the prompt builder rejects
// diarize=ON together with timestamps=WORD.
//
// The splitter is non-static and model-free so the unit test can
// exercise the grammar directly.

#pragma once

#include "transcribe-session.h"

#include <string>
#include <vector>

struct transcribe_model;
struct transcribe_run_params;

namespace transcribe::granite {

// IBM's verbatim speaker-attribution instruction (leading space matches
// the -plus prompt convention, same as the timestamps instruction).
extern const char k_saa_instruction[];

// True when this run requests the speaker-attribution task: the model
// advertises TRANSCRIBE_FEATURE_DIARIZATION (the -plus variant's GGUF
// capability KV) AND the caller passed diarize=ON. DEFAULT resolves OFF
// for granite: plain transcription is the family default the shipped
// WER numbers were measured against.
bool diarize_requested(const transcribe_model * model, const transcribe_run_params * params);

// Split a speaker-attributed transcript at "[Speaker N]" markers
// (optional ':' and one following space are consumed with the marker).
//
// Returns true when at least one marker was recognized: out_segments
// gets one entry per turn (marker stripped, speaker_id = N, t0/t1 = 0 —
// this task carries no timing information), out_speaker_segments one
// row per attributed turn (p = NaN), and out_full_text the marker-free
// text. Text before the first marker becomes a leading speaker_id == 0
// segment. Markers with N <= 0 are not recognized (they would collide
// with the "unattributed" sentinel) and pass through verbatim, as does
// any other unrecognized bracket span: splitting never drops bytes.
//
// Returns false when no marker is present; outputs are untouched and
// the caller keeps the plain single-segment path.
bool split_speaker_turns(const std::string &                                    raw,
                         std::vector<transcribe_session::SegmentEntry> &        out_segments,
                         std::vector<transcribe_session::SpeakerSegmentEntry> & out_speaker_segments,
                         std::string &                                          out_full_text);

}  // namespace transcribe::granite
