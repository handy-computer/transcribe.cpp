// Long-form pause segmentation for parakeet checkpoints that carry a VAD
// head. Port of kestrel 0.9.1 segment.py / vad.py.
//
// Audio longer than max_segment_seconds is cut at the midpoint of the last
// pause (a gap of at least min_pause_seconds between speech regions) that
// lies inside (min_segment_seconds, max_segment_seconds) of the running
// carry, else at the cap; segments are contiguous, never overlap, and a
// segment with no speech is skipped. Speech regions come from the VAD head,
// scanned one block of scan_block_seconds at a time.
//
// All time arithmetic is float64 and cut indices use round-half-to-even,
// as in the Python reference, so cut points match sample for sample.
// Host-only; no ggml.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace transcribe::parakeet::longform {

// [start, end) in seconds.
using Regions = std::vector<std::pair<double, double>>;

struct Params {
    int    sample_rate          = 16000;
    double frame_seconds        = 0.08;  // hop / sample_rate * subsampling_factor
    float  speech_threshold     = 0.5f;
    double min_speech_seconds   = 0.1;
    double min_gap_seconds      = 0.1;
    double max_segment_seconds  = 30.0;
    double min_segment_seconds  = 1.0;
    double min_pause_seconds    = 0.2;
    double scan_block_seconds   = 120.0;
    // AudioChunks constants: a block is never followed by a tail shorter
    // than max(window, 0.5 s).
    double boundary_window_secs = 0.1;
    double min_tail_seconds     = 0.5;
    // parakeet_features refuses to normalize anything shorter (samples).
    int    min_feature_samples  = 320;
};

struct Segment {
    int64_t start     = 0;  // first sample
    int64_t n_samples = 0;
};

// Python's round() on a float: nearest integer, ties to even.
int64_t round_half_even(double x);

// vad.speech_regions: runs of frames with p >= threshold, gaps shorter
// than min_gap bridged, runs shorter than min_speech dropped.
Regions speech_regions(const float * probs, int n_frames, const Params & p);

// segment.pauses_from_speech: gaps of at least min_pause between and
// around the speech regions, over [0, duration].
Regions pauses_from_speech(Regions speech, double duration, const Params & p);

// segment.next_cut: where a segment starting at zero ends.
double next_cut(const Regions & pauses, const Params & p);

// segment.fits_one_segment for 16 kHz input.
bool fits_one_segment(int64_t n_samples, const Params & p);

// AudioChunks.chunks(scan_block_seconds, boundary_search_seconds=0): the
// sample count of each scan block, in order.
std::vector<int64_t> scan_block_sizes(int64_t n_samples, const Params & p);

// Speech source for one scan block: fills `regions` (seconds, relative to
// the block start) for samples [block_start, block_start + block_len).
using SpeechFn = std::function<transcribe_status(int64_t block_start, int64_t block_len, Regions & regions)>;

// segment.pause_segments. A clip that fits one segment is returned whole
// without calling `speech`.
transcribe_status pause_segments(int64_t                n_samples,
                                 const Params &         p,
                                 const SpeechFn &       speech,
                                 std::vector<Segment> & out);

}  // namespace transcribe::parakeet::longform
