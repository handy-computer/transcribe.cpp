// Parakeet long-form segmenter. See longform.h.

#include "longform.h"

#include <algorithm>
#include <cmath>

namespace transcribe::parakeet::longform {

namespace {

// A pause source reports region edges on its own frame grid, so a gap of
// exactly min_pause comes out as a difference of two multiples of that
// frame (281.0 - 280.8 is 0.19999999999998863); ties go to the pause.
constexpr double kPauseEpsilon = 1e-6;

int64_t samples_for(double seconds, int rate) {
    return round_half_even(seconds * static_cast<double>(rate));
}

// segment._has_speech: whether any speech falls in [0, seconds).
bool has_speech(const Regions & regions, double seconds) {
    for (const auto & r : regions) {
        if (r.second > 0.0 && r.first < seconds) {
            return true;
        }
    }
    return false;
}

}  // namespace

int64_t round_half_even(double x) {
    // std::nearbyint honours the current rounding mode; the default
    // (FE_TONEAREST) is ties-to-even, Python's round().
    return static_cast<int64_t>(std::nearbyint(x));
}

Regions speech_regions(const float * probs, int n_frames, const Params & p) {
    // edges = flatnonzero(diff(r_[0, speaking, 0])) -> (start, end) frame pairs.
    Regions regions;
    int     run_start = -1;
    for (int i = 0; i <= n_frames; ++i) {
        const bool speaking = i < n_frames && probs[i] >= p.speech_threshold;
        if (speaking && run_start < 0) {
            run_start = i;
        } else if (!speaking && run_start >= 0) {
            const double start = static_cast<double>(run_start) * p.frame_seconds;
            const double end   = static_cast<double>(i) * p.frame_seconds;
            if (!regions.empty() && start - regions.back().second < p.min_gap_seconds) {
                regions.back().second = end;
            } else {
                regions.emplace_back(start, end);
            }
            run_start = -1;
        }
    }
    Regions kept;
    for (const auto & r : regions) {
        if (r.second - r.first >= p.min_speech_seconds) {
            kept.push_back(r);
        }
    }
    return kept;
}

Regions pauses_from_speech(Regions speech, double duration, const Params & p) {
    const double least = p.min_pause_seconds - kPauseEpsilon;
    std::sort(speech.begin(), speech.end());
    Regions pauses;
    double  previous = 0.0;
    for (const auto & r : speech) {
        if (r.first - previous >= least) {
            pauses.emplace_back(previous, r.first);
        }
        previous = std::max(previous, r.second);
    }
    if (duration - previous >= least) {
        pauses.emplace_back(previous, duration);
    }
    return pauses;
}

double next_cut(const Regions & pauses, const Params & p) {
    // Last pause fully inside (min_segment, max_segment), else any pause whose
    // midpoint lands there, else the cap.
    const std::pair<double, double> * last = nullptr;
    for (const auto & r : pauses) {
        if (r.first >= p.min_segment_seconds && r.second <= p.max_segment_seconds) {
            last = &r;
        }
    }
    if (last == nullptr) {
        for (const auto & r : pauses) {
            const double mid = (r.first + r.second) / 2;
            if (p.min_segment_seconds <= mid && mid <= p.max_segment_seconds) {
                last = &r;
            }
        }
    }
    if (last == nullptr) {
        return p.max_segment_seconds;
    }
    return (last->first + last->second) / 2;
}

bool fits_one_segment(int64_t n_samples, const Params & p) {
    return n_samples <= samples_for(p.max_segment_seconds, p.sample_rate);
}

std::vector<int64_t> scan_block_sizes(int64_t n_samples, const Params & p) {
    const int64_t        max_frames      = samples_for(p.scan_block_seconds, p.sample_rate);
    const int64_t        window_frames   = std::max<int64_t>(4, samples_for(p.boundary_window_secs, p.sample_rate));
    const int64_t        min_tail_frames = std::max(window_frames, samples_for(p.min_tail_seconds, p.sample_rate));
    std::vector<int64_t> sizes;
    int64_t              emitted = 0;
    while (emitted < n_samples) {
        const int64_t remaining = n_samples - emitted;
        // With boundary_search_seconds=0 the quiet-boundary search window is
        // empty, so a full block is shortened only to keep min_tail behind it.
        const int64_t boundary =
            remaining <= max_frames ? remaining : std::min(max_frames, remaining - min_tail_frames);
        sizes.push_back(boundary);
        emitted += boundary;
    }
    return sizes;
}

transcribe_status pause_segments(int64_t                n_samples,
                                 const Params &         p,
                                 const SpeechFn &       speech,
                                 std::vector<Segment> & out) {
    out.clear();
    if (n_samples <= 0) {
        return TRANSCRIBE_OK;
    }
    const double rate = static_cast<double>(p.sample_rate);
    if (fits_one_segment(n_samples, p)) {
        out.push_back({ 0, n_samples });
        return TRANSCRIBE_OK;
    }
    const int64_t cap_samples = samples_for(p.max_segment_seconds, p.sample_rate);

    // The carry is [consumed, consumed + carry_len) of the file; regions
    // are relative to its start.
    Regions regions;
    int64_t consumed    = 0;
    int64_t carry_len   = 0;
    int64_t block_start = 0;
    int     cuts        = 0;
    for (const int64_t block_len : scan_block_sizes(n_samples, p)) {
        const double offset = static_cast<double>(carry_len) / rate;
        Regions      block_regions;
        if (const transcribe_status st = speech(block_start, block_len, block_regions); st != TRANSCRIBE_OK) {
            return st;
        }
        for (const auto & r : block_regions) {
            regions.emplace_back(r.first + offset, r.second + offset);
        }
        carry_len += block_len;
        block_start += block_len;
        while (carry_len > cap_samples) {
            const double  cut     = next_cut(pauses_from_speech(regions, static_cast<double>(carry_len) / rate, p), p);
            const int64_t index   = round_half_even(cut * rate);
            const double  seconds = static_cast<double>(index) / rate;
            if (has_speech(regions, seconds)) {
                out.push_back({ consumed, index });
            }
            consumed += index;
            carry_len -= index;
            ++cuts;
            Regions shifted;
            for (const auto & r : regions) {
                if (r.second > seconds) {
                    shifted.emplace_back(std::max(r.first - seconds, 0.0), r.second - seconds);
                }
            }
            regions.swap(shifted);
        }
    }
    // A remainder too short to normalize, or with no speech, left behind by
    // a cut carries nothing and is dropped.
    if (cuts > 0 &&
        (carry_len < p.min_feature_samples || !has_speech(regions, static_cast<double>(carry_len) / rate))) {
        return TRANSCRIBE_OK;
    }
    out.push_back({ consumed, carry_len });
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::parakeet::longform
