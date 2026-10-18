// vad_segments_unit.cpp - the get_speech_timestamps_from_probs port
// (transcribe::vad_probs_to_segments) against vectors generated from
// silero-vad 6.2.3 by scripts/vad/gen_vectors.py.

#include "transcribe-vad.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct SegmentVector {
    double               threshold;
    double               neg_threshold;
    int32_t              min_speech_ms;
    int32_t              min_silence_ms;
    int32_t              speech_pad_ms;
    int32_t              max_speech_ms;
    int32_t              min_silence_at_max_speech_ms;
    bool                 use_max_possible_silence;
    int64_t              n_samples;
    const char *         probs;  // 'a' + 16 * p per frame
    std::vector<int64_t> segments;
};

const SegmentVector k_vectors[] = {
#include "fixtures/vad_segment_vectors.inc"
};

}  // namespace

int main() {
    int failures = 0;
    for (size_t v = 0; v < sizeof(k_vectors) / sizeof(k_vectors[0]); ++v) {
        const SegmentVector & t = k_vectors[v];
        transcribe_vad_params p;
        transcribe_vad_params_init(&p);
        p.threshold      = t.threshold;
        p.neg_threshold  = t.neg_threshold;
        p.min_speech_ms  = t.min_speech_ms;
        p.min_silence_ms = t.min_silence_ms;
        p.speech_pad_ms  = t.speech_pad_ms;
        p.max_speech_ms  = t.max_speech_ms;
        transcribe::VadSegmentParams sp;
        if (transcribe::resolve_vad_params(&p, sp) != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL vector %zu: params rejected\n", v);
            ++failures;
            continue;
        }
        // Not public: upstream's max-speech split knobs, pinned internally.
        sp.min_silence_at_max_speech_ms = t.min_silence_at_max_speech_ms;
        sp.use_max_possible_silence     = t.use_max_possible_silence;
        std::vector<float> probs;
        for (const char * c = t.probs; *c != '\0'; ++c) {
            probs.push_back(static_cast<float>(*c - 'a') / 16.0f);
        }
        std::vector<transcribe::VadSegmentEntry> got;
        transcribe::vad_probs_to_segments(probs.data(), static_cast<int64_t>(probs.size()), t.n_samples, 512, sp, got);
        bool ok = got.size() * 2 == t.segments.size();
        for (size_t i = 0; ok && i < got.size(); ++i) {
            ok = got[i].start_sample == t.segments[2 * i] && got[i].end_sample == t.segments[2 * i + 1];
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL vector %zu: %zu segments vs %zu expected\n", v, got.size(),
                         t.segments.size() / 2);
            ++failures;
        }
    }
    std::printf("vad_segments_unit: %d failure(s)\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
