// vad_lookahead_unit.cpp - VAD role dispatcher (transcribe-vad.cpp) with a
// fake lookahead model: frames are judged lookahead_frames late; feed returns
// judged frames only, flush and run step silent frames so every frame is
// judged; first_frame counts judged frames; the default threshold is the
// model's.

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe-vad.h"
#include "transcribe.h"
#include "transcribe/vad.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

constexpr int32_t k_frame = 4;
constexpr int32_t k_lag   = 2;

struct FakeSession : transcribe_vad_session {
    std::vector<float> hist;  // first sample of each stepped frame
};

int32_t fake_frame_samples(const transcribe_model *) {
    return k_frame;
}

transcribe_vad_session * fake_new_session() {
    return new FakeSession();
}

void fake_reset(transcribe_vad_session * s) {
    static_cast<FakeSession *>(s)->hist.clear();
}

// The value emitted at step s judges frame s - lag: that frame's first sample.
transcribe_status fake_score(transcribe_vad_session * s, const float * pcm, int64_t n, float * probs) {
    auto & hist = static_cast<FakeSession *>(s)->hist;
    for (int64_t i = 0; i < n; ++i) {
        hist.push_back(pcm[i * k_frame]);
        const int64_t step = static_cast<int64_t>(hist.size()) - 1;
        probs[i]           = step - k_lag >= 0 ? hist[static_cast<size_t>(step - k_lag)] : -1.0f;
    }
    return TRANSCRIBE_OK;
}

int32_t fake_lookahead(const transcribe_model *) {
    return k_lag;
}

double fake_threshold(const transcribe_model *) {
    return 0.25;
}

const transcribe::VadOps k_ops = { fake_frame_samples, fake_new_session, fake_reset,
                                   fake_score,         fake_lookahead,   fake_threshold };

transcribe::Arch make_arch() {
    transcribe::Arch a{};
    a.name = "fake-vad-lookahead";
    a.vad  = &k_ops;
    return a;
}

const transcribe::Arch k_arch = make_arch();

transcribe_vad_result get_result(transcribe_vad_session * s) {
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    return r;
}

void test_lookahead() {
    transcribe_model model;
    model.arch  = &k_arch;
    model.roles = TRANSCRIBE_ROLE_VAD;

    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    CHECK(transcribe_vad_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.frame_samples == k_frame && info.lookahead_frames == k_lag && info.threshold == 0.25);

    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(&model, nullptr, &s) == TRANSCRIBE_OK);

    // 7 frames, the last partial; frame t's first sample is its probability.
    std::vector<float> pcm;
    const float        p[7] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f };
    for (float v : p) {
        pcm.insert(pcm.end(), { v, 0.0f, 0.0f, 0.0f });
    }
    pcm.resize(pcm.size() - 2);
    const int n = static_cast<int>(pcm.size());

    // run judges every frame, in order.
    CHECK(transcribe_vad_run(s, pcm.data(), n, nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result r = get_result(s);
    CHECK(r.n_probs == 7 && r.first_frame == 0);
    const std::vector<float> want(p, p + 7);
    CHECK(std::vector<float>(transcribe_vad_probs(s), transcribe_vad_probs(s) + 7) == want);

    // Streams in any chunking give the same judged frames, numbered by frame.
    for (int chunk : { 1, 3, 4, 26 }) {
        std::vector<float> got;
        int64_t            next = 0;
        for (int off = 0; off < n; off += chunk) {
            CHECK(transcribe_vad_stream_feed(s, pcm.data() + off, std::min(chunk, n - off)) == TRANSCRIBE_OK);
            r = get_result(s);
            CHECK(r.first_frame == next);
            if (r.n_probs > 0) {
                got.insert(got.end(), transcribe_vad_probs(s), transcribe_vad_probs(s) + r.n_probs);
            }
            next += r.n_probs;
        }
        CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
        r = get_result(s);
        CHECK(r.first_frame == next);
        // Flush judges the padded frame and the lag frames before it.
        CHECK(r.n_probs == 1 + k_lag);
        got.insert(got.end(), transcribe_vad_probs(s), transcribe_vad_probs(s) + r.n_probs);
        CHECK(got == want);
    }

    // Flush of an empty stream judges nothing.
    CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
    CHECK(get_result(s).n_probs == 0);

    // The default threshold is the model's: 0.25 -> frames 1.. (p >= 0.25 from 0.3).
    transcribe_vad_params vp;
    transcribe_vad_params_init(&vp);
    CHECK(vp.threshold == -1.0);
    vp.min_speech_ms  = 0;
    vp.speech_pad_ms  = 0;
    vp.min_silence_ms = 0;
    CHECK(transcribe_vad_run(s, pcm.data(), n, &vp) == TRANSCRIBE_OK);
    CHECK(get_result(s).n_segments == 1);
    transcribe_vad_segment seg;
    transcribe_vad_segment_init(&seg);
    CHECK(transcribe_vad_get_segment(s, 0, &seg) == TRANSCRIBE_OK);
    CHECK(seg.start_sample == 2 * k_frame);
    vp.threshold = 0.65;
    CHECK(transcribe_vad_run(s, pcm.data(), n, &vp) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_get_segment(s, 0, &seg) == TRANSCRIBE_OK);
    CHECK(seg.start_sample == 6 * k_frame);
    vp.threshold = 1.5;
    CHECK(transcribe_vad_run(s, pcm.data(), n, &vp) == TRANSCRIBE_ERR_INVALID_ARG);

    transcribe_vad_session_free(s);
}

}  // namespace

int main() {
    test_lookahead();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("vad_lookahead_unit: OK\n");
    return EXIT_SUCCESS;
}
