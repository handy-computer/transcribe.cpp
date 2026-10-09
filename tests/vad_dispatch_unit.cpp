// vad_dispatch_unit.cpp - VAD role dispatcher (transcribe-vad.cpp) against a
// fake arch: role checks, params validation, frame accounting and zero
// padding for offline runs, stream buffering / flush / reset, failure paths,
// and the segmentation port against vectors from the reference.

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe-vad.h"
#include "transcribe.h"
#include "transcribe/vad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
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

constexpr int kFrame = 8;  // the fake model's frame size

// The fake model's "probability" of a frame is its first sample, so tests
// control the output through the PCM. It records every frame it scores, in
// stream order, and counts resets.
std::vector<float> g_scored;
int                g_resets      = 0;
bool               g_check_abort = false;
int                g_fail_after  = -1;  // fail the call that would pass this many frames
int                g_throw_score = 0;   // 1 = backend exception, 2 = allocation failure
int                g_throw_reset = 0;   // number of resets that should throw

int32_t fake_frame_samples(const transcribe_model *) {
    return kFrame;
}

transcribe_vad_session * fake_new_session() {
    return new transcribe_vad_session();
}

void fake_reset(transcribe_vad_session *) {
    ++g_resets;
    if (g_throw_reset > 0) {
        --g_throw_reset;
        throw std::bad_alloc();
    }
}

transcribe_status fake_score(transcribe_vad_session * s, const float * pcm, int64_t n, float * probs) {
    if (g_check_abort && s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    if (g_fail_after >= 0 && static_cast<int64_t>(g_scored.size() / kFrame) + n > g_fail_after) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    for (int64_t f = 0; f < n; ++f) {
        probs[f] = pcm[f * kFrame];
        for (int i = 0; i < kFrame; ++i) {
            g_scored.push_back(pcm[f * kFrame + i]);
        }
        if (g_throw_score == 1) {
            throw std::runtime_error("injected VAD score exception");
        }
        if (g_throw_score == 2) {
            throw std::bad_alloc();
        }
    }
    return TRANSCRIBE_OK;
}

const transcribe::VadOps k_ops = { fake_frame_samples, fake_new_session, fake_reset, fake_score };

transcribe::Arch make_arch() {
    transcribe::Arch a = {};
    a.name             = "fake-vad";
    a.vad              = &k_ops;
    return a;
}

const transcribe::Arch k_arch = make_arch();

struct Fixture {
    transcribe_model         model;
    transcribe_vad_session * session = nullptr;

    Fixture() {
        g_scored.clear();
        g_resets      = 0;
        g_check_abort = false;
        g_fail_after  = -1;
        g_throw_score = 0;
        g_throw_reset = 0;
        model.arch    = &k_arch;
        model.roles   = TRANSCRIBE_ROLE_VAD;
        CHECK(transcribe_vad_session_init(&model, nullptr, &session) == TRANSCRIBE_OK);
    }

    ~Fixture() { transcribe_vad_session_free(session); }

    transcribe_vad_result result() const {
        transcribe_vad_result r;
        transcribe_vad_result_init(&r);
        CHECK(transcribe_vad_get_result(session, &r) == TRANSCRIBE_OK);
        return r;
    }
};

// pcm[i] = i + 1, so frames are distinguishable and never all zero.
std::vector<float> ramp(int n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    return v;
}

void test_roles_and_info() {
    transcribe_model other;
    other.arch                 = &k_arch;
    other.roles                = TRANSCRIBE_ROLE_ASR;
    transcribe_vad_session * s = reinterpret_cast<transcribe_vad_session *>(1);
    CHECK(transcribe_vad_session_init(&other, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);
    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    CHECK(transcribe_vad_get_info(&other, &info) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    Fixture fx;
    CHECK(transcribe_vad_get_info(&fx.model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000);
    CHECK(info.frame_samples == kFrame);
    info.struct_size = 8;
    CHECK(transcribe_vad_get_info(&fx.model, &info) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    sp.n_threads = -1;
    CHECK(transcribe_vad_session_init(&fx.model, &sp, &s) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(s == nullptr);
}

void test_params() {
    transcribe_vad_params p;
    transcribe_vad_params_init(&p);
    CHECK(p.threshold == 0.5 && p.neg_threshold < 0.0 && p.min_speech_ms == 250 && p.min_silence_ms == 100 &&
          p.speech_pad_ms == 30 && p.max_speech_ms == 0 && p.min_silence_at_max_speech_ms == 98 &&
          p.use_max_possible_silence);

    transcribe::VadSegmentParams r;
    CHECK(transcribe::resolve_vad_params(nullptr, r) == TRANSCRIBE_OK);
    CHECK(r.neg_threshold == 0.5 - 0.15);  // max(threshold - 0.15, 0.01) in double
    p.threshold = 0.1;
    CHECK(transcribe::resolve_vad_params(&p, r) == TRANSCRIBE_OK);
    CHECK(r.neg_threshold == 0.01);

    const double bad_thr[] = { -0.1, 1.1, std::numeric_limits<double>::quiet_NaN() };
    for (const double t : bad_thr) {
        transcribe_vad_params_init(&p);
        p.threshold = t;
        CHECK(transcribe::resolve_vad_params(&p, r) == TRANSCRIBE_ERR_INVALID_ARG);
    }
    transcribe_vad_params_init(&p);
    p.neg_threshold = 0.6;  // above threshold
    CHECK(transcribe::resolve_vad_params(&p, r) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_vad_params_init(&p);
    p.min_speech_ms = -1;
    CHECK(transcribe::resolve_vad_params(&p, r) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_vad_params_init(&p);
    p.struct_size = 8;
    CHECK(transcribe::resolve_vad_params(&p, r) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

// An offline run scores ceil(n / frame) frames from a reset model, the last
// one zero padded, and resets again afterwards.
void test_run_frames() {
    Fixture                  fx;
    const std::vector<float> pcm = ramp(3 * kFrame + 3);
    const int                r0  = g_resets;
    CHECK(transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(g_resets == r0 + 2);
    CHECK(g_scored.size() == 4u * kFrame);
    for (size_t i = 0; i < g_scored.size(); ++i) {
        CHECK(g_scored[i] == (i < pcm.size() ? pcm[i] : 0.0f));
    }
    const transcribe_vad_result r = fx.result();
    CHECK(r.n_probs == 4 && r.first_frame == 0);
    const float * p = transcribe_vad_probs(fx.session);
    CHECK(p != nullptr && p[0] == 1.0f && p[3] == static_cast<float>(3 * kFrame + 1));

    // Malformed input leaves the previous result alone.
    std::vector<float> nan_pcm = pcm;
    nan_pcm[5]                 = std::numeric_limits<float>::infinity();
    CHECK(transcribe_vad_run(fx.session, nan_pcm.data(), static_cast<int>(nan_pcm.size()), nullptr) ==
          TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_run(fx.session, pcm.data(), 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_run(fx.session, nullptr, 8, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_vad_params bad;
    transcribe_vad_params_init(&bad);
    bad.threshold = 2.0;
    CHECK(transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), &bad) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(fx.result().n_probs == 4);
}

// Feeding in arbitrary pieces scores the same frames as one run; flush
// scores the zero-padded tail; first_frame tracks the stream position.
void test_stream() {
    Fixture                  fx;
    const std::vector<float> pcm = ramp(5 * kFrame + 5);
    std::vector<float>       probs;
    CHECK(transcribe_vad_stream_feed(fx.session, nullptr, 0) == TRANSCRIBE_OK);
    CHECK(fx.result().n_probs == 0 && fx.result().first_frame == 0);
    const int pieces[] = { 3, 0, 9, 7, 16, 2, 8 };  // 45 samples
    size_t    off      = 0;
    for (const int n : pieces) {
        CHECK(transcribe_vad_stream_feed(fx.session, pcm.data() + off, n) == TRANSCRIBE_OK);
        off += static_cast<size_t>(n);
        const transcribe_vad_result r = fx.result();
        CHECK(r.n_segments == 0);
        CHECK(r.first_frame == static_cast<int64_t>(probs.size()));
        const float * p = transcribe_vad_probs(fx.session);
        for (int i = 0; i < r.n_probs; ++i) {
            probs.push_back(p[i]);
        }
    }
    CHECK(off == pcm.size());
    CHECK(probs.size() == 5u);
    CHECK(transcribe_vad_stream_flush(fx.session) == TRANSCRIBE_OK);
    const transcribe_vad_result r = fx.result();
    CHECK(r.n_probs == 1 && r.first_frame == 5);
    CHECK(transcribe_vad_probs(fx.session)[0] == static_cast<float>(5 * kFrame + 1));
    CHECK(g_scored.size() == 6u * kFrame);
    for (size_t i = 0; i < g_scored.size(); ++i) {
        CHECK(g_scored[i] == (i < pcm.size() ? pcm[i] : 0.0f));
    }

    // After the flush a new stream starts at frame 0; a flush of an empty
    // stream produces nothing.
    CHECK(transcribe_vad_stream_flush(fx.session) == TRANSCRIBE_OK);
    CHECK(fx.result().n_probs == 0 && transcribe_vad_probs(fx.session) == nullptr);
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_OK);
    CHECK(fx.result().first_frame == 0 && fx.result().n_probs == 1);

    // reset drops the partial frame.
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), 3) == TRANSCRIBE_OK);
    transcribe_vad_stream_reset(fx.session);
    const size_t before = g_scored.size();
    CHECK(transcribe_vad_stream_flush(fx.session) == TRANSCRIBE_OK);
    CHECK(g_scored.size() == before);

    // Malformed feeds leave the stream alone.
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), 3) == TRANSCRIBE_OK);
    std::vector<float> bad(4, std::numeric_limits<float>::quiet_NaN());
    CHECK(transcribe_vad_stream_feed(fx.session, bad.data(), 4) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_stream_feed(fx.session, nullptr, 4) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), -1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_stream_feed(fx.session, nullptr, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data() + 3, kFrame - 3) == TRANSCRIBE_OK);
    CHECK(fx.result().n_probs == 1 && transcribe_vad_probs(fx.session)[0] == 1.0f);
}

void test_failures() {
    Fixture                  fx;
    const std::vector<float> pcm = ramp(4 * kFrame);
    CHECK(transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);

    // A backend failure zeroes the result and resets the stream.
    g_fail_after = 0;
    CHECK(transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_ERR_BACKEND);
    CHECK(fx.result().n_probs == 0 && fx.result().n_segments == 0);
    g_fail_after = -1;
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame + 2) == TRANSCRIBE_OK);
    g_fail_after = static_cast<int>(g_scored.size() / kFrame);
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_ERR_BACKEND);
    g_fail_after = -1;
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_OK);
    CHECK(fx.result().first_frame == 0);  // the stream restarted

    // Abort.
    g_check_abort = true;
    transcribe_vad_set_abort_callback(fx.session, [](void *) { return true; }, nullptr);
    CHECK(transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_ERR_ABORTED);
    CHECK(fx.result().n_probs == 0);
}

// Exceptions after validation have the same cleanup contract as status errors.
void test_exceptions() {
    for (int operation = 0; operation < 3; ++operation) {
        for (int kind = 1; kind <= 2; ++kind) {
            Fixture                  fx;
            const std::vector<float> pcm = ramp(4 * kFrame);
            CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame + 2) == TRANSCRIBE_OK);
            const int before = g_resets;
            g_throw_score    = kind;
            transcribe_status st;
            if (operation == 0) {
                st = transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), nullptr);
            } else if (operation == 1) {
                st = transcribe_vad_stream_feed(fx.session, pcm.data(), 2 * kFrame);
            } else {
                st = transcribe_vad_stream_flush(fx.session);
            }
            CHECK(st == (kind == 1 ? TRANSCRIBE_ERR_BACKEND : TRANSCRIBE_ERR_OOM));
            CHECK(g_resets > before);
            CHECK(fx.result().n_probs == 0 && fx.result().n_segments == 0 && fx.result().first_frame == 0);
            CHECK(transcribe_vad_probs(fx.session) == nullptr);
            CHECK(fx.session->pending.empty() && fx.session->frames_done == 0);
            g_throw_score = 0;
            CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_OK);
            CHECK(fx.result().first_frame == 0 && fx.result().n_probs == 1);
        }
    }

    // Throwing host callbacks are contained and also discard the stream.
    Fixture                  fx;
    const std::vector<float> pcm = ramp(2 * kFrame);
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame + 2) == TRANSCRIBE_OK);
    g_check_abort = true;
    transcribe_vad_set_abort_callback(
        fx.session, [](void *) -> bool { throw std::runtime_error("injected VAD abort callback exception"); }, nullptr);
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_ERR_BACKEND);
    CHECK(fx.result().n_probs == 0 && fx.session->frames_done == 0 && fx.session->pending.empty());
    transcribe_vad_set_abort_callback(fx.session, nullptr, nullptr);
    g_check_abort = false;

    // If cleanup itself throws, it is contained and retried before scoring.
    g_throw_score = 1;
    g_throw_reset = 1;
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_ERR_BACKEND);
    CHECK(fx.result().n_probs == 0 && fx.session->needs_reset);
    g_throw_score    = 0;
    const int before = g_resets;
    CHECK(transcribe_vad_stream_feed(fx.session, pcm.data(), kFrame) == TRANSCRIBE_OK);
    CHECK(g_resets == before + 1 && !fx.session->needs_reset);
    CHECK(fx.result().first_frame == 0 && fx.result().n_probs == 1);
}

// Segment accessors and the copy-out contract.
void test_segments_api() {
    Fixture            fx;
    // 16 kHz, 8-sample frames: speech for 600 frames (300 ms) after 100
    // silent ones, then silence.
    std::vector<float> pcm(static_cast<size_t>(1000 * kFrame), 0.0f);
    for (int f = 100; f < 700; ++f) {
        pcm[static_cast<size_t>(f * kFrame)] = 0.9f;
    }
    CHECK(transcribe_vad_run(fx.session, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    CHECK(fx.result().n_segments == 1);
    transcribe_vad_segment seg;
    transcribe_vad_segment_init(&seg);
    CHECK(transcribe_vad_get_segment(fx.session, 0, &seg) == TRANSCRIBE_OK);
    CHECK(seg.start_sample == 100 * kFrame - 480 && seg.end_sample == 700 * kFrame + 480);
    CHECK(transcribe_vad_get_segment(fx.session, 1, &seg) == TRANSCRIBE_OK);
    CHECK(seg.start_sample == 0 && seg.end_sample == 0);
    seg.struct_size = 4;
    CHECK(transcribe_vad_get_segment(fx.session, 0, &seg) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

}  // namespace

// Segmentation vectors generated from silero_vad.get_speech_timestamps_from_probs.
void run_segment_vectors(int & failures);

int main() {
    test_roles_and_info();
    test_params();
    test_run_frames();
    test_stream();
    test_failures();
    test_exceptions();
    test_segments_api();
    run_segment_vectors(g_failures);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("vad_dispatch_unit: ok\n");
    return EXIT_SUCCESS;
}

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

// Every vector's segments must match the reference sample for sample.
void run_segment_vectors(int & failures) {
    int n_bad = 0;
    for (size_t v = 0; v < sizeof(k_vectors) / sizeof(k_vectors[0]); ++v) {
        const SegmentVector & t = k_vectors[v];
        transcribe_vad_params p;
        transcribe_vad_params_init(&p);
        p.threshold                    = t.threshold;
        p.neg_threshold                = t.neg_threshold;
        p.min_speech_ms                = t.min_speech_ms;
        p.min_silence_ms               = t.min_silence_ms;
        p.speech_pad_ms                = t.speech_pad_ms;
        p.max_speech_ms                = t.max_speech_ms;
        p.min_silence_at_max_speech_ms = t.min_silence_at_max_speech_ms;
        p.use_max_possible_silence     = t.use_max_possible_silence;
        transcribe::VadSegmentParams sp;
        if (transcribe::resolve_vad_params(&p, sp) != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL vector %zu: params rejected\n", v);
            ++n_bad;
            continue;
        }
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
            ++n_bad;
        }
    }
    failures += n_bad;
}
