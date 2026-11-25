// turn_dispatch_unit.cpp - TURN role dispatcher (transcribe-turn.cpp) against
// two fake archs: a clip model (crop, threshold, NOT_IMPLEMENTED stream) and
// a streaming model with a lagged speech track (alignment, chunking, flush,
// run == feed + flush, segments), plus params validation and abort.

#include "transcribe-arch.h"
#include "transcribe-model.h"
#include "transcribe-turn.h"
#include "transcribe.h"
#include "transcribe/turn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

bool g_check_abort = false;

// ---- clip model: max 100 samples, p = the last sample, threshold 0.5 ----

transcribe::TurnModelInfo g_clip_info;
int64_t                   g_clip_last_n     = 0;
float                     g_clip_last_first = 0.0f;

const transcribe::TurnModelInfo & clip_info(const transcribe_model *) {
    return g_clip_info;
}

transcribe_turn_session * new_session() {
    return new transcribe_turn_session();
}

transcribe_status clip_score(transcribe_turn_session * s, const float * pcm, int64_t n, float * p) {
    if (g_check_abort && s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    g_clip_last_n     = n;
    g_clip_last_first = pcm[0];
    *p                = pcm[n - 1];
    return TRANSCRIBE_OK;
}

const transcribe::TurnOps k_clip_ops = { clip_info, new_session, clip_score, nullptr, nullptr };

// ---- streaming model: 4-sample frames, speech lag 2 ----
// turn score of a frame = its second sample; the speech score emitted at
// step s judges frame s - 2 and is that frame's first sample (0 before it).

constexpr int32_t k_frame = 4;
constexpr int32_t k_lag   = 2;

transcribe::TurnModelInfo g_stream_info;

struct StreamSession : transcribe_turn_session {
    std::vector<float> first_samples;  // per stepped frame
};

const transcribe::TurnModelInfo & stream_info(const transcribe_model *) {
    return g_stream_info;
}

transcribe_turn_session * stream_new_session() {
    return new StreamSession();
}

void stream_reset(transcribe_turn_session * s) {
    static_cast<StreamSession *>(s)->first_samples.clear();
}

transcribe_status stream_step(transcribe_turn_session * s, const float * pcm, int64_t n, float * turn, float * speech) {
    if (g_check_abort && s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    auto & hist = static_cast<StreamSession *>(s)->first_samples;
    for (int64_t i = 0; i < n; ++i) {
        const float * f = pcm + i * k_frame;
        hist.push_back(f[0]);
        const int64_t step = static_cast<int64_t>(hist.size()) - 1;
        turn[i]            = f[1];
        speech[i]          = step - k_lag >= 0 ? hist[static_cast<size_t>(step - k_lag)] : 0.0f;
    }
    return TRANSCRIBE_OK;
}

const transcribe::TurnOps k_stream_ops = { stream_info, stream_new_session, nullptr, stream_reset, stream_step };

// The same model with a batched step: one frame per session per call.
int g_batch_calls = 0;

transcribe_status stream_step_batch(transcribe_turn_session * const * sessions,
                                    int                               count,
                                    const float * const *             pcm,
                                    float *                           turn,
                                    float *                           speech) {
    ++g_batch_calls;
    for (int i = 0; i < count; ++i) {
        if (const auto st = stream_step(sessions[i], pcm[i], 1, turn + i, speech + i); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    return TRANSCRIBE_OK;
}

const transcribe::TurnOps k_stream_batch_ops = { stream_info,  stream_new_session, nullptr,
                                                 stream_reset, stream_step,        stream_step_batch };

transcribe::Arch make_arch(const char * name, const transcribe::TurnOps * ops) {
    transcribe::Arch a{};
    a.name = name;
    a.turn = ops;
    return a;
}

const transcribe::Arch k_clip_arch         = make_arch("fake-turn-clip", &k_clip_ops);
const transcribe::Arch k_stream_arch       = make_arch("fake-turn-stream", &k_stream_ops);
const transcribe::Arch k_stream_batch_arch = make_arch("fake-turn-stream-batch", &k_stream_batch_ops);

void setup_infos() {
    g_clip_info                    = {};
    g_clip_info.max_audio_samples  = 100;
    g_clip_info.defaults.threshold = 0.5;

    g_stream_info                                      = {};
    g_stream_info.frame_samples                        = k_frame;
    g_stream_info.has_speech                           = true;
    g_stream_info.speech_lookahead_frames              = k_lag;
    g_stream_info.defaults.threshold                   = 0.9;
    g_stream_info.defaults.speech_threshold            = 0.5;
    g_stream_info.defaults.pre_roll_ms                 = 0;
    g_stream_info.defaults.min_silence_before_check_ms = 0;
    g_stream_info.defaults.max_silence_ms              = 1000000;
    g_stream_info.defaults.min_speech_ms               = 0;
    g_stream_info.defaults.min_confident_speech_ms     = 0;
    g_stream_info.defaults.confident_speech_threshold  = 0.9;
}

transcribe_turn_result get_result(transcribe_turn_session * s) {
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    return r;
}

void test_roles_and_info() {
    transcribe_model model;
    model.arch                  = &k_clip_arch;
    model.roles                 = TRANSCRIBE_ROLE_VAD;  // no TURN bit
    transcribe_turn_session * s = reinterpret_cast<transcribe_turn_session *>(0x1);
    CHECK(transcribe_turn_session_init(&model, nullptr, &s) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);
    CHECK(s == nullptr);
    transcribe_turn_info info;
    transcribe_turn_info_init(&info);
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    model.roles = TRANSCRIBE_ROLE_TURN;
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.max_audio_ms == 6 && info.frame_samples == 0);
    CHECK(!info.has_speech && info.speech_lookahead_frames == 0 && info.threshold == 0.5);

    model.arch = &k_stream_arch;
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.max_audio_ms == 0 && info.frame_samples == k_frame && info.has_speech &&
          info.speech_lookahead_frames == k_lag && info.threshold == 0.9);

    // A short output struct is BAD_STRUCT_SIZE.
    info.struct_size = 8;
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

void test_clip() {
    transcribe_model model;
    model.arch                  = &k_clip_arch;
    model.roles                 = TRANSCRIBE_ROLE_TURN;
    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(&model, nullptr, &s) == TRANSCRIBE_OK);

    // Crop keeps the LAST 100 samples.
    std::vector<float> pcm(250);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = static_cast<float>(i) / 1000.0f;
    }
    pcm.back() = 0.7f;
    CHECK(transcribe_turn_run(s, pcm.data(), 250, nullptr) == TRANSCRIBE_OK);
    CHECK(g_clip_last_n == 100 && g_clip_last_first == 0.15f);
    transcribe_turn_result r = get_result(s);
    CHECK(r.p == 0.7f && r.ended && r.n_probs == 0 && r.n_speech_probs == 0 && r.n_segments == 0);
    CHECK(transcribe_turn_probs(s) == nullptr && transcribe_turn_speech_probs(s) == nullptr);

    // Shorter input is passed whole.
    CHECK(transcribe_turn_run(s, pcm.data(), 10, nullptr) == TRANSCRIBE_OK);
    CHECK(g_clip_last_n == 10);
    CHECK(!get_result(s).ended);  // p = 0.009

    // Threshold override; equality counts as ended.
    transcribe_turn_params p;
    transcribe_turn_params_init(&p);
    CHECK(p.threshold < 0 && p.family == nullptr);
    p.threshold = 0.009f;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_OK);
    CHECK(get_result(s).ended);

    // Invalid input leaves the last result.
    p.threshold = std::numeric_limits<double>::quiet_NaN();
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    p.threshold = 1.5;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(get_result(s).ended);
    pcm[3] = std::numeric_limits<float>::infinity();
    CHECK(transcribe_turn_run(s, pcm.data(), 10, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_run(s, pcm.data(), 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_run(s, nullptr, 10, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(get_result(s).ended);
    pcm[3] = 0.0f;

    // A family extension this model does not accept.
    struct transcribe_ext ext = { sizeof(struct transcribe_ext), 0x58585858u };
    transcribe_turn_params_init(&p);
    p.family = &ext;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    ext.size = 4;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    // A clip model does not stream.
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 10) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    transcribe_turn_stream_reset(s);

    // Abort clears the result.
    g_check_abort = true;
    transcribe_turn_set_abort_callback(s, [](void *) { return true; }, nullptr);
    CHECK(transcribe_turn_run(s, pcm.data(), 10, nullptr) == TRANSCRIBE_ERR_ABORTED);
    CHECK(get_result(s).p == 0.0f && !get_result(s).ended);
    g_check_abort = false;
    transcribe_turn_set_abort_callback(s, nullptr, nullptr);

    transcribe_turn_session_free(s);
}

// n frames; frame t: first sample = speech value, second = turn value.
std::vector<float> make_stream(const std::vector<float> & speech, const std::vector<float> & turn, int tail = 0) {
    std::vector<float> pcm;
    for (size_t t = 0; t < speech.size(); ++t) {
        pcm.insert(pcm.end(), { speech[t], turn[t], 0.0f, 0.0f });
    }
    if (tail > 0) {
        pcm.resize(pcm.size() - static_cast<size_t>(k_frame - tail));
    }
    return pcm;
}

void test_stream() {
    transcribe_model model;
    model.arch                  = &k_stream_arch;
    model.roles                 = TRANSCRIBE_ROLE_TURN;
    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(&model, nullptr, &s) == TRANSCRIBE_OK);

    const std::vector<float> speech = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f };
    const std::vector<float> turn   = { 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f };
    const std::vector<float> pcm    = make_stream(speech, turn, 2);  // last frame partial: 2 samples
    const int                n      = static_cast<int>(pcm.size());  // 26

    // Offline reference.
    CHECK(transcribe_turn_run(s, pcm.data(), n, nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r = get_result(s);
    CHECK(r.n_probs == 7 && r.first_frame == 0 && r.n_speech_probs == 7 && r.speech_first_frame == 0);
    const std::vector<float> run_turn(transcribe_turn_probs(s), transcribe_turn_probs(s) + 7);
    const std::vector<float> run_speech(transcribe_turn_speech_probs(s), transcribe_turn_speech_probs(s) + 7);
    CHECK(run_turn == turn);
    CHECK(run_speech == speech);
    CHECK(r.p == 0.07f && !r.ended);

    // Streamed in odd chunk sizes, then flushed: the same tracks, aligned.
    for (int chunk : { 1, 3, 5, 26 }) {
        std::vector<float> got_turn, got_speech;
        int64_t            next_turn = 0, next_speech = 0;
        auto               collect = [&] {
            transcribe_turn_result q = get_result(s);
            CHECK(q.first_frame == next_turn);
            CHECK(q.speech_first_frame == next_speech);
            CHECK(q.p == 0.0f && !q.ended && q.n_segments == 0);
            if (q.n_probs > 0) {
                got_turn.insert(got_turn.end(), transcribe_turn_probs(s), transcribe_turn_probs(s) + q.n_probs);
            }
            if (q.n_speech_probs > 0) {
                got_speech.insert(got_speech.end(), transcribe_turn_speech_probs(s),
                                  transcribe_turn_speech_probs(s) + q.n_speech_probs);
            }
            next_turn += q.n_probs;
            next_speech += q.n_speech_probs;
            // The speech track trails by the lag until the flush.
            CHECK(next_speech <= std::max<int64_t>(0, next_turn - k_lag) || chunk == -1);
        };
        for (int off = 0; off < n; off += chunk) {
            CHECK(transcribe_turn_stream_feed(s, pcm.data() + off, std::min(chunk, n - off)) == TRANSCRIBE_OK);
            collect();
        }
        CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
        {
            transcribe_turn_result q = get_result(s);
            CHECK(q.first_frame == next_turn && q.speech_first_frame == next_speech);
            got_turn.insert(got_turn.end(), transcribe_turn_probs(s), transcribe_turn_probs(s) + q.n_probs);
            got_speech.insert(got_speech.end(), transcribe_turn_speech_probs(s),
                              transcribe_turn_speech_probs(s) + q.n_speech_probs);
        }
        CHECK(got_turn == run_turn);
        CHECK(got_speech == run_speech);
        // The next feed starts a new stream at frame 0.
        CHECK(transcribe_turn_stream_feed(s, pcm.data(), 0) == TRANSCRIBE_OK);
        CHECK(get_result(s).first_frame == 0);
    }

    // Flush of an empty stream returns nothing.
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    CHECK(get_result(s).n_probs == 0 && get_result(s).n_speech_probs == 0);

    // Reset discards the stream; frame numbering restarts.
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 12) == TRANSCRIBE_OK);
    CHECK(get_result(s).n_probs == 3 && get_result(s).n_speech_probs == 1);
    transcribe_turn_stream_reset(s);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(s).first_frame == 0 && get_result(s).n_speech_probs == 0);

    // run discards the live stream.
    CHECK(transcribe_turn_run(s, pcm.data(), n, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(s).first_frame == 0);

    // Invalid feed input leaves the stream and result.
    CHECK(transcribe_turn_stream_feed(s, nullptr, 4) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), -1) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(get_result(s).n_probs == 1);

    // An aborted feed resets the stream and clears the result.
    g_check_abort = true;
    transcribe_turn_set_abort_callback(s, [](void *) { return true; }, nullptr);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 8) == TRANSCRIBE_ERR_ABORTED);
    CHECK(get_result(s).n_probs == 0);
    g_check_abort = false;
    transcribe_turn_set_abort_callback(s, nullptr, nullptr);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(s).first_frame == 0);

    transcribe_turn_session_free(s);
}

struct Tracks {
    std::vector<float> turn, speech;
    int64_t            next_turn = 0, next_speech = 0;
};

void collect(transcribe_turn_session * s, Tracks & t) {
    transcribe_turn_result r = get_result(s);
    CHECK(r.first_frame == t.next_turn && r.speech_first_frame == t.next_speech);
    if (r.n_probs > 0) {
        t.turn.insert(t.turn.end(), transcribe_turn_probs(s), transcribe_turn_probs(s) + r.n_probs);
    }
    if (r.n_speech_probs > 0) {
        t.speech.insert(t.speech.end(), transcribe_turn_speech_probs(s),
                        transcribe_turn_speech_probs(s) + r.n_speech_probs);
    }
    t.next_turn += r.n_probs;
    t.next_speech += r.n_speech_probs;
}

// feed_batch == each session fed alone, for ragged chunks, with and without
// a batched step; validation before any change; failures reset every stream.
void test_feed_batch(const transcribe::Arch * arch) {
    transcribe_model model;
    model.arch                    = arch;
    model.roles                   = TRANSCRIBE_ROLE_TURN;
    constexpr int             k_n = 3;
    transcribe_turn_session * batch[k_n]{};
    transcribe_turn_session * alone[k_n]{};
    for (int i = 0; i < k_n; ++i) {
        CHECK(transcribe_turn_session_init(&model, nullptr, &batch[i]) == TRANSCRIBE_OK);
        CHECK(transcribe_turn_session_init(&model, nullptr, &alone[i]) == TRANSCRIBE_OK);
    }
    // Three different streams of 9 frames.
    std::vector<std::vector<float>> pcm(k_n);
    for (int i = 0; i < k_n; ++i) {
        for (int t = 0; t < 9; ++t) {
            pcm[i].insert(pcm[i].end(), { 0.1f * t + i, 0.01f * t + 0.1f * i, 0.0f, 0.0f });
        }
    }
    const int chunks[k_n] = { 3, 4, 9 };  // samples per call
    Tracks    tb[k_n], ta[k_n];
    size_t    off[k_n] = {};
    for (int call = 0; call < 12; ++call) {
        const float * p[k_n];
        int           len[k_n];
        for (int i = 0; i < k_n; ++i) {
            len[i] = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunks[i]), pcm[i].size() - off[i]));
            p[i]   = len[i] > 0 ? pcm[i].data() + off[i] : nullptr;
        }
        CHECK(transcribe_turn_stream_feed_batch(batch, p, len, k_n) == TRANSCRIBE_OK);
        for (int i = 0; i < k_n; ++i) {
            collect(batch[i], tb[i]);
            CHECK(transcribe_turn_stream_feed(alone[i], p[i], len[i]) == TRANSCRIBE_OK);
            collect(alone[i], ta[i]);
            off[i] += static_cast<size_t>(len[i]);
        }
    }
    for (int i = 0; i < k_n; ++i) {
        CHECK(transcribe_turn_stream_flush(batch[i]) == TRANSCRIBE_OK);
        collect(batch[i], tb[i]);
        CHECK(transcribe_turn_stream_flush(alone[i]) == TRANSCRIBE_OK);
        collect(alone[i], ta[i]);
        CHECK(tb[i].turn == ta[i].turn && tb[i].speech == ta[i].speech);
        CHECK(tb[i].turn.size() == 9 && tb[i].speech.size() == 9);
    }

    // Invalid input changes nothing.
    CHECK(transcribe_turn_stream_feed(batch[0], pcm[0].data(), 4) == TRANSCRIBE_OK);
    const float *             p2[2]   = { pcm[0].data(), pcm[1].data() };
    int                       len2[2] = { 4, 4 };
    transcribe_turn_session * dup[2]  = { batch[0], batch[0] };
    CHECK(transcribe_turn_stream_feed_batch(dup, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_turn_session * none[2] = { batch[0], nullptr };
    CHECK(transcribe_turn_stream_feed_batch(none, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_stream_feed_batch(batch, p2, len2, 0) == TRANSCRIBE_ERR_INVALID_ARG);
    std::vector<float> bad(4, std::numeric_limits<float>::quiet_NaN());
    const float *      p3[2] = { pcm[0].data(), bad.data() };
    CHECK(transcribe_turn_stream_feed_batch(batch, p3, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_model other;
    other.arch                        = arch;
    other.roles                       = TRANSCRIBE_ROLE_TURN;
    transcribe_turn_session * foreign = nullptr;
    CHECK(transcribe_turn_session_init(&other, nullptr, &foreign) == TRANSCRIBE_OK);
    transcribe_turn_session * mixed[2] = { batch[0], foreign };
    CHECK(transcribe_turn_stream_feed_batch(mixed, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(get_result(batch[0]).n_probs == 1 && get_result(batch[0]).first_frame == 0);

    // An abort resets every listed stream and clears every result.
    g_check_abort = true;
    transcribe_turn_set_abort_callback(batch[1], [](void *) { return true; }, nullptr);
    CHECK(transcribe_turn_stream_feed_batch(batch, p2, len2, 2) == TRANSCRIBE_ERR_ABORTED);
    CHECK(get_result(batch[0]).n_probs == 0 && get_result(batch[1]).n_probs == 0);
    g_check_abort = false;
    transcribe_turn_set_abort_callback(batch[1], nullptr, nullptr);
    CHECK(transcribe_turn_stream_feed(batch[0], pcm[0].data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(batch[0]).first_frame == 0);

    transcribe_turn_session_free(foreign);
    for (int i = 0; i < k_n; ++i) {
        transcribe_turn_session_free(batch[i]);
        transcribe_turn_session_free(alone[i]);
    }
}

void test_feed_batch_clip() {
    transcribe_model model;
    model.arch                  = &k_clip_arch;
    model.roles                 = TRANSCRIBE_ROLE_TURN;
    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(&model, nullptr, &s) == TRANSCRIBE_OK);
    std::vector<float> pcm(8, 0.0f);
    const float *      p[1]   = { pcm.data() };
    int                len[1] = { 8 };
    CHECK(transcribe_turn_stream_feed_batch(&s, p, len, 1) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    transcribe_turn_session_free(s);
}

// Turn segmentation, worked by hand from inworld.sava.segmenter.
// 16 kHz, 640-sample (40 ms) frames, lag 3.
void test_segments() {
    transcribe::TurnPolicy p;
    p.threshold                   = 0.9;
    p.speech_threshold            = 0.5;
    p.pre_roll_ms                 = 80;   // 2 frames
    p.min_silence_before_check_ms = 0;
    p.max_silence_ms              = 200;  // 5 frames
    p.min_speech_ms               = 120;  // 3 frames
    p.min_confident_speech_ms     = 80;
    p.confident_speech_threshold  = 0.95;

    // frames:        0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19
    std::vector<float> speech = { 0, 0, 0, .9f, .9f, .9f, .9f, 0, 0, 0, .6f, 0, 0, 0, 0, 0, 0, 0, .96f, .96f };
    const int64_t      n      = static_cast<int64_t>(speech.size());
    std::vector<float> turn(static_cast<size_t>(n + 3), 0.0f);
    turn[10] = 0.95f;  // judged frame 7 (first pause frame) reads step 10: closes at step 10 -> end frame 11

    std::vector<transcribe::TurnSegmentEntry> out;
    transcribe::turn_frames_to_segments(speech.data(), turn.data(), n, n * 640 - 100, 640, 3, p, out);
    // Turn 1: opens at frame 3, start 1; closes by the model at end frame 11.
    // Turn 2: frame 10 only (40 ms speech, not confident) -> dropped; it
    //         closes by silence after 5 pause frames.
    // Turn 3: opens at 18, start 16; 80 ms of confident speech keeps it; open
    //         at the end -> closes at frame 20, clipped to the input.
    CHECK(out.size() == 2);
    if (out.size() == 2) {
        CHECK(out[0].start_sample == 1 * 640 && out[0].end_sample == 11 * 640 &&
              out[0].end_reason == TRANSCRIBE_TURN_END_MODEL);
        CHECK(out[1].start_sample == 16 * 640 && out[1].end_sample == n * 640 - 100 &&
              out[1].end_reason == TRANSCRIBE_TURN_END_AUDIO);
    }

    // Keep turn 2: start 8; its 5th pause frame is judged 15 (step 18), so it
    // closes by silence at end frame 19.
    p.min_speech_ms = 40;
    transcribe::turn_frames_to_segments(speech.data(), turn.data(), n, n * 640, 640, 3, p, out);
    CHECK(out.size() == 3);
    if (out.size() == 3) {
        CHECK(out[1].start_sample == 8 * 640 && out[1].end_sample == 19 * 640 &&
              out[1].end_reason == TRANSCRIBE_TURN_END_SILENCE);
    }
}

void test_params() {
    transcribe::TurnModelInfo info;
    info.defaults.threshold   = 0.9;
    info.defaults.pre_roll_ms = 800;
    transcribe::TurnPolicy out;
    CHECK(transcribe::resolve_turn_params(nullptr, info, out) == TRANSCRIBE_OK);
    CHECK(out.threshold == 0.9 && out.pre_roll_ms == 800);
    transcribe_turn_params p;
    transcribe_turn_params_init(&p);
    p.threshold   = 0.0;
    p.pre_roll_ms = 0;
    CHECK(transcribe::resolve_turn_params(&p, info, out) == TRANSCRIBE_OK);
    CHECK(out.threshold == 0.0 && out.pre_roll_ms == 0);
    p.speech_threshold = std::numeric_limits<double>::quiet_NaN();
    CHECK(transcribe::resolve_turn_params(&p, info, out) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_turn_params_init(&p);
    p.struct_size = 16;
    CHECK(transcribe::resolve_turn_params(&p, info, out) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

}  // namespace

int main() {
    setup_infos();
    test_roles_and_info();
    test_clip();
    test_stream();
    test_feed_batch(&k_stream_arch);
    g_batch_calls = 0;
    test_feed_batch(&k_stream_batch_arch);
    CHECK(g_batch_calls > 0);
    test_feed_batch_clip();
    test_segments();
    test_params();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("turn_dispatch_unit: OK\n");
    return EXIT_SUCCESS;
}
