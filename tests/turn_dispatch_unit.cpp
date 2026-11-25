// turn_dispatch_unit.cpp - TURN role dispatcher (transcribe-turn.cpp) against
// fake archs: a clip model (crop, threshold, no stream) and streaming models
// with and without a lagged speech track (alignment, chunking, flush, run ==
// feed + flush, feed_batch, thread counts), plus params validation, abort and
// throwing ops.

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

bool g_check_abort = false;

bool abort_now(void *) {
    return true;
}

void set_abort(transcribe_turn_session * s, bool on) {
    g_check_abort = on;
    transcribe_turn_set_abort_callback(s, on ? abort_now : nullptr, nullptr);
}

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

// ---- streaming models: 4-sample frames ----
// turn score of a frame = its second sample; with a speech track (lag 2) the
// speech score emitted at step s judges frame s - 2 and is that frame's first
// sample (0 before it).

constexpr int32_t k_frame = 4;
constexpr int32_t k_lag   = 2;

transcribe::TurnModelInfo g_stream_info;
transcribe::TurnModelInfo g_nospeech_info;
int                       g_step_n_threads_min = 0;
int                       g_step_n_threads_max = 0;

struct StreamSession : transcribe_turn_session {
    std::vector<float> first_samples;  // per stepped frame
};

const transcribe::TurnModelInfo & stream_info(const transcribe_model *) {
    return g_stream_info;
}

const transcribe::TurnModelInfo & nospeech_info(const transcribe_model *) {
    return g_nospeech_info;
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
    CHECK((speech != nullptr) == s->model->arch->turn->info(s->model).has_speech);
    g_step_n_threads_min = std::min(g_step_n_threads_min, s->n_threads);
    g_step_n_threads_max = std::max(g_step_n_threads_max, s->n_threads);
    auto & hist          = static_cast<StreamSession *>(s)->first_samples;
    for (int64_t i = 0; i < n; ++i) {
        const float * f = pcm + i * k_frame;
        hist.push_back(f[0]);
        const int64_t step = static_cast<int64_t>(hist.size()) - 1;
        turn[i]            = f[1];
        if (speech != nullptr) {
            speech[i] = step - k_lag >= 0 ? hist[static_cast<size_t>(step - k_lag)] : 0.0f;
        }
    }
    return TRANSCRIBE_OK;
}

// While g_throw_step is set the step throws; the reset throws the next
// g_throw_reset times.
bool g_throw_step  = false;
int  g_throw_reset = 0;
int  g_reset_calls = 0;

void throwing_reset(transcribe_turn_session * s) {
    ++g_reset_calls;
    if (g_throw_reset > 0) {
        --g_throw_reset;
        throw std::runtime_error("fake reset failure");
    }
    stream_reset(s);
}

transcribe_status throwing_step(transcribe_turn_session * s,
                                const float *             pcm,
                                int64_t                   n,
                                float *                   turn,
                                float *                   speech) {
    if (g_throw_step) {
        throw std::runtime_error("fake step failure");
    }
    return stream_step(s, pcm, n, turn, speech);
}

transcribe_status throwing_score(transcribe_turn_session *, const float *, int64_t, float *) {
    throw std::runtime_error("fake score failure");
}

using StepFn = transcribe_status (*)(transcribe_turn_session *, const float *, int64_t, float *, float *);

int g_batch_calls     = 0;
int g_batch_n_threads = -1;  // n_threads of the last batched step

// A batched step: Step on one frame of each session.
template <StepFn Step>
transcribe_status step_batch(transcribe_turn_session * const * sessions,
                             int                               count,
                             int                               n_threads,
                             const float * const *             pcm,
                             float *                           turn,
                             float *                           speech) {
    ++g_batch_calls;
    g_batch_n_threads = n_threads;
    for (int i = 0; i < count; ++i) {
        if (const auto st = Step(sessions[i], pcm[i], 1, turn + i, speech != nullptr ? speech + i : nullptr);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    return TRANSCRIBE_OK;
}

const transcribe::TurnOps k_clip_ops         = { clip_info, new_session, clip_score };
const transcribe::TurnOps k_stream_ops       = { stream_info, stream_new_session, nullptr, stream_reset, stream_step };
const transcribe::TurnOps k_stream_batch_ops = { stream_info,  stream_new_session, nullptr,
                                                 stream_reset, stream_step,        step_batch<stream_step> };
const transcribe::TurnOps k_nospeech_ops = { nospeech_info, stream_new_session, nullptr, stream_reset, stream_step };
const transcribe::TurnOps k_nospeech_batch_ops = { nospeech_info, stream_new_session, nullptr,
                                                   stream_reset,  stream_step,        step_batch<stream_step> };
const transcribe::TurnOps k_throw_ops = { stream_info, stream_new_session, nullptr, throwing_reset, throwing_step };
const transcribe::TurnOps k_throw_batch_ops = { stream_info,    stream_new_session, nullptr,
                                                throwing_reset, throwing_step,      step_batch<throwing_step> };
const transcribe::TurnOps k_throw_clip_ops  = { clip_info, new_session, throwing_score };

transcribe::Arch make_arch(const char * name, const transcribe::TurnOps * ops) {
    transcribe::Arch a{};
    a.name = name;
    a.turn = ops;
    return a;
}

const transcribe::Arch k_clip_arch           = make_arch("fake-turn-clip", &k_clip_ops);
const transcribe::Arch k_stream_arch         = make_arch("fake-turn-stream", &k_stream_ops);
const transcribe::Arch k_stream_batch_arch   = make_arch("fake-turn-stream-batch", &k_stream_batch_ops);
const transcribe::Arch k_nospeech_arch       = make_arch("fake-turn-nospeech", &k_nospeech_ops);
const transcribe::Arch k_nospeech_batch_arch = make_arch("fake-turn-nospeech-batch", &k_nospeech_batch_ops);
const transcribe::Arch k_throw_arch          = make_arch("fake-turn-throw", &k_throw_ops);
const transcribe::Arch k_throw_batch_arch    = make_arch("fake-turn-throw-batch", &k_throw_batch_ops);
const transcribe::Arch k_throw_clip_arch     = make_arch("fake-turn-throw-clip", &k_throw_clip_ops);

void setup_infos() {
    g_clip_info.max_audio_samples = 100;
    g_clip_info.threshold         = 0.5;

    g_stream_info.frame_samples           = k_frame;
    g_stream_info.has_speech              = true;
    g_stream_info.speech_lookahead_frames = k_lag;
    g_stream_info.threshold               = 0.9;

    g_nospeech_info.frame_samples = k_frame;
}

// Sets model to arch with the TURN role and opens a session on it.
transcribe_turn_session * open_session(transcribe_model & model, const transcribe::Arch * arch, int n_threads = 0) {
    model.arch  = arch;
    model.roles = TRANSCRIBE_ROLE_TURN;
    transcribe_turn_session_params sp;
    transcribe_turn_session_params_init(&sp);
    sp.n_threads                = n_threads;
    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(&model, &sp, &s) == TRANSCRIBE_OK);
    return s;
}

transcribe_turn_result get_result(transcribe_turn_session * s) {
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    return r;
}

struct Tracks {
    std::vector<float> turn, speech;
    int64_t            next_turn = 0, next_speech = 0;
};

// Appends a feed / flush result, checking it continues the tracks.
void collect(transcribe_turn_session * s, Tracks & t) {
    const transcribe_turn_result r = get_result(s);
    CHECK(r.first_frame == t.next_turn && r.speech_first_frame == t.next_speech);
    CHECK(r.p == 0.0f && !r.ended);
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

// Frame t: first sample = speech value, second = turn value; the last frame
// keeps `tail` samples when tail > 0.
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
    double thr = -1.0;
    CHECK(transcribe::resolve_turn_threshold(&model, nullptr, thr) == TRANSCRIBE_ERR_UNSUPPORTED_ROLE);

    model.roles = TRANSCRIBE_ROLE_TURN;
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.sample_rate == 16000 && info.max_audio_ms == 6 && info.frame_samples == 0);
    CHECK(!info.has_speech && info.threshold == 0.5);

    model.arch = &k_stream_arch;
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(info.max_audio_ms == 0 && info.frame_samples == k_frame && info.has_speech && info.threshold == 0.9);

    info.struct_size = 8;
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
}

void test_clip() {
    transcribe_model          model;
    transcribe_turn_session * s = open_session(model, &k_clip_arch);

    // Crop keeps the LAST 100 samples.
    std::vector<float> pcm(250);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = static_cast<float>(i) / 1000.0f;
    }
    pcm.back() = 0.7f;
    CHECK(transcribe_turn_run(s, pcm.data(), 250, nullptr) == TRANSCRIBE_OK);
    CHECK(g_clip_last_n == 100 && g_clip_last_first == 0.15f);
    transcribe_turn_result r = get_result(s);
    CHECK(r.p == 0.7f && r.ended && r.n_probs == 0 && r.n_speech_probs == 0);
    CHECK(transcribe_turn_probs(s) == nullptr && transcribe_turn_speech_probs(s) == nullptr);

    // Shorter input is passed whole.
    CHECK(transcribe_turn_run(s, pcm.data(), 10, nullptr) == TRANSCRIBE_OK);
    CHECK(g_clip_last_n == 10);
    CHECK(!get_result(s).ended);  // p = 0.009

    // Threshold override; equality counts as ended.
    transcribe_turn_params p;
    transcribe_turn_params_init(&p);
    CHECK(p.threshold == -1.0);
    p.threshold = 0.009f;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_OK);
    CHECK(get_result(s).ended);
    p.threshold = 0.0;
    CHECK(transcribe_turn_run(s, pcm.data(), 1, &p) == TRANSCRIBE_OK);
    CHECK(get_result(s).p == 0.0f && get_result(s).ended);

    // Invalid input leaves the last result.
    p.threshold = std::numeric_limits<double>::quiet_NaN();
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    p.threshold = 1.5;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_turn_params_init(&p);
    p.struct_size = 8;
    CHECK(transcribe_turn_run(s, pcm.data(), 10, &p) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    pcm[3] = std::numeric_limits<float>::infinity();
    CHECK(transcribe_turn_run(s, pcm.data(), 10, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_run(s, pcm.data(), 0, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_run(s, nullptr, 10, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(get_result(s).ended);
    pcm[3] = 0.0f;

    // A clip model does not stream.
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 10) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    const float * p1[1]   = { pcm.data() };
    int           len1[1] = { 8 };
    CHECK(transcribe_turn_stream_feed_batch(&s, p1, len1, 1) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
    transcribe_turn_stream_reset(s);

    // Abort clears the result.
    set_abort(s, true);
    CHECK(transcribe_turn_run(s, pcm.data(), 10, nullptr) == TRANSCRIBE_ERR_ABORTED);
    CHECK(get_result(s).p == 0.0f && !get_result(s).ended);
    set_abort(s, false);

    transcribe_turn_session_free(s);
}

void test_stream() {
    transcribe_model          model;
    transcribe_turn_session * s = open_session(model, &k_stream_arch);

    const std::vector<float> speech = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f };
    const std::vector<float> turn   = { 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f };
    const std::vector<float> pcm    = make_stream(speech, turn, 2);
    const int                n      = static_cast<int>(pcm.size());  // 26

    CHECK(transcribe_turn_run(s, pcm.data(), n, nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r = get_result(s);
    CHECK(r.n_probs == 7 && r.first_frame == 0 && r.n_speech_probs == 7 && r.speech_first_frame == 0);
    CHECK(std::vector<float>(transcribe_turn_probs(s), transcribe_turn_probs(s) + 7) == turn);
    CHECK(std::vector<float>(transcribe_turn_speech_probs(s), transcribe_turn_speech_probs(s) + 7) == speech);
    CHECK(r.p == 0.07f && !r.ended);

    // Streamed in odd chunk sizes, then flushed: the same tracks, aligned.
    for (int chunk : { 1, 3, 5, 26 }) {
        Tracks t;
        for (int off = 0; off < n; off += chunk) {
            CHECK(transcribe_turn_stream_feed(s, pcm.data() + off, std::min(chunk, n - off)) == TRANSCRIBE_OK);
            collect(s, t);
            CHECK(t.next_speech <= std::max<int64_t>(0, t.next_turn - k_lag));
        }
        CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
        collect(s, t);
        CHECK(t.turn == turn && t.speech == speech);
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
    set_abort(s, true);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 8) == TRANSCRIBE_ERR_ABORTED);
    CHECK(get_result(s).n_probs == 0);
    set_abort(s, false);
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(s).first_frame == 0);

    transcribe_turn_session_free(s);
}

// feed_batch == each session fed alone, for ragged chunks; validation before
// any change; an abort resets every stream.
void test_feed_batch(const transcribe::Arch * arch) {
    constexpr int             k_n = 3;
    transcribe_model          model;
    transcribe_turn_session * batch[k_n]{};
    transcribe_turn_session * alone[k_n]{};
    std::vector<float>        pcm[k_n];  // three different streams of 9 frames
    for (int i = 0; i < k_n; ++i) {
        batch[i] = open_session(model, arch);
        alone[i] = open_session(model, arch);
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

    // Invalid input changes nothing: a duplicate, a null session (first or
    // not), count 0, non-finite PCM, a session of another model.
    CHECK(transcribe_turn_stream_feed(batch[0], pcm[0].data(), 4) == TRANSCRIBE_OK);
    const float *             p2[2]   = { pcm[0].data(), pcm[1].data() };
    int                       len2[2] = { 4, 4 };
    transcribe_turn_session * dup[2]  = { batch[0], batch[0] };
    CHECK(transcribe_turn_stream_feed_batch(dup, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_turn_session * none[2] = { batch[0], nullptr };
    CHECK(transcribe_turn_stream_feed_batch(none, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_turn_session * first_none[2] = { nullptr, batch[0] };
    CHECK(transcribe_turn_stream_feed_batch(first_none, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_turn_stream_feed_batch(batch, p2, len2, 0) == TRANSCRIBE_ERR_INVALID_ARG);
    std::vector<float> bad(4, std::numeric_limits<float>::quiet_NaN());
    const float *      p3[2] = { pcm[0].data(), bad.data() };
    CHECK(transcribe_turn_stream_feed_batch(batch, p3, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_model          other;
    transcribe_turn_session * foreign  = open_session(other, arch);
    transcribe_turn_session * mixed[2] = { batch[0], foreign };
    CHECK(transcribe_turn_stream_feed_batch(mixed, p2, len2, 2) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(get_result(batch[0]).n_probs == 1 && get_result(batch[0]).first_frame == 0);

    set_abort(batch[1], true);
    CHECK(transcribe_turn_stream_feed_batch(batch, p2, len2, 2) == TRANSCRIBE_ERR_ABORTED);
    CHECK(get_result(batch[0]).n_probs == 0 && get_result(batch[1]).n_probs == 0);
    set_abort(batch[1], false);
    CHECK(transcribe_turn_stream_feed(batch[0], pcm[0].data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(batch[0]).first_frame == 0);

    transcribe_turn_session_free(foreign);
    for (int i = 0; i < k_n; ++i) {
        transcribe_turn_session_free(batch[i]);
        transcribe_turn_session_free(alone[i]);
    }
}

// feed_batch runs with sessions[0]'s thread count, batched or not, and
// leaves every session's own count as it was.
void test_feed_batch_threads(const transcribe::Arch * arch, bool batched) {
    constexpr int             k_n = 3;
    transcribe_model          model;
    transcribe_turn_session * s[k_n]{};
    for (int i = 0; i < k_n; ++i) {
        s[i] = open_session(model, arch, 3 - i);
    }
    std::vector<float> pcm(8, 0.25f);
    const float *      p[k_n]   = { pcm.data(), pcm.data(), pcm.data() };
    int                len[k_n] = { 0, 8, 8 };  // sessions[0] has no frame
    g_batch_n_threads           = -1;
    g_step_n_threads_min        = 1 << 30;
    g_step_n_threads_max        = -1;
    CHECK(transcribe_turn_stream_feed_batch(s, p, len, k_n) == TRANSCRIBE_OK);
    if (batched) {
        CHECK(g_batch_n_threads == 3);
    } else {
        CHECK(g_step_n_threads_min == 3 && g_step_n_threads_max == 3);
    }
    for (int i = 0; i < k_n; ++i) {
        CHECK(s[i]->n_threads == 3 - i);
        transcribe_turn_session_free(s[i]);
    }
}

// Without a speech track: no speech probabilities from run, feed, feed_batch
// or flush, and the step gets speech == nullptr.
void test_no_speech(const transcribe::Arch * arch) {
    constexpr int             k_n = 2;
    transcribe_model          model;
    transcribe_turn_session * s[k_n]{};
    for (int i = 0; i < k_n; ++i) {
        s[i] = open_session(model, arch);
    }
    transcribe_turn_info info;
    transcribe_turn_info_init(&info);
    CHECK(transcribe_turn_get_info(&model, &info) == TRANSCRIBE_OK);
    CHECK(!info.has_speech && info.frame_samples == k_frame);

    const std::vector<float> turn = { 0.01f, 0.02f, 0.03f };
    const std::vector<float> pcm  = make_stream({ 0.1f, 0.2f, 0.3f }, turn, 2);  // 10 samples

    CHECK(transcribe_turn_run(s[0], pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r = get_result(s[0]);
    CHECK(r.n_probs == 3 && r.n_speech_probs == 0 && r.speech_first_frame == 0);
    CHECK(transcribe_turn_speech_probs(s[0]) == nullptr);
    CHECK(std::vector<float>(transcribe_turn_probs(s[0]), transcribe_turn_probs(s[0]) + 3) == turn);

    const float * p[k_n]   = { pcm.data(), pcm.data() };
    int           len[k_n] = { 10, 6 };
    CHECK(transcribe_turn_stream_feed_batch(s, p, len, k_n) == TRANSCRIBE_OK);
    r = get_result(s[0]);
    CHECK(r.n_probs == 2 && r.n_speech_probs == 0 && r.speech_first_frame == 0);
    r = get_result(s[1]);
    CHECK(r.n_probs == 1 && r.n_speech_probs == 0);
    CHECK(transcribe_turn_stream_feed(s[1], pcm.data() + 6, 4) == TRANSCRIBE_OK);
    r = get_result(s[1]);
    CHECK(r.n_probs == 1 && r.first_frame == 1 && r.n_speech_probs == 0 && r.speech_first_frame == 0);
    CHECK(transcribe_turn_stream_flush(s[0]) == TRANSCRIBE_OK);
    r = get_result(s[0]);
    CHECK(r.n_probs == 1 && r.first_frame == 2 && r.n_speech_probs == 0);
    CHECK(transcribe_turn_probs(s[0])[0] == 0.03f);
    for (int i = 0; i < k_n; ++i) {
        transcribe_turn_session_free(s[i]);
    }
}

size_t history(transcribe_turn_session * s) {
    return static_cast<StreamSession *>(s)->first_samples.size();
}

// An exception from an arch op never escapes: the call fails with BACKEND,
// the result is cleared, the stream is reset, and a reset that itself threw
// is retried before the next step.
void test_throwing_ops(const transcribe::Arch * arch) {
    constexpr int             k_n = 2;
    transcribe_model          model;
    transcribe_turn_session * s[k_n]{};
    for (int i = 0; i < k_n; ++i) {
        s[i] = open_session(model, arch);
    }
    std::vector<float> pcm(12, 0.5f);
    auto               cleared = [&](int i) {
        return get_result(s[i]).n_probs == 0 && history(s[i]) == 0;
    };

    CHECK(transcribe_turn_stream_feed(s[0], pcm.data(), 8) == TRANSCRIBE_OK);
    CHECK(get_result(s[0]).n_probs == 2);
    g_throw_step = true;
    CHECK(transcribe_turn_stream_feed(s[0], pcm.data(), 4) == TRANSCRIBE_ERR_BACKEND);
    g_throw_step = false;
    CHECK(cleared(0));
    CHECK(transcribe_turn_stream_feed(s[0], pcm.data(), 4) == TRANSCRIBE_OK);
    CHECK(get_result(s[0]).first_frame == 0 && get_result(s[0]).n_probs == 1);

    // The cleanup's reset throws too: the next feed retries it first.
    g_throw_step  = true;
    g_throw_reset = 1;
    CHECK(transcribe_turn_stream_feed(s[0], pcm.data(), 4) == TRANSCRIBE_ERR_BACKEND);
    g_throw_step     = false;
    const int resets = g_reset_calls;
    CHECK(history(s[0]) == 1);  // the failed reset left the fake's state
    CHECK(transcribe_turn_stream_feed(s[0], pcm.data(), 4) == TRANSCRIBE_OK);
    CHECK(g_reset_calls == resets + 1);
    CHECK(history(s[0]) == 1 && get_result(s[0]).first_frame == 0);

    // run and flush.
    g_throw_step = true;
    CHECK(transcribe_turn_run(s[0], pcm.data(), 8, nullptr) == TRANSCRIBE_ERR_BACKEND);
    g_throw_step = false;
    CHECK(cleared(0));
    CHECK(transcribe_turn_stream_feed(s[0], pcm.data(), 2) == TRANSCRIBE_OK);
    g_throw_step = true;
    CHECK(transcribe_turn_stream_flush(s[0]) == TRANSCRIBE_ERR_BACKEND);
    g_throw_step = false;
    CHECK(cleared(0));

    // feed_batch: a throwing step resets every listed stream.
    const float * p[k_n]   = { pcm.data(), pcm.data() };
    int           len[k_n] = { 4, 4 };
    for (int i = 0; i < k_n; ++i) {
        CHECK(transcribe_turn_stream_feed(s[i], pcm.data(), 8) == TRANSCRIBE_OK);
    }
    g_throw_step = true;
    CHECK(transcribe_turn_stream_feed_batch(s, p, len, k_n) == TRANSCRIBE_ERR_BACKEND);
    g_throw_step = false;
    CHECK(cleared(0) && cleared(1));

    // feed_batch: s[1]'s retried reset throws while setting up; every listed
    // stream is reset, s[0] included though it was already set up.
    for (int i = 0; i < k_n; ++i) {
        CHECK(transcribe_turn_stream_feed(s[i], pcm.data(), 8) == TRANSCRIBE_OK);
    }
    g_throw_step  = true;
    g_throw_reset = 1;
    CHECK(transcribe_turn_stream_feed(s[1], pcm.data(), 4) == TRANSCRIBE_ERR_BACKEND);  // s[1] needs a reset
    g_throw_step  = false;
    g_throw_reset = 1;
    CHECK(transcribe_turn_stream_feed_batch(s, p, len, k_n) == TRANSCRIBE_ERR_BACKEND);
    CHECK(g_throw_reset == 0);
    for (int i = 0; i < k_n; ++i) {
        CHECK(cleared(i));
        CHECK(transcribe_turn_stream_feed(s[i], pcm.data(), 4) == TRANSCRIBE_OK);
        CHECK(get_result(s[i]).first_frame == 0);
        transcribe_turn_session_free(s[i]);
    }

    // A throwing score_clip.
    transcribe_model          clip;
    transcribe_turn_session * c = open_session(clip, &k_throw_clip_arch);
    CHECK(transcribe_turn_run(c, pcm.data(), 4, nullptr) == TRANSCRIBE_ERR_BACKEND);
    CHECK(get_result(c).p == 0.0f && !get_result(c).ended);
    transcribe_turn_session_free(c);
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
    test_feed_batch_threads(&k_stream_arch, false);
    test_feed_batch_threads(&k_stream_batch_arch, true);
    test_no_speech(&k_nospeech_arch);
    test_no_speech(&k_nospeech_batch_arch);
    transcribe_log_set(nullptr, nullptr);  // the thrown exceptions log at ERROR
    test_throwing_ops(&k_throw_arch);
    test_throwing_ops(&k_throw_batch_arch);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("turn_dispatch_unit: OK\n");
    return EXIT_SUCCESS;
}
