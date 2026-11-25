// sava_smoke.cpp - the toy sava fixture through the public TURN and VAD APIs:
// info, run, streams in any chunking and thread count bit-identical to the
// run, reset, flush, abort, the VAD role as the speech track, and
// transcribe_turn_stream_feed_batch over 1-8 and 130 sessions bit-identical
// to each run. The toy's 8- and 6-frame attention windows wrap their KV rings
// many times over a clip.

#include "transcribe.h"
#include "transcribe/turn.h"
#include "transcribe/vad.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

constexpr int k_frame = 640;
constexpr int k_lag   = 2;  // the toy's lookahead_frames

std::string fixture(const char * name) {
    return std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/" + name;
}

transcribe_model * load(const char * name) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend           = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * m = nullptr;
    transcribe_model_load_file(fixture(name).c_str(), &mp, &m);
    return m;
}

// Deterministic test signal: bursts of a chirp over low noise.
std::vector<float> signal(int n) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t           seed = 12345;
    for (int i = 0; i < n; ++i) {
        seed                      = seed * 1664525u + 1013904223u;
        const float noise         = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.01f;
        const float t             = static_cast<float>(i) / 16000.0f;
        const bool  burst         = (i / 8000) % 3 != 0;
        v[static_cast<size_t>(i)] = noise + (burst ? 0.3f * std::sin(6.2831853f * (200.0f + 400.0f * t) * t) : 0.0f);
    }
    return v;
}

struct Track {
    std::vector<float> turn;
    std::vector<float> speech;

    bool operator==(const Track & o) const { return turn == o.turn && speech == o.speech; }
};

transcribe_turn_session * turn_session(transcribe_model * m, int n_threads) {
    transcribe_turn_session_params sp;
    transcribe_turn_session_params_init(&sp);
    sp.n_threads                = n_threads;
    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    return s;
}

Track run_track(transcribe_turn_session * s, const std::vector<float> & pcm) {
    Track t;
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.first_frame == 0 && r.speech_first_frame == 0);
    if (const float * p = transcribe_turn_probs(s)) {
        t.turn.assign(p, p + r.n_probs);
    }
    if (const float * p = transcribe_turn_speech_probs(s)) {
        t.speech.assign(p, p + r.n_speech_probs);
    }
    CHECK(!t.turn.empty() && r.p == t.turn.back());
    return t;
}

// Appends a feed / flush result, checking it continues the tracks.
void collect(transcribe_turn_session * s, Track & t) {
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.first_frame == static_cast<int64_t>(t.turn.size()));
    CHECK(r.speech_first_frame == static_cast<int64_t>(t.speech.size()));
    CHECK(r.p == 0.0f && !r.ended);
    if (const float * p = transcribe_turn_probs(s)) {
        t.turn.insert(t.turn.end(), p, p + r.n_probs);
    }
    if (const float * p = transcribe_turn_speech_probs(s)) {
        t.speech.insert(t.speech.end(), p, p + r.n_speech_probs);
    }
}

// Feeds pcm in chunks of sizes[i % n] and flushes.
Track stream_track(transcribe_turn_session * s, const std::vector<float> & pcm, const std::vector<int> & sizes) {
    Track  t;
    size_t k = 0;
    for (size_t off = 0; off < pcm.size(); ++k) {
        const size_t n = std::min(static_cast<size_t>(sizes[k % sizes.size()]), pcm.size() - off);
        CHECK(transcribe_turn_stream_feed(s, pcm.data() + off, static_cast<int>(n)) == TRANSCRIBE_OK);
        collect(s, t);
        // The speech track trails the turn track by the lookahead.
        CHECK(static_cast<int64_t>(t.speech.size()) <=
              std::max<int64_t>(0, static_cast<int64_t>(t.turn.size()) - k_lag));
        off += n;
    }
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    collect(s, t);
    return t;
}

std::vector<float> vad_run(transcribe_vad_session * s, const std::vector<float> & pcm) {
    CHECK(transcribe_vad_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    const float * p = transcribe_vad_probs(s);
    return p != nullptr ? std::vector<float>(p, p + r.n_probs) : std::vector<float>();
}

std::vector<float> vad_stream(transcribe_vad_session * s, const std::vector<float> & pcm, int chunk) {
    std::vector<float>    out;
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    auto append = [&] {
        CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(out.size()));
        if (const float * p = transcribe_vad_probs(s)) {
            out.insert(out.end(), p, p + r.n_probs);
        }
    };
    for (size_t off = 0; off < pcm.size(); off += static_cast<size_t>(chunk)) {
        const int n = static_cast<int>(std::min(static_cast<size_t>(chunk), pcm.size() - off));
        CHECK(transcribe_vad_stream_feed(s, pcm.data() + off, n) == TRANSCRIBE_OK);
        append();
    }
    CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
    append();
    return out;
}

// A ragged chunk size for user u at call k: 0 (sometimes) to a few frames.
int ragged(int u, int k) {
    const uint32_t h        = static_cast<uint32_t>(u * 7919 + k * 104729) * 2654435761u;
    const int      sizes[6] = { 0, 13, 640, 777, 1280 + 3 * u, 2000 };
    return sizes[static_cast<int>(h >> 20) % 6];
}

// Feeds every session its pcm through transcribe_turn_stream_feed_batch, then
// flushes each. ragged_order: ragged chunks with a rotating first session (it
// owns the batch graph); else one frame per call in a fixed order.
std::vector<Track> batch_tracks(const std::vector<transcribe_turn_session *> & s,
                                const std::vector<std::vector<float>> &        pcm,
                                bool                                           ragged_order) {
    const size_t        n = s.size();
    std::vector<Track>  out(n);
    std::vector<size_t> off(n, 0);
    for (int k = 0;; ++k) {
        bool any = false;
        for (size_t i = 0; i < n; ++i) {
            any = any || off[i] < pcm[i].size();
        }
        if (!any) {
            break;
        }
        std::vector<transcribe_turn_session *> list(n);
        std::vector<const float *>             ptr(n);
        std::vector<int>                       len(n);
        for (size_t j = 0; j < n; ++j) {
            const size_t i     = ragged_order ? (j + static_cast<size_t>(k / 3)) % n : j;
            const int    chunk = ragged_order ? ragged(static_cast<int>(i), k) : k_frame;
            const size_t c     = std::min(static_cast<size_t>(chunk), pcm[i].size() - off[i]);
            list[j]            = s[i];
            ptr[j]             = c > 0 ? pcm[i].data() + off[i] : nullptr;
            len[j]             = static_cast<int>(c);
            off[i] += c;
        }
        CHECK(transcribe_turn_stream_feed_batch(list.data(), ptr.data(), len.data(), static_cast<int>(n)) ==
              TRANSCRIBE_OK);
        for (size_t i = 0; i < n; ++i) {
            collect(s[i], out[i]);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        CHECK(transcribe_turn_stream_flush(s[i]) == TRANSCRIBE_OK);
        collect(s[i], out[i]);
    }
    return out;
}

// feed_batch over 1..8 sessions with ragged chunks equals each session's run,
// bit for bit, also on reused sessions.
void test_batch(transcribe_model * m) {
    std::vector<std::vector<float>> pcm;
    std::vector<Track>              alone;
    transcribe_turn_session *       ref = turn_session(m, 1);
    for (int u = 0; u < 8; ++u) {
        // Different audio and lengths per user (40 to 110 frames).
        std::vector<float> x = signal((40 + 10 * u) * k_frame + 37 * u);
        for (float & v : x) {
            v *= 0.5f + 0.1f * static_cast<float>(u);
        }
        pcm.push_back(x);
        alone.push_back(run_track(ref, x));
    }
    transcribe_turn_session_free(ref);

    for (const int n_users : { 1, 2, 3, 5, 8 }) {
        for (const int threads : { 1, 4 }) {
            if (threads == 4 && n_users != 5) {
                continue;
            }
            std::vector<transcribe_turn_session *> s;
            for (int u = 0; u < n_users; ++u) {
                s.push_back(turn_session(m, threads));
            }
            const std::vector<std::vector<float>> in(pcm.begin(), pcm.begin() + n_users);
            const std::vector<Track>              want(alone.begin(), alone.begin() + n_users);
            for (int pass = 0; pass < 2; ++pass) {
                CHECK(batch_tracks(s, in, true) == want);
            }
            for (transcribe_turn_session * x : s) {
                transcribe_turn_session_free(x);
            }
        }
    }
}

// feed_batch over more sessions than one batch graph holds (the arch steps
// them in chunks of 64): 130 sessions with different audio and lengths, every
// one bit-identical to its run, in a fixed order (the cached chunk graphs are
// reused every call) and then in ragged order.
void test_batch_large(transcribe_model * m) {
    constexpr int                   n_users = 130;
    std::vector<std::vector<float>> pcm;
    std::vector<Track>              alone;
    transcribe_turn_session *       ref  = turn_session(m, 1);
    const std::vector<float>        base = signal(16 * k_frame + 200);
    for (int u = 0; u < n_users; ++u) {
        // 6 to 15 frames and a partial one, a per-user offset and gain.
        const size_t       len = static_cast<size_t>((6 + u % 10) * k_frame + 11 * (u % 7));
        const size_t       off = static_cast<size_t>(u % 13) * 17;
        std::vector<float> x(base.begin() + static_cast<std::ptrdiff_t>(off),
                             base.begin() + static_cast<std::ptrdiff_t>(off + len));
        for (float & v : x) {
            v *= 0.4f + 0.005f * static_cast<float>(u);
        }
        pcm.push_back(x);
        alone.push_back(run_track(ref, x));
    }
    transcribe_turn_session_free(ref);

    std::vector<transcribe_turn_session *> s;
    for (int u = 0; u < n_users; ++u) {
        s.push_back(turn_session(m, 1));
    }
    CHECK(batch_tracks(s, pcm, false) == alone);
    CHECK(batch_tracks(s, pcm, true) == alone);
    for (transcribe_turn_session * x : s) {
        transcribe_turn_session_free(x);
    }
}

bool abort_now(void *) {
    return true;
}

void test_model() {
    transcribe_model * m = load("arch_sava.gguf");
    CHECK(m != nullptr);
    if (m == nullptr) {
        return;
    }
    CHECK(transcribe_model_roles(m) == (TRANSCRIBE_ROLE_VAD | TRANSCRIBE_ROLE_TURN));

    transcribe_turn_info ti;
    transcribe_turn_info_init(&ti);
    CHECK(transcribe_turn_get_info(m, &ti) == TRANSCRIBE_OK);
    CHECK(ti.sample_rate == 16000 && ti.frame_samples == k_frame && ti.max_audio_ms == 0);
    CHECK(ti.has_speech && ti.threshold == 0.9);
    transcribe_vad_info vi;
    transcribe_vad_info_init(&vi);
    CHECK(transcribe_vad_get_info(m, &vi) == TRANSCRIBE_OK);
    CHECK(vi.frame_samples == k_frame);

    // 200 frames and a partial one: crosses the 64-frame graph blocks and
    // wraps every KV ring.
    const std::vector<float> pcm      = signal(200 * k_frame + 123);
    const size_t             n_frames = (pcm.size() + k_frame - 1) / k_frame;

    transcribe_turn_session * s   = turn_session(m, 1);
    const Track               run = run_track(s, pcm);
    CHECK(run.turn.size() == n_frames && run.speech.size() == n_frames);
    for (size_t i = 0; i < n_frames; ++i) {
        CHECK(std::isfinite(run.turn[i]) && run.turn[i] >= 0.0f && run.turn[i] <= 1.0f);
        CHECK(std::isfinite(run.speech[i]) && run.speech[i] >= 0.0f && run.speech[i] <= 1.0f);
    }

    // ended uses the model's threshold, or an override.
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.ended == (r.p >= 0.9f));
    transcribe_turn_params tp;
    transcribe_turn_params_init(&tp);
    tp.threshold = 0.0;
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), &tp) == TRANSCRIBE_OK);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK && r.ended);

    // Streaming in any chunking equals the run, bit for bit: single samples,
    // ragged sizes, whole frames, more than a graph block at once, the clip.
    const std::vector<std::vector<int>> chunkings = {
        { 1 },
        { 13, 1280, 777 },
        { k_frame },
        { k_frame + 1 },
        { 3 * k_frame },
        { 70 * k_frame + 3 },
        { static_cast<int>(pcm.size()) },
    };
    for (const std::vector<int> & sizes : chunkings) {
        CHECK(stream_track(s, pcm, sizes) == run);
    }

    // Thread count does not change a value.
    transcribe_turn_session * s4 = turn_session(m, 4);
    CHECK(run_track(s4, pcm) == run);
    CHECK(stream_track(s4, pcm, { 777 }) == run);
    transcribe_turn_session_free(s4);

    // Reset mid-stream: the next stream starts from scratch.
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 5 * k_frame + 7) == TRANSCRIBE_OK);
    transcribe_turn_stream_reset(s);
    CHECK(stream_track(s, pcm, { 1000 }) == run);

    // Flush with nothing fed steps nothing.
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.n_probs == 0 && r.n_speech_probs == 0);

    // Abort; the session runs again afterwards.
    transcribe_turn_set_abort_callback(s, abort_now, nullptr);
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_ERR_ABORTED);
    transcribe_turn_set_abort_callback(s, nullptr, nullptr);
    CHECK(run_track(s, pcm) == run);
    transcribe_turn_session_free(s);

    // The VAD role is the TURN session's speech track, run or streamed.
    transcribe_vad_session_params vp;
    transcribe_vad_session_params_init(&vp);
    vp.n_threads               = 1;
    transcribe_vad_session * v = nullptr;
    CHECK(transcribe_vad_session_init(m, &vp, &v) == TRANSCRIBE_OK);
    CHECK(vad_run(v, pcm) == run.speech);
    for (const int c : { 1, 641, 64 * k_frame + 5 }) {
        CHECK(vad_stream(v, pcm, c) == run.speech);
    }
    transcribe_vad_session_free(v);

    test_batch(m);
    test_batch_large(m);

    transcribe_model_free(m);
}

}  // namespace

int main() {
    test_model();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("sava_smoke: ok\n");
    return EXIT_SUCCESS;
}
