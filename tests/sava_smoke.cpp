// sava_smoke.cpp - the toy sava fixture through the public TURN and VAD APIs:
// load, info, a run on synthetic audio, streaming parity with the run (bit
// for bit) across ragged chunkings and thread counts, the VAD speech track,
// flush and reset; transcribe_turn_stream_feed_batch over 1-8 sessions in
// ragged chunks equals each session fed alone, bit for bit. The toy's 8- and
// 6-frame attention windows and 72-frame KV rings wrap many times over the
// clip.

#include "transcribe.h"
#include "transcribe/turn.h"
#include "transcribe/vad.h"

#include <cmath>
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
    const float * p  = transcribe_turn_probs(s);
    const float * sp = transcribe_turn_speech_probs(s);
    if (p != nullptr) {
        t.turn.assign(p, p + r.n_probs);
    }
    if (sp != nullptr) {
        t.speech.assign(sp, sp + r.n_speech_probs);
    }
    CHECK(!t.turn.empty() && r.p == t.turn.back());
    return t;
}

// Feed pcm in chunks of sizes[i % n], flush, and collect both tracks,
// checking the frame indices of every call.
Track stream_track(transcribe_turn_session * s, const std::vector<float> & pcm, const std::vector<int> & sizes) {
    Track                  t;
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    auto collect = [&]() {
        CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(t.turn.size()));
        CHECK(r.speech_first_frame == static_cast<int64_t>(t.speech.size()));
        CHECK(r.p == 0.0f && !r.ended);
        const float * p  = transcribe_turn_probs(s);
        const float * sp = transcribe_turn_speech_probs(s);
        if (p != nullptr) {
            t.turn.insert(t.turn.end(), p, p + r.n_probs);
        }
        if (sp != nullptr) {
            t.speech.insert(t.speech.end(), sp, sp + r.n_speech_probs);
        }
    };
    size_t k = 0;
    for (size_t off = 0; off < pcm.size(); ++k) {
        const size_t n = std::min(static_cast<size_t>(sizes[k % sizes.size()]), pcm.size() - off);
        CHECK(transcribe_turn_stream_feed(s, pcm.data() + off, static_cast<int>(n)) == TRANSCRIBE_OK);
        collect();
        // The speech track trails the turn track by the lookahead.
        CHECK(static_cast<int64_t>(t.speech.size()) <=
              std::max<int64_t>(0, static_cast<int64_t>(t.turn.size()) - k_lag));
        off += n;
    }
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    collect();
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
    for (size_t off = 0; off < pcm.size(); off += static_cast<size_t>(chunk)) {
        const int n = static_cast<int>(std::min(static_cast<size_t>(chunk), pcm.size() - off));
        CHECK(transcribe_vad_stream_feed(s, pcm.data() + off, n) == TRANSCRIBE_OK);
        CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(out.size()));
        const float * p = transcribe_vad_probs(s);
        if (p != nullptr) {
            out.insert(out.end(), p, p + r.n_probs);
        }
    }
    CHECK(transcribe_vad_stream_flush(s) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    const float * p = transcribe_vad_probs(s);
    if (p != nullptr) {
        out.insert(out.end(), p, p + r.n_probs);
    }
    return out;
}

// A ragged chunk size for user u at call k: 0 (sometimes) to a few frames.
int ragged(int u, int k) {
    const uint32_t h        = static_cast<uint32_t>(u * 7919 + k * 104729) * 2654435761u;
    const int      r        = static_cast<int>(h >> 20) % 6;
    const int      sizes[6] = { 0, 13, 640, 777, 1280 + 3 * u, 2000 };
    return sizes[r];
}

// Feed every user's pcm through transcribe_turn_stream_feed_batch in ragged
// chunks (rotating which session is listed first), then flush each, and
// collect each session's tracks with their frame indices checked.
std::vector<Track> batch_tracks(const std::vector<transcribe_turn_session *> & s,
                                const std::vector<std::vector<float>> &        pcm) {
    const size_t        n = s.size();
    std::vector<Track>  out(n);
    std::vector<size_t> off(n, 0);
    auto                collect = [&](size_t i) {
        transcribe_turn_result r;
        transcribe_turn_result_init(&r);
        CHECK(transcribe_turn_get_result(s[i], &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(out[i].turn.size()));
        CHECK(r.speech_first_frame == static_cast<int64_t>(out[i].speech.size()));
        if (const float * p = transcribe_turn_probs(s[i])) {
            out[i].turn.insert(out[i].turn.end(), p, p + r.n_probs);
        }
        if (const float * p = transcribe_turn_speech_probs(s[i])) {
            out[i].speech.insert(out[i].speech.end(), p, p + r.n_speech_probs);
        }
    };
    for (int k = 0;; ++k) {
        bool any = false;
        for (size_t i = 0; i < n; ++i) {
            any = any || off[i] < pcm[i].size();
        }
        if (!any) {
            break;
        }
        // Rotate the order so the session that owns the batch graph changes.
        std::vector<transcribe_turn_session *> list(n);
        std::vector<const float *>             ptr(n);
        std::vector<int>                       len(n);
        std::vector<size_t>                    who(n);
        for (size_t j = 0; j < n; ++j) {
            const size_t i = (j + static_cast<size_t>(k / 3)) % n;
            const size_t c = std::min(static_cast<size_t>(ragged(static_cast<int>(i), k)), pcm[i].size() - off[i]);
            who[j]         = i;
            list[j]        = s[i];
            ptr[j]         = c > 0 ? pcm[i].data() + off[i] : nullptr;
            len[j]         = static_cast<int>(c);
            off[i] += c;
        }
        CHECK(transcribe_turn_stream_feed_batch(list.data(), ptr.data(), len.data(), static_cast<int>(n)) ==
              TRANSCRIBE_OK);
        for (size_t j = 0; j < n; ++j) {
            collect(who[j]);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        CHECK(transcribe_turn_stream_flush(s[i]) == TRANSCRIBE_OK);
        collect(i);
    }
    return out;
}

// feed_batch over 1..8 sessions with ragged chunks equals each session fed
// alone (and, through stream == run, its run), bit for bit.
void test_batch(transcribe_model * m) {
    const int                       max_users = 8;
    std::vector<std::vector<float>> pcm;
    std::vector<Track>              alone;
    transcribe_turn_session *       ref = turn_session(m, 1);
    for (int u = 0; u < max_users; ++u) {
        // Different audio and lengths per user (40 to 110 frames).
        std::vector<float> x = signal((40 + 10 * u) * k_frame + 37 * u);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] *= 0.5f + 0.1f * static_cast<float>(u);
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
            for (int pass = 0; pass < 2; ++pass) {  // the second pass reuses the sessions
                const std::vector<Track> got = batch_tracks(s, in);
                for (int u = 0; u < n_users; ++u) {
                    CHECK(got[static_cast<size_t>(u)].turn == alone[static_cast<size_t>(u)].turn);
                    CHECK(got[static_cast<size_t>(u)].speech == alone[static_cast<size_t>(u)].speech);
                }
            }
            for (transcribe_turn_session * x : s) {
                transcribe_turn_session_free(x);
            }
        }
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

    // p is the last frame's turn score; a threshold override sets ended.
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    {
        CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.p == run.turn.back() && r.ended == (r.p >= 0.9f));
        transcribe_turn_params tp;
        transcribe_turn_params_init(&tp);
        tp.threshold = 0.0;
        CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), &tp) == TRANSCRIBE_OK);
        CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK && r.ended);
        tp.threshold = 1.5;  // out of range: rejected before the result is touched
        CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), &tp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK && r.ended);
    }

    // A second run from a fresh state gives the same result.
    const Track again = run_track(s, pcm);
    CHECK(again.turn == run.turn && again.speech == run.speech);

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
        const Track st = stream_track(s, pcm, sizes);
        CHECK(st.turn == run.turn);
        CHECK(st.speech == run.speech);
    }

    // Thread count does not change a value.
    transcribe_turn_session * s4 = turn_session(m, 4);
    const Track               r4 = run_track(s4, pcm);
    CHECK(r4.turn == run.turn && r4.speech == run.speech);
    const Track st4 = stream_track(s4, pcm, { 777 });
    CHECK(st4.turn == run.turn && st4.speech == run.speech);
    transcribe_turn_session_free(s4);

    // Reset mid-stream: the next stream starts from scratch.
    CHECK(transcribe_turn_stream_feed(s, pcm.data(), 5 * k_frame + 7) == TRANSCRIBE_OK);
    transcribe_turn_stream_reset(s);
    const Track after_reset = stream_track(s, pcm, { 1000 });
    CHECK(after_reset.turn == run.turn && after_reset.speech == run.speech);

    // Flush with nothing fed steps nothing.
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.n_probs == 0 && r.n_speech_probs == 0);

    // Abort.
    transcribe_turn_set_abort_callback(s, abort_now, nullptr);
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_ERR_ABORTED);
    transcribe_turn_set_abort_callback(s, nullptr, nullptr);
    const Track after_abort = run_track(s, pcm);
    CHECK(after_abort.turn == run.turn && after_abort.speech == run.speech);
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
