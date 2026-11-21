// sava_real_smoke.cpp - the real SAVA GGUF through the public TURN and VAD
// APIs on the CPU:
//   - sava-speech.wav (upstream's golden clip): the clip score and turn
//     segments of inworld.sava.run / sava.segment (inworld-sava 1.0.0);
//   - streams in ragged chunkings and at 1 and 4 threads are bit-identical to
//     a run, on jfk.wav and on dots.wav (884 frames, past the 512-frame
//     attention window);
//   - the VAD role returns the TURN session's speech track;
//   - transcribe_turn_stream_feed_batch over 4 users in ragged chunks equals
//     each user's run, bit for bit;
//   - digital silence and low noise hold no turn.
// Models: TRANSCRIBE_SAVA_GGUF, TRANSCRIBE_SAVA_F16_GGUF and
// TRANSCRIBE_SAVA_Q8_0_GGUF (else models/sava/sava-{F32,F16,Q8_0}.gguf), each
// tested when it exists; RC 77 skip when none does. The golden-clip checks
// hold for F16 and Q8_0 too (p within 1e-4, the same two turns).

#include "transcribe.h"
#include "transcribe/turn.h"
#include "transcribe/vad.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

bool load_sample(const char * name, std::vector<float> & pcm) {
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name, pcm, err)) {
        std::fprintf(stderr, "FAIL: %s: %s\n", name, err.c_str());
        ++g_failures;
        return false;
    }
    return true;
}

struct Track {
    std::vector<float>   turn;
    std::vector<float>   speech;
    std::vector<int64_t> segments;  // start, end, reason triples
    float                p     = 0.0f;
    bool                 ended = false;
};

transcribe_turn_session * turn_session(transcribe_model * m, int n_threads) {
    transcribe_turn_session_params sp;
    transcribe_turn_session_params_init(&sp);
    sp.n_threads                = n_threads;
    transcribe_turn_session * s = nullptr;
    CHECK(transcribe_turn_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    return s;
}

Track run(transcribe_turn_session * s, const std::vector<float> & pcm) {
    Track t;
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    t.p     = r.p;
    t.ended = r.ended;
    if (const float * p = transcribe_turn_probs(s)) {
        t.turn.assign(p, p + r.n_probs);
    }
    if (const float * p = transcribe_turn_speech_probs(s)) {
        t.speech.assign(p, p + r.n_speech_probs);
    }
    for (int i = 0; i < r.n_segments; ++i) {
        transcribe_turn_segment seg;
        transcribe_turn_segment_init(&seg);
        CHECK(transcribe_turn_get_segment(s, i, &seg) == TRANSCRIBE_OK);
        t.segments.insert(t.segments.end(), { seg.start_sample, seg.end_sample, seg.end_reason });
    }
    return t;
}

Track stream(transcribe_turn_session * s, const std::vector<float> & pcm, const std::vector<int> & sizes) {
    Track                  t;
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    auto collect = [&]() {
        CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(t.turn.size()));
        CHECK(r.speech_first_frame == static_cast<int64_t>(t.speech.size()));
        if (const float * p = transcribe_turn_probs(s)) {
            t.turn.insert(t.turn.end(), p, p + r.n_probs);
        }
        if (const float * p = transcribe_turn_speech_probs(s)) {
            t.speech.insert(t.speech.end(), p, p + r.n_speech_probs);
        }
    };
    size_t k = 0;
    for (size_t off = 0; off < pcm.size(); ++k) {
        const size_t n = std::min(static_cast<size_t>(sizes[k % sizes.size()]), pcm.size() - off);
        CHECK(transcribe_turn_stream_feed(s, pcm.data() + off, static_cast<int>(n)) == TRANSCRIBE_OK);
        collect();
        off += n;
    }
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    collect();
    return t;
}

void test_reference(transcribe_model * m) {
    std::vector<float> pcm;
    if (!load_sample("sava-speech.wav", pcm)) {
        return;
    }
    transcribe_turn_session * s = turn_session(m, 1);
    const Track               t = run(s, pcm);
    // inworld.sava.run: turn[-1] = 0.9989076852798462 (fp32 reference;
    // validate.py holds the per-frame tolerances).
    CHECK(t.turn.size() == 574 && t.speech.size() == 574);
    CHECK(std::fabs(t.p - 0.9989077f) < 1e-4f && t.ended);
    // sava.segment(curves, Policy(), 3): frames 134..319 (turn) and
    // 308..393 (silence), in samples.
    const std::vector<int64_t> want = { 85760,  204160, TRANSCRIBE_TURN_END_MODEL,
                                        197120, 251520, TRANSCRIBE_TURN_END_SILENCE };
    CHECK(t.segments == want);
    transcribe_turn_session_free(s);
}

void test_parity(transcribe_model * m, const char * name, const std::vector<std::vector<int>> & chunkings) {
    std::vector<float> pcm;
    if (!load_sample(name, pcm)) {
        return;
    }
    std::printf("parity: %s\n", name);
    transcribe_turn_session * s1  = turn_session(m, 1);
    const Track               ref = run(s1, pcm);
    for (const std::vector<int> & sizes : chunkings) {
        const Track t = stream(s1, pcm, sizes);
        CHECK(t.turn == ref.turn);
        CHECK(t.speech == ref.speech);
    }
    transcribe_turn_session * s4 = turn_session(m, 4);
    const Track               r4 = run(s4, pcm);
    CHECK(r4.turn == ref.turn && r4.speech == ref.speech && r4.segments == ref.segments);
    const Track t4 = stream(s4, pcm, chunkings.front());
    CHECK(t4.turn == ref.turn && t4.speech == ref.speech);
    transcribe_turn_session_free(s4);
    transcribe_turn_session_free(s1);

    // The VAD role is the same speech track.
    transcribe_vad_session_params vp;
    transcribe_vad_session_params_init(&vp);
    vp.n_threads               = 1;
    transcribe_vad_session * v = nullptr;
    CHECK(transcribe_vad_session_init(m, &vp, &v) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_run(v, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result vr;
    transcribe_vad_result_init(&vr);
    CHECK(transcribe_vad_get_result(v, &vr) == TRANSCRIBE_OK);
    const float * vp_probs = transcribe_vad_probs(v);
    CHECK(vp_probs != nullptr && std::vector<float>(vp_probs, vp_probs + vr.n_probs) == ref.speech);
    transcribe_vad_session_free(v);
}

void test_quiet(transcribe_model * m, const char * name) {
    std::vector<float> pcm;
    if (!load_sample(name, pcm)) {
        return;
    }
    transcribe_turn_session * s = turn_session(m, 1);
    const Track               t = run(s, pcm);
    CHECK(t.segments.empty() && !t.ended);
    for (const float p : t.speech) {
        CHECK(p < 0.5f);
    }
    transcribe_turn_session_free(s);
}

// A ragged chunk size for user u at call k: 0 (sometimes) to 2000 samples.
int ragged(int u, int k) {
    const uint32_t h        = static_cast<uint32_t>(u * 7919 + k * 104729) * 2654435761u;
    const int      sizes[6] = { 0, 13, 640, 777, 1280 + 3 * u, 2000 };
    return sizes[static_cast<int>(h >> 20) % 6];
}

// transcribe_turn_stream_feed_batch over 4 users with different audio and
// lengths, ragged chunks and a rotating first session, equals each user's
// run, bit for bit (stream == run holds per session).
void test_batch(transcribe_model * m) {
    const char *                    names[] = { "jfk.wav", "dots.wav", "sava-speech.wav", "noise.wav" };
    const size_t                    limit[] = { 176000, 16000 * 15, 16000 * 12, 16000 * 8 };
    std::vector<std::vector<float>> pcm;
    std::vector<Track>              want;
    transcribe_turn_session *       ref = turn_session(m, 1);
    for (size_t u = 0; u < 4; ++u) {
        std::vector<float> x;
        if (!load_sample(names[u], x)) {
            transcribe_turn_session_free(ref);
            return;
        }
        x.resize(std::min(x.size(), limit[u]));
        pcm.push_back(x);
        want.push_back(run(ref, x));
    }
    transcribe_turn_session_free(ref);

    std::vector<transcribe_turn_session *> s;
    for (size_t u = 0; u < 4; ++u) {
        s.push_back(turn_session(m, u == 0 ? 4 : 1));
    }
    std::vector<Track>  got(4);
    std::vector<size_t> off(4, 0);
    auto                collect = [&](size_t i) {
        transcribe_turn_result r;
        transcribe_turn_result_init(&r);
        CHECK(transcribe_turn_get_result(s[i], &r) == TRANSCRIBE_OK);
        CHECK(r.first_frame == static_cast<int64_t>(got[i].turn.size()));
        if (const float * p = transcribe_turn_probs(s[i])) {
            got[i].turn.insert(got[i].turn.end(), p, p + r.n_probs);
        }
        if (const float * p = transcribe_turn_speech_probs(s[i])) {
            got[i].speech.insert(got[i].speech.end(), p, p + r.n_speech_probs);
        }
    };
    for (int k = 0;; ++k) {
        bool any = false;
        for (size_t i = 0; i < 4; ++i) {
            any = any || off[i] < pcm[i].size();
        }
        if (!any) {
            break;
        }
        transcribe_turn_session * list[4];
        const float *             ptr[4];
        int                       len[4];
        size_t                    who[4];
        for (size_t j = 0; j < 4; ++j) {
            const size_t i = (j + static_cast<size_t>(k / 5)) % 4;
            const size_t c = std::min(static_cast<size_t>(ragged(static_cast<int>(i), k)), pcm[i].size() - off[i]);
            who[j]         = i;
            list[j]        = s[i];
            ptr[j]         = c > 0 ? pcm[i].data() + off[i] : nullptr;
            len[j]         = static_cast<int>(c);
            off[i] += c;
        }
        CHECK(transcribe_turn_stream_feed_batch(list, ptr, len, 4) == TRANSCRIBE_OK);
        for (size_t j = 0; j < 4; ++j) {
            collect(who[j]);
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        CHECK(transcribe_turn_stream_flush(s[i]) == TRANSCRIBE_OK);
        collect(i);
        CHECK(got[i].turn == want[i].turn);
        CHECK(got[i].speech == want[i].speech);
        transcribe_turn_session_free(s[i]);
    }
}

int test_file(const std::string & path) {
    std::printf("model: %s\n", path.c_str());
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    transcribe_model * m = nullptr;
    if (transcribe_model_load_file(path.c_str(), &mp, &m) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", path.c_str());
        ++g_failures;
        return 1;
    }
    CHECK(transcribe_model_roles(m) == (TRANSCRIBE_ROLE_VAD | TRANSCRIBE_ROLE_TURN));

    test_reference(m);
    test_parity(m, "jfk.wav",
                {
                    { 13, 1280, 777 },
                    { 640 },
                    { 1 },
                    { 641 },
                    { 64 * 640 + 3 }
    });
    test_parity(m, "dots.wav", { { 640 }, { 4999 } });
    test_quiet(m, "silence.wav");
    test_quiet(m, "noise.wav");
    test_batch(m);

    transcribe_model_free(m);
    return 1;
}

}  // namespace

int main() {
    struct Candidate {
        const char * env;
        const char * fallback;
    };

    const Candidate candidates[] = {
        { "TRANSCRIBE_SAVA_GGUF",      TRANSCRIBE_SAVA_DEFAULT_GGUF      },
        { "TRANSCRIBE_SAVA_F16_GGUF",  TRANSCRIBE_SAVA_DEFAULT_F16_GGUF  },
        { "TRANSCRIBE_SAVA_Q8_0_GGUF", TRANSCRIBE_SAVA_DEFAULT_Q8_0_GGUF },
    };
    int tested = 0;
    for (const Candidate & c : candidates) {
        const char *      env  = std::getenv(c.env);
        const std::string path = env != nullptr && env[0] != '\0' ? env : c.fallback;
        if (std::ifstream(path).good()) {
            tested += test_file(path);
        }
    }
    if (tested == 0) {
        std::fprintf(stderr, "SKIP: no SAVA GGUF found (set TRANSCRIBE_SAVA_GGUF / _F16_GGUF / _Q8_0_GGUF)\n");
        return 77;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("sava_real_smoke: ok\n");
    return EXIT_SUCCESS;
}
