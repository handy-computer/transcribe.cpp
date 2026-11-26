// sava_real_smoke.cpp - the real SAVA GGUFs through the public TURN and VAD
// APIs on the CPU: upstream's golden clip score (sava-speech.wav, inworld-sava
// 1.0.0, within 1e-4 for F32, F16 and Q8_0); streams in ragged chunkings and
// at 1 and 4 threads bit-identical to a run on jfk.wav and dots.wav (884
// frames, past the 512-frame attention window); the VAD role as the speech
// track; transcribe_turn_stream_feed_batch over 4 users bit-identical to each
// run; no turn in silence or low noise. TRANSCRIBE_SAVA_GGUF / _F16_GGUF /
// _Q8_0_GGUF, else models/sava/sava-{F32,F16,Q8_0}.gguf; RC 77 skip when none
// exists.

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

Track run(transcribe_turn_session * s, const std::vector<float> & pcm, transcribe_turn_result * out = nullptr) {
    Track t;
    CHECK(transcribe_turn_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    if (const float * p = transcribe_turn_probs(s)) {
        t.turn.assign(p, p + r.n_probs);
    }
    if (const float * p = transcribe_turn_speech_probs(s)) {
        t.speech.assign(p, p + r.n_speech_probs);
    }
    if (out != nullptr) {
        *out = r;
    }
    return t;
}

// Appends a feed / flush result, checking it continues the tracks.
void collect(transcribe_turn_session * s, Track & t) {
    transcribe_turn_result r;
    transcribe_turn_result_init(&r);
    CHECK(transcribe_turn_get_result(s, &r) == TRANSCRIBE_OK);
    CHECK(r.first_frame == static_cast<int64_t>(t.turn.size()));
    CHECK(r.speech_first_frame == static_cast<int64_t>(t.speech.size()));
    if (const float * p = transcribe_turn_probs(s)) {
        t.turn.insert(t.turn.end(), p, p + r.n_probs);
    }
    if (const float * p = transcribe_turn_speech_probs(s)) {
        t.speech.insert(t.speech.end(), p, p + r.n_speech_probs);
    }
}

Track stream(transcribe_turn_session * s, const std::vector<float> & pcm, const std::vector<int> & sizes) {
    Track  t;
    size_t k = 0;
    for (size_t off = 0; off < pcm.size(); ++k) {
        const size_t n = std::min(static_cast<size_t>(sizes[k % sizes.size()]), pcm.size() - off);
        CHECK(transcribe_turn_stream_feed(s, pcm.data() + off, static_cast<int>(n)) == TRANSCRIBE_OK);
        collect(s, t);
        off += n;
    }
    CHECK(transcribe_turn_stream_flush(s) == TRANSCRIBE_OK);
    collect(s, t);
    return t;
}

void test_reference(transcribe_model * m) {
    std::vector<float> pcm;
    if (!load_sample("sava-speech.wav", pcm)) {
        return;
    }
    transcribe_turn_session * s = turn_session(m, 1);
    transcribe_turn_result    r;
    const Track               t = run(s, pcm, &r);
    // inworld.sava.run: turn[-1] = 0.9989076852798462 (validate.py holds the
    // per-frame tolerances).
    CHECK(t.turn.size() == 574 && t.speech.size() == 574);
    CHECK(std::fabs(r.p - 0.9989077f) < 1e-4f && r.ended);
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
        CHECK(stream(s1, pcm, sizes) == ref);
    }
    transcribe_turn_session * s4 = turn_session(m, 4);
    CHECK(run(s4, pcm) == ref);
    CHECK(stream(s4, pcm, chunkings.front()) == ref);
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
    transcribe_turn_result    r;
    const Track               t = run(s, pcm, &r);
    CHECK(!r.ended);
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
// lengths, ragged chunks and a rotating first session equals each user's run,
// bit for bit.
void test_batch(transcribe_model * m) {
    constexpr size_t                k_n     = 4;
    const char *                    names[] = { "jfk.wav", "dots.wav", "sava-speech.wav", "noise.wav" };
    const size_t                    limit[] = { 176000, 16000 * 15, 16000 * 12, 16000 * 8 };
    std::vector<std::vector<float>> pcm(k_n);
    std::vector<Track>              want;
    transcribe_turn_session *       s[k_n];
    for (size_t u = 0; u < k_n; ++u) {
        if (!load_sample(names[u], pcm[u])) {
            return;
        }
    }
    for (size_t u = 0; u < k_n; ++u) {
        s[u] = turn_session(m, u == 0 ? 4 : 1);
        pcm[u].resize(std::min(pcm[u].size(), limit[u]));
        want.push_back(run(s[u], pcm[u]));
    }
    std::vector<Track> got(k_n);
    size_t             off[k_n] = {};
    for (int k = 0;; ++k) {
        bool any = false;
        for (size_t i = 0; i < k_n; ++i) {
            any = any || off[i] < pcm[i].size();
        }
        if (!any) {
            break;
        }
        transcribe_turn_session * list[k_n];
        const float *             ptr[k_n];
        int                       len[k_n];
        for (size_t j = 0; j < k_n; ++j) {
            const size_t i = (j + static_cast<size_t>(k / 5)) % k_n;
            const size_t c = std::min(static_cast<size_t>(ragged(static_cast<int>(i), k)), pcm[i].size() - off[i]);
            list[j]        = s[i];
            ptr[j]         = c > 0 ? pcm[i].data() + off[i] : nullptr;
            len[j]         = static_cast<int>(c);
            off[i] += c;
        }
        CHECK(transcribe_turn_stream_feed_batch(list, ptr, len, k_n) == TRANSCRIBE_OK);
        for (size_t i = 0; i < k_n; ++i) {
            collect(s[i], got[i]);
        }
    }
    for (size_t i = 0; i < k_n; ++i) {
        CHECK(transcribe_turn_stream_flush(s[i]) == TRANSCRIBE_OK);
        collect(s[i], got[i]);
        CHECK(got[i] == want[i]);
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
