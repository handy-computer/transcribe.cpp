// stream_offline_interleave_smoke.cpp - real-model gated test: interleaving
// streaming and offline runs on ONE session.
//
// The dispatcher releases per-run compute scratch (the scheduler) after every
// offline transcribe_run / transcribe_run_batch. Streaming sessions re-create
// the scheduler lazily, so a stream that follows an offline run, and an
// offline run that follows a stream, must behave exactly like the same call
// on a fresh session.
//
// Sequence on one session (per model):
//   stream -> run -> stream -> run -> stream(reset mid-way) -> run -> stream
// (transcribe_run_batch is exercised by the dispatcher unit test and the
// gigaam workspace smoke, so it is not part of the sequence.)
// Every stream result must equal the fresh-session stream result and every
// offline result must equal the fresh-session offline result.
//
// Runs on every streaming family whose model env var is set:
//   TRANSCRIBE_PARAKEET_UNIFIED_GGUF, TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF,
//   TRANSCRIBE_VOXTRAL_REALTIME_GGUF. Exit 77 when none is set. Any set model
//   that fails to open is a failure.

#include "transcribe.h"
#include "wav.h"

#include <algorithm>
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

constexpr int k_chunk_samples = 16000 / 10;  // 100 ms feeds

std::string text_of(transcribe_session * s) {
    const char * t = transcribe_full_text(s);
    return t == nullptr ? "" : t;
}

bool run_offline(transcribe_session * s, const std::vector<float> & pcm, std::string & out) {
    if (transcribe_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) != TRANSCRIBE_OK) {
        return false;
    }
    out = text_of(s);
    return true;
}

// Stream the clip in 100 ms feeds. If reset_at_feed >= 0, call
// transcribe_stream_reset after that many feeds and start over, so the
// result still covers the whole clip.
bool run_stream(transcribe_session * s, const std::vector<float> & pcm, int reset_at_feed, std::string & out) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (transcribe_stream_begin(s, nullptr, nullptr) != TRANSCRIBE_OK) {
            return false;
        }
        size_t pos   = 0;
        int    feeds = 0;
        bool   reset = false;
        while (pos < pcm.size()) {
            const size_t             take = std::min<size_t>(k_chunk_samples, pcm.size() - pos);
            transcribe_stream_update upd;
            transcribe_stream_update_init(&upd);
            if (transcribe_stream_feed(s, pcm.data() + pos, static_cast<int>(take), &upd) != TRANSCRIBE_OK) {
                return false;
            }
            pos += take;
            ++feeds;
            if (attempt == 0 && reset_at_feed >= 0 && feeds == reset_at_feed) {
                transcribe_stream_reset(s);
                reset = true;
                break;
            }
        }
        if (reset) {
            continue;  // second attempt streams the whole clip after the reset
        }
        transcribe_stream_update fin;
        transcribe_stream_update_init(&fin);
        if (transcribe_stream_finalize(s, &fin) != TRANSCRIBE_OK) {
            return false;
        }
        out = text_of(s);
        return true;
    }
    return false;
}

int test_model(const char * label, const char * path, const std::vector<float> & pcm) {
    std::printf("== %s: %s\n", label, path);
    const int            failures_before = g_failures;  // per-model verdict, not cumulative
    transcribe_session * fresh           = nullptr;
    if (transcribe_open(path, nullptr, nullptr, &fresh) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL %s: transcribe_open failed\n", label);
        return 1;
    }
    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    transcribe_model_get_capabilities(transcribe_get_model(fresh), &caps);
    if (!caps.supports_streaming) {
        std::fprintf(stderr, "FAIL %s: model does not advertise streaming\n", label);
        transcribe_close(fresh);
        return 1;
    }

    // Fresh-session baselines, each from its own session so neither path has
    // seen the other.
    std::string ref_stream;
    std::string ref_offline;
    CHECK(run_stream(fresh, pcm, -1, ref_stream));
    transcribe_close(fresh);
    transcribe_session * fresh2 = nullptr;
    CHECK(transcribe_open(path, nullptr, nullptr, &fresh2) == TRANSCRIBE_OK);
    CHECK(run_offline(fresh2, pcm, ref_offline));
    transcribe_close(fresh2);
    std::printf("   ref stream : '%s'\n   ref offline: '%s'\n", ref_stream.c_str(), ref_offline.c_str());
    CHECK(!ref_stream.empty());
    CHECK(!ref_offline.empty());

    // Interleave on one session.
    transcribe_session * s = nullptr;
    CHECK(transcribe_open(path, nullptr, nullptr, &s) == TRANSCRIBE_OK);
    std::string t;
    CHECK(run_stream(s, pcm, -1, t) && t == ref_stream);
    CHECK(run_offline(s, pcm, t) && t == ref_offline);
    CHECK(run_stream(s, pcm, -1, t) && t == ref_stream);
    CHECK(run_offline(s, pcm, t) && t == ref_offline);
    CHECK(run_stream(s, pcm, /*reset_at_feed=*/5, t) && t == ref_stream);
    CHECK(run_offline(s, pcm, t) && t == ref_offline);
    CHECK(run_stream(s, pcm, -1, t) && t == ref_stream);
    transcribe_close(s);
    std::printf("   interleave: %s\n", g_failures == failures_before ? "ok" : "FAILED");
    return 0;
}

}  // namespace

int main() {
    struct Entry {
        const char * label;
        const char * env;
    };

    const Entry entries[] = {
        { "parakeet-unified",    "TRANSCRIBE_PARAKEET_UNIFIED_GGUF"         },
        { "moonshine-streaming", "TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF" },
        { "voxtral-realtime",    "TRANSCRIBE_VOXTRAL_REALTIME_GGUF"         },
    };

    std::vector<float> pcm;
    std::string        err;
    const std::string  sample = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    if (!transcribe_cli::load_wav_mono_16k(sample, pcm, err) || pcm.empty()) {
        std::fprintf(stderr, "FAIL could not load %s: %s\n", sample.c_str(), err.c_str());
        return EXIT_FAILURE;
    }

    int n_run = 0;
    for (const Entry & e : entries) {
        const char * path = std::getenv(e.env);
        if (path == nullptr || path[0] == '\0') {
            continue;
        }
        ++n_run;
        if (test_model(e.label, path, pcm) != 0) {
            ++g_failures;
        }
    }
    if (n_run == 0) {
        std::fprintf(stderr, "stream_offline_interleave_smoke: no streaming model env var set; skipping\n");
        return 77;
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
