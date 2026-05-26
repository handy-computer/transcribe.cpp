// moonshine_streaming_stream_parity.cpp - real-model gated test that
// the streaming path produces an identical FINAL transcript to
// transcribe_run on the same audio across a range of chunk sizes,
// and that mid-stream partial transcripts behave correctly.
//
// Asserts (per chunk size):
//   1. Final transcript after stream_finalize equals one-shot.
//   2. revision is non-decreasing across feeds and finalize.
//   3. n_committed_tokens is non-decreasing across feeds, and equals
//      transcribe_n_tokens after finalize (everything committed).
//   4. Whenever result_changed=true at feed i, the full_text differs
//      from the previous feed's snapshot.
//   5. A post-stream transcribe_run still produces the same text
//      (no residual context state corrupting one-shot reuse).
//
// Gating: built only when TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON; at run
// time TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF must point at a tiny
// GGUF. Exits 77 (CTest "skipped") when the env var is unset or the
// file is missing.

#include "transcribe.h"
#include "transcribe/moonshine_streaming.h"

#include "wav.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

int run_one_shot(transcribe_session *      ctx,
                 const std::vector<float> & pcm,
                 std::string &              out_text)
{
    transcribe_run_params rp = transcribe_run_default_params();
    const transcribe_status st = transcribe_run(
        ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "one-shot transcribe_run failed: status=%d (%s)\n",
                     static_cast<int>(st), transcribe_status_string(st));
        return 1;
    }
    const char * t = transcribe_full_text(ctx);
    out_text = (t == nullptr) ? "" : t;
    return 0;
}

int run_streaming(transcribe_session *      ctx,
                  const std::vector<float> & pcm,
                  int                        chunk_samples,
                  int                        min_decode_interval_ms,
                  int *                      out_n_partial_updates,
                  std::string &              out_text)
{
    transcribe_run_params       rp = TRANSCRIBE_RUN_PARAMS_INIT;
    transcribe_stream_params sp = TRANSCRIBE_STREAM_PARAMS_INIT;
    transcribe_moonshine_streaming_stream_ext ms =
        TRANSCRIBE_MOONSHINE_STREAMING_STREAM_EXT_INIT;
    ms.min_decode_interval_ms = min_decode_interval_ms;
    sp.family = &ms.ext;
    transcribe_status st = transcribe_stream_begin(ctx, &rp, &sp);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "stream_begin failed: status=%d (%s)\n",
                     static_cast<int>(st), transcribe_status_string(st));
        return 1;
    }

    int feed_count           = 0;
    int n_partial_updates    = 0;
    size_t pos               = 0;
    int    last_revision     = 0;
    int    last_committed    = 0;
    std::string last_text;
    while (pos < pcm.size()) {
        const size_t take = std::min<size_t>(
            static_cast<size_t>(chunk_samples), pcm.size() - pos);
        transcribe_stream_update upd = TRANSCRIBE_STREAM_UPDATE_INIT;
        st = transcribe_stream_feed(ctx, pcm.data() + pos,
                                    static_cast<int>(take), &upd);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr,
                         "stream_feed[%d] failed: status=%d (%s)\n",
                         feed_count, static_cast<int>(st),
                         transcribe_status_string(st));
            return 1;
        }

        // Revision is monotonic non-decreasing.
        if (upd.revision < last_revision) {
            std::fprintf(stderr,
                         "FAIL feed[%d] revision regressed (%d -> %d)\n",
                         feed_count, last_revision, upd.revision);
            ++g_failures;
        }
        // n_committed_tokens is monotonic non-decreasing.
        const int committed_now = transcribe_stream_n_committed_tokens(ctx);
        if (committed_now < last_committed) {
            std::fprintf(stderr,
                         "FAIL feed[%d] n_committed_tokens regressed "
                         "(%d -> %d)\n", feed_count,
                         last_committed, committed_now);
            ++g_failures;
        }
        // When the family signals result_changed, the visible text
        // must actually differ from the last snapshot.
        if (upd.result_changed) {
            const char * t = transcribe_full_text(ctx);
            const std::string cur = (t == nullptr) ? "" : t;
            if (cur == last_text) {
                std::fprintf(stderr,
                             "FAIL feed[%d] result_changed=true but text "
                             "unchanged ('%s')\n",
                             feed_count, cur.c_str());
                ++g_failures;
            }
            last_text = cur;
            ++n_partial_updates;
        }

        last_revision  = upd.revision;
        last_committed = committed_now;
        pos += take;
        ++feed_count;
    }

    transcribe_stream_update fin_upd = TRANSCRIBE_STREAM_UPDATE_INIT;
    st = transcribe_stream_finalize(ctx, &fin_upd);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "stream_finalize failed: status=%d (%s)\n",
                     static_cast<int>(st), transcribe_status_string(st));
        return 1;
    }
    CHECK(fin_upd.is_final);
    if (fin_upd.revision < last_revision) {
        std::fprintf(stderr,
                     "FAIL finalize revision regressed (%d -> %d)\n",
                     last_revision, fin_upd.revision);
        ++g_failures;
    }

    // At finalize, the whole transcript is committed.
    const int n_tokens_final           = transcribe_n_tokens(ctx);
    const int n_committed_tokens_final =
        transcribe_stream_n_committed_tokens(ctx);
    if (n_tokens_final > 0 &&
        n_committed_tokens_final != n_tokens_final)
    {
        std::fprintf(stderr,
                     "FAIL finalize: n_committed_tokens=%d != "
                     "n_tokens=%d\n",
                     n_committed_tokens_final, n_tokens_final);
        ++g_failures;
    }

    const char * t = transcribe_full_text(ctx);
    out_text = (t == nullptr) ? "" : t;
    if (out_n_partial_updates != nullptr) {
        *out_n_partial_updates = n_partial_updates;
    }
    return 0;
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
    const char * model_path = std::getenv("TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr,
                     "skipping: TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF unset "
                     "or missing\n");
        return 77;
    }

    const std::string sample_path =
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    if (!file_exists(sample_path)) {
        std::fprintf(stderr, "skipping: %s missing\n", sample_path.c_str());
        return 77;
    }

    std::vector<float> pcm;
    {
        std::string err;
        if (!transcribe_cli::load_wav_mono_16k(sample_path, pcm, err) ||
            pcm.empty())
        {
            std::fprintf(stderr, "failed to load %s: %s\n",
                         sample_path.c_str(), err.c_str());
            return 1;
        }
    }

    transcribe_model_load_params  mp = transcribe_model_load_default_params();
    transcribe_session_params cp = transcribe_session_default_params();

    transcribe_model *   model = nullptr;
    transcribe_session * ctx   = nullptr;

    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }
    if (transcribe_session_init(model, &cp, &ctx) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "context init failed\n");
        transcribe_model_free(model);
        return 1;
    }

    // Capabilities sanity: streaming must be advertised, lookahead and
    // chunk hints must be non-zero (Phase 4b-encoder publishes both).
    {
        transcribe_capabilities caps = TRANSCRIBE_CAPABILITIES_INIT;
        CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
        CHECK(caps.supports_streaming);
        CHECK(caps.streaming_lookahead_ms     > 0);
        CHECK(caps.streaming_chunk_ms         > 0);
        CHECK(caps.streaming_lookahead_ms_min > 0);
    }

    // Reference: one-shot transcribe_run.
    std::string ref_text;
    if (run_one_shot(ctx, pcm, ref_text) != 0) {
        transcribe_session_free(ctx);
        transcribe_model_free(model);
        return 1;
    }
    CHECK(!ref_text.empty());
    std::fprintf(stdout, "ref: %s\n", ref_text.c_str());

    // Streaming parity across a range of chunk sizes. 1 ms is a
    // pathological "every PCM sample basically gets its own feed"
    // case; the others cover typical end-user buffer sizes. Throttle
    // = 0 means decode on every advance (matches Phase 4b-full v0
    // behavior).
    const int chunk_ms_choices[] = { 1, 20, 40, 80, 160, 500, 1000 };
    for (int chunk_ms : chunk_ms_choices) {
        const int chunk_samples = std::max(1, chunk_ms * 16000 / 1000);
        std::string stream_text;
        int        n_partial_updates = 0;
        if (run_streaming(ctx, pcm, chunk_samples,
                          /*min_decode_interval_ms=*/0,
                          &n_partial_updates, stream_text) != 0)
        {
            ++g_failures;
            continue;
        }
        if (stream_text != ref_text) {
            std::fprintf(stderr,
                         "FAIL chunk_ms=%d: streamed text differs from one-shot\n"
                         "  ref:    '%s'\n"
                         "  stream: '%s'\n",
                         chunk_ms, ref_text.c_str(), stream_text.c_str());
            ++g_failures;
        } else {
            std::fprintf(stdout, "ok  chunk_ms=%-5d  %s\n",
                         chunk_ms, stream_text.c_str());
        }

        // After finalize, transcribe_run on the same context should
        // also produce the same text (snapshot is replaced atomically).
        // This catches any residual state on the context after
        // stream_finalize that would corrupt a subsequent one-shot.
        std::string post_one_shot;
        if (run_one_shot(ctx, pcm, post_one_shot) != 0) {
            ++g_failures;
            continue;
        }
        if (post_one_shot != ref_text) {
            std::fprintf(stderr,
                         "FAIL chunk_ms=%d: post-stream one-shot text differs\n"
                         "  ref:  '%s'\n"
                         "  post: '%s'\n",
                         chunk_ms, ref_text.c_str(), post_one_shot.c_str());
            ++g_failures;
        }
    }

    // Decode-interval throttle. jfk is ~11 s; feeding 20 ms chunks
    // would naïvely trigger ~550 decodes (one per encoder frame).
    // With the default 240 ms throttle, that should drop to ~46
    // partial updates plus finalize. We assert: (a) final text still
    // matches one-shot, (b) the partial-update count for the
    // throttled run is materially smaller than for the no-throttle
    // run, and (c) a 500 ms throttle reduces the count even more.
    {
        const int chunk_samples = std::max(1, 20 * 16000 / 1000);

        std::string s_no_throttle;
        int n_updates_no_throttle = 0;
        if (run_streaming(ctx, pcm, chunk_samples,
                          /*min_decode_interval_ms=*/0,
                          &n_updates_no_throttle, s_no_throttle) != 0)
        {
            ++g_failures;
        }
        CHECK(s_no_throttle == ref_text);

        std::string s_default_throttle;
        int n_updates_default = 0;
        if (run_streaming(ctx, pcm, chunk_samples,
                          /*min_decode_interval_ms=*/-1, // family default 240
                          &n_updates_default, s_default_throttle) != 0)
        {
            ++g_failures;
        }
        CHECK(s_default_throttle == ref_text);

        std::string s_long_throttle;
        int n_updates_long = 0;
        if (run_streaming(ctx, pcm, chunk_samples,
                          /*min_decode_interval_ms=*/500,
                          &n_updates_long, s_long_throttle) != 0)
        {
            ++g_failures;
        }
        CHECK(s_long_throttle == ref_text);

        std::fprintf(stdout,
                     "throttle: 20ms feeds, partial updates: "
                     "none=%d  default(240ms)=%d  long(500ms)=%d\n",
                     n_updates_no_throttle, n_updates_default, n_updates_long);

        if (n_updates_default >= n_updates_no_throttle) {
            std::fprintf(stderr,
                         "FAIL default throttle did not reduce partial-update "
                         "count (%d >= %d)\n",
                         n_updates_default, n_updates_no_throttle);
            ++g_failures;
        }
        if (n_updates_long >= n_updates_default) {
            std::fprintf(stderr,
                         "FAIL 500ms throttle did not reduce partial-update "
                         "count vs default (%d >= %d)\n",
                         n_updates_long, n_updates_default);
            ++g_failures;
        }
        // 240ms throttle on 11s audio should yield somewhere around
        // 11000/240 ≈ 46 updates. Allow generous slack for the
        // exact warmup boundary and "no new frames at the very end"
        // skips.
        if (n_updates_default < 20 || n_updates_default > 80) {
            std::fprintf(stderr,
                         "FAIL default-throttle update count out of expected "
                         "20..80 range: %d\n", n_updates_default);
            ++g_failures;
        }
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "all checks passed\n");
    return 0;
}
