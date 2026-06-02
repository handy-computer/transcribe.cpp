// stream_committed_pointer_stability.cpp - real-model gated test that
// stream committed_text is append-only for the life of the stream.
//
// The filename is kept for continuity with the original test draft, but
// the contract under test is now the text-view API: row pointers remain
// raw snapshot views and may be invalidated by every feed.
//
// Asserts:
//   1. committed_text never rewrites previously exposed bytes.
//   2. tentative_text is empty after finalize.
//   3. full_text remains readable as the raw final hypothesis.
//
// Gating: built only when TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON; at run
// time TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF must point at a tiny
// GGUF. Exits 77 (CTest "skipped") when the env var is unset or the file
// is missing.

#include "transcribe.h"

#include "wav.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
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

std::string read_committed_text(transcribe_session * ctx,
                                std::string &        tentative,
                                std::string &        full,
                                const char *         where)
{
    transcribe_stream_text text;
    transcribe_stream_text_init(&text);
    CHECK(transcribe_stream_get_text(ctx, &text) == TRANSCRIBE_OK);
    CHECK(text.committed_text != nullptr);
    CHECK(text.tentative_text != nullptr);
    CHECK(text.full_text != nullptr);
    if (text.committed_text == nullptr ||
        text.tentative_text == nullptr ||
        text.full_text == nullptr)
    {
        std::fprintf(stderr, "FAIL %s: null stream text pointer\n", where);
        return "";
    }
    tentative = text.tentative_text;
    full      = text.full_text;
    return text.committed_text;
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
    const char * model_path =
        std::getenv("TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF");
    if (model_path == nullptr || *model_path == '\0' ||
        !file_exists(model_path))
    {
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

    struct transcribe_session * ctx = nullptr;
    if (transcribe_open(model_path, nullptr, nullptr, &ctx) != TRANSCRIBE_OK ||
        ctx == nullptr)
    {
        std::fprintf(stderr, "transcribe_open failed for %s\n", model_path);
        return 1;
    }

    if (transcribe_stream_begin(ctx, nullptr, nullptr) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "stream_begin failed\n");
        transcribe_session_free(ctx);
        return 1;
    }

    // Small chunks => many feeds => maximal opportunity for the committed
    // prefix to grow incrementally and for a stale pointer to surface.
    const int chunk_samples = 250 * 16000 / 1000; // 250 ms
    std::string prev_committed;
    int pos = 0;
    while (pos < static_cast<int>(pcm.size())) {
        int take = chunk_samples;
        if (pos + take > static_cast<int>(pcm.size())) {
            take = static_cast<int>(pcm.size()) - pos;
        }
        const transcribe_status st =
            transcribe_stream_feed(ctx, pcm.data() + pos, take, nullptr);
        CHECK(st == TRANSCRIBE_OK);
        if (st != TRANSCRIBE_OK) break;

        std::string tentative;
        std::string full;
        const std::string committed =
            read_committed_text(ctx, tentative, full, "feed");
        CHECK(committed.rfind(prev_committed, 0) == 0);
        prev_committed = committed;
        pos += take;
    }

    const transcribe_status fin =
        transcribe_stream_finalize(ctx, nullptr);
    CHECK(fin == TRANSCRIBE_OK);

    std::string tentative;
    std::string full;
    const std::string committed =
        read_committed_text(ctx, tentative, full, "finalize");
    CHECK(committed.rfind(prev_committed, 0) == 0);
    CHECK(tentative.empty());
    CHECK(!full.empty());

    if (g_failures == 0) {
        std::fprintf(stderr,
                     "PASS: committed_text stayed append-only across "
                     "the stream (%zu bytes)\n",
                     committed.size());
    }

    transcribe_session_free(ctx);
    return g_failures == 0 ? 0 : 1;
}
