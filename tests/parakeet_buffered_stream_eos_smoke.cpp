// The 7.30 s JFK cut ends with "you" still inside the RNN-T emission
// lag, exercising EOS silence lookahead. The 7.28 s cut is exactly
// C+R+5*C at the default C=R=13-frame geometry, so finalize must flush
// retained right context even though no unread samples remain.

#include "transcribe.h"
#include "transcribe/parakeet.h"
#include "wav.h"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
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

constexpr int     k_sample_rate           = 16000;
constexpr size_t  k_encoder_frame_samples = 1280;
constexpr size_t  k_raw_cut_samples       = 116800;
constexpr size_t  k_boundary_samples      = (13 + 13 + 5 * 13) * k_encoder_frame_samples;
constexpr int64_t k_right_ms              = 1040;

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::string normalize(const std::string & text) {
    std::string out;
    bool        pending_space = false;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            if (pending_space && !out.empty()) {
                out.push_back(' ');
            }
            pending_space = false;
            out.push_back(static_cast<char>(std::tolower(c)));
        } else {
            pending_space = true;
        }
    }
    return out;
}

bool contains_tail(const std::string & text) {
    return normalize(text).find("country can do for you") != std::string::npos;
}

bool run_stream(transcribe_session * ctx, const std::vector<float> & pcm, std::string & text) {
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    transcribe_parakeet_buffered_stream_ext ext;
    transcribe_parakeet_buffered_stream_ext_init(&ext);
    sp.family = &ext.ext;

    transcribe_status st = transcribe_stream_begin(ctx, &rp, &sp);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "stream_begin failed: %s\n", transcribe_status_string(st));
        return false;
    }

    size_t  pos            = 0;
    int64_t last_committed = 0;
    int     feed_index     = 0;
    while (pos < pcm.size()) {
        const size_t             wanted = (feed_index % 2 == 0) ? 255 : 257;
        const size_t             take   = std::min(wanted, pcm.size() - pos);
        transcribe_stream_update update;
        transcribe_stream_update_init(&update);
        st = transcribe_stream_feed(ctx, pcm.data() + pos, static_cast<int>(take), &update);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "stream_feed failed: %s\n", transcribe_status_string(st));
            return false;
        }
        pos += take;
        ++feed_index;

        const int64_t expected_input = static_cast<int64_t>(pos) * 1000 / k_sample_rate;
        CHECK(update.input_received_ms == expected_input);
        CHECK(update.audio_committed_ms >= last_committed);
        CHECK(update.audio_committed_ms <= update.input_received_ms);
        CHECK(update.buffered_ms == update.input_received_ms - update.audio_committed_ms);
        if (update.audio_committed_ms > 0) {
            CHECK(update.input_received_ms - update.audio_committed_ms >= k_right_ms);
        }
        last_committed = update.audio_committed_ms;
    }

    transcribe_stream_update final;
    transcribe_stream_update_init(&final);
    st = transcribe_stream_finalize(ctx, &final);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "stream_finalize failed: %s\n", transcribe_status_string(st));
        return false;
    }

    const int64_t expected_ms = static_cast<int64_t>(pcm.size()) * 1000 / k_sample_rate;
    CHECK(final.is_final);
    CHECK(final.input_received_ms == expected_ms);
    CHECK(final.audio_committed_ms == expected_ms);
    CHECK(final.buffered_ms == 0);

    const char * full_text = transcribe_full_text(ctx);
    text                   = full_text == nullptr ? "" : full_text;
    for (int i = 0; i < transcribe_n_tokens(ctx); ++i) {
        transcribe_token token;
        transcribe_token_init(&token);
        CHECK(transcribe_get_token(ctx, i, &token) == TRANSCRIBE_OK);
        CHECK(token.t0_ms <= token.t1_ms);
        CHECK(token.t1_ms <= expected_ms);
    }
    return true;
}

}  // namespace

int main() {
    const char * model_path = std::getenv("TRANSCRIBE_PARAKEET_UNIFIED_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr, "skipping: TRANSCRIBE_PARAKEET_UNIFIED_GGUF unset or missing\n");
        return 77;
    }

    const std::string  sample_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    std::vector<float> full_pcm;
    std::string        error;
    if (!transcribe_cli::load_wav_mono_16k(sample_path, full_pcm, error) || full_pcm.size() < k_raw_cut_samples) {
        std::fprintf(stderr, "failed to load %s: %s\n", sample_path.c_str(), error.c_str());
        return 1;
    }

    transcribe_model_load_params model_params;
    transcribe_model_load_params_init(&model_params);
    transcribe_session_params session_params;
    transcribe_session_params_init(&session_params);
    transcribe_model *   model = nullptr;
    transcribe_session * ctx   = nullptr;
    if (transcribe_model_load_file(model_path, &model_params, &model) != TRANSCRIBE_OK ||
        transcribe_session_init(model, &session_params, &ctx) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "failed to initialize model/session\n");
        transcribe_model_free(model);
        return 1;
    }

    std::vector<float>    raw(full_pcm.begin(), full_pcm.begin() + k_raw_cut_samples);
    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    CHECK(transcribe_run(ctx, raw.data(), static_cast<int>(raw.size()), &run_params) == TRANSCRIBE_OK);
    const char *      batch_result = transcribe_full_text(ctx);
    const std::string batch_text   = batch_result == nullptr ? "" : batch_result;
    CHECK(contains_tail(batch_text));

    std::string raw_text;
    CHECK(run_stream(ctx, raw, raw_text));
    CHECK(contains_tail(raw_text));
    const std::string batch_normalized = normalize(batch_text);
    const std::string raw_normalized   = normalize(raw_text);
    CHECK(batch_normalized.compare(0, raw_normalized.size(), raw_normalized) == 0);

    std::vector<float> boundary(full_pcm.begin(), full_pcm.begin() + k_boundary_samples);
    std::string        boundary_text;
    CHECK(run_stream(ctx, boundary, boundary_text));
    CHECK(contains_tail(boundary_text));

    transcribe_session_free(ctx);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "parakeet_buffered_stream_eos_smoke: %d failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
