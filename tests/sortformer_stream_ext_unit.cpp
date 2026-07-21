// sortformer_stream_ext_unit.cpp - Sortformer streaming operating-point run
// extension (TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM, RUN slot).
//
// Covers, against a real GGUF (env-gated, RC 77 skip):
//
//   1. transcribe_model_accepts_ext_kind: SFST accepted on _RUN only;
//      foreign kinds rejected.
//   2. transcribe_sortformer_stream_ext_init stamps size/kind/preset.
//   3. Pre-clear rejection: a wrong-kind ext and an out-of-range preset
//      both fail with INVALID_ARG and PRESERVE the previous result
//      (run_validate fires before clear_result).
//   4. Ext preset == env preset parity: running the committed 2-speaker
//      oracle mix with ext=VERY_HIGH_LATENCY produces exactly the same
//      speaker-segment rows as TRANSCRIBE_SORTFORMER_STREAM_PRESET=
//      very_high_latency (the DER-validated env path), and a different
//      geometry (LOW_LATENCY) is accepted and produces rows.
//
// Gated by TRANSCRIBE_SORTFORMER_GGUF (same pattern as the other
// real-model tests) because the preset resolution runs inside run() and
// needs real weights to produce comparable segments.

#include "transcribe.h"
#include "transcribe/sortformer.h"
#include "wav.h"

#include <sys/stat.h>

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

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<transcribe_speaker_segment> read_segments(const transcribe_session * session) {
    std::vector<transcribe_speaker_segment> rows;
    const int                               n = transcribe_n_speaker_segments(session);
    for (int i = 0; i < n; ++i) {
        transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        if (transcribe_get_speaker_segment(session, i, &row) == TRANSCRIBE_OK) {
            rows.push_back(row);
        }
    }
    return rows;
}

bool same_segments(const std::vector<transcribe_speaker_segment> & a,
                   const std::vector<transcribe_speaker_segment> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].t0_ms != b[i].t0_ms || a[i].t1_ms != b[i].t1_ms || a[i].speaker_id != b[i].speaker_id) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_SORTFORMER_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "sortformer_stream_ext_unit: TRANSCRIBE_SORTFORMER_GGUF not set; skipping.\n"
                     "Re-run with TRANSCRIBE_SORTFORMER_GGUF=models/diar_streaming_sortformer_4spk-v2.1/"
                     "diar_streaming_sortformer_4spk-v2.1-F32.gguf\n");
        return 77;
    }
    const std::string gguf = env;
    if (!file_exists(gguf)) {
        std::fprintf(stderr, "sortformer_stream_ext_unit: file not found: %s\n", gguf.c_str());
        return 77;
    }
    const std::string wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/sortformer-2spk-mix.wav";
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::fprintf(stderr, "sortformer_stream_ext_unit: wav load: %s\n", wav_err.c_str());
        return 77;
    }

    // The parity leg compares the ext path against the env path, so the
    // environment must start clean of the validation overrides.
    ::unsetenv("TRANSCRIBE_SORTFORMER_STREAM_PRESET");

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = TRANSCRIBE_BACKEND_CPU;  // deterministic float order for the parity leg
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(gguf.c_str(), &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }

    // 1. Kind+slot probe.
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                             TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, 0x4E524857u /* WHRN */));

    // 2. Init function stamps the header + default.
    transcribe_sortformer_stream_ext ext;
    transcribe_sortformer_stream_ext_init(&ext);
    CHECK(ext.ext.size == sizeof(ext));
    CHECK(ext.ext.kind == TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM);
    CHECK(ext.preset == TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);

    struct transcribe_session * session = nullptr;
    if (transcribe_session_init(model, nullptr, &session) != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL: session create\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);

    // Baseline run (env preset path, the DER-validated harness route).
    ::setenv("TRANSCRIBE_SORTFORMER_STREAM_PRESET", "very_high_latency", 1);
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    const std::vector<transcribe_speaker_segment> env_rows = read_segments(session);
    ::unsetenv("TRANSCRIBE_SORTFORMER_STREAM_PRESET");
    CHECK(!env_rows.empty());  // the oracle mix has two speakers

    // 3. Pre-clear rejection preserves the previous result.
    {
        transcribe_sortformer_stream_ext bad;
        transcribe_sortformer_stream_ext_init(&bad);
        bad.ext.kind = 0x4E524857u;  // WHRN: wrong family kind
        rp.family    = &bad.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
              TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), env_rows));  // snapshot intact

        transcribe_sortformer_stream_ext oor;
        transcribe_sortformer_stream_ext_init(&oor);
        oor.preset = static_cast<transcribe_sortformer_preset>(99);
        rp.family  = &oor.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) ==
              TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), env_rows));
    }

    // 4a. Ext preset == env preset parity (same operating point, same rows).
    {
        transcribe_sortformer_stream_ext vh;
        transcribe_sortformer_stream_ext_init(&vh);
        vh.preset = TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY;
        rp.family = &vh.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
        CHECK(same_segments(read_segments(session), env_rows));
    }

    // 4b. A different geometry is accepted and produces speaker rows.
    {
        transcribe_sortformer_stream_ext lo;
        transcribe_sortformer_stream_ext_init(&lo);
        lo.preset = TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY;
        rp.family = &lo.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
        CHECK(!read_segments(session).empty());
    }

    transcribe_session_free(session);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "sortformer_stream_ext_unit: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("sortformer_stream_ext_unit: OK\n");
    return 0;
}
