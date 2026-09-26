// nemotron3_diar_stream_unit.cpp - Nemotron-3-Diarization operating-point
// extensions (N3DR on the RUN slot, N3DS on the STREAM slot) and push-audio
// live diarization.
//
// Covers, against a real GGUF (env-gated, RC 77 skip):
//
//   1. transcribe_model_accepts_ext_kind: N3DR on _RUN only, N3DS on
//      _STREAM only; foreign kinds rejected. Init functions stamp
//      size/kind/preset.
//   2. Pre-clear rejection: a wrong-kind or out-of-range RUN ext fails with
//      INVALID_ARG and PRESERVES the previous result; an out-of-range STREAM
//      ext fails transcribe_stream_begin before the snapshot is touched.
//   3. Push-audio at LOW_LATENCY on the 8-speaker oracle, fed in 173 ms
//      pieces (not hop-aligned): speaker rows appear DURING feed, audio
//      progress is monotonic, an open turn is reported and later extends,
//      and the finalized rows equal transcribe_run over the same audio at
//      the same preset. Repeated on the non-aligned 91.337 s clip (partial
//      final chunk + floor(n/160) framing at finalize).
//   4. RUN ext preset == env preset parity (the validation hook route).
//   5. Input / memory contract: sub-hop input (< 160 samples) is an empty OK
//      result (NeMo's floor(n/160) = 0 frames) on run, push-audio and batch,
//      and all three agree on short clips; n_ctx is a documented no-op (same
//      rows, unbounded limits); an abort mid-run returns ERR_ABORTED.
//
// Gated by TRANSCRIBE_NEMOTRON3_DIAR_GGUF.

#include "transcribe.h"
#include "transcribe/nemotron3_diar.h"
#include "wav.h"

#include <sys/stat.h>

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

int count_speakers(const std::vector<transcribe_speaker_segment> & rows) {
    std::vector<int> ids;
    for (const auto & r : rows) {
        if (std::find(ids.begin(), ids.end(), r.speaker_id) == ids.end()) {
            ids.push_back(r.speaker_id);
        }
    }
    return static_cast<int>(ids.size());
}

// Push-audio vs whole-file run at one preset.
void check_push_audio(transcribe_session *             session,
                      const std::vector<float> &       pcm,
                      transcribe_nemotron3_diar_preset preset,
                      const char *                     label) {
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_nemotron3_diar_run_ext rx;
    transcribe_nemotron3_diar_run_ext_init(&rx);
    rx.preset = preset;
    rp.family = &rx.ext;
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    const std::vector<transcribe_speaker_segment> run_rows = read_segments(session);
    CHECK(count_speakers(run_rows) == 8);

    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    transcribe_nemotron3_diar_stream_ext sx;
    transcribe_nemotron3_diar_stream_ext_init(&sx);
    sx.preset = preset;
    sp.family = &sx.ext;
    CHECK(transcribe_stream_begin(session, nullptr, &sp) == TRANSCRIBE_OK);

    const int                               piece           = 2768;  // 173 ms: not a multiple of the 160-sample hop
    size_t                                  pos             = 0;
    int64_t                                 last_committed  = 0;
    int                                     feeds_with_rows = 0, changed_feeds = 0;
    bool                                    saw_open_extend = false;
    std::vector<transcribe_speaker_segment> prev;
    while (pos < pcm.size()) {
        const int                take = static_cast<int>(std::min<size_t>(piece, pcm.size() - pos));
        transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        CHECK(transcribe_stream_feed(session, pcm.data() + pos, take, &upd) == TRANSCRIBE_OK);
        pos += static_cast<size_t>(take);
        CHECK(upd.audio_committed_ms >= last_committed);
        CHECK(upd.audio_committed_ms <= upd.input_received_ms);
        last_committed = upd.audio_committed_ms;
        changed_feeds += upd.result_changed ? 1 : 0;
        const std::vector<transcribe_speaker_segment> rows = read_segments(session);
        feeds_with_rows += rows.empty() ? 0 : 1;
        // An open turn: same (speaker, t0) as last feed, later t1.
        for (const auto & r : rows) {
            for (const auto & q : prev) {
                if (r.speaker_id == q.speaker_id && r.t0_ms == q.t0_ms && r.t1_ms > q.t1_ms) {
                    saw_open_extend = true;
                }
            }
        }
        prev = rows;
    }
    transcribe_stream_update fin;
    transcribe_stream_update_init(&fin);
    CHECK(transcribe_stream_finalize(session, &fin) == TRANSCRIBE_OK);
    CHECK(fin.is_final);
    const std::vector<transcribe_speaker_segment> stream_rows = read_segments(session);

    CHECK(feeds_with_rows > 0);
    CHECK(changed_feeds > 0);
    CHECK(saw_open_extend);
    CHECK(same_segments(stream_rows, run_rows));
    std::printf("  %s: run rows=%zu stream rows=%zu feeds-with-rows=%d changed=%d final committed=%lld ms\n", label,
                run_rows.size(), stream_rows.size(), feeds_with_rows, changed_feeds,
                static_cast<long long>(fin.audio_committed_ms));
}

int g_abort_polls = 0;
int g_abort_after = 0;

bool abort_after_n(void * /*user*/) {
    return ++g_abort_polls > g_abort_after;
}

// run, push-audio (one feed) and batch over the same short clip.
void check_short_input(transcribe_session * session, const std::vector<float> & pcm, int n) {
    std::vector<float>    clip(pcm.begin() + 16000 * 44, pcm.begin() + 16000 * 44 + n);
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    const transcribe_status st_run   = transcribe_run(session, clip.data(), n, &rp);
    const auto              run_rows = read_segments(session);
    CHECK(st_run == TRANSCRIBE_OK);

    CHECK(transcribe_stream_begin(session, nullptr, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_feed(session, clip.data(), n, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_finalize(session, nullptr) == TRANSCRIBE_OK);
    CHECK(same_segments(read_segments(session), run_rows));

    const float * ptrs[2] = { clip.data(), clip.data() };
    const int     lens[2] = { n, n };
    CHECK(transcribe_run_batch(session, ptrs, lens, 2, &rp) == TRANSCRIBE_OK);
    for (int i = 0; i < 2; ++i) {
        CHECK(transcribe_batch_status(session, i) == TRANSCRIBE_OK);
        CHECK(transcribe_batch_n_speaker_segments(session, i) == static_cast<int>(run_rows.size()));
    }
    if (n < 160) {
        CHECK(run_rows.empty());
    }
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "nemotron3_diar_stream_unit: TRANSCRIBE_NEMOTRON3_DIAR_GGUF not set; skipping.\n"
                     "Re-run with TRANSCRIBE_NEMOTRON3_DIAR_GGUF=models/Nemotron-3-Diarization/"
                     "Nemotron-3-Diarization-BF16.gguf\n");
        return 77;
    }
    const std::string gguf = env;
    if (!file_exists(gguf)) {
        std::fprintf(stderr, "nemotron3_diar_stream_unit: file not found: %s\n", gguf.c_str());
        return 77;
    }
    std::vector<float> pcm, pcm_trunc;
    std::string        wav_err;
    const std::string  dir = TRANSCRIBE_TEST_SAMPLES_DIR;
    if (!transcribe_cli::load_wav_mono_16k(dir + "/nemotron3-diar-8spk-mix.wav", pcm, wav_err) ||
        !transcribe_cli::load_wav_mono_16k(dir + "/nemotron3-diar-8spk-mix-trunc.wav", pcm_trunc, wav_err)) {
        std::fprintf(stderr, "nemotron3_diar_stream_unit: wav load: %s\n", wav_err.c_str());
        return 77;
    }
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                      = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(gguf.c_str(), &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }

    // 1. Kind + slot probe, init stamping.
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN));
    CHECK(
        !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN));
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                            TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_STREAM));
    CHECK(
        !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_STREAM));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, 0x54534653u /* SFST */));
    {
        transcribe_nemotron3_diar_run_ext rx;
        transcribe_nemotron3_diar_run_ext_init(&rx);
        CHECK(rx.ext.size == sizeof(rx));
        CHECK(rx.ext.kind == TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN);
        CHECK(rx.preset == TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT);
        transcribe_nemotron3_diar_stream_ext sx;
        transcribe_nemotron3_diar_stream_ext_init(&sx);
        CHECK(sx.ext.size == sizeof(sx));
        CHECK(sx.ext.kind == TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_STREAM);
        CHECK(sx.preset == TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT);
    }

    struct transcribe_session * session = nullptr;
    if (transcribe_session_init(model, nullptr, &session) != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL: session create\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // Baseline: default preset (= VERY_HIGH_LATENCY) via the env hook route.
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    ::setenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET", "very_high_latency", 1);
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    const std::vector<transcribe_speaker_segment> env_rows = read_segments(session);
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");
    CHECK(count_speakers(env_rows) == 8);

    // 2. Pre-clear rejection preserves the previous result.
    {
        transcribe_nemotron3_diar_run_ext bad;
        transcribe_nemotron3_diar_run_ext_init(&bad);
        bad.ext.kind = 0x54534653u;  // SFST: another family's kind
        rp.family    = &bad.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), env_rows));

        transcribe_nemotron3_diar_run_ext oor;
        transcribe_nemotron3_diar_run_ext_init(&oor);
        oor.preset = static_cast<transcribe_nemotron3_diar_preset>(99);
        rp.family  = &oor.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), env_rows));

        transcribe_stream_params sp;
        transcribe_stream_params_init(&sp);
        transcribe_nemotron3_diar_stream_ext soor;
        transcribe_nemotron3_diar_stream_ext_init(&soor);
        soor.preset = static_cast<transcribe_nemotron3_diar_preset>(99);
        sp.family   = &soor.ext;
        CHECK(transcribe_stream_begin(session, nullptr, &sp) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(same_segments(read_segments(session), env_rows));
    }

    // 4. RUN ext preset == env preset.
    {
        transcribe_nemotron3_diar_run_ext vh;
        transcribe_nemotron3_diar_run_ext_init(&vh);
        vh.preset = TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY;
        rp.family = &vh.ext;
        CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
        CHECK(same_segments(read_segments(session), env_rows));
    }

    // 5. Input / memory contract.
    for (const int n : { 1, 159, 160, 400, 1601, 8000 }) {
        check_short_input(session, pcm, n);
    }
    {
        transcribe_session_params sp;
        transcribe_session_params_init(&sp);
        sp.n_ctx                         = 16;  // ignored: no lowerable context
        struct transcribe_session * s_nc = nullptr;
        CHECK(transcribe_session_init(model, &sp, &s_nc) == TRANSCRIBE_OK);
        transcribe_session_limits lim;
        transcribe_session_limits_init(&lim);
        CHECK(transcribe_session_get_limits(s_nc, &lim) == TRANSCRIBE_OK);
        CHECK(lim.effective_max_audio_ms == 0);
        transcribe_run_params rp0;
        transcribe_run_params_init(&rp0);
        CHECK(transcribe_run(s_nc, pcm.data(), static_cast<int>(pcm.size()), &rp0) == TRANSCRIBE_OK);
        CHECK(same_segments(read_segments(s_nc), env_rows));

        g_abort_polls = 0;
        g_abort_after = 2;  // pass the entry poll + one chunk, then abort
        transcribe_set_abort_callback(s_nc, abort_after_n, nullptr);
        CHECK(transcribe_run(s_nc, pcm.data(), static_cast<int>(pcm.size()), &rp0) == TRANSCRIBE_ERR_ABORTED);
        CHECK(transcribe_was_aborted(s_nc));
        transcribe_set_abort_callback(s_nc, nullptr, nullptr);
        transcribe_session_free(s_nc);
    }

    // 3. Push-audio == whole-file run.
    check_push_audio(session, pcm, TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY, "low_latency 92s");
    check_push_audio(session, pcm_trunc, TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY, "low_latency 91.337s");
    check_push_audio(session, pcm_trunc, TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY,
                     "very_high_latency 91.337s");

    transcribe_session_free(session);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "nemotron3_diar_stream_unit: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("nemotron3_diar_stream_unit: OK\n");
    return 0;
}
