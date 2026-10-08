// langid.cpp - transcribe-cli LANGID driver: ranked language candidates for
// one file through include/transcribe/langid.h.
//
// Output lines are stable (scripts/validate.py parses `language:`):
//   language: <code> index=<i> p=<p>
//   candidate: <rank> <code> index=<i> p=<p> logit=<z>
// --top N prints (and writes with -o) only the first N ranked candidates.

#include "transcribe/langid.h"

#include "cli.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int transcribe_cli::run_langid_file(const cli_args &           args,
                                    transcribe_model *         model,
                                    const std::vector<float> & pcm,
                                    std::ofstream *            output) {
    if (args.stream_chunk_ms > 0) {
        std::fprintf(stderr, "stream: the language ID path has no streaming entry point; drop --stream-chunk-ms\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_langid_session_params sp;
    transcribe_langid_session_params_init(&sp);
    sp.n_threads                        = args.n_threads;
    transcribe_langid_session * session = nullptr;
    transcribe_status           st      = transcribe_langid_session_init(model, &sp, &session);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "langid session init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    std::vector<const char *> allowed;
    for (const std::string & code : args.langid_allow) {
        allowed.push_back(code.c_str());
    }
    transcribe_langid_params lp;
    transcribe_langid_params_init(&lp);
    lp.allowed   = allowed.empty() ? nullptr : allowed.data();
    lp.n_allowed = static_cast<int32_t>(allowed.size());

    // Only the first info.max_audio_ms of a longer file is scored.
    transcribe_langid_info info;
    transcribe_langid_info_init(&info);
    transcribe_langid_get_info(model, &info);
    const double file_s   = static_cast<double>(pcm.size()) / 16000.0;
    const double scored_s = std::min(file_s, static_cast<double>(info.max_audio_ms) / 1000.0);

    // --repeat N runs transcribe_langid_run() N times for steady-state perf
    // measurements.
    for (int r = 0; r < args.repeat; ++r) {
        st = transcribe_langid_run(session, pcm.data(), static_cast<int>(pcm.size()), &lp);
        if (st != TRANSCRIBE_OK) {
            break;
        }
    }
    std::printf("run: %s\n", transcribe_status_string(st));
    bool output_ok = true;
    if (st == TRANSCRIBE_OK) {
        transcribe_langid_result res;
        transcribe_langid_result_init(&res);
        transcribe_langid_get_result(session, &res);
        const int n_print =
            args.langid_top > 0 ? std::min(args.langid_top, static_cast<int>(res.n_candidates)) : res.n_candidates;

        std::string lines;
        for (int i = 0; i < n_print; ++i) {
            transcribe_langid_candidate c;
            transcribe_langid_candidate_init(&c);
            transcribe_langid_get_candidate(session, i, &c);
            if (i == 0) {
                std::printf("language: %s index=%d p=%.6f\n", c.code, c.index, static_cast<double>(c.p));
                std::printf("  name:         %s\n", c.name);
                std::printf("  allowed_mass: %.6f (%d allowed)\n", static_cast<double>(res.allowed_mass),
                            res.n_candidates);
                std::printf("  scored:       %.2f s of %.2f s\n", scored_s, file_s);
            }
            char line[160];
            std::snprintf(line, sizeof(line), "candidate: %d %s index=%d p=%.6f logit=%.4f\n", i + 1, c.code, c.index,
                          static_cast<double>(c.p), static_cast<double>(c.logit));
            std::printf("  %s", line);
            lines += line;
        }
        output_ok = write_output_file(output, args.output_path, lines.c_str());
    }

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    transcribe_langid_get_timings(session, &tm);
    const double total_ms = tm.mel_ms + tm.encode_ms;
    if (st == TRANSCRIBE_OK && total_ms > 0.0 && scored_s > 0.0) {
        std::printf("  realtime:   %.0fx (%.1f ms for %.1f s; mel %.1f ms, encode %.1f ms)\n",
                    scored_s * 1000.0 / total_ms, total_ms, scored_s, static_cast<double>(tm.mel_ms),
                    static_cast<double>(tm.encode_ms));
    }

    transcribe_langid_session_free(session);
    transcribe_model_free(model);
    return st == TRANSCRIBE_OK && output_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
