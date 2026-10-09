// vad.cpp - transcribe-cli VAD driver: speech segments for one file through
// include/transcribe/vad.h.
//
// Output lines are stable (scripts/validate.py parses `segment:`):
//   segment: <i> start=<sample> end=<sample> t0=<s> t1=<s>
// -o writes the segment lines.

#include "transcribe/vad.h"

#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int transcribe_cli::run_vad_file(const cli_args &           args,
                                 transcribe_model *         model,
                                 const std::vector<float> & pcm,
                                 double                     duration_s,
                                 std::ofstream *            output) {
    // The CLI drives the offline path only; streaming is a library API
    // (docs/vad.md).
    if (args.stream_chunk_ms > 0) {
        std::fprintf(stderr, "stream: the VAD path has no streaming CLI mode; drop --stream-chunk-ms\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    sp.n_threads                     = args.n_threads;
    transcribe_vad_session * session = nullptr;
    transcribe_status        st      = transcribe_vad_session_init(model, &sp, &session);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "vad session init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_vad_info info;
    transcribe_vad_info_init(&info);
    transcribe_vad_get_info(model, &info);

    transcribe_vad_params vp;
    transcribe_vad_params_init(&vp);
    if (args.vad_threshold >= 0.0) {
        vp.threshold = args.vad_threshold;
    }

    // --repeat N runs transcribe_vad_run() N times for steady-state perf
    // measurements.
    for (int r = 0; r < args.repeat; ++r) {
        st = transcribe_vad_run(session, pcm.data(), static_cast<int>(pcm.size()), &vp);
        if (st != TRANSCRIBE_OK) {
            break;
        }
    }
    std::printf("run: %s\n", transcribe_status_string(st));
    bool output_ok = true;
    if (st == TRANSCRIBE_OK) {
        transcribe_vad_result res;
        transcribe_vad_result_init(&res);
        transcribe_vad_get_result(session, &res);
        std::printf("speech segments: %d\n", res.n_segments);
        std::string lines;
        double      speech_s = 0.0;
        for (int i = 0; i < res.n_segments; ++i) {
            transcribe_vad_segment seg;
            transcribe_vad_segment_init(&seg);
            transcribe_vad_get_segment(session, i, &seg);
            const double t0 = static_cast<double>(seg.start_sample) / info.sample_rate;
            const double t1 = static_cast<double>(seg.end_sample) / info.sample_rate;
            speech_s += t1 - t0;
            char line[160];
            std::snprintf(line, sizeof(line), "segment: %d start=%lld end=%lld t0=%.3f t1=%.3f\n", i,
                          static_cast<long long>(seg.start_sample), static_cast<long long>(seg.end_sample), t0, t1);
            std::printf("  %s", line);
            lines += line;
        }
        std::printf("  frames:     %d (%d samples each)\n", res.n_probs, info.frame_samples);
        std::printf("  speech:     %.2f s of %.2f s (threshold %.2f)\n", speech_s, duration_s, vp.threshold);
        output_ok = write_output_file(output, args.output_path, lines.c_str());
    }

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    transcribe_vad_get_timings(session, &tm);
    const double total_ms = tm.encode_ms + tm.decode_ms;
    if (st == TRANSCRIBE_OK && total_ms > 0.0 && duration_s > 0.0) {
        std::printf("  realtime:   %.0fx (%.1f ms for %.1f s; encode %.1f ms, decode %.1f ms)\n",
                    duration_s * 1000.0 / total_ms, total_ms, duration_s, static_cast<double>(tm.encode_ms),
                    static_cast<double>(tm.decode_ms));
    }

    transcribe_vad_session_free(session);
    transcribe_model_free(model);
    return st == TRANSCRIBE_OK && output_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
