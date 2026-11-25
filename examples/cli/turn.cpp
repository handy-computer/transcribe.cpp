// turn.cpp - transcribe-cli TURN driver: end-of-turn detection for one file
// through include/transcribe/turn.h.
//
// The file is scored as one clip ("has the turn ended at the end of this
// audio?"). Output lines are stable (scripts parse them):
//   turn: p=<probability> ended=<0|1> threshold=<cut-off>
// -o writes this line.

#include "transcribe/turn.h"

#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int transcribe_cli::run_turn_file(const cli_args &           args,
                                  transcribe_model *         model,
                                  const std::vector<float> & pcm,
                                  double                     duration_s,
                                  std::ofstream *            output) {
    // The CLI drives the clip path only; streaming is a library API
    // (docs/turn.md).
    if (args.stream_chunk_ms > 0) {
        std::fprintf(stderr, "stream: the turn path has no streaming CLI mode; drop --stream-chunk-ms\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_turn_session_params sp;
    transcribe_turn_session_params_init(&sp);
    sp.n_threads                      = args.n_threads;
    transcribe_turn_session * session = nullptr;
    transcribe_status         st      = transcribe_turn_session_init(model, &sp, &session);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "turn session init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    transcribe_turn_info info;
    transcribe_turn_info_init(&info);
    transcribe_turn_get_info(model, &info);

    transcribe_turn_params tp;
    transcribe_turn_params_init(&tp);
    if (args.turn_threshold >= 0.0) {
        tp.threshold = args.turn_threshold;
    }
    const double threshold = tp.threshold >= 0.0 ? tp.threshold : info.threshold;

    // --repeat N runs transcribe_turn_run() N times for steady-state perf
    // measurements.
    for (int r = 0; r < args.repeat; ++r) {
        st = transcribe_turn_run(session, pcm.data(), static_cast<int>(pcm.size()), &tp);
        if (st != TRANSCRIBE_OK) {
            break;
        }
    }
    std::printf("run: %s\n", transcribe_status_string(st));
    bool output_ok = true;
    if (st == TRANSCRIBE_OK) {
        transcribe_turn_result res;
        transcribe_turn_result_init(&res);
        transcribe_turn_get_result(session, &res);
        std::string lines;
        char        line[200];
        std::snprintf(line, sizeof(line), "turn: p=%.6f ended=%d threshold=%.2f\n", static_cast<double>(res.p),
                      res.ended ? 1 : 0, threshold);
        std::printf("%s", line);
        lines += line;
        if (info.max_audio_ms > 0 && duration_s * 1000.0 > info.max_audio_ms) {
            std::printf("  scored:     the last %.1f s of %.2f s\n", info.max_audio_ms / 1000.0, duration_s);
        }
        if (info.frame_samples > 0) {
            std::printf("  frames:     %d (%d samples each)\n", res.n_probs, info.frame_samples);
        }
        output_ok = write_output_file(output, args.output_path, lines.c_str());
    }

    transcribe_timings tm;
    transcribe_timings_init(&tm);
    transcribe_turn_get_timings(session, &tm);
    const double total_ms = tm.mel_ms + tm.encode_ms + tm.decode_ms;
    if (st == TRANSCRIBE_OK && total_ms > 0.0) {
        std::printf("  time:       %.1f ms (mel %.1f ms, encode %.1f ms, decode %.1f ms)\n", total_ms,
                    static_cast<double>(tm.mel_ms), static_cast<double>(tm.encode_ms),
                    static_cast<double>(tm.decode_ms));
    }

    transcribe_turn_session_free(session);
    transcribe_model_free(model);
    return st == TRANSCRIBE_OK && output_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
