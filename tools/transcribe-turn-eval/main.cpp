// transcribe-turn-eval - score a stream of clips with one TURN model.
//
//   transcribe-turn-eval MODEL.gguf [--threads N] [--backend cpu|auto] < clips.bin
//
// stdin: repeated records of int32 n_samples followed by n_samples float32
// samples (16 kHz mono). stdout: one line per clip, "<p> <ended>", in input
// order. Errors go to stderr and end the run with a non-zero exit code.

#include "transcribe.h"
#include "transcribe/turn.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [--threads N] [--backend cpu|auto] < clips.bin\n", argv[0]);
        return 2;
    }
    int                        n_threads = 0;
    transcribe_backend_request backend   = TRANSCRIBE_BACKEND_AUTO;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char * b = argv[++i];
            backend        = std::strcmp(b, "cpu") == 0 ? TRANSCRIBE_BACKEND_CPU : TRANSCRIBE_BACKEND_AUTO;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    transcribe_model_load_params lp;
    transcribe_model_load_params_init(&lp);
    lp.backend               = backend;
    transcribe_model * model = nullptr;
    transcribe_status  st    = transcribe_model_load_file(argv[1], &lp, &model);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "load: %s\n", transcribe_status_string(st));
        return 1;
    }
    transcribe_turn_session_params sp;
    transcribe_turn_session_params_init(&sp);
    sp.n_threads                      = n_threads;
    transcribe_turn_session * session = nullptr;
    st                                = transcribe_turn_session_init(model, &sp, &session);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "session: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return 1;
    }

    std::vector<float> pcm;
    int                rc = 0;
    for (;;) {
        int32_t n = 0;
        if (std::fread(&n, sizeof(n), 1, stdin) != 1) {
            break;
        }
        if (n <= 0) {
            std::fprintf(stderr, "bad record length %d\n", n);
            rc = 1;
            break;
        }
        pcm.resize(static_cast<size_t>(n));
        if (std::fread(pcm.data(), sizeof(float), pcm.size(), stdin) != pcm.size()) {
            std::fprintf(stderr, "truncated record\n");
            rc = 1;
            break;
        }
        st = transcribe_turn_run(session, pcm.data(), n, nullptr);
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "run: %s\n", transcribe_status_string(st));
            rc = 1;
            break;
        }
        transcribe_turn_result r;
        transcribe_turn_result_init(&r);
        transcribe_turn_get_result(session, &r);
        std::printf("%.9g %d\n", static_cast<double>(r.p), r.ended ? 1 : 0);
        std::fflush(stdout);
    }
    transcribe_turn_session_free(session);
    transcribe_model_free(model);
    return rc;
}
