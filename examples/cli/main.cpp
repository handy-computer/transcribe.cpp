// transcribe-cli - example CLI driver for transcribe.cpp
//
// Pass 2 stub: this CLI exercises the public ABI end-to-end with the
// stub library. It loads a WAV file (must be 16 kHz mono float32 source
// or downmixable to mono), prints duration / sample count, optionally
// asks the library to load a model (which will return NOT_IMPLEMENTED
// in pass 2), and prints the resulting status string.
//
// Once the loader and decoder land, this same CLI will run real
// transcription. Argument parsing, log sink wiring, and result printing
// will grow in place rather than being rewritten.
//
// Usage:
//   transcribe-cli [options] audio.wav
// Options:
//   -m, --model PATH   GGUF model file (optional in pass 2)
//   -l, --language ISO BCP-47-ish language hint (e.g. en, de)
//   -t, --translate    set task to TRANSLATE
//   -q, --quiet        suppress library log output
//   -h, --help         show this help

#include "transcribe.h"
#include "wav.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct cli_args {
    std::string wav_path;
    std::string model_path;
    std::string language;
    std::string batch_file; // --batch: one wav path per line
    bool        translate = false;
    bool        quiet     = false;
    bool        batch_jsonl = false; // --batch-jsonl: output JSONL
    int         repeat    = 1;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [options] audio.wav\n"
        "       %s [options] --batch file.list -m model.gguf\n"
        "options:\n"
        "  -m, --model PATH      GGUF model file\n"
        "  -l, --language ISO    BCP-47-ish language hint (e.g. en, de)\n"
        "  -t, --translate       set task to TRANSLATE\n"
        "  -q, --quiet           suppress library log output\n"
        "  -r, --repeat N        run N times per file (benchmark)\n"
        "  --batch FILE          batch mode: FILE has one wav path per line\n"
        "  --batch-jsonl         output one JSON line per file (for batch)\n"
        "  -h, --help            show this help\n",
        argv0, argv0);
}

bool parse_args(int argc, char ** argv, cli_args & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "-m" || a == "--model") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.model_path = v;
        } else if (a == "-l" || a == "--language") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.language = v;
        } else if (a == "-t" || a == "--translate") {
            out.translate = true;
        } else if (a == "-q" || a == "--quiet") {
            out.quiet = true;
        } else if (a == "-r" || a == "--repeat") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.repeat = std::atoi(v);
            if (out.repeat < 1) out.repeat = 1;
        } else if (a == "--batch") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.batch_file = v;
        } else if (a == "--batch-jsonl") {
            out.batch_jsonl = true;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return false;
        } else {
            if (!out.wav_path.empty()) {
                std::fprintf(stderr, "error: multiple positional arguments\n");
                return false;
            }
            out.wav_path = a;
        }
    }
    if (out.wav_path.empty() && out.batch_file.empty()) {
        std::fprintf(stderr, "error: missing audio.wav or --batch\n");
        return false;
    }
    if (!out.wav_path.empty() && !out.batch_file.empty()) {
        std::fprintf(stderr, "error: cannot combine positional audio.wav with --batch\n");
        return false;
    }
    return true;
}

void log_cb(transcribe_log_level level, const char * msg, void * userdata) {
    (void)userdata;
    const char * prefix = "[?]";
    switch (level) {
        case TRANSCRIBE_LOG_LEVEL_NONE:  return;
        case TRANSCRIBE_LOG_LEVEL_INFO:  prefix = "[info]";  break;
        case TRANSCRIBE_LOG_LEVEL_WARN:  prefix = "[warn]";  break;
        case TRANSCRIBE_LOG_LEVEL_ERROR: prefix = "[error]"; break;
        case TRANSCRIBE_LOG_LEVEL_DEBUG: prefix = "[debug]"; break;
        case TRANSCRIBE_LOG_LEVEL_CONT:  prefix = "";        break;
    }
    std::fprintf(stderr, "%s %s%s", prefix, msg,
                 (msg && *msg && msg[std::strlen(msg) - 1] == '\n') ? "" : "\n");
}

} // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Install the log sink ONCE at startup, before any models or contexts
    // exist. This is the only supported usage model in 0.x; see the
    // threading contract in transcribe.h.
    if (!args.quiet) {
        transcribe_log_set(log_cb, nullptr);
    }

    // ---- Batch mode: --batch reads a file list, one wav path per
    // line. Loads the model ONCE and reuses the context across all
    // files. Outputs one JSONL line per file to stdout when
    // --batch-jsonl is set, otherwise the same human-readable format
    // as single-file mode. ------------------------------------------------
    if (!args.batch_file.empty()) {
        if (args.model_path.empty()) {
            std::fprintf(stderr, "error: --batch requires --model\n");
            return EXIT_FAILURE;
        }

        // Read the file list.
        std::vector<std::string> wav_paths;
        {
            std::ifstream fin(args.batch_file);
            if (!fin) {
                std::fprintf(stderr, "error: cannot open batch file %s\n",
                             args.batch_file.c_str());
                return EXIT_FAILURE;
            }
            std::string line;
            while (std::getline(fin, line)) {
                // Trim trailing whitespace / carriage return.
                while (!line.empty() &&
                       (line.back() == '\n' || line.back() == '\r' ||
                        line.back() == ' '  || line.back() == '\t'))
                {
                    line.pop_back();
                }
                if (!line.empty()) {
                    wav_paths.push_back(line);
                }
            }
        }
        if (wav_paths.empty()) {
            std::fprintf(stderr, "error: batch file is empty\n");
            return EXIT_FAILURE;
        }
        if (!args.batch_jsonl) {
            std::fprintf(stderr, "batch: %zu files\n", wav_paths.size());
        }

        // Load model once.
        struct transcribe_model_params mp = transcribe_model_default_params();
        struct transcribe_model *      model = nullptr;
        const transcribe_status        load_st =
            transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        if (load_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "model load: %s\n",
                         transcribe_status_string(load_st));
            return EXIT_FAILURE;
        }

        // Init context once (reused across all files via run()).
        struct transcribe_context_params cp = transcribe_context_default_params();
        struct transcribe_context *      ctx = nullptr;
        const transcribe_status          init_st =
            transcribe_context_init(model, &cp, &ctx);
        if (init_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "context init: %s\n",
                         transcribe_status_string(init_st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        struct transcribe_params rp = transcribe_default_params();
        if (args.translate)         rp.task     = TRANSCRIBE_TASK_TRANSLATE;
        if (!args.language.empty()) rp.language = args.language.c_str();

        int n_ok = 0;
        int n_fail = 0;
        for (size_t i = 0; i < wav_paths.size(); ++i) {
            const std::string & wav = wav_paths[i];

            // Load wav.
            std::vector<float> pcm;
            std::string        wav_err;
            if (!transcribe_cli::load_wav_mono_16k(wav, pcm, wav_err)) {
                if (args.batch_jsonl) {
                    std::printf("{\"file\":\"%s\",\"text\":\"\","
                                "\"error\":\"wav: %s\"}\n",
                                wav.c_str(), wav_err.c_str());
                } else {
                    std::fprintf(stderr, "SKIP %s: %s\n",
                                 wav.c_str(), wav_err.c_str());
                }
                ++n_fail;
                std::fflush(stdout);
                continue;
            }

            // Run.
            transcribe_status run_st = transcribe_run(
                ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);

            const char * text = "";
            if (run_st == TRANSCRIBE_OK) {
                const char * t = transcribe_full_text(ctx);
                if (t && *t) text = t;
                ++n_ok;
            } else {
                ++n_fail;
            }

            // Output.
            if (args.batch_jsonl) {
                struct transcribe_timings tm = transcribe_get_timings(ctx);
                // Minimal JSON escaping: the only characters in
                // transcribed text that need escaping are quote and
                // backslash. The text is short ASCII in practice.
                std::string escaped;
                for (const char * p = text; *p; ++p) {
                    if (*p == '"')       escaped += "\\\"";
                    else if (*p == '\\') escaped += "\\\\";
                    else                 escaped += *p;
                }
                std::printf("{\"file\":\"%s\",\"text\":\"%s\","
                            "\"mel_ms\":%.1f,\"encode_ms\":%.1f,"
                            "\"decode_ms\":%.1f}\n",
                            wav.c_str(), escaped.c_str(),
                            (double)tm.mel_ms, (double)tm.encode_ms,
                            (double)tm.decode_ms);
            } else {
                std::printf("[%zu/%zu] %s\n", i + 1, wav_paths.size(), wav.c_str());
                std::printf("  text: %s\n", text);
            }
            std::fflush(stdout);
        }

        if (!args.batch_jsonl) {
            std::fprintf(stderr, "batch: %d ok, %d failed out of %zu\n",
                         n_ok, n_fail, wav_paths.size());
        }

        transcribe_context_free(ctx);
        transcribe_model_free(model);
        return n_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    // ---- Single-file mode (original path). -----------------------------
    std::vector<float> pcm;
    std::string        load_err;
    if (!transcribe_cli::load_wav_mono_16k(args.wav_path, pcm, load_err)) {
        std::fprintf(stderr, "wav: %s\n", load_err.c_str());
        return EXIT_FAILURE;
    }

    const double duration_s = static_cast<double>(pcm.size()) / 16000.0;
    std::printf("audio: %s\n", args.wav_path.c_str());
    std::printf("  samples:    %zu\n", pcm.size());
    std::printf("  duration:   %.3f s\n", duration_s);
    std::printf("  sample rate 16000 Hz mono float32\n");

    if (!args.model_path.empty()) {
        struct transcribe_model_params mp = transcribe_model_default_params();
        struct transcribe_model *      model = nullptr;
        const transcribe_status        st =
            transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        std::printf("model: %s -> %s\n",
                    args.model_path.c_str(),
                    transcribe_status_string(st));
        if (st != TRANSCRIBE_OK) {
            return EXIT_FAILURE;
        }
        std::printf("  backend:    %s\n", transcribe_model_backend(model));

        struct transcribe_context_params cp = transcribe_context_default_params();
        struct transcribe_context *      ctx = nullptr;
        const transcribe_status          init_st =
            transcribe_context_init(model, &cp, &ctx);
        if (init_st != TRANSCRIBE_OK) {
            std::fprintf(stderr,
                         "context init: %s\n",
                         transcribe_status_string(init_st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        struct transcribe_params rp = transcribe_default_params();
        if (args.translate)         rp.task     = TRANSCRIBE_TASK_TRANSLATE;
        if (!args.language.empty()) rp.language = args.language.c_str();

        // --repeat N runs transcribe_run() N times for steady-state
        // perf measurements.
        transcribe_status run_st = TRANSCRIBE_OK;
        for (int r = 0; r < args.repeat; ++r) {
            run_st = transcribe_run(
                ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            if (run_st != TRANSCRIBE_OK) break;
        }
        std::printf("run: %s\n", transcribe_status_string(run_st));
        if (run_st == TRANSCRIBE_OK) {
            const char * text = transcribe_full_text(ctx);
            std::printf("text: %s\n", (text && *text) ? text : "(empty)");
        }

        transcribe_print_timings(ctx);

        transcribe_context_free(ctx);
        transcribe_model_free(model);

        if (run_st != TRANSCRIBE_OK) {
            return EXIT_FAILURE;
        }
    } else {
        std::printf("model: (none specified, skipping load)\n");
    }

    return EXIT_SUCCESS;
}
