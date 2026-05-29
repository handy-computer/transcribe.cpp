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
#include "transcribe/parakeet.h"
#include "transcribe/whisper.h"
#include "wav.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Minimal JSON string escape: covers the characters MUST be escaped by
// the JSON spec (quote, backslash, control chars). Transcribed text is
// short UTF-8 in practice; we don't need unicode escaping.
std::string json_escape(const char * s) {
    std::string out;
    for (const char * p = s ? s : ""; *p; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

// Build the ",\"segments\":[...]" fragment when the context has real
// segment timestamps. Returns an empty string when the result kind is
// NONE (the library still populates a single dummy entry in that case,
// but with zero timing — don't pollute the JSON with it).
std::string segments_json(const transcribe_session * ctx) {
    if (transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_NONE) {
        return {};
    }
    const int n_seg = transcribe_n_segments(ctx);
    if (n_seg <= 0) return {};
    std::string out = ",\"segments\":[";
    for (int s = 0; s < n_seg; ++s) {
        if (s > 0) out += ",";
        struct transcribe_segment seg; transcribe_segment_init(&seg);
        (void)transcribe_get_segment(ctx, s, &seg);
        char head[96];
        std::snprintf(head, sizeof(head),
                      "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"text\":\"",
                      static_cast<long long>(seg.t0_ms),
                      static_cast<long long>(seg.t1_ms));
        out += head;
        out += json_escape(seg.text != nullptr ? seg.text : "");
        out += "\"}";
    }
    out += "]";
    return out;
}

struct cli_args {
    std::string wav_path;
    std::string model_path;
    std::string language;
    std::string target_language; // --target-language: target lang for translation
    std::string batch_file; // --batch: one wav path per line
    bool        translate = false;
    bool        quiet     = false;
    bool        batch_jsonl = false; // --batch-jsonl: output JSONL
    int         repeat    = 1;
    int         n_threads = 0; // 0 = library default (all cores)
    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;
    transcribe_backend_request backend = TRANSCRIBE_BACKEND_AUTO;
    transcribe_timestamp_kind timestamps = TRANSCRIBE_TIMESTAMPS_NONE;

    // Whisper-family knobs. Ignored for non-Whisper models.
    std::string initial_prompt;                // --initial-prompt TEXT
    bool        whisper_set                  = false;
    bool        condition_on_prev_tokens     = false; // --condition-on-prev-tokens
    enum transcribe_whisper_prompt_condition prompt_condition =
        TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;       // --prompt-condition first|all

    // SenseVoice / FunASR-Nano family knobs. The `--itn` flag is shared:
    // it routes to whichever family the loaded model belongs to. Ignored
    // by non-ITN-aware families.
    bool        use_itn                      = false; // --itn
    bool        itn_set                      = false;
    bool        keep_special_tags            = false; // --raw-tokens

    // Canary family knobs. Ignored by non-Canary families.
    bool        canary_pnc                   = true;  // default: punctuation+caps on
    bool        canary_pnc_set               = false; // --pnc / --no-pnc set this

    // Streaming demo: when > 0, the single-file path feeds the WAV
    // through transcribe_stream_begin/feed/finalize in fixed-size
    // ms-aligned chunks instead of one transcribe_run call. Requires
    // the loaded model to advertise supports_streaming. Set by
    // --stream-chunk-ms N.
    int         stream_chunk_ms              = 0;
    // Parakeet streaming: pick a right-context (lookahead) setting
    // from the model's training menu. -1 = model default (max
    // accuracy / max latency); 0/1/6/13 select the published
    // nemotron-speech-streaming-en-0.6b settings. Set by
    // --stream-att-right N. Ignored when stream_chunk_ms == 0.
    int         stream_att_right             = -1;
    // Parakeet buffered streaming (parakeet-unified-en-0.6b): override
    // the (L, C, R) attention context tuple in milliseconds. -1 = use
    // the model's default (highest-accuracy row of the training menu).
    // Frame-aligned: the lib rounds each value down to the nearest
    // post-subsample frame (80ms at 4x subsampling). Ignored when
    // stream_chunk_ms == 0 or when the model is not buffered-streaming.
    int         stream_buf_left_ms           = -1;
    int         stream_buf_chunk_ms          = -1;
    int         stream_buf_right_ms          = -1;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [options] audio.wav\n"
        "       %s [options] --batch file.list -m model.gguf\n"
        "options:\n"
        "  -m, --model PATH      GGUF model file\n"
        "  -l, --language ISO    BCP-47-ish language hint (e.g. en, de)\n"
        "  -t, --translate       set task to TRANSLATE\n"
        "  --target-language ISO target language for translation (e.g. de, es, fr)\n"
        "  -q, --quiet           suppress library log output\n"
        "  -r, --repeat N        run N times per file (benchmark)\n"
        "  --threads N           CPU threads (default: all cores)\n"
        "  --kv-type TYPE        flash-attn KV type: auto, f32, f16 (default: auto)\n"
        "  --backend TYPE        compute backend: auto, cpu, cpu_accel, metal, vulkan (default: auto)\n"
        "  --timestamps TYPE     timestamps: auto, none, segment, word, token (default: none)\n"
        "  --batch FILE          batch mode: FILE has one wav path per line\n"
        "  --batch-jsonl         output one JSON line per file (for batch)\n"
        "  --initial-prompt TEXT (whisper) initial prompt text for context biasing\n"
        "  --condition-on-prev-tokens (whisper) carry prev-chunk tokens across chunks\n"
        "  --prompt-condition T  (whisper) prompt placement: first|all (default: first)\n"
        "  --itn                 (sensevoice/funasr-nano) enable inverse text normalization\n"
        "  --pnc                 (canary) emit punctuation and capitalization (default)\n"
        "  --no-pnc              (canary) emit lowercase de-punctuated text\n"
        "  --raw-tokens          keep <|...|> control tokens in output text\n"
        "  --stream-chunk-ms N   single-file: drive the streaming API by feeding\n"
        "                        N-ms PCM slices; requires model to advertise\n"
        "                        supports_streaming\n"
        "  --stream-att-right R  (parakeet streaming) pick the right-context\n"
        "                        setting from the model's training menu;\n"
        "                        nemotron-speech-streaming-en-0.6b accepts\n"
        "                        R in {0,1,6,13}; default = model's first choice\n"
        "  --stream-buf-left-ms N  (parakeet-unified buffered streaming)\n"
        "                        left-context size in ms; -1 = model default\n"
        "  --stream-buf-chunk-ms N (parakeet-unified buffered streaming)\n"
        "                        chunk size in ms; -1 = model default\n"
        "  --stream-buf-right-ms N (parakeet-unified buffered streaming)\n"
        "                        right-context (lookahead) size in ms;\n"
        "                        -1 = model default\n"
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
        } else if (a == "--target-language") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.target_language = v;
        } else if (a == "-t" || a == "--translate") {
            out.translate = true;
        } else if (a == "-q" || a == "--quiet") {
            out.quiet = true;
        } else if (a == "-r" || a == "--repeat") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.repeat = std::atoi(v);
            if (out.repeat < 1) out.repeat = 1;
        } else if (a == "--threads") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.n_threads = std::atoi(v);
            if (out.n_threads < 1) out.n_threads = 1;
        } else if (a == "--kv-type") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            const std::string vs = v;
            if      (vs == "auto") out.kv_type = TRANSCRIBE_KV_TYPE_AUTO;
            else if (vs == "f32")  out.kv_type = TRANSCRIBE_KV_TYPE_F32;
            else if (vs == "f16")  out.kv_type = TRANSCRIBE_KV_TYPE_F16;
            else {
                std::fprintf(stderr, "error: --kv-type must be auto, f32, or f16\n");
                return false;
            }
        } else if (a == "--backend") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            const std::string vs = v;
            if      (vs == "auto")      out.backend = TRANSCRIBE_BACKEND_AUTO;
            else if (vs == "cpu")       out.backend = TRANSCRIBE_BACKEND_CPU;
            else if (vs == "cpu_accel") out.backend = TRANSCRIBE_BACKEND_CPU_ACCEL;
            else if (vs == "metal")     out.backend = TRANSCRIBE_BACKEND_METAL;
            else if (vs == "vulkan")    out.backend = TRANSCRIBE_BACKEND_VULKAN;
            else if (vs == "cuda")      out.backend = TRANSCRIBE_BACKEND_CUDA;
            else {
                std::fprintf(stderr, "error: --backend must be auto, cpu, cpu_accel, metal, vulkan, or cuda\n");
                return false;
            }
        } else if (a == "--timestamps") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            const std::string vs = v;
            if      (vs == "auto")    out.timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
            else if (vs == "none")    out.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
            else if (vs == "segment") out.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
            else if (vs == "word")    out.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
            else if (vs == "token")   out.timestamps = TRANSCRIBE_TIMESTAMPS_TOKEN;
            else {
                std::fprintf(stderr,
                             "error: --timestamps must be auto, none, segment, word, or token\n");
                return false;
            }
        } else if (a == "--batch") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.batch_file = v;
        } else if (a == "--batch-jsonl") {
            out.batch_jsonl = true;
        } else if (a == "--initial-prompt") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.initial_prompt = v;
            out.whisper_set    = true;
        } else if (a == "--condition-on-prev-tokens") {
            out.condition_on_prev_tokens = true;
            out.whisper_set              = true;
        } else if (a == "--prompt-condition") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            const std::string vs = v;
            if      (vs == "first") out.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;
            else if (vs == "all")   out.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;
            else {
                std::fprintf(stderr,
                             "error: --prompt-condition must be first or all\n");
                return false;
            }
            out.whisper_set = true;
        } else if (a == "--itn") {
            out.use_itn = true;
            out.itn_set = true;
        } else if (a == "--pnc") {
            out.canary_pnc     = true;
            out.canary_pnc_set = true;
        } else if (a == "--no-pnc") {
            out.canary_pnc     = false;
            out.canary_pnc_set = true;
        } else if (a == "--raw-tokens") {
            out.keep_special_tags = true;
        } else if (a == "--stream-chunk-ms") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.stream_chunk_ms = std::atoi(v);
            if (out.stream_chunk_ms <= 0) {
                std::fprintf(stderr,
                             "error: --stream-chunk-ms must be > 0\n");
                return false;
            }
        } else if (a == "--stream-att-right") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.stream_att_right = std::atoi(v);
            if (out.stream_att_right < 0) {
                std::fprintf(stderr,
                             "error: --stream-att-right must be >= 0\n");
                return false;
            }
        } else if (a == "--stream-buf-left-ms") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.stream_buf_left_ms = std::atoi(v);
        } else if (a == "--stream-buf-chunk-ms") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.stream_buf_chunk_ms = std::atoi(v);
        } else if (a == "--stream-buf-right-ms") {
            const char * v = take_value(a.c_str());
            if (!v) return false;
            out.stream_buf_right_ms = std::atoi(v);
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
    if (out.stream_chunk_ms > 0 && out.repeat > 1) {
        std::fprintf(stderr,
                     "error: --stream-chunk-ms cannot be combined with --repeat\n");
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
        struct transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
        mp.backend = args.backend;
        struct transcribe_model *      model = nullptr;
        const transcribe_status        load_st =
            transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        if (load_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "model load: %s\n",
                         transcribe_status_string(load_st));
            return EXIT_FAILURE;
        }

        // Init context once (reused across all files via run()).
        struct transcribe_session_params cp; transcribe_session_params_init(&cp);
        cp.n_threads = args.n_threads;
        cp.kv_type   = args.kv_type;
        struct transcribe_session *      ctx = nullptr;
        const transcribe_status          init_st =
            transcribe_session_init(model, &cp, &ctx);
        if (init_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "context init: %s\n",
                         transcribe_status_string(init_st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        struct transcribe_run_params rp; transcribe_run_params_init(&rp);
        if (args.translate)         rp.task     = TRANSCRIBE_TASK_TRANSLATE;
        if (!args.language.empty()) rp.language = args.language.c_str();
        if (!args.target_language.empty()) rp.target_language = args.target_language.c_str();
        rp.timestamps = args.timestamps;

        if (args.itn_set) {
            rp.itn = args.use_itn ? TRANSCRIBE_ITN_MODE_ON
                                  : TRANSCRIBE_ITN_MODE_OFF;
        }
        if (args.canary_pnc_set) {
            rp.pnc = args.canary_pnc ? TRANSCRIBE_PNC_MODE_ON
                                     : TRANSCRIBE_PNC_MODE_OFF;
        }

        // Whisper run extension. Allocated outside rp's scope so its
        // bytes outlive the per-file loop below; the library copies
        // initial_prompt/prompt_tokens before transcribe_run returns,
        // but rp aliases &wx.ext for the run call itself.
        struct transcribe_whisper_run_ext wx; transcribe_whisper_run_ext_init(&wx);
        if (args.whisper_set) {
            if (!args.initial_prompt.empty()) {
                wx.initial_prompt = args.initial_prompt.c_str();
            }
            wx.condition_on_prev_tokens = args.condition_on_prev_tokens;
            wx.prompt_condition         = args.prompt_condition;
            if (transcribe_model_accepts_ext_kind(
                    model,
                    TRANSCRIBE_EXT_SLOT_RUN,
                    TRANSCRIBE_EXT_KIND_WHISPER_RUN))
            {
                rp.family = &wx.ext;
            }
        }

        if (args.keep_special_tags) {
            rp.keep_special_tags = true;
        }

        // Emit a batch header line once, before any per-file output. Carries
        // the one-shot load time so downstream WER tooling can record it
        // without parsing stderr. Per-file lines follow on subsequent lines.
        if (args.batch_jsonl) {
            struct transcribe_timings load_tm; transcribe_timings_init(&load_tm);
            (void)transcribe_get_timings(ctx, &load_tm);
            std::printf("{\"type\":\"batch_header\",\"load_ms\":%.1f}\n",
                        (double)load_tm.load_ms);
            std::fflush(stdout);
        }

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

            // Run. When --stream-chunk-ms > 0, drive the streaming API
            // for this utterance (begin/feed/finalize) so the WER
            // harness can measure cache-aware streaming output.
            transcribe_status run_st = TRANSCRIBE_OK;
            if (args.stream_chunk_ms > 0) {
                struct transcribe_stream_params sp; transcribe_stream_params_init(&sp);
                struct transcribe_parakeet_stream_ext pkt_sp;
                transcribe_parakeet_stream_ext_init(&pkt_sp);
                struct transcribe_parakeet_buffered_stream_ext pkt_buf_sp;
                transcribe_parakeet_buffered_stream_ext_init(&pkt_buf_sp);
                const bool want_cache_aware = (args.stream_att_right >= 0);
                const bool want_buffered =
                    args.stream_buf_left_ms  >= 0 ||
                    args.stream_buf_chunk_ms >= 0 ||
                    args.stream_buf_right_ms >= 0;
                if (want_cache_aware && transcribe_model_accepts_ext_kind(
                        model,
                        TRANSCRIBE_EXT_SLOT_STREAM,
                        TRANSCRIBE_EXT_KIND_PARAKEET_STREAM))
                {
                    pkt_sp.att_context_right = args.stream_att_right;
                    sp.family = &pkt_sp.ext;
                } else if (want_buffered && transcribe_model_accepts_ext_kind(
                        model,
                        TRANSCRIBE_EXT_SLOT_STREAM,
                        TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM))
                {
                    pkt_buf_sp.left_ms  = args.stream_buf_left_ms;
                    pkt_buf_sp.chunk_ms = args.stream_buf_chunk_ms;
                    pkt_buf_sp.right_ms = args.stream_buf_right_ms;
                    sp.family = &pkt_buf_sp.ext;
                }
                run_st = transcribe_stream_begin(ctx, &rp, &sp);
                if (run_st == TRANSCRIBE_OK) {
                    const int chunk_samples =
                        std::max(1, args.stream_chunk_ms * 16000 / 1000);
                    size_t pos = 0;
                    while (pos < pcm.size()) {
                        const size_t take = std::min<size_t>(
                            static_cast<size_t>(chunk_samples),
                            pcm.size() - pos);
                        struct transcribe_stream_update upd; transcribe_stream_update_init(&upd);
                        run_st = transcribe_stream_feed(
                            ctx, pcm.data() + pos,
                            static_cast<int>(take), &upd);
                        if (run_st != TRANSCRIBE_OK) break;
                        pos += take;
                    }
                    if (run_st == TRANSCRIBE_OK) {
                        struct transcribe_stream_update fin_upd; transcribe_stream_update_init(&fin_upd);
                        run_st = transcribe_stream_finalize(ctx, &fin_upd);
                    }
                }
            } else {
                run_st = transcribe_run(
                    ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            }

            const char * text = "";
            if (run_st == TRANSCRIBE_OK) {
                const char * t = transcribe_full_text(ctx);
                if (t && *t) text = t;
                ++n_ok;
            } else {
                ++n_fail;
            }

            // Output. When run_st != OK we still emit a JSON line so
            // batch consumers see one record per input wav, but we tag
            // it with the error string. Without this the wav-load error
            // path (above) reports errors but transcribe_run failures
            // (e.g. unsupported language) produce empty hyp_text with no
            // error tag, so downstream tools count them as silent
            // successes.
            if (args.batch_jsonl) {
                struct transcribe_timings tm; transcribe_timings_init(&tm);
                (void)transcribe_get_timings(ctx, &tm);
                const std::string escaped  = json_escape(text);
                const std::string segments = segments_json(ctx);
                std::string err_field;
                if (run_st != TRANSCRIBE_OK) {
                    err_field = ",\"error\":\"";
                    err_field += json_escape(transcribe_status_string(run_st));
                    err_field += "\"";
                }
                std::printf("{\"file\":\"%s\",\"text\":\"%s\"%s,"
                            "\"mel_ms\":%.1f,\"encode_ms\":%.1f,"
                            "\"decode_ms\":%.1f%s}\n",
                            wav.c_str(), escaped.c_str(),
                            segments.c_str(),
                            (double)tm.mel_ms, (double)tm.encode_ms,
                            (double)tm.decode_ms,
                            err_field.c_str());
            } else {
                std::printf("[%zu/%zu] %s", i + 1, wav_paths.size(), wav.c_str());
                if (run_st != TRANSCRIBE_OK) {
                    std::printf("  ERROR: %s\n", transcribe_status_string(run_st));
                } else {
                    std::printf("\n  text: %s\n", text);
                }
            }
            std::fflush(stdout);
        }

        if (!args.batch_jsonl) {
            std::fprintf(stderr, "batch: %d ok, %d failed out of %zu\n",
                         n_ok, n_fail, wav_paths.size());
        }

        transcribe_session_free(ctx);
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
        struct transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
        mp.backend = args.backend;
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

        struct transcribe_session_params cp; transcribe_session_params_init(&cp);
        cp.n_threads = args.n_threads;
        cp.kv_type   = args.kv_type;
        struct transcribe_session *      ctx = nullptr;
        const transcribe_status          init_st =
            transcribe_session_init(model, &cp, &ctx);
        if (init_st != TRANSCRIBE_OK) {
            std::fprintf(stderr,
                         "context init: %s\n",
                         transcribe_status_string(init_st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        struct transcribe_run_params rp; transcribe_run_params_init(&rp);
        if (args.translate)         rp.task     = TRANSCRIBE_TASK_TRANSLATE;
        if (!args.language.empty()) rp.language = args.language.c_str();
        if (!args.target_language.empty()) rp.target_language = args.target_language.c_str();
        rp.timestamps = args.timestamps;

        if (args.itn_set) {
            rp.itn = args.use_itn ? TRANSCRIBE_ITN_MODE_ON
                                  : TRANSCRIBE_ITN_MODE_OFF;
        }
        if (args.canary_pnc_set) {
            rp.pnc = args.canary_pnc ? TRANSCRIBE_PNC_MODE_ON
                                     : TRANSCRIBE_PNC_MODE_OFF;
        }

        struct transcribe_whisper_run_ext wx; transcribe_whisper_run_ext_init(&wx);
        if (args.whisper_set) {
            if (!args.initial_prompt.empty()) {
                wx.initial_prompt = args.initial_prompt.c_str();
            }
            wx.condition_on_prev_tokens = args.condition_on_prev_tokens;
            wx.prompt_condition         = args.prompt_condition;
            if (transcribe_model_accepts_ext_kind(
                    model,
                    TRANSCRIBE_EXT_SLOT_RUN,
                    TRANSCRIBE_EXT_KIND_WHISPER_RUN))
            {
                rp.family = &wx.ext;
            }
        }

        if (args.keep_special_tags) {
            rp.keep_special_tags = true;
        }

        // Streaming demo: drive transcribe_stream_begin/feed/finalize
        // with fixed-size PCM chunks. Families with true per-feed
        // partial decoding (moonshine_streaming) flip
        // update.result_changed whenever the transcript advances; the
        // CLI prints the live tentative text on each such feed.
        // Families that only commit at finalize keep result_changed
        // false until the finalize call.
        transcribe_status run_st = TRANSCRIBE_OK;
        if (args.stream_chunk_ms > 0) {
            struct transcribe_capabilities caps; transcribe_capabilities_init(&caps);
            const transcribe_status caps_st =
                transcribe_model_get_capabilities(model, &caps);
            if (caps_st != TRANSCRIBE_OK || !caps.supports_streaming) {
                std::fprintf(stderr,
                             "stream: model does not advertise "
                             "supports_streaming; use a streaming-capable "
                             "model or drop --stream-chunk-ms\n");
                transcribe_session_free(ctx);
                transcribe_model_free(model);
                return EXIT_FAILURE;
            }

            const int chunk_samples =
                std::max(1, args.stream_chunk_ms * 16000 / 1000);
            std::printf("stream: chunk=%d ms (%d samples)\n",
                        args.stream_chunk_ms, chunk_samples);

            struct transcribe_stream_params sp; transcribe_stream_params_init(&sp);
            struct transcribe_parakeet_stream_ext pkt_sp;
            transcribe_parakeet_stream_ext_init(&pkt_sp);
            struct transcribe_parakeet_buffered_stream_ext pkt_buf_sp;
            transcribe_parakeet_buffered_stream_ext_init(&pkt_buf_sp);
            const bool want_cache_aware = (args.stream_att_right >= 0);
            const bool want_buffered =
                args.stream_buf_left_ms  >= 0 ||
                args.stream_buf_chunk_ms >= 0 ||
                args.stream_buf_right_ms >= 0;
            if (want_cache_aware && transcribe_model_accepts_ext_kind(
                    model,
                    TRANSCRIBE_EXT_SLOT_STREAM,
                    TRANSCRIBE_EXT_KIND_PARAKEET_STREAM))
            {
                pkt_sp.att_context_right = args.stream_att_right;
                sp.family = &pkt_sp.ext;
                std::printf("stream: att_context_right=%d\n",
                            args.stream_att_right);
            } else if (want_buffered && transcribe_model_accepts_ext_kind(
                    model,
                    TRANSCRIBE_EXT_SLOT_STREAM,
                    TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM))
            {
                pkt_buf_sp.left_ms  = args.stream_buf_left_ms;
                pkt_buf_sp.chunk_ms = args.stream_buf_chunk_ms;
                pkt_buf_sp.right_ms = args.stream_buf_right_ms;
                sp.family = &pkt_buf_sp.ext;
                std::printf("stream: buffered (L,C,R)_ms=(%d,%d,%d)\n",
                            args.stream_buf_left_ms,
                            args.stream_buf_chunk_ms,
                            args.stream_buf_right_ms);
            }
            run_st = transcribe_stream_begin(ctx, &rp, &sp);
            if (run_st != TRANSCRIBE_OK) {
                std::fprintf(stderr,
                             "stream_begin: %s\n",
                             transcribe_status_string(run_st));
            } else {
                size_t pos = 0;
                int    feed_n = 0;
                while (pos < pcm.size()) {
                    const size_t take = std::min<size_t>(
                        static_cast<size_t>(chunk_samples),
                        pcm.size() - pos);
                    struct transcribe_stream_update upd; transcribe_stream_update_init(&upd);
                    run_st = transcribe_stream_feed(
                        ctx, pcm.data() + pos,
                        static_cast<int>(take), &upd);
                    if (run_st != TRANSCRIBE_OK) {
                        std::fprintf(stderr,
                                     "stream_feed[%d]: %s\n",
                                     feed_n,
                                     transcribe_status_string(run_st));
                        break;
                    }
                    pos += take;
                    std::printf("  feed[%2d]: input=%lld ms buffered=%lld ms",
                                feed_n,
                                (long long)upd.input_received_ms,
                                (long long)upd.buffered_ms);
                    if (upd.result_changed) {
                        const char * partial = transcribe_full_text(ctx);
                        std::printf("  partial=\"%s\"",
                                    (partial && *partial) ? partial : "");
                    }
                    std::printf("\n");
                    ++feed_n;
                }
                if (run_st == TRANSCRIBE_OK) {
                    struct transcribe_stream_update fin_upd; transcribe_stream_update_init(&fin_upd);
                    run_st = transcribe_stream_finalize(ctx, &fin_upd);
                    std::printf("  finalize: status=%s "
                                "revision=%d input=%lld ms committed=%lld ms\n",
                                transcribe_status_string(run_st),
                                fin_upd.revision,
                                (long long)fin_upd.input_received_ms,
                                (long long)fin_upd.audio_committed_ms);
                }
            }
        } else {
            // --repeat N runs transcribe_run() N times for steady-state
            // perf measurements.
            for (int r = 0; r < args.repeat; ++r) {
                run_st = transcribe_run(
                    ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
                if (run_st != TRANSCRIBE_OK) break;
            }
        }
        std::printf("run: %s\n", transcribe_status_string(run_st));
        if (run_st == TRANSCRIBE_OK) {
            const char * text = transcribe_full_text(ctx);
            std::printf("text: %s\n", (text && *text) ? text : "(empty)");

            const char * dl = transcribe_detected_language(ctx);
            if (dl && *dl) {
                std::printf("detected-language: %s\n", dl);
            }

            const transcribe_timestamp_kind ret_kind =
                transcribe_returned_timestamp_kind(ctx);
            const int n_seg = transcribe_n_segments(ctx);
            if (n_seg > 0 && ret_kind != TRANSCRIBE_TIMESTAMPS_NONE) {
                std::printf("segments: %d\n", n_seg);
                for (int i = 0; i < n_seg; ++i) {
                    struct transcribe_segment seg; transcribe_segment_init(&seg);
                    (void)transcribe_get_segment(ctx, i, &seg);
                    std::printf("  [%7.2f -> %7.2f] %s\n",
                                seg.t0_ms / 1000.0, seg.t1_ms / 1000.0,
                                (seg.text != nullptr) ? seg.text : "");
                }
            }
            if (ret_kind == TRANSCRIBE_TIMESTAMPS_WORD ||
                ret_kind == TRANSCRIBE_TIMESTAMPS_TOKEN) {
                const int n_wrd = transcribe_n_words(ctx);
                std::printf("words: %d\n", n_wrd);
                for (int i = 0; i < n_wrd; ++i) {
                    struct transcribe_word wrd; transcribe_word_init(&wrd);
                    (void)transcribe_get_word(ctx, i, &wrd);
                    std::printf("  [%7.2f -> %7.2f] %s\n",
                                wrd.t0_ms / 1000.0, wrd.t1_ms / 1000.0,
                                (wrd.text != nullptr) ? wrd.text : "");
                }
            }
            if (ret_kind == TRANSCRIBE_TIMESTAMPS_TOKEN) {
                const int n_tok = transcribe_n_tokens(ctx);
                std::printf("tokens: %d\n", n_tok);
                for (int i = 0; i < n_tok; ++i) {
                    struct transcribe_token tok; transcribe_token_init(&tok);
                    (void)transcribe_get_token(ctx, i, &tok);
                    std::printf("  [%7.2f -> %7.2f] p=%.3f %s\n",
                                tok.t0_ms / 1000.0, tok.t1_ms / 1000.0, tok.p,
                                (tok.text != nullptr) ? tok.text : "");
                }
            }
        }

        transcribe_print_timings(ctx);

        {
            struct transcribe_timings tm; transcribe_timings_init(&tm);
            (void)transcribe_get_timings(ctx, &tm);
            const double total_ms = tm.mel_ms + tm.encode_ms + tm.decode_ms;
            if (total_ms > 0.0 && duration_s > 0.0) {
                std::printf("  realtime:   %.0fx (%.1f ms for %.1f s)\n",
                            (duration_s * 1000.0) / total_ms,
                            total_ms, duration_s);
            }
        }

        transcribe_session_free(ctx);
        transcribe_model_free(model);

        if (run_st != TRANSCRIBE_OK) {
            return EXIT_FAILURE;
        }
    } else {
        std::printf("model: (none specified, skipping load)\n");
    }

    return EXIT_SUCCESS;
}
