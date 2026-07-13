// transcribe-cli - example CLI driver for transcribe.cpp
//
// Loads a 16 kHz mono float32 WAV (or downmixable to mono), loads a GGUF
// model, and transcribes it through the public ABI. Supports single-file
// and batch modes, offline and streaming paths, and per-family knobs.
// Run with --help for the full option list.

#include "transcribe.h"
#include "transcribe/parakeet.h"
#include "transcribe/voxtral_realtime.h"
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
        if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20) {
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
    if (n_seg <= 0) {
        return {};
    }
    std::string out = ",\"segments\":[";
    for (int s = 0; s < n_seg; ++s) {
        if (s > 0) {
            out += ",";
        }
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        (void) transcribe_get_segment(ctx, s, &seg);
        char head[96];
        std::snprintf(head, sizeof(head), "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"text\":\"",
                      static_cast<long long>(seg.t0_ms), static_cast<long long>(seg.t1_ms));
        out += head;
        out += json_escape(seg.text != nullptr ? seg.text : "");
        out += "\"}";
    }
    out += "]";
    return out;
}

// Batch variant: segments JSON for utterance `i` of a transcribe_run_batch
// result, using the indexed transcribe_batch_* accessors.
std::string batch_segments_json(const transcribe_session * ctx, int i) {
    if (transcribe_batch_returned_timestamp_kind(ctx, i) == TRANSCRIBE_TIMESTAMPS_NONE) {
        return {};
    }
    const int n_seg = transcribe_batch_n_segments(ctx, i);
    if (n_seg <= 0) {
        return {};
    }
    std::string out = ",\"segments\":[";
    for (int s = 0; s < n_seg; ++s) {
        if (s > 0) {
            out += ",";
        }
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        (void) transcribe_batch_get_segment(ctx, i, s, &seg);
        char head[96];
        std::snprintf(head, sizeof(head), "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"text\":\"",
                      static_cast<long long>(seg.t0_ms), static_cast<long long>(seg.t1_ms));
        out += head;
        out += json_escape(seg.text != nullptr ? seg.text : "");
        out += "\"}";
    }
    out += "]";
    return out;
}

struct cli_args {
    std::string                wav_path;
    std::string                model_path;
    std::string                language;
    std::string                target_language;   // --target-language: target lang for translation
    std::string                batch_file;        // --batch: one wav path per line
    int                        batch_size   = 0;  // --batch-size: >1 groups utterances into
                                                  // transcribe_run_batch calls (offline only).
                                                  // 0/1 keeps the per-file serial loop.
    bool                       translate    = false;
    bool                       quiet        = false;
    bool                       list_devices = false;  // --list-devices: print devices and exit
    bool                       batch_jsonl  = false;  // --batch-jsonl: output JSONL
    int                        repeat       = 1;
    int                        n_threads    = 0;      // 0 = library default (all cores)
    int                        n_ctx        = 0;      // 0 = model's true max; >0 lowers the cap
    transcribe_kv_type         kv_type      = TRANSCRIBE_KV_TYPE_AUTO;
    transcribe_backend_request backend      = TRANSCRIBE_BACKEND_AUTO;
    int                        gpu_device   = 0;  // --device N: 0 = auto, >0 = registry index
    transcribe_timestamp_kind  timestamps   = TRANSCRIBE_TIMESTAMPS_AUTO;

    // Whisper-family knobs. Ignored for non-Whisper models.
    std::string                              initial_prompt;                    // --initial-prompt TEXT
    bool                                     whisper_set              = false;
    bool                                     temperature_set          = false;  // --temperature F
    float                                    temperature              = 0.0f;   // tier-0 sampling temp
    bool                                     condition_on_prev_tokens = false;  // --condition-on-prev-tokens
    enum transcribe_whisper_prompt_condition prompt_condition =
        TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;                                // --prompt-condition first|all

    // SenseVoice / FunASR-Nano family knobs. The `--itn` flag is shared:
    // it routes to whichever family the loaded model belongs to. Ignored
    // by non-ITN-aware families.
    bool use_itn           = false;  // --itn
    bool itn_set           = false;
    bool keep_special_tags = false;  // --raw-tokens

    // Canary family knobs. Ignored by non-Canary families.
    bool canary_pnc     = true;   // default: punctuation+caps on
    bool canary_pnc_set = false;  // --pnc / --no-pnc set this

    // Streaming demo: when > 0, the single-file path feeds the WAV
    // through transcribe_stream_begin/feed/finalize in fixed-size
    // ms-aligned chunks instead of one transcribe_run call. Requires
    // the loaded model to advertise supports_streaming. Set by
    // --stream-chunk-ms N.
    int stream_chunk_ms      = 0;
    // Parakeet streaming: pick a right-context (lookahead) setting
    // from the model's training menu. -1 = model default (max
    // accuracy / max latency); 0/1/6/13 select the published
    // nemotron-speech-streaming-en-0.6b settings. Set by
    // --stream-att-right N. Ignored when stream_chunk_ms == 0.
    int stream_att_right     = -1;
    // Parakeet buffered streaming (parakeet-unified-en-0.6b): override
    // the (L, C, R) attention context tuple in milliseconds. -1 = use
    // the model's default (highest-accuracy row of the training menu).
    // Frame-aligned: the lib rounds each value down to the nearest
    // post-subsample frame (80ms at 4x subsampling). Ignored when
    // stream_chunk_ms == 0 or when the model is not buffered-streaming.
    int stream_buf_left_ms   = -1;
    int stream_buf_chunk_ms  = -1;
    int stream_buf_right_ms  = -1;
    // Voxtral Realtime streaming: transcription delay in 12.5 Hz audio
    // tokens (80 ms each). -1 = model default (6 = 480 ms). Set by
    // --stream-voxtral-delay N. Ignored when stream_chunk_ms == 0 or when
    // the model is not voxtral_realtime.
    int stream_voxtral_delay = -1;
    // Speculative-decode draft length passed through to
    // transcribe_run_params::spec_k_drafts on the offline path. -1 = family
    // default (each family picks its tuned K). 0 = explicitly off. >0 =
    // explicit K. Silently ignored by families without
    // supports_spec_decode. Set by --spec-k-drafts N.
    int spec_k_drafts        = -1;
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
                 "  --n-ctx N             session context/KV cap in tokens (bounds decoder\n"
                 "                        KV memory; cannot extend the model): 0 = model\n"
                 "                        max, >max is clamped down. Lowers the effective\n"
                 "                        max audio.\n"
                 "  --kv-type TYPE        flash-attn KV type: auto, f32, f16 (default: auto)\n"
                 "  --backend TYPE        compute backend: auto, cpu, cpu_accel, metal, vulkan, cuda\n"
                 "                        (default: auto)\n"
                 "  --device N            GPU device index from --list-devices: 0 = auto\n"
                 "                        (first of kind), >0 selects that registry index\n"
                 "  --timestamps TYPE     timestamps: auto, none, segment, word, token (default: auto)\n"
                 "  --batch FILE          batch mode: FILE has one wav path per line\n"
                 "  --batch-jsonl         output one JSON line per file (for batch)\n"
                 "  --batch-size N        group N utterances into one transcribe_run_batch\n"
                 "                        call (offline only; 0/1 = per-file serial loop)\n"
                 "  --initial-prompt TEXT (whisper) initial prompt text for context biasing\n"
                 "  --temperature F       (whisper) tier-0 sampling temperature (default 0 = greedy)\n"
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
                 "  --stream-voxtral-delay N (voxtral_realtime streaming) transcription\n"
                 "                        delay in 12.5 Hz tokens (80 ms each); valid\n"
                 "                        N = 1..15 (80..1200 ms) or 30 (2400 ms);\n"
                 "                        -1 = model default (6 = 480 ms)\n"
                 "  --spec-k-drafts N     speculative-decode draft length on offline\n"
                 "                        autoregressive families that advertise\n"
                 "                        supports_spec_decode. -1 = family default,\n"
                 "                        0 = off, > 0 = explicit K. Silently ignored\n"
                 "                        by families without spec support.\n"
                 "  --list-devices        list registered compute devices (with memory)\n"
                 "                        and exit; ignores all other options\n"
                 "  -h, --help            show this help\n",
                 argv0, argv0);
}

// Print every registered compute device and its live memory, then return an
// exit code. Used by --list-devices. Calls transcribe_init_backends_default()
// first so dynamic-backend builds register their modules (no-op when the
// backends are compiled in).
int list_devices_main() {
    const transcribe_status st = transcribe_init_backends_default();
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "warning: transcribe_init_backends_default() returned %d; "
                     "listing whatever registered\n",
                     (int) st);
    }
    const int n = transcribe_backend_device_count();
    if (n <= 0) {
        std::fprintf(stderr, "no compute devices registered\n");
        return EXIT_FAILURE;
    }
    std::printf("%d compute device(s):\n", n);
    for (int i = 0; i < n; ++i) {
        struct transcribe_backend_device d;
        transcribe_backend_device_init(&d);
        if (transcribe_get_backend_device(i, &d) != TRANSCRIBE_OK) {
            continue;
        }
        const char * type_str = d.device_type == TRANSCRIBE_DEVICE_TYPE_CPU   ? "cpu" :
                                d.device_type == TRANSCRIBE_DEVICE_TYPE_GPU   ? "gpu" :
                                d.device_type == TRANSCRIBE_DEVICE_TYPE_IGPU  ? "igpu" :
                                d.device_type == TRANSCRIBE_DEVICE_TYPE_ACCEL ? "accel" :
                                                                                "?";
        const double gib      = 1024.0 * 1024.0 * 1024.0;
        std::printf("  [%d] %s\n", i, (d.description && *d.description) ? d.description : d.name);
        std::printf("      name=%s  kind=%s  type=%s  id=%s\n", d.name ? d.name : "?", d.kind ? d.kind : "?", type_str,
                    (d.device_id && *d.device_id) ? d.device_id : "(none)");
        std::printf("      memory: %.2f GiB total, %.2f GiB free\n", (double) d.memory_total / gib,
                    (double) d.memory_free / gib);
    }
    return EXIT_SUCCESS;
}

bool parse_args(int argc, char ** argv, cli_args & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string a          = argv[i];
        auto              take_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--list-devices") {
            out.list_devices = true;
        } else if (a == "-m" || a == "--model") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.model_path = v;
        } else if (a == "-l" || a == "--language") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.language = v;
        } else if (a == "--target-language") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.target_language = v;
        } else if (a == "-t" || a == "--translate") {
            out.translate = true;
        } else if (a == "-q" || a == "--quiet") {
            out.quiet = true;
        } else if (a == "-r" || a == "--repeat") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.repeat = std::atoi(v);
            if (out.repeat < 1) {
                out.repeat = 1;
            }
        } else if (a == "--n-ctx") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.n_ctx = std::atoi(v);
            if (out.n_ctx < 0) {
                std::fprintf(stderr, "error: --n-ctx must be >= 0 (0 = model max)\n");
                return false;
            }
        } else if (a == "--threads") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.n_threads = std::atoi(v);
            if (out.n_threads < 1) {
                out.n_threads = 1;
            }
        } else if (a == "--kv-type") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            const std::string vs = v;
            if (vs == "auto") {
                out.kv_type = TRANSCRIBE_KV_TYPE_AUTO;
            } else if (vs == "f32") {
                out.kv_type = TRANSCRIBE_KV_TYPE_F32;
            } else if (vs == "f16") {
                out.kv_type = TRANSCRIBE_KV_TYPE_F16;
            } else {
                std::fprintf(stderr, "error: --kv-type must be auto, f32, or f16\n");
                return false;
            }
        } else if (a == "--backend") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            const std::string vs = v;
            if (vs == "auto") {
                out.backend = TRANSCRIBE_BACKEND_AUTO;
            } else if (vs == "cpu") {
                out.backend = TRANSCRIBE_BACKEND_CPU;
            } else if (vs == "cpu_accel") {
                out.backend = TRANSCRIBE_BACKEND_CPU_ACCEL;
            } else if (vs == "metal") {
                out.backend = TRANSCRIBE_BACKEND_METAL;
            } else if (vs == "vulkan") {
                out.backend = TRANSCRIBE_BACKEND_VULKAN;
            } else if (vs == "cuda") {
                out.backend = TRANSCRIBE_BACKEND_CUDA;
            } else {
                std::fprintf(stderr, "error: --backend must be auto, cpu, cpu_accel, metal, vulkan, or cuda\n");
                return false;
            }
        } else if (a == "--device") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.gpu_device = std::atoi(v);
            if (out.gpu_device < 0) {
                std::fprintf(stderr, "error: --device must be >= 0 (0 = auto)\n");
                return false;
            }
        } else if (a == "--timestamps") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            const std::string vs = v;
            if (vs == "auto") {
                out.timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
            } else if (vs == "none") {
                out.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
            } else if (vs == "segment") {
                out.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
            } else if (vs == "word") {
                out.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
            } else if (vs == "token") {
                out.timestamps = TRANSCRIBE_TIMESTAMPS_TOKEN;
            } else {
                std::fprintf(stderr, "error: --timestamps must be auto, none, segment, word, or token\n");
                return false;
            }
        } else if (a == "--batch") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.batch_file = v;
        } else if (a == "--batch-jsonl") {
            out.batch_jsonl = true;
        } else if (a == "--batch-size") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.batch_size = std::atoi(v);
            if (out.batch_size < 0) {
                std::fprintf(stderr, "error: --batch-size must be >= 0\n");
                return false;
            }
        } else if (a == "--initial-prompt") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.initial_prompt = v;
            out.whisper_set    = true;
        } else if (a == "--temperature") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.temperature     = static_cast<float>(std::atof(v));
            out.temperature_set = true;
            out.whisper_set     = true;
        } else if (a == "--condition-on-prev-tokens") {
            out.condition_on_prev_tokens = true;
            out.whisper_set              = true;
        } else if (a == "--prompt-condition") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            const std::string vs = v;
            if (vs == "first") {
                out.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;
            } else if (vs == "all") {
                out.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;
            } else {
                std::fprintf(stderr, "error: --prompt-condition must be first or all\n");
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
            if (!v) {
                return false;
            }
            out.stream_chunk_ms = std::atoi(v);
            if (out.stream_chunk_ms <= 0) {
                std::fprintf(stderr, "error: --stream-chunk-ms must be > 0\n");
                return false;
            }
        } else if (a == "--stream-att-right") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.stream_att_right = std::atoi(v);
            if (out.stream_att_right < 0) {
                std::fprintf(stderr, "error: --stream-att-right must be >= 0\n");
                return false;
            }
        } else if (a == "--stream-buf-left-ms") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.stream_buf_left_ms = std::atoi(v);
        } else if (a == "--stream-buf-chunk-ms") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.stream_buf_chunk_ms = std::atoi(v);
        } else if (a == "--stream-buf-right-ms") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.stream_buf_right_ms = std::atoi(v);
        } else if (a == "--stream-voxtral-delay") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.stream_voxtral_delay = std::atoi(v);
        } else if (a == "--spec-k-drafts") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.spec_k_drafts = std::atoi(v);
            if (out.spec_k_drafts < -1) {
                std::fprintf(stderr, "error: --spec-k-drafts must be -1 (family default), 0 (off), or > 0\n");
                return false;
            }
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
    // --list-devices is a standalone query handled before any audio is
    // needed, so skip the audio-input requirement for it.
    if (out.list_devices) {
        return true;
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
        std::fprintf(stderr, "error: --stream-chunk-ms cannot be combined with --repeat\n");
        return false;
    }
    return true;
}

void log_cb(transcribe_log_level level, const char * msg, void * userdata) {
    (void) userdata;
    const char * prefix = "[?]";
    switch (level) {
        case TRANSCRIBE_LOG_LEVEL_NONE:
            return;
        case TRANSCRIBE_LOG_LEVEL_INFO:
            prefix = "[info]";
            break;
        case TRANSCRIBE_LOG_LEVEL_WARN:
            prefix = "[warn]";
            break;
        case TRANSCRIBE_LOG_LEVEL_ERROR:
            prefix = "[error]";
            break;
        case TRANSCRIBE_LOG_LEVEL_DEBUG:
            prefix = "[debug]";
            break;
        case TRANSCRIBE_LOG_LEVEL_CONT:
            prefix = "";
            break;
    }
    std::fprintf(stderr, "%s %s%s", prefix, msg, (msg && *msg && msg[std::strlen(msg) - 1] == '\n') ? "" : "\n");
}

}  // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Device listing is a standalone query: no model, no audio. Honor it
    // before any other setup so `--list-devices` works on its own.
    if (args.list_devices) {
        if (!args.quiet) {
            transcribe_log_set(log_cb, nullptr);
        }
        return list_devices_main();
    }

    // Install the log sink ONCE at startup, before any models or contexts
    // exist. This is the only supported usage model in 0.x; see the
    // threading contract in transcribe.h.
    if (!args.quiet) {
        transcribe_log_set(log_cb, nullptr);
    }

    // Batch mode: --batch reads a file list, one wav path per line. Loads
    // the model ONCE and reuses the context across all files. Outputs one
    // JSONL line per file to stdout when --batch-jsonl is set, otherwise
    // the same human-readable format as single-file mode.
    if (!args.batch_file.empty()) {
        if (args.model_path.empty()) {
            std::fprintf(stderr, "error: --batch requires --model\n");
            return EXIT_FAILURE;
        }

        std::vector<std::string> wav_paths;
        {
            std::ifstream fin(args.batch_file);
            if (!fin) {
                std::fprintf(stderr, "error: cannot open batch file %s\n", args.batch_file.c_str());
                return EXIT_FAILURE;
            }
            std::string line;
            while (std::getline(fin, line)) {
                while (!line.empty() &&
                       (line.back() == '\n' || line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
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

        struct transcribe_model_load_params mp;
        transcribe_model_load_params_init(&mp);
        mp.backend                        = args.backend;
        mp.gpu_device                     = args.gpu_device;
        struct transcribe_model * model   = nullptr;
        const transcribe_status   load_st = transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        if (load_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "model load: %s\n", transcribe_status_string(load_st));
            return EXIT_FAILURE;
        }

        struct transcribe_session_params cp;
        transcribe_session_params_init(&cp);
        cp.n_threads                        = args.n_threads;
        cp.n_ctx                            = args.n_ctx;
        cp.kv_type                          = args.kv_type;
        struct transcribe_session * ctx     = nullptr;
        const transcribe_status     init_st = transcribe_session_init(model, &cp, &ctx);
        if (init_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "context init: %s\n", transcribe_status_string(init_st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        struct transcribe_run_params rp;
        transcribe_run_params_init(&rp);
        if (args.translate) {
            rp.task = TRANSCRIBE_TASK_TRANSLATE;
        }
        if (!args.language.empty()) {
            rp.language = args.language.c_str();
        }
        if (!args.target_language.empty()) {
            rp.target_language = args.target_language.c_str();
        }
        rp.timestamps    = args.timestamps;
        rp.spec_k_drafts = args.spec_k_drafts;

        if (args.itn_set) {
            rp.itn = args.use_itn ? TRANSCRIBE_ITN_MODE_ON : TRANSCRIBE_ITN_MODE_OFF;
        }
        if (args.canary_pnc_set) {
            rp.pnc = args.canary_pnc ? TRANSCRIBE_PNC_MODE_ON : TRANSCRIBE_PNC_MODE_OFF;
        }

        // Whisper run extension. Allocated outside rp's scope so its
        // bytes outlive the per-file loop below; the library copies
        // initial_prompt/prompt_tokens before transcribe_run returns,
        // but rp aliases &wx.ext for the run call itself.
        struct transcribe_whisper_run_ext wx;
        transcribe_whisper_run_ext_init(&wx);
        if (args.whisper_set) {
            if (!args.initial_prompt.empty()) {
                wx.initial_prompt = args.initial_prompt.c_str();
            }
            wx.condition_on_prev_tokens = args.condition_on_prev_tokens;
            if (args.temperature_set) {
                wx.temperature = args.temperature;
            }
            wx.prompt_condition = args.prompt_condition;
            if (transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN)) {
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
            struct transcribe_timings load_tm;
            transcribe_timings_init(&load_tm);
            (void) transcribe_get_timings(ctx, &load_tm);
            std::printf("{\"type\":\"batch_header\",\"load_ms\":%.1f}\n", (double) load_tm.load_ms);
            std::fflush(stdout);
        }

        int n_ok        = 0;
        int n_truncated = 0;  // result-bearing: hit the generation cap, partial hyp emitted
        int n_fail      = 0;  // no usable result (wav load / backend / unsupported / whole-batch)

        // Offline batched path: group up to batch_size utterances into one
        // transcribe_run_batch call. Mutually exclusive with the streaming
        // path (--stream-chunk-ms), which is per-utterance by construction.
        // Per-file JSONL output is byte-identical in shape to the serial
        // path so the WER harness (scripts/wer/run.py) consumes either.
        const bool use_batched = args.batch_size > 1 && args.stream_chunk_ms <= 0;
        if (use_batched) {
            const size_t total = wav_paths.size();
            const size_t group = static_cast<size_t>(args.batch_size);
            for (size_t base = 0; base < total; base += group) {
                const size_t end = std::min(total, base + group);

                // Load each wav in the group. A wav that fails to load is
                // emitted as an error line and excluded from the batch, so
                // its failure cannot abort its neighbours.
                std::vector<std::vector<float>> pcms;
                std::vector<const float *>      pcm_ptrs;
                std::vector<int>                n_samps;
                std::vector<size_t>             src_index;  // -> wav_paths
                for (size_t i = base; i < end; ++i) {
                    std::vector<float> pcm;
                    std::string        wav_err;
                    if (!transcribe_cli::load_wav_mono_16k(wav_paths[i], pcm, wav_err)) {
                        if (args.batch_jsonl) {
                            std::printf(
                                "{\"file\":\"%s\",\"text\":\"\","
                                "\"error\":\"wav: %s\"}\n",
                                wav_paths[i].c_str(), wav_err.c_str());
                        } else {
                            std::fprintf(stderr, "SKIP %s: %s\n", wav_paths[i].c_str(), wav_err.c_str());
                        }
                        ++n_fail;
                        std::fflush(stdout);
                        continue;
                    }
                    pcms.push_back(std::move(pcm));
                    src_index.push_back(i);
                }
                for (auto & p : pcms) {
                    pcm_ptrs.push_back(p.data());
                    n_samps.push_back(static_cast<int>(p.size()));
                }
                if (pcms.empty()) {
                    continue;
                }

                const transcribe_status bst =
                    transcribe_run_batch(ctx, pcm_ptrs.data(), n_samps.data(), static_cast<int>(pcm_ptrs.size()), &rp);
                if (bst != TRANSCRIBE_OK && transcribe_batch_n_results(ctx) == 0) {
                    // Whole-batch failure (no per-utterance results): emit an
                    // error line per file in the group and continue.
                    for (size_t k = 0; k < src_index.size(); ++k) {
                        if (args.batch_jsonl) {
                            std::printf(
                                "{\"file\":\"%s\",\"text\":\"\","
                                "\"error\":\"%s\"}\n",
                                wav_paths[src_index[k]].c_str(), json_escape(transcribe_status_string(bst)).c_str());
                        }
                        ++n_fail;
                    }
                    std::fflush(stdout);
                    continue;
                }

                for (size_t k = 0; k < src_index.size(); ++k) {
                    const std::string &     wav = wav_paths[src_index[k]];
                    const transcribe_status ust = transcribe_batch_status(ctx, static_cast<int>(k));
                    // OUTPUT_TRUNCATED is result-bearing: the partial transcript is
                    // preserved and readable via transcribe_batch_full_text (see
                    // transcribe.h). Emit it as the hyp so downstream tooling scores
                    // the partial rather than an empty string; the error field below
                    // still tags it so the truncation stays visible.
                    const bool   result_present = ust == TRANSCRIBE_OK || ust == TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
                    const char * text           = "";
                    if (result_present) {
                        const char * t = transcribe_batch_full_text(ctx, static_cast<int>(k));
                        if (t && *t) {
                            text = t;
                        }
                    }
                    if (ust == TRANSCRIBE_OK) {
                        ++n_ok;
                    } else if (ust == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
                        ++n_truncated;
                    } else {
                        ++n_fail;
                    }
                    if (args.batch_jsonl) {
                        const std::string escaped  = json_escape(text);
                        const std::string segments = batch_segments_json(ctx, static_cast<int>(k));
                        std::string       err_field;
                        if (ust != TRANSCRIBE_OK) {
                            err_field = ",\"error\":\"";
                            err_field += json_escape(transcribe_status_string(ust));
                            err_field += "\"";
                        }
                        // Per-utterance timings: the batched encoder time is
                        // amortized across the batch, decode is real per-utt,
                        // so summing across utterances gives the true encode vs
                        // decode split for the whole batched run.
                        struct transcribe_timings tm;
                        transcribe_timings_init(&tm);
                        (void) transcribe_batch_get_timings(ctx, static_cast<int>(k), &tm);
                        std::printf(
                            "{\"file\":\"%s\",\"text\":\"%s\"%s,"
                            "\"mel_ms\":%.1f,\"encode_ms\":%.1f,"
                            "\"decode_ms\":%.1f%s}\n",
                            wav.c_str(), escaped.c_str(), segments.c_str(), (double) tm.mel_ms, (double) tm.encode_ms,
                            (double) tm.decode_ms, err_field.c_str());
                    } else {
                        std::printf("[%zu/%zu] %s", src_index[k] + 1, total, wav.c_str());
                        if (ust == TRANSCRIBE_OK) {
                            std::printf("\n  text: %s\n", text);
                        } else if (ust == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
                            std::printf("  (truncated)\n  text: %s\n", text);
                        } else {
                            std::printf("  ERROR: %s\n", transcribe_status_string(ust));
                        }
                    }
                    std::fflush(stdout);
                }
            }
        } else {
            for (size_t i = 0; i < wav_paths.size(); ++i) {
                const std::string & wav = wav_paths[i];

                std::vector<float> pcm;
                std::string        wav_err;
                if (!transcribe_cli::load_wav_mono_16k(wav, pcm, wav_err)) {
                    if (args.batch_jsonl) {
                        std::printf(
                            "{\"file\":\"%s\",\"text\":\"\","
                            "\"error\":\"wav: %s\"}\n",
                            wav.c_str(), wav_err.c_str());
                    } else {
                        std::fprintf(stderr, "SKIP %s: %s\n", wav.c_str(), wav_err.c_str());
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
                    struct transcribe_stream_params sp;
                    transcribe_stream_params_init(&sp);
                    struct transcribe_parakeet_stream_ext pkt_sp;
                    transcribe_parakeet_stream_ext_init(&pkt_sp);
                    struct transcribe_parakeet_buffered_stream_ext pkt_buf_sp;
                    transcribe_parakeet_buffered_stream_ext_init(&pkt_buf_sp);
                    struct transcribe_voxtral_realtime_stream_ext vx_sp;
                    transcribe_voxtral_realtime_stream_ext_init(&vx_sp);
                    const bool want_cache_aware = (args.stream_att_right >= 0);
                    const bool want_buffered =
                        args.stream_buf_left_ms >= 0 || args.stream_buf_chunk_ms >= 0 || args.stream_buf_right_ms >= 0;
                    if (want_cache_aware && transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                                              TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
                        pkt_sp.att_context_right = args.stream_att_right;
                        sp.family                = &pkt_sp.ext;
                    } else if (want_buffered &&
                               transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                                 TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM)) {
                        pkt_buf_sp.left_ms  = args.stream_buf_left_ms;
                        pkt_buf_sp.chunk_ms = args.stream_buf_chunk_ms;
                        pkt_buf_sp.right_ms = args.stream_buf_right_ms;
                        sp.family           = &pkt_buf_sp.ext;
                    } else if (args.stream_voxtral_delay != -1 &&
                               transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                                 TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM)) {
                        vx_sp.num_delay_tokens = args.stream_voxtral_delay;
                        sp.family              = &vx_sp.ext;
                    }
                    run_st = transcribe_stream_begin(ctx, &rp, &sp);
                    if (run_st == TRANSCRIBE_OK) {
                        const int chunk_samples = std::max(1, args.stream_chunk_ms * 16000 / 1000);
                        size_t    pos           = 0;
                        while (pos < pcm.size()) {
                            const size_t take = std::min<size_t>(static_cast<size_t>(chunk_samples), pcm.size() - pos);
                            struct transcribe_stream_update upd;
                            transcribe_stream_update_init(&upd);
                            run_st = transcribe_stream_feed(ctx, pcm.data() + pos, static_cast<int>(take), &upd);
                            if (run_st != TRANSCRIBE_OK) {
                                break;
                            }
                            pos += take;
                        }
                        if (run_st == TRANSCRIBE_OK) {
                            struct transcribe_stream_update fin_upd;
                            transcribe_stream_update_init(&fin_upd);
                            run_st = transcribe_stream_finalize(ctx, &fin_upd);
                        }
                    }
                } else {
                    run_st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
                }

                // OUTPUT_TRUNCATED is result-bearing: the partial transcript is
                // preserved and readable via transcribe_full_text (see transcribe.h).
                // Emit it as the hyp so downstream tooling scores the partial rather
                // than an empty string; the error field below still tags it so the
                // truncation stays visible.
                const bool   result_present = run_st == TRANSCRIBE_OK || run_st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
                const char * text           = "";
                if (result_present) {
                    const char * t = transcribe_full_text(ctx);
                    if (t && *t) {
                        text = t;
                    }
                }
                if (run_st == TRANSCRIBE_OK) {
                    ++n_ok;
                } else if (run_st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
                    ++n_truncated;
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
                    struct transcribe_timings tm;
                    transcribe_timings_init(&tm);
                    (void) transcribe_get_timings(ctx, &tm);
                    const std::string escaped  = json_escape(text);
                    const std::string segments = segments_json(ctx);
                    std::string       err_field;
                    if (run_st != TRANSCRIBE_OK) {
                        err_field = ",\"error\":\"";
                        err_field += json_escape(transcribe_status_string(run_st));
                        err_field += "\"";
                    }
                    std::printf(
                        "{\"file\":\"%s\",\"text\":\"%s\"%s,"
                        "\"mel_ms\":%.1f,\"encode_ms\":%.1f,"
                        "\"decode_ms\":%.1f%s}\n",
                        wav.c_str(), escaped.c_str(), segments.c_str(), (double) tm.mel_ms, (double) tm.encode_ms,
                        (double) tm.decode_ms, err_field.c_str());
                } else {
                    std::printf("[%zu/%zu] %s", i + 1, wav_paths.size(), wav.c_str());
                    if (run_st == TRANSCRIBE_OK) {
                        std::printf("\n  text: %s\n", text);
                    } else if (run_st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
                        std::printf("  (truncated)\n  text: %s\n", text);
                    } else {
                        std::printf("  ERROR: %s\n", transcribe_status_string(run_st));
                    }
                }
                std::fflush(stdout);
            }
        }

        if (!args.batch_jsonl) {
            std::fprintf(stderr, "batch: %d ok, %d truncated, %d failed out of %zu\n", n_ok, n_truncated, n_fail,
                         wav_paths.size());
        }

        transcribe_session_free(ctx);
        transcribe_model_free(model);
        // OUTPUT_TRUNCATED is result-bearing and does not fail the batch, but
        // hard per-utterance failures must remain visible to automation.
        return n_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    // Single-file mode.
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
        struct transcribe_model_load_params mp;
        transcribe_model_load_params_init(&mp);
        mp.backend                      = args.backend;
        mp.gpu_device                   = args.gpu_device;
        struct transcribe_model * model = nullptr;
        const transcribe_status   st    = transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        std::printf("model: %s -> %s\n", args.model_path.c_str(), transcribe_status_string(st));
        if (st != TRANSCRIBE_OK) {
            return EXIT_FAILURE;
        }
        std::printf("  backend:    %s\n", transcribe_model_backend(model));
        if (const char * dn = transcribe_model_meta_val_str(model, "general.name"); dn[0]) {
            std::printf("  name:       %s\n", dn);
        }
        if (const char * lic = transcribe_model_meta_val_str(model, "general.license"); lic[0]) {
            std::printf("  license:    %s\n", lic);
        }

        struct transcribe_session_params cp;
        transcribe_session_params_init(&cp);
        cp.n_threads                        = args.n_threads;
        cp.n_ctx                            = args.n_ctx;
        cp.kv_type                          = args.kv_type;
        struct transcribe_session * ctx     = nullptr;
        const transcribe_status     init_st = transcribe_session_init(model, &cp, &ctx);
        if (init_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "context init: %s\n", transcribe_status_string(init_st));
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        // Surface the effective input-length limit so it's obvious how much
        // audio this session accepts (reflects --n-ctx). 0 means "no practical
        // limit" — the family chunks internally or is unbounded. See
        // docs/input-limits.md.
        {
            struct transcribe_session_limits lim;
            transcribe_session_limits_init(&lim);
            if (transcribe_session_get_limits(ctx, &lim) == TRANSCRIBE_OK) {
                if (lim.effective_max_audio_ms > 0) {
                    std::printf("  max audio:  %.1f s", (double) lim.effective_max_audio_ms / 1000.0);
                    if (lim.effective_n_ctx > 0) {
                        std::printf("  (context %d tok, ~%lld MiB KV max)", lim.effective_n_ctx,
                                    (long long) (lim.max_kv_bytes >> 20));
                    }
                    std::printf("\n");
                } else if (lim.effective_n_ctx > 0) {
                    // Capped family whose context is too small to fit any audio
                    // plus a prompt (e.g. an aggressively low --n-ctx).
                    std::printf(
                        "  max audio:  ~0 s (context %d tok too small for "
                        "audio + prompt)\n",
                        lim.effective_n_ctx);
                } else {
                    std::printf("  max audio:  unbounded (long audio chunked internally)\n");
                }
            }
        }

        struct transcribe_run_params rp;
        transcribe_run_params_init(&rp);
        if (args.translate) {
            rp.task = TRANSCRIBE_TASK_TRANSLATE;
        }
        if (!args.language.empty()) {
            rp.language = args.language.c_str();
        }
        if (!args.target_language.empty()) {
            rp.target_language = args.target_language.c_str();
        }
        rp.timestamps    = args.timestamps;
        rp.spec_k_drafts = args.spec_k_drafts;

        if (args.itn_set) {
            rp.itn = args.use_itn ? TRANSCRIBE_ITN_MODE_ON : TRANSCRIBE_ITN_MODE_OFF;
        }
        if (args.canary_pnc_set) {
            rp.pnc = args.canary_pnc ? TRANSCRIBE_PNC_MODE_ON : TRANSCRIBE_PNC_MODE_OFF;
        }

        struct transcribe_whisper_run_ext wx;
        transcribe_whisper_run_ext_init(&wx);
        if (args.whisper_set) {
            if (!args.initial_prompt.empty()) {
                wx.initial_prompt = args.initial_prompt.c_str();
            }
            wx.condition_on_prev_tokens = args.condition_on_prev_tokens;
            if (args.temperature_set) {
                wx.temperature = args.temperature;
            }
            wx.prompt_condition = args.prompt_condition;
            if (transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN)) {
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
            struct transcribe_capabilities caps;
            transcribe_capabilities_init(&caps);
            const transcribe_status caps_st = transcribe_model_get_capabilities(model, &caps);
            if (caps_st != TRANSCRIBE_OK || !caps.supports_streaming) {
                std::fprintf(stderr,
                             "stream: model does not advertise "
                             "supports_streaming; use a streaming-capable "
                             "model or drop --stream-chunk-ms\n");
                transcribe_session_free(ctx);
                transcribe_model_free(model);
                return EXIT_FAILURE;
            }

            const int chunk_samples = std::max(1, args.stream_chunk_ms * 16000 / 1000);
            std::printf("stream: chunk=%d ms (%d samples)\n", args.stream_chunk_ms, chunk_samples);

            struct transcribe_stream_params sp;
            transcribe_stream_params_init(&sp);
            struct transcribe_parakeet_stream_ext pkt_sp;
            transcribe_parakeet_stream_ext_init(&pkt_sp);
            struct transcribe_parakeet_buffered_stream_ext pkt_buf_sp;
            transcribe_parakeet_buffered_stream_ext_init(&pkt_buf_sp);
            struct transcribe_voxtral_realtime_stream_ext vx_sp;
            transcribe_voxtral_realtime_stream_ext_init(&vx_sp);
            const bool want_cache_aware = (args.stream_att_right >= 0);
            const bool want_buffered =
                args.stream_buf_left_ms >= 0 || args.stream_buf_chunk_ms >= 0 || args.stream_buf_right_ms >= 0;
            if (want_cache_aware && transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                                      TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
                pkt_sp.att_context_right = args.stream_att_right;
                sp.family                = &pkt_sp.ext;
                std::printf("stream: att_context_right=%d\n", args.stream_att_right);
            } else if (want_buffered &&
                       transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                         TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM)) {
                pkt_buf_sp.left_ms  = args.stream_buf_left_ms;
                pkt_buf_sp.chunk_ms = args.stream_buf_chunk_ms;
                pkt_buf_sp.right_ms = args.stream_buf_right_ms;
                sp.family           = &pkt_buf_sp.ext;
                std::printf("stream: buffered (L,C,R)_ms=(%d,%d,%d)\n", args.stream_buf_left_ms,
                            args.stream_buf_chunk_ms, args.stream_buf_right_ms);
            } else if (args.stream_voxtral_delay != -1 &&
                       transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                         TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM)) {
                vx_sp.num_delay_tokens = args.stream_voxtral_delay;
                sp.family              = &vx_sp.ext;
                std::printf("stream: voxtral num_delay_tokens=%d\n", args.stream_voxtral_delay);
            }
            run_st = transcribe_stream_begin(ctx, &rp, &sp);
            if (run_st != TRANSCRIBE_OK) {
                std::fprintf(stderr, "stream_begin: %s\n", transcribe_status_string(run_st));
            } else {
                size_t pos    = 0;
                int    feed_n = 0;
                while (pos < pcm.size()) {
                    const size_t take = std::min<size_t>(static_cast<size_t>(chunk_samples), pcm.size() - pos);
                    struct transcribe_stream_update upd;
                    transcribe_stream_update_init(&upd);
                    run_st = transcribe_stream_feed(ctx, pcm.data() + pos, static_cast<int>(take), &upd);
                    if (run_st != TRANSCRIBE_OK) {
                        std::fprintf(stderr, "stream_feed[%d]: %s\n", feed_n, transcribe_status_string(run_st));
                        break;
                    }
                    pos += take;
                    std::printf("  feed[%2d]: input=%lld ms buffered=%lld ms", feed_n,
                                (long long) upd.input_received_ms, (long long) upd.buffered_ms);
                    if (upd.result_changed) {
                        const char * partial = transcribe_full_text(ctx);
                        std::printf("  partial=\"%s\"", (partial && *partial) ? partial : "");
                    }
                    std::printf("\n");
                    ++feed_n;
                }
                if (run_st == TRANSCRIBE_OK) {
                    struct transcribe_stream_update fin_upd;
                    transcribe_stream_update_init(&fin_upd);
                    run_st = transcribe_stream_finalize(ctx, &fin_upd);
                    std::printf(
                        "  finalize: status=%s "
                        "revision=%d input=%lld ms committed=%lld ms\n",
                        transcribe_status_string(run_st), fin_upd.revision, (long long) fin_upd.input_received_ms,
                        (long long) fin_upd.audio_committed_ms);
                }
            }
        } else {
            // --repeat N runs transcribe_run() N times for steady-state
            // perf measurements.
            for (int r = 0; r < args.repeat; ++r) {
                run_st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
                if (run_st != TRANSCRIBE_OK) {
                    break;
                }
            }
        }
        std::printf("run: %s\n", transcribe_status_string(run_st));
        // OUTPUT_TRUNCATED and ABORTED are non-OK but preserve the partial
        // transcript (see docs/input-limits.md), so show the result for them
        // too — just flagged.
        const bool result_present =
            run_st == TRANSCRIBE_OK || run_st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED || run_st == TRANSCRIBE_ERR_ABORTED;
        if (result_present) {
            const char * text = transcribe_full_text(ctx);
            std::printf("text: %s\n", (text && *text) ? text : "(empty)");

            // A truncated decode hit the model's context/output budget before
            // end-of-stream; the text above is incomplete.
            if (transcribe_was_truncated(ctx)) {
                std::printf(
                    "  note:      output truncated (hit the model's "
                    "context/generation cap before end-of-stream); "
                    "transcript is incomplete\n");
            }

            const char * dl = transcribe_detected_language(ctx);
            if (dl && *dl) {
                std::printf("detected-language: %s\n", dl);
            }

            const transcribe_timestamp_kind ret_kind = transcribe_returned_timestamp_kind(ctx);
            const int                       n_seg    = transcribe_n_segments(ctx);
            if (n_seg > 0 && ret_kind != TRANSCRIBE_TIMESTAMPS_NONE) {
                std::printf("segments: %d\n", n_seg);
                for (int i = 0; i < n_seg; ++i) {
                    struct transcribe_segment seg;
                    transcribe_segment_init(&seg);
                    (void) transcribe_get_segment(ctx, i, &seg);
                    std::printf("  [%7.2f -> %7.2f] %s\n", seg.t0_ms / 1000.0, seg.t1_ms / 1000.0,
                                (seg.text != nullptr) ? seg.text : "");
                }
            }
            if (ret_kind == TRANSCRIBE_TIMESTAMPS_WORD || ret_kind == TRANSCRIBE_TIMESTAMPS_TOKEN) {
                const int n_wrd = transcribe_n_words(ctx);
                std::printf("words: %d\n", n_wrd);
                for (int i = 0; i < n_wrd; ++i) {
                    struct transcribe_word wrd;
                    transcribe_word_init(&wrd);
                    (void) transcribe_get_word(ctx, i, &wrd);
                    std::printf("  [%7.2f -> %7.2f] %s\n", wrd.t0_ms / 1000.0, wrd.t1_ms / 1000.0,
                                (wrd.text != nullptr) ? wrd.text : "");
                }
            }
            if (ret_kind == TRANSCRIBE_TIMESTAMPS_TOKEN) {
                const int n_tok = transcribe_n_tokens(ctx);
                std::printf("tokens: %d\n", n_tok);
                for (int i = 0; i < n_tok; ++i) {
                    struct transcribe_token tok;
                    transcribe_token_init(&tok);
                    (void) transcribe_get_token(ctx, i, &tok);
                    std::printf("  [%7.2f -> %7.2f] p=%.3f %s\n", tok.t0_ms / 1000.0, tok.t1_ms / 1000.0, tok.p,
                                (tok.text != nullptr) ? tok.text : "");
                }
            }
        }

        transcribe_print_timings(ctx);

        {
            struct transcribe_timings tm;
            transcribe_timings_init(&tm);
            (void) transcribe_get_timings(ctx, &tm);
            const double total_ms = tm.mel_ms + tm.encode_ms + tm.decode_ms;
            if (total_ms > 0.0 && duration_s > 0.0) {
                std::printf("  realtime:   %.0fx (%.1f ms for %.1f s)\n", (duration_s * 1000.0) / total_ms, total_ms,
                            duration_s);
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
