// transcribe-cli - example CLI driver for transcribe.cpp
//
// Loads a 16 kHz mono float32 WAV (or downmixable to mono), loads a GGUF
// model, and transcribes it through the public ABI. Supports single-file
// and batch modes, offline and streaming paths, and per-family knobs.
// Run with --help for the full option list.

#include "cli.h"
#include "wav.h"

#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using transcribe_cli::cli_args;

namespace {

bool parse_device_index(const char * text, int & out) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    const char * end    = text + std::strlen(text);
    int          parsed = 0;
    const auto   result = std::from_chars(text, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed < 0) {
        return false;
    }
    out = parsed;
    return true;
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [options] audio.wav\n"
                 "       %s [options] --batch file.list -m model.gguf\n"
                 "options:\n"
                 "  -m, --model PATH      GGUF model file\n"
                 "  -l, --language ISO    BCP-47-ish language hint (e.g. en, de)\n"
                 "  -t, --translate       set task to TRANSLATE\n"
                 "  --task T              transcribe, translate or instruct (instruct: --prompt\n"
                 "                        is the instruction; output is free text)\n"
                 "  --vocabulary TERMS    comma-separated custom terms, priority order;\n"
                 "                        repeatable (models with the vocabulary feature)\n"
                 "  --vocabulary-file P   custom terms, one per line\n"
                 "  --prompt TEXT         context text, or the instruction for --task instruct\n"
                 "  --prefix TEXT         transcript text the model continues from\n"
                 "  --target-language ISO target language for translation (e.g. de, es, fr)\n"
                 "  -q, --quiet           suppress library log output\n"
                 "  -r, --repeat N        run N times per file (benchmark)\n"
                 "  -o, --output PATH     write text or speaker segments to PATH (stdout unchanged)\n"
                 "  --threads N           CPU threads (default: all cores)\n"
                 "  --n-ctx N             session context/KV cap in tokens (bounds decoder\n"
                 "                        KV memory; cannot extend the model): 0 = model\n"
                 "                        max, >max is clamped down. Lowers the effective\n"
                 "                        max audio.\n"
                 "  --kv-type TYPE        flash-attn KV type: auto, f32, f16 (default: auto)\n"
                 "  --backend TYPE        compute backend: auto, cpu, cpu_accel, metal, vulkan, cuda, rocm\n"
                 "                        (default: auto)\n"
                 "  --device N            exact device index from --list-devices, including 0\n"
                 "                        (default: automatic device selection)\n"
                 "  --timestamps TYPE     timestamps: auto, none, segment, word, token (default: auto)\n"
                 "  --batch FILE          batch mode: FILE has one wav path per line\n"
                 "  --batch-jsonl         output one JSON line per file (for batch)\n"
                 "  --batch-size N        group N utterances into one transcribe_run_batch\n"
                 "                        call (offline only; 0/1 = per-file serial loop)\n"
                 "  --initial-prompt TEXT (whisper) initial prompt text for context biasing\n"
                 "  --temperature F       (whisper) tier-0 sampling temperature (default 0 = greedy)\n"
                 "  --condition-on-prev-tokens (whisper) carry prev-chunk tokens across chunks\n"
                 "  --prompt-condition T  (whisper) prompt placement: first|all (default: first)\n"
                 "  --itn                 (sensevoice/funasr-nano) enable inverse text\n"
                 "                        normalization (sensevoice: on unless --no-itn)\n"
                 "  --no-itn              (sensevoice/funasr-nano) emit the upstream\n"
                 "                        spoken-form text instead\n"
                 "  --pnc                 (canary) emit punctuation and capitalization (default)\n"
                 "  --no-pnc              (canary) emit lowercase de-punctuated text\n"
                 "  --diarize             (moss/granite-plus) speaker attribution: segments carry\n"
                 "                        speaker ids; granite-plus requests its speaker task\n"
                 "  --no-diarize          disable speaker attribution (the library default)\n"
                 "  --allow CODES         (language ID) comma-separated labels to choose from\n"
                 "  --top N               (language ID) print only the best N ranked\n"
                 "                        candidates (0 = all; default)\n"
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
    const int n = transcribe_device_count();
    if (n <= 0) {
        std::fprintf(stderr, "no compute devices registered\n");
        return EXIT_FAILURE;
    }
    std::printf("%d compute device(s):\n", n);
    for (int i = 0; i < n; ++i) {
        struct transcribe_device_info d;
        transcribe_device_info_init(&d);
        if (transcribe_device_get_info(transcribe_device_get(i), &d) != TRANSCRIBE_OK) {
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
            out.instruct  = false;
        } else if (a == "--task") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            const std::string t = v;
            out.translate       = t == "translate";
            out.instruct        = t == "instruct";
            if (!out.translate && !out.instruct && t != "transcribe") {
                std::fprintf(stderr, "error: --task must be transcribe, translate or instruct\n");
                return false;
            }
        } else if (a == "--vocabulary" || a == "--vocabulary-file") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            std::string text = v;
            char        sep  = ',';
            if (a == "--vocabulary-file") {
                std::ifstream f(v);
                if (!f) {
                    std::fprintf(stderr, "error: cannot read %s\n", v);
                    return false;
                }
                text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                if (text.rfind("\xEF\xBB\xBF", 0) == 0) {
                    text.erase(0, 3);  // UTF-8 byte-order mark
                }
                sep = '\n';
            }
            size_t start = 0;
            while (start <= text.size()) {
                size_t end = text.find(sep, start);
                if (end == std::string::npos) {
                    end = text.size();
                }
                size_t a0 = start, b0 = end;
                while (a0 < b0 && std::isspace(static_cast<unsigned char>(text[a0]))) {
                    ++a0;
                }
                while (b0 > a0 && std::isspace(static_cast<unsigned char>(text[b0 - 1]))) {
                    --b0;
                }
                if (b0 > a0) {
                    out.vocabulary.emplace_back(text.substr(a0, b0 - a0));
                }
                start = end + 1;
            }
        } else if (a == "--prompt") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.prompt = v;
        } else if (a == "--prefix") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.prefix = v;
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
            } else if (vs == "rocm") {
                out.backend = TRANSCRIBE_BACKEND_ROCM;
            } else {
                std::fprintf(stderr, "error: --backend must be auto, cpu, cpu_accel, metal, vulkan, cuda, or rocm\n");
                return false;
            }
        } else if (a == "--device") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            if (!parse_device_index(v, out.device_index)) {
                std::fprintf(stderr, "error: --device must be an integer index >= 0\n");
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
        } else if (a == "-o" || a == "--output") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.output_path = v;
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
        } else if (a == "--no-itn") {
            out.use_itn = false;
            out.itn_set = true;
        } else if (a == "--pnc") {
            out.canary_pnc     = true;
            out.canary_pnc_set = true;
        } else if (a == "--no-pnc") {
            out.canary_pnc     = false;
            out.canary_pnc_set = true;
        } else if (a == "--allow") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            const std::string list = v;
            size_t            pos  = 0;
            while (pos <= list.size()) {
                const size_t comma = list.find(',', pos);
                const size_t end   = comma == std::string::npos ? list.size() : comma;
                if (end > pos) {
                    out.langid_allow.push_back(list.substr(pos, end - pos));
                }
                pos = end + 1;
            }
            if (out.langid_allow.empty()) {
                std::fprintf(stderr, "error: --allow needs at least one label\n");
                return false;
            }
        } else if (a == "--top") {
            const char * v = take_value(a.c_str());
            if (!v) {
                return false;
            }
            out.langid_top = std::atoi(v);
            if (out.langid_top < 0) {
                std::fprintf(stderr, "error: --top must be >= 0\n");
                return false;
            }
        } else if (a == "--diarize") {
            out.diarize     = true;
            out.diarize_set = true;
        } else if (a == "--no-diarize") {
            out.diarize     = false;
            out.diarize_set = true;
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
    if (!out.prefix.empty() && !out.batch_file.empty()) {
        std::fprintf(stderr, "error: --prefix describes one utterance and cannot be combined with --batch\n");
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

// One file: load the audio and (with -m) the model, print both, then hand the
// model to the driver for its role. The driver takes ownership of the model.
int run_file(const cli_args & args, std::ofstream * output) {
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

    if (args.model_path.empty()) {
        std::printf("model: (none specified, skipping load)\n");
        return EXIT_SUCCESS;
    }

    struct transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = args.backend;
    mp.device  = args.device_index >= 0 ? transcribe_device_get(args.device_index) : nullptr;
    if (args.device_index >= 0 && mp.device == nullptr) {
        std::fprintf(stderr, "error: --device index %d is not available\n", args.device_index);
        return EXIT_FAILURE;
    }
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

    const uint32_t roles = transcribe_model_roles(model);
    if ((roles & TRANSCRIBE_ROLE_ASR) != 0) {
        return transcribe_cli::run_asr_file(args, model, pcm, duration_s, output);
    }
    if ((roles & TRANSCRIBE_ROLE_LANGID) != 0) {
        return transcribe_cli::run_langid_file(args, model, pcm, output);
    }
    return transcribe_cli::run_diarize_file(args, model, pcm, duration_s, output);
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

    std::ofstream   output_file;
    std::ofstream * output = nullptr;
    if (!args.output_path.empty()) {
        output_file.open(args.output_path, std::ios::binary | std::ios::trunc);
        if (!output_file) {
            std::fprintf(stderr, "error: cannot open %s for writing\n", args.output_path.c_str());
            return EXIT_FAILURE;
        }
        output = &output_file;
    }

    if (!args.batch_file.empty()) {
        return transcribe_cli::run_asr_batch(args, output);
    }

    return run_file(args, output);
}
