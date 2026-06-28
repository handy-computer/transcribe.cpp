// transcribe-bench - one-shot perf measurement for a (model, sample) pair.
//
// Developer-loop tool. Loads a model once, runs --warmup untimed
// iterations, then --iters measured iterations, and emits a JSON
// document that a Python driver aggregates across a matrix of
// models and samples. Progress lines go to stderr; the JSON goes
// to --json-out (or stdout if omitted).
//
// Schema version: "transcribe-bench-v2". Reports "rtf_wall_mean"
// (wall-clock RTF, the user-visible number) and "rtf_compute_mean"
// (phase-timer sum RTF). The Python driver accepts both v1 and v2 schemas.

#include "transcribe.h"
#include "wav.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct bench_args {
    std::string                model_path;
    std::string                sample_path;
    std::string                json_out;
    int                        iters     = 2;
    int                        warmup    = 1;
    int                        n_threads = 0;
    bool                       quiet     = false;
    transcribe_backend_request backend   = TRANSCRIBE_BACKEND_AUTO;
    int                        gpu_device = 0;  // --device N: 0 = auto, >0 = index
    // Passed through to transcribe_run_params::spec_k_drafts. -1 = family
    // default, 0 = spec decode off, > 0 = explicit draft length. Silently
    // ignored by families without supports_spec_decode. Set by
    // --spec-k-drafts N.
    int                        spec_k_drafts = -1;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model PATH --sample PATH [options]\n"
        "options:\n"
        "  --model PATH       GGUF model file (required)\n"
        "  --sample PATH      WAV file, 16 kHz mono (required)\n"
        "  --iters N          measured iterations (default 2)\n"
        "  --warmup N         untimed warmup iterations (default 1)\n"
        "  --json-out PATH    write result JSON here (default: stdout)\n"
        "  --quiet            suppress progress lines on stderr\n"
        "  --threads N        CPU threads (default 0 = library default)\n"
        "  --backend KIND     request a specific backend:\n"
        "                       auto|cpu|cpu_accel|metal|vulkan|cuda (default auto)\n"
        "                     cpu is strict CPU (no GPU, no BLAS/AMX).\n"
        "                     cpu_accel is CPU + host-memory accelerators\n"
        "                       (BLAS/AMX) when the build includes them.\n"
        "  --device N         GPU device index: 0 = auto (first of kind),\n"
        "                       >0 selects that ggml registry index\n"
        "  --spec-k-drafts N  speculative-decode draft length on the offline\n"
        "                     path: -1 = family default, 0 = off, > 0 = K.\n"
        "                     Ignored by families without spec support.\n"
        "  -h, --help         show this help\n",
        argv0);
}

// Parse --backend KIND into the public enum. Returns false and logs a
// diagnostic on an unknown value.
bool parse_backend_kind(const char *                   s,
                        transcribe_backend_request &   out)
{
    if (s == nullptr) return false;
    if (std::strcmp(s, "auto")      == 0) { out = TRANSCRIBE_BACKEND_AUTO;      return true; }
    if (std::strcmp(s, "cpu")       == 0) { out = TRANSCRIBE_BACKEND_CPU;       return true; }
    if (std::strcmp(s, "cpu_accel") == 0) { out = TRANSCRIBE_BACKEND_CPU_ACCEL; return true; }
    if (std::strcmp(s, "metal")     == 0) { out = TRANSCRIBE_BACKEND_METAL;     return true; }
    if (std::strcmp(s, "vulkan")    == 0) { out = TRANSCRIBE_BACKEND_VULKAN;    return true; }
    if (std::strcmp(s, "cuda")      == 0) { out = TRANSCRIBE_BACKEND_CUDA;      return true; }
    std::fprintf(stderr,
        "error: --backend value '%s' not recognized "
        "(expected one of: auto, cpu, cpu_accel, metal, vulkan, cuda)\n", s);
    return false;
}

bool parse_args(int argc, char ** argv, bench_args & out) {
    auto need_val = [&](int & i, const char * name) -> const char * {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s requires an argument\n", name);
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--model")    { auto v = need_val(i, "--model");    if (!v) return false; out.model_path  = v; }
        else if (a == "--sample")   { auto v = need_val(i, "--sample");   if (!v) return false; out.sample_path = v; }
        else if (a == "--json-out") { auto v = need_val(i, "--json-out"); if (!v) return false; out.json_out    = v; }
        else if (a == "--iters")    { auto v = need_val(i, "--iters");    if (!v) return false; out.iters     = std::atoi(v); if (out.iters  < 1) out.iters  = 1; }
        else if (a == "--warmup")   { auto v = need_val(i, "--warmup");   if (!v) return false; out.warmup    = std::atoi(v); if (out.warmup < 0) out.warmup = 0; }
        else if (a == "--threads")  { auto v = need_val(i, "--threads");  if (!v) return false; out.n_threads = std::atoi(v); if (out.n_threads < 0) out.n_threads = 0; }
        else if (a == "--spec-k-drafts") { auto v = need_val(i, "--spec-k-drafts"); if (!v) return false; out.spec_k_drafts = std::atoi(v); if (out.spec_k_drafts < -1) { std::fprintf(stderr, "error: --spec-k-drafts must be -1 (family default), 0 (off), or > 0\n"); return false; } }
        else if (a == "--quiet")    { out.quiet = true; }
        else if (a == "--backend")  {
            auto v = need_val(i, "--backend"); if (!v) return false;
            if (!parse_backend_kind(v, out.backend)) return false;
        }
        else if (a == "--device")   {
            auto v = need_val(i, "--device"); if (!v) return false;
            out.gpu_device = std::atoi(v);
            if (out.gpu_device < 0) { std::fprintf(stderr, "error: --device must be >= 0 (0 = auto)\n"); return false; }
        }
        else {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return false;
        }
    }
    if (out.model_path.empty() || out.sample_path.empty()) {
        std::fprintf(stderr, "error: --model and --sample are required\n");
        return false;
    }
    return true;
}

// Minimal JSON string escaping - inputs are paths and a backend
// name, only backslash and double-quote need handling.
std::string json_escape(const char * s) {
    std::string o;
    if (!s) return o;
    for (const char * p = s; *p; ++p) {
        if      (*p == '"')  o += "\\\"";
        else if (*p == '\\') o += "\\\\";
        else                 o += *p;
    }
    return o;
}

struct iter_timings {
    double mel_ms, encode_ms, decode_ms, total_ms, wall_ms;
};

struct stat_summary { double min_v, max_v, mean_v; };

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void append_fmt(std::string & out, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    char buf[256];
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return; }
    if (static_cast<size_t>(n) < sizeof(buf)) {
        out.append(buf, static_cast<size_t>(n));
    } else {
        // vsnprintf reports the would-be length; fall back to a heap
        // buffer so long fields (e.g. hyp_text) aren't truncated and
        // we don't read past the stack buffer.
        std::string big(static_cast<size_t>(n) + 1, '\0');
        std::vsnprintf(big.data(), big.size(), fmt, ap2);
        out.append(big.data(), static_cast<size_t>(n));
    }
    va_end(ap2);
}

stat_summary compute_stats(const std::vector<iter_timings> & iters,
                           double (*pick)(const iter_timings &)) {
    stat_summary s{0.0, 0.0, 0.0};
    if (iters.empty()) return s;
    s.min_v = s.max_v = pick(iters[0]);
    double sum = 0.0;
    for (const auto & it : iters) {
        const double v = pick(it);
        if (v < s.min_v) s.min_v = v;
        if (v > s.max_v) s.max_v = v;
        sum += v;
    }
    s.mean_v = sum / static_cast<double>(iters.size());
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    bench_args args;
    if (!parse_args(argc, argv, args)) { print_usage(argv[0]); return EXIT_FAILURE; }

    if (args.quiet) transcribe_log_set(nullptr, nullptr);
    const bool quiet = args.quiet;

    if (!quiet) std::fprintf(stderr, "loading sample %s\n", args.sample_path.c_str());
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(args.sample_path, pcm, wav_err)) {
        std::fprintf(stderr, "error: wav: %s\n", wav_err.c_str());
        return EXIT_FAILURE;
    }
    const double sample_duration_s = static_cast<double>(pcm.size()) / 16000.0;

    // Load model. --backend propagates directly to
    // transcribe_model_load_params::backend, which the per-family loader
    // consumes via transcribe::load_common::init_backends.
    if (!quiet) std::fprintf(stderr, "loading model %s\n", args.model_path.c_str());
    struct transcribe_model_load_params mp; transcribe_model_load_params_init(&mp);
    mp.backend = args.backend;
    mp.gpu_device = args.gpu_device;
    struct transcribe_model *      model = nullptr;
    if (const transcribe_status st =
            transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "error: model load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }
    const char * backend_c = transcribe_model_backend(model);
    const std::string backend = (backend_c && *backend_c) ? backend_c : "unknown";

    struct transcribe_session_params cp; transcribe_session_params_init(&cp);
    cp.n_threads = args.n_threads;
    struct transcribe_session * ctx = nullptr;
    if (const transcribe_status st = transcribe_session_init(model, &cp, &ctx);
        st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "error: context init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // load_ms is a model-scoped one-shot, captured before any run.
    double load_ms = 0.0;
    {
        struct transcribe_timings load_tm; transcribe_timings_init(&load_tm);
        (void)transcribe_get_timings(ctx, &load_tm);
        load_ms = static_cast<double>(load_tm.load_ms);
    }

    struct transcribe_run_params rp; transcribe_run_params_init(&rp);
    rp.spec_k_drafts = args.spec_k_drafts;

    // Warmup (untimed).
    for (int w = 0; w < args.warmup; ++w) {
        if (!quiet) std::fprintf(stderr, "warmup %d/%d\n", w + 1, args.warmup);
        if (const transcribe_status st = transcribe_run(
                ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "error: warmup %d: %s\n",
                         w + 1, transcribe_status_string(st));
            transcribe_session_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
    }

    // Measured iterations. Empirically the library replaces per-run
    // counters rather than accumulating, so reset-before-run +
    // read-after-run cleanly extracts this run's work regardless of
    // whether the semantics are reset/accumulate/replace.
    std::vector<iter_timings> measured;
    measured.reserve(args.iters);
    for (int it = 0; it < args.iters; ++it) {
        if (!quiet) std::fprintf(stderr, "iter %d/%d\n", it + 1, args.iters);
        transcribe_reset_timings(ctx);
        const auto t0 = std::chrono::steady_clock::now();
        const transcribe_status st = transcribe_run(
            ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
        const auto t1 = std::chrono::steady_clock::now();
        if (st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "error: run %d: %s\n",
                         it + 1, transcribe_status_string(st));
            transcribe_session_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
        struct transcribe_timings after; transcribe_timings_init(&after);
        (void)transcribe_get_timings(ctx, &after);
        iter_timings row;
        row.mel_ms    = static_cast<double>(after.mel_ms);
        row.encode_ms = static_cast<double>(after.encode_ms);
        row.decode_ms = static_cast<double>(after.decode_ms);
        row.total_ms  = row.mel_ms + row.encode_ms + row.decode_ms;
        row.wall_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        measured.push_back(row);
    }

    const char * hyp_text = transcribe_full_text(ctx);
    const size_t hyp_text_len = hyp_text ? std::strlen(hyp_text) : 0;

    // Collect token IDs for the driver to hash. Emitting them here
    // (rather than hashing in C++) keeps this binary free of a SHA256
    // dependency; the driver is already Python, where hashlib is stdlib.
    const int n_tokens = transcribe_n_tokens(ctx);
    std::string token_ids_csv;
    token_ids_csv.reserve(static_cast<size_t>(n_tokens) * 6);
    for (int i = 0; i < n_tokens; ++i) {
        if (i > 0) token_ids_csv.push_back(',');
        struct transcribe_token tok; transcribe_token_init(&tok);
        (void)transcribe_get_token(ctx, i, &tok);
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", tok.id);
        token_ids_csv.append(buf);
    }

    if (!quiet) std::fprintf(stderr, "done\n");

    // Summary.
    const stat_summary s_mel    = compute_stats(measured, [](const iter_timings & t) { return t.mel_ms;    });
    const stat_summary s_encode = compute_stats(measured, [](const iter_timings & t) { return t.encode_ms; });
    const stat_summary s_decode = compute_stats(measured, [](const iter_timings & t) { return t.decode_ms; });
    const stat_summary s_total  = compute_stats(measured, [](const iter_timings & t) { return t.total_ms;  });
    const stat_summary s_wall   = compute_stats(measured, [](const iter_timings & t) { return t.wall_ms;   });
    // RTF based on wall time (the user-visible number — includes
    // uninstrumented run overhead like scheduler alloc, tensor upload,
    // thread fan-out). rtf_compute is the phase-timer sum for backward
    // compat with v1 consumers, but wall is the number to use for
    // perf decisions.
    const double rtf_wall_mean = (s_wall.mean_v > 0.0)
        ? sample_duration_s / (s_wall.mean_v / 1000.0) : 0.0;
    const double rtf_compute_mean = (s_total.mean_v > 0.0)
        ? sample_duration_s / (s_total.mean_v / 1000.0) : 0.0;

    // Hand-rolled JSON. Pretty-printed with 2-space indent; whitespace
    // is irrelevant to the driver but convenient for humans.
    std::string out;
    out.reserve(2048);
    auto stat = [&out](const char * key, const stat_summary & s, bool last) {
        append_fmt(out, "    \"%s\": {\"min\": %.3f, \"max\": %.3f, \"mean\": %.3f}%s\n",
                   key, s.min_v, s.max_v, s.mean_v, last ? "" : ",");
    };

    out += "{\n";
    append_fmt(out, "  \"schema\": \"transcribe-bench-v2\",\n");
    append_fmt(out, "  \"model_path\": \"%s\",\n",  json_escape(args.model_path.c_str()).c_str());
    append_fmt(out, "  \"sample_path\": \"%s\",\n", json_escape(args.sample_path.c_str()).c_str());
    append_fmt(out, "  \"sample_duration_s\": %.3f,\n", sample_duration_s);
    append_fmt(out, "  \"iters\": %d,\n",  args.iters);
    append_fmt(out, "  \"warmup\": %d,\n", args.warmup);
    append_fmt(out, "  \"spec_k_drafts\": %d,\n", args.spec_k_drafts);
    append_fmt(out, "  \"backend\": \"%s\",\n", json_escape(backend.c_str()).c_str());
    append_fmt(out, "  \"load_ms\": %.3f,\n", load_ms);
    out += "  \"per_iter\": [\n";
    for (size_t i = 0; i < measured.size(); ++i) {
        const auto & r = measured[i];
        append_fmt(out, "    {\"mel_ms\": %.3f, \"encode_ms\": %.3f, "
                        "\"decode_ms\": %.3f, \"total_ms\": %.3f, \"wall_ms\": %.3f}%s\n",
                   r.mel_ms, r.encode_ms, r.decode_ms, r.total_ms, r.wall_ms,
                   (i + 1 < measured.size()) ? "," : "");
    }
    out += "  ],\n";
    out += "  \"summary\": {\n";
    stat("mel_ms",    s_mel,    false);
    stat("encode_ms", s_encode, false);
    stat("decode_ms", s_decode, false);
    stat("total_ms",  s_total,  false);
    stat("wall_ms",   s_wall,   true);
    out += "  },\n";
    append_fmt(out, "  \"rtf_wall_mean\": %.3f,\n", rtf_wall_mean);
    append_fmt(out, "  \"rtf_compute_mean\": %.3f,\n", rtf_compute_mean);
    append_fmt(out, "  \"hyp_text_len\": %zu,\n", hyp_text_len);
    append_fmt(out, "  \"hyp_text\": \"%s\",\n",
               hyp_text ? json_escape(hyp_text).c_str() : "");
    append_fmt(out, "  \"n_tokens\": %d,\n", n_tokens);
    append_fmt(out, "  \"token_ids_csv\": \"%s\"\n", token_ids_csv.c_str());
    out += "}\n";

    int exit_code = EXIT_SUCCESS;
    if (!args.json_out.empty()) {
        std::FILE * f = std::fopen(args.json_out.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "error: cannot open %s for writing\n", args.json_out.c_str());
            exit_code = EXIT_FAILURE;
        } else {
            std::fwrite(out.data(), 1, out.size(), f);
            std::fclose(f);
        }
    } else {
        std::fwrite(out.data(), 1, out.size(), stdout);
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);
    return exit_code;
}
