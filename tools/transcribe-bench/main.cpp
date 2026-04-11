// transcribe-bench - one-shot perf measurement for a (model, sample) pair.
//
// Developer-loop tool. Loads a model once, runs --warmup untimed
// iterations, then --iters measured iterations, and emits a JSON
// document that a Python driver aggregates across a matrix of
// models and samples. Progress lines go to stderr; the JSON goes
// to --json-out (or stdout if omitted).
//
// Schema is frozen at "transcribe-bench-v1". Do not change the
// schema without coordinating with the Python driver.

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
        "  --threads N        CPU threads (default 0 = all cores)\n"
        "  --backend KIND     request a specific backend: auto|cpu|metal|vulkan\n"
        "                     (default auto). cpu is strict CPU — no GPU,\n"
        "                     no BLAS/AMX.\n"
        "  -h, --help         show this help\n",
        argv0);
}

// Parse --backend KIND into the public enum. Returns false and logs a
// diagnostic on an unknown value.
bool parse_backend_kind(const char *                   s,
                        transcribe_backend_request &   out)
{
    if (s == nullptr) return false;
    if (std::strcmp(s, "auto")   == 0) { out = TRANSCRIBE_BACKEND_AUTO;   return true; }
    if (std::strcmp(s, "cpu")    == 0) { out = TRANSCRIBE_BACKEND_CPU;    return true; }
    if (std::strcmp(s, "metal")  == 0) { out = TRANSCRIBE_BACKEND_METAL;  return true; }
    if (std::strcmp(s, "vulkan") == 0) { out = TRANSCRIBE_BACKEND_VULKAN; return true; }
    std::fprintf(stderr,
        "error: --backend value '%s' not recognized "
        "(expected one of: auto, cpu, metal, vulkan)\n", s);
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
        else if (a == "--quiet")    { out.quiet = true; }
        else if (a == "--backend")  {
            auto v = need_val(i, "--backend"); if (!v) return false;
            if (!parse_backend_kind(v, out.backend)) return false;
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
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) out.append(buf, static_cast<size_t>(n));
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

    // Load WAV.
    if (!quiet) std::fprintf(stderr, "loading sample %s\n", args.sample_path.c_str());
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(args.sample_path, pcm, wav_err)) {
        std::fprintf(stderr, "error: wav: %s\n", wav_err.c_str());
        return EXIT_FAILURE;
    }
    const double sample_duration_s = static_cast<double>(pcm.size()) / 16000.0;

    // Load model. --backend propagates directly to
    // transcribe_model_params::backend, which the per-family loader
    // consumes via transcribe::load_common::init_backends.
    if (!quiet) std::fprintf(stderr, "loading model %s\n", args.model_path.c_str());
    struct transcribe_model_params mp = transcribe_model_default_params();
    mp.backend = args.backend;
    struct transcribe_model *      model = nullptr;
    if (const transcribe_status st =
            transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
        st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "error: model load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }
    const char * backend_c = transcribe_model_backend(model);
    const std::string backend = (backend_c && *backend_c) ? backend_c : "unknown";

    // Init context.
    struct transcribe_context_params cp = transcribe_context_default_params();
    cp.n_threads = args.n_threads;
    struct transcribe_context * ctx = nullptr;
    if (const transcribe_status st = transcribe_context_init(model, &cp, &ctx);
        st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "error: context init: %s\n", transcribe_status_string(st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // load_ms is a model-scoped one-shot, captured before any run.
    const double load_ms = static_cast<double>(transcribe_get_timings(ctx).load_ms);

    struct transcribe_params rp = transcribe_default_params();

    // Warmup (untimed).
    for (int w = 0; w < args.warmup; ++w) {
        if (!quiet) std::fprintf(stderr, "warmup %d/%d\n", w + 1, args.warmup);
        if (const transcribe_status st = transcribe_run(
                ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "error: warmup %d: %s\n",
                         w + 1, transcribe_status_string(st));
            transcribe_context_free(ctx);
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
            transcribe_context_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }
        const struct transcribe_timings after = transcribe_get_timings(ctx);
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
    if (!quiet) std::fprintf(stderr, "done\n");

    // Summary.
    const stat_summary s_mel    = compute_stats(measured, [](const iter_timings & t) { return t.mel_ms;    });
    const stat_summary s_encode = compute_stats(measured, [](const iter_timings & t) { return t.encode_ms; });
    const stat_summary s_decode = compute_stats(measured, [](const iter_timings & t) { return t.decode_ms; });
    const stat_summary s_total  = compute_stats(measured, [](const iter_timings & t) { return t.total_ms;  });
    const stat_summary s_wall   = compute_stats(measured, [](const iter_timings & t) { return t.wall_ms;   });
    const double rtf_mean = (s_total.mean_v > 0.0)
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
    append_fmt(out, "  \"schema\": \"transcribe-bench-v1\",\n");
    append_fmt(out, "  \"model_path\": \"%s\",\n",  json_escape(args.model_path.c_str()).c_str());
    append_fmt(out, "  \"sample_path\": \"%s\",\n", json_escape(args.sample_path.c_str()).c_str());
    append_fmt(out, "  \"sample_duration_s\": %.3f,\n", sample_duration_s);
    append_fmt(out, "  \"iters\": %d,\n",  args.iters);
    append_fmt(out, "  \"warmup\": %d,\n", args.warmup);
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
    append_fmt(out, "  \"rtf_mean\": %.3f,\n", rtf_mean);
    append_fmt(out, "  \"hyp_text_len\": %zu\n", hyp_text_len);
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

    transcribe_context_free(ctx);
    transcribe_model_free(model);
    return exit_code;
}
