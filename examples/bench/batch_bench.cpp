// batch_bench.cpp - offline batch throughput sweep for transcribe.cpp.
//
// Loads a model + session ONCE, then times transcribe_run_batch over a
// sweep of batch sizes, reporting per-utterance latency and wall time per
// batch. Model load is excluded, so per_utt_ms is directly comparable
// across batch sizes.
//
// The batch is built by repeating a single input clip N times, which
// isolates the device-occupancy effect of batching from input-length
// variance. (Real workloads vary; this is a controlled throughput probe,
// not a WER measurement.) Output is a JSON array on stdout, one object per
// batch size.
//
// Usage:
//   transcribe-batch-bench -m model.gguf clip.wav \
//       [--backend auto|cpu|metal|...] [--batch-sizes 1,2,4,8,16,32] \
//       [--iters 3] [--threads 0] [--out report.json]

#include "transcribe.h"
#include "wav.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
               steady_clock::now().time_since_epoch())
        .count();
}

transcribe_backend_request parse_backend(const char * s) {
    if (!std::strcmp(s, "cpu"))       return TRANSCRIBE_BACKEND_CPU;
    if (!std::strcmp(s, "cpu_accel")) return TRANSCRIBE_BACKEND_CPU_ACCEL;
    if (!std::strcmp(s, "metal"))     return TRANSCRIBE_BACKEND_METAL;
    if (!std::strcmp(s, "vulkan"))    return TRANSCRIBE_BACKEND_VULKAN;
    if (!std::strcmp(s, "cuda"))      return TRANSCRIBE_BACKEND_CUDA;
    return TRANSCRIBE_BACKEND_AUTO;
}

std::vector<int> parse_sizes(const std::string & csv) {
    std::vector<int> out;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        const std::string tok = csv.substr(i, j - i);
        if (!tok.empty()) {
            const int n = std::atoi(tok.c_str());
            if (n > 0) out.push_back(n);
        }
        i = j + 1;
    }
    return out;
}

std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n')        { o += "\\n"; }
        else                        { o += c; }
    }
    return o;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string wav_path;
    std::string out_path;
    std::string sizes_csv = "1,2,4,8,16,32";
    transcribe_backend_request backend = TRANSCRIBE_BACKEND_AUTO;
    int iters = 3;
    int n_threads = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model")          model_path = next(a.c_str());
        else if (a == "--backend")                backend = parse_backend(next(a.c_str()));
        else if (a == "--batch-sizes")            sizes_csv = next(a.c_str());
        else if (a == "--iters")                  iters = std::atoi(next(a.c_str()));
        else if (a == "--threads")                n_threads = std::atoi(next(a.c_str()));
        else if (a == "--out")                    out_path = next(a.c_str());
        else if (!a.empty() && a[0] != '-')       wav_path = a;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || wav_path.empty()) {
        std::fprintf(stderr,
            "usage: %s -m model.gguf clip.wav [--backend B] "
            "[--batch-sizes 1,2,4,8] [--iters N] [--threads N] [--out f.json]\n",
            argv[0]);
        return 2;
    }
    if (iters < 1) iters = 1;

    struct transcribe_model_load_params lp; transcribe_model_load_params_init(&lp);
    lp.backend = backend;
    struct transcribe_session_params sp; transcribe_session_params_init(&sp);
    sp.n_threads = n_threads;
    struct transcribe_session * s = nullptr;
    if (auto st = transcribe_open(model_path.c_str(), &lp, &sp, &s);
        st != TRANSCRIBE_OK)
    {
        std::fprintf(stderr, "open failed: %s\n", transcribe_status_string(st));
        return 1;
    }
    const char * dev = transcribe_model_backend(transcribe_get_model(s));

    std::vector<float> pcm;
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, err)) {
        std::fprintf(stderr, "wav: %s\n", err.c_str());
        transcribe_session_free(s);
        return 1;
    }
    const double audio_s = (double)pcm.size() / 16000.0;
    const std::vector<int> sizes = parse_sizes(sizes_csv);

    std::string json = "[\n";
    for (size_t si = 0; si < sizes.size(); ++si) {
        const int N = sizes[si];
        std::vector<const float *> pcm_ptrs(N, pcm.data());
        std::vector<int>           n_samps(N, (int)pcm.size());

        // Warmup (allocates the batch graph / buffers for this shape).
        auto wst = transcribe_run_batch(s, pcm_ptrs.data(), n_samps.data(), N, nullptr);
        std::string row_err;
        if (wst != TRANSCRIBE_OK) {
            row_err = transcribe_status_string(wst);
        }

        const double t_sweep0 = now_ms();
        double wall_sum = 0.0;
        for (int it = 0; it < iters && row_err.empty(); ++it) {
            const double t0 = now_ms();
            auto st = transcribe_run_batch(s, pcm_ptrs.data(), n_samps.data(), N, nullptr);
            const double t1 = now_ms();
            if (st != TRANSCRIBE_OK) { row_err = transcribe_status_string(st); break; }
            wall_sum += (t1 - t0);
        }
        const double elapsed_s = (now_ms() - t_sweep0) / 1000.0;
        const double wall_mean = row_err.empty() ? wall_sum / iters : 0.0;
        const double per_utt   = row_err.empty() ? wall_mean / N : 0.0;

        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "  {\"n_batch\": %d, \"wall_mean_ms\": %.3f, \"per_utt_ms\": %.4f, "
            "\"audio_s\": %.3f, \"n_iters\": %d, \"elapsed_s\": %.4f, "
            "\"backend\": \"%s\", \"error\": %s}%s\n",
            N, wall_mean, per_utt, audio_s, iters, elapsed_s,
            dev ? dev : "",
            row_err.empty() ? "null" : ("\"" + json_escape(row_err) + "\"").c_str(),
            (si + 1 < sizes.size()) ? "," : "");
        json += buf;

        std::fprintf(stderr,
            "n_batch=%-4d  per_utt=%8.2f ms  wall=%9.2f ms  %s\n",
            N, per_utt, wall_mean, row_err.empty() ? "" : row_err.c_str());
    }
    json += "]\n";

    if (!out_path.empty()) {
        FILE * f = std::fopen(out_path.c_str(), "w");
        if (f) { std::fputs(json.c_str(), f); std::fclose(f); }
        std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    } else {
        std::fputs(json.c_str(), stdout);
    }

    transcribe_session_free(s);
    return 0;
}
