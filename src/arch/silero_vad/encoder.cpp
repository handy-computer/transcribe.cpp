// arch/silero_vad/encoder.cpp - STFT windows (host) and the batched encoder
// graph: STFT magnitude, four convs and the LSTM input projection for a
// block of frames.
//
// Activations are frame-major with each frame's features time-major, so a
// frame's whole receptive field is one column and every stage is one
// mul_mat over the block:
//
//   windows  ne [n_fft, t0 * n]           host (build_windows)
//   stft     basis . windows              ne [2 * nbp, t0 * n]     nbp = n_bins_pad
//   mag      sqrt(re^2 + im^2)            ne [nbp, t0 * n] == [t0 * nbp, n]
//   enc.i    relu(dense_i . x + b_i)      ne [t_{i+1} * C_{i+1}, n]
//   gates_in W_ih . enc.3 + b_ih + b_hh   ne [4H, n]
//
// The dense_i matrices (weights.cpp) are the k3 / pad 1 / stride s convs
// unrolled over the frame's t_i time steps.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "silero_vad.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"

#include <cstddef>
#include <cstring>
#include <vector>

namespace transcribe::silero_vad {

namespace {

constexpr const char * kTag       = "silero_vad";
constexpr int          kGraphSize = 64;

// Name a tensor and keep it readable after compute when dumping.
ggml_tensor * dump_point(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    debug::mark_tensor_for_dump(t);
    return t;
}

// Build the encoder graph for n frames into s.compute_ctx.
transcribe_status build_graph(Session & s, const Model & m, int n) {
    const HParams & hp = m.hparams;
    const Weights & w  = m.weights;

    if (s.compute_ctx != nullptr) {
        ggml_free(s.compute_ctx);
        s.compute_ctx = nullptr;
    }
    s.graph_n = 0;

    ggml_init_params ip{};
    ip.mem_size   = ggml_graph_overhead_custom(kGraphSize, false) + kGraphSize * ggml_tensor_overhead();
    ip.no_alloc   = true;
    s.compute_ctx = ggml_init(ip);
    if (s.compute_ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_context * ctx = s.compute_ctx;
    GraphTensors & g   = s.g;
    g                  = GraphTensors{};

    g.windows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_fft, static_cast<int64_t>(hp.t_len[0]) * n);
    ggml_set_name(g.windows, "fe.windows");
    ggml_set_input(g.windows);

    // [re (nbp) | im (nbp)] per window; the padded bins are zero.
    ggml_tensor * spec  = ggml_mul_mat(ctx, w.stft_basis_pad, g.windows);
    ggml_tensor * re    = ggml_view_2d(ctx, spec, hp.n_bins_pad, spec->ne[1], spec->nb[1], 0);
    ggml_tensor * im    = ggml_view_2d(ctx, spec, hp.n_bins_pad, spec->ne[1], spec->nb[1],
                                       static_cast<size_t>(hp.n_bins_pad) * ggml_element_size(spec));
    ggml_tensor * power = ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im));
    ggml_tensor * mag   = ggml_sqrt(ctx, power);
    g.stft_mag          = dump_point(mag, "fe.stft_mag");

    ggml_tensor * x = ggml_reshape_2d(ctx, mag, static_cast<int64_t>(hp.n_bins_pad) * hp.t_len[0], n);
    for (int i = 0; i < kNumEncoder; ++i) {
        x = ggml_mul_mat(ctx, w.dense_w[i], x);
        x = ggml_add(ctx, x, w.dense_b[i]);
        x = ggml_relu(ctx, x);
        char name[32];
        snprintf(name, sizeof(name), "enc.%d.out", i);
        g.enc[i] = dump_point(x, name);
    }

    ggml_tensor * gates = ggml_mul_mat(ctx, w.lstm_w_ih, x);
    gates               = ggml_add(ctx, gates, w.lstm_b);
    ggml_set_name(gates, "dec.gates_in");
    ggml_set_output(gates);
    g.gates_in = gates;

    g.graph = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_build_forward_expand(g.graph, gates);

    if (s.sched == nullptr) {
        auto & backends = const_cast<std::vector<ggml_backend_t> &>(m.plan.scheduler_list);
        s.sched = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()), kGraphSize,
                                         /*parallel=*/false, /*op_offload=*/true);
        if (s.sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: scheduler allocation failed", kTag);
            return TRANSCRIBE_ERR_OOM;
        }
    }
    ggml_backend_sched_reset(s.sched);
    if (!ggml_backend_sched_alloc_graph(s.sched, g.graph)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: graph allocation failed (%d frames)", kTag, n);
        return TRANSCRIBE_ERR_OOM;
    }
    s.graph_n = n;
    return TRANSCRIBE_OK;
}

// Append the first n_real frames (columns of ne[1] / n_graph rows each).
void append_dump(std::vector<float> & dst, const ggml_tensor * t, int n_graph, int n_real) {
    const size_t n =
        static_cast<size_t>(ggml_nelements(t)) / static_cast<size_t>(n_graph) * static_cast<size_t>(n_real);
    const size_t off = dst.size();
    dst.resize(off + n);
    ggml_backend_tensor_get(t, dst.data() + off, 0, n * sizeof(float));
}

}  // namespace

void build_windows(Session & s, const HParams & hp, const float * pcm, int n, float * out) {
    const int ctx_n = hp.context_samples;
    const int chunk = hp.chunk;
    float *   buf   = s.chunk_buf.data();  // [padded]
    for (int f = 0; f < n; ++f) {
        const float * frame = pcm + static_cast<size_t>(f) * static_cast<size_t>(hp.frame_samples);
        // chunk = [context | frame], then torch "reflect" on the right
        // (edge sample not repeated).
        std::memcpy(buf, s.context.data(), static_cast<size_t>(ctx_n) * sizeof(float));
        std::memcpy(buf + ctx_n, frame, static_cast<size_t>(hp.frame_samples) * sizeof(float));
        for (int j = 0; j < hp.reflect_pad; ++j) {
            buf[chunk + j] = buf[chunk - 2 - j];
        }
        float * dst = out + static_cast<size_t>(f) * static_cast<size_t>(hp.t_len[0]) * static_cast<size_t>(hp.n_fft);
        for (int t = 0; t < hp.t_len[0]; ++t) {
            std::memcpy(dst + static_cast<size_t>(t) * static_cast<size_t>(hp.n_fft),
                        buf + static_cast<size_t>(t) * static_cast<size_t>(hp.hop),
                        static_cast<size_t>(hp.n_fft) * sizeof(float));
        }
        // The next frame's context is this chunk's tail.
        std::memcpy(s.context.data(), buf + chunk - ctx_n, static_cast<size_t>(ctx_n) * sizeof(float));
    }
}

transcribe_status encode_block(Session &     s,
                               const Model & m,
                               const float * windows,
                               int           n,
                               int           n_real,
                               float *       gates_in) {
    if (s.graph_n != n || s.compute_ctx == nullptr) {
        if (auto st = build_graph(s, m, n); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    const HParams & hp      = m.hparams;
    // Small blocks are cheaper on one thread than on a freshly spun-up pool.
    const int       threads = n >= 16 ? s.n_threads : 1;
    configure_sched_n_threads(s.sched, threads);

    ggml_backend_tensor_set(s.g.windows, windows, 0, ggml_nbytes(s.g.windows));
    if (const ggml_status gs = ggml_backend_sched_graph_compute(s.sched, s.g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: graph compute failed (%d)", kTag, static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_backend_tensor_get(s.g.gates_in, gates_in, 0,
                            static_cast<size_t>(n_real) * 4 * static_cast<size_t>(hp.hidden) * sizeof(float));

    if (debug::enabled()) {
        // Strip the padded bins: [t0 * n][nbp] -> [t0 * n][n_bins].
        std::vector<float> mag;
        append_dump(mag, s.g.stft_mag, n, n_real);
        const size_t rows = static_cast<size_t>(hp.t_len[0]) * static_cast<size_t>(n_real);
        for (size_t r = 0; r < rows; ++r) {
            const float * src = mag.data() + r * static_cast<size_t>(hp.n_bins_pad);
            s.trace.stft_mag.insert(s.trace.stft_mag.end(), src, src + hp.n_bins);
        }
        for (int i = 0; i < kNumEncoder; ++i) {
            append_dump(s.trace.enc[i], s.g.enc[i], n, n_real);
        }
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::silero_vad
