// pre_encode_tile_unit.cpp - ConvPolicy::pre_encode_time_chunk is exact.
//
// build_pre_encode can run the DwStridingSubsampling stack over overlapping
// mel windows instead of the whole utterance (the stack's activations are
// otherwise linear in audio length: 3 GB at 306 s for cohere/canary-qwen).
// The window arithmetic (stride-8 aligned starts, one dropped frame on the
// left, balanced window lengths, the utterance edges left untouched) is the
// part that can silently go wrong, so this pins it on random weights:
// every window size, against the whole-utterance stack, across mel lengths
// that hit every residue mod 8 and the one-window / many-window boundaries,
// for batch 1 and 2, on every compute device present.

#include "conformer/conformer.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace conf = transcribe::conformer;

int g_failures = 0;

constexpr int kMels  = 16;  // F' = 2 after three stride-2 stages
constexpr int kChan  = 8;
constexpr int kModel = 12;

uint32_t g_rng = 0x2468ace1u;

float next_uniform() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>(g_rng >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
}

struct HostWeights {
    std::vector<float> conv0_w, conv0_b, conv2_w, conv2_b, conv3_w, conv3_b;
    std::vector<float> conv5_w, conv5_b, conv6_w, conv6_b, out_w, out_b;
};

std::vector<float> rnd(size_t n, float scale) {
    std::vector<float> v(n);
    for (float & x : v) {
        x = next_uniform() * scale;
    }
    return v;
}

// Runs build_pre_encode on `be` and returns the [d_model, T_enc, B] output,
// or an empty vector if the device cannot run the graph.
std::vector<float> run(ggml_backend_t             be,
                       const HostWeights &        hw,
                       const std::vector<float> & mel,
                       int                        t_mel,
                       int                        n_batch,
                       int                        chunk,
                       bool                       f32_paths,
                       int64_t *                  t_enc_out) {
    ggml_init_params ip  = { 4096 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);

    auto new_t = [&](int64_t a, int64_t b, int64_t c, int64_t d) {
        return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d);
    };
    conf::PreEncodeView pe;
    pe.conv0_w                           = new_t(3, 3, 1, kChan);
    pe.conv0_b                           = new_t(kChan, 1, 1, 1);
    pe.conv2_w                           = new_t(3, 3, 1, kChan);
    pe.conv2_b                           = new_t(kChan, 1, 1, 1);
    pe.conv3_w                           = new_t(1, 1, kChan, kChan);
    pe.conv3_b                           = new_t(kChan, 1, 1, 1);
    pe.conv5_w                           = new_t(3, 3, 1, kChan);
    pe.conv5_b                           = new_t(kChan, 1, 1, 1);
    pe.conv6_w                           = new_t(1, 1, kChan, kChan);
    pe.conv6_b                           = new_t(kChan, 1, 1, 1);
    pe.out_w                             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (kMels / 8) * kChan, kModel);
    pe.out_b                             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kModel);
    ggml_tensor *              mel_in    = new_t(t_mel, kMels, 1, n_batch);
    ggml_tensor *              weights[] = { pe.conv0_w, pe.conv0_b, pe.conv2_w, pe.conv2_b, pe.conv3_w, pe.conv3_b,
                                             pe.conv5_w, pe.conv5_b, pe.conv6_w, pe.conv6_b, pe.out_w,   pe.out_b };
    const std::vector<float> * host[] = { &hw.conv0_w, &hw.conv0_b, &hw.conv2_w, &hw.conv2_b, &hw.conv3_w, &hw.conv3_b,
                                          &hw.conv5_w, &hw.conv5_b, &hw.conv6_w, &hw.conv6_b, &hw.out_w,   &hw.out_b };

    conf::ConvPolicy policy{};
    policy.direct_dw_in_pre_encode    = true;
    policy.pre_encode_time_chunk      = chunk;
    // f32_paths: direct conv0 + F32-im2col pointwise convs. Otherwise the
    // vendored ggml_conv_2d's F16 im2col (cohere / canary-qwen's config).
    policy.direct_conv0_in_pre_encode = f32_paths;
    policy.pre_encode_f32_pointwise   = f32_paths;

    ggml_tensor *      out = conf::build_pre_encode(ctx, pe, mel_in, policy, "pe", "pre_encode_tile_unit");
    std::vector<float> res;
    if (out == nullptr) {
        std::fprintf(stderr, "FAIL build_pre_encode returned null (t_mel=%d chunk=%d)\n", t_mel, chunk);
        ++g_failures;
        ggml_free(ctx);
        return res;
    }
    out              = ggml_cont(ctx, out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
            ggml_free(ctx);
            return res;  // device cannot run it: caller skips
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (size_t i = 0; i < sizeof(weights) / sizeof(weights[0]); ++i) {
        ggml_backend_tensor_set(weights[i], host[i]->data(), 0, ggml_nbytes(weights[i]));
    }
    ggml_backend_tensor_set(mel_in, mel.data(), 0, ggml_nbytes(mel_in));
    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
        res.resize(static_cast<size_t>(ggml_nelements(out)));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        *t_enc_out = out->ne[1];
    } else {
        std::fprintf(stderr, "FAIL graph compute (t_mel=%d chunk=%d)\n", t_mel, chunk);
        ++g_failures;
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

void run_device(ggml_backend_dev_t dev, const HostWeights & hw) {
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be == nullptr) {
        return;
    }
    const char * name = ggml_backend_name(be);
    for (bool f32_paths : { false, true }) {
        int          n_cases = 0;
        double       worst   = 0.0;
        // Same inputs through the same per-element ops, so any difference is a
        // backend choosing another matmul kernel for a different extent, i.e.
        // accumulation order and, on Vulkan, the kernel's internal precision
        // (up to ~2e-4 at the 1-2 frame windows here). Metal is bit-identical.
        // A window-arithmetic error (wrong halo, misaligned start, dropped or
        // duplicated frame) is O(0.1) on these weights, so the bound sits well
        // between the two.
        const double tol     = 1e-3;

        // Mel lengths: every residue mod 8, around the window boundaries for the
        // small windows below (T_enc = ceil(T_mel / 8)).
        const int t_mels[] = { 8, 9, 31, 32, 33, 39, 40, 41, 47, 63, 64, 65, 100, 129, 203 };
        for (int n_batch : { 1, 2 }) {
            for (int t_mel : t_mels) {
                const std::vector<float> mel   = rnd(static_cast<size_t>(t_mel) * kMels * n_batch, 2.0f);
                int64_t                  t_ref = 0;
                const std::vector<float> ref   = run(be, hw, mel, t_mel, n_batch, /*chunk=*/0, f32_paths, &t_ref);
                if (ref.empty()) {
                    std::fprintf(stderr, "note: %s cannot run the pre_encode graph, skipped\n", name);
                    ggml_backend_free(be);
                    return;
                }
                for (int chunk : { 1, 2, 3, 4, 5, 7, 16 }) {
                    int64_t                  t_got = 0;
                    const std::vector<float> got   = run(be, hw, mel, t_mel, n_batch, chunk, f32_paths, &t_got);
                    ++n_cases;
                    if (t_got != t_ref || got.size() != ref.size()) {
                        std::fprintf(stderr, "FAIL %s t_mel=%d B=%d chunk=%d: T_enc %lld vs %lld\n", name, t_mel,
                                     n_batch, chunk, static_cast<long long>(t_got), static_cast<long long>(t_ref));
                        ++g_failures;
                        continue;
                    }
                    double e = 0.0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i])) /
                                         (1.0 + std::fabs(static_cast<double>(ref[i])));
                        if (!(d <= e)) {
                            e = d;
                        }
                    }
                    if (!(e <= worst)) {
                        worst = e;
                    }
                    if (!(e < tol)) {
                        std::fprintf(stderr, "FAIL %s t_mel=%d B=%d chunk=%d f32=%d: max rel %.3e\n", name, t_mel,
                                     n_batch, chunk, f32_paths ? 1 : 0, e);
                        ++g_failures;
                    }
                }
            }
        }
        std::printf("%-8s %-9s %d tiled cases, worst rel diff vs whole-utterance %.2e\n", name,
                    f32_paths ? "f32" : "f16-im2col", n_cases, worst);
    }
    ggml_backend_free(be);
}

}  // namespace

int main() {
    HostWeights hw;
    hw.conv0_w = rnd(3 * 3 * kChan, 0.5f);
    hw.conv0_b = rnd(kChan, 0.2f);
    hw.conv2_w = rnd(3 * 3 * kChan, 0.5f);
    hw.conv2_b = rnd(kChan, 0.2f);
    hw.conv3_w = rnd(kChan * kChan, 0.5f);
    hw.conv3_b = rnd(kChan, 0.2f);
    hw.conv5_w = rnd(3 * 3 * kChan, 0.5f);
    hw.conv5_b = rnd(kChan, 0.2f);
    hw.conv6_w = rnd(kChan * kChan, 0.5f);
    hw.conv6_b = rnd(kChan, 0.2f);
    hw.out_w   = rnd(static_cast<size_t>(kMels / 8) * kChan * kModel, 0.5f);
    hw.out_b   = rnd(kModel, 0.2f);

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        run_device(ggml_backend_dev_get(i), hw);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("OK\n");
    return EXIT_SUCCESS;
}
