// ecapa_tdnn_gemm_unit.cpp - the ecapa_tdnn CPU kernels against scalar
// references: the AVX2 GEMM (src/arch/ecapa_tdnn/cpu_gemm.h) and the fused
// time reductions / softmax (cpu_ops.h).
//
// GEMM cases cover what the graph uses: F16 and F32 weights, 1..3 segments
// with row offsets (a dilated conv's taps), every epilogue,
// odd N (partial 6-wide tiles), K not a multiple of the k block, and 1..16
// threads; every result must be bit-identical across thread counts and
// within FMA-rounding distance of an fp64 reference. Skips (RC 77) when the
// CPU or build has no AVX2 kernel.

#include "arch/ecapa_tdnn/cpu_gemm.h"
#include "arch/ecapa_tdnn/cpu_ops.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace transcribe::ecapa_tdnn;

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

struct Case {
    const char *   name;
    ggml_type      wtype;
    int64_t        M, K, N;
    int            n_segs;    // segments share one weight tensor (n_mats = n_segs)
    int64_t        row_step;  // segment s reads x from row s * row_step
    gemm::Epilogue ep;
    bool           bias;
};

std::vector<float> rnd(size_t n, std::mt19937 & rng, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float>              v(n);
    for (auto & x : v) {
        x = nd(rng);
    }
    return v;
}

bool run_case(const Case & c) {
    std::mt19937  rng(1234);
    const int64_t Q      = c.n_segs;
    const int64_t x_rows = c.N + (c.n_segs - 1) * c.row_step;

    std::vector<float> wf = rnd(static_cast<size_t>(Q * c.M * c.K), rng, 0.05f);
    std::vector<float> xf = rnd(static_cast<size_t>(x_rows * c.K), rng, 1.0f);
    std::vector<float> bf = rnd(static_cast<size_t>(c.M), rng, 0.3f);
    std::vector<float> sf = rnd(static_cast<size_t>(c.M), rng, 1.0f);
    std::vector<float> hf = rnd(static_cast<size_t>(c.M), rng, 0.3f);

    // Weights as the kernel sees them: round through F16 when F16.
    std::vector<ggml_fp16_t> wh;
    if (c.wtype == GGML_TYPE_F16) {
        wh.resize(wf.size());
        ggml_fp32_to_fp16_row(wf.data(), wh.data(), static_cast<int64_t>(wf.size()));
        ggml_fp16_to_fp32_row(wh.data(), wf.data(), static_cast<int64_t>(wf.size()));
    }

    // fp64 reference.
    std::vector<double> ref(static_cast<size_t>(c.M * c.N));
    for (int64_t j = 0; j < c.N; ++j) {
        for (int64_t i = 0; i < c.M; ++i) {
            double acc = 0.0;
            for (int64_t s = 0; s < Q; ++s) {
                const float * w = wf.data() + static_cast<size_t>((s * c.M + i) * c.K);
                const float * x = xf.data() + static_cast<size_t>((j + s * c.row_step) * c.K);
                for (int64_t k = 0; k < c.K; ++k) {
                    acc += static_cast<double>(w[k]) * static_cast<double>(x[k]);
                }
            }
            if (c.ep != gemm::Epilogue::None) {
                double v = acc + (c.bias ? bf[static_cast<size_t>(i)] : 0.0);
                v        = v > 0.0 ? v : 0.0;
                v        = v * sf[static_cast<size_t>(i)] + hf[static_cast<size_t>(i)];
                if (c.ep == gemm::Epilogue::ReluBnTanh) {
                    v = std::tanh(v);
                }
                acc = v;
            }
            ref[static_cast<size_t>(j * c.M + i)] = acc;
        }
    }

    std::vector<float> first;
    bool               ok = true;
    for (int nt : { 1, 3, 8, 16 }) {
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, nt);
        ggml_init_params      ip{ 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context *        ctx  = ggml_init(ip);
        ggml_tensor *         w    = ggml_new_tensor_3d(ctx, c.wtype, c.K, c.M, Q);
        ggml_tensor *         x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, x_rows);
        ggml_tensor *         b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.M);
        ggml_tensor *         s    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.M);
        ggml_tensor *         h    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.M);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (c.wtype == GGML_TYPE_F16) {
            ggml_backend_tensor_set(w, wh.data(), 0, ggml_nbytes(w));
        } else {
            ggml_backend_tensor_set(w, wf.data(), 0, ggml_nbytes(w));
        }
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(b, bf.data(), 0, ggml_nbytes(b));
        ggml_backend_tensor_set(s, sf.data(), 0, ggml_nbytes(s));
        ggml_backend_tensor_set(h, hf.data(), 0, ggml_nbytes(h));
        CHECK(gemm::pack_weight(w), "%s: pack_weight failed", c.name);

        gemm::Arena   arena;
        gemm::Segment segs[gemm::kMaxSegs];
        for (int i = 0; i < c.n_segs; ++i) {
            segs[i] = { w, i, x, i * c.row_step };
        }
        const bool    epi = c.ep != gemm::Epilogue::None;
        ggml_tensor * y   = gemm::mul_mat(ctx, arena, segs, c.n_segs, c.N, c.ep, epi && c.bias ? b : nullptr,
                                          epi ? s : nullptr, epi ? h : nullptr);
        ggml_cgraph * g   = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        CHECK(ggml_gallocr_alloc_graph(ga, g), "%s: graph alloc failed", c.name);
        // Run twice: the second compute exercises the work-stealing state
        // left behind by the first.
        for (int rep = 0; rep < 2; ++rep) {
            CHECK(ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS, "%s: compute failed", c.name);
        }

        std::vector<float> out(static_cast<size_t>(c.M * c.N));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

        double worst = 0.0;
        for (int64_t j = 0; j < c.N; ++j) {
            for (int64_t i = 0; i < c.M; ++i) {
                const double got = out[static_cast<size_t>(j * c.M + i)];
                const double r   = ref[static_cast<size_t>(j * c.M + i)];
                worst            = std::fmax(worst, std::fabs(got - r) / (1.0 + std::fabs(r)));
            }
        }
        CHECK(worst < 1e-4, "%s t=%d: max rel err %.3e", c.name, nt, worst);
        ok &= worst < 1e-4;
        if (first.empty()) {
            first = out;
        } else {
            const bool same = std::memcmp(first.data(), out.data(), out.size() * sizeof(float)) == 0;
            CHECK(same, "%s t=%d: result differs from t=1", c.name, nt);
            ok &= same;
        }

        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }
    return ok;
}

// time_stats / attn_stats against fp64 references on a [C, T] tensor.
void run_reductions() {
    const int64_t      C = 600, T = 517;  // C not a multiple of the 256 channel block
    std::mt19937       rng(99);
    std::vector<float> xf  = rnd(static_cast<size_t>(C * T), rng, 1.0f);
    std::vector<float> lf  = rnd(static_cast<size_t>(C * T), rng, 2.0f);
    std::vector<float> bf  = rnd(static_cast<size_t>(C), rng, 1.0f);
    static const float eps = 1e-12f;

    for (int nt : { 1, 5, 8 }) {
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, nt);
        ggml_init_params      ip{ 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context *        ctx = ggml_init(ip);
        ggml_tensor *         x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
        ggml_tensor *         l   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
        ggml_tensor *         b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(l, lf.data(), 0, ggml_nbytes(l));
        ggml_backend_tensor_set(b, bf.data(), 0, ggml_nbytes(b));

        ggml_tensor * st = cpu::time_stats(ctx, x, &eps);
        ggml_tensor * at = cpu::attn_stats(ctx, x, l, b, &eps);
        ggml_tensor * mn = cpu::time_mean(ctx, x);
        ggml_cgraph * g  = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, st);
        ggml_build_forward_expand(g, at);
        ggml_build_forward_expand(g, mn);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, g);
        CHECK(ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS, "reductions: compute failed");

        std::vector<float> vst(static_cast<size_t>(2 * C)), vat(static_cast<size_t>(2 * C)),
            vmn(static_cast<size_t>(C));
        ggml_backend_tensor_get(st, vst.data(), 0, ggml_nbytes(st));
        ggml_backend_tensor_get(at, vat.data(), 0, ggml_nbytes(at));
        ggml_backend_tensor_get(mn, vmn.data(), 0, ggml_nbytes(mn));

        double e_st = 0, e_at = 0, e_mn = 0;
        for (int64_t c = 0; c < C; ++c) {
            double s = 0, q = 0, mx = -1e30, z = 0, mu = 0, sg = 0;
            for (int64_t t = 0; t < T; ++t) {
                s += xf[static_cast<size_t>(t * C + c)];
            }
            const double mean = s / T;
            for (int64_t t = 0; t < T; ++t) {
                const double d = xf[static_cast<size_t>(t * C + c)] - mean;
                q += d * d;
                mx =
                    std::fmax(mx, static_cast<double>(lf[static_cast<size_t>(t * C + c)]) + bf[static_cast<size_t>(c)]);
            }
            for (int64_t t = 0; t < T; ++t) {
                z +=
                    std::exp(static_cast<double>(lf[static_cast<size_t>(t * C + c)]) + bf[static_cast<size_t>(c)] - mx);
            }
            for (int64_t t = 0; t < T; ++t) {
                const double w = std::exp(static_cast<double>(lf[static_cast<size_t>(t * C + c)]) +
                                          bf[static_cast<size_t>(c)] - mx) /
                                 z;
                mu += w * xf[static_cast<size_t>(t * C + c)];
            }
            for (int64_t t = 0; t < T; ++t) {
                const double w = std::exp(static_cast<double>(lf[static_cast<size_t>(t * C + c)]) +
                                          bf[static_cast<size_t>(c)] - mx) /
                                 z;
                const double d = xf[static_cast<size_t>(t * C + c)] - mu;
                sg += w * d * d;
            }
            e_mn = std::fmax(e_mn, std::fabs(vmn[static_cast<size_t>(c)] - mean));
            e_st = std::fmax(e_st, std::fabs(vst[static_cast<size_t>(c)] - mean));
            e_st = std::fmax(e_st, std::fabs(vst[static_cast<size_t>(C + c)] - std::sqrt(q / T)));
            e_at = std::fmax(e_at, std::fabs(vat[static_cast<size_t>(c)] - mu));
            e_at = std::fmax(e_at, std::fabs(vat[static_cast<size_t>(C + c)] - std::sqrt(sg)));
        }
        CHECK(e_mn < 1e-5 && e_st < 1e-5 && e_at < 1e-4, "reductions t=%d: mean %.2e stats %.2e attn %.2e", nt, e_mn,
              e_st, e_at);

        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }
}

}  // namespace

int main() {
    run_reductions();

    if (!gemm::available()) {
        std::printf("ecapa_tdnn_gemm_unit: no AVX2 GEMM on this CPU/build; reductions ok=%d, GEMM skipped\n",
                    g_fail == 0);
        return g_fail == 0 ? 77 : 1;
    }

    const Case cases[] = {
        { "f16 1x1 relu-bn",        GGML_TYPE_F16, 64, 96,  37,  1, 0, gemm::Epilogue::ReluBn,     true  },
        { "f16 mfa 3-seg",          GGML_TYPE_F16, 48, 300, 61,  3, 0, gemm::Epilogue::ReluBn,     true  },
        { "f16 none tanh-free",     GGML_TYPE_F16, 32, 513, 13,  1, 0, gemm::Epilogue::None,       false },
        { "f16 tanh",               GGML_TYPE_F16, 16, 40,  7,   1, 0, gemm::Epilogue::ReluBnTanh, true  },
        { "f32 dilated conv 3-tap", GGML_TYPE_F32, 32, 24,  50,  3, 4, gemm::Epilogue::ReluBn,     true  },
        { "f32 im2col no-bias",     GGML_TYPE_F32, 16, 320, 101, 1, 0, gemm::Epilogue::ReluBn,     false },
        { "f16 N=1",                GGML_TYPE_F16, 16, 33,  1,   2, 1, gemm::Epilogue::None,       false },
    };
    for (const Case & c : cases) {
        run_case(c);
    }
    if (g_fail == 0) {
        std::printf("ecapa_tdnn_gemm_unit: ok\n");
    }
    return g_fail == 0 ? 0 : 1;
}
