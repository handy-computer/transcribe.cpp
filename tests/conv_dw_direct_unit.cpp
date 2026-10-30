// conv_dw_direct_unit.cpp - CONV_2D_DW (ggml_conv_2d_dw_direct) on every
// compute device present, against a host reference and the im2col forms.
//
// The conformer conv helpers used to route Metal to im2col + mul_mat on the
// strength of a comment saying CONV_2D_DW was unsupported there. The direct
// op is what keeps the depthwise convs from inflating the activation by the
// kernel width (the 1.9 GB granite block conv, the 2.3 GB conformer
// pre-encode conv at 306 s), so the policy rests on this test instead:
//
//   * every device that reports CONV_2D_DW support computes the shapes the
//     encoders use (k=15 1-D "same" conv, k=3 s=2 2-D pre-encode conv with
//     the offline p=1 and the causal p=2 padding) within F32 rounding of a
//     double-precision host reference, for F32 and F16 kernels;
//   * Metal and CPU must report support: a regression there fails, it does
//     not skip;
//   * the direct op is batch-invariant: each utterance of an N=3 batch is
//     bit-identical to the same utterance run alone (the batch-parity
//     requirement conv_1d_dw_f32's B-dependent algorithm switch breaks);
//   * the direct op agrees with the im2col helpers it replaces, within F16
//     rounding (the im2col routes round to F16 on some backends), where the
//     device runs both.

#include "conformer/conformer.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace conf = transcribe::conformer;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

struct Case {
    const char * name;
    int          kw, kh;  // kernel
    int          w, h;    // spatial input (W fast)
    int          c;       // channels
    int          s0, s1, p0, p1;
};

// Shapes the encoders use, at toy sizes. Odd extents exercise the strided
// tail; the 1-D case is W=time, H=1 as granite / conformer blocks build it.
const Case kCases[] = {
    { "block-1d-k15-same",    15, 1, 37, 1,  12, 1, 1, 7, 0 },
    { "pre-encode-k3s2-p1",   3,  3, 17, 23, 8,  2, 2, 1, 1 },
    { "pre-encode-k3s2-p2",   3,  3, 17, 23, 8,  2, 2, 2, 2 },
    { "pre-encode-k3s2-even", 3,  3, 16, 24, 8,  2, 2, 1, 1 },
};

int64_t out_size(int64_t in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

uint32_t g_rng = 0x12345678u;

float next_uniform() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>(g_rng >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
}

// Depthwise reference in double: out[n][c][oh][ow].
std::vector<float> reference(const Case &               cs,
                             const std::vector<float> & kernel,
                             const std::vector<float> & data,
                             int                        n_batch) {
    const int64_t      ow = out_size(cs.w, cs.kw, cs.s0, cs.p0);
    const int64_t      oh = out_size(cs.h, cs.kh, cs.s1, cs.p1);
    std::vector<float> out(static_cast<size_t>(ow * oh * cs.c * n_batch));
    for (int n = 0; n < n_batch; ++n) {
        for (int c = 0; c < cs.c; ++c) {
            for (int64_t y = 0; y < oh; ++y) {
                for (int64_t x = 0; x < ow; ++x) {
                    double acc = 0.0;
                    for (int ky = 0; ky < cs.kh; ++ky) {
                        for (int kx = 0; kx < cs.kw; ++kx) {
                            const int64_t iy = y * cs.s1 - cs.p1 + ky;
                            const int64_t ix = x * cs.s0 - cs.p0 + kx;
                            if (iy < 0 || iy >= cs.h || ix < 0 || ix >= cs.w) {
                                continue;
                            }
                            const float d = data[static_cast<size_t>(((n * cs.c + c) * cs.h + iy) * cs.w + ix)];
                            const float k = kernel[static_cast<size_t>((c * cs.kh + ky) * cs.kw + kx)];
                            acc += static_cast<double>(d) * static_cast<double>(k);
                        }
                    }
                    out[static_cast<size_t>(((n * cs.c + c) * oh + y) * ow + x)] = static_cast<float>(acc);
                }
            }
        }
    }
    return out;
}

enum class Algo { Direct, Im2col };

struct RunResult {
    bool               supported = false;
    std::vector<float> out;
};

// Build and run one depthwise conv on `be`. Host kernel values must already
// be representable in `ktype` (F16 cases pre-round them).
RunResult run(ggml_backend_t             be,
              const Case &               cs,
              ggml_type                  ktype,
              Algo                       algo,
              const std::vector<float> & kernel,
              const std::vector<float> & data,
              int                        n_batch) {
    RunResult        res;
    ggml_init_params ip  = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return res;
    }

    ggml_tensor * k = ggml_new_tensor_4d(ctx, ktype, cs.kw, cs.kh, 1, cs.c);
    ggml_tensor * d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cs.w, cs.h, cs.c, n_batch);

    ggml_tensor * out = nullptr;
    if (algo == Algo::Direct) {
        out = conf::conv_2d_dw_direct_f32(ctx, k, d, cs.s0, cs.s1, cs.p0, cs.p1, 1, 1);
    } else if (cs.h == 1) {
        // The 1-D helper takes [W, C, B]; it is single-utterance in its
        // im2col form, so the caller only asks for n_batch == 1 here.
        ggml_tensor * k3 = ggml_reshape_3d(ctx, k, cs.kw, 1, cs.c);
        ggml_tensor * d3 = ggml_reshape_3d(ctx, d, cs.w, cs.c, n_batch);
        out              = conf::conv_1d_dw_f32(ctx, k3, d3, cs.s0, cs.p0, 1);
    } else {
        out = conf::conv_2d_dw_f32(ctx, k, d, cs.s0, cs.s1, cs.p0, cs.p1, 1, 1);
    }
    out = ggml_cont(ctx, out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Every non-view node must run on this device, or the case is
    // unsupported there (the caller decides whether that is a failure).
    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (!ggml_backend_supports_op(be, node)) {
            supported = false;
        }
    }
    if (!supported) {
        ggml_free(ctx);
        return res;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) {
        ggml_free(ctx);
        return res;
    }

    if (ktype == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> k16(kernel.size());
        for (size_t i = 0; i < kernel.size(); ++i) {
            k16[i] = ggml_fp32_to_fp16(kernel[i]);
        }
        ggml_backend_tensor_set(k, k16.data(), 0, ggml_nbytes(k));
    } else {
        ggml_backend_tensor_set(k, kernel.data(), 0, ggml_nbytes(k));
    }
    ggml_backend_tensor_set(d, data.data(), 0, ggml_nbytes(d));

    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
        res.supported = true;
        res.out.resize(static_cast<size_t>(ggml_nelements(out)));
        ggml_backend_tensor_get(out, res.out.data(), 0, ggml_nbytes(out));
    } else {
        std::fprintf(stderr, "FAIL graph compute on %s\n", ggml_backend_name(be));
        ++g_failures;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// max |a-b| / (1 + |ref|) over all elements.
double max_rel(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])) /
                         (1.0 + std::fabs(static_cast<double>(b[i])));
        if (!(e <= worst)) {
            worst = e;  // also latches NaN
        }
    }
    return worst;
}

bool must_support(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
    if (t == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }
    const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    return reg != nullptr && std::strcmp(reg, "Metal") == 0;
}

void run_device(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
    if (t == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return;  // BLAS etc.: not a standalone compute backend
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be == nullptr) {
        std::fprintf(stderr, "note: could not init %s, skipped\n", ggml_backend_dev_name(dev));
        return;
    }
    const char * name     = ggml_backend_name(be);
    const bool   required = must_support(dev);

    for (const Case & cs : kCases) {
        for (ggml_type ktype : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            const int n_batch = 3;
            const int kn      = cs.kw * cs.kh * cs.c;
            const int dn      = cs.w * cs.h * cs.c;

            std::vector<float> kernel(static_cast<size_t>(kn));
            for (float & v : kernel) {
                v = next_uniform();
                if (ktype == GGML_TYPE_F16) {
                    v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
                }
            }
            std::vector<float> data(static_cast<size_t>(dn * n_batch));
            for (float & v : data) {
                v = next_uniform();
            }
            const std::vector<float> ref = reference(cs, kernel, data, n_batch);

            const RunResult direct = run(be, cs, ktype, Algo::Direct, kernel, data, n_batch);
            if (!direct.supported) {
                if (required) {
                    std::fprintf(stderr, "FAIL %s: CONV_2D_DW unsupported for %s (%s kernel)\n", name, cs.name,
                                 ggml_type_name(ktype));
                    ++g_failures;
                } else {
                    std::fprintf(stderr, "note: %s: CONV_2D_DW unsupported for %s (%s kernel), skipped\n", name,
                                 cs.name, ggml_type_name(ktype));
                }
                continue;
            }

            // Direct vs the double reference: F32 rounding only.
            const double e_ref = max_rel(direct.out, ref);
            std::printf("%-8s %-22s %-4s direct  vs ref    %.2e\n", name, cs.name, ggml_type_name(ktype), e_ref);
            CHECK(e_ref < 1e-5);

            // Batch invariance: utterance n of the batch == utterance n alone.
            const size_t per_out = direct.out.size() / n_batch;
            for (int n = 0; n < n_batch; ++n) {
                std::vector<float> one(data.begin() + static_cast<std::ptrdiff_t>(n) * dn,
                                       data.begin() + static_cast<std::ptrdiff_t>(n + 1) * dn);
                const RunResult    solo = run(be, cs, ktype, Algo::Direct, kernel, one, 1);
                CHECK(solo.supported);
                CHECK(solo.out.size() == per_out);
                if (solo.out.size() == per_out) {
                    const bool same =
                        std::memcmp(solo.out.data(), direct.out.data() + n * per_out, per_out * sizeof(float)) == 0;
                    if (!same) {
                        std::fprintf(stderr, "FAIL %s %s %s: batch item %d differs from solo run\n", name, cs.name,
                                     ggml_type_name(ktype), n);
                        ++g_failures;
                    }
                }
            }

            // Direct vs the im2col helper it replaces (single utterance), a
            // cross-check rather than the accuracy gate (that is the reference
            // above): the F16-kernel im2col rounds the activations to F16,
            // and Vulkan's im2col + mul_mat route carries F16-level error even
            // for F32 kernels (~7e-4 against the reference), so the bound is
            // the F16 one throughout.
            std::vector<float> first(data.begin(), data.begin() + dn);
            const RunResult    im2col = run(be, cs, ktype, Algo::Im2col, kernel, first, 1);
            if (im2col.supported) {
                std::vector<float> direct0(direct.out.begin(),
                                           direct.out.begin() + static_cast<std::ptrdiff_t>(per_out));
                const double       e   = max_rel(direct0, im2col.out);
                const double       tol = 5e-3;
                std::printf("%-8s %-22s %-4s direct  vs im2col %.2e\n", name, cs.name, ggml_type_name(ktype), e);
                CHECK(e < tol);
            } else {
                std::fprintf(stderr, "note: %s: im2col form unsupported for %s (%s kernel)\n", name, cs.name,
                             ggml_type_name(ktype));
            }
        }
    }
    ggml_backend_free(be);
}

}  // namespace

int main() {
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        run_device(ggml_backend_dev_get(i));
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("OK\n");
    return EXIT_SUCCESS;
}
