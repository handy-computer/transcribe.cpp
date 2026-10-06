// arch/ecapa_tdnn/cpu_ops.cpp - fused CPU kernels; see cpu_ops.h.
//
// Threading: elementwise kernels split the frame axis (rows of [C, T]);
// time reductions split the channel axis into blocks, so every thread walks
// all T rows over its own channels and each channel's accumulation order is
// the same sequential-over-t order as ggml's reductions, for any thread
// count.

#include "cpu_ops.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#    define ECAPA_OPS_X86 1
#    include <immintrin.h>
#else
#    define ECAPA_OPS_X86 0
#endif

namespace transcribe::ecapa_tdnn::cpu {

namespace {

#if ECAPA_OPS_X86
#    define ECAPA_AVX2 __attribute__((target("avx2,fma")))

// ggml's AVX2 expf (ggml-cpu/vec.h ggml_v_expf, ARM optimized-routines
// polynomial, max error 1.45 ulp): ggml's own soft_max uses it on AVX2
// builds, so this keeps the attention softmax on the arithmetic the stock
// graph had.
ECAPA_AVX2 inline __m256 v_expf(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b =
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f), _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256  k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c =
        _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g  = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                        _mm256_set1_epi32(static_cast<int>(0x82000000u)));
    const __m256  s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000)));
    const __m256  s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d =
        _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                         _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

bool cpu_has_avx2_fma() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

// r[c] = exp(r[c] - mx[c]) for c in [0, n), returning nothing; sums go to sum.
ECAPA_AVX2 void exp_row_avx2(float * r, const float * mx, double * sum, size_t n) {
    size_t c = 0;
    for (; c + 8 <= n; c += 8) {
        const __m256 e = v_expf(_mm256_sub_ps(_mm256_loadu_ps(r + c), _mm256_loadu_ps(mx + c)));
        _mm256_storeu_ps(r + c, e);
        alignas(32) float ev[8];
        _mm256_store_ps(ev, e);
        for (int i = 0; i < 8; ++i) {
            sum[c + static_cast<size_t>(i)] += static_cast<double>(ev[i]);
        }
    }
    for (; c < n; ++c) {
        const float e = std::exp(r[c] - mx[c]);
        r[c]          = e;
        sum[c] += static_cast<double>(e);
    }
}
#endif

// Elsewhere (ARM included) this is libm expf, which is not the NEON
// ggml_v_expf ggml's soft_max uses there: the softmax is within a few ulp of
// the stock graph's, not bit-identical.
void exp_row(float * r, const float * mx, double * sum, size_t n) {
#if ECAPA_OPS_X86
    static const bool avx2 = cpu_has_avx2_fma();
    if (avx2) {
        exp_row_avx2(r, mx, sum, n);
        return;
    }
#endif
    for (size_t c = 0; c < n; ++c) {
        const float e = std::exp(r[c] - mx[c]);
        r[c]          = e;
        sum[c] += static_cast<double>(e);
    }
}

// Per-node flags passed through the custom op's userdata. Static storage:
// the scheduler reads userdata at compute time, long after graph build.
struct EpilogueParams {
    bool has_bias;
    bool tanh_after;
};

const float * f32(const ggml_tensor * t) {
    return static_cast<const float *>(t->data);
}

float * f32_mut(ggml_tensor * t) {
    return static_cast<float *>(t->data);
}

size_t row_floats(const ggml_tensor * t) {
    return t->nb[1] / sizeof(float);
}

void thread_rows(int64_t n, int ith, int nth, int64_t & r0, int64_t & r1) {
    const int64_t per = (n + nth - 1) / nth;
    r0                = std::min<int64_t>(per * ith, n);
    r1                = std::min<int64_t>(r0 + per, n);
}

// Time reductions walk all T rows over a channel block. Blocks of
// kChanBlock channels (1 KiB per row) are dealt to the threads round robin:
// wide enough that each row read is a few whole cache lines, narrow enough
// that 3072 channels still spread over 8+ threads.
constexpr int64_t kChanBlock = 256;

template <typename F> void for_channel_blocks(int64_t C, int ith, int nth, F && f) {
    for (int64_t c0 = static_cast<int64_t>(ith) * kChanBlock; c0 < C; c0 += static_cast<int64_t>(nth) * kChanBlock) {
        f(c0, std::min(C, c0 + kChanBlock));
    }
}

// y = max(x + b, 0) * s + sh  (and tanh) over rows [r0, r1) of a [C, T]
// activation. The add / max / mul / add sequence is evaluated in float in
// the same order as ggml's add -> relu -> mul -> add chain, so the result is
// bit-identical to the stock ops.
void epilogue_rows(const float * in,
                   size_t        in_stride,
                   float *       out,
                   size_t        out_stride,
                   int64_t       C,
                   int64_t       r0,
                   int64_t       r1,
                   const float * b,
                   const float * s,
                   const float * sh,
                   bool          tanh_after) {
    for (int64_t t = r0; t < r1; ++t) {
        const float * x = in + static_cast<size_t>(t) * in_stride;
        float *       y = out + static_cast<size_t>(t) * out_stride;
        if (b != nullptr) {
            for (int64_t c = 0; c < C; ++c) {
                float v = x[c] + b[c];
                v       = v > 0.0f ? v : 0.0f;
                y[c]    = v * s[c] + sh[c];
            }
        } else {
            for (int64_t c = 0; c < C; ++c) {
                const float v = x[c] > 0.0f ? x[c] : 0.0f;
                y[c]          = v * s[c] + sh[c];
            }
        }
        if (tanh_after) {
            for (int64_t c = 0; c < C; ++c) {
                y[c] = std::tanh(y[c]);
            }
        }
    }
}

// dst is a view of src[0] (in place). src: [z, (b), scale, shift].
void epilogue_inplace_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto *  p  = static_cast<const EpilogueParams *>(ud);
    const int64_t C  = dst->ne[0];
    int           i  = 1;
    const float * b  = p->has_bias ? f32(dst->src[i++]) : nullptr;
    const float * s  = f32(dst->src[i++]);
    const float * sh = f32(dst->src[i++]);
    int64_t       r0 = 0;
    int64_t       r1 = 0;
    thread_rows(dst->ne[1], ith, nth, r0, r1);
    epilogue_rows(f32(dst), row_floats(dst), f32_mut(dst), row_floats(dst), C, r0, r1, b, s, sh, p->tanh_after);
}

// dst is a view of src[0]; src: [z0, z1, z2, b, scale, shift].
void epilogue_sum3_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) ud;
    const int64_t C   = dst->ne[0];
    const float * z1  = f32(dst->src[1]);
    const float * z2  = f32(dst->src[2]);
    const float * b   = f32(dst->src[3]);
    const float * s   = f32(dst->src[4]);
    const float * sh  = f32(dst->src[5]);
    float *       y   = f32_mut(dst);
    const size_t  st  = row_floats(dst);
    const size_t  st1 = row_floats(dst->src[1]);
    const size_t  st2 = row_floats(dst->src[2]);
    int64_t       r0  = 0;
    int64_t       r1  = 0;
    thread_rows(dst->ne[1], ith, nth, r0, r1);
    for (int64_t t = r0; t < r1; ++t) {
        float *       yr  = y + static_cast<size_t>(t) * st;
        const float * z1r = z1 + static_cast<size_t>(t) * st1;
        const float * z2r = z2 + static_cast<size_t>(t) * st2;
        for (int64_t c = 0; c < C; ++c) {
            // ((z0 + z1) + z2) + b: the stock graph's add order.
            float v = yr[c] + z1r[c];
            v       = v + z2r[c];
            v       = v + b[c];
            v       = v > 0.0f ? v : 0.0f;
            yr[c]   = v * s[c] + sh[c];
        }
    }
}

// dst [C]; src: [x [C, T]].
void time_mean_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) ud;
    const ggml_tensor * x  = dst->src[0];
    const int64_t       T  = x->ne[1];
    const size_t        st = row_floats(x);
    float *             y  = f32_mut(dst);
    double              acc[kChanBlock];
    for_channel_blocks(x->ne[0], ith, nth, [&](int64_t c0, int64_t c1) {
        const int64_t n = c1 - c0;
        std::fill(acc, acc + n, 0.0);
        for (int64_t t = 0; t < T; ++t) {
            const float * r = f32(x) + static_cast<size_t>(t) * st + c0;
            for (int64_t c = 0; c < n; ++c) {
                acc[c] += static_cast<double>(r[c]);
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            // ggml_mean: (float) sum, then /= (float) n.
            y[c0 + c] = static_cast<float>(acc[c]) / static_cast<float>(T);
        }
    });
}

// dst [C, T]; src: [x [C, T], s [C], res [C, T]].
void scale_add_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) ud;
    const ggml_tensor * x   = dst->src[0];
    const float *       s   = f32(dst->src[1]);
    const ggml_tensor * res = dst->src[2];
    const int64_t       C   = dst->ne[0];
    int64_t             r0  = 0;
    int64_t             r1  = 0;
    thread_rows(dst->ne[1], ith, nth, r0, r1);
    for (int64_t t = r0; t < r1; ++t) {
        const float * xr = f32(x) + static_cast<size_t>(t) * row_floats(x);
        const float * rr = f32(res) + static_cast<size_t>(t) * row_floats(res);
        float *       yr = f32_mut(dst) + static_cast<size_t>(t) * row_floats(dst);
        for (int64_t c = 0; c < C; ++c) {
            const float v = xr[c] * s[c];
            yr[c]         = v + rr[c];
        }
    }
}

// dst [2C] = [mean, sqrt(max(var, eps))]; src: [x [C, T]]; ud: const float * eps.
void time_stats_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    const float         eps = *static_cast<const float *>(ud);
    const ggml_tensor * x   = dst->src[0];
    const int64_t       C   = x->ne[0];
    const int64_t       T   = x->ne[1];
    const size_t        st  = row_floats(x);
    float *             y   = f32_mut(dst);
    double              acc[kChanBlock];
    float               mean[kChanBlock];
    for_channel_blocks(C, ith, nth, [&](int64_t c0, int64_t c1) {
        const int64_t n = c1 - c0;
        std::fill(acc, acc + n, 0.0);
        for (int64_t t = 0; t < T; ++t) {
            const float * r = f32(x) + static_cast<size_t>(t) * st + c0;
            for (int64_t c = 0; c < n; ++c) {
                acc[c] += static_cast<double>(r[c]);
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            mean[c] = static_cast<float>(acc[c]) / static_cast<float>(T);
            acc[c]  = 0.0;
        }
        for (int64_t t = 0; t < T; ++t) {
            const float * r = f32(x) + static_cast<size_t>(t) * st + c0;
            for (int64_t c = 0; c < n; ++c) {
                const float d = r[c] - mean[c];
                acc[c] += static_cast<double>(d * d);
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            float var     = static_cast<float>(acc[c]) / static_cast<float>(T);
            var           = var < eps ? eps : var;
            y[c0 + c]     = mean[c];
            y[C + c0 + c] = std::sqrt(var);
        }
    });
}

// In place over src[0] = logits [C, T]: softmax over t of (logits + bias)
// per channel. src: [logits, bias].
void softmax_time_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) ud;
    const float * b  = f32(dst->src[1]);
    const int64_t T  = dst->ne[1];
    const size_t  st = row_floats(dst);
    float *       l  = f32_mut(dst);
    float         mx[kChanBlock];
    double        sum[kChanBlock];
    float         inv[kChanBlock];
    for_channel_blocks(dst->ne[0], ith, nth, [&](int64_t c0, int64_t c1) {
        const int64_t n = c1 - c0;
        std::fill(mx, mx + n, -INFINITY);
        std::fill(sum, sum + n, 0.0);
        for (int64_t t = 0; t < T; ++t) {
            float * r = l + static_cast<size_t>(t) * st + c0;
            for (int64_t c = 0; c < n; ++c) {
                const float v = r[c] + b[c0 + c];
                r[c]          = v;
                mx[c]         = v > mx[c] ? v : mx[c];
            }
        }
        for (int64_t t = 0; t < T; ++t) {
            exp_row(l + static_cast<size_t>(t) * st + c0, mx, sum, static_cast<size_t>(n));
        }
        for (int64_t c = 0; c < n; ++c) {
            inv[c] = static_cast<float>(1.0 / sum[c]);  // ggml_vec_scale_f32(dp, 1.0/sum)
        }
        for (int64_t t = 0; t < T; ++t) {
            float * r = l + static_cast<size_t>(t) * st + c0;
            for (int64_t c = 0; c < n; ++c) {
                r[c] *= inv[c];
            }
        }
    });
}

// dst [2C] = [mu, sigma]; src: [x [C, T], w [C, T]]; ud: const float * eps.
void weighted_stats_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    const float         eps = *static_cast<const float *>(ud);
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * w   = dst->src[1];
    const int64_t       C   = x->ne[0];
    const int64_t       T   = x->ne[1];
    const size_t        sx  = row_floats(x);
    const size_t        sw  = row_floats(w);
    float *             y   = f32_mut(dst);
    double              acc[kChanBlock];
    float               mu[kChanBlock];
    for_channel_blocks(C, ith, nth, [&](int64_t c0, int64_t c1) {
        const int64_t n = c1 - c0;
        std::fill(acc, acc + n, 0.0);
        for (int64_t t = 0; t < T; ++t) {
            const float * xr = f32(x) + static_cast<size_t>(t) * sx + c0;
            const float * wr = f32(w) + static_cast<size_t>(t) * sw + c0;
            for (int64_t c = 0; c < n; ++c) {
                acc[c] += static_cast<double>(wr[c] * xr[c]);
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            mu[c]  = static_cast<float>(acc[c]);
            acc[c] = 0.0;
        }
        for (int64_t t = 0; t < T; ++t) {
            const float * xr = f32(x) + static_cast<size_t>(t) * sx + c0;
            const float * wr = f32(w) + static_cast<size_t>(t) * sw + c0;
            for (int64_t c = 0; c < n; ++c) {
                const float d = xr[c] - mu[c];
                acc[c] += static_cast<double>(wr[c] * (d * d));
            }
        }
        for (int64_t c = 0; c < n; ++c) {
            float sig     = static_cast<float>(acc[c]);
            sig           = sig < eps ? eps : sig;
            y[c0 + c]     = mu[c];
            y[C + c0 + c] = std::sqrt(sig);
        }
    });
}

// Res2Net step parameters, packed into the custom op's userdata pointer
// (they are small integers, and this keeps the graph free of side storage).
struct Res2Params {
    bool has_z;
    int  next;  // -1 = no next chunk
    int  kernel;
    int  dilation;
};

void * pack_res2(const Res2Params & p) {
    const uintptr_t v = (p.has_z ? 1u : 0u) | (static_cast<uintptr_t>(p.next + 1) & 0xffu) << 1 |
                        (static_cast<uintptr_t>(p.kernel) & 0xffu) << 9 |
                        (static_cast<uintptr_t>(p.dilation) & 0xffu) << 17;
    return reinterpret_cast<void *>(v);
}

Res2Params unpack_res2(const void * ud) {
    const uintptr_t v = reinterpret_cast<uintptr_t>(ud);
    return { (v & 1u) != 0, static_cast<int>((v >> 1) & 0xffu) - 1, static_cast<int>((v >> 9) & 0xffu),
             static_cast<int>((v >> 17) & 0xffu) };
}

// src: [h, (z, b, scale, shift)].
void res2_step_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    const Res2Params    p          = unpack_res2(ud);
    const ggml_tensor * h          = dst->src[0];
    const int64_t       w          = dst->ne[0];
    const int64_t       T          = h->ne[1];
    const int64_t       pad        = static_cast<int64_t>(p.dilation) * (p.kernel - 1) / 2;
    const int64_t       n_pad_rows = p.next >= 0 ? T + 2 * pad : 0;
    const int64_t       n_rows     = n_pad_rows + (p.has_z ? T : 0);
    const size_t        sd         = row_floats(dst);
    const size_t        sh_        = row_floats(h);

    const float * z  = nullptr;
    const float * b  = nullptr;
    const float * s  = nullptr;
    const float * sh = nullptr;
    size_t        sz = 0;
    if (p.has_z) {
        z  = f32(dst->src[1]);
        sz = row_floats(dst->src[1]);
        b  = f32(dst->src[2]);
        s  = f32(dst->src[3]);
        sh = f32(dst->src[4]);
    }

    // y(t) into out[0..w): ((Z_0 + Z_1) + ... + Z_{K-1}) + b, ReLU, BN.
    auto y_at = [&](int64_t t, float * out) {
        const float * r0 = z + static_cast<size_t>(t) * sz;
        for (int64_t c = 0; c < w; ++c) {
            out[c] = r0[c];
        }
        for (int k = 1; k < p.kernel; ++k) {
            const float * rk = z + static_cast<size_t>(t + static_cast<int64_t>(k) * p.dilation) * sz +
                               static_cast<size_t>(k) * static_cast<size_t>(w);
            for (int64_t c = 0; c < w; ++c) {
                out[c] += rk[c];
            }
        }
        for (int64_t c = 0; c < w; ++c) {
            float v = out[c] + b[c];
            v       = v > 0.0f ? v : 0.0f;
            out[c]  = v * s[c] + sh[c];
        }
    };

    std::vector<float> tmp(static_cast<size_t>(w));
    int64_t            r0 = 0;
    int64_t            r1 = 0;
    thread_rows(n_rows, ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r) {
        float * out = f32_mut(dst) + static_cast<size_t>(r) * sd;
        if (r >= n_pad_rows) {
            y_at(r - n_pad_rows, out);
            continue;
        }
        // torch "reflect" (edge not repeated); T > pad is checked at build.
        int64_t t = r - pad;
        if (t < 0) {
            t = -t;
        } else if (t >= T) {
            t = 2 * (T - 1) - t;
        }
        const float * hr = f32(h) + static_cast<size_t>(t) * sh_ + static_cast<size_t>(p.next) * static_cast<size_t>(w);
        if (p.has_z) {
            y_at(t, tmp.data());
            for (int64_t c = 0; c < w; ++c) {
                out[c] = hr[c] + tmp[static_cast<size_t>(c)];
            }
        } else {
            for (int64_t c = 0; c < w; ++c) {
                out[c] = hr[c];
            }
        }
    }
}

// dst [w, T + 2p] = reflect(h[next] + y); src: [h, y]; ud: packed params.
void res2_pad_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    const Res2Params    p   = unpack_res2(ud);
    const ggml_tensor * h   = dst->src[0];
    const ggml_tensor * y   = dst->src[1];
    const int64_t       w   = dst->ne[0];
    const int64_t       T   = h->ne[1];
    const int64_t       pad = static_cast<int64_t>(p.dilation) * (p.kernel - 1) / 2;
    int64_t             r0  = 0;
    int64_t             r1  = 0;
    thread_rows(dst->ne[1], ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r) {
        int64_t t = r - pad;  // torch "reflect", edge not repeated
        if (t < 0) {
            t = -t;
        } else if (t >= T) {
            t = 2 * (T - 1) - t;
        }
        const float * hr = f32(h) + static_cast<size_t>(t) * row_floats(h) + static_cast<size_t>(p.next * w);
        const float * yr = f32(y) + static_cast<size_t>(t) * row_floats(y);
        float *       o  = f32_mut(dst) + static_cast<size_t>(r) * row_floats(dst);
        for (int64_t c = 0; c < w; ++c) {
            o[c] = hr[c] + yr[c];
        }
    }
}

// dst [C, T]; src: [h, y_1 .. y_n] (y_i are [w, T] views).
void res2_gather_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) ud;
    const ggml_tensor * h = dst->src[0];
    const int64_t       C = dst->ne[0];
    int                 n = 0;
    while (n + 1 < GGML_MAX_SRC && dst->src[n + 1] != nullptr) {
        ++n;
    }
    const int64_t w  = dst->src[1]->ne[0];
    int64_t       r0 = 0;
    int64_t       r1 = 0;
    thread_rows(dst->ne[1], ith, nth, r0, r1);
    for (int64_t t = r0; t < r1; ++t) {
        float *       out = f32_mut(dst) + static_cast<size_t>(t) * row_floats(dst);
        const float * hr  = f32(h) + static_cast<size_t>(t) * row_floats(h);
        std::copy(hr, hr + w, out);
        for (int i = 1; i <= n; ++i) {
            const ggml_tensor * y  = dst->src[i];
            const float *       yr = f32(y) + static_cast<size_t>(t) * row_floats(y);
            std::copy(yr, yr + w, out + static_cast<size_t>(i) * static_cast<size_t>(w));
        }
        (void) C;
    }
}

const EpilogueParams k_ep_bias{ true, false };
const EpilogueParams k_ep_bias_tanh{ true, true };
const EpilogueParams k_ep_nobias{ false, false };
const EpilogueParams k_ep_nobias_tanh{ false, true };

void * ud(const void * p) {
    return const_cast<void *>(p);
}

}  // namespace

ggml_tensor * epilogue(ggml_context * ctx,
                       ggml_tensor *  z,
                       ggml_tensor *  b,
                       ggml_tensor *  scale,
                       ggml_tensor *  shift,
                       bool           tanh_after,
                       bool           inplace) {
    const EpilogueParams * p =
        b != nullptr ? (tanh_after ? &k_ep_bias_tanh : &k_ep_bias) : (tanh_after ? &k_ep_nobias_tanh : &k_ep_nobias);
    ggml_tensor * args[3];
    int           n = 0;
    if (b != nullptr) {
        args[n++] = b;
    }
    args[n++] = scale;
    args[n++] = shift;
    if (!inplace) {
        z = ggml_cont(ctx, z);
    }
    return ggml_custom_inplace(ctx, z, args, n, epilogue_inplace_fn, GGML_N_TASKS_MAX, ud(p));
}

ggml_tensor * epilogue_sum3(ggml_context * ctx,
                            ggml_tensor *  z0,
                            ggml_tensor *  z1,
                            ggml_tensor *  z2,
                            ggml_tensor *  b,
                            ggml_tensor *  scale,
                            ggml_tensor *  shift,
                            bool           inplace) {
    ggml_tensor * args[5] = { z1, z2, b, scale, shift };
    if (!inplace) {
        z0 = ggml_cont(ctx, z0);
    }
    return ggml_custom_inplace(ctx, z0, args, 5, epilogue_sum3_fn, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * time_mean(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * args[1] = { x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], 1, 1, 1, args, 1, time_mean_fn, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * scale_add(ggml_context * ctx, ggml_tensor * x, ggml_tensor * s, ggml_tensor * res) {
    ggml_tensor * args[3] = { x, s, res };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1, args, 3, scale_add_fn, GGML_N_TASKS_MAX,
                          nullptr);
}

ggml_tensor * time_stats(ggml_context * ctx, ggml_tensor * x, const float * eps) {
    ggml_tensor * args[1] = { x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 2 * x->ne[0], 1, 1, 1, args, 1, time_stats_fn, GGML_N_TASKS_MAX, ud(eps));
}

ggml_tensor * attn_stats(ggml_context * ctx,
                         ggml_tensor *  x,
                         ggml_tensor *  logits,
                         ggml_tensor *  bias,
                         const float *  eps) {
    ggml_tensor * sm_args[1] = { bias };
    ggml_tensor * w          = ggml_custom_inplace(ctx, logits, sm_args, 1, softmax_time_fn, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * args[2]    = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 2 * x->ne[0], 1, 1, 1, args, 2, weighted_stats_fn, GGML_N_TASKS_MAX,
                          ud(eps));
}

Res2Step res2_step(ggml_context * ctx,
                   ggml_tensor *  h,
                   int            next,
                   int            width,
                   ggml_tensor *  z,
                   ggml_tensor *  b,
                   ggml_tensor *  scale,
                   ggml_tensor *  shift,
                   int            kernel,
                   int            dilation) {
    Res2Step      r;
    const int64_t T          = h->ne[1];
    const int64_t pad        = static_cast<int64_t>(dilation) * (kernel - 1) / 2;
    const int64_t n_pad_rows = next >= 0 ? T + 2 * pad : 0;
    const int64_t n_rows     = n_pad_rows + (z != nullptr ? T : 0);

    ggml_tensor * args[5] = { h, z, b, scale, shift };
    const int     n_args  = z != nullptr ? 5 : 1;
    r.out = ggml_custom_4d(ctx, GGML_TYPE_F32, width, n_rows, 1, 1, args, n_args, res2_step_fn, GGML_N_TASKS_MAX,
                           pack_res2({ z != nullptr, next, kernel, dilation }));
    if (next >= 0) {
        r.pad = ggml_view_2d(ctx, r.out, width, n_pad_rows, r.out->nb[1], 0);
    }
    if (z != nullptr) {
        r.y = ggml_view_2d(ctx, r.out, width, T, r.out->nb[1], static_cast<size_t>(n_pad_rows) * r.out->nb[1]);
    }
    return r;
}

Res2Step res2_pad(ggml_context * ctx, ggml_tensor * h, int next, int width, ggml_tensor * y, int kernel, int dilation) {
    Res2Step      r;
    const int64_t T       = h->ne[1];
    const int64_t pad     = static_cast<int64_t>(dilation) * (kernel - 1) / 2;
    ggml_tensor * args[2] = { h, y };
    r.out = ggml_custom_4d(ctx, GGML_TYPE_F32, width, T + 2 * pad, 1, 1, args, 2, res2_pad_fn, GGML_N_TASKS_MAX,
                           pack_res2({ false, next, kernel, dilation }));
    r.pad = r.out;
    return r;
}

ggml_tensor * res2_gather(ggml_context * ctx, ggml_tensor * h, ggml_tensor * const * ys, int n_ys) {
    ggml_tensor * args[GGML_MAX_SRC - 1] = {};
    args[0]                              = h;
    for (int i = 0; i < n_ys; ++i) {
        args[i + 1] = ys[i];
    }
    return ggml_custom_4d(ctx, GGML_TYPE_F32, h->ne[0], h->ne[1], 1, 1, args, n_ys + 1, res2_gather_fn,
                          GGML_N_TASKS_MAX, nullptr);
}

}  // namespace transcribe::ecapa_tdnn::cpu
