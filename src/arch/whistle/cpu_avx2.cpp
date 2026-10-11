// arch/whistle/cpu_avx2.cpp - AVX2 + FMA Whistle CPU kernels (see cpu_avx2.h).

#include "cpu_avx2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#if WHISTLE_HAVE_AVX2_KERNELS
#    include <immintrin.h>
#    define WHISTLE_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif

namespace transcribe::whistle::avx2 {

#if WHISTLE_HAVE_AVX2_KERNELS

bool available() {
    static const bool ok = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    }();
    return ok;
}

namespace {

// ggml_v_expf (ggml-cpu/vec.h, AVX2 + FMA): max error 1.45 + 0.5 ulp.
WHISTLE_AVX2 inline __m256 v_expf(__m256 x) {
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

// out = A^T (Z B) for Z [16][32], B [32][32] (row j), A [16][16] (row i,
// column k): out[k*32 + l] = sum_ij z[i*32 + j] a[i][k] b[j][l].
WHISTLE_AVX2 void kron_16x32(const float * z, const float * a, const float * b, float * tmp, float * out) {
    for (int i = 0; i < 16; i += 2) {
        __m256 c0[4], c1[4];
        for (int v = 0; v < 4; ++v) {
            c0[v] = _mm256_setzero_ps();
            c1[v] = _mm256_setzero_ps();
        }
        const float * z0 = z + i * 32;
        const float * z1 = z0 + 32;
        for (int j = 0; j < 32; ++j) {
            const float * br = b + j * 32;
            const __m256  s0 = _mm256_broadcast_ss(z0 + j);
            const __m256  s1 = _mm256_broadcast_ss(z1 + j);
            for (int v = 0; v < 4; ++v) {
                const __m256 bv = _mm256_loadu_ps(br + 8 * v);
                c0[v]           = _mm256_fmadd_ps(s0, bv, c0[v]);
                c1[v]           = _mm256_fmadd_ps(s1, bv, c1[v]);
            }
        }
        for (int v = 0; v < 4; ++v) {
            _mm256_storeu_ps(tmp + i * 32 + 8 * v, c0[v]);
            _mm256_storeu_ps(tmp + (i + 1) * 32 + 8 * v, c1[v]);
        }
    }
    for (int k = 0; k < 16; k += 2) {
        __m256 c0[4], c1[4];
        for (int v = 0; v < 4; ++v) {
            c0[v] = _mm256_setzero_ps();
            c1[v] = _mm256_setzero_ps();
        }
        for (int i = 0; i < 16; ++i) {
            const float * tr = tmp + i * 32;
            const __m256  s0 = _mm256_broadcast_ss(a + i * 16 + k);
            const __m256  s1 = _mm256_broadcast_ss(a + i * 16 + k + 1);
            for (int v = 0; v < 4; ++v) {
                const __m256 tv = _mm256_loadu_ps(tr + 8 * v);
                c0[v]           = _mm256_fmadd_ps(s0, tv, c0[v]);
                c1[v]           = _mm256_fmadd_ps(s1, tv, c1[v]);
            }
        }
        for (int v = 0; v < 4; ++v) {
            _mm256_storeu_ps(out + k * 32 + 8 * v, c0[v]);
            _mm256_storeu_ps(out + (k + 1) * 32 + 8 * v, c1[v]);
        }
    }
}

}  // namespace

bool hmlp_supported(const HmlpHost & h) {
    return available() && h.d == 512 && h.ba == 16 && h.bb == 32;
}

// Same steps as encoder.cpp hmlp_cpu:
//   cond = 1 + softmax(x @ cond_v) @ cond_u
//   z = kron0(x * d1); z = silu(z[perm1] * d2 * cond + b2)
//   z = kron1(z); z = kron2(z[perm2] * d3); out = z * d4
WHISTLE_AVX2 void hmlp_row(const HmlpHost & h, const float * x, float * out, float * scratch) {
    constexpr int d    = 512;
    float *       z    = scratch;
    float *       y    = scratch + d;
    float *       tmp  = scratch + 2 * d;
    float *       cond = scratch + 3 * d;

    // cond logits: 4 interleaved accumulators over k.
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int k = 0; k < d; k += 4) {
        for (int q = 0; q < 4; ++q) {
            acc[q] = _mm256_fmadd_ps(_mm256_loadu_ps(h.cond_v + (k + q) * 8), _mm256_broadcast_ss(x + k + q), acc[q]);
        }
    }
    alignas(32) float c[8];
    _mm256_store_ps(c, _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
    float mx = c[0];
    for (int r = 1; r < 8; ++r) {
        mx = std::max(mx, c[r]);
    }
    float sum = 0.0f;
    for (float & v : c) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (float & v : c) {
        v /= sum;
    }
    for (int k = 0; k < d; k += 8) {
        __m256 a = _mm256_set1_ps(1.0f);
        for (int r = 0; r < 8; ++r) {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(h.cond_u + r * d + k), _mm256_set1_ps(c[r]), a);
        }
        _mm256_storeu_ps(cond + k, a);
    }

    for (int k = 0; k < d; k += 8) {
        _mm256_storeu_ps(z + k, _mm256_mul_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(h.d1 + k)));
    }
    kron_16x32(z, h.a[0], h.b[0], tmp, y);
    const __m256 one = _mm256_set1_ps(1.0f);
    for (int k = 0; k < d; k += 8) {
        const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(h.perm1 + k));
        __m256        v = _mm256_mul_ps(_mm256_i32gather_ps(y, p, 4), _mm256_loadu_ps(h.d2 + k));
        v               = _mm256_fmadd_ps(v, _mm256_loadu_ps(cond + k), _mm256_loadu_ps(h.b2 + k));
        v               = _mm256_div_ps(v, _mm256_add_ps(one, v_expf(_mm256_sub_ps(_mm256_setzero_ps(), v))));
        _mm256_storeu_ps(z + k, v);
    }
    kron_16x32(z, h.a[1], h.b[1], tmp, y);
    for (int k = 0; k < d; k += 8) {
        const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(h.perm2 + k));
        _mm256_storeu_ps(z + k, _mm256_mul_ps(_mm256_i32gather_ps(y, p, 4), _mm256_loadu_ps(h.d3 + k)));
    }
    kron_16x32(z, h.a[2], h.b[2], tmp, y);
    for (int k = 0; k < d; k += 8) {
        _mm256_storeu_ps(out + k, _mm256_mul_ps(_mm256_loadu_ps(y + k), _mm256_loadu_ps(h.d4 + k)));
    }
}

namespace {

WHISTLE_AVX2 inline float hsum(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    x        = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x        = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}

// Horizontal sums of two vectors -> {sum a, sum b}.
WHISTLE_AVX2 inline void hsum2(__m256 a, __m256 b, float & sa, float & sb) {
    const __m256 ab = _mm256_hadd_ps(a, b);
    __m128       x  = _mm_add_ps(_mm256_extractf128_ps(ab, 1), _mm256_castps256_ps128(ab));
    x               = _mm_hadd_ps(x, x);
    sa              = _mm_cvtss_f32(x);
    sb              = _mm_cvtss_f32(_mm_movehdup_ps(x));
}

}  // namespace

namespace {

// Softmax numerators of one score row in place (tail to a multiple of 8 set to
// 0); returns 1 / sum.
WHISTLE_AVX2 inline float softmax_exp(float * sc, int n) {
    const int n8 = (n + 7) / 8 * 8;
    for (int i = n; i < n8; ++i) {
        sc[i] = -INFINITY;
    }
    __m256 mv = _mm256_set1_ps(-INFINITY);
    for (int i = 0; i < n8; i += 8) {
        mv = _mm256_max_ps(mv, _mm256_loadu_ps(sc + i));
    }
    __m128 m4       = _mm_max_ps(_mm256_extractf128_ps(mv, 1), _mm256_castps256_ps128(mv));
    m4              = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4              = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const __m256 mx = _mm256_set1_ps(_mm_cvtss_f32(m4));
    __m256       sv = _mm256_setzero_ps();
    for (int i = 0; i < n8; i += 8) {
        const __m256 e = v_expf(_mm256_sub_ps(_mm256_loadu_ps(sc + i), mx));
        _mm256_storeu_ps(sc + i, e);
        sv = _mm256_add_ps(sv, e);
    }
    return 1.0f / hsum(sv);
}

template <int NQ>
WHISTLE_AVX2 void attn_multi_fixed(const float * q,
                                   size_t        qs,
                                   const float * k,
                                   size_t        ks,
                                   int           D,
                                   const float * vt,
                                   size_t        vs,
                                   int           DV,
                                   int           n,
                                   const float * mask,
                                   size_t        ms,
                                   float         scale,
                                   float *       sc,
                                   float *       out,
                                   size_t        os,
                                   float *       probs,
                                   size_t        ps) {
    const size_t ss = static_cast<size_t>((n + 7) / 8 * 8);  // score row stride
    // scores: two keys x NQ queries per pass over D
    int          t  = 0;
    for (; t + 2 <= n; t += 2) {
        const float * k0 = k + static_cast<size_t>(t) * ks;
        const float * k1 = k0 + ks;
        __m256        a0[NQ], a1[NQ];
        for (int j = 0; j < NQ; ++j) {
            a0[j] = _mm256_setzero_ps();
            a1[j] = _mm256_setzero_ps();
        }
        for (int c = 0; c < D; c += 8) {
            const __m256 kv0 = _mm256_loadu_ps(k0 + c);
            const __m256 kv1 = _mm256_loadu_ps(k1 + c);
            for (int j = 0; j < NQ; ++j) {
                const __m256 qv = _mm256_loadu_ps(q + j * qs + c);
                a0[j]           = _mm256_fmadd_ps(qv, kv0, a0[j]);
                a1[j]           = _mm256_fmadd_ps(qv, kv1, a1[j]);
            }
        }
        for (int j = 0; j < NQ; ++j) {
            float s0, s1;
            hsum2(a0[j], a1[j], s0, s1);
            s0 *= scale;
            s1 *= scale;
            if (mask != nullptr) {
                s0 += mask[j * ms + t];
                s1 += mask[j * ms + t + 1];
            }
            sc[j * ss + t]     = s0;
            sc[j * ss + t + 1] = s1;
        }
    }
    for (; t < n; ++t) {
        const float * kt = k + static_cast<size_t>(t) * ks;
        for (int j = 0; j < NQ; ++j) {
            __m256 a = _mm256_setzero_ps();
            for (int c = 0; c < D; c += 8) {
                a = _mm256_fmadd_ps(_mm256_loadu_ps(q + j * qs + c), _mm256_loadu_ps(kt + c), a);
            }
            sc[j * ss + t] = hsum(a) * scale + (mask != nullptr ? mask[j * ms + t] : 0.0f);
        }
    }
    float inv[NQ];
    for (int j = 0; j < NQ; ++j) {
        inv[j] = softmax_exp(sc + j * ss, n);
        if (probs != nullptr) {
            for (int u = 0; u < n; ++u) {
                probs[j * ps + u] = sc[j * ss + u] * inv[j];
            }
        }
    }
    // out: two output dims x NQ queries per pass over the positions. Runs over
    // n rounded up to 8 (callers guarantee Vt rows that long): the scores
    // there are exactly 0, and the V slack is finite (zeroed cache slack).
    const int nv = (n + 7) / 8 * 8;
    for (int i = 0; i < DV; i += 2) {
        const float * v0 = vt + static_cast<size_t>(i) * vs;
        const float * v1 = v0 + vs;
        __m256        a0[NQ], a1[NQ];
        for (int j = 0; j < NQ; ++j) {
            a0[j] = _mm256_setzero_ps();
            a1[j] = _mm256_setzero_ps();
        }
        for (int u = 0; u < nv; u += 8) {
            const __m256 vv0 = _mm256_loadu_ps(v0 + u);
            const __m256 vv1 = _mm256_loadu_ps(v1 + u);
            for (int j = 0; j < NQ; ++j) {
                const __m256 pv = _mm256_loadu_ps(sc + j * ss + u);
                a0[j]           = _mm256_fmadd_ps(pv, vv0, a0[j]);
                a1[j]           = _mm256_fmadd_ps(pv, vv1, a1[j]);
            }
        }
        for (int j = 0; j < NQ; ++j) {
            float o0, o1;
            hsum2(a0[j], a1[j], o0, o1);
            out[j * os + i]     = o0 * inv[j];
            out[j * os + i + 1] = o1 * inv[j];
        }
    }
}

}  // namespace

WHISTLE_AVX2 void attn_multi(int           nq,
                             const float * q,
                             size_t        qs,
                             const float * k,
                             size_t        ks,
                             int           D,
                             const float * vt,
                             size_t        vs,
                             int           DV,
                             int           n,
                             const float * mask,
                             size_t        ms,
                             float         scale,
                             float *       sc,
                             float *       out,
                             size_t        os,
                             float *       probs,
                             size_t        ps) {
    switch (nq) {
        case 4:
            return attn_multi_fixed<4>(q, qs, k, ks, D, vt, vs, DV, n, mask, ms, scale, sc, out, os, probs, ps);
        case 5:
            return attn_multi_fixed<5>(q, qs, k, ks, D, vt, vs, DV, n, mask, ms, scale, sc, out, os, probs, ps);
        default:
            for (int j = 0; j < nq; ++j) {
                attn_multi_fixed<1>(q + j * qs, qs, k, ks, D, vt, vs, DV, n, mask != nullptr ? mask + j * ms : nullptr,
                                    ms, scale, sc, out + j * os, os, probs != nullptr ? probs + j * ps : nullptr, ps);
            }
    }
}

WHISTLE_AVX2 void sigmoid_mul(const float * o, const float * g, float * out, int64_t n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int64_t      i   = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 e = v_expf(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(g + i)));
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(o + i), _mm256_add_ps(one, e)));
    }
    for (; i < n; ++i) {
        out[i] = o[i] * (1.0f / (1.0f + std::exp(-g[i])));
    }
}

namespace {

// Row order of the scales / int32 results after the hadd in q8x8_gemm.
constexpr int kQ8x8Perm[8] = { 0, 1, 4, 5, 2, 3, 6, 7 };

struct BlockQ8_0 {
    uint16_t d;
    int8_t   qs[32];
};

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block");

template <int NC>
WHISTLE_AVX2 inline void q8x8_cols(const uint8_t * W,
                                   int64_t         nbk,
                                   const Q8Act *   cols,
                                   int64_t         g,
                                   float *         out,
                                   size_t          ldo) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256        acc[NC];
    for (int j = 0; j < NC; ++j) {
        acc[j] = _mm256_setzero_ps();
    }
    const uint8_t * wg = W + static_cast<size_t>(g) * static_cast<size_t>(nbk) * kQ8x8Block;
    for (int64_t b = 0; b < nbk; ++b) {
        const uint8_t * wb = wg + static_cast<size_t>(b) * kQ8x8Block;
        __m256i         lo[NC], hi[NC];
        for (int j = 0; j < NC; ++j) {
            lo[j] = _mm256_setzero_si256();
            hi[j] = _mm256_setzero_si256();
        }
        for (int ch = 0; ch < 4; ++ch) {
            const __m256i wl = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb + 16 + ch * 64));
            const __m256i wh = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wb + 16 + ch * 64 + 32));
            const __m256i al = _mm256_sign_epi8(wl, wl);
            const __m256i ah = _mm256_sign_epi8(wh, wh);
            for (int j = 0; j < NC; ++j) {
                int64_t a8;
                std::memcpy(&a8, cols[j * nbk + b].qs + ch * 8, sizeof(a8));
                const __m256i ac = _mm256_set1_epi64x(a8);
                lo[j] = _mm256_add_epi32(lo[j],
                                         _mm256_madd_epi16(_mm256_maddubs_epi16(al, _mm256_sign_epi8(ac, wl)), ones));
                hi[j] = _mm256_add_epi32(hi[j],
                                         _mm256_madd_epi16(_mm256_maddubs_epi16(ah, _mm256_sign_epi8(ac, wh)), ones));
            }
        }
        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(wb)));
        for (int j = 0; j < NC; ++j) {
            const __m256 sc = _mm256_mul_ps(wd, _mm256_set1_ps(cols[j * nbk + b].d));
            acc[j]          = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_hadd_epi32(lo[j], hi[j])), sc, acc[j]);
        }
    }
    for (int j = 0; j < NC; ++j) {
        alignas(32) float t[8];
        _mm256_store_ps(t, acc[j]);
        float * o = out + j * ldo + g * 8;
        for (int r = 0; r < 8; ++r) {
            o[kQ8x8Perm[r]] = t[r];
        }
    }
}

}  // namespace

void q8x8_pack(void * data, int64_t rows, int64_t k) {
    const int64_t        nbk = k / 32;
    const size_t         gsz = static_cast<size_t>(nbk) * kQ8x8Block;
    std::vector<uint8_t> tmp(gsz);
    for (int64_t g = 0; g < rows / 8; ++g) {
        uint8_t *         base = static_cast<uint8_t *>(data) + static_cast<size_t>(g) * gsz;
        const BlockQ8_0 * src  = reinterpret_cast<const BlockQ8_0 *>(base);  // 8 rows x nbk blocks
        for (int64_t b = 0; b < nbk; ++b) {
            uint8_t * o = tmp.data() + static_cast<size_t>(b) * kQ8x8Block;
            for (int r = 0; r < 8; ++r) {
                std::memcpy(o + 2 * r, &src[kQ8x8Perm[r] * nbk + b].d, 2);
            }
            for (int ch = 0; ch < 4; ++ch) {
                for (int r = 0; r < 8; ++r) {
                    std::memcpy(o + 16 + ch * 64 + (r / 4) * 32 + (r % 4) * 8, src[r * nbk + b].qs + ch * 8, 8);
                }
            }
        }
        std::memcpy(base, tmp.data(), gsz);
    }
}

// ggml-cpu/arch/x86/quants.c quantize_row_q8_0 (AVX2 path), with the scale
// kept as its fp16 value converted back to f32.
WHISTLE_AVX2 void q8_quantize(const float * x, int64_t k, Q8Act * out) {
    for (int64_t i = 0; i < k / 32; ++i, x += 32) {
        __m256       v0         = _mm256_loadu_ps(x);
        __m256       v1         = _mm256_loadu_ps(x + 8);
        __m256       v2         = _mm256_loadu_ps(x + 16);
        __m256       v3         = _mm256_loadu_ps(x + 24);
        const __m256 signBit    = _mm256_set1_ps(-0.0f);
        __m256       maxAbs     = _mm256_andnot_ps(signBit, v0);
        maxAbs                  = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs                  = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs                  = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));
        __m128 max4             = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4                    = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4                    = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float   maxScalar = _mm_cvtss_f32(max4);
        const float   d         = maxScalar / 127.f;
        const __m128i dh        = _mm_cvtps_ph(_mm_set_ss(d), _MM_FROUND_TO_NEAREST_INT);
        out[i].d                = _mm_cvtss_f32(_mm_cvtph_ps(dh));
        const float  id         = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul        = _mm256_set1_ps(id);
        __m256i      i0         = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST));
        __m256i      i1         = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST));
        __m256i      i2         = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST));
        __m256i      i3         = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST));
        i0                      = _mm256_packs_epi32(i0, i1);
        i2                      = _mm256_packs_epi32(i2, i3);
        i0                      = _mm256_packs_epi16(i0, i2);
        const __m256i perm      = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0                      = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out[i].qs), i0);
    }
}

WHISTLE_AVX2 void q8x8_gemm(const uint8_t * W,
                            int64_t         k,
                            const Q8Act *   cols,
                            int             ncols,
                            int64_t         g0,
                            int64_t         g1,
                            float *         out,
                            size_t          ldo) {
    const int64_t nbk = k / 32;
    for (int64_t g = g0; g < g1; ++g) {
        int c = 0;
        for (; c + 4 <= ncols; c += 4) {
            q8x8_cols<4>(W, nbk, cols + c * nbk, g, out + c * ldo, ldo);
        }
        switch (ncols - c) {
            case 3:
                q8x8_cols<3>(W, nbk, cols + c * nbk, g, out + c * ldo, ldo);
                break;
            case 2:
                q8x8_cols<2>(W, nbk, cols + c * nbk, g, out + c * ldo, ldo);
                break;
            case 1:
                q8x8_cols<1>(W, nbk, cols + c * nbk, g, out + c * ldo, ldo);
                break;
            default:
                break;
        }
    }
}

namespace {

// conv0 + SiLU for conv0 output row iy: dst [W1][C].
WHISTLE_AVX2 void stem_conv0_row(const float * mel,
                                 int64_t       W0,
                                 int64_t       H0,
                                 size_t        ms,
                                 const float * w0,
                                 int64_t       C,
                                 int64_t       W1,
                                 int64_t       iy,
                                 float *       dst) {
    const __m256 one = _mm256_set1_ps(1.0f);
    for (int64_t ox = 0; ox < W1; ++ox) {
        float v[9];
        for (int kt = 0; kt < 3; ++kt) {
            const int64_t y = iy * 2 - 1 + kt;
            for (int kf = 0; kf < 3; ++kf) {
                const int64_t x = ox * 2 - 1 + kf;
                v[kt * 3 + kf]  = y < 0 || y >= H0 || x < 0 || x >= W0 ? 0.0f : mel[static_cast<size_t>(y) * ms + x];
            }
        }
        float * o = dst + ox * C;
        for (int64_t c = 0; c < C; c += 8) {
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(w0 + c), _mm256_set1_ps(v[0]));
            for (int j = 1; j < 9; ++j) {  // separate mul + add: the stem_conv0_cpu sums
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(w0 + j * C + c), _mm256_set1_ps(v[j])));
            }
            // ggml_v_silu
            const __m256 e = v_expf(_mm256_sub_ps(_mm256_setzero_ps(), acc));
            _mm256_storeu_ps(o + c, _mm256_div_ps(acc, _mm256_add_ps(one, e)));
        }
    }
}

}  // namespace

WHISTLE_AVX2 void stem_front(const float * mel,
                             int64_t       W0,
                             int64_t       H0,
                             size_t        ms,
                             const float * w0,
                             const float * wdw,
                             int64_t       C,
                             float *       out,
                             int64_t       oy0,
                             int64_t       oy1,
                             float *       scratch) {
    const int64_t W1 = (W0 - 1) / 2 + 1, H1 = (H0 - 1) / 2 + 1, W2 = (W1 - 1) / 2 + 1;
    int64_t       tag[3] = { -1, -1, -1 };  // conv0 row held in each scratch slot
    for (int64_t oy = oy0; oy < oy1; ++oy) {
        const float * rows[3] = { nullptr, nullptr, nullptr };
        for (int ky = 0; ky < 3; ++ky) {
            const int64_t iy = oy * 2 - 1 + ky;
            if (iy < 0 || iy >= H1) {
                continue;
            }
            float * slot = scratch + (iy % 3) * W1 * C;
            if (tag[iy % 3] != iy) {
                stem_conv0_row(mel, W0, H0, ms, w0, C, W1, iy, slot);
                tag[iy % 3] = iy;
            }
            rows[ky] = slot;
        }
        for (int64_t ox = 0; ox < W2; ++ox) {
            float * o = out + (oy * W2 + ox) * C;
            for (int64_t c = 0; c < C; c += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (int ky = 0; ky < 3; ++ky) {
                    if (rows[ky] == nullptr) {
                        continue;
                    }
                    for (int kx = 0; kx < 3; ++kx) {
                        const int64_t ix = ox * 2 - 1 + kx;
                        if (ix < 0 || ix >= W1) {
                            continue;
                        }
                        // GGML_F32_VEC_FMA(sum, k, s)
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(wdw + (ky * 3 + kx) * C + c),
                                              _mm256_loadu_ps(rows[ky] + ix * C + c), sum);
                    }
                }
                _mm256_storeu_ps(o + c, sum);
            }
        }
    }
}

WHISTLE_AVX2 double log_sum_exp(const float * x, int64_t n) {
    __m256  mv = _mm256_set1_ps(-INFINITY);
    int64_t i  = 0;
    for (; i + 8 <= n; i += 8) {
        mv = _mm256_max_ps(mv, _mm256_loadu_ps(x + i));
    }
    __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(mv, 1), _mm256_castps256_ps128(mv));
    m4        = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4        = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    float mx  = _mm_cvtss_f32(m4);
    for (int64_t j = i; j < n; ++j) {
        mx = std::max(mx, x[j]);
    }
    const __m256 mxv = _mm256_set1_ps(mx);
    __m256d      s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    for (i = 0; i + 8 <= n; i += 8) {
        const __m256 e = v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i), mxv));
        s0             = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm256_castps256_ps128(e)));
        s1             = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm256_extractf128_ps(e, 1)));
    }
    alignas(32) double t[4];
    _mm256_store_pd(t, _mm256_add_pd(s0, s1));
    double sum = (t[0] + t[1]) + (t[2] + t[3]);
    for (; i < n; ++i) {
        sum += std::exp(static_cast<double>(x[i]) - mx);
    }
    return mx + std::log(sum);
}

#else  // !WHISTLE_HAVE_AVX2_KERNELS

bool available() {
    return false;
}

bool hmlp_supported(const HmlpHost &) {
    return false;
}

void hmlp_row(const HmlpHost &, const float *, float *, float *) {}

void sigmoid_mul(const float *, const float *, float *, int64_t) {}

void q8x8_pack(void *, int64_t, int64_t) {}

double log_sum_exp(const float *, int64_t) {
    return 0.0;
}

void stem_front(const float *,
                int64_t,
                int64_t,
                size_t,
                const float *,
                const float *,
                int64_t,
                float *,
                int64_t,
                int64_t,
                float *) {}

void q8_quantize(const float *, int64_t, Q8Act *) {}

void q8x8_gemm(const uint8_t *, int64_t, const Q8Act *, int, int64_t, int64_t, float *, size_t) {}

void attn_multi(int,
                const float *,
                size_t,
                const float *,
                size_t,
                int,
                const float *,
                size_t,
                int,
                int,
                const float *,
                size_t,
                float,
                float *,
                float *,
                size_t,
                float *,
                size_t) {}

#endif

}  // namespace transcribe::whistle::avx2
