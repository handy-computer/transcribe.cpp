// arch/ecapa_tdnn/cpu_gemm.cpp - see cpu_gemm.h.
//
// Kernel: a 16 x NR register tile (two 8-float rows of accumulators per
// output column, NR <= 6 columns): per k it widens 16 packed weights, then
// broadcasts one activation value per column and issues two FMAs per column.
// The activations are read in place (each output column's K values are
// contiguous in ggml's [K, N] layout), so nothing is packed at run time.
//
// Work split: the output is cut into row blocks of kRowBlock rows and column
// blocks sized so one block's activations stay in L2; blocks are dealt to
// the ggml threads round robin.
//
// Built with -ffp-contract=off (src/CMakeLists.txt): the epilogue's multiply
// and add must round separately, as the stock ggml_mul / ggml_add do.

#include "cpu_gemm.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#    define ECAPA_GEMM_X86 1
#    include <immintrin.h>
#else
#    define ECAPA_GEMM_X86 0
#endif

namespace transcribe::ecapa_tdnn::gemm {

namespace {

#if ECAPA_GEMM_X86

#    define ECAPA_TARGET __attribute__((target("avx2,fma,f16c")))

constexpr int     kNR = 6;    // output columns per micro-tile
constexpr int64_t kKC = 256;  // k block: 16 x 256 F16 weights = 8 KiB in L1
constexpr int64_t kNC = 192;  // column block: 256 x 192 packed F32 = 192 KiB in L2

// 16 x 6 micro-tile over k in [0, kc): a = packed weight panel at k0
// (16 values per k), bp = packed activations (6 values per k). Writes the
// raw partial sums to acc[12] (two __m256 per column); the caller loads /
// stores / applies the epilogue. The accumulators are named locals so the
// compiler keeps all twelve in registers through the k loop.
#    define ECAPA_FMA6(W0, W1, BP)                     \
        do {                                           \
            __m256 xb = _mm256_broadcast_ss((BP) + 0); \
            a00       = _mm256_fmadd_ps(W0, xb, a00);  \
            a01       = _mm256_fmadd_ps(W1, xb, a01);  \
            xb        = _mm256_broadcast_ss((BP) + 1); \
            a10       = _mm256_fmadd_ps(W0, xb, a10);  \
            a11       = _mm256_fmadd_ps(W1, xb, a11);  \
            xb        = _mm256_broadcast_ss((BP) + 2); \
            a20       = _mm256_fmadd_ps(W0, xb, a20);  \
            a21       = _mm256_fmadd_ps(W1, xb, a21);  \
            xb        = _mm256_broadcast_ss((BP) + 3); \
            a30       = _mm256_fmadd_ps(W0, xb, a30);  \
            a31       = _mm256_fmadd_ps(W1, xb, a31);  \
            xb        = _mm256_broadcast_ss((BP) + 4); \
            a40       = _mm256_fmadd_ps(W0, xb, a40);  \
            a41       = _mm256_fmadd_ps(W1, xb, a41);  \
            xb        = _mm256_broadcast_ss((BP) + 5); \
            a50       = _mm256_fmadd_ps(W0, xb, a50);  \
            a51       = _mm256_fmadd_ps(W1, xb, a51);  \
        } while (0)

template <bool F16> ECAPA_TARGET void tile_core(const void * a, const float * bp, int64_t kc, __m256 * acc) {
    __m256 a00 = acc[0], a01 = acc[1], a10 = acc[2], a11 = acc[3], a20 = acc[4], a21 = acc[5];
    __m256 a30 = acc[6], a31 = acc[7], a40 = acc[8], a41 = acc[9], a50 = acc[10], a51 = acc[11];
    if constexpr (F16) {
        const uint16_t * ap = static_cast<const uint16_t *>(a);
        for (int64_t k = 0; k < kc; ++k) {
            const __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ap)));
            const __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ap + 8)));
            ECAPA_FMA6(w0, w1, bp);
            ap += 16;
            bp += kNR;
        }
    } else {
        const float * ap = static_cast<const float *>(a);
        for (int64_t k = 0; k < kc; ++k) {
            const __m256 w0 = _mm256_loadu_ps(ap);
            const __m256 w1 = _mm256_loadu_ps(ap + 8);
            ECAPA_FMA6(w0, w1, bp);
            ap += 16;
            bp += kNR;
        }
    }
    acc[0] = a00, acc[1] = a01, acc[2] = a10, acc[3] = a11, acc[4] = a20, acc[5] = a21;
    acc[6] = a30, acc[7] = a31, acc[8] = a40, acc[9] = a41, acc[10] = a50, acc[11] = a51;
}

#    undef ECAPA_FMA6

template <bool F16>
ECAPA_TARGET void tile(const void *  a,
                       const float * bp,
                       int64_t       kc,
                       float *       y,
                       size_t        ldy,
                       int           n_valid,
                       bool          first,
                       bool          last,
                       Epilogue      ep,
                       const float * b,
                       const float * s,
                       const float * sh) {
    alignas(32) __m256 acc[2 * kNR];
    for (int n = 0; n < kNR; ++n) {
        if (first || n >= n_valid) {
            acc[2 * n]     = _mm256_setzero_ps();
            acc[2 * n + 1] = _mm256_setzero_ps();
        } else {
            acc[2 * n]     = _mm256_loadu_ps(y + static_cast<size_t>(n) * ldy);
            acc[2 * n + 1] = _mm256_loadu_ps(y + static_cast<size_t>(n) * ldy + 8);
        }
    }
    tile_core<F16>(a, bp, kc, acc);

    if (last && ep != Epilogue::None) {
        const __m256 zero = _mm256_setzero_ps();
        const __m256 s0   = _mm256_loadu_ps(s);
        const __m256 s1   = _mm256_loadu_ps(s + 8);
        const __m256 h0   = _mm256_loadu_ps(sh);
        const __m256 h1   = _mm256_loadu_ps(sh + 8);
        for (int n = 0; n < kNR; ++n) {
            __m256 v0 = acc[2 * n];
            __m256 v1 = acc[2 * n + 1];
            if (b != nullptr) {
                v0 = _mm256_add_ps(v0, _mm256_loadu_ps(b));
                v1 = _mm256_add_ps(v1, _mm256_loadu_ps(b + 8));
            }
            v0             = _mm256_max_ps(v0, zero);
            v1             = _mm256_max_ps(v1, zero);
            acc[2 * n]     = _mm256_add_ps(_mm256_mul_ps(v0, s0), h0);
            acc[2 * n + 1] = _mm256_add_ps(_mm256_mul_ps(v1, s1), h1);
        }
    }
    for (int n = 0; n < n_valid; ++n) {
        float * yc = y + static_cast<size_t>(n) * ldy;
        _mm256_storeu_ps(yc, acc[2 * n]);
        _mm256_storeu_ps(yc + 8, acc[2 * n + 1]);
        if (last && ep == Epilogue::ReluBnTanh) {
            for (int r = 0; r < kPanel; ++r) {
                yc[r] = std::tanh(yc[r]);
            }
        }
    }
}

// Pack columns [j0, j0 + nc) x k [k0, k0 + kc) of x (column j's K values at
// x + j*ldx) into kNR-wide k-major micro-panels, zero filling past nc.
void pack_x(const float * x, size_t ldx, int64_t j0, int64_t nc, int64_t k0, int64_t kc, float * bp) {
    const int64_t n_panels = (nc + kNR - 1) / kNR;
    for (int64_t p = 0; p < n_panels; ++p) {
        float * dst = bp + static_cast<size_t>(p * kc * kNR);
        for (int n = 0; n < kNR; ++n) {
            const int64_t j = p * kNR + n;
            if (j >= nc) {
                for (int64_t k = 0; k < kc; ++k) {
                    dst[static_cast<size_t>(k * kNR + n)] = 0.0f;
                }
                continue;
            }
            const float * src = x + static_cast<size_t>(j0 + j) * ldx + static_cast<size_t>(k0);
            for (int64_t k = 0; k < kc; ++k) {
                dst[static_cast<size_t>(k * kNR + n)] = src[k];
            }
        }
    }
}

// Work split: the output is cut into (row block x column block) jobs dealt
// to the threads round robin. A row block's weight slice for one k block
// (kMB x kKC F16 = 128 KiB) and the packed activations of one column block
// (kKC x <= kNC F32 = 192 KiB) both stay in L2 while the micro-tiles sweep
// them. The k order inside every output element is fixed (segments, then k
// blocks, then k), so the result does not depend on the thread count.
constexpr int64_t kMB = 256;

// Claim the next job of thread `owner`'s range [owner*per, ...). A slot
// packs (generation << 32 | jobs taken); a stale generation reads as zero.
int64_t claim(Desc & d, int owner, uint32_t gen, int64_t limit) {
    std::atomic<uint64_t> & a   = d.slot[owner];
    uint64_t                cur = a.load(std::memory_order_relaxed);
    for (;;) {
        const uint32_t g     = static_cast<uint32_t>(cur >> 32);
        const int64_t  taken = g == gen ? static_cast<int64_t>(cur & 0xffffffffu) : 0;
        if (taken >= limit) {
            return -1;
        }
        const uint64_t next = (static_cast<uint64_t>(gen) << 32) | static_cast<uint64_t>(taken + 1);
        if (a.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
            return taken;
        }
    }
}

void gemm_fn(ggml_tensor * dst, int ith, int nth, void * ud) {
    Desc & d = *static_cast<Desc *>(ud);
    if (nth > Desc::kMaxThreads) {
        nth = Desc::kMaxThreads;
        if (ith >= nth) {
            return;
        }
    }
    // The generation advances once per compute: the last thread to arrive
    // in compute n bumps it for compute n + 1. Threads of one compute all
    // read the same value because nobody arrives before every thread has
    // read it (the read happens before this thread's own arrival).
    const uint32_t gen = d.generation.load(std::memory_order_acquire) + 1;
    const int64_t  M   = d.M;
    const int64_t  N   = d.N;

    const int64_t mb   = std::min(kMB, M);
    const int64_t n_rb = (M + mb - 1) / mb;
    int64_t       n_cb = (N + kNC - 1) / kNC;
    if (n_rb * n_cb < 4 * nth) {
        // Too few jobs to keep every thread busy: narrow the column blocks.
        n_cb = std::min<int64_t>((N + kNR - 1) / kNR, (4 * nth + n_rb - 1) / n_rb);
    }
    int64_t cb_w = (N + n_cb - 1) / n_cb;
    cb_w         = (cb_w + kNR - 1) / kNR * kNR;
    n_cb         = (N + cb_w - 1) / cb_w;

    const float * bias  = d.b_src >= 0 ? static_cast<const float *>(dst->src[d.b_src]->data) : nullptr;
    const float * scale = d.scale_src >= 0 ? static_cast<const float *>(dst->src[d.scale_src]->data) : nullptr;
    const float * shift = d.shift_src >= 0 ? static_cast<const float *>(dst->src[d.shift_src]->data) : nullptr;
    const size_t  ldy   = dst->nb[1] / sizeof(float);
    float *       y0    = static_cast<float *>(dst->data);

    int64_t n_blocks = 0;  // k blocks over all segments
    for (int s = 0; s < d.n_segs; ++s) {
        n_blocks += (dst->src[d.segs[s].w_src]->ne[0] + kKC - 1) / kKC;
    }

    thread_local std::vector<float> bp;
    bp.resize(static_cast<size_t>(std::max(cb_w, int64_t{ kNR }) * kKC));

    const int64_t n_jobs = n_rb * n_cb;
    const int64_t per    = (n_jobs + nth - 1) / nth;

    // Own range first, then steal from the others in ring order.
    auto next_job = [&]() -> int64_t {
        for (int r = 0; r < nth; ++r) {
            const int     owner = (ith + r) % nth;
            const int64_t lo    = owner * per;
            const int64_t lim   = std::min(per, n_jobs - lo);
            if (lim <= 0) {
                continue;
            }
            const int64_t k = claim(d, owner, gen, lim);
            if (k >= 0) {
                return lo + k;
            }
        }
        return -1;
    };

    for (int64_t job = next_job(); job >= 0; job = next_job()) {
        const int64_t rb       = job % n_rb;
        const int64_t cb       = job / n_rb;
        const int64_t i0       = rb * mb;
        const int64_t i1       = std::min(M, i0 + mb);
        const int64_t j0       = cb * cb_w;
        const int64_t ncb      = std::min(cb_w, N - j0);
        const int64_t n_panels = (ncb + kNR - 1) / kNR;

        int64_t blk = 0;
        for (int s = 0; s < d.n_segs; ++s) {
            const ggml_tensor * w   = dst->src[d.segs[s].w_src];
            const ggml_tensor * x   = dst->src[d.segs[s].x_src];
            const int64_t       K   = w->ne[0];
            const bool          f16 = w->type == GGML_TYPE_F16;
            const size_t        esz = ggml_type_size(w->type);
            const size_t        ldx = x->nb[1] / sizeof(float);
            const float *       xs  = static_cast<const float *>(x->data) + static_cast<size_t>(d.segs[s].x_row0) * ldx;
            const char * wm = static_cast<const char *>(w->data) + static_cast<size_t>(d.segs[s].mat) * M * K * esz;

            for (int64_t k0 = 0; k0 < K; k0 += kKC, ++blk) {
                const int64_t kc    = std::min(kKC, K - k0);
                const bool    first = blk == 0;
                const bool    last  = blk == n_blocks - 1;
                pack_x(xs, ldx, j0, ncb, k0, kc, bp.data());

                for (int64_t i = i0; i < i1; i += kPanel) {
                    const char *  a  = wm + (static_cast<size_t>(i) * K + static_cast<size_t>(k0) * kPanel) * esz;
                    const float * bb = bias ? bias + i : nullptr;
                    const float * ss = scale ? scale + i : nullptr;
                    const float * hh = shift ? shift + i : nullptr;
                    for (int64_t p = 0; p < n_panels; ++p) {
                        const int64_t jj = j0 + p * kNR;
                        const int     nv = static_cast<int>(std::min<int64_t>(kNR, j0 + ncb - jj));
                        float *       y  = y0 + static_cast<size_t>(jj) * ldy + static_cast<size_t>(i);
                        const float * pb = bp.data() + static_cast<size_t>(p * kc * kNR);
                        if (f16) {
                            tile<true>(a, pb, kc, y, ldy, nv, first, last, d.ep, bb, ss, hh);
                        } else {
                            tile<false>(a, pb, kc, y, ldy, nv, first, last, d.ep, bb, ss, hh);
                        }
                    }
                }
            }
        }
    }
    if (d.arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == nth) {
        d.arrived.store(0, std::memory_order_relaxed);
        d.generation.store(gen, std::memory_order_release);
    }
}

bool cpu_has_kernel() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

#endif  // ECAPA_GEMM_X86

}  // namespace

bool available() {
#if ECAPA_GEMM_X86
    static const bool ok = cpu_has_kernel();
    return ok;
#else
    return false;
#endif
}

bool pack_weight(ggml_tensor * t) {
    if (t == nullptr || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || !ggml_is_contiguous(t)) {
        return false;
    }
    const int64_t K = t->ne[0];
    const int64_t M = t->ne[1];
    const int64_t Q = t->ne[2] * t->ne[3];
    if (M % kPanel != 0 || K <= 0) {
        return false;
    }
    const size_t         esz = ggml_type_size(t->type);
    std::vector<uint8_t> src(ggml_nbytes(t));
    std::vector<uint8_t> dst(src.size());
    ggml_backend_tensor_get(t, src.data(), 0, src.size());
    for (int64_t q = 0; q < Q; ++q) {
        const uint8_t * sm = src.data() + static_cast<size_t>(q * M * K) * esz;
        uint8_t *       dm = dst.data() + static_cast<size_t>(q * M * K) * esz;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                const size_t from = static_cast<size_t>(i * K + k);
                const size_t to   = static_cast<size_t>((i / kPanel) * K * kPanel + k * kPanel + (i % kPanel));
                std::memcpy(dm + to * esz, sm + from * esz, esz);
            }
        }
    }
    ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
    return true;
}

ggml_tensor * mul_mat(ggml_context *  ctx,
                      Arena &         arena,
                      const Segment * segs,
                      int             n_segs,
                      int64_t         N,
                      Epilogue        ep,
                      ggml_tensor *   b,
                      ggml_tensor *   scale,
                      ggml_tensor *   shift) {
#if ECAPA_GEMM_X86
    if (n_segs < 1 || n_segs > kMaxSegs) {
        return nullptr;
    }
    Desc & d = arena.emplace_back();
    d.n_segs = n_segs;
    d.M      = segs[0].w->ne[1];
    d.N      = N;
    d.ep     = ep;

    ggml_tensor * args[GGML_MAX_SRC] = {};
    int           n                  = 0;
    auto          add                = [&](ggml_tensor * t) {
        for (int i = 0; i < n; ++i) {
            if (args[i] == t) {
                return i;
            }
        }
        args[n] = t;
        return n++;
    };
    for (int s = 0; s < n_segs; ++s) {
        d.segs[s].w_src  = add(segs[s].w);
        d.segs[s].mat    = segs[s].mat;
        d.segs[s].x_src  = add(segs[s].x);
        d.segs[s].x_row0 = segs[s].x_row0;
    }
    if (b != nullptr) {
        d.b_src = add(b);
    }
    if (scale != nullptr) {
        d.scale_src = add(scale);
    }
    if (shift != nullptr) {
        d.shift_src = add(shift);
    }

    return ggml_custom_4d(ctx, GGML_TYPE_F32, d.M, N, 1, 1, args, n, gemm_fn, GGML_N_TASKS_MAX, &d);
#else
    (void) ctx;
    (void) arena;
    (void) segs;
    (void) n_segs;
    (void) N;
    (void) ep;
    (void) b;
    (void) scale;
    (void) shift;
    return nullptr;
#endif
}

}  // namespace transcribe::ecapa_tdnn::gemm
