// arch/whistle/cpu_avx2.h - AVX2 + FMA forms of the hottest Whistle CPU
// kernels, chosen at run time (the library itself targets baseline x86-64).
//
// These reassociate sums and use FMA, so they round differently from the
// generic ggml graphs they replace (rounding-order only; same math). Without
// AVX2 + FMA + F16C (or off GCC / Clang x86-64) those graphs run instead.

#pragma once

#include "weights.h"

#include <cstddef>
#include <cstdint>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#    define WHISTLE_HAVE_AVX2_KERNELS 1
#else
#    define WHISTLE_HAVE_AVX2_KERNELS 0
#endif

namespace transcribe::whistle::avx2 {

// True when the CPU has AVX2 + FMA + F16C and this build carries the kernels.
bool available();

// HadamardMLP for the 512 = 16 x 32 Kronecker shape (hmlp_supported).
bool hmlp_supported(const HmlpHost & h);
// One token: x [d] -> out [d]; scratch holds 4 * d floats.
void hmlp_row(const HmlpHost & h, const float * x, float * out, float * scratch);

// Attention for nq queries sharing K / Vt (beams of one utterance, GQA heads
// of one KV head), loading K and V once for all of them: scores[t] = scale *
// <q, K_t> + mask[t] over t < n, softmax, out[i] = sum_t p[t] * Vt_i[t], with
// K_t = k + t * ks, Vt_i = vt + i * vs (floats); D % 8 == 0, DV % 2 == 0. Query j: q + j * qs,
// mask + j * ms (mask may be null), out + j * os, probs + j * ps (probs may
// be null). scratch holds nq * (n + 8) floats. Vt rows must hold n rounded
// up to a multiple of 8 finite values (the PV loop reads that far).
void attn_multi(int           nq,
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
                float *       scratch,
                float *       out,
                size_t        os,
                float *       probs,
                size_t        ps);

// out[i] = o[i] * sigmoid(g[i]), i < n (ggml_v_expf for the exponential).
void sigmoid_mul(const float * o, const float * g, float * out, int64_t n);

// ---------------------------------------------------------------------------
// Q8_0 matmul on 8-row interleaved weights ("q8x8").
//
// Layout per (8-row group, 32-wide block): the 8 rows' fp16 scales (in
// kQ8x8Perm order), then 4 chunks of [rows 0-3 | rows 4-7] x 8 int8 values:
// the same 272 bytes as the 8 Q8_0 blocks it replaces, so a weight is packed
// in place. Activations are quantized to Q8_0 exactly as ggml's AVX2
// quantize_row_q8_0 does, so results differ from ggml's Q8_0 matmul only in
// the float accumulation order.
//
// Interim: ggml's CPU_REPACK has no x86 Q8_0 layout yet. Once the vendored
// ggml has one, drop this and let the repack buffer take these weights.
// ---------------------------------------------------------------------------
constexpr size_t kQ8x8Block = 272;

struct Q8Act {
    float  d;  // fp16-rounded scale
    int8_t qs[32];
};

// In place: data holds rows x k Q8_0 (rows % 8 == 0, k % 32 == 0).
void q8x8_pack(void * data, int64_t rows, int64_t k);
// One activation row x [k] -> k / 32 blocks.
void q8_quantize(const float * x, int64_t k, Q8Act * out);
// out[c * ldo + r] = <row r of W, column c> for rows in groups [g0, g1) and
// the ncols columns (column c's blocks at cols + c * nbk, nbk = k / 32).
void q8x8_gemm(const uint8_t * W,
               int64_t         k,
               const Q8Act *   cols,
               int             ncols,
               int64_t         g0,
               int64_t         g1,
               float *         out,
               size_t          ldo);

// Encoder stem front, channel-first: conv0 (1 -> C, 3x3, stride 2, pad 1,
// sum order as encoder.cpp stem_conv0_cpu), SiLU (as ggml's AVX2 SiLU), then
// the depthwise 3x3 stride-2 pad-1 conv (as ggml's CWHN depthwise kernel), so
// the result is bit-identical to running those three ops; the conv0 output
// is only ever held three rows at a time. mel [W0, H0] with row stride
// mel_stride floats; w0 / wdw ne [C, 9] (row j = kt * 3 + kf); out [C, W2, H2].
// Computes output rows [oy0, oy1). C % 8 == 0. scratch: 3 * W1 * C floats.
void stem_front(const float * mel,
                int64_t       W0,
                int64_t       H0,
                size_t        mel_stride,
                const float * w0,
                const float * wdw,
                int64_t       C,
                float *       out,
                int64_t       oy0,
                int64_t       oy1,
                float *       scratch);

// log(sum_i exp(x_i)) of one row: max-shifted, ggml_v_expf terms summed in
// double.
double log_sum_exp(const float * x, int64_t n);

}  // namespace transcribe::whistle::avx2
