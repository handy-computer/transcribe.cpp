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
