// arch/silero_vad/decoder.cpp - the recurrent decoder on the host: one
// LSTMCell step and the sigmoid head per frame.
//
// torch.lstm_cell, gate order i, f, g, o:
//   gates = (W_ih x + b_ih + b_hh) + W_hh h      first term from the encoder graph
//   c'    = sigmoid(f) * c + sigmoid(i) * tanh(g)
//   h'    = sigmoid(o) * tanh(c')
//   p     = sigmoid(head . relu(h') + head_b)
//
// The step is a 128 x 512 matvec plus elementwise work, far too small for a
// graph launch per frame; the loop below runs it straight from host copies.

#include "ggml.h"
#include "silero_vad.h"
#include "transcribe-debug.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace transcribe::silero_vad {

namespace {

// Gates per register tile of the W_hh h product; 4H must be a multiple.
constexpr int kTile = 32;

// exp(x) for the gate nonlinearities: x = n ln2 + r with a two-constant
// Cody-Waite split, exp(r) as a degree-7 polynomial, 2^n from the exponent
// bits. Max relative error 8.3e-8 over the clamped range, so sigmoid and
// tanh below are within 9e-8 absolute of the exact value (libm's sigmoid
// is too). libm's expf / tanhf are not inlined and cost 3-4x as much in
// this loop, which runs 4 * hidden times per frame.
inline float fast_expf(float x) {
    x                  = std::fmin(std::fmax(x, -87.0f), 88.0f);
    const float n      = std::nearbyint(x * 1.44269504088896341f);
    float       r      = std::fma(n, -0.693359375f, x);
    r                  = std::fma(n, 2.12194440e-4f, r);
    float p            = 1.9875691500e-4f;
    p                  = std::fma(p, r, 1.3981999507e-3f);
    p                  = std::fma(p, r, 8.3334519073e-3f);
    p                  = std::fma(p, r, 4.1665795894e-2f);
    p                  = std::fma(p, r, 1.6666665459e-1f);
    p                  = std::fma(p, r, 5.0000001201e-1f);
    p                  = std::fma(p * r, r, r) + 1.0f;
    const int32_t bits = (static_cast<int32_t>(n) + 127) << 23;
    float         scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

inline float sigmoidf(float x) {
    return 1.0f / (1.0f + fast_expf(-x));
}

// Odd and cancellation-free in 1 + e: tanh|x| = (1 - e) / (1 + e), e = exp(-2|x|).
inline float tanhf_fast(float x) {
    const float e = fast_expf(-2.0f * std::fabs(x));
    return std::copysign((1.0f - e) / (1.0f + e), x);
}

}  // namespace

void decode_block(Session & s, const Model & m, const float * gates_in, int n, float * probs) {
    const int     H     = m.hparams.hidden;
    const size_t  G     = 4 * static_cast<size_t>(H);
    const float * w_hh  = m.host.w_hh_t.data();  // [H][4H]
    const float * hw    = m.host.head_w.data();
    float *       h     = s.h.data();
    float *       c     = s.c.data();
    float *       gates = s.gates.data();
    const bool    dump  = debug::enabled();
    GGML_ASSERT(G % kTile == 0);

    for (int f = 0; f < n; ++f) {
        // gates = gates_in + W_hh h. Tiles of kTile gates stay in registers
        // across the H-long reduction, so W_hh^T streams through once.
        const float * gin = gates_in + static_cast<size_t>(f) * G;
        for (size_t g0 = 0; g0 < G; g0 += kTile) {
            float acc[kTile];
            for (int u = 0; u < kTile; ++u) {
                acc[u] = gin[g0 + u];
            }
            for (int j = 0; j < H; ++j) {
                const float   hj  = h[j];
                const float * row = w_hh + static_cast<size_t>(j) * G + g0;
                for (int u = 0; u < kTile; ++u) {
                    acc[u] += row[u] * hj;
                }
            }
            for (int u = 0; u < kTile; ++u) {
                gates[g0 + u] = acc[u];
            }
        }
        float logit = 0.0f;
        for (int j = 0; j < H; ++j) {
            const float i_t = sigmoidf(gates[j]);
            const float f_t = sigmoidf(gates[H + j]);
            const float g_t = tanhf_fast(gates[2 * H + j]);
            const float o_t = sigmoidf(gates[3 * H + j]);
            c[j]            = f_t * c[j] + i_t * g_t;
            h[j]            = o_t * tanhf_fast(c[j]);
            logit += hw[j] * (h[j] > 0.0f ? h[j] : 0.0f);
        }
        probs[f] = sigmoidf(logit + m.host.head_b);

        if (dump) {
            s.trace.lstm_h.insert(s.trace.lstm_h.end(), h, h + H);
            s.trace.lstm_c.insert(s.trace.lstm_c.end(), c, c + H);
            s.trace.probs.push_back(probs[f]);
        }
    }
}

}  // namespace transcribe::silero_vad
