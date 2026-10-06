// arch/ecapa_tdnn/cpu_ops.h - fused CPU kernels for the ECAPA-TDNN graph.
//
// INTERNAL to src/arch/ecapa_tdnn/. Used only when the graph runs on the
// ggml CPU backend (graph.cpp's `cpu_ops` build); every other backend keeps
// the stock-op graph. Each builder returns a GGML_OP_CUSTOM node that runs on
// all of the scheduler's CPU threads.
//
// Why these exist: with stock ggml ops the activations are channel-innermost
// ne = [C, T], so every reduction over time needs a transposed copy first,
// and ggml's CPU MEAN / SUM_ROWS run on one thread. The epilogue chain
// (bias -> ReLU -> BN scale -> BN shift) is four full passes over the
// tensor, each with a thread barrier. The kernels below do each of those in
// one multithreaded pass over the [C, T] layout, with the same float
// arithmetic per element and the same sequential double accumulation over
// time that ggml's reductions use (time_mean / time_stats are bit-identical
// to transpose + ggml_mean).
//
// Every input must be F32 and contiguous; vectors are ne = [C].

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace transcribe::ecapa_tdnn::cpu {

// y = max(z + b, 0) * scale + shift, then tanh(y) if `tanh_after`.
// z is ne = [C, T]; b may be null. In place over z when `inplace`.
ggml_tensor * epilogue(ggml_context * ctx,
                       ggml_tensor *  z,
                       ggml_tensor *  b,
                       ggml_tensor *  scale,
                       ggml_tensor *  shift,
                       bool           tanh_after,
                       bool           inplace);

// y = max(z0 + z1 + z2 + b, 0) * scale + shift (the MFA stage's three
// per-block products). In place over z0 when `inplace`.
ggml_tensor * epilogue_sum3(ggml_context * ctx,
                            ggml_tensor *  z0,
                            ggml_tensor *  z1,
                            ggml_tensor *  z2,
                            ggml_tensor *  b,
                            ggml_tensor *  scale,
                            ggml_tensor *  shift,
                            bool           inplace);

// Mean over time of x ne = [C, T], returned as ne = [C].
ggml_tensor * time_mean(ggml_context * ctx, ggml_tensor * x);

// y = x * s + res: x, res ne = [C, T], s ne = [C]. New tensor.
ggml_tensor * scale_add(ggml_context * ctx, ggml_tensor * x, ggml_tensor * s, ggml_tensor * res);

// Uniform statistics over time of x ne = [C, T]: ne = [2C] holding
// [mean, sqrt(max(var, *eps))], biased variance. `eps` must outlive compute.
ggml_tensor * time_stats(ggml_context * ctx, ggml_tensor * x, const float * eps);

// Attentive statistics: per channel c, w = softmax_t(logits[c, t] + bias[c]),
// mu = sum_t w x, sigma = sqrt(max(sum_t w (x - mu)^2, *eps)). x and logits
// ne = [C, T], bias ne = [C]; returns ne = [2C] = [mu, sigma].
ggml_tensor * attn_stats(ggml_context * ctx, ggml_tensor * x, ggml_tensor * logits, ggml_tensor * bias,
                         const float * eps);

// One Res2Net step. Chunk i of the block (i >= 1) is
//   y_i = BN(ReLU(sum_k Z[k*w + c, t + k*d] + b))
// where Z = W_stack @ pad_i is the stacked-tap product ne = [K*w, T + 2p]
// of the reflect-padded chunk input (p = d*(K-1)/2). The step computes y_i
// and, when there is a next chunk, that chunk's reflect-padded input
//   pad_{i+1} = reflect(h[next] + y_i)            ne = [w, T + 2p]
// in one pass. The first step of a block has no Z: it only pads h[1].
//
// The returned tensor packs both, rows [0, T + 2p) = pad (when next >= 0)
// followed by T rows of y (when z != nullptr); use res2_pad / res2_y.
struct Res2Step {
    ggml_tensor * out = nullptr;
    ggml_tensor * pad = nullptr;  // view, ne = [w, T + 2p], contiguous
    ggml_tensor * y   = nullptr;  // view, ne = [w, T]
};

Res2Step res2_step(ggml_context * ctx,
                   ggml_tensor *  h,      // block input [C, T]
                   int            next,   // chunk index of h to pad next, or -1
                   int            width,  // chunk width w
                   ggml_tensor *  z,      // [K*w, T + 2p] or nullptr
                   ggml_tensor *  b,
                   ggml_tensor *  scale,
                   ggml_tensor *  shift,
                   int            kernel,
                   int            dilation);

// pad_{next} = reflect(h[next] + y) for a GEMM-produced y ne = [w, T]:
// returns a Res2Step whose `pad` is set (out == pad).
Res2Step res2_pad(ggml_context * ctx, ggml_tensor * h, int next, int width, ggml_tensor * y, int kernel, int dilation);

// Res2Net output [C, T]: chunk 0 from h, chunk i from ys[i - 1] (each [w, T]).
ggml_tensor * res2_gather(ggml_context * ctx, ggml_tensor * h, ggml_tensor * const * ys, int n_ys);

}  // namespace transcribe::ecapa_tdnn::cpu
