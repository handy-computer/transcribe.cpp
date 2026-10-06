// arch/ecapa_tdnn/cpu_gemm.h - x86 AVX2 GEMM for the ECAPA-TDNN CPU graph.
//
// INTERNAL to src/arch/ecapa_tdnn/. Optional: available() is false on
// non-x86 builds and on x86 CPUs without AVX2 + FMA + F16C, and the graph
// then uses ggml_mul_mat.
//
// Why: on AVX2, ggml's F16 x F32 matmul rounds the activations to F16 and
// widens both operands inside its innermost loop, which runs ~1.7x slower
// than its F32 path. This kernel keeps the weights F16 in memory (repacked
// once at load into 16-row panels), widens them in registers, keeps the
// activations F32, and accumulates in F32. It also fuses what the graph
// would otherwise do in separate passes over the output:
//   - several (weight, activation) segments summed into one output, which
//     covers both the MFA stage (three blocks) and a dilated k-tap
//     convolution (one segment per tap, the activation shifted by k*d rows);
//   - the TDNN epilogue max(acc + b, 0) * scale + shift, optional tanh.
//
// Every output element is one F32 FMA chain over k in segment order, so the
// result does not depend on the thread count.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>

struct ggml_context;
struct ggml_tensor;

namespace transcribe::ecapa_tdnn::gemm {

// True when this build and this CPU can run the kernel.
bool available();

// Rows per weight panel. A weight with M output rows is usable when
// M % kPanel == 0.
constexpr int kPanel = 16;

// Most segments one mul_mat sums: the MFA stage's three blocks, or a k-tap
// convolution with k <= 3. A model with wider Res2Net kernels keeps those
// weights on ggml_mul_mat (model.cpp only packs them when k <= kMaxSegs).
constexpr int kMaxSegs = 3;

// Repack a weight in place into panel-major order. `t` is a ggml weight
// ne = [K, M, n_mats] (rows of K contiguous, F32 or F16), on a host buffer.
// Matrix q's row i, column k moves to
//   q*M*K + (i / 16)*K*16 + k*16 + (i % 16)          (elements)
// After this only the GEMM may read `t`. Returns false (and leaves `t`
// untouched) if the type or shape is unsupported.
bool pack_weight(ggml_tensor * t);

enum class Epilogue : int32_t {
    None    = 0,  // y = acc
    ReluBn  = 1,  // y = max(acc + b, 0) * scale + shift (b optional)
    ReluBnTanh = 2,
};

// One operand pair: rows [0, M) of packed weight matrix `mat` of `w` times
// the activation `x` ne = [K, *] starting at row `x_row0`.
struct Segment {
    ggml_tensor * w      = nullptr;
    int           mat    = 0;
    ggml_tensor * x      = nullptr;
    int64_t       x_row0 = 0;
};

// Per-node descriptor, read by the kernel at compute time. Owned by the
// caller's Arena, which must outlive the graph's computation (a deque, so
// growing it never moves an existing descriptor).
struct Desc {
    struct Seg {
        int     w_src  = -1;  // dst->src index of the packed weight
        int     mat    = 0;
        int     x_src  = -1;  // dst->src index of the activation
        int64_t x_row0 = 0;
    };

    Seg      segs[kMaxSegs];
    int      n_segs    = 0;
    int64_t  M         = 0;
    int64_t  N         = 0;
    Epilogue ep        = Epilogue::None;
    int      b_src     = -1;
    int      scale_src = -1;
    int      shift_src = -1;

    // Work stealing between the threads of one compute (cpu_gemm.cpp).
    // Slot t holds thread t's progress through its own job range, tagged
    // with the compute generation so a reused graph starts clean without a
    // reset pass (custom ops have no barrier to reset behind).
    static constexpr int kMaxThreads = 256;
    std::atomic<uint64_t> slot[kMaxThreads] = {};
    std::atomic<uint32_t> generation{ 0 };
    std::atomic<int32_t>  arrived{ 0 };
};

using Arena = std::deque<Desc>;

// y[M, N] = epilogue(sum over segments of W_s @ x_s[x_row0 : x_row0 + N]).
// Returns a new [M, N] tensor. b / scale / shift are F32 vectors ne = [M]
// (b may be null; all null for Epilogue::None). 1..kMaxSegs segments.
ggml_tensor * mul_mat(ggml_context *  ctx,
                      Arena &         arena,
                      const Segment * segs,
                      int             n_segs,
                      int64_t         N,
                      Epilogue        ep,
                      ggml_tensor *   b,
                      ggml_tensor *   scale,
                      ggml_tensor *   shift);

}  // namespace transcribe::ecapa_tdnn::gemm
