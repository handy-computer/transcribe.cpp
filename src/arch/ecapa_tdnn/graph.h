// arch/ecapa_tdnn/graph.h - ECAPA-TDNN forward graph builder.
//
// INTERNAL to src/arch/ecapa_tdnn/. The LANGID ops table's run hook builds
// and computes this graph once per call.
//
// Every activation is ggml `ne = [C, T]` (channel-innermost); see the
// helper notes at the top of graph.cpp. Topology is independent of T, so the
// scheduler's allocator settles after the first call.

#pragma once

#include "cpu_gemm.h"
#include "ggml.h"
#include "weights.h"

#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::ecapa_tdnn {

struct Model;

// Named intermediates for the numerical-parity harness. Each is a borrowed
// pointer into the compute context and is already marked as a graph output
// when TRANSCRIBE_DUMP_DIR is set (otherwise the scheduler may reuse its
// buffer). The dump names are the contract with the reference dumper
// (scripts/dump_reference_ecapa_tdnn_speechbrain.py); each tensor's ggml `ne`
// is reversed on disk, so `ne = [C, T]` lands as the reference's `[T, C]`.
struct Dumps {
    ggml_tensor * blk0_out              = nullptr;  // enc.blk.0.out          [C, T]
    ggml_tensor * blk1_tdnn1_out        = nullptr;  // enc.blk.1.tdnn1.out    [C, T]
    ggml_tensor * blk1_res2_out         = nullptr;  // enc.blk.1.res2.out     [C, T]
    ggml_tensor * blk1_se_out           = nullptr;  // enc.blk.1.se.out       [C, T]
    ggml_tensor * blk_out[kNumSeBlocks] = { nullptr, nullptr, nullptr };
    // enc.blk.{1,2,3}.out    [C, T]
    ggml_tensor * mfa_out               = nullptr;  // enc.mfa.out            [Cm, T]
    // asp.attn's output BEFORE the transpose: ggml ne = [Cm, T] lands on disk
    // as [T, Cm], which is the orientation the reference dumps. Marking the
    // transposed copy instead would be a silent SHAPE mismatch.
    ggml_tensor * asp_attn_logits       = nullptr;  // enc.asp.attn_logits    [Cm, T]
    ggml_tensor * asp_out               = nullptr;  // enc.asp.out            [2*Cm]
    ggml_tensor * emb                   = nullptr;  // enc.emb                [emb]
    ggml_tensor * cls_hidden            = nullptr;  // cls.hidden             [hid]
    ggml_tensor * cls_logits            = nullptr;  // cls.logits_raw         [n_labels]
};

struct GraphBuild {
    // Stage-0 input: the host-built im2col of the reflect-padded log-mel,
    // ne = [blk0_cols, T] f32. Row t holds mel_pad[t + k*d0] for k = 0..K0-1
    // back to back, then zeros up to blk0_cols (build_blk0_im2col).
    ggml_tensor * blk0_in = nullptr;

    // Reflect-padding gather indices, one per SERes2Net block, I32
    // ne = [T + 2*pad(i+1)]. Only the stock-op graph uses them; the CPU
    // graph pads inside its Res2Net kernel and leaves these null.
    ggml_tensor * idx[kNumSeBlocks] = { nullptr, nullptr, nullptr };

    // Res2Net chunk numbers 0..kRes2NetScale-1, I32. Only the stock-op graph
    // uses them (set_rows targets when writing a chunk's output in place).
    ggml_tensor * chunk_ids = nullptr;

    // Output.
    ggml_tensor * logits = nullptr;  // ne = [n_labels]

    Dumps dumps{};

    ggml_cgraph * graph = nullptr;
};

// Build the full forward graph for T frames into `ctx` (a fresh no_alloc
// context). `cpu_ops` selects the fused CPU kernels (cpu_ops.h); it must only
// be set when every node runs on the ggml CPU backend. Weights the model
// packed for the AVX2 GEMM (Model::gemm_conv / gemm_lin) are routed through
// it; `arena` holds those nodes' descriptors and must outlive the compute.
// Returns a build with `graph == nullptr` on invalid input.
GraphBuild build_graph(ggml_context * ctx, const Model & model, int T, bool cpu_ops, gemm::Arena * arena);

// Fill `out` ([T, hp.blk0_cols()] frame-major, i.e. ggml ne = [cols, T])
// with the stage-0 im2col of the frame-major log-mel `mel` [T, n_mels]:
// out[t, k*n_mels + m] = mel[reflect(t + k*d0 - p0), m], zeros past K0*n_mels.
void build_blk0_im2col(const HParams & hp, const float * mel, int T, std::vector<float> & out);

}  // namespace transcribe::ecapa_tdnn
