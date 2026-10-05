// arch/ecapa_tdnn/graph.h - ECAPA-TDNN forward graph builder.
//
// INTERNAL to src/arch/ecapa_tdnn/. The LANGID ops table's run hook builds
// and computes this graph once per call.
//
// Every activation is ggml `ne = [C, T]` (channel-innermost); see the
// helper notes at the top of graph.cpp. Topology is independent of T, so the
// scheduler's allocator settles after the first call.

#pragma once

#include "ggml.h"
#include "weights.h"

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
    // Log-mel input, ne = [n_mels, T] f32. The caller uploads the front end's
    // frame-major buffer verbatim (frame t's n_mels values contiguous).
    ggml_tensor * mel_in = nullptr;

    // Reflect-padding gather indices, one per SERes2Net block, I32
    // ne = [T + 2*pad(i+1)]. Stage 0 reuses idx[0] (read_hparams enforces
    // pad(0) == pad(1)).
    ggml_tensor * idx[kNumSeBlocks] = { nullptr, nullptr, nullptr };

    // Output.
    ggml_tensor * logits = nullptr;  // ne = [n_labels]

    Dumps dumps{};

    ggml_cgraph * graph = nullptr;
};

// Build the full forward graph for T frames into `ctx` (a fresh no_alloc
// context). Returns a build with `graph == nullptr` on invalid input.
GraphBuild build_graph(ggml_context * ctx, const Model & model, int T);

}  // namespace transcribe::ecapa_tdnn
