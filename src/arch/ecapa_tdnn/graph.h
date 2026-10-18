// arch/ecapa_tdnn/graph.h - ECAPA-TDNN forward graph builder.
//
// INTERNAL to src/arch/ecapa_tdnn/. Activations are ggml ne = [C, T].

#pragma once

#include "ggml.h"
#include "weights.h"

#include <cstddef>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::ecapa_tdnn {

struct Model;

// Graph node capacity (the built graph is ~571 nodes, independent of T).
constexpr size_t kGraphSize = 2048;

// Named intermediates for the parity dumps (marked as graph outputs when
// TRANSCRIBE_DUMP_DIR is set). Names match
// scripts/dump_reference_ecapa_tdnn_speechbrain.py.
struct Dumps {
    ggml_tensor * blk0_out              = nullptr;  // enc.blk.0.out          [C, T]
    ggml_tensor * blk1_tdnn1_out        = nullptr;  // enc.blk.1.tdnn1.out    [C, T]
    ggml_tensor * blk1_res2_out         = nullptr;  // enc.blk.1.res2.out     [C, T]
    ggml_tensor * blk1_se_out           = nullptr;  // enc.blk.1.se.out       [C, T]
    ggml_tensor * blk_out[kNumSeBlocks] = { nullptr, nullptr, nullptr };
    // enc.blk.{1,2,3}.out    [C, T]
    ggml_tensor * mfa_out               = nullptr;  // enc.mfa.out            [Cm, T]
    ggml_tensor * asp_attn_logits       = nullptr;  // enc.asp.attn_logits    [Cm, T]
    ggml_tensor * asp_out               = nullptr;  // enc.asp.out            [2*Cm]
    ggml_tensor * emb                   = nullptr;  // enc.emb                [emb]
    ggml_tensor * cls_hidden            = nullptr;  // cls.hidden             [hid]
    ggml_tensor * cls_logits            = nullptr;  // cls.logits_raw         [n_labels]
};

struct GraphBuild {
    // Stage-0 input: im2col of the log-mel, ne = [blk0_cols, T] (build_blk0_im2col).
    ggml_tensor * blk0_in = nullptr;

    // Reflect-padding gather indices, one per SERes2Net block, I32
    // ne = [T + 2*pad(i+1)].
    ggml_tensor * idx[kNumSeBlocks] = { nullptr, nullptr, nullptr };

    // 0..kRes2NetScale-1, I32: set_rows targets for the Res2Net chunks.
    ggml_tensor * chunk_ids = nullptr;

    // Output.
    ggml_tensor * logits = nullptr;  // ne = [n_labels]

    Dumps dumps{};

    ggml_cgraph * graph = nullptr;
};

// Build the forward graph for T frames into a fresh no_alloc `ctx`.
// Requires T > hp.pad(i) for every stage.
GraphBuild build_graph(ggml_context * ctx, const Model & model, int T);

// Fill `out` ([T, hp.blk0_cols()] frame-major, i.e. ggml ne = [cols, T])
// with the stage-0 im2col of the frame-major log-mel `mel` [T, n_mels]:
// out[t, k*n_mels + m] = mel[reflect(t + k*d0 - p0), m], zeros past K0*n_mels.
void build_blk0_im2col(const HParams & hp, const float * mel, int T, std::vector<float> & out);

}  // namespace transcribe::ecapa_tdnn
