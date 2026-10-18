// arch/ecapa_tdnn/weights.h - ECAPA-TDNN hyperparameters, tensor catalogue,
// and the per-instance weight slots.
//
// INTERNAL to src/arch/ecapa_tdnn/.
//
// The names and shapes below are the loader contract with
// scripts/convert-ecapa_tdnn.py. ggml `ne` is fast-to-slow, so a PyTorch
// Linear [OC, IC] lands as ne = [IC, OC] and a PyTorch Conv1d [OC, IC, K] is
// stored TAP-MAJOR as ne = [IC, OC, K] (numpy [K, OC, IC]) so every tap is a
// contiguous [IC, OC] matrix. Names follow the tools/transcribe-quantize
// rules: `.bias` and `.bn.` stay F32, `.conv.weight` (k>1 kernels) stays
// F32 / F16, every other `.weight` is a quantizable matmul operand.
//
//   frontend.mel_filterbank            F32      ne=[n_freq, n_mels]   mel-major
//
//   blk.0.conv.weight                  F32/F16  ne=[n_mels, C, K0]
//   blk.0.conv.bias                    F32      [C]
//   blk.0.bn.scale / .shift            F32      [C]
//
//   blk.{1,2,3}.tdnn1.weight           quant    ne=[C, C]
//   blk.{1,2,3}.tdnn1.bias             F32      [C]
//   blk.{1,2,3}.tdnn1.bn.scale / .shift         [C]
//   blk.{1,2,3}.res2.{0..S-2}.conv.weight  F32/F16  ne=[C/S, C/S, Ki]
//   blk.{1,2,3}.res2.{0..S-2}.conv.bias    F32      [C/S]
//   blk.{1,2,3}.res2.{0..S-2}.bn.scale / .shift     [C/S]
//   blk.{1,2,3}.tdnn2.weight           quant    ne=[C, C]
//   blk.{1,2,3}.tdnn2.bias             F32      [C]
//   blk.{1,2,3}.tdnn2.bn.scale / .shift         [C]
//   blk.{1,2,3}.se.c1.weight           quant    ne=[C, se]     .bias [se]
//   blk.{1,2,3}.se.c2.weight           quant    ne=[se, C]     .bias [C]
//
//   mfa.w{1,2,3}.weight                quant    ne=[C, Cm]  (Cm = 3C)
//   mfa.bias                           F32      [Cm]
//   mfa.bn.scale / .shift              F32      [Cm]
//
//   asp.tdnn.{x,mean,std}.weight       quant    ne=[Cm, att]
//   asp.tdnn.bias                      F32      [att]
//   asp.tdnn.bn.scale / .shift         F32      [att]
//   asp.attn.weight                    quant    ne=[att, Cm]   .bias [Cm]
//
//   fc.weight                          quant    ne=[2*Cm, emb]  .bias [emb]   (asp_bn folded in)
//   cls.l1.weight                      quant    ne=[emb, hid]   .bias [hid]   (cls.bn0 folded in)
//   cls.out.weight                     quant    ne=[hid, n_lab] .bias [n_lab] (cls.bn1 folded in)
//
// "quant" is TRANSCRIBE_QUANT_LINEAR_TYPES; k>1 kernels are
// TRANSCRIBE_QUANT_CONV_TYPES. Every dimension above is derived from HParams
// so the tiny test fixture and the real checkpoint share one path.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::ecapa_tdnn {

// Number of TDNN stages described by stt.ecapa_tdnn.channels: four embedding
// blocks (one plain TDNNBlock + three SERes2Net) plus the MFA stage.
constexpr int kNumStages = 5;

// SERes2Net blocks, i.e. stages 1..3.
constexpr int kNumSeBlocks = 3;

// Res2Net scale this implementation supports (fixes the weight-slot count).
constexpr int kRes2NetScale = 8;

// Sub-convolutions inside one Res2Net block: the first chunk is passed
// through unchanged, so there are scale - 1 of them.
constexpr int kRes2NetSubs = kRes2NetScale - 1;

// Every stt.* KV the family reads, plus the label count.
struct HParams {
    int32_t sample_rate = 0;

    // Front end (stt.frontend.*).
    int32_t     mel_n_fft  = 0;
    int32_t     mel_hop    = 0;
    int32_t     mel_win    = 0;
    int32_t     mel_n_mels = 0;
    std::string mel_window;    // "hamming_periodic"
    std::string mel_pad_mode;  // "constant"
    float       mel_log_floor = 0.0f;
    float       mel_top_db    = 0.0f;
    std::string mel_normalize;  // "sentence_mean"

    // Embedding network (stt.ecapa_tdnn.*).
    std::vector<int32_t> channels;      // kNumStages entries
    std::vector<int32_t> kernel_sizes;  // kNumStages entries
    std::vector<int32_t> dilations;     // kNumStages entries
    int32_t              res2net_scale      = 0;
    int32_t              se_channels        = 0;
    int32_t              attention_channels = 0;
    float                asp_eps            = 0.0f;
    int32_t              embedding_dim      = 0;

    // Classifier (stt.ecapa_tdnn.classifier_*).
    int32_t classifier_hidden = 0;
    float   leaky_slope       = 0.0f;

    // Set by load() from the label table.
    int32_t n_labels = 0;

    // ---- derived ------------------------------------------------------
    int32_t c_block() const { return channels.empty() ? 0 : channels[0]; }

    int32_t c_mfa() const { return channels.size() < kNumStages ? 0 : channels[kNumStages - 1]; }

    int32_t c_chunk() const { return res2net_scale > 0 ? c_block() / res2net_scale : 0; }

    int32_t n_freq() const { return mel_n_fft / 2 + 1; }

    // Inner width of the stage-0 im2col matmul: K0 * n_mels rounded up to a
    // multiple of 32 (every CPU tinyBLAS tile width divides it), zero padded.
    int32_t blk0_cols() const {
        const int32_t n = kernel_sizes.empty() ? 0 : kernel_sizes[0] * mel_n_mels;
        return (n + 31) / 32 * 32;
    }

    // Reflect padding on each side of stage `i`: d * (k - 1) / 2, the
    // "same"-padding width SpeechBrain's Conv1d uses.
    int32_t pad(int i) const {
        return dilations[static_cast<size_t>(i)] * (kernel_sizes[static_cast<size_t>(i)] - 1) / 2;
    }
};

// One TDNNBlock's BatchNorm, stored as an affine map (the converter
// cannot fold it into the conv: TDNNBlock is conv -> ReLU -> BN).
struct BnAffine {
    ggml_tensor * scale = nullptr;
    ggml_tensor * shift = nullptr;
};

// A k=1 TDNNBlock: 1x1 conv -> ReLU -> BN.
struct TdnnLayer {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
    BnAffine      bn;
};

// One of the scale-1 dilated 3-tap convolutions inside a Res2Net block.
struct Res2Sub {
    ggml_tensor * w = nullptr;  // ne=[c_chunk, c_chunk, K]
    ggml_tensor * b = nullptr;  // [c_chunk]
    BnAffine      bn;
};

// Squeeze-and-excitation: mean over T -> 1x1 -> ReLU -> 1x1 -> sigmoid -> scale.
struct SeBlock {
    ggml_tensor * c1_w = nullptr;  // ne=[C, se]
    ggml_tensor * c1_b = nullptr;  // [se]
    ggml_tensor * c2_w = nullptr;  // ne=[se, C]
    ggml_tensor * c2_b = nullptr;  // [C]
};

struct SeRes2NetBlock {
    TdnnLayer tdnn1;
    Res2Sub   res2[kRes2NetSubs];
    TdnnLayer tdnn2;
    SeBlock   se;
};

struct Weights {
    // Front end. Read back to host memory at load time and handed to the
    // MelFrontend; never touched by the graph.
    ggml_tensor * mel_filters = nullptr;  // ne=[n_freq, n_mels]

    // Stage 0: plain TDNNBlock, n_mels -> C, k = kernel_sizes[0].
    ggml_tensor * blk0_w        = nullptr;  // ne=[n_mels, C, K0]
    // Derived at load (model.cpp, not in the GGUF): blk0_w regrouped as one
    // [K0 * n_mels, C] matrix, zero-padded on the inner axis to
    // HParams::blk0_cols(), so stage 0 is a single aligned matmul against
    // the host-built im2col of the mel. Same type as blk0_w.
    ggml_tensor * blk0_w_im2col = nullptr;  // ne=[blk0_cols, C]
    ggml_tensor * blk0_b        = nullptr;  // [C]
    BnAffine      blk0_bn;

    // Stages 1..3.
    SeRes2NetBlock blocks[kNumSeBlocks];

    // Multi-layer feature aggregation. The 3C -> 3C 1x1 conv is split into
    // three C -> 3C blocks applied to the three SERes2Net outputs and summed,
    // which avoids materialising the concat.
    ggml_tensor * mfa_w[kNumSeBlocks] = { nullptr, nullptr, nullptr };
    ggml_tensor * mfa_b               = nullptr;  // [Cm]
    BnAffine      mfa_bn;

    // Attentive statistics pooling. asp.tdnn's 3*Cm -> att weight is split
    // into the x / mean / std blocks.
    ggml_tensor * asp_wx = nullptr;  // ne=[Cm, att]
    ggml_tensor * asp_wm = nullptr;
    ggml_tensor * asp_ws = nullptr;
    ggml_tensor * asp_b  = nullptr;      // [att]
    BnAffine      asp_bn;
    ggml_tensor * asp_attn_w = nullptr;  // ne=[att, Cm]
    ggml_tensor * asp_attn_b = nullptr;  // [Cm]

    // Embedding head (asp_bn folded in).
    ggml_tensor * fc_w = nullptr;  // ne=[2*Cm, emb]
    ggml_tensor * fc_b = nullptr;  // [emb]

    // Classifier (cls.bn0 / cls.bn1 folded forward; the LeakyReLUs still run).
    ggml_tensor * cls_l1_w  = nullptr;  // ne=[emb, hid]
    ggml_tensor * cls_l1_b  = nullptr;  // [hid]
    ggml_tensor * cls_out_w = nullptr;  // ne=[hid, n_labels]
    ggml_tensor * cls_out_b = nullptr;  // [n_labels]
};

// Read every stt.* KV the family needs and reject shapes the graph cannot
// build. Does not set n_labels.
transcribe_status read_hparams(const gguf_context * gguf, HParams & hp);

// Bind every tensor in the catalogue above to a borrowed pointer in
// `ctx_meta`, validating type and shape against `hp`. On failure the partly
// built `w` is indeterminate and the caller must discard the model.
transcribe_status build_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w);

}  // namespace transcribe::ecapa_tdnn
