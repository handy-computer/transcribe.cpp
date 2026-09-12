// arch/granite5_ctc/weights.h - Granite Speech 5.0 TurboCTC tensor
// catalog and hparams.
//
// INTERNAL to src/arch/granite5_ctc/. Defines:
//
//   - Granite5CtcHParams: architecture KV that drives tensor shapes.
//     Read from stt.granite5_ctc.* and stt.frontend.* at load time,
//     before any tensors are allocated.
//
//   - Granite5CtcWeights: named ggml_tensor* slots. Borrowed pointers —
//     the tensors live in the model's ctx_meta / backend buffer.
//
// Architecture pattern: encoder-ctc. One weight prefix:
//
//   enc.*  Conformer encoder (16 blocks, hidden=1024, Shaw block-local
//          attention over context_size=128 blocks, GLU conv with
//          BatchNorm, in-block stride-2 subsampling on blocks 0 and 1,
//          self-conditioned CTC at block 7) plus the tied CTC head.
//
// There is no projector, no LM and no decoder: `enc.ctc_proj` is both
// the mid-layer self-conditioning head and the final CTC head
// (`_tied_weights_keys` in the reference ties ctc_head to encoder.out).
//
// Tensor naming mirrors the converter's output and, where the module is
// identical, granite 4.x's (`scripts/convert-granite.py`). Linear
// weights use PyTorch [out, in] order (ggml ne[0]=in, ne[1]=out).
// The depthwise Conv1d kernel uses PyTorch (out, in/groups, k) order
// (ggml ne[0]=k, ne[1]=1, ne[2]=inner_dim). The two pointwise convs are
// nn.Linear in this checkpoint, so they are 2-D [in, out] — NOT the 3-D
// [1, in, out] granite 4.x carries.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::granite5_ctc {

// Hyperparameters

struct Granite5CtcHParams {
    std::string variant;

    // Encoder (granite_speech5_encoder).
    int32_t enc_n_layers         = 0;  // 16
    int32_t enc_hidden           = 0;  // 1024
    int32_t enc_n_heads          = 0;  // 8
    int32_t enc_head_dim         = 0;  // 128
    int32_t enc_intermediate     = 0;  // 4096 (FFN inner)
    int32_t enc_input_dim        = 0;  // 320 ((80 mel + 80 delta) * 2 stack)
    int32_t enc_output_dim       = 0;  // 16384 (BPE CTC vocabulary)
    int32_t enc_conv_kernel_size = 0;  // 7
    int32_t enc_conv_expansion   = 0;  // 2 (conv inner = hidden * expansion)
    int32_t enc_context_size     = 0;  // 128 (block size for local attention)
    int32_t enc_max_pos_emb      = 0;  // 512 (Shaw table is 2*this + 1 rows)

    // Index of the block AFTER which the self-conditioned CTC injection
    // fires, 0-indexed: the reference triggers on
    // `layer_idx + 1 == num_hidden_layers // 2`, so with 16 layers the
    // injection lands on the output of block 7.
    int32_t enc_self_cond_layer = 0;  // 8 (== num_hidden_layers / 2)

    // Blocks that halve the time axis. [0, 1] on this variant: the
    // depthwise conv runs at stride 2 and the residual is mean-pooled
    // over frame pairs. Stored sorted; membership is checked per block.
    std::vector<int32_t> enc_subsample_layers;

    // CTC blank. Doubles as pad_token_id upstream.
    int32_t blank_id = 0;

    // Frontend (torchaudio MelSpectrogram + deltas + frame stacking).
    std::string fe_type;                    // "mel"
    int32_t     fe_num_mels        = 0;     // 80
    int32_t     fe_sample_rate     = 0;     // 16000
    int32_t     fe_n_fft           = 0;     // 512
    int32_t     fe_win_length      = 0;     // 400
    int32_t     fe_hop_length      = 0;     // 160
    std::string fe_window;                  // "hann_periodic"
    std::string fe_normalize;               // "per_utterance"
    std::string fe_pad_mode;                // "reflect"
    std::string fe_mel_norm;                // "htk"
    float       fe_dither          = 0.0f;  // 0.0
    bool        fe_deltas          = true;
    int32_t     fe_delta_win_length = 0;    // 3
    float       fe_logmel_floor_db  = 0.0f; // 8.0
    int32_t     fe_stack_factor     = 0;    // 2
};

transcribe_status read_granite5_ctc_hparams(const gguf_context * gguf, Granite5CtcHParams & hp);

// Weight slots

// Top-level encoder tensors. `ctc_proj` is tied: it produces the
// mid-layer self-conditioning logits AND the final CTC logits.
struct Granite5CtcEncTop {
    ggml_tensor * input_linear_w = nullptr;  // [input_dim,  hidden]
    ggml_tensor * input_linear_b = nullptr;  // [hidden]
    ggml_tensor * ctc_proj_w     = nullptr;  // [hidden,     output_dim]
    ggml_tensor * ctc_proj_b     = nullptr;  // [output_dim]
    ggml_tensor * ctc_bypass_w   = nullptr;  // [output_dim, hidden]
    ggml_tensor * ctc_bypass_b   = nullptr;  // [hidden]
};

// One Conformer block. Every linear carries a bias except q/k/v, where
// the reference hard-codes bias=False (config.attention_bias reaches the
// FFN linears, not the attention projections).
struct Granite5CtcEncBlock {
    // FF1 (first macaron half).
    ggml_tensor * norm_ff1_w  = nullptr;  // [hidden]
    ggml_tensor * norm_ff1_b  = nullptr;
    ggml_tensor * ff1_lin1_w  = nullptr;  // [hidden, intermediate]
    ggml_tensor * ff1_lin1_b  = nullptr;  // [intermediate]
    ggml_tensor * ff1_lin2_w  = nullptr;  // [intermediate, hidden]
    ggml_tensor * ff1_lin2_b  = nullptr;  // [hidden]

    // Block-local Shaw self-attention. attn.kv is the converter-side
    // concatenation of k_proj and v_proj (K rows first).
    ggml_tensor * norm_attn_w      = nullptr;  // [hidden]
    ggml_tensor * norm_attn_b      = nullptr;
    ggml_tensor * attn_q_w         = nullptr;  // [hidden, inner_dim]
    ggml_tensor * attn_kv_w        = nullptr;  // [hidden, 2*inner_dim]
    ggml_tensor * attn_out_w       = nullptr;  // [inner_dim, hidden]
    ggml_tensor * attn_out_b       = nullptr;  // [hidden]
    ggml_tensor * attn_rel_pos_emb = nullptr;  // [head_dim, 2*max_pos_emb+1]

    // Conv module.
    ggml_tensor * norm_conv_w         = nullptr;  // [hidden]
    ggml_tensor * norm_conv_b         = nullptr;
    ggml_tensor * conv_pointwise1_w   = nullptr;  // [hidden, 2*inner_dim]
    ggml_tensor * conv_pointwise1_b   = nullptr;  // [2*inner_dim]
    ggml_tensor * conv_depthwise_w    = nullptr;  // [k, 1, inner_dim]
    ggml_tensor * conv_bn_w           = nullptr;  // [inner_dim]
    ggml_tensor * conv_bn_b           = nullptr;  // [inner_dim]
    ggml_tensor * conv_bn_mean        = nullptr;  // [inner_dim]
    ggml_tensor * conv_bn_var         = nullptr;  // [inner_dim]
    ggml_tensor * conv_pointwise2_w   = nullptr;  // [inner_dim, hidden]
    ggml_tensor * conv_pointwise2_b   = nullptr;  // [hidden]
    // Precomputed at load (model.cpp fuse_batch_norm); the four raw BN
    // tensors above only feed that fusion.
    ggml_tensor * conv_bn_fused_scale = nullptr;  // [inner_dim]
    ggml_tensor * conv_bn_fused_bias  = nullptr;  // [inner_dim]

    // FF2 (second macaron half).
    ggml_tensor * norm_ff2_w = nullptr;
    ggml_tensor * norm_ff2_b = nullptr;
    ggml_tensor * ff2_lin1_w = nullptr;
    ggml_tensor * ff2_lin1_b = nullptr;
    ggml_tensor * ff2_lin2_w = nullptr;
    ggml_tensor * ff2_lin2_b = nullptr;

    // Per-block final LayerNorm.
    ggml_tensor * norm_out_w = nullptr;
    ggml_tensor * norm_out_b = nullptr;
};

struct Granite5CtcWeights {
    Granite5CtcEncTop                    enc_top;
    std::vector<Granite5CtcEncBlock>     enc_blocks;

    // Baked frontend buffers (optional; absent on a hand-built GGUF).
    ggml_tensor * frontend_mel_filterbank = nullptr;  // [n_freq, n_mels]
    ggml_tensor * frontend_window         = nullptr;  // [win_length]
};

transcribe_status build_granite5_ctc_weights(ggml_context *             ctx_meta,
                                             const Granite5CtcHParams & hp,
                                             Granite5CtcWeights &       weights);

}  // namespace transcribe::granite5_ctc
