// arch/parakeet/weights.h - canonical Parakeet tensor catalog and
// per-instance weight slots.
//
// This header is INTERNAL to src/arch/parakeet/. It defines:
//
//   - ParakeetHParams: the architecture KV the loader reads from
//     stt.parakeet.* / stt.frontend.* before allocating any tensors.
//     Every dim that drives a tensor shape lives here.
//
//   - ParakeetWeights: a struct of named borrowed ggml_tensor* slots,
//     one per logical weight in a Parakeet model. The actual storage
//     is owned by the model's ggml_context (allocated by
//     gguf_init_from_file with no_alloc=false). The slots are
//     borrowed pointers and must not outlive the ggml_context.
//
// The corresponding source file walks the canonical tensor name list
// and validates each tensor (presence + shape) against the hparams,
// following llama.cpp's `create_tensor`-with-explicit-shape pattern.
// Required by default; the few honestly-optional tensors are wrapped
// in a NOT_REQUIRED flag.
//
// Naming conventions baked in here (and matched by the converter):
//
//   - Linear weight tensors are stored in PyTorch order [out, in].
//     ggml_mul_mat(W, x) takes the activation row vector x and reads
//     ne0 of W as the input dim — so PyTorch [out, in] is what we
//     want.
//   - Conv2d weight tensors are stored in PyTorch OIHW
//     [out_channels, in_channels, kH, kW], matching NeMo's native
//     layout. Phase 4 figures out whether to feed this directly to
//     ggml_conv_2d or transpose again at runtime — that's not 2C's
//     problem.
//   - Conv1d weight tensors (the depthwise/pointwise convs inside
//     the Conformer block) are stored as PyTorch
//     [out_channels, in_channels, kernel] for the pointwise convs and
//     [out_channels, kernel, 1] for the depthwise (kernel * groups,
//     where groups=out_channels=in_channels for depthwise).
//   - LSTM gate matrices (Wx, Wh) are stored as a single concatenated
//     [4*hidden, input] tensor in PyTorch order (i, f, g, o gates
//     stacked along the row dimension). Bias is concatenated [4*hidden].
//     This matches NeMo's safetensors layout exactly so the converter
//     does not transpose.
//   - LayerNorm and RMSNorm have separate weight + bias tensors of
//     shape [d_model].
//   - BatchNorm carries running_mean and running_var alongside
//     weight + bias. We keep all four; folding into the surrounding
//     conv happens at compute time in phase 4 (encoder is in eval
//     mode at inference and BN reduces to an affine transform).

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
//
// Read from stt.parakeet.* / stt.frontend.* via read_parakeet_hparams().
// Every field that controls a tensor shape MUST live here so the
// loader can validate weight shapes deterministically against KV
// (rather than against tensor sizes the converter happened to write).

// Decoder head dispatch. Resolved from stt.parakeet.head_kind (string KV).
// Legacy v2/v3 GGUFs predate the KV; absent KV defaults to TDT, which is
// what those GGUFs are. Drives conditional KV reads in
// read_parakeet_hparams (predictor/joint/tdt only when applicable) and
// the per-head decoder dispatch in model.cpp run().
enum class HeadKind { TDT, RNNT, CTC };

struct ParakeetHParams {
    HeadKind head_kind = HeadKind::TDT;

    // Encoder (Conformer).
    int32_t enc_n_layers          = 0;
    int32_t enc_d_model           = 0;
    int32_t enc_n_heads           = 0;
    int32_t enc_d_ff              = 0; // d_model * ff_expansion_factor
    int32_t enc_conv_kernel       = 0; // depthwise conv kernel inside the Conformer block
    int32_t enc_subsampling_factor   = 0; // total downsampling on the time axis
    int32_t enc_subsampling_channels = 0; // intermediate channel count in pre_encode
    int32_t enc_pos_emb_max_len   = 0;
    bool    enc_use_bias          = false;
    // NeMo's RelPositionalEncoding multiplies x by sqrt(d_model) when
    // `xscaling=True` is set in the encoder cfg. This pre-block scaling
    // is structurally different from how v2/v3/tdt-* (xscaling=False)
    // feed the pre_encode output into block 0. Default to false so
    // legacy GGUFs without this KV (v2/v3 etc.) keep working unchanged.
    bool    enc_xscaling          = false;
    // Local attention window. -1 / -1 = full attention (the default for
    // most variants). When non-negative, each query attends only to keys
    // within an [left, right]-bounded set whose exact semantics depend
    // on enc_att_context_style below.
    //
    // The two styles in use today:
    //
    //   AttContextStyle::Regular         — every other variant. When
    //     both left/right are >= 0 (parakeet-tdt_ctc-1.1b sets
    //     [128, 128]), pos_emb is shortened to (left+right+1) and the
    //     band mask sits in matrix_bd via the existing pad/slice path,
    //     matching NeMo's LocalAttRelPositionalEncoding.
    //
    //   AttContextStyle::ChunkedLimited  — nemotron-speech-streaming-en-0.6b
    //     ([70, 13]). Full RelPositionalEncoding (pos_emb stays at
    //     2T-1); a separate chunked mask is built host-side and added
    //     to matrix_bd. chunk_size = right+1, left_chunks = left/chunk_size.
    int32_t enc_att_context_left  = -1;
    int32_t enc_att_context_right = -1;

    // Self-attention context style (introduced by streaming variants).
    // Resolved from stt.parakeet.encoder.att_context_style (string KV);
    // optional, defaults to "regular" so every legacy GGUF stays on the
    // existing path.
    enum class AttContextStyle { Regular, ChunkedLimited };
    AttContextStyle enc_att_context_style = AttContextStyle::Regular;

    // Depthwise convolution context window inside the Conformer block.
    // -1 / -1 = symmetric (kernel-1)/2 on both sides (every offline
    // variant). NeMo's `conv_context_size = "causal"` emits
    // [kernel-1, 0] (nemotron-speech-streaming-en-0.6b uses kernel=9,
    // so [8, 0]) so the depthwise conv only consumes left-context.
    int32_t enc_conv_context_left  = -1;
    int32_t enc_conv_context_right = -1;

    // Conv-module normalisation choice. NeMo's offline default is
    // BatchNorm; streaming variants use LayerNorm to avoid stat
    // brittleness across chunks. With LayerNorm the GGUF carries the
    // same `conv.bn.weight` / `conv.bn.bias` tensor names but omits
    // running_mean / running_var, and the conformer applies an
    // unfused LayerNorm (compute per-channel mean/std at inference,
    // then affine-transform with bn_w / bn_b).
    enum class ConvNormType { BatchNorm, LayerNorm };
    ConvNormType enc_conv_norm_type = ConvNormType::BatchNorm;

    // Predictor (RNN-T prediction network). TDT and RNNT only; zero for CTC.
    int32_t pred_hidden   = 0;
    int32_t pred_n_layers = 0;
    int32_t pred_vocab    = 0; // includes the +1 "start" / blank-row entry of the embed matrix
                               // (raw token vocab size is pred_vocab - 1)

    // Joint network. TDT and RNNT only; zero for CTC.
    int32_t joint_hidden            = 0;
    int32_t joint_num_extra_outputs = 0; // TDT durations count; 0 for RNNT
    // Joint output activation. NeMo Parakeet 0.6B v2/v3 ship "relu"
    // (verified against config.json on both variants); the rnnt.py
    // reference defaults to "tanh" but no published Parakeet variant
    // we know about uses tanh. Read from stt.parakeet.joint.activation
    // and validated against the small allow-list the C++ joint forward
    // implements. Without this read at load time the C++ side would
    // hard-code an activation choice and silently produce wrong logits
    // on a model that asked for a different one.
    std::string joint_activation;

    // TDT decoding parameters.
    //
    // tdt_durations: the duration values the joint network's "extra
    // outputs" axis predicts (e.g. [0,1,2,3,4] for the published 0.6B
    // variants). The argmax over duration logits selects an index into
    // this list; the chosen integer is how many encoder frames the TDT
    // decode driver advances after emitting a token. PLAN.md fixes the
    // KV name as `stt.parakeet.tdt.durations` (i32 array). Length must
    // equal joint_num_extra_outputs (we cross-validate at load time).
    std::vector<int32_t> tdt_durations;

    // tdt_max_symbols: the per-encoder-frame "stuck" cap from NeMo's
    // greedy_batch decoder. Optional KV with a documented default of 10
    // (matches both published v2 and v3 configs). When the decoder has
    // emitted N consecutive zero-duration tokens for the same frame
    // without advancing time, it forces a +1 advance to break the
    // loop. Set to 0 to disable.
    int32_t tdt_max_symbols = 10;

    // Frontend (mel feature extractor). The full set of stt.frontend.*
    // KV the converter emits and the phase 3 C++ frontend reads. PLAN.md
    // declares this list as the *complete* set the converter and loader
    // must agree on, so the loader is the gate that catches any future
    // converter that drops a field. Fields irrelevant to Parakeet
    // (LFR window/shift, CMVN mean/inv_stddev) are not in this struct
    // because no published Parakeet variant uses them — when SenseVoice
    // or another LFR/CMVN family lands, those fields move to a
    // family-agnostic FrontendParams struct.
    std::string fe_type;          // "mel" for parakeet
    int32_t     fe_num_mels    = 0;
    int32_t     fe_sample_rate = 0;
    int32_t     fe_n_fft       = 0;
    int32_t     fe_win_length  = 0;  // samples
    int32_t     fe_hop_length  = 0;  // samples
    std::string fe_window;        // "hann" for parakeet
    std::string fe_normalize;     // "per_feature" for parakeet
    float       fe_dither       = 0.0f;
    float       fe_pre_emphasis = 0.0f;  // 0.0 == disabled
    float       fe_f_min        = 0.0f;
    float       fe_f_max        = 0.0f;

    // CTC head. Populated by build_parakeet_weights when head_kind=CTC,
    // by reading the leading dim of head.ctc.weight (shape
    // [vocab+1, d_model, 1] in PyTorch -> ggml ne [1, d_model, vocab+1]).
    // Zero for TDT/RNNT.
    int32_t head_ctc_n_classes = 0;

    // Derived. Convenience accessors keyed off the above.
    int32_t enc_head_dim() const { return enc_n_heads > 0 ? enc_d_model / enc_n_heads : 0; }
    int32_t joint_n_classes() const { return (pred_vocab - 1) + joint_num_extra_outputs + 1; }
};

// Read every required stt.parakeet.* / stt.frontend.* KV into hp.
// Returns:
//   TRANSCRIBE_OK              on success.
//   TRANSCRIBE_ERR_INVALID_ARG if gguf is null.
//   TRANSCRIBE_ERR_GGUF        if a required key is missing or has the
//                              wrong type, or if cross-field invariants
//                              don't hold (e.g. d_model not divisible
//                              by n_heads).
transcribe_status read_parakeet_hparams(const gguf_context * gguf,
                                        ParakeetHParams &    hp);

// ---------------------------------------------------------------------------
// Weight slots
// ---------------------------------------------------------------------------
//
// Every ggml_tensor pointer below is a borrowed pointer into the
// model's ggml_context (the one that owns the data buffer). They are
// invalidated when the model is freed.

struct ParakeetPreEncode {
    // dw_striding subsampling: 1 standard conv + 2 depthwise-separable
    // conv pairs = 8x downsampling on the time axis. The numeric
    // indices in the names below come from NeMo's torch.nn.Sequential
    // layout (1 and 4 are activation modules, hence skipped).
    ggml_tensor * conv0_w = nullptr; // [out=channels,        in=1,        kh=3, kw=3]
    ggml_tensor * conv0_b = nullptr; // [channels]
    ggml_tensor * conv2_w = nullptr; // [out=channels,        in=1 (dw),   kh=3, kw=3]
    ggml_tensor * conv2_b = nullptr; // [channels]
    ggml_tensor * conv3_w = nullptr; // [out=channels,        in=channels, kh=1, kw=1]
    ggml_tensor * conv3_b = nullptr; // [channels]
    ggml_tensor * conv5_w = nullptr; // [out=channels,        in=1 (dw),   kh=3, kw=3]
    ggml_tensor * conv5_b = nullptr; // [channels]
    ggml_tensor * conv6_w = nullptr; // [out=channels,        in=channels, kh=1, kw=1]
    ggml_tensor * conv6_b = nullptr; // [channels]
    // Final projection from flattened [channels * (num_mels / subsampling)] to d_model.
    ggml_tensor * out_w   = nullptr;
    ggml_tensor * out_b   = nullptr;
};

// One Conformer block. Same shape contract for every block; the loader
// builds n_layers of these. The bias slots (`*_b` on the linear/conv
// layers, NOT on the layer norms / BN / pos_bias) are populated only
// when `enc_use_bias=true` — left null for the v2/v3 baseline. The
// shared `transcribe::conformer::BlockView` already accepts nullable
// biases on every linear/conv slot.
struct ParakeetBlock {
    // Macaron feed-forward 1.
    ggml_tensor * norm_ff1_w = nullptr; // [d_model]
    ggml_tensor * norm_ff1_b = nullptr; // [d_model]
    ggml_tensor * ff1_lin1_w = nullptr; // [d_ff,    d_model]
    ggml_tensor * ff1_lin1_b = nullptr; // [d_ff],    nullable (use_bias)
    ggml_tensor * ff1_lin2_w = nullptr; // [d_model, d_ff]
    ggml_tensor * ff1_lin2_b = nullptr; // [d_model], nullable (use_bias)

    // Self-attention with relative positional encoding.
    ggml_tensor * norm_attn_w  = nullptr; // [d_model]
    ggml_tensor * norm_attn_b  = nullptr; // [d_model]
    ggml_tensor * attn_q_w     = nullptr; // [d_model, d_model]
    ggml_tensor * attn_q_b     = nullptr; // [d_model], nullable (use_bias)
    ggml_tensor * attn_k_w     = nullptr; // [d_model, d_model]
    ggml_tensor * attn_k_b     = nullptr; // [d_model], nullable (use_bias)
    ggml_tensor * attn_v_w     = nullptr; // [d_model, d_model]
    ggml_tensor * attn_v_b     = nullptr; // [d_model], nullable (use_bias)
    ggml_tensor * attn_out_w   = nullptr; // [d_model, d_model]
    ggml_tensor * attn_out_b   = nullptr; // [d_model], nullable (use_bias)
    ggml_tensor * attn_pos_w   = nullptr; // [d_model, d_model]   (linear_pos is bias-free in NeMo even when use_bias=true)
    ggml_tensor * attn_pos_u   = nullptr; // [n_heads, head_dim]   learned rel-pos bias u
    ggml_tensor * attn_pos_v   = nullptr; // [n_heads, head_dim]   learned rel-pos bias v

    // Convolution module: pointwise -> GLU -> depthwise -> BN -> activation -> pointwise.
    ggml_tensor * norm_conv_w  = nullptr; // [d_model]
    ggml_tensor * norm_conv_b  = nullptr; // [d_model]
    ggml_tensor * conv_pw1_w   = nullptr; // [2*d_model, d_model, 1]   conv1d k=1; 2x for GLU
    ggml_tensor * conv_pw1_b   = nullptr; // [2*d_model], nullable (use_bias)
    ggml_tensor * conv_dw_w    = nullptr; // [d_model,   1,       k]   depthwise conv1d
    ggml_tensor * conv_dw_b    = nullptr; // [d_model],   nullable (use_bias)
    ggml_tensor * conv_pw2_w   = nullptr; // [d_model,   d_model, 1]   conv1d k=1
    ggml_tensor * conv_pw2_b   = nullptr; // [d_model],   nullable (use_bias)
    ggml_tensor * conv_bn_w    = nullptr; // [d_model]   BN scale
    ggml_tensor * conv_bn_b    = nullptr; // [d_model]   BN shift
    ggml_tensor * conv_bn_rm   = nullptr; // [d_model]   running_mean
    ggml_tensor * conv_bn_rv   = nullptr; // [d_model]   running_var

    // Fused BN parameters (computed at load time, replaces batch_norm()
    // in the graph with a single mul + add):
    //   fused_scale = bn_w / sqrt(bn_rv + eps)
    //   fused_bias  = bn_b - bn_rm * fused_scale
    ggml_tensor * conv_bn_fused_scale = nullptr; // [d_model]
    ggml_tensor * conv_bn_fused_bias  = nullptr; // [d_model]

    // Macaron feed-forward 2.
    ggml_tensor * norm_ff2_w = nullptr; // [d_model]
    ggml_tensor * norm_ff2_b = nullptr; // [d_model]
    ggml_tensor * ff2_lin1_w = nullptr; // [d_ff,    d_model]
    ggml_tensor * ff2_lin1_b = nullptr; // [d_ff],    nullable (use_bias)
    ggml_tensor * ff2_lin2_w = nullptr; // [d_model, d_ff]
    ggml_tensor * ff2_lin2_b = nullptr; // [d_model], nullable (use_bias)

    // Final per-block layer norm.
    ggml_tensor * norm_out_w = nullptr; // [d_model]
    ggml_tensor * norm_out_b = nullptr; // [d_model]
};

struct ParakeetPredictor {
    // Token embedding. The +1 row is the "start of sequence / no
    // previous token" embedding NeMo prepends to vocab_size.
    ggml_tensor * embed_w = nullptr; // [pred_vocab, pred_hidden]

    // 2-layer LSTM by default for v2/v3 0.6b. Concatenated gate
    // weights in PyTorch (i, f, g, o) order, 4*pred_hidden rows.
    struct LstmLayer {
        ggml_tensor * Wx = nullptr; // [4*pred_hidden, pred_hidden]   (input-to-hidden, layer 0 input dim is pred_hidden because of the embed)
        ggml_tensor * Wh = nullptr; // [4*pred_hidden, pred_hidden]   (hidden-to-hidden)
        ggml_tensor * b  = nullptr; // [4*pred_hidden]
    };
    std::vector<LstmLayer> lstm; // pred_n_layers entries
};

struct ParakeetJoint {
    ggml_tensor * enc_w  = nullptr; // [joint_hidden, enc_d_model]
    ggml_tensor * enc_b  = nullptr; // [joint_hidden]
    ggml_tensor * pred_w = nullptr; // [joint_hidden, pred_hidden]
    ggml_tensor * pred_b = nullptr; // [joint_hidden]
    ggml_tensor * out_w  = nullptr; // [joint_n_classes, joint_hidden]
    ggml_tensor * out_b  = nullptr; // [joint_n_classes]
};

// CTC head. Single 1×1 Conv1d projecting d_model -> vocab+1. Loaded
// only when head_kind=CTC; predictor + joint stay empty in that case.
struct ParakeetCtcHead {
    ggml_tensor * weight = nullptr; // [vocab+1, d_model, 1]   conv1d k=1
    ggml_tensor * bias   = nullptr; // [vocab+1]
};

struct ParakeetWeights {
    ParakeetPreEncode          pre_encode;
    std::vector<ParakeetBlock> blocks; // hp.enc_n_layers entries
    ParakeetPredictor          predictor; // empty when head_kind=CTC
    ParakeetJoint              joint;     // empty when head_kind=CTC
    ParakeetCtcHead            ctc_head;  // populated when head_kind=CTC
};

// Walk the canonical tensor list, look up each tensor by name in the
// gguf, validate its shape against hp, and store the borrowed pointer
// in the matching slot of weights. The ggml_tensor objects must
// already exist in ctx_meta — typically because the caller invoked
// gguf_init_from_file with no_alloc=false and ctx=&ctx_meta, which
// both creates ggml_tensors AND reads the data section.
//
// Follows llama.cpp's load_tensors pattern: every required tensor
// must be present with the exact expected shape; missing or
// shape-mismatched tensors return TRANSCRIBE_ERR_GGUF with a log
// message naming the offending tensor.
//
// On failure the partially-built `weights` is left in an
// indeterminate state — the caller is expected to throw the whole
// model away.
transcribe_status build_parakeet_weights(ggml_context *          ctx_meta,
                                         const ParakeetHParams & hp,
                                         ParakeetWeights &       weights);

} // namespace transcribe::parakeet
