// arch/parakeet/decoder.h - Parakeet TDT decoder.
//
// Phase 5 of the encoder/decoder bring-up. Implements the host-side
// predictor (2-layer LSTM) + joint network forward + TDT greedy
// decode driver. Reference: parakeet-mlx
//
//   /tmp/parakeet-mlx/parakeet_mlx/rnnt.py        (PredictNetwork, JointNetwork)
//   /tmp/parakeet-mlx/parakeet_mlx/parakeet.py    (ParakeetTDT.decode_greedy)
//
// Why this is on host (not a ggml backend graph): the per-step compute
// is small (a 640-wide LSTM step is ~6.5 MFLOPs; the joint pass is
// another ~2.7 MFLOPs for v2 / ~12 MFLOPs for v3), and an 11-second
// jfk.wav clip emits ~70 tokens, so the total decoder cost is well
// under a millisecond. Building a backend graph would add lifetime
// complexity (per-step input/output buffers, allocator reuse, state
// snapshotting on blank emission) for no measurable speedup. Both
// CPU and Metal are dominated by the encoder forward (~63 ms on M4
// Max) — the decoder is rounding error.
//
// Memory cost: a load-time host mirror of the predictor + joint
// weights is ~35 MB on v2 and ~73 MB on v3, against an existing
// ~2.4 GB encoder weight footprint. The mirror is built once in
// build_host_decoder_weights via ggml_backend_tensor_get (universal
// across host buffers, Metal unified memory, and future discrete
// GPUs).
//
// This header is INTERNAL to src/arch/parakeet/. The public C ABI
// only sees ParakeetContext::result populated by Parakeet::run.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::parakeet {

struct ParakeetModel;
struct ParakeetHParams;

// ---------------------------------------------------------------------------
// Host weight mirrors
// ---------------------------------------------------------------------------
//
// Layout convention: every weight matrix is stored in PyTorch
// row-major order [out, in], which means flat element (r, c) lives at
// `r * in + c`. This matches the bytes the converter writes (NeMo
// safetensors → no transpose for linear layers / LSTM gates), and it
// matches the ggml ne=[in, out] fast-to-slow layout — the same bytes,
// read with the row-major convention. See decoder.cpp for the matmul
// helper that operationalizes this.
//
// LSTM gate ordering: [i, f, g, o] (PyTorch / MLX standard) packed
// into a single [4*hidden, in] matrix. Wx (input-to-hidden) and Wh
// (hidden-to-hidden) are both stored this way; bias is a single
// concatenated [4*hidden] vector (NeMo's prednet uses one bias, not
// PyTorch's redundant W_ih + W_hh pair).

struct HostLstmLayer {
    std::vector<float> Wx;  // [4*pred_hidden, pred_hidden]
    std::vector<float> Wh;  // [4*pred_hidden, pred_hidden]
    std::vector<float> b;   // [4*pred_hidden]
};

struct HostPredictor {
    int                       pred_hidden = 0;
    int                       pred_vocab  = 0; // includes the +1 start row
    std::vector<float>        embed_w;          // [pred_vocab, pred_hidden]
    std::vector<HostLstmLayer> lstm;            // pred_n_layers entries
};

struct HostJoint {
    int         d_enc       = 0; // encoder d_model
    int         pred_hidden = 0;
    int         joint_h     = 0;
    int         joint_n     = 0; // total output classes (vocab+blank+durations)
    std::string activation;       // "relu" / "sigmoid" / "tanh"
    std::vector<float> enc_w;    // [joint_h, d_enc]
    std::vector<float> enc_b;    // [joint_h]
    std::vector<float> pred_w;   // [joint_h, pred_hidden]
    std::vector<float> pred_b;   // [joint_h]
    std::vector<float> out_w;    // [joint_n, joint_h]
    std::vector<float> out_b;    // [joint_n]
};

// Bundle of everything the decoder needs at run time. Built once at
// load() time and stored on ParakeetModel; const-after-construction
// and shared across every context derived from the model.
struct HostDecoderWeights {
    HostPredictor          predictor;
    HostJoint              joint;
    std::vector<int32_t>   tdt_durations;
    int                    tdt_max_symbols = 0;
    int                    blank_id        = 0; // == pred_vocab - 1
    int                    n_vocab         = 0; // == pred_vocab - 1; the SP vocab size
};

// Build host mirrors from a loaded ParakeetModel. Reads tensor bytes
// via ggml_backend_tensor_get so it works on every backend. Returns
// TRANSCRIBE_OK on success; on failure logs to stderr and leaves
// `out` in an indeterminate state.
transcribe_status build_host_decoder_weights(const ParakeetModel & model,
                                             HostDecoderWeights &  out);

// ---------------------------------------------------------------------------
// LSTM hidden state
// ---------------------------------------------------------------------------
//
// One (h, c) pair per LSTM layer. The decoder owns one "current" state
// (committed) and one "scratch" state (computed each iteration; only
// committed when a non-blank token is emitted). Both are sized once
// in reset() and reused across decode steps.

struct LstmState {
    // h[layer] and c[layer] are each `pred_hidden` floats.
    std::vector<std::vector<float>> h;
    std::vector<std::vector<float>> c;

    void reset(int n_layers, int pred_hidden);
};

// ---------------------------------------------------------------------------
// Decode result
// ---------------------------------------------------------------------------
//
// One emitted (non-blank) TDT token. `step_at_emit` is the encoder
// frame index when the token was emitted; `duration_frames` is the
// encoder-frame jump the joint network selected from the durations
// table. The decoder driver in Parakeet::run multiplies these by the
// model's frame-to-seconds ratio (subsampling_factor * hop_length /
// sample_rate) to convert to milliseconds for the public result
// accessors.

struct TdtToken {
    int   id              = 0;
    float p               = 0.0f; // entropy-based confidence
    int   step_at_emit    = 0;
    int   duration_frames = 0;
};

// Run TDT greedy decode end-to-end against an encoder output buffer.
//
// Inputs:
//   w           - the decoder weights mirror
//   enc_out     - host pointer to T_enc * d_enc fp32 floats. Layout
//                 is fast-to-slow [d_enc, T_enc] = row-major [T_enc, d_enc]:
//                 frame `t` lives at `enc_out + t * d_enc`. Matches
//                 the byte layout of a ggml encoder output tensor with
//                 ne=[d_enc, T_enc, 1, 1] read via
//                 ggml_backend_tensor_get.
//   T_enc       - number of encoder frames
//   d_enc       - encoder d_model (must equal w.joint.d_enc)
//
// Outputs:
//   out_tokens  - appended with one TdtToken per non-blank emission
//
// Side effects:
//   When TRANSCRIBE_DUMP_DIR is set, dumps a small set of decoder
//   intermediates (first-step embed, first-step LSTM h/c, first-step
//   joint logits) for the bring-up comparison harness. Production
//   builds (env var unset) pay zero cost.
//
// Returns TRANSCRIBE_OK on success. The decoder cannot fail at run
// time except via invalid arguments — input validation (T_enc > 0,
// d_enc matches the joint mirror, etc.) is enforced by the family
// driver before this is called.
transcribe_status decode_tdt_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    std::vector<TdtToken> &    out_tokens);

} // namespace transcribe::parakeet
