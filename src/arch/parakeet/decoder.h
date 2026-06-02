// arch/parakeet/decoder.h - Parakeet TDT decoder.
//
// Phase 5 of the encoder/decoder bring-up. Implements the host-side
// predictor (2-layer LSTM) + joint network forward + TDT greedy
// decode driver.
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
// only sees ParakeetSession::result populated by Parakeet::run.

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
// LSTM gate ordering: [i, f, g, o] (PyTorch standard) packed into a
// single [4*hidden, in] matrix. Wx (input-to-hidden) and Wh
// (hidden-to-hidden) are both stored this way; bias is the combined
// W_ih + W_hh bias pre-summed at conversion time.

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

// CTC head mirror. NeMo's `decoder.decoder_layers.0` is a 1×1 Conv1d
// projecting d_enc -> n_classes (= vocab + 1 blank). Stored row-major
// [n_classes, d_enc] so per-frame logits = W @ enc_t + b. The host
// decode path runs entirely on fp32 (same `read_tensor_to_f32` as the
// predictor/joint mirrors), so any of the GET_LIN-allowed source
// types are safe.
struct HostCtcHead {
    int                 n_classes = 0; // == vocab + 1
    int                 blank_id  = 0; // NeMo convention: blank lives at n_classes - 1
    int                 d_enc     = 0;
    std::vector<float>  weight;        // [n_classes, d_enc]
    std::vector<float>  bias;          // [n_classes]
};

// Decoder head selector. Mirrors ParakeetHParams::HeadKind so the
// host decoder can dispatch without depending on weights.h. Stays in
// sync via the assignment in build_host_decoder_weights.
enum class HostHeadKind { TDT, RNNT, CTC };

// Bundle of everything the decoder needs at run time. Built once at
// load() time and stored on ParakeetModel; const-after-construction
// and shared across every context derived from the model. For
// head_kind=CTC the predictor + joint mirrors stay empty; for
// head_kind=RNNT the tdt_durations vector is empty and only blank/non-blank
// symbol-cap logic from `tdt_max_symbols` is reused.
struct HostDecoderWeights {
    HostHeadKind           head_kind        = HostHeadKind::TDT;
    HostPredictor          predictor;        // empty for CTC
    HostJoint              joint;            // empty for CTC
    HostCtcHead            ctc_head;         // empty for TDT/RNNT
    std::vector<int32_t>   tdt_durations;    // empty for RNNT/CTC
    int                    tdt_max_symbols  = 0;
    int                    blank_id         = 0; // unified: TDT/RNNT == pred_vocab - 1; CTC == ctc_head.blank_id
    int                    n_vocab          = 0; // raw SP vocab size (excludes blank)
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

// Run RNNT greedy decode end-to-end. Same predictor + joint code as TDT,
// but the joint emits `vocab+1` logits (no duration extras) and the
// step rule is "blank → advance one frame, non-blank → emit + stay,
// capped by tdt_max_symbols". Per-emit `duration_frames` is fixed at 1
// — the public token timestamps approximate the reference's
// (encoder-frame indexed) emission.
//
// Same input/output contract as decode_tdt_greedy. Reuses the same
// TdtToken result type so the model.cpp result-builder can stay
// head-agnostic.
transcribe_status decode_rnnt_greedy(const HostDecoderWeights & w,
                                     const float *              enc_out,
                                     int                        T_enc,
                                     int                        d_enc,
                                     std::vector<TdtToken> &    out_tokens);

// Streaming variant of RNN-T greedy decode. Consumes T_enc_new
// encoder frames (the chunk just produced by the streaming encoder
// graph) and APPENDS the emitted tokens to out_tokens. LSTM state and
// previous-token id are carried across calls via state_io and
// last_token_io. frame_offset is the absolute encoder-frame index of
// the first frame in this chunk (so emitted tokens'
// step_at_emit lands in stream-wide coordinates).
//
// state_io must have been reset to a fresh "start of sequence" state
// at stream_begin (LstmState::reset for n_layers, pred_hidden, plus
// last_token_io = -1 for the SOS embedding zero).
//
// No timing log is printed (per-chunk noise on the CLI).
transcribe_status decode_rnnt_greedy_streaming(
    const HostDecoderWeights & w,
    const float *              enc_out,
    int                        T_enc_new,
    int                        d_enc,
    LstmState &                state_io,
    int &                      last_token_io,
    int                        frame_offset,
    std::vector<TdtToken> &    out_tokens);

// Run CTC greedy decode end-to-end. Per-frame: logits = W @ enc[t] + b,
// argmax. The collapse rule is "drop adjacent duplicates, then drop
// blanks" — standard CTC greedy (see e.g. NeMo's GreedyCTCInfer).
// `step_at_emit` is the encoder-frame index of the (post-collapse)
// emission; `duration_frames` is set to 1 for compatibility with the
// public token timestamps.
//
// Same TdtToken result type as the transducer paths so model.cpp can
// build the public result hierarchy uniformly.
transcribe_status decode_ctc_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    std::vector<TdtToken> &    out_tokens);

} // namespace transcribe::parakeet
