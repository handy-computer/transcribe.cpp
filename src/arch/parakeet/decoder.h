// arch/parakeet/decoder.h - Parakeet TDT decoder.
//
// Implements the predictor (2-layer LSTM) + joint network forward +
// greedy decode driver (TDT, RNN-T, and CTC heads).
//
// Execution model. The decoder runs entirely as ggml graphs on ONE shared,
// per-decode CPU threadpool: the predictor LSTM (per-step graph), the
// per-utterance encoder projection (one GEMM), and the joint network (pred_w
// proj + sum + activation + out_w proj, one graph). PredGraph owns the backend
// and threadpool; the enc_proj and joint graphs borrow it, so the whole decode
// shares a single pool with no oversubscription. Every decode call builds its
// own graphs (reentrant); the weights are model-resident fp32 ggml tensors
// (HostPredictor::lstm_w_* and HostJoint::gw_*/g_pred_*), built once at load.
//
// This is a deliberate "one threading system" design: the decoder used to run
// host fp32 matmuls on a bespoke worker pool, which coexisted badly with ggml's
// own pool (teardown / oversubscription / MSVC-vcomp bugs). Routing everything
// through ggml deleted that second system. The CPU backend is always available
// (the decoder runs on host even when the model is on a GPU), so a graph build
// failure is a hard decode error rather than a host fallback.
//
// All weights are fp32 (the LSTM and joint matmuls are n=1 GEMV per step, where
// quantization buys no usable bandwidth and would only add drift); fp32 keeps
// decode numerically faithful to the historical host reference. Any joint
// quantization is a model/quant-level concern, not done here.
//
// Memory cost: a load-time host + resident-ggml mirror of the predictor + joint
// weights (~35 MB on v2, ~73 MB on v3) against the ~2.4 GB encoder footprint.
// The mirror is built once in build_host_decoder_weights via
// ggml_backend_tensor_get (universal across host buffers, Metal unified
// memory, and discrete GPUs).
//
// This header is INTERNAL to src/arch/parakeet/. The public C ABI
// only sees ParakeetSession::result populated by Parakeet::run.

#pragma once

#include "transcribe.h"

#include "ggml.h"
#include "ggml-backend.h"

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
    // Host fp32 mirrors — LOAD-TIME SCRATCH ONLY: filled at load, uploaded into
    // the resident ggml tensors below by build_pred_weights, then freed.
    std::vector<float> Wx;  // [4*pred_hidden, pred_hidden] (freed after load)
    std::vector<float> Wh;  // [4*pred_hidden, pred_hidden] (freed after load)
    std::vector<float> b;   // [4*pred_hidden]              (freed after load)

    // Resident fp32 ggml mirrors of the three weights above, consumed by the
    // ggml predictor graph. Row-major [4*H, H] host bytes map to ggml ne
    // fast-to-slow [H, 4*H] (the mul_mat operand); bias is [4*H]. Borrowed
    // pointers into HostPredictor::lstm_w_ctx — owned/freed by ~HostPredictor.
    ggml_tensor * g_Wx = nullptr;
    ggml_tensor * g_Wh = nullptr;
    ggml_tensor * g_b  = nullptr;
};

struct HostPredictor {
    int                       pred_hidden = 0;
    int                       pred_vocab  = 0; // includes the +1 start row
    std::vector<float>        embed_w;          // [pred_vocab, pred_hidden]
    std::vector<HostLstmLayer> lstm;            // pred_n_layers entries

    // --- resident ggml LSTM weights (immutable, model-owned) ---
    // fp32 mirror of the per-layer Wx/Wh/b, made resident once at load so the
    // per-decode-call PredGraph reads them without re-uploading. Same
    // immutable-half / mutable-per-call split as HostJoint::gw_* + the
    // stack-local JointGraph. fp32 (not native quant): the per-step LSTM
    // matmuls are n=1 GEMV — tinyBLAS skips them (n<2) and they gain no usable
    // weight-bandwidth benefit, so fp32 stays closest to the host reference at
    // no speed cost. Owned here; freed by ~HostPredictor. On build failure
    // lstm_ready stays false and the decode reports a hard error.
    ggml_context *        lstm_w_ctx     = nullptr;
    ggml_backend_t        lstm_w_backend = nullptr; // alloc-only; never compute'd
    ggml_backend_buffer_t lstm_w_buf     = nullptr;
    bool                  lstm_ready     = false;

    HostPredictor() = default;
    ~HostPredictor();
    HostPredictor(const HostPredictor &)             = delete;
    HostPredictor & operator=(const HostPredictor &) = delete;
    HostPredictor(HostPredictor &&)                  = delete;
    HostPredictor & operator=(HostPredictor &&)      = delete;
};

struct HostJoint {
    int         d_enc       = 0; // encoder d_model
    int         pred_hidden = 0;
    int         joint_h     = 0;
    int         joint_n     = 0; // total output classes (vocab+blank+durations)
    std::string activation;       // "relu" / "sigmoid" / "tanh"

    // Host fp32 weight mirrors — LOAD-TIME SCRATCH ONLY. read_tensor_to_f32
    // fills these at load; build_joint_weight uploads them into the resident
    // ggml tensors below and then frees them, so the decode hot path reads only
    // the resident tensors. (out_w is not mirrored here — gw_w is built straight
    // from the model tensor.)
    std::vector<float> enc_w;    // [joint_h, d_enc]       (freed after load)
    std::vector<float> enc_b;    // [joint_h]              (freed after load)
    std::vector<float> pred_w;   // [joint_h, pred_hidden] (freed after load)
    std::vector<float> pred_b;   // [joint_h]              (freed after load)
    std::vector<float> out_b;    // [joint_n]              (freed after load)

    // --- resident ggml weights (immutable, model-owned) ---
    // The whole joint runs as one ggml graph (build_joint_graph), so every joint
    // weight is resident as an fp32 ggml tensor: the encoder projection
    // (g_enc_w/g_enc_b), the predictor projection (g_pred_w/g_pred_b), and the
    // output projection (gw_w/gw_b). fp32 keeps decode faithful to the host
    // reference; any joint quantization is a model/quant-level concern. Built
    // once at load and only read after — safe to share across every context;
    // the mutable per-decode state lives in a stack-local JointGraph (see
    // decoder.cpp), which is what keeps concurrent decode reentrant. Owned here;
    // freed by ~HostJoint.
    ggml_context *        w_ctx     = nullptr;
    ggml_backend_t        w_backend = nullptr; // alloc-only; never graph_compute'd
    ggml_backend_buffer_t w_buf     = nullptr;
    ggml_tensor *         g_enc_w   = nullptr; // [d_enc, joint_h] fp32 weight
    ggml_tensor *         g_enc_b   = nullptr; // [joint_h] fp32 bias
    ggml_tensor *         g_pred_w  = nullptr; // [pred_hidden, joint_h] fp32 weight
    ggml_tensor *         g_pred_b  = nullptr; // [joint_h] fp32 bias
    ggml_tensor *         gw_w      = nullptr; // [joint_h, joint_n] fp32 weight
    ggml_tensor *         gw_b      = nullptr; // [joint_n] fp32 bias
    bool                  w_ready   = false;

    HostJoint() = default;
    ~HostJoint();
    HostJoint(const HostJoint &)             = delete;
    HostJoint & operator=(const HostJoint &) = delete;
    HostJoint(HostJoint &&)                  = delete;
    HostJoint & operator=(HostJoint &&)      = delete;
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
                                    int                        n_threads,
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
                                     int                        n_threads,
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
    int                        n_threads,
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
                                    int                        n_threads,
                                    std::vector<TdtToken> &    out_tokens);

} // namespace transcribe::parakeet
