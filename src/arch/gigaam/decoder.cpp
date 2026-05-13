// arch/gigaam/decoder.cpp - GigaAM RNN-T greedy decode on host.
//
// The decoder is tiny (1 LSTM layer of 320, single joint linear) so the
// per-step compute fits cleanly in CPU cache; running it on host avoids
// backend dispatch overhead for the inner symbol-loop. Mirrors
// parakeet's host-decode pattern.
//
// PyTorch nn.LSTM stores its gate weights as a single concatenated
// [4*hidden, hidden] tensor in (i, f, g, o) gate order. The converter
// emits Wx and Wh in that layout (ggml ne=[hidden, 4*hidden]) and
// collapses bias_ih + bias_hh into a single [4*hidden] bias. The host
// LSTM step here reads them in numpy-row-major [4*hidden, hidden]
// orientation, which matches the byte layout directly.

#include "decoder.h"

#include "gigaam.h"
#include "transcribe-debug.h"
#include "weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace transcribe::gigaam {

namespace {

// Copy a ggml tensor's data into a host f32 vector. The tensor must be
// fp32; assertion would be loud and clear if the converter ever switched
// dtypes on us.
void readback_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        out.clear();
        return;
    }
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    out.assign(n, 0.0f);
    if (n == 0) return;
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
}

inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// y = W @ x + b, where W is [out, in] row-major (numpy shape), x is
// [in], b is [out]. b may be nullptr to skip the add.
void matvec_add(const float * W,
                const float * b,
                const float * x,
                int           out,
                int           in_dim,
                float *       y)
{
    for (int i = 0; i < out; ++i) {
        float acc = (b != nullptr) ? b[i] : 0.0f;
        const float * row = W + static_cast<size_t>(i) * in_dim;
        for (int j = 0; j < in_dim; ++j) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

} // namespace

// Build host buffers for the predictor + joint weights. Reads back from
// the backend exactly once at load time; the decode loop then iterates
// the host floats.
transcribe_status build_host_decoder_weights(const GigaamWeights & w,
                                             const GigaamHParams & hp,
                                             HostDecoderWeights &  out)
{
    if (hp.head_kind == HeadKind::CTC) {
        readback_f32(w.ctc_head.weight, out.ctc_w);
        readback_f32(w.ctc_head.bias,   out.ctc_b);
        return TRANSCRIBE_OK;
    }
    // RNN-T.
    readback_f32(w.predictor.embed_w, out.pred_embed);
    out.lstm_Wx.resize(hp.pred_n_layers);
    out.lstm_Wh.resize(hp.pred_n_layers);
    out.lstm_b.resize(hp.pred_n_layers);
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        readback_f32(w.predictor.lstm[i].Wx, out.lstm_Wx[i]);
        readback_f32(w.predictor.lstm[i].Wh, out.lstm_Wh[i]);
        readback_f32(w.predictor.lstm[i].b,  out.lstm_b[i]);
    }
    readback_f32(w.joint.enc_w,  out.joint_enc_w);
    readback_f32(w.joint.enc_b,  out.joint_enc_b);
    readback_f32(w.joint.pred_w, out.joint_pred_w);
    readback_f32(w.joint.pred_b, out.joint_pred_b);
    readback_f32(w.joint.out_w,  out.joint_out_w);
    readback_f32(w.joint.out_b,  out.joint_out_b);
    return TRANSCRIBE_OK;
}

// One LSTM step (single batch). Returns the new hidden vector (= LSTM
// output) into `h_out`, and the new cell state into `c_out`. Inputs:
//   x:      [pred_hidden] embedding (or zero on the first step)
//   h_prev: [pred_hidden]
//   c_prev: [pred_hidden]
//   Wx:     [4*pred_hidden, pred_hidden] flat row-major
//   Wh:     [4*pred_hidden, pred_hidden]
//   b:      [4*pred_hidden] (already collapsed bias_ih + bias_hh)
// Gate order is PyTorch (i, f, g, o), each block size `pred_hidden`.
static void lstm_step(const float * x,
                      const float * h_prev,
                      const float * c_prev,
                      const float * Wx,
                      const float * Wh,
                      const float * b,
                      int           H,
                      float *       h_out,
                      float *       c_out,
                      float *       scratch_gates)
{
    // scratch_gates: [4*H]. Compute Wx @ x + Wh @ h_prev + b in place.
    for (int i = 0; i < 4 * H; ++i) {
        float acc = b[i];
        const float * wx_row = Wx + static_cast<size_t>(i) * H;
        const float * wh_row = Wh + static_cast<size_t>(i) * H;
        for (int j = 0; j < H; ++j) {
            acc += wx_row[j] * x[j] + wh_row[j] * h_prev[j];
        }
        scratch_gates[i] = acc;
    }

    // Split gates: i = 0..H, f = H..2H, g = 2H..3H, o = 3H..4H.
    const float * gi = scratch_gates;
    const float * gf = scratch_gates + H;
    const float * gg = scratch_gates + 2 * H;
    const float * go = scratch_gates + 3 * H;

    for (int j = 0; j < H; ++j) {
        const float i_t = sigmoidf(gi[j]);
        const float f_t = sigmoidf(gf[j]);
        const float g_t = std::tanh(gg[j]);
        const float o_t = sigmoidf(go[j]);
        const float c_new = f_t * c_prev[j] + i_t * g_t;
        c_out[j] = c_new;
        h_out[j] = o_t * std::tanh(c_new);
    }
}

// One joint forward pass for batch=1:
//   enc_proj  = enc_w @ f_t  + enc_b           [joint_hidden]
//   pred_proj = pred_w @ g_t + pred_b          [joint_hidden]
//   logits    = out_w @ relu(enc_proj + pred_proj) + out_b   [n_classes]
//
// Returns the argmax token id.
static int joint_argmax(const HostDecoderWeights & h,
                        const float *              f_t,    // [enc_d_model]
                        const float *              g_t,    // [pred_hidden]
                        int                        enc_d,
                        int                        pred_d,
                        int                        joint_h,
                        int                        n_classes,
                        std::vector<float> &       scratch_join, // size joint_h
                        std::vector<float> &       scratch_logits)
{
    scratch_join.assign(joint_h, 0.0f);
    scratch_logits.assign(n_classes, 0.0f);

    // enc_proj.
    matvec_add(h.joint_enc_w.data(),
               h.joint_enc_b.data(),
               f_t, joint_h, enc_d,
               scratch_join.data());
    // pred_proj added in place + ReLU fused.
    std::vector<float> pj(joint_h, 0.0f);
    matvec_add(h.joint_pred_w.data(),
               h.joint_pred_b.data(),
               g_t, joint_h, pred_d,
               pj.data());
    for (int i = 0; i < joint_h; ++i) {
        float v = scratch_join[i] + pj[i];
        scratch_join[i] = v > 0.0f ? v : 0.0f;
    }
    // logits = out_w @ relu + out_b. The reference applies log_softmax
    // but argmax is invariant under it, so we skip.
    matvec_add(h.joint_out_w.data(),
               h.joint_out_b.data(),
               scratch_join.data(), n_classes, joint_h,
               scratch_logits.data());

    int   best_idx = 0;
    float best_val = scratch_logits[0];
    for (int i = 1; i < n_classes; ++i) {
        if (scratch_logits[i] > best_val) {
            best_val = scratch_logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

// Run RNN-T greedy decode against the encoder output. `encoded` is a
// host-side buffer laid out [T_enc * d_model] in T-major order (each
// contiguous d_model-sized chunk is one time step's encoder vector) —
// matches the rnnt.encoded reference dump.
//
// Returns the emitted token sequence (excluding blanks) and the
// per-token encoder frame indices.
transcribe_status decode_rnnt_greedy(const HostDecoderWeights & host,
                                     const GigaamHParams &      hp,
                                     const float *              encoded,
                                     int                        T_enc,
                                     int                        max_symbols_per_step,
                                     std::vector<int> &         out_tokens,
                                     std::vector<int> &         out_frames)
{
    out_tokens.clear();
    out_frames.clear();

    const int H        = hp.pred_hidden;
    const int enc_d    = hp.enc_d_model;
    const int joint_h  = hp.joint_hidden;
    const int n_class  = hp.joint_n_classes;
    const int blank_id = n_class - 1;

    std::vector<float> h_cur(H, 0.0f);
    std::vector<float> c_cur(H, 0.0f);
    std::vector<float> h_next(H, 0.0f);
    std::vector<float> c_next(H, 0.0f);
    std::vector<float> gates(4 * H, 0.0f);
    std::vector<float> pred_in(H, 0.0f);   // embed lookup output
    std::vector<float> pred_out(H, 0.0f);  // LSTM output
    std::vector<float> jh(joint_h, 0.0f);
    std::vector<float> logits(n_class, 0.0f);

    bool fresh = true; // first step uses zero input + zero state (reference predict(None, None))

    for (int t = 0; t < T_enc; ++t) {
        const float * f_t = encoded + static_cast<size_t>(t) * enc_d;

        int sym_count = 0;
        for (;;) {
            // Predictor input.
            if (fresh) {
                // Zero embedding, zero state.
                std::fill(pred_in.begin(), pred_in.end(), 0.0f);
            } else {
                // Embed the previously emitted token.
                const int prev = out_tokens.back();
                std::memcpy(pred_in.data(),
                            host.pred_embed.data() +
                                static_cast<size_t>(prev) * H,
                            H * sizeof(float));
            }

            // LSTM step (1 layer; multi-layer would chain pred_in / pred_out).
            lstm_step(pred_in.data(),
                      h_cur.data(), c_cur.data(),
                      host.lstm_Wx[0].data(),
                      host.lstm_Wh[0].data(),
                      host.lstm_b[0].data(),
                      H,
                      h_next.data(), c_next.data(),
                      gates.data());
            pred_out = h_next; // LSTM output = h_next for the last layer

            const int tok = joint_argmax(host, f_t, pred_out.data(),
                                          enc_d, H, joint_h, n_class,
                                          jh, logits);

            if (tok == blank_id) {
                // Advance time. Do NOT commit state.
                break;
            }
            // Emit token; commit state.
            out_tokens.push_back(tok);
            out_frames.push_back(t);
            h_cur = h_next;
            c_cur = c_next;
            fresh = false;
            ++sym_count;
            if (max_symbols_per_step > 0 && sym_count >= max_symbols_per_step) {
                break;
            }
        }
    }

    return TRANSCRIBE_OK;
}

// CTC greedy decode. The 1×1 Conv1d head is a per-frame linear:
//   logits[t, :] = ctc_w @ encoded[t, :] + ctc_b
// Layout of host.ctc_w after readback: ggml ne=[1, d_model, n_classes]
// is byte-equivalent to a row-major [n_classes, d_model] matrix (the
// k=1 axis collapses), which is what matvec_add expects.
transcribe_status decode_ctc_greedy(const HostDecoderWeights & host,
                                    const GigaamHParams &      hp,
                                    const float *              encoded,
                                    int                        T_enc,
                                    std::vector<int> &         out_tokens,
                                    std::vector<int> &         out_frames)
{
    out_tokens.clear();
    out_frames.clear();

    const int d_model   = hp.enc_d_model;
    const int n_classes = hp.head_n_classes;
    const int blank_id  = n_classes - 1;

    if (encoded == nullptr || T_enc <= 0 || d_model <= 0 || n_classes <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (static_cast<int>(host.ctc_w.size()) != n_classes * d_model ||
        static_cast<int>(host.ctc_b.size()) != n_classes)
    {
        std::fprintf(stderr,
                     "gigaam decoder (ctc): host head shape mismatch "
                     "(ctc_w=%zu expected=%d, ctc_b=%zu expected=%d)\n",
                     host.ctc_w.size(), n_classes * d_model,
                     host.ctc_b.size(), n_classes);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // logits[T_enc, n_classes] row-major.
    std::vector<float> logits(
        static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes), 0.0f);
    for (int t = 0; t < T_enc; ++t) {
        const float * frame = encoded + static_cast<size_t>(t) * d_model;
        float * row = logits.data() + static_cast<size_t>(t) * n_classes;
        matvec_add(host.ctc_w.data(), host.ctc_b.data(),
                   frame, n_classes, d_model, row);
    }

    // Dump raw logits before log_softmax (matches reference dump
    // `ctc.logits.raw`). Shape: [T_enc, n_classes] row-major.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32(
            "ctc.logits.raw", logits.data(),
            static_cast<long long>(logits.size()),
            shape, 2, "decoder.ctc.logits_raw");
    }

    // Per-frame log_softmax in place.
    for (int t = 0; t < T_enc; ++t) {
        float * row = logits.data() + static_cast<size_t>(t) * n_classes;
        float max_v = row[0];
        for (int c = 1; c < n_classes; ++c) {
            if (row[c] > max_v) max_v = row[c];
        }
        double sum = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            sum += std::exp(static_cast<double>(row[c] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int c = 0; c < n_classes; ++c) {
            row[c] -= log_sum;
        }
    }

    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32(
            "ctc.log_probs", logits.data(),
            static_cast<long long>(logits.size()),
            shape, 2, "decoder.ctc.log_probs");
    }

    // Greedy collapse: drop runs of the same label, then drop blanks.
    int prev_label = -1;
    for (int t = 0; t < T_enc; ++t) {
        const float * row = logits.data() + static_cast<size_t>(t) * n_classes;
        int   best = 0;
        float best_v = row[0];
        for (int c = 1; c < n_classes; ++c) {
            if (row[c] > best_v) { best_v = row[c]; best = c; }
        }
        if (best == prev_label) continue; // run-length collapse
        prev_label = best;
        if (best == blank_id) continue;
        out_tokens.push_back(best);
        out_frames.push_back(t);
    }

    return TRANSCRIBE_OK;
}

} // namespace transcribe::gigaam
