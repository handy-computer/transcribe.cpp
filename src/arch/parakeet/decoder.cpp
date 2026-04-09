// arch/parakeet/decoder.cpp - Parakeet TDT decoder implementation.
//
// See decoder.h for the API contract and the rationale for running
// the decoder on host instead of via a backend graph. The
// implementation is intentionally dependency-light: <vector>, <cmath>,
// <cstring> for the math; the only ggml include is for tensor reads
// at load time.
//
// Numerical-accuracy strategy: every step is a small enough operation
// (a few hundred dot products of length 640) that compounding error
// stays within ~1e-4 of the parakeet-mlx reference on CPU. The
// per-step dump points wired below feed scripts/compare_tensors.py
// for the bring-up loop and are gated on TRANSCRIBE_DUMP_DIR.

#include "decoder.h"

#include "parakeet.h"
#include "weights.h"

#include "transcribe-debug.h"

#include "ggml.h"
#include "ggml-backend.h"

#if TRANSCRIBE_HAS_BLAS
#  ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Host weight extraction
// ---------------------------------------------------------------------------

namespace {

// Read all elements of a ggml_tensor into a host fp32 vector,
// dequantizing as needed via ggml_get_type_traits()->to_float. The
// host decoder runs entirely on fp32 (linear / lstm_step / joint_step
// in this file are all hand-rolled fp32 matmuls), so quantized weight
// tensors are dequantized exactly once at load time and the per-decode
// hot path pays no readback or dequant cost.
//
// Source dtype support is whatever ggml's type traits register a
// to_float implementation for: F32, F16, BF16, all the Q* and IQ*
// quants. Tensors with no to_float fall through to the diagnostic
// (which would be a configuration mistake — the loader's per-slot
// allowlist in weights.cpp should never accept such a type).
//
// ggml_backend_tensor_get is the universal backend API: it's a memcpy
// on host buffers, a readback on discrete GPUs. We stage the raw
// (possibly quantized) bytes into a local buffer first, then walk the
// type's to_float to materialize fp32 in `out`.
//
// Returns false on missing-trait / size errors (logs a diagnostic
// naming the offending tensor).
bool read_tensor_to_f32(const ggml_tensor * t,
                        std::vector<float> & out)
{
    if (t == nullptr) {
        std::fprintf(stderr, "parakeet decoder: read_tensor_to_f32: null tensor\n");
        return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        std::fprintf(stderr,
                     "parakeet decoder: tensor \"%s\" has 0 bytes\n", t->name);
        return false;
    }
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        std::fprintf(stderr,
                     "parakeet decoder: tensor \"%s\" has nelem=%lld\n",
                     t->name, static_cast<long long>(nelem));
        return false;
    }

    out.resize(static_cast<size_t>(nelem));

    // F32 fast path. ggml's type_traits[GGML_TYPE_F32] entry does
    // NOT set a to_float (the identity case is left null upstream),
    // so we must short-circuit here rather than dispatch through it.
    // Read straight from the backend into the output buffer.
    if (t->type == GGML_TYPE_F32) {
        if (nbytes != static_cast<size_t>(nelem) * sizeof(float)) {
            std::fprintf(stderr,
                         "parakeet decoder: tensor \"%s\" f32 nbytes %zu "
                         "!= nelem*sizeof(float) %zu\n",
                         t->name, nbytes,
                         static_cast<size_t>(nelem) * sizeof(float));
            return false;
        }
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
        return true;
    }

    // Quantized / non-fp32 path. Stage the raw bytes off the backend,
    // then walk the type's to_float to materialize fp32. F16, BF16,
    // and every Q*/IQ* register a to_float in ggml.c's type_traits
    // table.
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        std::fprintf(stderr,
                     "parakeet decoder: tensor \"%s\" type %s has no to_float\n",
                     t->name, ggml_type_name(t->type));
        return false;
    }

    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data(), nelem);
    return true;
}

} // namespace

transcribe_status build_host_decoder_weights(const ParakeetModel & model,
                                             HostDecoderWeights &  out)
{
    const ParakeetHParams &  hp = model.hparams;
    const ParakeetWeights &  w  = model.weights;

    // ----- Predictor mirror -----
    out.predictor.pred_hidden = hp.pred_hidden;
    out.predictor.pred_vocab  = hp.pred_vocab;

    if (!read_tensor_to_f32(w.predictor.embed_w, out.predictor.embed_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    // Sanity-check the flattened size against the catalog shape. The
    // loader already validated ne against hparams; this is a cheap
    // belt-and-braces in case future surgery to either side gets out
    // of sync without updating the other.
    {
        const size_t expected =
            static_cast<size_t>(hp.pred_vocab) * static_cast<size_t>(hp.pred_hidden);
        if (out.predictor.embed_w.size() != expected) {
            std::fprintf(stderr,
                         "parakeet decoder: pred.embed.weight size %zu "
                         "!= pred_vocab*pred_hidden %zu\n",
                         out.predictor.embed_w.size(), expected);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    out.predictor.lstm.assign(hp.pred_n_layers, HostLstmLayer{});
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        const auto & lw = w.predictor.lstm[i];
        auto       & lh = out.predictor.lstm[i];
        if (!read_tensor_to_f32(lw.Wx, lh.Wx)) return TRANSCRIBE_ERR_GGUF;
        if (!read_tensor_to_f32(lw.Wh, lh.Wh)) return TRANSCRIBE_ERR_GGUF;
        if (!read_tensor_to_f32(lw.b,  lh.b))  return TRANSCRIBE_ERR_GGUF;

        const size_t gates    = static_cast<size_t>(4 * hp.pred_hidden);
        const size_t mat_size = gates * static_cast<size_t>(hp.pred_hidden);
        if (lh.Wx.size() != mat_size || lh.Wh.size() != mat_size) {
            std::fprintf(stderr,
                         "parakeet decoder: pred.lstm.%d Wx/Wh wrong size\n", i);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (lh.b.size() != gates) {
            std::fprintf(stderr,
                         "parakeet decoder: pred.lstm.%d bias wrong size\n", i);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ----- Joint mirror -----
    out.joint.d_enc       = hp.enc_d_model;
    out.joint.pred_hidden = hp.pred_hidden;
    out.joint.joint_h     = hp.joint_hidden;
    out.joint.joint_n     = hp.joint_n_classes();
    out.joint.activation  = hp.joint_activation;

    if (!read_tensor_to_f32(w.joint.enc_w,  out.joint.enc_w))  return TRANSCRIBE_ERR_GGUF;
    if (!read_tensor_to_f32(w.joint.enc_b,  out.joint.enc_b))  return TRANSCRIBE_ERR_GGUF;
    if (!read_tensor_to_f32(w.joint.pred_w, out.joint.pred_w)) return TRANSCRIBE_ERR_GGUF;
    if (!read_tensor_to_f32(w.joint.pred_b, out.joint.pred_b)) return TRANSCRIBE_ERR_GGUF;
    if (!read_tensor_to_f32(w.joint.out_w,  out.joint.out_w))  return TRANSCRIBE_ERR_GGUF;
    if (!read_tensor_to_f32(w.joint.out_b,  out.joint.out_b))  return TRANSCRIBE_ERR_GGUF;

    // ----- TDT params -----
    out.tdt_durations   = hp.tdt_durations;
    out.tdt_max_symbols = hp.tdt_max_symbols;
    out.n_vocab         = hp.pred_vocab - 1; // raw SP vocab size
    out.blank_id        = hp.pred_vocab - 1; // blank lives at vocab_size

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// LSTM state
// ---------------------------------------------------------------------------

void LstmState::reset(int n_layers, int pred_hidden) {
    h.resize(static_cast<size_t>(n_layers));
    c.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        h[static_cast<size_t>(layer)].assign(
            static_cast<size_t>(pred_hidden), 0.0f);
        c[static_cast<size_t>(layer)].assign(
            static_cast<size_t>(pred_hidden), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

namespace {

// y = W @ x + b   where W is row-major [out, in], x is [in], y is [out].
//
// "Row-major [out, in]" means W[r, c] = W_flat[r * in + c]. The result
// at row r is the dot product of row r with x. b is added in place;
// pass nullptr for no bias. y is overwritten (not accumulated).
//
// When BLAS is available (Accelerate on Apple, OpenBLAS/MKL/BLIS
// elsewhere), routes through cblas_sgemv for ~10-15x speedup over the
// naive loop. The BLAS reduction order differs from our naive row-wise
// accumulation, so joint logits may drift by ~1e-4 — argmax is robust
// to that. Fallback: scalar loop when no BLAS is linked.
inline void linear(const float * W,
                   const float * x,
                   const float * b,
                   int           out_dim,
                   int           in_dim,
                   float *       y)
{
#if TRANSCRIBE_HAS_BLAS
    // y = 1.0 * W @ x + 0.0 * y
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim,
                1.0f, W, in_dim, x, 1,
                0.0f, y, 1);
    if (b != nullptr) {
#ifdef __APPLE__
        catlas_saxpby(out_dim, 1.0f, b, 1, 1.0f, y, 1);
#else
        cblas_saxpy(out_dim, 1.0f, b, 1, y, 1);
#endif
    }
#else
    for (int r = 0; r < out_dim; ++r) {
        const float * row = W + static_cast<size_t>(r) * static_cast<size_t>(in_dim);
        float acc = 0.0f;
        for (int c = 0; c < in_dim; ++c) {
            acc += row[c] * x[c];
        }
        y[r] = acc + (b != nullptr ? b[r] : 0.0f);
    }
#endif
}

inline float sigmoidf(float x) {
    // The two-branch form keeps the exp argument away from +inf,
    // avoiding NaN propagation on extreme logits the model never
    // produces in practice.
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

// One LSTM time step. Reads `prev_h` and `prev_c` (each `H` floats),
// applies the input projection, hidden projection, gate equations,
// and writes new state into `new_h` / `new_c`. Caller must ensure
// `new_h` / `new_c` are pre-sized to H. Allocates an inline gate
// buffer of size 4*H on the stack via std::vector with a single
// reusable scratch passed in by the caller.
//
// Gate ordering: [i, f, g, o] (PyTorch / MLX standard, matches NeMo
// safetensors layout).
//
// Math (one bias term — NeMo's prednet stores a single concatenated
// bias rather than PyTorch's redundant W_ih + W_hh bias pair):
//
//     gates = Wx @ x + Wh @ h_prev + b
//     i, f, g, o = split(gates, 4, axis=0)
//     i = sigmoid(i)
//     f = sigmoid(f)
//     g = tanh(g)
//     o = sigmoid(o)
//     c_new = f * c_prev + i * g
//     h_new = o * tanh(c_new)
void lstm_step(const HostLstmLayer & layer,
               const float *         x,
               const float *         prev_h,
               const float *         prev_c,
               int                   H,
               std::vector<float> &  scratch_gates,
               std::vector<float> &  scratch_hh,
               float *               new_h,
               float *               new_c)
{
    const int four_H = 4 * H;
    if (static_cast<int>(scratch_gates.size()) < four_H) {
        scratch_gates.resize(static_cast<size_t>(four_H));
    }
    if (static_cast<int>(scratch_hh.size()) < four_H) {
        scratch_hh.resize(static_cast<size_t>(four_H));
    }

    // gates = Wx @ x + b
    linear(layer.Wx.data(), x, layer.b.data(), four_H, H, scratch_gates.data());
    // gates += Wh @ prev_h
    linear(layer.Wh.data(), prev_h, nullptr, four_H, H, scratch_hh.data());
    for (int i = 0; i < four_H; ++i) {
        scratch_gates[i] += scratch_hh[i];
    }

    const float * gi = scratch_gates.data() + 0 * H;
    const float * gf = scratch_gates.data() + 1 * H;
    const float * gg = scratch_gates.data() + 2 * H;
    const float * go = scratch_gates.data() + 3 * H;

    for (int j = 0; j < H; ++j) {
        const float i_t = sigmoidf(gi[j]);
        const float f_t = sigmoidf(gf[j]);
        const float g_t = std::tanh(gg[j]);
        const float o_t = sigmoidf(go[j]);
        const float c_new = f_t * prev_c[j] + i_t * g_t;
        new_c[j] = c_new;
        new_h[j] = o_t * std::tanh(c_new);
    }
}

// Run the predictor for one decode step.
//
// `last_token` is the token id from the previous decode step, or -1
// for the start state. The embed lookup goes through the [pred_vocab,
// pred_hidden] table; the start state matches parakeet-mlx's
// `embedded_y = mx.zeros(...)` branch in PredictNetwork.__call__.
//
// Reads from `prev_state`, writes new state into `new_state`. The
// caller arranges `new_state` to be the only state mutated, so blank
// emissions can simply discard `new_state` instead of restoring a
// snapshot.
//
// Returns a pointer (borrowed) into `new_state.h.back()` — the last
// LSTM layer's new hidden state — which is the predictor "decoder
// output" the joint network will consume.
const float * predictor_step(const HostPredictor & predictor,
                             int                   last_token,
                             const LstmState &     prev_state,
                             LstmState &           new_state,
                             std::vector<float> &  scratch_x,
                             std::vector<float> &  scratch_gates,
                             std::vector<float> &  scratch_hh)
{
    const int H = predictor.pred_hidden;
    if (static_cast<int>(scratch_x.size()) < H) {
        scratch_x.resize(static_cast<size_t>(H));
    }

    // Embed lookup (or start-state zeros).
    if (last_token < 0) {
        std::fill_n(scratch_x.data(), H, 0.0f);
    } else {
        const size_t row_off =
            static_cast<size_t>(last_token) * static_cast<size_t>(H);
        std::memcpy(scratch_x.data(),
                    predictor.embed_w.data() + row_off,
                    static_cast<size_t>(H) * sizeof(float));
    }

    // Layer 0 takes the embed; subsequent layers take the previous
    // layer's new hidden state. The LSTM step writes new state in
    // place into `new_state.h[layer]` and `new_state.c[layer]`.
    const float * layer_input = scratch_x.data();
    for (size_t layer = 0; layer < predictor.lstm.size(); ++layer) {
        lstm_step(predictor.lstm[layer],
                  layer_input,
                  prev_state.h[layer].data(),
                  prev_state.c[layer].data(),
                  H,
                  scratch_gates,
                  scratch_hh,
                  new_state.h[layer].data(),
                  new_state.c[layer].data());
        layer_input = new_state.h[layer].data();
    }

    return new_state.h.back().data();
}

// Run the joint network for one decode step.
//
// Inputs are one encoder frame (`d_enc` floats) and one predictor
// hidden state (`pred_hidden` floats). Output is the full
// `joint_n` logits vector. Both projections share the joint_h
// hidden width; they're added then passed through the configured
// activation, then a final linear collapses to joint_n.
//
//   enc_proj  = enc_w  @ enc_frame  + enc_b      [joint_h]
//   pred_proj = pred_w @ pred_state + pred_b     [joint_h]
//   summed    = enc_proj + pred_proj             [joint_h]
//   activated = activation(summed)               [joint_h]
//   logits    = out_w @ activated   + out_b      [joint_n]
//
// Activation is one of {relu, sigmoid, tanh}; the loader's
// allow-list guarantees we'll see exactly one of those at run
// time. Parakeet 0.6B v2/v3 ship "relu".
void joint_step(const HostJoint &     j,
                const float *         enc_frame,
                const float *         pred_state,
                std::vector<float> &  scratch_enc_proj,
                std::vector<float> &  scratch_pred_proj,
                std::vector<float> &  scratch_summed,
                std::vector<float> &  out_logits)
{
    if (static_cast<int>(scratch_enc_proj.size())  < j.joint_h) scratch_enc_proj.resize(static_cast<size_t>(j.joint_h));
    if (static_cast<int>(scratch_pred_proj.size()) < j.joint_h) scratch_pred_proj.resize(static_cast<size_t>(j.joint_h));
    if (static_cast<int>(scratch_summed.size())    < j.joint_h) scratch_summed.resize(static_cast<size_t>(j.joint_h));
    if (static_cast<int>(out_logits.size())        < j.joint_n) out_logits.resize(static_cast<size_t>(j.joint_n));

    linear(j.enc_w.data(),  enc_frame,  j.enc_b.data(),  j.joint_h, j.d_enc,
           scratch_enc_proj.data());
    linear(j.pred_w.data(), pred_state, j.pred_b.data(), j.joint_h, j.pred_hidden,
           scratch_pred_proj.data());

    for (int i = 0; i < j.joint_h; ++i) {
        scratch_summed[i] = scratch_enc_proj[i] + scratch_pred_proj[i];
    }

    if (j.activation == "relu") {
        for (int i = 0; i < j.joint_h; ++i) {
            if (scratch_summed[i] < 0.0f) scratch_summed[i] = 0.0f;
        }
    } else if (j.activation == "sigmoid") {
        for (int i = 0; i < j.joint_h; ++i) {
            scratch_summed[i] = sigmoidf(scratch_summed[i]);
        }
    } else { // "tanh" — loader's allow-list ensures this is the only remaining case.
        for (int i = 0; i < j.joint_h; ++i) {
            scratch_summed[i] = std::tanh(scratch_summed[i]);
        }
    }

    linear(j.out_w.data(), scratch_summed.data(), j.out_b.data(),
           j.joint_n, j.joint_h, out_logits.data());
}

// Argmax over a contiguous fp32 range. Returns the index of the
// largest value; ties go to the first occurrence (matches numpy /
// MLX argmax convention). The decoder uses this for both the token
// and duration argmaxes.
int argmax_range(const float * data, int n) {
    int   best_i = 0;
    float best_v = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

// Entropy-based confidence over a token-logit slice. Mirrors
// parakeet-mlx's ParakeetTDT.decode_greedy:
//
//     token_probs = softmax(token_logits)
//     entropy     = -sum(p * log(p + 1e-10))
//     max_entropy = log(vocab_size + 1)
//     confidence  = 1 - entropy / max_entropy
//
// In our terms `vocab_size + 1 == pred_vocab == n_token_classes`.
// The result lives in [0, 1] modulo the +1e-10 epsilon (which
// matches parakeet-mlx's reference verbatim, including its slight
// negative-bias for nearly-uniform distributions).
float token_confidence(const float * token_logits,
                       int           n_token_classes,
                       std::vector<float> & scratch_probs) {
    // Numerically stable softmax: subtract max before exp.
    float max_logit = token_logits[0];
    for (int i = 1; i < n_token_classes; ++i) {
        if (token_logits[i] > max_logit) max_logit = token_logits[i];
    }
    if (static_cast<int>(scratch_probs.size()) < n_token_classes) {
        scratch_probs.resize(static_cast<size_t>(n_token_classes));
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float e = std::exp(token_logits[i] - max_logit);
        scratch_probs[static_cast<size_t>(i)] = e;
        sum_exp += static_cast<double>(e);
    }
    const float inv_sum = static_cast<float>(1.0 / sum_exp);
    double entropy = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float p = scratch_probs[static_cast<size_t>(i)] * inv_sum;
        // +1e-10 mirrors parakeet-mlx exactly.
        entropy -= static_cast<double>(p) *
                   std::log(static_cast<double>(p) + 1e-10);
    }
    const double max_entropy = std::log(static_cast<double>(n_token_classes));
    if (max_entropy <= 0.0) {
        return 1.0f;
    }
    return static_cast<float>(1.0 - entropy / max_entropy);
}

} // namespace

// ---------------------------------------------------------------------------
// Greedy decode driver
// ---------------------------------------------------------------------------

transcribe_status decode_tdt_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    std::vector<TdtToken> &    out_tokens)
{
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        std::fprintf(stderr,
                     "parakeet decoder: enc d_model mismatch (got %d, "
                     "expected %d)\n", d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int n_layers     = static_cast<int>(w.predictor.lstm.size());
    const int H            = w.predictor.pred_hidden;
    const int n_token_cls  = w.predictor.pred_vocab; // == vocab_size + 1
    const int n_dur        = static_cast<int>(w.tdt_durations.size());
    const int blank_id     = w.blank_id;

    // Two LSTM states, both pre-sized: `state` is the committed
    // state we read from each iteration; `next_state` is where the
    // predictor writes the new step's outputs. On a non-blank
    // emission we swap; on a blank emission we just discard the
    // contents of `next_state` (overwritten on the next iteration).
    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Per-call scratch reused across every decode step.
    std::vector<float> scratch_x;
    std::vector<float> scratch_gates;
    std::vector<float> scratch_hh;
    std::vector<float> scratch_enc_proj;
    std::vector<float> scratch_pred_proj;
    std::vector<float> scratch_summed;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = -1; // sentinel: no previous token (start state)
    int step        = 0;
    int new_symbols = 0;

    // Honest cap: a 30-second clip emits a few hundred tokens at
    // most. Bound the loop iterations as a runaway-protection net
    // in case the joint produces pathological logits we haven't
    // anticipated. The cap is generous; legitimate transcriptions
    // never approach it.
    const int max_iters = 16 * T_enc + 1024;
    int iter = 0;
    int64_t t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        // ----- Predictor (one LSTM step) -----
        const int64_t t0 = ggml_time_us();
        const float * decoder_out = predictor_step(
            w.predictor, last_token, state, next_state,
            scratch_x, scratch_gates, scratch_hh);
        const int64_t t1 = ggml_time_us();

        // ----- Joint -----
        const float * enc_frame =
            enc_out + static_cast<size_t>(step) * static_cast<size_t>(d_enc);
        joint_step(w.joint, enc_frame, decoder_out,
                   scratch_enc_proj, scratch_pred_proj, scratch_summed,
                   logits);
        const int64_t t2 = ggml_time_us();
        t_pred_us  += t1 - t0;
        t_joint_us += t2 - t1;

        // ----- Argmax (token + duration) -----
        const float * token_logits    = logits.data();
        const float * duration_logits = logits.data() + n_token_cls;

        const int pred_token = argmax_range(token_logits, n_token_cls);
        const int decision   = argmax_range(duration_logits, n_dur);
        const int duration   = w.tdt_durations[static_cast<size_t>(decision)];

        // ----- Optional dump (first iteration only, for bring-up) -----
        //
        // Dumping every iteration would flood the dump dir on a long
        // clip; the bring-up only needs a few well-chosen sample
        // points to verify each component matches parakeet-mlx. The
        // first decode step (start state, encoder frame 0) gives us
        // four reference points: predictor input (the embed lookup,
        // here all zeros), per-layer LSTM h, the joint logits, the
        // argmax decision. The Python `dump_reference.py decode`
        // subcommand mirrors these.
        if (iter == 1 && transcribe::debug::enabled()) {
            // Predictor scratch_x at start: zeros (vector of length H).
            const long long s_h = H;
            transcribe::debug::dump_host_f32(
                "dec.embed.0", scratch_x.data(), s_h, &s_h, 1,
                "decoder.embed");

            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(
                    name_h, next_state.h[layer].data(), s_h, &s_h, 1,
                    "decoder.lstm");
                transcribe::debug::dump_host_f32(
                    name_c, next_state.c[layer].data(), s_h, &s_h, 1,
                    "decoder.lstm");
            }

            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32(
                "dec.joint.0", logits.data(), s_n, &s_n, 1,
                "decoder.joint");
        }

        // ----- TDT emit + state advance -----
        const bool is_blank = (pred_token == blank_id);
        if (!is_blank) {
            const int64_t tc0 = ggml_time_us();
            const float p = token_confidence(token_logits, n_token_cls,
                                             scratch_probs);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = duration;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state); // commit
        }

        // Step / stuck advance. Mirrors parakeet-mlx exactly:
        //   step += duration
        //   new_symbols += 1
        //   if duration != 0: new_symbols = 0
        //   elif max_symbols and new_symbols >= max_symbols:
        //       step += 1; new_symbols = 0
        step += duration;
        new_symbols += 1;
        if (duration != 0) {
            new_symbols = 0;
        } else if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
            step += 1;
            new_symbols = 0;
        } else if (is_blank && w.tdt_max_symbols > 0) {
            // A blank with duration 0 leaves (step, last_token, state)
            // unchanged, so every following iteration is identical until
            // max_symbols forces the step to advance. Fast-forward those
            // repeated no-op loops without changing the observable result.
            const int skip = w.tdt_max_symbols - new_symbols;
            if (skip > 0 && iter + skip < max_iters) {
                iter += skip;
                step += 1;
                new_symbols = 0;
            }
        }
    }

    std::fprintf(stderr,
                 "decoder: %d iters, %d tokens, T_enc=%d  "
                 "pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
                 "total=%.1f ms  per_iter=%.0f us\n",
                 iter, static_cast<int>(out_tokens.size()), T_enc,
                 t_pred_us  / 1000.0,
                 t_joint_us / 1000.0,
                 t_conf_us  / 1000.0,
                 (t_pred_us + t_joint_us + t_conf_us) / 1000.0,
                 static_cast<double>(t_pred_us + t_joint_us) / std::max(iter, 1));

    if (iter >= max_iters) {
        std::fprintf(stderr,
                     "parakeet decoder: hit iteration cap (%d) — pathological "
                     "logits or loop bug\n", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

} // namespace transcribe::parakeet
