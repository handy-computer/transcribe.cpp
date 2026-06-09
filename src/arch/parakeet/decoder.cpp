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
// stays within ~1e-4 of the reference on CPU. The
// per-step dump points wired below feed scripts/compare_tensors.py
// for the bring-up loop and are gated on TRANSCRIBE_DUMP_DIR.

#include "decoder.h"

#include "parakeet.h"
#include "weights.h"

#include "transcribe-debug.h"
#include "transcribe-log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

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
#include <cstdlib>
#include <cstring>
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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: read_tensor_to_f32: null tensor");
        return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder: tensor \"%s\" has 0 bytes", t->name);
        return false;
    }
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder: tensor \"%s\" has nelem=%lld",
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet decoder: tensor \"%s\" f32 nbytes %zu "
                         "!= nelem*sizeof(float) %zu",
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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder: tensor \"%s\" type %s has no to_float",
                     t->name, ggml_type_name(t->type));
        return false;
    }

    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data(), nelem);
    return true;
}

// Make the joint output-projection weight + bias resident as ggml tensors on
// the model (the immutable half). Built once at load; only ever read after
// that, so it is safe to share across every context. The per-step compute
// graph that consumes this is built per decode call (build_joint_graph below).
// On any failure it frees its partial state and returns false, so the caller
// can retry (native→fp32) or fall back to the host matmul (w_ready stays false).
//
// `use_native` selects the weight dtype:
//   - true  (default): keep the source quant dtype. ggml streams the quantized
//     bytes (~4× less weight bandwidth) and quantizes the activation on the fly
//     — the same regime the encoder runs in, WER-validated equal to fp32 on
//     English. This is the faster path on bandwidth-bound CPUs/iGPUs.
//   - false (fallback): dequantize the weight to fp32 once, so ggml accumulates
//     fp32×fp32 with no activation down-conversion. Numerically faithful to the
//     host reference (max rel logit diff ~3e-7 vs ~3e-3 for a Q4 native weight).
// The caller tries native first; TRANSCRIBE_JOINT_FP32=1 forces fp32.
bool build_joint_weight(HostJoint & j, const ggml_tensor * src_out_w,
                        bool use_native) {
    const int joint_h = j.joint_h;
    const int joint_n = j.joint_n;

    // Free any partial state so the caller can cleanly retry (e.g. native→fp32).
    auto fail = [&]() -> bool {
        if (j.w_buf     != nullptr) { ggml_backend_buffer_free(j.w_buf); j.w_buf = nullptr; }
        if (j.w_ctx     != nullptr) { ggml_free(j.w_ctx);                j.w_ctx = nullptr; }
        if (j.w_backend != nullptr) { ggml_backend_free(j.w_backend);    j.w_backend = nullptr; }
        j.gw_w = nullptr; j.gw_b = nullptr; j.w_ready = false;
        return false;
    };

    const ggml_type w_type = use_native ? src_out_w->type : GGML_TYPE_F32;

    // A CPU backend used ONLY to allocate the resident weight buffer — never
    // graph_compute'd (each decode call owns its own compute backend), so it
    // carries no per-step state and is safe to keep on the shared model.
    j.w_backend = ggml_backend_cpu_init();
    if (j.w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed");
        return fail();
    }

    ggml_init_params ip {};
    ip.mem_size   = ggml_tensor_overhead() * 4;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    j.w_ctx = ggml_init(ip);
    if (j.w_ctx == nullptr) return fail();

    j.gw_w = ggml_new_tensor_2d(j.w_ctx, w_type, joint_h, joint_n);
    j.gw_b = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_n);

    j.w_buf = ggml_backend_alloc_ctx_tensors(j.w_ctx, j.w_backend);
    if (j.w_buf == nullptr) return fail();

    // Upload the weight (verbatim native dtype, or dequantized to fp32 under
    // TRANSCRIBE_JOINT_FP32) and the fp32 bias.
    if (w_type == src_out_w->type) {
        const size_t nb = ggml_nbytes(src_out_w);
        std::vector<uint8_t> tmp(nb);
        ggml_backend_tensor_get(src_out_w, tmp.data(), 0, nb);
        ggml_backend_tensor_set(j.gw_w, tmp.data(), 0, nb);
    } else {
        // fp32 graph weight from a non-fp32 source: dequantize once into fp32.
        std::vector<float> tmp;
        if (!read_tensor_to_f32(src_out_w, tmp)) return fail();
        ggml_backend_tensor_set(j.gw_w, tmp.data(), 0,
                                tmp.size() * sizeof(float));
    }
    ggml_backend_tensor_set(j.gw_b, j.out_b.data(), 0,
                            static_cast<size_t>(joint_n) * sizeof(float));

    j.w_ready = true;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Per-call joint compute graph
// ---------------------------------------------------------------------------
//
// The MUTABLE half of the joint output projection. Built fresh on the stack in
// each decode call (cheap: it allocates only the [joint_h] activation input,
// the matmul node, and the [joint_n] logits output — the weight is the shared
// resident HostJoint::gw_w/gw_b, not re-uploaded), and torn down at function
// exit. Because every concurrent decode owns its own JointGraph (its own
// compute backend + I/O tensors), decode on contexts sharing one model is
// reentrant — nothing per-step is shared.
namespace {

struct JointGraph {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         act     = nullptr; // [joint_h] fp32 input
    ggml_tensor *         logits  = nullptr; // [joint_n] fp32 output
    bool                  ready   = false;

    JointGraph() = default;
    ~JointGraph() {
        if (buf     != nullptr) ggml_backend_buffer_free(buf);
        if (ctx     != nullptr) ggml_free(ctx);
        if (backend != nullptr) ggml_backend_free(backend);
    }
    JointGraph(const JointGraph &)             = delete;
    JointGraph & operator=(const JointGraph &) = delete;
};

// Build a per-call compute graph around the model-resident weight j.gw_w/gw_b.
// Returns false (and leaves g.ready == false) if the weight is absent or any
// ggml step fails; the caller then uses the host matmul fallback. n_threads is
// the resolved (>0) decode thread count.
bool build_joint_graph(JointGraph & g, const HostJoint & j, int n_threads) {
    if (!j.w_ready || j.gw_w == nullptr || j.gw_b == nullptr) return false;

    auto fail = [&]() -> bool {
        if (g.buf     != nullptr) { ggml_backend_buffer_free(g.buf); g.buf = nullptr; }
        if (g.ctx     != nullptr) { ggml_free(g.ctx);                g.ctx = nullptr; }
        if (g.backend != nullptr) { ggml_backend_free(g.backend);    g.backend = nullptr; }
        g.act = nullptr; g.logits = nullptr; g.graph = nullptr; g.ready = false;
        return false;
    };

    g.backend = ggml_backend_cpu_init();
    if (g.backend == nullptr) return fail();
    ggml_backend_cpu_set_n_threads(g.backend, std::max(1, n_threads));

    ggml_init_params ip {};
    ip.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx = ggml_init(ip);
    if (g.ctx == nullptr) return fail();

    g.act = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.joint_h);
    ggml_set_input(g.act);
    // References the shared resident weight (in j.w_ctx) — ggml stores the
    // pointer; the CPU backend reads its bytes directly at compute time.
    ggml_tensor * mm = ggml_mul_mat(g.ctx, j.gw_w, g.act); // [joint_n, 1]
    g.logits = ggml_add(g.ctx, mm, j.gw_b);                // [joint_n, 1]
    ggml_set_output(g.logits);

    // Allocates only the tensors created in g.ctx (act + op results); the
    // shared weight/bias live in another ctx and are already resident.
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) return fail();

    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    g.ready = true;
    return true;
}

} // namespace

HostJoint::~HostJoint() {
    if (w_buf     != nullptr) ggml_backend_buffer_free(w_buf);
    if (w_ctx     != nullptr) ggml_free(w_ctx);
    if (w_backend != nullptr) ggml_backend_free(w_backend);
}

// Resolve a decode thread count: n_threads <= 0 means "auto" → min(8, cores),
// matching the encoder default. Threaded through to the joint compute backend
// and linear()'s OpenMP loop per decode call; no process-global state is set,
// so concurrent decodes on one model do not stomp each other.
static int resolve_decode_threads(int n_threads) {
    return n_threads > 0
        ? n_threads
        : std::min(8, std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
}

transcribe_status build_host_decoder_weights(const ParakeetModel & model,
                                             HostDecoderWeights &  out)
{
    const ParakeetHParams &  hp = model.hparams;
    const ParakeetWeights &  w  = model.weights;

    // Head-kind dispatch. CTC skips predictor/joint entirely; RNNT and
    // TDT share the predictor + joint mirror code below.
    switch (hp.head_kind) {
        case HeadKind::TDT:  out.head_kind = HostHeadKind::TDT;  break;
        case HeadKind::RNNT: out.head_kind = HostHeadKind::RNNT; break;
        case HeadKind::CTC:  out.head_kind = HostHeadKind::CTC;  break;
    }

    if (hp.head_kind == HeadKind::CTC) {
        // ----- CTC head mirror -----
        //
        // Source weight has ggml ne [1, d_model, n_classes] (PyTorch
        // [n_classes, d_model, 1] flat). The bytes are already laid out
        // exactly the way the host decoder wants to consume them
        // (row-major [n_classes, d_model]); read_tensor_to_f32 just
        // copies them into a flat fp32 vector. Bias is [n_classes].
        out.ctc_head.n_classes = hp.head_ctc_n_classes;
        out.ctc_head.blank_id  = hp.head_ctc_n_classes - 1; // NeMo convention
        out.ctc_head.d_enc     = hp.enc_d_model;

        if (!read_tensor_to_f32(w.ctc_head.weight, out.ctc_head.weight)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(w.ctc_head.bias, out.ctc_head.bias)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t expected_w =
            static_cast<size_t>(out.ctc_head.n_classes) *
            static_cast<size_t>(out.ctc_head.d_enc);
        if (out.ctc_head.weight.size() != expected_w) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet decoder: head.ctc.weight size %zu "
                         "!= n_classes*d_enc %zu",
                         out.ctc_head.weight.size(), expected_w);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (static_cast<int>(out.ctc_head.bias.size()) != out.ctc_head.n_classes) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet decoder: head.ctc.bias size %zu "
                         "!= n_classes %d",
                         out.ctc_head.bias.size(), out.ctc_head.n_classes);
            return TRANSCRIBE_ERR_GGUF;
        }

        out.tdt_durations.clear();
        out.tdt_max_symbols = 0;
        out.blank_id = out.ctc_head.blank_id;
        out.n_vocab  = out.ctc_head.n_classes - 1;
        return TRANSCRIBE_OK;
    }

    // ----- Predictor mirror (TDT, RNNT) -----
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet decoder: pred.embed.weight size %zu "
                         "!= pred_vocab*pred_hidden %zu",
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet decoder: pred.lstm.%d Wx/Wh wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (lh.b.size() != gates) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet decoder: pred.lstm.%d bias wrong size", i);
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

    // Make the dominant out_w projection weight resident as a ggml tensor.
    // Default to the native-quant weight (faster, WER-validated); fall back to
    // the fp32-dequant weight on failure, then to the host matmul (w_ready stays
    // false) if even that fails. TRANSCRIBE_JOINT_FP32=1 forces the fp32 path.
    if (w.joint.out_w != nullptr) {
        const int ne0 = static_cast<int>(w.joint.out_w->ne[0]);
        const int ne1 = static_cast<int>(w.joint.out_w->ne[1]);
        if (ne0 != out.joint.joint_h || ne1 != out.joint.joint_n) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: out_w ne [%d,%d] != [joint_h=%d, joint_n=%d]; "
                "using host matmul",
                ne0, ne1, out.joint.joint_h, out.joint.joint_n);
        } else {
            const bool force_fp32 = std::getenv("TRANSCRIBE_JOINT_FP32") != nullptr;
            bool ok = false;
            if (!force_fp32) {
                ok = build_joint_weight(out.joint, w.joint.out_w, /*use_native=*/true);
                if (!ok) {
                    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                        "parakeet decoder: native joint weight failed; retrying fp32");
                }
            }
            if (!ok) {
                ok = build_joint_weight(out.joint, w.joint.out_w, /*use_native=*/false);
            }
            if (!ok) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: joint ggml weight build failed; using host matmul");
            }
        }
    }

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
                   float *       y,
                   int           n_threads)
{
#if TRANSCRIBE_HAS_BLAS
    (void) n_threads; // BLAS manages its own threads
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
    // Rows are independent, so larger projections parallelize cleanly across
    // cores. Gate on out_dim (>= 2048) to thread the LSTM gate matmuls
    // (4*640 = 2560) and the CTC head, while the tiny 640-wide enc/pred
    // projections stay serial and don't pay OpenMP fork/join overhead. The
    // thread count is passed in per decode (resolved min(8, cores) default) via
    // a num_threads clause — no process-global omp_set_num_threads, so
    // concurrent decodes don't stomp each other. The joint out_w projection
    // runs in its own ggml graph, not here.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(out_dim >= 2048) \
        num_threads(n_threads > 0 ? n_threads : 1)
#endif
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
// Gate ordering: [i, f, g, o] (PyTorch standard, matches NeMo
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
               float *               new_c,
               int                   n_threads)
{
    const int four_H = 4 * H;
    if (static_cast<int>(scratch_gates.size()) < four_H) {
        scratch_gates.resize(static_cast<size_t>(four_H));
    }
    if (static_cast<int>(scratch_hh.size()) < four_H) {
        scratch_hh.resize(static_cast<size_t>(four_H));
    }

    // gates = Wx @ x + b
    linear(layer.Wx.data(), x, layer.b.data(), four_H, H, scratch_gates.data(), n_threads);
    // gates += Wh @ prev_h
    linear(layer.Wh.data(), prev_h, nullptr, four_H, H, scratch_hh.data(), n_threads);
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
// pred_hidden] table; the start state is an all-zeros embedding
// (matching PredictNetwork's "no previous token" branch).
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
                             std::vector<float> &  scratch_hh,
                             int                   n_threads)
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
                  new_state.c[layer].data(),
                  n_threads);
        layer_input = new_state.h[layer].data();
    }

    return new_state.h.back().data();
}

// Run the joint network for one decode step.
//
// `enc_proj` is the precomputed encoder projection for this frame
// (enc_w @ enc_frame + enc_b, `joint_h` floats). The caller batches
// all T_enc projections via sgemm before the decode loop so they
// are computed once rather than redundantly per iteration.
//
//   pred_proj = pred_w @ pred_state + pred_b     [joint_h]
//   summed    = enc_proj + pred_proj             [joint_h]
//   activated = activation(summed)               [joint_h]
//   logits    = out_w @ activated   + out_b      [joint_n]
//
// Activation is one of {relu, sigmoid, tanh}; the loader's
// allow-list guarantees we'll see exactly one of those at run
// time. Parakeet 0.6B v2/v3 ship "relu".
void joint_step(const HostJoint &     j,
                const JointGraph &    g,
                int                   n_threads,
                const float *         enc_proj,
                const float *         pred_state,
                std::vector<float> &  scratch_pred_proj,
                std::vector<float> &  scratch_summed,
                std::vector<float> &  out_logits)
{
    if (static_cast<int>(scratch_pred_proj.size()) < j.joint_h) scratch_pred_proj.resize(static_cast<size_t>(j.joint_h));
    if (static_cast<int>(scratch_summed.size())    < j.joint_h) scratch_summed.resize(static_cast<size_t>(j.joint_h));
    if (static_cast<int>(out_logits.size())        < j.joint_n) out_logits.resize(static_cast<size_t>(j.joint_n));

    linear(j.pred_w.data(), pred_state, j.pred_b.data(), j.joint_h, j.pred_hidden,
           scratch_pred_proj.data(), n_threads);

    for (int i = 0; i < j.joint_h; ++i) {
        scratch_summed[i] = enc_proj[i] + scratch_pred_proj[i];
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

    // Output projection: per-call ggml graph (native/fp32 weight, threaded
    // matmul, bias folded into the graph) when available, else the host matmul.
    if (g.ready) {
        ggml_backend_tensor_set(g.act, scratch_summed.data(), 0,
                                static_cast<size_t>(j.joint_h) * sizeof(float));
        ggml_backend_graph_compute(g.backend, g.graph);
        ggml_backend_tensor_get(g.logits, out_logits.data(), 0,
                                static_cast<size_t>(j.joint_n) * sizeof(float));
    } else {
        linear(j.out_w.data(), scratch_summed.data(), j.out_b.data(),
               j.joint_n, j.joint_h, out_logits.data(), n_threads);
    }

    // Apply log_softmax over the full joint output (vocab+blank+durations
    // for TDT, vocab+blank for RNNT) to match NeMo's CPU-inference
    // representation of `joint_after_projection`. NeMo's RNNTJoint applies
    // `log_softmax(dim=-1)` when running on CPU and `self.log_softmax`
    // is None (the default), or when `self.log_softmax=True`. Applying
    // it here:
    //   - leaves token-argmax and duration-argmax invariant (subtracting
    //     a constant from all elements of either sub-range preserves
    //     argmax),
    //   - leaves token_confidence invariant (it re-softmaxes the token
    //     sub-range, which absorbs any uniform shift),
    //   - lets `dec.joint.0` dump compare element-wise against NeMo's
    //     reference dump (representation match — without this, our raw
    //     logits sit ~+25 to +35 above NeMo's log-softmax-normalized
    //     values, and the only flat tolerance that fits is 100/100,
    //     which is loose enough to hide real numerical divergence).
    //
    // Cost: ~joint_n adds + 1 logsumexp. For joint_n=1030 that's ~2K
    // ops per decode iter, negligible vs the joint_h × pred_hidden +
    // joint_h × joint_n matmuls (~1.3M ops).
    {
        float max_v = out_logits[0];
        for (int i = 1; i < j.joint_n; ++i) {
            if (out_logits[i] > max_v) max_v = out_logits[i];
        }
        double sum = 0.0;
        for (int i = 0; i < j.joint_n; ++i) {
            sum += std::exp(static_cast<double>(out_logits[i] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int i = 0; i < j.joint_n; ++i) {
            out_logits[i] -= log_sum;
        }
    }
}

// Precompute the joint encoder projection for `T` frames into `out`
// (row-major [T, joint_h]): out[t] = enc_w @ enc_out[t] + enc_b. The
// projection is decode-state-independent, so batching it into one sgemm (or T
// sgemv calls without BLAS) before the greedy loop eliminates redundant
// per-iteration work and gives BLAS a large matrix. Shared by the TDT, RNN-T,
// and streaming RNN-T drivers.
void precompute_enc_proj(const HostJoint &    j,
                         const float *        enc_out,
                         int                  T,
                         int                  d_enc,
                         std::vector<float> & out,
                         int                  n_threads)
{
    const int joint_h = j.joint_h;
    out.resize(static_cast<size_t>(T) * static_cast<size_t>(joint_h));
#if TRANSCRIBE_HAS_BLAS
    (void) n_threads;
    // C[T, joint_h] = enc_out[T, d_enc] @ enc_w^T[d_enc, joint_h]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, joint_h, d_enc,
                1.0f, enc_out, d_enc,
                j.enc_w.data(), d_enc,
                0.0f, out.data(), joint_h);
#else
    for (int t = 0; t < T; ++t) {
        const float * frame = enc_out + static_cast<size_t>(t) * static_cast<size_t>(d_enc);
        float * proj = out.data() + static_cast<size_t>(t) * static_cast<size_t>(joint_h);
        linear(j.enc_w.data(), frame, nullptr, joint_h, d_enc, proj, n_threads);
    }
#endif
    // Add bias to every row.
    for (int t = 0; t < T; ++t) {
        float * proj = out.data() + static_cast<size_t>(t) * static_cast<size_t>(joint_h);
        for (int k = 0; k < joint_h; ++k) {
            proj[k] += j.enc_b[static_cast<size_t>(k)];
        }
    }
}

// Argmax over a contiguous fp32 range. Returns the index of the
// largest value; ties go to the first occurrence (matches the numpy
// argmax convention). The decoder uses this for both the token and
// duration argmaxes.
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

// Entropy-based confidence over a token-logit slice. Mirrors the
// reference ParakeetTDT.decode_greedy path:
//
//     token_probs = softmax(token_logits)
//     entropy     = -sum(p * log(p + 1e-10))
//     max_entropy = log(vocab_size + 1)
//     confidence  = 1 - entropy / max_entropy
//
// In our terms `vocab_size + 1 == pred_vocab == n_token_classes`.
// The result lives in [0, 1] modulo the +1e-10 epsilon (which
// matches the reference verbatim, including its slight
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
        // +1e-10 matches the reference.
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
                                    int                        n_threads,
                                    std::vector<TdtToken> &    out_tokens)
{
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder: enc d_model mismatch (got %d, "
                     "expected %d)", d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt           = resolve_decode_threads(n_threads);
    const int n_layers     = static_cast<int>(w.predictor.lstm.size());
    const int H            = w.predictor.pred_hidden;
    const int n_token_cls  = w.predictor.pred_vocab; // == vocab_size + 1
    const int n_dur        = static_cast<int>(w.tdt_durations.size());
    const int blank_id     = w.blank_id;

    // Per-call joint compute graph around the shared resident weight. Owns its
    // own backend + I/O tensors, so concurrent decodes don't share mutable
    // state. Falls back to the host matmul if the graph isn't available.
    JointGraph jg;
    build_joint_graph(jg, w.joint, nt);

    // Two LSTM states, both pre-sized: `state` is the committed
    // state we read from each iteration; `next_state` is where the
    // predictor writes the new step's outputs. On a non-blank
    // emission we swap; on a blank emission we just discard the
    // contents of `next_state` (overwritten on the next iteration).
    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (decode-state
    // independent; one sgemm before the loop). See precompute_enc_proj.
    const int joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t t_enc_proj_start = ggml_time_us();
    precompute_enc_proj(w.joint, enc_out, T_enc, d_enc, enc_proj_all, nt);
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    // Per-call scratch reused across every decode step.
    std::vector<float> scratch_x;
    std::vector<float> scratch_gates;
    std::vector<float> scratch_hh;
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

    // On a blank emission `(last_token, state)` are preserved, so the
    // predictor would deterministically write the same `next_state` on the
    // following iteration. Skip the LSTM unroll and reuse the previous
    // `decoder_out`; recompute on a non-blank emission, where
    // `state.swap(next_state)` and `last_token` change. Bit-exact with the
    // unconditional path. Complements the `duration==0` fast-forward
    // below: that fast-forward only fires when `tdt_max_symbols > 0` and
    // duration is zero. Blanks with `duration > 0` advance `step` but
    // still leave `(last_token, state)` unchanged, and this cache catches
    // them.
    bool predictor_dirty = true;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        // ----- Predictor (one LSTM step) -----
        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step(
                w.predictor, last_token, state, next_state,
                scratch_x, scratch_gates, scratch_hh, nt);
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        // ----- Joint (using precomputed encoder projection) -----
        const float * enc_proj =
            enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        joint_step(w.joint, jg, nt, enc_proj, decoder_out,
                   scratch_pred_proj, scratch_summed,
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
        // points to verify each component matches the NeMo reference.
        // The first decode step (start state, encoder frame 0) gives
        // us four reference points: predictor input (the embed lookup,
        // here all zeros), per-layer LSTM h, the joint logits, the
        // argmax decision. The Python `dump_reference_parakeet_nemo.py
        // decode` subcommand mirrors these.
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
            predictor_dirty = true;
        }

        // Step / stuck advance. Matches the reference:
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

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                 "decoder: %d iters, %d tokens, T_enc=%d  "
                 "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
                 "total=%.1f ms  per_iter=%.0f us",
                 iter, static_cast<int>(out_tokens.size()), T_enc,
                 t_enc_proj_us / 1000.0,
                 t_pred_us  / 1000.0,
                 t_joint_us / 1000.0,
                 t_conf_us  / 1000.0,
                 (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0,
                 static_cast<double>(t_pred_us + t_joint_us) / std::max(iter, 1));

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder: hit iteration cap (%d) — pathological "
                     "logits or loop bug", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode
// ---------------------------------------------------------------------------
//
// Same predictor + joint forward as TDT, but the joint emits exactly
// `vocab+1` logits (no duration extras). Step rule:
//
//   per iteration:
//     run predictor (one LSTM step)
//     run joint -> token_logits[vocab+1]
//     argmax -> pred_token
//     if pred_token == blank:
//         step += 1
//         new_symbols = 0
//     else:
//         emit token, swap predictor state, last_token = pred_token
//         new_symbols += 1
//         if max_symbols and new_symbols >= max_symbols:
//             step += 1; new_symbols = 0   # break out of stuck-on-frame loop
//
// Mirrors NeMo's `RNNTGreedyDecodeInfer.step_per_frame` (without
// duration). Matches the reference dump points (`dec.embed.0`,
// `dec.lstm.{layer}.{h,c}.0`, `dec.joint.0`) emitted on iter 1.

transcribe_status decode_rnnt_greedy(const HostDecoderWeights & w,
                                     const float *              enc_out,
                                     int                        T_enc,
                                     int                        d_enc,
                                     int                        n_threads,
                                     std::vector<TdtToken> &    out_tokens)
{
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder (rnnt): enc d_model mismatch (got %d, "
                     "expected %d)", d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt           = resolve_decode_threads(n_threads);
    const int n_layers     = static_cast<int>(w.predictor.lstm.size());
    const int H            = w.predictor.pred_hidden;
    const int n_token_cls  = w.predictor.pred_vocab; // == vocab + 1
    const int blank_id     = w.blank_id;

    // Per-call joint compute graph (see decode_tdt_greedy).
    JointGraph jg;
    build_joint_graph(jg, w.joint, nt);

    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (same as TDT).
    const int joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t t_enc_proj_start = ggml_time_us();
    precompute_enc_proj(w.joint, enc_out, T_enc, d_enc, enc_proj_all, nt);
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    std::vector<float> scratch_x;
    std::vector<float> scratch_gates;
    std::vector<float> scratch_hh;
    std::vector<float> scratch_pred_proj;
    std::vector<float> scratch_summed;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = -1;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc + 1024;
    int iter = 0;
    int64_t t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // On a blank emission `(last_token, state)` are preserved, so the
    // predictor would deterministically write the same `next_state` on the
    // following iteration. Skip the LSTM unroll and reuse the previous
    // `decoder_out`; recompute on a non-blank emission, where
    // `state.swap(next_state)` and `last_token` change. Bit-exact with the
    // unconditional path. On a 0.6B FastConformer RNN-T with a typical
    // 4-5× iters/token ratio this elides 60-80% of predictor work and
    // ~halves decode wall time.
    bool predictor_dirty = true;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step(
                w.predictor, last_token, state, next_state,
                scratch_x, scratch_gates, scratch_hh, nt);
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        const float * enc_proj =
            enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        joint_step(w.joint, jg, nt, enc_proj, decoder_out,
                   scratch_pred_proj, scratch_summed,
                   logits);
        const int64_t t2 = ggml_time_us();
        t_pred_us  += t1 - t0;
        t_joint_us += t2 - t1;

        // RNNT joint output is just `n_token_cls` floats (no duration extras).
        const float * token_logits = logits.data();
        const int pred_token = argmax_range(token_logits, n_token_cls);

        if (iter == 1 && transcribe::debug::enabled()) {
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

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step       += 1;
            new_symbols = 0;
        } else {
            const int64_t tc0 = ggml_time_us();
            const float p = token_confidence(token_logits, n_token_cls,
                                             scratch_probs);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step       += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                 "decoder (rnnt): %d iters, %d tokens, T_enc=%d  "
                 "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
                 "total=%.1f ms",
                 iter, static_cast<int>(out_tokens.size()), T_enc,
                 t_enc_proj_us / 1000.0,
                 t_pred_us  / 1000.0,
                 t_joint_us / 1000.0,
                 t_conf_us  / 1000.0,
                 (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0);

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder (rnnt): hit iteration cap (%d) — "
                     "pathological logits or loop bug", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode — streaming
// ---------------------------------------------------------------------------
//
// Same algorithm as decode_rnnt_greedy but with externally-managed
// LSTM state (state_io) and previous token (last_token_io). The
// chunk's encoder frames are decoded in stream-wide coordinates
// (step_at_emit = frame_offset + local_step). No timing log.
transcribe_status decode_rnnt_greedy_streaming(
    const HostDecoderWeights & w,
    const float *              enc_out,
    int                        T_enc_new,
    int                        d_enc,
    LstmState &                state_io,
    int &                      last_token_io,
    int                        frame_offset,
    int                        n_threads,
    std::vector<TdtToken> &    out_tokens)
{
    if (enc_out == nullptr || T_enc_new <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // This driver implements pure RNN-T greedy search: every non-blank
    // advances the predictor by one symbol and every blank advances the
    // encoder frame by exactly one. A TDT head's duration predictions
    // would be silently ignored here (collapsing every blank to a single
    // frame) and a CTC head carries no predictor state at all. Both of
    // today's streaming parakeet variants are RNN-T, so this only guards
    // a hypothetical future TDT/CTC streaming model — but it fails loud at
    // the exact point of mis-decode rather than emitting wrong tokens.
    if (w.head_kind != HostHeadKind::RNNT) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder (rnnt-stream): streaming decode "
                     "requires an RNN-T head; this model's head is not "
                     "RNN-T");
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder (rnnt-stream): enc d_model mismatch "
                     "(got %d, expected %d)", d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt           = resolve_decode_threads(n_threads);
    const int n_layers     = static_cast<int>(w.predictor.lstm.size());
    const int H            = w.predictor.pred_hidden;
    const int n_token_cls  = w.predictor.pred_vocab;
    const int blank_id     = w.blank_id;

    // Per-call joint compute graph (see decode_tdt_greedy).
    JointGraph jg;
    build_joint_graph(jg, w.joint, nt);

    // Validate state_io shape; reset if degenerate (paranoid guard).
    if (static_cast<int>(state_io.h.size()) != n_layers ||
        static_cast<int>(state_io.c.size()) != n_layers)
    {
        state_io.reset(n_layers, H);
        last_token_io = -1;
    }
    for (int l = 0; l < n_layers; ++l) {
        if (static_cast<int>(state_io.h[l].size()) != H ||
            static_cast<int>(state_io.c[l].size()) != H)
        {
            state_io.reset(n_layers, H);
            last_token_io = -1;
            break;
        }
    }

    // Precompute encoder projections for this chunk only.
    const int joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    precompute_enc_proj(w.joint, enc_out, T_enc_new, d_enc, enc_proj_all, nt);

    // Working state. We mutate `state` and `next_state`; on return we
    // copy `state` (the COMMITTED state after the last non-blank
    // emission, or unchanged for an all-blank chunk) back into
    // state_io. last_token is committed similarly.
    LstmState state     = state_io;
    LstmState next_state;
    next_state.reset(n_layers, H);

    std::vector<float> scratch_x;
    std::vector<float> scratch_gates;
    std::vector<float> scratch_hh;
    std::vector<float> scratch_pred_proj;
    std::vector<float> scratch_summed;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = last_token_io;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc_new + 1024;
    int iter = 0;

    bool predictor_dirty = true;

    while (step < T_enc_new && iter < max_iters) {
        ++iter;

        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step(
                w.predictor, last_token, state, next_state,
                scratch_x, scratch_gates, scratch_hh, nt);
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }

        const float * enc_proj = enc_proj_all.data() +
            static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        joint_step(w.joint, jg, nt, enc_proj, decoder_out,
                   scratch_pred_proj, scratch_summed, logits);

        const float * token_logits = logits.data();
        const int pred_token = argmax_range(token_logits, n_token_cls);

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step       += 1;
            new_symbols = 0;
        } else {
            const float p = token_confidence(token_logits, n_token_cls,
                                             scratch_probs);
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = frame_offset + step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step       += 1;
                new_symbols = 0;
            }
        }
    }

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder (rnnt-stream): hit iteration cap (%d)",
                     max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Commit final state. After the loop, `state` is the post-last-
    // non-blank state (the predictor's "current" since the last emit).
    state_io      = std::move(state);
    last_token_io = last_token;
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// CTC greedy decode
// ---------------------------------------------------------------------------
//
// Per-frame: logits[t] = W @ enc[t] + b   ->   log_softmax   ->   argmax.
// Collapse: drop adjacent duplicates ("aaab" -> "ab"), then drop the
// blank label. Standard CTC greedy (see NeMo GreedyCTCInfer).
//
// Dump points (gated on TRANSCRIBE_DUMP_DIR):
//   - dec.ctc.logprobs   : full [T_enc, n_classes] log-softmax
//   - dec.ctc.logprobs.0 : frame 0 of the same
// These match the Stage 2 oracle's reference dumps.

transcribe_status decode_ctc_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    int                        n_threads,
                                    std::vector<TdtToken> &    out_tokens)
{
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.ctc_head.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet decoder (ctc): enc d_model mismatch "
                     "(got %d, expected %d)", d_enc, w.ctc_head.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt        = resolve_decode_threads(n_threads);
    const int n_classes = w.ctc_head.n_classes;
    const int blank_id  = w.ctc_head.blank_id;

    // Compute T_enc frames of logits: [T_enc, n_classes] = enc_out[T_enc, d_enc] @ W^T + b.
    std::vector<float> logits_all(
        static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes));

    const int64_t t_proj_start = ggml_time_us();
#if TRANSCRIBE_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T_enc, n_classes, d_enc,
                1.0f, enc_out, d_enc,
                w.ctc_head.weight.data(), d_enc,
                0.0f, logits_all.data(), n_classes);
#else
    for (int t = 0; t < T_enc; ++t) {
        const float * frame = enc_out + static_cast<size_t>(t) * static_cast<size_t>(d_enc);
        float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        linear(w.ctc_head.weight.data(), frame, nullptr, n_classes, d_enc, row, nt);
    }
#endif
    for (int t = 0; t < T_enc; ++t) {
        float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        for (int c = 0; c < n_classes; ++c) {
            row[c] += w.ctc_head.bias[static_cast<size_t>(c)];
        }
    }
    const int64_t t_proj_us = ggml_time_us() - t_proj_start;

    // Per-frame log-softmax in place. The reference dumps log-probs,
    // not raw logits.
    for (int t = 0; t < T_enc; ++t) {
        float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
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

    // Optional dumps. Frame 0 is shape [n_classes]; full is
    // [T_enc, n_classes] in row-major (== ggml ne fast-to-slow
    // [n_classes, T_enc]).
    if (transcribe::debug::enabled()) {
        const long long s_n = n_classes;
        transcribe::debug::dump_host_f32(
            "dec.ctc.logprobs.0", logits_all.data(), s_n, &s_n, 1,
            "decoder.ctc.logprobs.0");
        const long long shape_full[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32(
            "dec.ctc.logprobs", logits_all.data(),
            static_cast<long long>(logits_all.size()),
            shape_full, 2, "decoder.ctc.logprobs");
    }

    // Greedy collapse: drop runs of identical labels, then drop blanks.
    // Per-emit `step_at_emit` is the encoder frame at which the label
    // first changed; duration_frames = 1.
    int prev_label = -1;
    for (int t = 0; t < T_enc; ++t) {
        const float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        const int label = argmax_range(row, n_classes);
        if (label == prev_label) continue; // run-length collapse
        prev_label = label;
        if (label == blank_id) continue;
        TdtToken tok;
        tok.id              = label;
        // CTC greedy confidence = exp(top log-prob); this is the
        // model's own probability for the chosen label, in [0, 1].
        tok.p               = std::exp(row[label]);
        tok.step_at_emit    = t;
        tok.duration_frames = 1;
        out_tokens.push_back(tok);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                 "decoder (ctc): T_enc=%d  proj=%.1f ms  emitted=%d",
                 T_enc, t_proj_us / 1000.0,
                 static_cast<int>(out_tokens.size()));

    return TRANSCRIBE_OK;
}

} // namespace transcribe::parakeet
