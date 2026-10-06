// arch/ecapa_tdnn/model.cpp - ECAPA-TDNN family handler (LANGID role).
//
// load() reads the hyperparameters and label table, builds the tensor
// catalogue, uploads the weights to the chosen backend, and constructs the
// shared log-mel front end from the GGUF's own filterbank. The LANGID run
// hook drives the front end host-side, builds and computes the graph in
// graph.cpp, and hands one logit per label back to the role dispatcher
// (transcribe-langid.cpp), which owns the softmax / ranking.

#include "ecapa_tdnn.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "graph.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-path.h"
#include "weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

namespace transcribe::ecapa_tdnn {

namespace {

constexpr const char * kTag = "ecapa_tdnn";

// stt.variant is optional in the GGUF; this is the family's only shipped
// checkpoint and the default when the key is absent.
constexpr const char k_default_variant[] = "lang-id-voxlingua107-ecapa";

// Nodes reserved in the scheduler. Must be >= the graph's node count; see
// graph.cpp's kGraphSize. The built graph is 571 nodes and is independent of
// T, so this is ~3.5x headroom.
constexpr size_t k_sched_graph_size = 2048;

// Metadata arena for one per-call graph build. The published configuration
// uses 0.36 MiB of it (the graph is independent of T); the rest is headroom
// so a future graph change cannot silently truncate.
constexpr size_t k_compute_ctx_bytes = 4u * 1024u * 1024u;

int64_t now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

// Frees the gguf_context on every exit path of load().
struct GgufGuard {
    gguf_context * ctx = nullptr;

    GgufGuard() = default;

    ~GgufGuard() {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }

    GgufGuard(const GgufGuard &)             = delete;
    GgufGuard & operator=(const GgufGuard &) = delete;
};

// Reflect-padding gather indices for a [C, T] activation:
//   [p, p-1, ..., 1, 0, 1, ..., T-1, T-2, ..., T-1-p]
// i.e. torch's "reflect" mode, which mirrors WITHOUT repeating the edge
// frame. Requires T > p, which the 500 ms audio minimum guarantees.
void build_reflect_indices(int T, int p, std::vector<int32_t> & out) {
    out.resize(static_cast<size_t>(T) + 2 * static_cast<size_t>(p));
    for (int j = 0; j < p; ++j) {
        out[static_cast<size_t>(j)] = p - j;
    }
    for (int t = 0; t < T; ++t) {
        out[static_cast<size_t>(p + t)] = t;
    }
    for (int j = 0; j < p; ++j) {
        out[static_cast<size_t>(p + T + j)] = T - 2 - j;
    }
}

// Read stt.langid.labels.* into a validated label table (H6). Aliases are
// optional; a present-but-mistyped key is a converter bug.
transcribe_status read_labels(const gguf_context * gguf, LangidLabels & out) {
    std::vector<std::string> codes;
    std::vector<std::string> names;
    std::vector<std::string> aliases;
    if (read_string_array_kv(gguf, "stt.langid.labels.codes", codes) != KvResult::Ok ||
        read_string_array_kv(gguf, "stt.langid.labels.names", names) != KvResult::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: stt.langid.labels.codes / names missing or not string arrays", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (read_string_array_kv(gguf, "stt.langid.labels.aliases", aliases) == KvResult::BadType) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: stt.langid.labels.aliases is not a string array", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    return build_langid_labels(std::move(codes), std::move(names), aliases, kTag, out);
}

// Q8_0 is a download format here, not a compute format: every Q8_0 weight is
// widened to F16 at load. ggml's Q8_0 matmuls also quantize the ACTIVATIONS
// to 8 bits, and on FLEURS that, not the 8-bit weights, is what moves the
// decision; the same weights computed in F16 score at F32's accuracy
// (agreement figures: docs/models/lang-id-voxlingua107-ecapa.md). The cost
// is F16 memory for those weights, and F16 speed.
//
// Retype in place: the tensors come from the no_alloc gguf context and have
// no data or views yet, so only type and strides change. Returns the count.
int widen_q8_0_weights(ggml_context * ctx_meta) {
    int n = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (t->type != GGML_TYPE_Q8_0) {
            continue;
        }
        t->type  = GGML_TYPE_F16;
        t->nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = t->nb[i - 1] * static_cast<size_t>(t->ne[i - 1]);
        }
        ++n;
    }
    return n;
}

// load_common::stream_tensor_data, plus the Q8_0 -> F16 widening: a tensor
// whose file type is Q8_0 and whose bound type is F16 is dequantized with
// ggml's own Q8_0 reference and rounded to F16 on the way in. Every other
// tensor must be bound with its file type and is copied as-is.
transcribe_status stream_weights(const std::string & path, const gguf_context * gguf, ggml_context * ctx_meta) {
    std::ifstream fin(path_from_utf8(path), std::ios::binary);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to reopen %s for tensor data", kTag, path.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    const size_t             data_offset = gguf_get_data_offset(gguf);
    const ggml_to_float_t    q8_to_f32   = ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float;
    std::vector<uint8_t>     staging;
    std::vector<float>       f32;
    std::vector<ggml_fp16_t> f16;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t != nullptr; t = ggml_get_next_tensor(ctx_meta, t)) {
        const int64_t idx = gguf_find_tensor(gguf, t->name);
        if (idx < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: tensor \"%s\" not in gguf data", kTag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }
        const ggml_type file_type = gguf_get_tensor_type(gguf, idx);
        const bool      widen     = file_type == GGML_TYPE_Q8_0 && t->type == GGML_TYPE_F16;
        if (file_type != t->type && !widen) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: tensor \"%s\" is %s in the file but bound as %s", kTag, t->name,
                    ggml_type_name(file_type), ggml_type_name(t->type));
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t nbytes = gguf_get_tensor_size(gguf, idx);
        if (!widen && nbytes != ggml_nbytes(t)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: tensor \"%s\" size mismatch", kTag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }

        fin.seekg(static_cast<std::streamoff>(data_offset) +
                  static_cast<std::streamoff>(gguf_get_tensor_offset(gguf, idx)));
        if (staging.size() < nbytes) {
            staging.resize(nbytes);
        }
        fin.read(reinterpret_cast<char *>(staging.data()), static_cast<std::streamsize>(nbytes));
        if (!fin) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: short read for tensor \"%s\" (%zu bytes)", kTag, t->name, nbytes);
            return TRANSCRIBE_ERR_GGUF;
        }

        if (!widen) {
            ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
            continue;
        }
        const int64_t n = ggml_nelements(t);
        if (nbytes != ggml_row_size(GGML_TYPE_Q8_0, n)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: Q8_0 tensor \"%s\" size mismatch", kTag, t->name);
            return TRANSCRIBE_ERR_GGUF;
        }
        f32.resize(static_cast<size_t>(n));
        f16.resize(static_cast<size_t>(n));
        q8_to_f32(staging.data(), f32.data(), n);
        ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
        ggml_backend_tensor_set(t, f16.data(), 0, ggml_nbytes(t));
    }
    return TRANSCRIBE_OK;
}

// Build Weights::blk0_w_im2col from blk0_w: the tap-major [n_mels, C, K0]
// kernel regrouped to [K0*n_mels, C] (column k*n_mels + m of output row c is
// tap k, mel m) and zero padded to hp.blk0_cols(). Same type as blk0_w.
transcribe_status build_derived_weights(Model & m) {
    const HParams & hp     = m.hparams;
    ggml_tensor *   src    = m.weights.blk0_w;
    const int64_t   n_mels = hp.mel_n_mels;
    const int64_t   C      = hp.c_block();
    const int64_t   K      = hp.kernel_sizes[0];
    const int64_t   cols   = hp.blk0_cols();

    ggml_init_params ip{};
    ip.mem_size   = 4 * ggml_tensor_overhead();
    ip.no_alloc   = true;
    m.ctx_derived = ggml_init(ip);
    if (m.ctx_derived == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_tensor * dst = ggml_new_tensor_2d(m.ctx_derived, src->type, cols, C);
    ggml_set_name(dst, "blk.0.conv.weight.im2col");
    m.derived_buffer = ggml_backend_alloc_ctx_tensors(m.ctx_derived, m.plan.primary);
    if (m.derived_buffer == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m.derived_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t         esz = ggml_type_size(src->type);  // F32 / F16, block size 1
    std::vector<uint8_t> in(ggml_nbytes(src));
    std::vector<uint8_t> out(ggml_nbytes(dst), 0);         // zero bits are 0.0 in F32 and F16
    ggml_backend_tensor_get(src, in.data(), 0, in.size());
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t c = 0; c < C; ++c) {
            const uint8_t * s = in.data() + static_cast<size_t>(k) * src->nb[2] + static_cast<size_t>(c) * src->nb[1];
            uint8_t * d = out.data() + static_cast<size_t>(c) * dst->nb[1] + static_cast<size_t>(k * n_mels) * esz;
            std::memcpy(d, s, static_cast<size_t>(n_mels) * esz);
        }
    }
    ggml_backend_tensor_set(dst, out.data(), 0, out.size());
    m.weights.blk0_w_im2col = dst;
    return TRANSCRIBE_OK;
}

// Repack weights for the AVX2 GEMM (cpu_gemm.h), in two all-or-nothing
// groups the graph routes as a unit:
//   - the k>1 kernels (F32 in every shipped file): this GEMM beats ggml's on
//     their small shapes, and stage 0's K = 320 is not a tinyBLAS shape;
//   - the T-wide 1x1 weights, only when F16 (F16 files, and Q8_0 after
//     widening): ggml widens F16 operands in its inner loop, ~1.7x slower.
//     For F32 1x1 weights ggml's tinyBLAS is as fast as this GEMM.
// A group is skipped (left on ggml_mul_mat) unless every weight fits the
// GEMM: F16/F32, contiguous, rows a multiple of the panel, and at most
// kMaxSegs matrices (one GEMM segment per tap).
transcribe_status pack_gemm_weights(Model & m) {
    if (!gemm::available()) {
        return TRANSCRIBE_OK;
    }
    Weights & w = m.weights;

    auto fits = [](const std::vector<ggml_tensor *> & ts, bool need_f16) {
        for (const ggml_tensor * t : ts) {
            if (t->ne[1] % gemm::kPanel != 0 || t->ne[2] * t->ne[3] > gemm::kMaxSegs || !ggml_is_contiguous(t) ||
                (need_f16 && t->type != GGML_TYPE_F16) || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32)) {
                return false;
            }
        }
        return true;
    };
    // After fits() this cannot fail; if it ever does, some weights are
    // already packed and unreadable by ggml_mul_mat, so the load fails.
    auto pack = [&](const std::vector<ggml_tensor *> & ts) {
        for (ggml_tensor * t : ts) {
            if (!gemm::pack_weight(t)) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: GEMM weight repack failed for %s", kTag, t->name);
                return false;
            }
        }
        return true;
    };

    std::vector<ggml_tensor *> conv{ w.blk0_w_im2col };
    std::vector<ggml_tensor *> lin;
    for (SeRes2NetBlock & b : w.blocks) {
        for (Res2Sub & r : b.res2) {
            conv.push_back(r.w);
        }
        lin.push_back(b.tdnn1.w);
        lin.push_back(b.tdnn2.w);
    }
    for (ggml_tensor * t : w.mfa_w) {
        lin.push_back(t);
    }
    lin.push_back(w.asp_wx);
    lin.push_back(w.asp_attn_w);

    m.gemm_conv = fits(conv, /*need_f16=*/false);
    m.gemm_lin  = fits(lin, /*need_f16=*/true);
    if ((m.gemm_conv && !pack(conv)) || (m.gemm_lin && !pack(lin))) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: AVX2 GEMM for %s%s%s", kTag, m.gemm_conv ? "conv" : "",
            m.gemm_conv && m.gemm_lin ? " + " : "", m.gemm_lin ? "F16 1x1" : (m.gemm_conv ? "" : "nothing"));
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = now_us();

    // Owned by RAII until the very end, so a failure (or a throw) anywhere
    // below, label construction included, frees everything built so far (H4).
    auto m     = std::make_unique<Model>();
    m->arch    = &arch;
    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();

    if (auto st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    const HParams & hp = m->hparams;

    if (auto st = read_labels(loader.gguf(), m->labels); st != TRANSCRIBE_OK) {
        return st;
    }
    if (static_cast<int32_t>(m->labels.codes.size()) != hp.n_labels) {
        return TRANSCRIBE_ERR_GGUF;  // unreachable: both read the same KV
    }

    // ---- front end ------------------------------------------------------
    //
    // The filterbank is SpeechBrain's own triangle geometry and is never
    // rebuilt here. It is stored ne = [n_freq, n_mels], so the flat host
    // vector is already mel-major, which is what MelConfig wants.
    {
        const size_t expected = static_cast<size_t>(hp.mel_n_mels) * static_cast<size_t>(hp.n_freq());

        MelConfig cfg;
        cfg.sample_rate = hp.sample_rate;
        cfg.n_mels      = hp.mel_n_mels;
        cfg.n_fft       = hp.mel_n_fft;
        cfg.win_length  = hp.mel_win;
        cfg.hop_length  = hp.mel_hop;
        cfg.window_type = hp.mel_window;
        cfg.pad_mode    = hp.mel_pad_mode;
        cfg.log_floor   = hp.mel_log_floor;
        cfg.top_db      = hp.mel_top_db;
        cfg.normalize   = hp.mel_normalize;

        const auto rr = load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.mel_filterbank",
                                                             expected, kTag, cfg.filterbank);
        if (rr != load_common::ReadF32Result::Ok) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "%s: frontend.mel_filterbank missing or unreadable (expected %zu f32 values)", kTag, expected);
            return TRANSCRIBE_ERR_GGUF;
        }

        m->mel = std::make_unique<MelFrontend>(cfg);
        if (m->mel->status() != TRANSCRIBE_OK) {
            return m->mel->status();
        }
    }

    // ---- weights ---------------------------------------------------------
    //
    // Reopen the file with no_alloc so ctx_meta holds tensor structs whose
    // data pointers are filled in by the backend allocation below.
    GgufGuard guard;
    {
        gguf_init_params init_params{};
        init_params.no_alloc = true;
        init_params.ctx      = &m->ctx_meta;
        guard.ctx            = gguf_init_from_file(loader.path().c_str(), init_params);
        if (guard.ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to reopen \"%s\" for tensor data", kTag,
                    loader.path().c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    if (auto st = build_weights(m->ctx_meta, hp, m->weights); st != TRANSCRIBE_OK) {
        return st;
    }

    // After build_weights, which validated each weight's file type.
    if (const int n_widened = widen_q8_0_weights(m->ctx_meta); n_widened > 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s: %d Q8_0 weights widened to F16 at load", kTag, n_widened);
    }

    const transcribe_backend_request backend_req = params != nullptr ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = load_common::init_backends(backend_req, params != nullptr ? params->device : nullptr, kTag, m->plan);
        st != TRANSCRIBE_OK) {
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    m->backend_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (m->backend_buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: ggml_backend_alloc_ctx_tensors failed", kTag);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m->backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = stream_weights(loader.path(), guard.ctx, m->ctx_meta); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = build_derived_weights(*m); st != TRANSCRIBE_OK) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: derived weight allocation failed", kTag);
        return st;
    }

    // Fused CPU kernels only when the whole graph runs on the CPU backend:
    // a custom op has no GPU implementation, and the scheduler would bounce
    // every one of them back to the host.
    m->cpu_ops = m->plan.primary_kind == BackendKind::Cpu && m->plan.scheduler_list.size() == 1;
    if (m->cpu_ops) {
        if (auto st = pack_gemm_weights(*m); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    m->roles = TRANSCRIBE_ROLE_LANGID;
    // The abort callback is honored (transcribe_langid_set_abort_callback).
    set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = now_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// forward pass
// ---------------------------------------------------------------------------
//
// On success the returned GraphBuild's tensors are live: they belong to
// s.compute_ctx, which stays alive until the dispatcher releases the scratch,
// and their data belongs to the scheduler's allocation for the graph that was
// just computed. The caller reads outputs with ggml_backend_tensor_get.
transcribe_status forward(Session & s, const Model & m, const float * pcm, int n_samples, GraphBuild & gb_out) {
    debug::init();

    const HParams & hp        = m.hparams;
    const int       n_threads = s.n_threads > 0 ? s.n_threads : default_n_threads();

    // ---- front end (host side) -------------------------------------------
    const int64_t t_mel_start = now_us();
    int           T           = 0;
    if (auto st = m.mel->compute(pcm, n_samples, s.mel_buf, T, n_threads); st != TRANSCRIBE_OK) {
        return st;
    }
    s.t_mel_us = now_us() - t_mel_start;

    if (T <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: front end produced no frames for %d samples", kTag, n_samples);
        return TRANSCRIBE_ERR_INPUT_TOO_SHORT;
    }
    if (s.poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    if (debug::enabled()) {
        const long long shape[2] = { T, hp.mel_n_mels };
        debug::dump_host_f32("fe.mel", s.mel_buf.data(), static_cast<long long>(s.mel_buf.size()), shape, 2,
                             "frontend");
    }

    // reflect_rows / the Res2Net kernel need T > pad (the 500 ms LANGID
    // minimum gives T >= 51 frames; the largest pad here is 4).
    for (int i = 0; i < kNumStages - 1; ++i) {
        if (T <= hp.pad(i)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: T=%d is too short for reflect padding %d", kTag, T, hp.pad(i));
            return TRANSCRIBE_ERR_INPUT_TOO_SHORT;
        }
    }

    // ---- stage-0 im2col and reflect-padding indices ------------------------
    build_blk0_im2col(hp, s.mel_buf.data(), T, s.im2col_buf);
    if (!m.cpu_ops) {
        for (int i = 0; i < kNumSeBlocks; ++i) {
            build_reflect_indices(T, hp.pad(i + 1), s.idx_buf[i]);
        }
    }

    // ---- graph -------------------------------------------------------------
    //
    // A fresh no_alloc metadata context per call: the previous build's tensor
    // structs are dead the moment the scheduler is reset.
    if (s.compute_ctx != nullptr) {
        ggml_free(s.compute_ctx);
        s.compute_ctx = nullptr;
    }
    s.gemm_arena.clear();
    {
        if (s.graph_arena.size() != k_compute_ctx_bytes) {
            s.graph_arena.resize(k_compute_ctx_bytes);
        }
        ggml_init_params init_params{};
        init_params.mem_size   = s.graph_arena.size();
        init_params.mem_buffer = s.graph_arena.data();
        init_params.no_alloc   = true;
        s.compute_ctx          = ggml_init(init_params);
        if (s.compute_ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: compute context allocation failed", kTag);
            return TRANSCRIBE_ERR_OOM;
        }
    }

    GraphBuild gb = build_graph(s.compute_ctx, m, T, m.cpu_ops, &s.gemm_arena);
    if (gb.graph == nullptr || gb.logits == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (s.sched == nullptr) {
        // Cast away const for the scheduler list only: the scheduler borrows
        // the model's backends and never mutates the model.
        auto & backends = const_cast<std::vector<ggml_backend_t> &>(m.plan.scheduler_list);
        s.sched         = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()),
                                                 k_sched_graph_size, /*parallel=*/false, /*op_offload=*/true);
        if (s.sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: scheduler allocation failed", kTag);
            return TRANSCRIBE_ERR_OOM;
        }
    }
    configure_sched_n_threads(s.sched, n_threads);

    ggml_backend_sched_reset(s.sched);
    if (!ggml_backend_sched_alloc_graph(s.sched, gb.graph)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: graph allocation failed", kTag);
        return TRANSCRIBE_ERR_OOM;
    }

    // ---- inputs -------------------------------------------------------------
    // Frame-major [T, cols], byte-identical to ggml ne = [cols, T].
    ggml_backend_tensor_set(gb.blk0_in, s.im2col_buf.data(), 0, s.im2col_buf.size() * sizeof(float));
    for (int i = 0; i < kNumSeBlocks; ++i) {
        if (gb.idx[i] != nullptr) {
            ggml_backend_tensor_set(gb.idx[i], s.idx_buf[i].data(), 0, s.idx_buf[i].size() * sizeof(int32_t));
        }
    }
    if (gb.chunk_ids != nullptr) {
        int32_t ids[kRes2NetScale];
        for (int i = 0; i < kRes2NetScale; ++i) {
            ids[i] = i;
        }
        ggml_backend_tensor_set(gb.chunk_ids, ids, 0, sizeof(ids));
    }

    // ---- compute -------------------------------------------------------------
    const int64_t t_enc_start = now_us();
    if (const ggml_status gs = ggml_backend_sched_graph_compute(s.sched, gb.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: graph compute failed (%d)", kTag, static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    s.t_encode_us = now_us() - t_enc_start;

    gb_out = gb;
    return TRANSCRIBE_OK;
}

// Write the contract's stage tensors. No-op unless TRANSCRIBE_DUMP_DIR is set.
void dump_stages(const GraphBuild & gb) {
    if (!debug::enabled()) {
        return;
    }

    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) {
            debug::dump_tensor(name, t, stage);
        }
    };

    try_dump("enc.blk.0.out", gb.dumps.blk0_out, "encoder");
    try_dump("enc.blk.1.tdnn1.out", gb.dumps.blk1_tdnn1_out, "encoder");
    try_dump("enc.blk.1.res2.out", gb.dumps.blk1_res2_out, "encoder");
    try_dump("enc.blk.1.se.out", gb.dumps.blk1_se_out, "encoder");  // stock-op graph only
    try_dump("enc.blk.1.out", gb.dumps.blk_out[0], "encoder");
    try_dump("enc.blk.2.out", gb.dumps.blk_out[1], "encoder");
    try_dump("enc.blk.3.out", gb.dumps.blk_out[2], "encoder");
    try_dump("enc.mfa.out", gb.dumps.mfa_out, "encoder");
    try_dump("enc.asp.attn_logits", gb.dumps.asp_attn_logits, "encoder");
    try_dump("enc.asp.out", gb.dumps.asp_out, "encoder");
    try_dump("enc.emb", gb.dumps.emb, "encoder");
    try_dump("cls.hidden", gb.dumps.cls_hidden, "classifier");
    try_dump("cls.logits_raw", gb.dumps.cls_logits, "classifier");
}

// SpeechBrain's classifier ends in log_softmax; the role needs raw logits, so
// the log-probabilities are recomputed here purely for the validation dump.
// Stable form: z - (max + log(sum(exp(z - max)))).
void dump_log_probs(const std::vector<float> & logits) {
    if (!debug::enabled() || logits.empty()) {
        return;
    }

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double      sum       = 0.0;
    for (const float v : logits) {
        sum += std::exp(static_cast<double>(v - max_logit));
    }
    const double log_z = static_cast<double>(max_logit) + std::log(sum);

    std::vector<float> lp(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        lp[i] = static_cast<float>(static_cast<double>(logits[i]) - log_z);
    }

    const long long shape[1] = { static_cast<long long>(lp.size()) };
    debug::dump_host_f32("cls.log_probs", lp.data(), static_cast<long long>(lp.size()), shape, 1, "classifier");
}

// ---------------------------------------------------------------------------
// LANGID ops
// ---------------------------------------------------------------------------

const LangidLabels & langid_labels(const transcribe_model * model) {
    return static_cast<const Model *>(model)->labels;
}

transcribe_langid_session * langid_new_session() {
    return new Session();
}

transcribe_status langid_run(transcribe_langid_session * session,
                             const float *               pcm,
                             int                         n_samples,
                             std::vector<float> &        logits) {
    auto &       s = static_cast<Session &>(*session);
    const auto & m = static_cast<const Model &>(*session->model);
    if (s.poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    GraphBuild gb{};
    if (auto st = forward(s, m, pcm, n_samples, gb); st != TRANSCRIBE_OK) {
        return st;
    }

    logits.assign(static_cast<size_t>(m.hparams.n_labels), 0.0f);
    ggml_backend_tensor_get(gb.logits, logits.data(), 0, logits.size() * sizeof(float));

    dump_stages(gb);
    dump_log_probs(logits);
    return TRANSCRIBE_OK;
}

const LangidOps k_langid_ops = { langid_labels, langid_new_session, langid_run };

}  // namespace

// ---------------------------------------------------------------------------
// Destructor, registry entry
// ---------------------------------------------------------------------------

Model::~Model() {
    mel.reset();
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
    }
    if (ctx_derived != nullptr) {
        ggml_free(ctx_derived);
    }
    if (derived_buffer != nullptr) {
        safe_buffer_free(derived_buffer);
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

const Arch arch = {
    /* .name             = */ "ecapa_tdnn",
    /* .load             = */ load,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ &k_langid_ops,
};

}  // namespace transcribe::ecapa_tdnn
