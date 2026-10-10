// arch/whistle/model.cpp - Whistle family handler.
//
// Load / init_context / run / run_batch for the Whistle encoder-decoder.
// Flow per utterance:
//   host log-mel + no-speech gate -> encoder graph -> memory
//   (enc + sigmoid(pe_gate) * sinusoid) -> cross K/V -> prompt <s> <|lang|>
//   (auto language = argmax over the language tokens after <s>) -> beam
//   search (the reference engine decodes with 5 beams) -> optional word
//   timestamps by DTW over cross-attention of the chosen hypothesis.
//
// Provenance of every op: reports/porting/whistle/forward-map.md.

#include "decoder.h"
#include "encoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-path.h"
#include "transcribe-repetition-guard.h"
#include "weights.h"
#include "whistle.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#    include <Accelerate/Accelerate.h>
#    include <sys/sysctl.h>
#endif

namespace transcribe::whistle {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, WhistleModel>);
static_assert(std::is_base_of_v<transcribe_session, WhistleSession>);

namespace {

constexpr const char k_default_variant[] = "whistle";
constexpr int        k_beam_size         = 5;
constexpr int        k_prompt_len        = 2;  // <s> <|lang|>
constexpr size_t     k_graph_mem         = 96u * 1024u * 1024u;
// Scheduler hash-set capacity: >= nodes + leafs of the largest graph (the
// encoder, ~1.5k). It is reset per graph, i.e. every decode step, so it is
// kept near the need rather than at a generic maximum.
constexpr size_t     k_sched_nodes       = 8192;

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Default CPU thread count (n_threads <= 0). Whistle's graphs are many small,
// barrier-separated ops, so the slowest participating core paces every op:
// on Apple hybrid CPUs, adding efficiency cores makes both encode and decode
// slower (M4 4P+6E, german 29 s Q8_0: 145 ms at 4 threads, 186 ms at 5, 203 ms
// at the library default of 8). Use the performance-core count there, capped
// by the library default; other hosts keep the library default.
int whistle_default_threads() {
    const int lib = transcribe::default_n_threads();
#if defined(__APPLE__)
    int    perf = 0;
    size_t len  = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0) {
        return std::max(1, std::min(perf, lib));
    }
#endif
    return lib;
}

std::vector<float> read_host(ggml_tensor * t) {
    std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// Aux / cache lifetime
// ---------------------------------------------------------------------------

void WhistleAux::free() {
    if (buffer != nullptr) {
        safe_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

void WhistleDecCache::free() {
    if (buffer != nullptr) {
        safe_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    k_self.clear();
    v_self.clear();
    tap_q.clear();
    tap_k.clear();
    tap_v.clear();
    k_cross.clear();
    v_cross.clear();
    eg_val.clear();
    n_ctx = n_seq = n_utt = T_enc = tap_len = eg_len = 0;
}

bool dec_cache_init(WhistleDecCache &      cache,
                    ggml_backend_t         backend,
                    const WhistleHParams & hp,
                    int                    n_ctx,
                    int                    n_seq,
                    int                    n_utt,
                    int                    T_enc) {
    cache.free();
    const int L        = hp.dec_n_layers;
    const int n_sites  = static_cast<int>(hp.engram_sites.size());
    const int n_tensor = L * 7 + n_sites;

    ggml_init_params p{};
    p.mem_size = static_cast<size_t>(n_tensor + 8) * ggml_tensor_overhead();
    p.no_alloc = true;
    cache.ctx  = ggml_init(p);
    if (cache.ctx == nullptr) {
        return false;
    }
    cache.n_ctx   = n_ctx;
    cache.n_seq   = n_seq;
    cache.n_utt   = n_utt;
    cache.T_enc   = T_enc;
    cache.tap_len = hp.dec_taps;
    cache.eg_len  = hp.engram_conv_dilation * (hp.engram_conv_taps - 1) + 1;

    const int64_t qk = hp.qk_head_dim, vd = hp.v_head_dim, nh = hp.n_heads, nkv = hp.n_kv_heads;
    // The transposed V caches are read with the position axis zero-padded to a
    // multiple of 4 (decoder.cpp pad_k4); the slack stays zero (cleared below).
    const int64_t n_ctx4 = (n_ctx + 3) / 4 * 4, T_enc4 = (T_enc + 3) / 4 * 4;
    for (int l = 0; l < L; ++l) {
        cache.k_self.push_back(ggml_new_tensor_4d(cache.ctx, GGML_TYPE_F32, qk, n_ctx, nkv, n_seq));
        cache.v_self.push_back(ggml_new_tensor_4d(cache.ctx, GGML_TYPE_F32, n_ctx4, vd, nkv, n_seq));
        cache.tap_q.push_back(ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, nh * qk, cache.tap_len, n_seq));
        cache.tap_k.push_back(ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, nkv * qk, cache.tap_len, n_seq));
        cache.tap_v.push_back(ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, nkv * vd, cache.tap_len, n_seq));
        cache.k_cross.push_back(ggml_new_tensor_4d(cache.ctx, GGML_TYPE_F32, qk, T_enc, nh, n_utt));
        cache.v_cross.push_back(ggml_new_tensor_4d(cache.ctx, GGML_TYPE_F32, T_enc4, vd, nh, n_utt));
    }
    for (int s = 0; s < n_sites; ++s) {
        cache.eg_val.push_back(ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, hp.d_model, cache.eg_len, n_seq));
    }
    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        cache.free();
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);
    return true;
}

WhistleModel::~WhistleModel() {
    aux.free();
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (ctx_fused != nullptr) {
        ggml_free(ctx_fused);
        ctx_fused = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    if (repack_buffer != nullptr) {
        safe_buffer_free(repack_buffer);
        repack_buffer = nullptr;
    }
    if (head_buffer != nullptr) {
        safe_buffer_free(head_buffer);
        head_buffer = nullptr;
    }
    if (ctx_head != nullptr) {
        ggml_free(ctx_head);
        ctx_head = nullptr;
    }
    if (threadpool != nullptr) {
        if (plan.primary != nullptr) {
            ggml_backend_cpu_set_threadpool(plan.primary, nullptr);
        }
        ggml_threadpool_free(threadpool);
        threadpool = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

WhistleSession::~WhistleSession() {
    cache.free();
}

namespace {

// Load-time aux: host scalars, integer permutations, mHC bias vectors with
// the per-layer lane offsets of architecture.py::Stack folded in.
// ggml's CPU repack buffer type (interleaved weights for the i8mm / dotprod
// GEMM kernels), or nullptr when the CPU backend does not provide one.
ggml_backend_buffer_type_t cpu_repack_buft(ggml_backend_dev_t & dev) {
    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return nullptr;
    }
    auto get = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    if (get == nullptr) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * b = get(dev); b != nullptr && *b != nullptr; ++b) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) {
            return *b;
        }
    }
    return nullptr;
}

// Rows [row0, row0 + rows) of a GGUF weight, one part of a fused tensor.
struct FusePart {
    const ggml_tensor * src  = nullptr;
    int64_t             row0 = 0;
    int64_t             rows = 0;
};

struct FuseGroup {
    ggml_tensor *         fused = nullptr;
    std::vector<FusePart> parts;
};

// Plans the load-time projection fusions (WhistleAttn::fused, WhistleMhc::
// phi_fused) and creates the fused tensors in m.ctx_fused. A group is fused
// only when every part has the same type; `skip` receives the GGUF tensors
// that are then never allocated or streamed.
transcribe_status plan_fusion(WhistleModel &                     m,
                              std::vector<FuseGroup> &           groups,
                              std::vector<const ggml_tensor *> & skip) {
    WhistleWeights & w     = m.weights;
    const int        lanes = m.hparams.mhc_lanes;
    ggml_init_params p{};
    p.mem_size  = 512 * ggml_tensor_overhead();
    p.no_alloc  = true;
    m.ctx_fused = ggml_init(p);
    if (m.ctx_fused == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    auto add = [&](std::vector<FusePart> parts, const char * name) -> ggml_tensor * {
        int64_t rows = 0;
        for (const FusePart & fp : parts) {
            if (fp.src == nullptr || fp.src->type != parts[0].src->type || fp.src->ne[0] != parts[0].src->ne[0]) {
                return nullptr;
            }
            rows += fp.rows;
        }
        ggml_tensor * t = ggml_new_tensor_2d(m.ctx_fused, parts[0].src->type, parts[0].src->ne[0], rows);
        ggml_set_name(t, name);
        groups.push_back({ t, std::move(parts) });
        return t;
    };
    auto whole = [](const ggml_tensor * t) {
        return FusePart{ t, 0, t->ne[1] };
    };
    auto fuse_attn = [&](WhistleAttn & a, bool with_kv, const std::string & name) {
        std::vector<FusePart> parts = { whole(a.q) };
        if (with_kv) {
            parts.push_back(whole(a.k));
            parts.push_back(whole(a.v));
        }
        parts.push_back(whole(a.gate));
        a.fused = add(parts, name.c_str());
        if (a.fused != nullptr) {
            for (const FusePart & fp : parts) {
                skip.push_back(fp.src);
            }
        }
    };
    for (size_t i = 0; i < w.enc_blocks.size(); ++i) {
        fuse_attn(w.enc_blocks[i].attn, true, "enc.blocks." + std::to_string(i) + ".attn.qkvg");
    }
    for (size_t i = 0; i < w.dec_blocks.size(); ++i) {
        fuse_attn(w.dec_blocks[i].attn, true, "dec.blocks." + std::to_string(i) + ".attn.qkvg");
        fuse_attn(w.dec_blocks[i].cross, false, "dec.blocks." + std::to_string(i) + ".cross.qg");
    }
    auto fuse_phi = [&](WhistleMhc & mh, int L, const char * pfx) {
        std::vector<ggml_tensor *> out;
        for (int l = 0; l < L; ++l) {
            ggml_tensor * t = add(
                {
                    { mh.phi_pre,  (int64_t) l * lanes,         lanes                   },
                    { mh.phi_post, (int64_t) l * lanes,         lanes                   },
                    { mh.phi_res,  (int64_t) l * lanes * lanes, (int64_t) lanes * lanes }
            },
                (std::string(pfx) + ".mhc.phi." + std::to_string(l)).c_str());
            if (t == nullptr) {  // all layers or none: the graphs index phi_fused by layer
                groups.resize(groups.size() - out.size());
                return;
            }
            out.push_back(t);
        }
        mh.phi_fused = std::move(out);
        skip.insert(skip.end(), { mh.phi_pre, mh.phi_post, mh.phi_res });
    };
    fuse_phi(w.enc_mhc, m.hparams.enc_n_layers, "enc");
    fuse_phi(w.dec_mhc, m.hparams.dec_n_layers, "dec");
    return TRANSCRIBE_OK;
}

// Linear weights read only as a full mul_mat src0 (fused or not). Weights
// read through a view or get_rows (unfused mHC phi rows, token_embd / the
// tied head, Engram tables) must keep the plain layout.
std::vector<ggml_tensor *> repack_candidates(const WhistleWeights & w) {
    std::vector<ggml_tensor *> out = { w.stem.pw_1, w.stem.pw_2, w.stem.out };
    auto                       lin = [&](const WhistleAttn & a, bool with_kv) {
        if (a.fused != nullptr) {
            out.push_back(a.fused);
            if (!with_kv) {
                out.insert(out.end(), { a.k, a.v });
            }
        } else {
            out.insert(out.end(), { a.q, a.k, a.v, a.gate });
        }
        out.push_back(a.out);
    };
    for (const WhistleEncBlock & b : w.enc_blocks) {
        lin(b.attn, true);
        out.insert(out.end(), { b.conv_pw1, b.conv_pw2 });
    }
    for (const WhistleDecBlock & b : w.dec_blocks) {
        lin(b.attn, true);
        lin(b.cross, false);
    }
    for (const WhistleEngram & e : w.engrams) {
        out.insert(out.end(), { e.key_proj, e.value_proj });
    }
    out.insert(out.end(), w.enc_mhc.phi_fused.begin(), w.enc_mhc.phi_fused.end());
    out.insert(out.end(), w.dec_mhc.phi_fused.begin(), w.dec_mhc.phi_fused.end());
    return out;
}

// GGUF type of every weight whose in-memory type differs from the file's
// (see prepare_cpu_weights).
using DiskTypes = std::unordered_map<const ggml_tensor *, ggml_type>;

// Matmul weights whose Hadamard rotation is folded back out at load (see
// prepare_cpu_weights).
using FoldSet = std::unordered_set<const ggml_tensor *>;

// CPU only, two load-time rewrites of the linear weights:
//
// 1. Low-bit expansion. Q2_K matmul weights, and Q4_K ones in Hadamard-domain
//    files (the small-download format), are expanded to Q8_0. ggml's CPU has
//    no repacked Q2_K kernel on Arm and its Q4_K path is slower than Q8_0's.
// 2. Hadamard fold (Hadamard-domain files). Every matmul weight that reads a
//    hada_in() input is stored C = W blockdiag(H_g); the rotation is applied
//    back to the weights (W = C H_g, H_g involutory) so the graphs skip the
//    per-activation transform (WhistleHada::h stays null). The Engram
//    projections are left rotated: they read Hadamard-domain table rows
//    directly, with no hada_in.
//
// Q8_0 then holds the plain-domain weights to within its own rounding, the
// same representation as a Q8_0 file quantized from F32. Must run before
// plan_fusion (fused tensors take their parts' type).
void prepare_cpu_weights(WhistleModel & m, DiskTypes & disk, FoldSet & fold) {
    if (m.plan.primary_kind != transcribe::BackendKind::Cpu) {
        return;
    }
    WhistleWeights &           w    = m.weights;
    // Linear weights behind a hada_in() input (all Hadamard-domain in HR files).
    std::vector<ggml_tensor *> lin  = { w.stem.pw_1, w.stem.pw_2, w.stem.out, w.token_embd };
    auto                       attn = [&](const WhistleAttn & a) {
        lin.insert(lin.end(), { a.q, a.k, a.v, a.gate, a.out });
    };
    for (const WhistleEncBlock & b : w.enc_blocks) {
        attn(b.attn);
        lin.insert(lin.end(), { b.conv_pw1, b.conv_pw2 });
    }
    for (const WhistleDecBlock & b : w.dec_blocks) {
        attn(b.attn);
        attn(b.cross);
    }
    for (const WhistleMhc * mh : { &w.enc_mhc, &w.dec_mhc }) {
        lin.insert(lin.end(), { mh->phi_pre, mh->phi_post, mh->phi_res });
    }
    const int g = m.hparams.hadamard_group;
    if (g > 0) {
        for (ggml_tensor * t : lin) {
            if (t != nullptr && t->ne[0] % g == 0 &&
                (t->type == GGML_TYPE_F32 || ggml_get_type_traits(t->type)->to_float != nullptr)) {
                fold.insert(t);
            }
        }
        // All or nothing: the graphs drop hada_in for every one of them.
        size_t n_lin = 0;
        for (ggml_tensor * t : lin) {
            n_lin += t != nullptr ? 1 : 0;
        }
        if (fold.size() != n_lin) {
            fold.clear();
        }
    }
    for (const WhistleEngram & e : w.engrams) {
        lin.insert(lin.end(), { e.key_proj, e.value_proj });
    }
    for (ggml_tensor * t : lin) {
        const bool low =
            t != nullptr && (t->type == GGML_TYPE_Q2_K || (t->type == GGML_TYPE_Q4_K && m.hparams.hadamard_group > 0));
        if (!low || t->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) != 0 || disk.count(t) != 0) {
            continue;
        }
        disk[t]  = t->type;
        t->type  = GGML_TYPE_Q8_0;
        t->nb[0] = ggml_type_size(t->type);
        t->nb[1] = ggml_row_size(t->type, t->ne[0]);
        for (int i = 2; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
        }
    }
}

// In-place orthonormal fast Walsh-Hadamard transform of every g-block of x
// (Sylvester order, scaled by 1/sqrt(g)): x_b <- H_g x_b, the same H_g the
// graphs apply (hada_in). Double accumulation; load time only.
void fwht_blocks(float * x, int64_t n, int g) {
    std::vector<double> y(static_cast<size_t>(g));
    const double        s = 1.0 / std::sqrt(static_cast<double>(g));
    for (int64_t b = 0; b + g <= n; b += g) {
        for (int i = 0; i < g; ++i) {
            y[static_cast<size_t>(i)] = x[b + i];
        }
        for (int h = 1; h < g; h *= 2) {
            for (int i = 0; i < g; i += 2 * h) {
                for (int j = i; j < i + h; ++j) {
                    const double u = y[static_cast<size_t>(j)], v = y[static_cast<size_t>(j + h)];
                    y[static_cast<size_t>(j)]     = u + v;
                    y[static_cast<size_t>(j + h)] = u - v;
                }
            }
        }
        for (int i = 0; i < g; ++i) {
            x[b + i] = static_cast<float>(y[static_cast<size_t>(i)] * s);
        }
    }
}

// Whether the CPU device runs mul_mat(w, x) with w in `buft` (same probe as
// llama.cpp: a dummy op whose src0 lives in a zero-size buffer of that type).
bool buft_supports_matmul(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft, const ggml_tensor * w) {
    ggml_init_params p{};
    p.mem_size         = 4 * ggml_tensor_overhead();
    p.no_alloc         = true;
    ggml_context * ctx = ggml_init(p);
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor *         wc  = ggml_dup_tensor(ctx, w);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, 0);
    wc->buffer                = buf;
    ggml_tensor * op          = ggml_mul_mat(ctx, wc, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], 8));
    const bool    ok          = buf != nullptr && ggml_backend_dev_supports_op(dev, op);
    if (buf != nullptr) {
        safe_buffer_free(buf);
    }
    ggml_free(ctx);
    return ok;
}

// Allocates the weights (ctx_meta minus `skip`, plus ctx_fused): repackable
// linears in a CPU_REPACK buffer (CPU primary only), everything else in the
// primary backend's buffer. Uploads into the repack buffer are repacked on
// the fly by ggml_backend_tensor_set.
transcribe_status alloc_weights(WhistleModel & m, const std::vector<const ggml_tensor *> & skip) {
    std::vector<ggml_tensor *> all;
    for (ggml_context * c : { m.ctx_meta, m.ctx_fused }) {
        for (ggml_tensor * t = ggml_get_first_tensor(c); t != nullptr; t = ggml_get_next_tensor(c, t)) {
            if (std::find(skip.begin(), skip.end(), t) == skip.end()) {
                all.push_back(t);
            }
        }
    }
    ggml_backend_dev_t         dev = nullptr;
    ggml_backend_buffer_type_t rp =
        m.plan.primary_kind == transcribe::BackendKind::Cpu ? cpu_repack_buft(dev) : nullptr;
    std::vector<ggml_tensor *> repack;
    if (rp != nullptr) {
        for (ggml_tensor * t : repack_candidates(m.weights)) {
            if (t != nullptr && std::find(all.begin(), all.end(), t) != all.end() && buft_supports_matmul(dev, rp, t)) {
                repack.push_back(t);
            }
        }
    }
    auto is_repack = [&](const ggml_tensor * t) {
        return std::find(repack.begin(), repack.end(), t) != repack.end();
    };
    ggml_backend_buffer_type_t main_buft = ggml_backend_get_default_buffer_type(m.plan.primary);
    size_t                     main_size = 0, rp_size = 0;
    for (ggml_tensor * t : all) {
        ggml_backend_buffer_type_t b = is_repack(t) ? rp : main_buft;
        (is_repack(t) ? rp_size : main_size) +=
            GGML_PAD(ggml_backend_buft_get_alloc_size(b, t), ggml_backend_buft_get_alignment(b));
    }
    m.backend_buffer = ggml_backend_buft_alloc_buffer(main_buft, main_size);
    if (m.backend_buffer == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m.backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_tallocr ta_main = ggml_tallocr_new(m.backend_buffer);
    ggml_tallocr ta_rp{};
    if (!repack.empty()) {
        m.repack_buffer = ggml_backend_buft_alloc_buffer(rp, rp_size);
        if (m.repack_buffer == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(m.repack_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ta_rp = ggml_tallocr_new(m.repack_buffer);
    }
    for (ggml_tensor * t : all) {
        if (ggml_tallocr_alloc(is_repack(t) ? &ta_rp : &ta_main, t) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_OOM;
        }
    }
    if (!repack.empty()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "whistle: %zu linear weights in the CPU repack layout", repack.size());
    }
    return TRANSCRIBE_OK;
}

// Uploads tensor data from the GGUF: every allocated ctx_meta tensor, and
// each fused tensor assembled from its parts' rows.
transcribe_status stream_weights(WhistleModel &                           m,
                                 const std::string &                      path,
                                 const gguf_context *                     gguf_data,
                                 const std::vector<FuseGroup> &           groups,
                                 const std::vector<const ggml_tensor *> & skip,
                                 const DiskTypes &                        disk,
                                 const FoldSet &                          fold) {
    std::ifstream fin(transcribe::path_from_utf8(path), std::ios::binary);
    if (!fin) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: failed to reopen %s for tensor data", path.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    const size_t         data_offset = gguf_get_data_offset(gguf_data);
    std::vector<uint8_t> staging, raw;
    std::vector<float>   f32;
    auto                 read_raw = [&](const ggml_tensor * t, size_t off, size_t n, uint8_t * dst) {
        const int64_t idx = gguf_find_tensor(gguf_data, t->name);
        if (idx < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: tensor \"%s\" not in gguf data", t->name);
            return false;
        }
        fin.seekg(static_cast<std::streamoff>(data_offset + gguf_get_tensor_offset(gguf_data, idx) + off));
        fin.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(n));
        if (!fin) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: short read for tensor \"%s\"", t->name);
            return false;
        }
        return true;
    };
    // Rows [row0, row0 + rows) of t in t's in-memory type (expanding from the
    // file's type when prepare_cpu_weights changed it, and folding the
    // Hadamard rotation for the tensors in `fold`).
    auto read = [&](const ggml_tensor * t, int64_t row0, int64_t rows, uint8_t * dst) {
        const auto      it     = disk.find(t);
        const bool      folded = fold.count(t) != 0;
        const ggml_type ftype  = it == disk.end() ? t->type : it->second;
        const size_t    row    = ggml_row_size(ftype, t->ne[0]);
        if (it == disk.end() && !folded) {
            return read_raw(t, static_cast<size_t>(row0) * row, static_cast<size_t>(rows) * row, dst);
        }
        raw.resize(static_cast<size_t>(rows) * row);
        if (!read_raw(t, static_cast<size_t>(row0) * row, raw.size(), raw.data())) {
            return false;
        }
        const int64_t n = rows * t->ne[0];
        f32.resize(static_cast<size_t>(n));
        if (ftype == GGML_TYPE_F32) {
            std::memcpy(f32.data(), raw.data(), static_cast<size_t>(n) * sizeof(float));
        } else {
            ggml_get_type_traits(ftype)->to_float(raw.data(), f32.data(), n);
        }
        if (folded) {
            fwht_blocks(f32.data(), n, m.hparams.hadamard_group);
        }
        ggml_quantize_chunk(t->type, f32.data(), dst, 0, rows, t->ne[0], nullptr);
        return true;
    };
    for (ggml_tensor * t = ggml_get_first_tensor(m.ctx_meta); t != nullptr; t = ggml_get_next_tensor(m.ctx_meta, t)) {
        if (std::find(skip.begin(), skip.end(), t) != skip.end()) {
            continue;
        }
        const size_t nbytes = ggml_nbytes(t);
        staging.resize(std::max(staging.size(), nbytes));
        if (!read(t, 0, ggml_nrows(t), staging.data())) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
    }
    for (const FuseGroup & g : groups) {
        const size_t nbytes = ggml_nbytes(g.fused);
        staging.resize(std::max(staging.size(), nbytes));
        size_t pos = 0;
        for (const FusePart & fp : g.parts) {
            const size_t row = ggml_row_size(fp.src->type, fp.src->ne[0]);
            if (!read(fp.src, fp.row0, fp.rows, staging.data() + pos)) {
                return TRANSCRIBE_ERR_GGUF;
            }
            pos += static_cast<size_t>(fp.rows) * row;
        }
        ggml_backend_tensor_set(g.fused, staging.data(), 0, nbytes);
    }
    return TRANSCRIBE_OK;
}

// CPU repack: the tied head [d, vocab] has a row count (8199) the repacked
// kernels cannot take, so it runs on the generic path. Build a copy padded
// with zero rows to a multiple of 8 in the repacked layout (zero bytes are a
// zero row in every supported type). token_embd stays for the embedding lookup.
void build_padded_head(WhistleModel & m) {
    ggml_backend_dev_t         dev = nullptr;
    ggml_backend_buffer_type_t rp =
        m.plan.primary_kind == transcribe::BackendKind::Cpu ? cpu_repack_buft(dev) : nullptr;
    const ggml_tensor * te = m.weights.token_embd;
    if (rp == nullptr || te == nullptr || te->ne[1] % 8 == 0) {
        return;
    }
    ggml_init_params p{};
    p.mem_size = 2 * ggml_tensor_overhead();
    p.no_alloc = true;
    m.ctx_head = ggml_init(p);
    if (m.ctx_head == nullptr) {
        return;
    }
    ggml_tensor * hd = ggml_new_tensor_2d(m.ctx_head, te->type, te->ne[0], (te->ne[1] + 7) / 8 * 8);
    ggml_set_name(hd, "dec.head_rp");
    if (!buft_supports_matmul(dev, rp, hd)) {
        return;
    }
    m.head_buffer = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_head, rp);
    if (m.head_buffer == nullptr) {
        return;
    }
    ggml_backend_buffer_set_usage(m.head_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> bytes(ggml_nbytes(hd), 0);
    ggml_backend_tensor_get(te, bytes.data(), 0, ggml_nbytes(te));
    ggml_backend_tensor_set(hd, bytes.data(), 0, bytes.size());
    m.weights.head_rp = hd;
}

transcribe_status build_aux(WhistleModel & m) {
    const WhistleHParams & hp    = m.hparams;
    const WhistleWeights & w     = m.weights;
    WhistleAux &           aux   = m.aux;
    const int              lanes = hp.mhc_lanes;

    // Every HadamardMLP, for the transposed weight copies below.
    std::vector<WhistleHmlp *> hmlps;
    for (WhistleEncBlock & b : m.weights.enc_blocks) {
        hmlps.push_back(&b.hmlp_0);
        hmlps.push_back(&b.hmlp);
    }
    for (WhistleDecBlock & b : m.weights.dec_blocks) {
        hmlps.push_back(&b.hmlp);
    }

    ggml_init_params p{};
    p.mem_size = (16 + 8 * hmlps.size()) * ggml_tensor_overhead();
    p.no_alloc = true;
    aux.ctx    = ggml_init(p);
    if (aux.ctx == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    aux.perm1         = ggml_new_tensor_1d(aux.ctx, GGML_TYPE_I32, hp.hada_n);
    aux.perm2         = ggml_new_tensor_1d(aux.ctx, GGML_TYPE_I32, hp.hada_n);
    aux.enc_pre_bias  = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, lanes, hp.enc_n_layers);
    aux.enc_post_bias = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, lanes, hp.enc_n_layers);
    aux.dec_pre_bias  = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, lanes, hp.dec_n_layers);
    aux.dec_post_bias = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, lanes, hp.dec_n_layers);
    // Folded weights (prepare_cpu_weights) take plain inputs: no rotation.
    const int g       = m.hada_folded ? 0 : hp.hadamard_group;
    if (g > 0) {
        m.weights.hada.h    = ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, g, g);
        m.weights.hada.fwht = aux.cpu_fused_ops;
    }
    auto new_t = [&](const ggml_tensor * src) {
        return ggml_new_tensor_2d(aux.ctx, GGML_TYPE_F32, src->ne[1], src->ne[0]);
    };
    for (WhistleHmlp * h : hmlps) {
        h->cond_v_t = new_t(h->cond_v);
        h->cond_u_t = new_t(h->cond_u);
        for (int k = 0; k < 3; ++k) {
            h->wa_t[k] = new_t(h->wa[k]);
            h->wb_t[k] = new_t(h->wb[k]);
        }
    }
    aux.buffer = ggml_backend_alloc_ctx_tensors(aux.ctx, m.plan.primary);
    if (aux.buffer == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }

    if (g > 0) {  // Sylvester Walsh-Hadamard: H[i][j] = (-1)^popcount(i & j) / sqrt(g)
        std::vector<float> hv(static_cast<size_t>(g) * g);
        const float        s = 1.0f / std::sqrt(static_cast<float>(g));
        for (int i = 0; i < g; ++i) {
            for (int j = 0; j < g; ++j) {
                unsigned bits = static_cast<unsigned>(i & j), parity = 0;
                for (; bits != 0; bits &= bits - 1) {
                    parity ^= 1u;
                }
                hv[static_cast<size_t>(i) * g + j] = parity != 0 ? -s : s;
            }
        }
        ggml_backend_tensor_set(m.weights.hada.h, hv.data(), 0, hv.size() * sizeof(float));
    }

    for (int k = 0; k < 2; ++k) {
        const std::vector<float> pf = read_host(k == 0 ? w.hada_perm1 : w.hada_perm2);
        std::vector<int32_t>     pi(pf.size());
        for (size_t i = 0; i < pf.size(); ++i) {
            pi[i] = static_cast<int32_t>(std::lround(pf[i]));
            if (pi[i] < 0 || pi[i] >= hp.hada_n) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: hada.perm%d entry %zu out of range", k + 1, i);
                return TRANSCRIBE_ERR_GGUF;
            }
        }
        ggml_backend_tensor_set(k == 0 ? aux.perm1 : aux.perm2, pi.data(), 0, pi.size() * sizeof(int32_t));
    }

    auto mhc_host = [&](const WhistleMhc & mh, int L, ggml_tensor * pre_t, ggml_tensor * post_t,
                        std::vector<float> & a_pre, std::vector<float> & a_post, std::vector<float> & a_res) {
        a_pre                          = read_host(mh.a_pre);
        a_post                         = read_host(mh.a_post);
        a_res                          = read_host(mh.a_res);
        const std::vector<float> bpre  = read_host(mh.b_pre);
        const std::vector<float> bpost = read_host(mh.b_post);
        std::vector<float>       pre(bpre.size()), post(bpost.size());
        for (int l = 0; l < L; ++l) {
            for (int n = 0; n < lanes; ++n) {
                const float lane                        = (l % lanes) == n ? 1.0f : 0.0f;
                pre[static_cast<size_t>(l) * lanes + n] = bpre[static_cast<size_t>(l) * lanes + n] + 8.0f * lane - 4.0f;
                post[static_cast<size_t>(l) * lanes + n] =
                    bpost[static_cast<size_t>(l) * lanes + n] - 4.0f * (1.0f - lane);
            }
        }
        ggml_backend_tensor_set(pre_t, pre.data(), 0, pre.size() * sizeof(float));
        ggml_backend_tensor_set(post_t, post.data(), 0, post.size() * sizeof(float));
        return std::make_pair(pre, post);
    };
    const auto enc_bias = mhc_host(w.enc_mhc, hp.enc_n_layers, aux.enc_pre_bias, aux.enc_post_bias, aux.enc_a_pre,
                                   aux.enc_a_post, aux.enc_a_res);
    const auto dec_bias = mhc_host(w.dec_mhc, hp.dec_n_layers, aux.dec_pre_bias, aux.dec_post_bias, aux.dec_a_pre,
                                   aux.dec_a_post, aux.dec_a_res);

    for (const WhistleEncBlock & b : w.enc_blocks) {
        aux.enc_attn_gate.push_back(sigmoidf(read_host(b.attn_gate)[0]));
    }
    for (const WhistleDecBlock & b : w.dec_blocks) {
        aux.dec_attn_gate.push_back(sigmoidf(read_host(b.attn_gate)[0]));
        aux.dec_cross_gate.push_back(sigmoidf(read_host(b.cross_gate)[0]));
    }
    aux.pe_gate = sigmoidf(read_host(w.pe_gate)[0]);

    // HadamardMLP graphs contract against transposed weights; transpose once.
    auto set_transposed = [&](ggml_tensor * src, ggml_tensor * dst) {
        const std::vector<float> v  = read_host(src);
        const int64_t            n0 = src->ne[0], n1 = src->ne[1];
        std::vector<float>       t(v.size());
        for (int64_t i1 = 0; i1 < n1; ++i1) {
            for (int64_t i0 = 0; i0 < n0; ++i0) {
                t[static_cast<size_t>(i0 * n1 + i1)] = v[static_cast<size_t>(i1 * n0 + i0)];
            }
        }
        ggml_backend_tensor_set(dst, t.data(), 0, t.size() * sizeof(float));
    };
    for (WhistleHmlp * h : hmlps) {
        set_transposed(h->cond_v, h->cond_v_t);
        set_transposed(h->cond_u, h->cond_u_t);
        for (int k = 0; k < 3; ++k) {
            set_transposed(h->wa[k], h->wa_t[k]);
            set_transposed(h->wb[k], h->wb_t[k]);
        }
    }

    // ZCRMSNorm multiplies by (1 + scale): fold the +1 into every norm weight
    // once here instead of recomputing it in each graph (float x + 1.0f is
    // the same rounding as ggml_scale_bias(x, 1, 1)).
    std::vector<ggml_tensor *> norms = { w.enc_final_norm, w.dec_final_norm };
    for (const WhistleEncBlock & b : w.enc_blocks) {
        norms.insert(norms.end(), { b.norm_hmlp_0, b.norm_in, b.norm_post_attn, b.norm_conv, b.norm_conv_out,
                                    b.norm_hmlp, b.attn.q_norm, b.attn.k_norm });
    }
    for (const WhistleDecBlock & b : w.dec_blocks) {
        norms.insert(norms.end(), { b.norm_in, b.norm_post_attn, b.norm_hmlp, b.norm_cross, b.norm_post_cross,
                                    b.attn.q_norm, b.attn.k_norm, b.cross.q_norm, b.cross.k_norm });
    }
    for (ggml_tensor * t : norms) {
        std::vector<float> v = read_host(t);
        for (float & x : v) {
            x += 1.0f;
        }
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    }

    // Fused CPU mHC: per-layer scalars and host bias vectors.
    if (aux.cpu_fused_ops) {
        const std::vector<float> enc_bres = read_host(w.enc_mhc.b_res);
        const std::vector<float> dec_bres = read_host(w.dec_mhc.b_res);
        std::vector<float> &     hb       = aux.mhc_bias_host;
        hb.clear();
        for (const std::vector<float> * v :
             { &enc_bias.first, &enc_bias.second, &enc_bres, &dec_bias.first, &dec_bias.second, &dec_bres }) {
            hb.insert(hb.end(), v->begin(), v->end());
        }
        const float * base = hb.data();
        auto          fill = [&](std::vector<MhcHost> & out, int L, const std::vector<float> & a_pre,
                                 const std::vector<float> & a_post, const std::vector<float> & a_res) {
            const float * pre  = base;
            const float * post = pre + static_cast<size_t>(L) * lanes;
            const float * bres = post + static_cast<size_t>(L) * lanes;
            out.resize(static_cast<size_t>(L));
            for (int l = 0; l < L; ++l) {
                MhcHost & mh = out[static_cast<size_t>(l)];
                mh.d         = hp.d_model;
                mh.lanes     = lanes;
                mh.a_pre     = a_pre[static_cast<size_t>(l)];
                mh.a_post    = a_post[static_cast<size_t>(l)];
                mh.a_res     = a_res[static_cast<size_t>(l)];
                mh.pre_bias  = pre + static_cast<size_t>(l) * lanes;
                mh.post_bias = post + static_cast<size_t>(l) * lanes;
                mh.b_res     = bres + static_cast<size_t>(l) * lanes * lanes;
            }
            base = bres + static_cast<size_t>(L) * lanes * lanes;
        };
        fill(aux.enc_mhc_host, hp.enc_n_layers, aux.enc_a_pre, aux.enc_a_post, aux.enc_a_res);
        fill(aux.dec_mhc_host, hp.dec_n_layers, aux.dec_a_pre, aux.dec_a_post, aux.dec_a_res);
    }

    // Fused CPU HadamardMLP: weights live in host memory on the CPU backend.
    if (aux.cpu_fused_ops) {
        for (int k = 0; k < 2; ++k) {
            const std::vector<float> pf = read_host(k == 0 ? w.hada_perm1 : w.hada_perm2);
            std::vector<int32_t> &   pi = k == 0 ? aux.perm1_host : aux.perm2_host;
            pi.resize(pf.size());
            for (size_t i = 0; i < pf.size(); ++i) {
                pi[i] = static_cast<int32_t>(std::lround(pf[i]));
            }
        }
        auto f32 = [](const ggml_tensor * t) {
            return static_cast<const float *>(t->data);
        };
        aux.hmlp_host.resize(hmlps.size());
        for (size_t i = 0; i < hmlps.size(); ++i) {
            const WhistleHmlp & h  = *hmlps[i];
            HmlpHost &          hh = aux.hmlp_host[i];
            hh.d                   = hp.hada_n;
            hh.ba                  = static_cast<int>(h.wa[0]->ne[0]);
            hh.bb                  = static_cast<int>(h.wb[0]->ne[0]);
            hh.d1                  = f32(h.d1);
            hh.d2                  = f32(h.d2);
            hh.b2                  = f32(h.b2);
            hh.d3                  = f32(h.d3);
            hh.d4                  = f32(h.d4);
            for (int k = 0; k < 3; ++k) {
                hh.a[k] = f32(h.wa[k]);
                hh.b[k] = f32(h.wb[k]);
            }
            hh.cond_v      = f32(h.cond_v);
            hh.cond_u      = f32(h.cond_u);
            hh.perm1       = aux.perm1_host.data();
            hh.perm2       = aux.perm2_host.data();
            hmlps[i]->host = &hh;
        }
    }
    return TRANSCRIBE_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// load / init_context
// ---------------------------------------------------------------------------

namespace {

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<WhistleModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? std::string(k_default_variant) : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_whistle_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    if (m->tok.eos_id() < 0 || m->tok.bos_id() < 0 || m->tok.n_tokens() != m->hparams.vocab_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: tokenizer special ids / vocab size inconsistent with hparams");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Hard input cap: the engine rejects > 30 s (max_audio_samples).
    m->caps.max_audio_ms = static_cast<int64_t>(m->hparams.max_audio_samples) * 1000 / m->hparams.fe_sample_rate;

    // Decoder context bounds the transcript, not the audio (audio lives in
    // cross-KV), so the audio limit comes from caps.
    m->limits.has_context_cap    = true;
    m->limits.audio_from_caps    = true;
    m->limits.model_max_ctx      = m->hparams.dec_max_seq;
    m->limits.prompt_overhead    = k_prompt_len;
    m->limits.gen_reserve        = m->hparams.dec_max_seq - k_prompt_len;
    m->limits.ms_per_audio_token = 8.0 * m->hparams.fe_hop_length * 1000.0 / m->hparams.fe_sample_rate;
    m->limits.kv_elems_per_ctx_token =
        (int64_t) m->hparams.n_kv_heads * (m->hparams.qk_head_dim + m->hparams.v_head_dim) * m->hparams.dec_n_layers;

    // Filterbank stays on the host (the frontend is host-side).
    {
        using R               = load_common::ReadF32Result;
        const size_t fb_elems = static_cast<size_t>(m->hparams.fe_num_mels) * (m->hparams.fe_n_fft / 2 + 1);
        const auto   rc = load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.mel_filterbank",
                                                               fb_elems, "whistle", m->aux.mel_fb);
        if (rc != R::Ok) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: frontend.mel_filterbank missing or malformed");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    gguf_init_params init_params{};
    init_params.no_alloc     = true;
    init_params.ctx          = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (const transcribe_status st = build_whistle_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    const transcribe_device_t  device      = (params != nullptr) ? params->device : nullptr;
    // AUTO -> CPU: the CPU path (repacked weights + fused custom ops) is ~5x
    // faster than Metal for this 55M-parameter beam-search model (M4, jfk
    // Q8_0: 56 ms CPU vs 300 ms Metal). An explicit backend or device request
    // is honored as given.
    if (backend_req == TRANSCRIBE_BACKEND_AUTO && device == nullptr) {
        backend_req = TRANSCRIBE_BACKEND_CPU;
    }
    if (const transcribe_status st = transcribe::load_common::init_backends(backend_req, device, "whistle", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    // Fused CPU custom ops read weights through host pointers (CPU buffers
    // only) and keep per-token lane scratch on the stack (<= 16 lanes).
    m->aux.cpu_fused_ops = (m->plan.primary_kind == transcribe::BackendKind::Cpu ||
                            m->plan.primary_kind == transcribe::BackendKind::Accel) &&
                           m->hparams.mhc_lanes <= 16;
    m->backend           = ggml_backend_name(m->plan.primary);
    m->primary_backend   = m->plan.primary;

    DiskTypes disk_types;
    FoldSet   fold;
    prepare_cpu_weights(*m, disk_types, fold);
    m->hada_folded = !fold.empty();
    std::vector<FuseGroup>           fuse_groups;
    std::vector<const ggml_tensor *> fuse_skip;
    if (plan_fusion(*m, fuse_groups, fuse_skip) != TRANSCRIBE_OK || alloc_weights(*m, fuse_skip) != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: weight buffer allocation failed");
        return TRANSCRIBE_ERR_OOM;
    }
    if (const transcribe_status st =
            stream_weights(*m, loader.path(), gguf_data, fuse_groups, fuse_skip, disk_types, fold);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    build_padded_head(*m);
    if (const transcribe_status st = build_aux(*m); st != TRANSCRIBE_OK) {
        return st;
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto cc       = std::make_unique<WhistleSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads > 0 ? params->n_threads : whistle_default_threads();
    cc->kv_type   = params->kv_type;
    cc->n_ctx     = transcribe_session_params_n_ctx(params);
    *out_ctx      = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Graph execution helpers
// ---------------------------------------------------------------------------

bool new_compute_ctx(WhistleSession * cc, size_t mem) {
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    if (cc->graph_meta.size() < mem) {
        cc->graph_meta.assign(mem, 0);
    }
    ggml_init_params p{};
    p.mem_size      = cc->graph_meta.size();
    p.mem_buffer    = cc->graph_meta.data();
    p.no_alloc      = true;
    cc->compute_ctx = ggml_init(p);
    return cc->compute_ctx != nullptr;
}

transcribe_status ensure_sched(WhistleSession * cc, WhistleModel * cm) {
    if (cc->sched == nullptr) {
        cc->sched =
            ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                   static_cast<int>(cm->plan.scheduler_list.size()), k_sched_nodes, false, true);
        if (cc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    return TRANSCRIBE_OK;
}

// Attaches a persistent worker pool sized to the session's thread count to
// the CPU primary backend (no-op on other backends).
void ensure_threadpool(WhistleSession * cc, WhistleModel * cm) {
    // 0 = library default; resolve it the way the sched thread count is.
    const int n = transcribe::configure_sched_n_threads(nullptr, cc->n_threads);
    if (cm->plan.primary_kind != transcribe::BackendKind::Cpu || n <= 1) {
        return;
    }
    if (cm->threadpool != nullptr && cm->threadpool_threads == n) {
        return;
    }
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n);
    ggml_threadpool_t      tp  = ggml_threadpool_new(&tpp);
    if (tp == nullptr) {
        return;  // fall back to per-compute pools
    }
    ggml_backend_cpu_set_threadpool(cm->plan.primary, tp);
    if (cm->threadpool != nullptr) {
        ggml_threadpool_free(cm->threadpool);
    }
    cm->threadpool         = tp;
    cm->threadpool_threads = n;
}

transcribe_status compute_graph(WhistleSession * cc, ggml_cgraph * gf, const char * what) {
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, gf)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: %s graph allocation failed - out of memory", what);
        return TRANSCRIBE_ERR_OOM;
    }
    return TRANSCRIBE_OK;
}

transcribe_status run_graph(WhistleSession * cc, ggml_cgraph * gf, const char * what) {
    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
    if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, gf); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: %s compute failed (%d)", what, static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// Encoder for one utterance: mel (already computed) -> host [T_enc][d].
transcribe_status encode(WhistleSession *     cc,
                         WhistleModel *       cm,
                         const MelResult &    mel,
                         std::vector<float> & enc_out,
                         int &                T_enc) {
    const bool dumps = transcribe::debug::enabled();
    if (!new_compute_ctx(cc, k_graph_mem)) {
        return TRANSCRIBE_ERR_OOM;
    }
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights, cm->hparams, cm->aux, mel.n_frames, dumps);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (auto st = compute_graph(cc, eb.graph, "encoder"); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_backend_tensor_set(eb.mel_in, mel.mel.data(), 0, mel.mel.size() * sizeof(float));
    std::vector<int32_t> pos(static_cast<size_t>(eb.T_enc));
    for (int i = 0; i < eb.T_enc; ++i) {
        pos[i] = i;
    }
    if (eb.pos_in->buffer != nullptr) {
        ggml_backend_tensor_set(eb.pos_in, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    if (auto st = run_graph(cc, eb.graph, "encoder"); st != TRANSCRIBE_OK) {
        return st;
    }
    if (dumps) {
        const long long mshape[2] = { mel.n_frames, cm->hparams.fe_num_mels };
        transcribe::debug::dump_host_f32("enc.mel.in", mel.mel.data(), static_cast<long long>(mel.mel.size()), mshape,
                                         2, "encoder.mel");
        transcribe::debug::dump_tensor("enc.stem.out", eb.dumps.stem_out, "encoder.stem");
        for (size_t i = 0; i < eb.dumps.block_out.size(); ++i) {
            char nm[48];
            std::snprintf(nm, sizeof(nm), "enc.blk.%zu.out", i);
            transcribe::debug::dump_tensor(nm, eb.dumps.block_out[i], "encoder.block");
        }
        transcribe::debug::dump_tensor("enc.final", eb.out, "encoder.final");
    }
    T_enc = eb.T_enc;
    enc_out.resize(static_cast<size_t>(T_enc) * cm->hparams.d_model);
    ggml_backend_tensor_get(eb.out, enc_out.data(), 0, enc_out.size() * sizeof(float));
    return TRANSCRIBE_OK;
}

// mem = enc + sigmoid(pe_gate) * [sin(p w_i), cos(p w_i)], w_i = 10000^(-2i/d).
void build_memory(const WhistleModel & cm, const std::vector<float> & enc, int T, std::vector<float> & mem) {
    const int d = cm.hparams.d_model;
    const int h = d / 2;
    mem         = enc;
    for (int t = 0; t < T; ++t) {
        float * row = mem.data() + static_cast<size_t>(t) * d;
        for (int i = 0; i < h; ++i) {
            const double w = std::pow(10000.0, -2.0 * i / d);
            row[i] += cm.aux.pe_gate * static_cast<float>(std::sin(t * w));
            row[h + i] += cm.aux.pe_gate * static_cast<float>(std::cos(t * w));
        }
    }
}

transcribe_status fill_cross(WhistleSession *           cc,
                             WhistleModel *             cm,
                             int                        utt,
                             const std::vector<float> & mem,
                             int                        T_utt) {
    if (!new_compute_ctx(cc, 8u * 1024u * 1024u)) {
        return TRANSCRIBE_ERR_OOM;
    }
    CrossKvBuild cb = build_cross_kv_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->cache, utt, T_utt);
    if (auto st = compute_graph(cc, cb.graph, "cross_kv"); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_backend_tensor_set(cb.mem_in, mem.data(), 0, mem.size() * sizeof(float));
    return run_graph(cc, cb.graph, "cross_kv");
}

// One decode step for every slot. hist[s] holds tokens[0..pos] of slot s.
// Returns logits [n_seq][vocab]; optionally the cross-attention probabilities
// of the alignment layers (the second half of the decoder) for every slot:
// per layer l >= L/2 one block laid out [T_enc, beam, n_heads, n_utt].
transcribe_status decode_step(WhistleSession *                      cc,
                              WhistleModel *                        cm,
                              int                                   pos,
                              const std::vector<std::vector<int>> & hist,
                              const std::vector<int> &              utt_T,
                              std::vector<float> &                  logits,
                              std::vector<float> *                  xattn) {
    const WhistleHParams & hp = cm->hparams;
    const int              S  = cc->cache.n_seq;
    if (!new_compute_ctx(cc, k_graph_mem)) {
        return TRANSCRIBE_ERR_OOM;
    }
    StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, hp, cm->aux, cc->cache, pos, xattn != nullptr);
    if (auto st = compute_graph(cc, sb.graph, "decoder step"); st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<int32_t> tok(S), posv(S, pos), rows(static_cast<size_t>(hp.engram_n_tables) * S);
    for (int s = 0; s < S; ++s) {
        tok[s] = hist[s][static_cast<size_t>(pos)];
        engram_rows(hist[s], pos, hp, rows.data() + static_cast<size_t>(s) * hp.engram_n_tables);
    }
    ggml_backend_tensor_set(sb.tok_in, tok.data(), 0, tok.size() * sizeof(int32_t));
    ggml_backend_tensor_set(sb.pos_in, posv.data(), 0, posv.size() * sizeof(int32_t));
    if (sb.eg_lo_in != nullptr) {  // paired Engram tables
        std::vector<int32_t> lo(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            lo[i] = rows[i] & 1;
            rows[i] >>= 1;
        }
        ggml_backend_tensor_set(sb.eg_lo_in, lo.data(), 0, lo.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(sb.eg_idx_in, rows.data(), 0, rows.size() * sizeof(int32_t));
    std::vector<float> egm(static_cast<size_t>(hp.engram_n_tables));
    for (size_t oi = 0, r = 0; oi < hp.engram_orders.size(); ++oi) {
        for (int h = 0; h < hp.engram_heads(); ++h, ++r) {
            egm[r] = pos >= hp.engram_orders[oi] - 1 ? 1.0f : 0.0f;
        }
    }
    ggml_backend_tensor_set(sb.eg_mask_in, egm.data(), 0, egm.size() * sizeof(float));
    if (sb.xmask_in != nullptr) {
        const int          T       = cc->cache.T_enc;
        const int          per_utt = S / cc->cache.n_utt;
        std::vector<float> xm(static_cast<size_t>(T) * S, 0.0f);
        for (int s = 0; s < S; ++s) {
            const int Tu = utt_T[static_cast<size_t>(s / per_utt)];
            for (int t = Tu; t < T; ++t) {
                xm[static_cast<size_t>(s) * T + t] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(sb.xmask_in, xm.data(), 0, xm.size() * sizeof(float));
    }
    if (auto st = run_graph(cc, sb.graph, "decoder step"); st != TRANSCRIBE_OK) {
        return st;
    }
    logits.resize(static_cast<size_t>(hp.vocab_size) * S);
    ggml_backend_tensor_get(sb.logits, logits.data(), 0, logits.size() * sizeof(float));
    if (xattn != nullptr) {
        const int    l0  = hp.dec_n_layers / 2;
        const size_t blk = static_cast<size_t>(cc->cache.T_enc) * hp.n_heads * S;
        xattn->resize(static_cast<size_t>(hp.dec_n_layers - l0) * blk);
        for (int l = l0; l < hp.dec_n_layers; ++l) {
            ggml_backend_tensor_get(sb.cross_attn[l], xattn->data() + static_cast<size_t>(l - l0) * blk, 0,
                                    blk * sizeof(float));
        }
    }
    return TRANSCRIBE_OK;
}

// n_pos: self-attention positions written so far (only these are moved).
transcribe_status reorder_cache(WhistleSession * cc, const std::vector<int32_t> & src, int n_pos) {
    bool identity = true;
    for (size_t i = 0; i < src.size(); ++i) {
        identity = identity && src[i] == static_cast<int32_t>(i);
    }
    if (identity) {
        return TRANSCRIBE_OK;
    }
    if (!new_compute_ctx(cc, 4u * 1024u * 1024u)) {
        return TRANSCRIBE_ERR_OOM;
    }
    ReorderBuild rb = build_reorder_graph(cc->compute_ctx, cc->cache, n_pos);
    if (auto st = compute_graph(cc, rb.graph, "beam reorder"); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_backend_tensor_set(rb.src_in, src.data(), 0, src.size() * sizeof(int32_t));
    for (ggml_tensor * t : { rb.src_k, rb.src_v }) {
        std::vector<int32_t> rep(static_cast<size_t>(ggml_nelements(t)));
        for (size_t i = 0; i < rep.size(); ++i) {
            rep[i] = src[i % src.size()];
        }
        ggml_backend_tensor_set(t, rep.data(), 0, rep.size() * sizeof(int32_t));
    }
    return run_graph(cc, rb.graph, "beam reorder");
}

// Log-sum-exp of x[0..n) in double (max-shifted). log p_i = x_i - lse.
double log_sum_exp(const float * x, int n, std::vector<double> & scratch) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        mx = std::max(mx, static_cast<double>(x[i]));
    }
    double sum = 0.0;
#if defined(__APPLE__)
    // Vectorized double exp (vForce); same precision class as std::exp.
    scratch.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        scratch[static_cast<size_t>(i)] = static_cast<double>(x[i]) - mx;
    }
    vvexp(scratch.data(), scratch.data(), &n);
    for (int i = 0; i < n; ++i) {
        sum += scratch[static_cast<size_t>(i)];
    }
#else
    (void) scratch;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(x[i]) - mx);
    }
#endif
    return mx + std::log(sum);
}

// Indices of the k largest x (k <= 16), best first; ties keep the lower
// index first. log-softmax is x - const per row, so this is also the top-k
// of the log-probabilities without materializing them.
int top_k_desc(const float * x, int n, int k, int * idx) {
    float val[16];
    int   m = 0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (m == k && !(v > val[m - 1])) {
            continue;
        }
        int j = m < k ? m++ : m - 1;
        for (; j > 0 && val[j - 1] < v; --j) {
            val[j] = val[j - 1];
            idx[j] = idx[j - 1];
        }
        val[j] = v;
        idx[j] = i;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Beam search (one utterance, or n_utt utterances in lockstep).
// ---------------------------------------------------------------------------

struct Hyp {
    std::vector<int>    toks;  // generated text tokens (no prompt, no eos)
    std::vector<double> lps;   // per-token log-probabilities
    double              score    = 0.0;
    // Finished hypotheses: the decode position whose logits produced </s> and
    // the cache slot holding the hypothesis at that position (XattnCapture).
    int                 fin_pos  = -1;
    int                 fin_slot = -1;
};

// Word-timing capture during the beam search: the alignment layers' cross-
// attention for every slot at every decode position, plus the beam back-
// pointers, so the winning hypothesis' rows are read back without a second
// teacher-forced decode.
struct XattnCapture {
    std::vector<std::vector<float>>   steps;   // [pos] -> decode_step xattn blocks
    std::vector<std::vector<int32_t>> parent;  // [pos] -> slot at pos - 1 of each slot (pos >= 1)
};

// Whisper MaximumLikelihoodRanker with length_penalty=None: rank finished
// hypotheses by sum log p / length. Fitted against the engine on 300
// LibriSpeech rows (raw sum and GNMT alpha 0.6 / 2.0 fit worse).
double length_penalty(int len) {
    return std::max(1, len);
}

struct UttDecode {
    int              lang_token = 0;
    std::vector<Hyp> live;
    std::vector<int> live_slot;  // slot of each live hypothesis
    std::vector<Hyp> finished;
    bool             done       = false;
    bool             hit_budget = false;
};

// The engine decodes with 5 beams (model card; no width knob in its C API).
int beam_width() {
    return k_beam_size;
}

// Runs prompt + beam search for n_utt utterances sharing the cache (slot
// s belongs to utterance s / beam). lang_tokens[u] < 0 = auto-detect.
transcribe_status beam_decode(WhistleSession *         cc,
                              WhistleModel *           cm,
                              const std::vector<int> & utt_T,
                              std::vector<int>         lang_tokens,
                              std::vector<UttDecode> & out,
                              XattnCapture *           cap = nullptr) {
    const WhistleHParams & hp    = cm->hparams;
    const int              n_utt = cc->cache.n_utt;
    const int              beam  = cc->cache.n_seq / n_utt;
    const int              S     = cc->cache.n_seq;
    const int              V     = hp.vocab_size;
    const int              eos   = cm->tok.eos_id();
    const int              bos   = cm->tok.bos_id();
    const int              n_ctx = cc->cache.n_ctx;

    out.assign(static_cast<size_t>(n_utt), UttDecode{});
    std::vector<std::vector<int>> hist(static_cast<size_t>(S), std::vector<int>{ bos });
    std::vector<float>            logits;
    std::vector<double>           lp;

    // pos 0: <s>. Language detection reads these logits.
    std::vector<float> xa;
    auto               record = [&](const std::vector<int32_t> & parent) {
        if (cap != nullptr) {
            cap->steps.push_back(std::move(xa));
            cap->parent.push_back(parent);
            xa.clear();
        }
    };
    std::vector<int32_t> identity(static_cast<size_t>(S));
    for (int s = 0; s < S; ++s) {
        identity[static_cast<size_t>(s)] = s;
    }
    if (auto st = decode_step(cc, cm, 0, hist, utt_T, logits, cap != nullptr ? &xa : nullptr); st != TRANSCRIBE_OK) {
        return st;
    }
    record(identity);
    for (int u = 0; u < n_utt; ++u) {
        if (lang_tokens[u] < 0) {
            const float * row  = logits.data() + static_cast<size_t>(u * beam) * V;
            int           best = hp.lang_token_ids[0];
            for (int32_t id : hp.lang_token_ids) {
                if (row[id] > row[best]) {
                    best = id;
                }
            }
            lang_tokens[u] = best;
        }
        out[u].lang_token = lang_tokens[u];
        for (int k = 0; k < beam; ++k) {
            hist[static_cast<size_t>(u * beam + k)].push_back(lang_tokens[u]);
        }
    }

    // pos 1: <|lang|> -> distribution over the first text token.
    if (auto st = decode_step(cc, cm, 1, hist, utt_T, logits, cap != nullptr ? &xa : nullptr); st != TRANSCRIBE_OK) {
        return st;
    }
    record(identity);
    if (transcribe::debug::enabled()) {
        const long long shape[1] = { V };
        transcribe::debug::dump_host_f32("dec.logits_raw", logits.data(), V, shape, 1, "decoder.logits_raw");
    }
    for (int u = 0; u < n_utt; ++u) {
        out[u].live.assign(1, Hyp{});
        out[u].live_slot.assign(1, u * beam);
    }

    for (int pos = 1;; ++pos) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        // Expand: logits at `pos` predict the token at pos + 1.
        std::vector<int32_t> src(static_cast<size_t>(S));
        std::vector<int>     next_tok(static_cast<size_t>(S), eos);
        bool                 any_live = false;
        for (int u = 0; u < n_utt; ++u) {
            UttDecode & ud = out[u];
            for (int k = 0; k < beam; ++k) {
                src[static_cast<size_t>(u * beam + k)] = u * beam;
            }
            if (ud.done) {
                continue;
            }

            struct Cand {
                double score;
                int    parent;
                int    tok;
                double lp;
            };

            std::vector<Cand> cands;
            for (size_t h = 0; h < ud.live.size(); ++h) {
                const int     slot = ud.live_slot[h];
                const float * row  = logits.data() + static_cast<size_t>(slot) * V;
                const double  lse  = log_sum_exp(row, V, lp);
                int           top[16];
                const int     topk = top_k_desc(row, V, std::min({ V, beam + 1, 16 }), top);
                for (int i = 0; i < topk; ++i) {
                    const double l = static_cast<double>(row[top[i]]) - lse;
                    cands.push_back({ ud.live[h].score + l, static_cast<int>(h), top[i], l });
                }
            }
            std::stable_sort(cands.begin(), cands.end(),
                             [](const Cand & a, const Cand & b) { return a.score > b.score; });

            std::vector<Hyp> next_live;
            std::vector<int> parent_slot;
            std::vector<Hyp> newly_finished;
            for (const Cand & c : cands) {
                const Hyp & par = ud.live[static_cast<size_t>(c.parent)];
                if (c.tok == eos) {
                    Hyp f      = par;
                    f.score    = c.score;
                    f.fin_pos  = pos;
                    f.fin_slot = ud.live_slot[static_cast<size_t>(c.parent)];
                    f.lps.push_back(c.lp);
                    newly_finished.push_back(std::move(f));
                } else {
                    Hyp nh = par;
                    nh.toks.push_back(c.tok);
                    nh.lps.push_back(c.lp);
                    nh.score = c.score;
                    next_live.push_back(std::move(nh));
                    parent_slot.push_back(ud.live_slot[static_cast<size_t>(c.parent)]);
                    if (static_cast<int>(next_live.size()) == beam) {
                        break;
                    }
                }
            }
            for (Hyp & f : newly_finished) {
                if (static_cast<int>(ud.finished.size()) >= beam) {
                    break;
                }
                ud.finished.push_back(std::move(f));
            }
            const bool budget = pos + 1 >= n_ctx;
            if (static_cast<int>(ud.finished.size()) >= beam || budget || next_live.empty()) {
                if (budget && ud.finished.empty()) {
                    ud.hit_budget = true;
                    ud.finished   = next_live;
                }
                ud.done = true;
                continue;
            }
            ud.live = std::move(next_live);
            ud.live_slot.clear();
            for (size_t k = 0; k < ud.live.size(); ++k) {
                const int slot                      = u * beam + static_cast<int>(k);
                src[static_cast<size_t>(slot)]      = parent_slot[k];
                next_tok[static_cast<size_t>(slot)] = ud.live[k].toks.back();
                ud.live_slot.push_back(slot);
            }
            any_live = true;
        }
        if (!any_live) {
            break;
        }

        // Reorder histories + caches, then feed the chosen tokens at pos + 1.
        std::vector<std::vector<int>> new_hist(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s) {
            new_hist[static_cast<size_t>(s)] = hist[static_cast<size_t>(src[static_cast<size_t>(s)])];
            new_hist[static_cast<size_t>(s)].push_back(next_tok[static_cast<size_t>(s)]);
        }
        hist.swap(new_hist);
        if (auto st = reorder_cache(cc, src, pos + 1); st != TRANSCRIBE_OK) {
            return st;
        }
        if (auto st = decode_step(cc, cm, pos + 1, hist, utt_T, logits, cap != nullptr ? &xa : nullptr);
            st != TRANSCRIBE_OK) {
            return st;
        }
        record(src);
    }
    return TRANSCRIBE_OK;
}

const Hyp * best_hyp(const UttDecode & ud) {
    const Hyp * best = nullptr;
    double      bs   = -std::numeric_limits<double>::infinity();
    for (const Hyp & h : ud.finished) {
        const double s = h.score / length_penalty(static_cast<int>(h.toks.size()));
        if (best == nullptr || s > bs) {
            best = &h;
            bs   = s;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Word timestamps: DTW over the chosen hypothesis' cross-attention.
// ---------------------------------------------------------------------------

std::vector<std::pair<int, int>> dtw_path(const std::vector<double> & cost, int N, int M) {
    std::vector<double> D(static_cast<size_t>(N + 1) * (M + 1), std::numeric_limits<double>::infinity());
    std::vector<int8_t> tr(static_cast<size_t>(N + 1) * (M + 1), 0);
    auto                at = [&](int i, int j) -> size_t {
        return static_cast<size_t>(i) * (M + 1) + j;
    };
    D[at(0, 0)] = 0.0;
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= M; ++j) {
            const double c0 = D[at(i - 1, j - 1)], c1 = D[at(i - 1, j)], c2 = D[at(i, j - 1)];
            int          k  = 0;
            double       mn = c0;
            if (c1 < mn) {
                mn = c1;
                k  = 1;
            }
            if (c2 < mn) {
                mn = c2;
                k  = 2;
            }
            D[at(i, j)]  = cost[static_cast<size_t>(i - 1) * M + (j - 1)] + mn;
            tr[at(i, j)] = static_cast<int8_t>(k);
        }
    }
    std::vector<std::pair<int, int>> path;
    int                              i = N, j = M;
    while (i > 0 && j > 0) {
        path.emplace_back(i - 1, j - 1);
        const int k = tr[at(i, j)];
        if (k == 0) {
            --i;
            --j;
        } else if (k == 1) {
            --i;
        } else {
            --j;
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void jumps_from_att(const std::vector<double> & att, int H, int R, int T_enc, std::vector<int> & jumps);

// Alignment recipe, probed on the engine (forward-map Capabilities):
//   - cross-attention of the second half of the decoder layers, all heads
//     (perturbing layers 0-3 never moves the engine's word times; every
//     head of layers 4-7 does);
//   - rows = decoder positions 0 .. n+1 (<s>, then the position predicting
//     each text token, then the one predicting </s>), standardized per head
//     over the row (token) axis, averaged over heads; DTW on -matrix;
//   - token k spans [jump(k), jump(k+1)) where jump(r) is the first frame
//     the path spends on row r; a word ends at the jump after its last
//     non-punctuation token, so trailing punctuation does not swallow the
//     following pause;
//   - the first word cannot start before the speech onset (onset_frame).
// On jfk + 14 LibriSpeech clips this lands 97.8% of word boundaries
// within +-1 frame (96.5% exact); see docs/porting/families/whistle.md.
//
// jumps receives n + 1 entries (one per text token, then the </s> row).
// Also dumps dec.logits_raw.gen8 (teacher-forced) for validation.
transcribe_status align_tokens(WhistleSession *           cc,
                               WhistleModel *             cm,
                               const std::vector<float> & mem,
                               int                        T_enc,
                               int                        lang_token,
                               const std::vector<int> &   toks,
                               std::vector<int> &         jumps) {
    const WhistleHParams & hp = cm->hparams;
    const int              n  = static_cast<int>(toks.size());
    jumps.assign(static_cast<size_t>(n + 1), 0);
    if (!dec_cache_init(cc->cache, cm->plan.primary, hp, hp.dec_max_seq, 1, 1, T_enc)) {
        return TRANSCRIBE_ERR_OOM;
    }
    if (auto st = fill_cross(cc, cm, 0, mem, T_enc); st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<std::vector<int>> hist(1, std::vector<int>{ cm->tok.bos_id(), lang_token });
    hist[0].insert(hist[0].end(), toks.begin(), toks.end());
    const std::vector<int> utt_T{ T_enc };
    std::vector<float>     logits, xa;

    const int           l0 = hp.dec_n_layers / 2;
    const int           H  = (hp.dec_n_layers - l0) * hp.n_heads;
    const int           R  = n + 2;
    std::vector<double> att(static_cast<size_t>(H) * R * T_enc, 0.0);  // [head][row][frame]
    for (int pos = 0; pos < R; ++pos) {
        if (auto st = decode_step(cc, cm, pos, hist, utt_T, logits, &xa); st != TRANSCRIBE_OK) {
            return st;
        }
        if (pos == 9 && transcribe::debug::enabled()) {
            const long long shape[1] = { hp.vocab_size };
            transcribe::debug::dump_host_f32("dec.logits_raw.gen8", logits.data(), hp.vocab_size, shape, 1,
                                             "decoder.logits_raw.gen8");
        }
        for (int h = 0; h < H; ++h) {
            const float * src = xa.data() + static_cast<size_t>(h) * T_enc;
            double *      dst = att.data() + (static_cast<size_t>(h) * R + pos) * T_enc;
            for (int t = 0; t < T_enc; ++t) {
                dst[t] = src[t];
            }
        }
    }
    jumps_from_att(att, H, R, T_enc, jumps);
    return TRANSCRIBE_OK;
}

// Standardize each head over the R rows, average heads, DTW on -matrix; then
// jumps[k] = first frame of row k + 1 (n = R - 2 text tokens, n + 1 jumps).
void jumps_from_att(const std::vector<double> & att, int H, int R, int T_enc, std::vector<int> & jumps) {
    const int n = R - 2;
    jumps.assign(static_cast<size_t>(n + 1), 0);
    std::vector<double> cost(static_cast<size_t>(R) * T_enc, 0.0);
    for (int h = 0; h < H; ++h) {
        const double * a = att.data() + static_cast<size_t>(h) * R * T_enc;
        for (int t = 0; t < T_enc; ++t) {
            double mu = 0.0, var = 0.0;
            for (int r = 0; r < R; ++r) {
                mu += a[static_cast<size_t>(r) * T_enc + t];
            }
            mu /= R;
            for (int r = 0; r < R; ++r) {
                const double dv = a[static_cast<size_t>(r) * T_enc + t] - mu;
                var += dv * dv;
            }
            const double sd = std::sqrt(var / R) + 1e-9;
            for (int r = 0; r < R; ++r) {
                cost[static_cast<size_t>(r) * T_enc + t] -= (a[static_cast<size_t>(r) * T_enc + t] - mu) / sd / H;
            }
        }
    }
    const auto       path = dtw_path(cost, R, T_enc);
    std::vector<int> first(static_cast<size_t>(R), -1);
    for (const auto & [ri, fj] : path) {
        if (first[static_cast<size_t>(ri)] < 0) {
            first[static_cast<size_t>(ri)] = fj;
        }
    }
    for (int k = 0; k <= n; ++k) {
        jumps[static_cast<size_t>(k)] = std::max(0, first[static_cast<size_t>(k + 1)]);
    }
}

// The alignment rows of hypothesis `h` (utterance u) from a beam capture:
// att [head][row][frame] over rows 0 .. fin_pos. False when the hypothesis
// did not finish inside the search (the caller then runs align_tokens).
bool att_from_capture(const XattnCapture &   cap,
                      const WhistleHParams & hp,
                      int                    S,
                      int                    n_utt,
                      int                    T_cache,
                      int                    u,
                      int                    T_utt,
                      const Hyp &            h,
                      std::vector<double> &  att,
                      int &                  R) {
    R = h.fin_pos + 1;
    if (h.fin_pos < 0 || R != static_cast<int>(h.toks.size()) + 2 || static_cast<int>(cap.steps.size()) < R) {
        return false;
    }
    const int    l0 = hp.dec_n_layers / 2, nh = hp.n_heads, nb = S / n_utt;
    const int    H   = (hp.dec_n_layers - l0) * nh;
    const size_t blk = static_cast<size_t>(T_cache) * nh * S;
    att.assign(static_cast<size_t>(H) * R * T_utt, 0.0);
    int slot = h.fin_slot;
    for (int pos = R - 1; pos >= 0; --pos) {
        const std::vector<float> & st = cap.steps[static_cast<size_t>(pos)];
        const int                  k  = slot % nb;
        for (int li = 0; li < hp.dec_n_layers - l0; ++li) {
            for (int hh = 0; hh < nh; ++hh) {
                const float * src =
                    st.data() + static_cast<size_t>(li) * blk +
                    static_cast<size_t>(T_cache) * (k + static_cast<size_t>(nb) * (hh + static_cast<size_t>(nh) * u));
                double * dst = att.data() + (static_cast<size_t>(li * nh + hh) * R + pos) * T_utt;
                for (int t = 0; t < T_utt; ++t) {
                    dst[t] = src[t];
                }
            }
        }
        slot = cap.parent[static_cast<size_t>(pos)][static_cast<size_t>(slot)];
    }
    return true;
}

// Speech onset in 80 ms frames: the first 40 ms block whose energy is within
// 19.5 dB of the loudest block, rounded up. Fitted to the engine's first-word
// starts (15 clips: all within +-1 frame, 80% exact).
int speech_onset_frame(const float * pcm, int n_samples) {
    constexpr int blk = 640;
    const int     nb  = n_samples / blk;
    if (nb <= 0) {
        return 0;
    }
    std::vector<double> e(static_cast<size_t>(nb));
    double              mx = -1e300;
    for (int b = 0; b < nb; ++b) {
        double s = 0.0;
        for (int i = 0; i < blk; ++i) {
            const double v = pcm[static_cast<size_t>(b) * blk + i];
            s += v * v;
        }
        e[static_cast<size_t>(b)] = 10.0 * std::log10(s / blk + 1e-20);
        mx                        = std::max(mx, e[static_cast<size_t>(b)]);
    }
    for (int b = 0; b < nb; ++b) {
        if (e[static_cast<size_t>(b)] > mx - 19.5) {
            return (b + 1) / 2;
        }
    }
    return 0;
}

// Byte length of the UTF-8 sequence starting with `c`.
size_t utf8_len(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return u < 0x80 ? 1 : (u >> 5) == 0x6 ? 2 : (u >> 4) == 0xE ? 3 : (u >> 3) == 0x1E ? 4 : 1;
}

// Punctuation-only piece (after stripping U+2581): no letters or digits.
bool is_punct_piece(const std::string & piece) {
    std::string p = piece;
    while (p.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xE2 && static_cast<unsigned char>(p[1]) == 0x96 &&
           static_cast<unsigned char>(p[2]) == 0x81) {
        p.erase(0, 3);
    }
    if (p.empty()) {
        return false;
    }
    static const char * const k_punct[] = { "\xC2\xAB",     "\xC2\xBB",     "\xE2\x80\x9E", "\xE2\x80\x9C",
                                            "\xE2\x80\x9D", "\xE2\x80\x98", "\xE2\x80\x99", "\xE2\x80\x94",
                                            "\xE2\x80\x93", "\xE2\x80\xA6", "\xC2\xBF",     "\xC2\xA1" };
    size_t                    i         = 0;
    while (i < p.size()) {
        const unsigned char c = static_cast<unsigned char>(p[i]);
        if (c < 0x80) {
            if (std::isalnum(c)) {
                return false;
            }
            ++i;
            continue;
        }
        bool matched = false;
        for (const char * q : k_punct) {
            const size_t ql = std::strlen(q);
            if (p.compare(i, ql, q) == 0) {
                i += ql;
                matched = true;
                break;
            }
        }
        if (!matched) {
            return false;  // any other non-ASCII codepoint counts as a letter
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Result assembly
// ---------------------------------------------------------------------------

bool piece_starts_word(const std::string & p) {
    return p.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xE2 && static_cast<unsigned char>(p[1]) == 0x96 &&
           static_cast<unsigned char>(p[2]) == 0x81;
}

void commit_result(WhistleSession *         cc,
                   WhistleModel *           cm,
                   const Hyp *              h,
                   bool                     with_words,
                   const std::vector<int> & jumps,
                   int                      onset_frame,
                   int64_t                  audio_ms) {
    cc->tokens.clear();
    cc->words.clear();
    cc->segments.clear();
    cc->full_text.clear();
    cc->raw_text.clear();
    cc->has_result  = true;
    cc->result_kind = with_words ? TRANSCRIBE_TIMESTAMPS_WORD : TRANSCRIBE_TIMESTAMPS_NONE;
    if (h == nullptr || h->toks.empty()) {
        return;
    }
    const Tokenizer & tok  = cm->tok;
    std::string       full = tok.decode(h->toks.data(), static_cast<int>(h->toks.size()));
    cc->raw_text           = full;
    if (!full.empty() && full.front() == ' ') {
        full.erase(full.begin());
    }
    constexpr int64_t frame_ms = 80;
    auto              ms       = [&](int frame) {
        return std::min<int64_t>(audio_ms, frame * frame_ms);
    };

    transcribe_session::SegmentEntry seg;
    seg.text = full;
    if (with_words) {
        const int        n = static_cast<int>(h->toks.size());
        std::vector<int> word_first;
        for (int i = 0; i < n; ++i) {
            transcribe_session::TokenEntry te;
            te.id    = h->toks[i];
            te.text  = tok.decode(&h->toks[i], 1);
            te.p     = static_cast<float>(std::exp(h->lps[i]));
            te.t0_ms = ms(jumps[i]);
            te.t1_ms = ms(jumps[i + 1]);
            if (i == 0 || piece_starts_word(tok.token(h->toks[i]))) {
                transcribe_session::WordEntry we;
                we.first_token = i;
                cc->words.push_back(we);
                word_first.push_back(i);
            }
            transcribe_session::WordEntry & we = cc->words.back();
            we.text += te.text;
            we.n_tokens += 1;
            te.word_index = static_cast<int>(cc->words.size()) - 1;
            cc->tokens.push_back(std::move(te));
        }
        for (size_t w = 0; w < cc->words.size(); ++w) {
            transcribe_session::WordEntry & we   = cc->words[w];
            const int                       f    = we.first_token;
            int                             core = f + we.n_tokens - 1;
            while (core > f && is_punct_piece(tok.token(h->toks[core]))) {
                --core;
            }
            const int last = f + we.n_tokens - 1;
            // The engine keeps trailing punctuation inside the word span when
            // it carries a quote (e.g. `,"`), or when the next word opens with
            // punctuation (e.g. `▁"`); fitted on 12k LibriSpeech words.
            if (core < last) {
                bool keep = false;
                for (int k = core + 1; k <= last; ++k) {
                    keep = keep || tok.token(h->toks[k]).find('"') != std::string::npos;
                }
                if (w + 1 < cc->words.size()) {
                    std::string nxt = tok.token(h->toks[cc->words[w + 1].first_token]);
                    if (piece_starts_word(nxt)) {
                        nxt.erase(0, 3);
                    }
                    keep = keep || (!nxt.empty() && is_punct_piece(nxt.substr(0, utf8_len(nxt[0]))));
                }
                if (keep) {
                    core = last;
                }
            }
            int start = jumps[f];
            int end   = jumps[core + 1];
            if (w == 0) {
                start = std::max(start, std::min(onset_frame, end - 1));
            }
            we.t0_ms = ms(start);
            we.t1_ms = ms(std::max(end, start));
            if (!we.text.empty() && we.text.front() == ' ') {
                we.text.erase(we.text.begin());
            }
        }
        seg.t0_ms    = cc->words.front().t0_ms;
        seg.t1_ms    = cc->words.back().t1_ms;
        seg.n_words  = static_cast<int>(cc->words.size());
        seg.n_tokens = static_cast<int>(cc->tokens.size());
    }
    cc->segments.push_back(std::move(seg));
    cc->full_text = std::move(full);
}

std::string lang_code(const WhistleModel & cm, int token) {
    for (size_t i = 0; i < cm.hparams.lang_token_ids.size(); ++i) {
        if (cm.hparams.lang_token_ids[i] == token) {
            return cm.hparams.languages[i];
        }
    }
    return {};
}

int lang_token_for(const WhistleModel & cm, const char * language) {
    if (language == nullptr || language[0] == '\0') {
        return -1;
    }
    for (size_t i = 0; i < cm.hparams.languages.size(); ++i) {
        if (cm.hparams.languages[i] == language) {
            return cm.hparams.lang_token_ids[i];
        }
    }
    return -2;
}

bool want_words(const transcribe_run_params * params) {
    if (params == nullptr) {
        return false;
    }
    return params->timestamps == TRANSCRIBE_TIMESTAMPS_WORD || params->timestamps == TRANSCRIBE_TIMESTAMPS_TOKEN ||
           params->timestamps == TRANSCRIBE_TIMESTAMPS_SEGMENT;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<WhistleSession *>(session);
    auto * cm = static_cast<WhistleModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    transcribe::debug::init();
    const WhistleHParams & hp = cm->hparams;

    // Input-length contract: the engine rejects audio over 30 s.
    if (n_samples > hp.max_audio_samples) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "whistle run: input too long - %d samples exceed the %d the model supports (%.0f s). See "
                "transcribe_capabilities.max_audio_ms.",
                n_samples, hp.max_audio_samples, hp.max_audio_samples / static_cast<double>(hp.fe_sample_rate));
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }
    const int lang_req = lang_token_for(*cm, params != nullptr ? params->language : nullptr);
    if (lang_req == -2) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t audio_ms = static_cast<int64_t>(n_samples) * 1000 / hp.fe_sample_rate;

    cc->clear_result();

    // ----- frontend + no-speech gate -----
    const int64_t t_mel = ggml_time_us();
    MelResult     mel;
    if (!compute_mel(pcm, n_samples, cm->aux.mel_fb, hp.fe_num_mels, mel,
                     transcribe::configure_sched_n_threads(nullptr, cc->n_threads))) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle run: audio too short for one frame");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    cc->t_mel_us = ggml_time_us() - t_mel;
    if (mel.energy_spread_db <= kNoSpeechSpreadDb) {
        // No speech: empty transcript and no detected language (engine parity).
        cc->has_result  = true;
        cc->result_kind = want_words(params) ? TRANSCRIBE_TIMESTAMPS_WORD : TRANSCRIBE_TIMESTAMPS_NONE;
        return TRANSCRIBE_OK;
    }

    ensure_threadpool(cc, cm);
    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) {
        return st;
    }

    // ----- encoder -----
    const int64_t t_enc = ggml_time_us();
    int           T_enc = 0;
    if (auto st = encode(cc, cm, mel, cc->enc_host, T_enc); st != TRANSCRIBE_OK) {
        return st;
    }
    cc->t_encode_us = ggml_time_us() - t_enc;
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, hp.d_model };
        transcribe::debug::dump_host_f32("dec.enc_out", cc->enc_host.data(),
                                         static_cast<long long>(cc->enc_host.size()), shape, 2, "decoder.enc_out");
    }
    build_memory(*cm, cc->enc_host, T_enc, cc->mem_host);

    // ----- decoder -----
    const int64_t t_dec = ggml_time_us();
    const int     beam  = beam_width();
    int           n_ctx = hp.dec_max_seq;
    if (cc->n_ctx > k_prompt_len && cc->n_ctx < n_ctx) {
        n_ctx = cc->n_ctx;
    }
    if (!dec_cache_init(cc->cache, cm->plan.primary, hp, n_ctx, beam, 1, T_enc)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle run: decoder cache allocation failed - out of memory");
        return TRANSCRIBE_ERR_OOM;
    }
    if (auto st = fill_cross(cc, cm, 0, cc->mem_host, T_enc); st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<UttDecode>  res;
    const std::vector<int>  utt_T{ T_enc };
    // Word timing reads the cross-attention captured during the search; debug
    // runs keep the teacher-forced pass (it also writes the gen8 dump).
    const bool              capture = want_words(params) && !transcribe::debug::enabled();
    XattnCapture            cap;
    const transcribe_status st = beam_decode(cc, cm, utt_T, { lang_req }, res, capture ? &cap : nullptr);
    if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_ABORTED) {
        return st;
    }
    const UttDecode & ud = res[0];
    const Hyp *       h  = best_hyp(ud);
    if (st == TRANSCRIBE_ERR_ABORTED && h == nullptr && !ud.live.empty()) {
        h = &ud.live[0];
    }
    cc->detected_language = lang_code(*cm, ud.lang_token);
    if (ud.hit_budget) {
        cc->was_truncated = true;
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "whistle run: output truncated at the %d-token decoder context", n_ctx);
    }

    std::vector<int>    jumps;
    const bool          words = want_words(params) && h != nullptr && st == TRANSCRIBE_OK;
    std::vector<double> att;
    int                 R = 0;
    if (words && capture &&
        att_from_capture(cap, hp, cc->cache.n_seq, cc->cache.n_utt, cc->cache.T_enc, 0, T_enc, *h, att, R)) {
        jumps_from_att(att, (hp.dec_n_layers - hp.dec_n_layers / 2) * hp.n_heads, R, T_enc, jumps);
    } else if ((words || transcribe::debug::enabled()) && h != nullptr && st == TRANSCRIBE_OK) {
        // Validation runs also need the teacher-forced gen8 dump.
        if (auto ast = align_tokens(cc, cm, cc->mem_host, T_enc, ud.lang_token, h->toks, jumps); ast != TRANSCRIBE_OK) {
            return ast;
        }
    }
    commit_result(cc, cm, h, words, jumps, words ? speech_onset_frame(pcm, n_samples) : 0, audio_ms);
    cc->t_decode_us = ggml_time_us() - t_dec;
    if (st == TRANSCRIBE_ERR_ABORTED) {
        return st;
    }
    return cc->truncation_status();
}

// ---------------------------------------------------------------------------
// run_batch: per-utterance frontend + encoder (serial, so each encoder output
// is bit-identical to the single-shot path), then ONE beam search over all
// utterances x beams in lockstep. Cross K/V carries an utterance dimension;
// frames past an utterance's own T_enc are masked to -inf.
// ---------------------------------------------------------------------------

transcribe_status run_batch(transcribe_session *          session,
                            const float * const *         pcm,
                            const int *                   n_samples,
                            int                           n,
                            const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<WhistleSession *>(session);
    auto * cm = static_cast<WhistleModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe::debug::init();
    const WhistleHParams & hp       = cm->hparams;
    const int              lang_req = lang_token_for(*cm, params != nullptr ? params->language : nullptr);
    if (lang_req == -2) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const bool words = want_words(params);
    ensure_threadpool(cc, cm);
    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) {
        return st;
    }

    // ----- per-utterance frontend + gate + encoder -----
    struct Utt {
        transcribe_status  status = TRANSCRIBE_OK;
        bool               speech = false;
        int                T_enc  = 0;
        std::vector<float> mem;
        int64_t            mel_us = 0, enc_us = 0;
    };

    std::vector<Utt> utts(static_cast<size_t>(n));
    for (int b = 0; b < n; ++b) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        Utt & u = utts[static_cast<size_t>(b)];
        if (pcm[b] == nullptr || n_samples[b] <= 0) {
            u.status = TRANSCRIBE_ERR_INVALID_ARG;
            continue;
        }
        if (n_samples[b] > hp.max_audio_samples) {
            u.status = TRANSCRIBE_ERR_INPUT_TOO_LONG;
            continue;
        }
        const int64_t t0 = ggml_time_us();
        MelResult     mel;
        if (!compute_mel(pcm[b], n_samples[b], cm->aux.mel_fb, hp.fe_num_mels, mel,
                         transcribe::configure_sched_n_threads(nullptr, cc->n_threads))) {
            u.status = TRANSCRIBE_ERR_INVALID_ARG;
            continue;
        }
        u.mel_us = ggml_time_us() - t0;
        if (mel.energy_spread_db <= kNoSpeechSpreadDb) {
            continue;  // no speech: empty result
        }
        const int64_t      t1 = ggml_time_us();
        std::vector<float> enc;
        if (auto st = encode(cc, cm, mel, enc, u.T_enc); st != TRANSCRIBE_OK) {
            return st;
        }
        u.enc_us = ggml_time_us() - t1;
        if (transcribe::debug::enabled()) {
            char            nm[48];
            const long long shape[2] = { u.T_enc, hp.d_model };
            std::snprintf(nm, sizeof(nm), "dec.enc_out.b%d", b);
            transcribe::debug::dump_host_f32(nm, enc.data(), static_cast<long long>(enc.size()), shape, 2,
                                             "decoder.enc_out");
        }
        build_memory(*cm, enc, u.T_enc, u.mem);
        u.speech = true;
    }

    // ----- batched beam decode over the utterances with speech -----
    std::vector<int> active;
    int              T_max = 0;
    for (int b = 0; b < n; ++b) {
        if (utts[static_cast<size_t>(b)].speech) {
            active.push_back(b);
            T_max = std::max(T_max, utts[static_cast<size_t>(b)].T_enc);
        }
    }
    std::vector<UttDecode> res;
    transcribe_status      dst   = TRANSCRIBE_OK;
    const int64_t          t_dec = ggml_time_us();
    // Word timing from the beam capture (see run()) for single-utterance
    // batches only; multi-utterance batches keep the teacher-forced pass
    // (bounded memory, word timings identical to serial).
    XattnCapture           cap;
    int                    n_seq_b = 0, n_utt_b = 0, T_cache_b = 0;
    if (!active.empty()) {
        const int na    = static_cast<int>(active.size());
        const int beam  = beam_width();
        int       n_ctx = hp.dec_max_seq;
        if (cc->n_ctx > k_prompt_len && cc->n_ctx < n_ctx) {
            n_ctx = cc->n_ctx;
        }
        if (!dec_cache_init(cc->cache, cm->plan.primary, hp, n_ctx, na * beam, na, T_max)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle run_batch: decoder cache allocation failed - out of memory");
            return TRANSCRIBE_ERR_OOM;
        }
        std::vector<int> utt_T;
        for (int i = 0; i < na; ++i) {
            const Utt & u = utts[static_cast<size_t>(active[i])];
            if (auto st = fill_cross(cc, cm, i, u.mem, u.T_enc); st != TRANSCRIBE_OK) {
                return st;
            }
            utt_T.push_back(u.T_enc);
        }
        n_seq_b   = cc->cache.n_seq;
        n_utt_b   = cc->cache.n_utt;
        T_cache_b = cc->cache.T_enc;
        dst       = beam_decode(cc, cm, utt_T, std::vector<int>(static_cast<size_t>(na), lang_req), res,
                                words && na == 1 && !transcribe::debug::enabled() ? &cap : nullptr);
        if (dst != TRANSCRIBE_OK && dst != TRANSCRIBE_ERR_ABORTED) {
            return dst;
        }
    }
    const int64_t dec_us = ggml_time_us() - t_dec;

    // ----- capture (word alignment per utterance) -----
    const int n_active = std::max<int>(1, static_cast<int>(active.size()));
    for (int b = 0, ai = 0; b < n; ++b) {
        const Utt & u = utts[static_cast<size_t>(b)];
        if (u.status != TRANSCRIBE_OK) {
            transcribe_session::ResultSet rs;
            rs.status = u.status;
            cc->batch_results.push_back(std::move(rs));
            continue;
        }
        cc->clear_result();
        cc->t_mel_us         = u.mel_us;
        cc->t_encode_us      = u.enc_us;
        cc->t_decode_us      = u.speech ? dec_us / n_active : 0;
        transcribe_status st = TRANSCRIBE_OK;
        if (!u.speech) {
            cc->has_result  = true;
            cc->result_kind = words ? TRANSCRIBE_TIMESTAMPS_WORD : TRANSCRIBE_TIMESTAMPS_NONE;
        } else {
            const UttDecode & ud  = res[static_cast<size_t>(ai++)];
            const Hyp *       h   = best_hyp(ud);
            cc->detected_language = lang_code(*cm, ud.lang_token);
            std::vector<int>    jumps;
            const bool          w = words && h != nullptr && dst == TRANSCRIBE_OK;
            std::vector<double> att;
            int                 R = 0;
            if (w && !cap.steps.empty() &&
                att_from_capture(cap, hp, n_seq_b, n_utt_b, T_cache_b, ai - 1, u.T_enc, *h, att, R)) {
                jumps_from_att(att, (hp.dec_n_layers - hp.dec_n_layers / 2) * hp.n_heads, R, u.T_enc, jumps);
            } else if (w) {
                if (auto ast = align_tokens(cc, cm, u.mem, u.T_enc, ud.lang_token, h->toks, jumps);
                    ast != TRANSCRIBE_OK) {
                    return ast;
                }
            }
            const int64_t audio_ms = static_cast<int64_t>(n_samples[b]) * 1000 / hp.fe_sample_rate;
            commit_result(cc, cm, h, w, jumps, w ? speech_onset_frame(pcm[b], n_samples[b]) : 0, audio_ms);
            if (ud.hit_budget) {
                cc->was_truncated = true;
                st                = TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
            }
        }
        cc->batch_results.push_back(cc->capture_result(st));
    }
    // Leave the scratch slot mirroring batch_results[0].
    if (!cc->batch_results.empty()) {
        const transcribe_session::ResultSet & r0 = cc->batch_results.front();
        cc->tokens                               = r0.tokens;
        cc->words                                = r0.words;
        cc->segments                             = r0.segments;
        cc->full_text                            = r0.full_text;
        cc->raw_text                             = r0.raw_text;
        cc->detected_language                    = r0.detected_language;
        cc->result_kind                          = r0.result_kind;
        cc->has_result                           = r0.has_result;
    }
    return dst;
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "whistle",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ run_batch,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

}  // namespace transcribe::whistle
