// arch/cohere/model.cpp - Cohere ASR family handler.
//
// Load/init_context/run lifecycle for the Cohere ASR encoder-decoder
// model. The encoder is a conformer (identical to Parakeet except FFN
// has bias). The decoder is an autoregressive Transformer that runs
// on the ggml graph (not host-side like Parakeet's LSTM).

#include "cohere.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::cohere {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   CohereModel>);
static_assert(std::is_base_of_v<transcribe_context, CohereContext>);

CohereContext::~CohereContext() {
    kv_cache.free();
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    encoder_out = nullptr;
}

// ---------------------------------------------------------------------------
// KV cache initialization.
// ---------------------------------------------------------------------------

bool kv_cache_init(CohereKvCache & cache,
                   ggml_backend_t  backend,
                   int             n_ctx,
                   int             T_enc,
                   int             n_state,
                   int             n_layer,
                   ggml_type       kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
                     "cohere kv_cache: unsupported kv_type=%d "
                     "(only F16/F32)\n", static_cast<int>(kv_type));
        return false;
    }

    // Allocate 4 tensors: self K, self V, cross K, cross V.
    const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;

    ggml_init_params params {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "cohere kv_cache: ggml_init failed\n");
        return false;
    }

    const int64_t self_elements  = static_cast<int64_t>(n_state) * n_layer * n_ctx;
    const int64_t cross_elements = static_cast<int64_t>(n_state) * n_layer * T_enc;

    cache.self_k  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.self_v  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

    ggml_set_name(cache.self_k,  "kv_self_k");
    ggml_set_name(cache.self_v,  "kv_self_v");
    ggml_set_name(cache.cross_k, "kv_cross_k");
    ggml_set_name(cache.cross_v, "kv_cross_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        std::fprintf(stderr, "cohere kv_cache: buffer alloc failed\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }

    // Zero out the buffers.
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx  = n_ctx;
    cache.T_enc  = T_enc;
    cache.n      = 0;
    cache.head   = 0;
    cache.cross_populated = false;

    const size_t total_bytes =
        ggml_nbytes(cache.self_k) + ggml_nbytes(cache.self_v) +
        ggml_nbytes(cache.cross_k) + ggml_nbytes(cache.cross_v);
    std::fprintf(stderr,
                 "cohere kv_cache: allocated %.1f MB (%s) "
                 "(self: %d ctx x %d layers, cross: %d T_enc x %d layers)\n",
                 static_cast<double>(total_bytes) / (1024.0 * 1024.0),
                 ggml_type_name(kv_type),
                 n_ctx, n_layer, T_enc, n_layer);

    return true;
}

CohereModel::~CohereModel() {
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        ggml_backend_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
    }
    if (conv_pw_f32_ctx != nullptr) {
        ggml_free(conv_pw_f32_ctx);
        conv_pw_f32_ctx = nullptr;
    }
    if (conv_pw_f32_buffer != nullptr) {
        ggml_backend_buffer_free(conv_pw_f32_buffer);
        conv_pw_f32_buffer = nullptr;
    }
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin();
         it != plan.scheduler_list.rend(); ++it)
    {
        ggml_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

namespace {

constexpr float kBnEps = 1e-5f;

transcribe_status fuse_batch_norm(CohereModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t d = m.hparams.enc_d_model;
    const size_t tensor_bytes = static_cast<size_t>(d) * sizeof(float);

    const size_t ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params = {ctx_size, nullptr, true};
    m.bn_fused_ctx = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) return TRANSCRIBE_ERR_BACKEND;

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
    }

    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(
        m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) return TRANSCRIBE_ERR_BACKEND;

    std::vector<float> bn_w(d), bn_b(d), rm(d), rv(d);
    std::vector<float> fused_s(d), fused_b(d);

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        ggml_backend_tensor_get(b.conv_bn_w,  bn_w.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_b,  bn_b.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rm, rm.data(),   0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rv, rv.data(),   0, tensor_bytes);

        for (int64_t c = 0; c < d; ++c) {
            const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
            fused_s[c] = s;
            fused_b[c] = bn_b[c] - rm[c] * s;
        }

        ggml_backend_tensor_set(b.conv_bn_fused_scale, fused_s.data(), 0, tensor_bytes);
        ggml_backend_tensor_set(b.conv_bn_fused_bias,  fused_b.data(), 0, tensor_bytes);
    }

    return TRANSCRIBE_OK;
}

// Fold each encoder layer's Q bias into its pos_bias_u / pos_bias_v
// tensors at load time. In rel_pos_mhsa the math is
//   q_u = (W_q x + q_b) + pos_u = W_q x + (q_b + pos_u)
//   q_v = (W_q x + q_b) + pos_v = W_q x + (q_b + pos_v)
// so pre-adding q_b into pos_u/pos_v is mathematically identical and
// drops the explicit `q = q + q_b` graph op (48 fewer ops per encoder).
// After fusion we null out attn_q_b so the graph builder naturally
// skips the add via its existing `if (attn_q_b != nullptr)` guard.
transcribe_status fuse_encoder_q_bias(CohereModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t d_model  = m.hparams.enc_d_model;
    const int64_t n_heads  = m.hparams.enc_n_heads;
    const int64_t head_dim = n_heads > 0 ? d_model / n_heads : 0;

    if (head_dim * n_heads != d_model) {
        std::fprintf(stderr,
            "cohere: d_model (%lld) != head_dim*n_heads; "
            "skipping encoder Q-bias fusion\n", (long long)d_model);
        return TRANSCRIBE_OK;
    }

    const size_t nbytes = static_cast<size_t>(d_model) * sizeof(float);
    std::vector<float> q_bias(d_model);
    std::vector<float> pos_u (d_model);
    std::vector<float> pos_v (d_model);

    size_t fused = 0;
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        if (b.attn_q_b   == nullptr ||
            b.attn_pos_u == nullptr ||
            b.attn_pos_v == nullptr) {
            continue;
        }

        ggml_backend_tensor_get(b.attn_q_b,   q_bias.data(), 0, nbytes);
        ggml_backend_tensor_get(b.attn_pos_u, pos_u.data(),  0, nbytes);
        ggml_backend_tensor_get(b.attn_pos_v, pos_v.data(),  0, nbytes);

        // pos_u/v are [head_dim, n_heads] with ne[0]=head_dim.
        // q_b is [d_model] = [head_dim*n_heads]. In memory the two
        // share the same element layout (same linear offsets), so
        // a flat element-wise add is correct.
        for (int64_t j = 0; j < d_model; ++j) {
            pos_u[j] += q_bias[j];
            pos_v[j] += q_bias[j];
        }

        ggml_backend_tensor_set(b.attn_pos_u, pos_u.data(), 0, nbytes);
        ggml_backend_tensor_set(b.attn_pos_v, pos_v.data(), 0, nbytes);

        // Drop the reference so the graph builder skips the add.
        // Tensor memory stays allocated inside ctx_meta.
        b.attn_q_b = nullptr;
        ++fused;
    }

    if (fused > 0) {
        std::fprintf(stderr,
            "cohere: fused Q bias into pos_u/pos_v for %zu encoder blocks\n",
            fused);
    }
    return TRANSCRIBE_OK;
}

// On a CPU primary backend, dequantize the conformer 1×1 pointwise
// conv weights (pw1, pw2) from F16 back to F32. Step 3 moved them to
// F16 in the GGUF to halve the Vulkan/Metal matmul cost, but Zen 2
// (and anything else without native F16 compute) pays an F16→F32
// upconvert per-element that outweighs the bandwidth win — a ~600 ms
// regression on a 9.3 s q4_k_m CPU encode.
//
// The cleanest fix without shipping two GGUFs is to hoist the
// conversion to load time: dequantize into a separate backend buffer
// and point the weight slots at the F32 copies. Vulkan/Metal/CUDA
// primary backends skip this step and keep the F16 weights. The
// original F16 tensors stay allocated in the main weight buffer
// (unused) — dropping them individually isn't supported by ggml's
// backend buffer model, and the ~235 MB cost is acceptable for the
// CPU-only path.
//
// Weight collection is the only family-specific bit: we walk
// m.weights.blocks and hand the shared helper a list of (slot, src)
// pairs. The helper does the backend check, allocation, dequantize,
// and repoint.
transcribe_status promote_conv_pw_to_f32_on_cpu(CohereModel & m) {
    std::vector<load_common::ConvPwF32Slot> slots;
    slots.reserve(m.weights.blocks.size() * 2);
    for (auto & b : m.weights.blocks) {
        if (b.conv_pw1_w != nullptr && b.conv_pw1_w->type == GGML_TYPE_F16) {
            slots.push_back({&b.conv_pw1_w, b.conv_pw1_w});
        }
        if (b.conv_pw2_w != nullptr && b.conv_pw2_w->type == GGML_TYPE_F16) {
            slots.push_back({&b.conv_pw2_w, b.conv_pw2_w});
        }
    }
    return load_common::promote_conv_pw_f16_to_f32_on_cpu(
        m.plan, slots, "cohere",
        &m.conv_pw_f32_ctx, &m.conv_pw_f32_buffer);
}

constexpr const char k_default_variant[] = "cohere-asr";

// Forward declarations for the Arch trait below.
extern transcribe_status load        (Loader &, const transcribe_model_params *,
                                      transcribe_model **);
extern transcribe_status init_context(transcribe_model *, const transcribe_context_params *,
                                      transcribe_context **);
extern transcribe_status run         (transcribe_context *, const float *, int,
                                      const transcribe_params *);

transcribe_status load(
    Loader &                          loader,
    const transcribe_model_params *   params,
    transcribe_model **               out_model)
{
    // params->backend is consumed below in the backend init block;
    // params->gpu_device is reserved per the public header contract
    // and is not yet honored (multi-device selection is a future
    // release).

    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<CohereModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    if (loader.variant().empty()) {
        m->variant = k_default_variant;
    } else {
        m->variant = loader.variant();
    }
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Tokenizer.
    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }

    // Hparams.
    if (const transcribe_status st = read_cohere_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Set vocab_size from tokenizer.
    m->hparams.vocab_size = m->tok.n_tokens();
    // Set special token IDs from tokenizer.
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    // Hard-fail at load time if the tokenizer did not supply an EOS
    // token id. Previously this was papered over at decode time with a
    // fallback to token id 3 plus a stderr warning; that is worse than
    // refusing the model because (a) id 3 is a random guess that only
    // happens to match this family's current checkpoint, and (b) a
    // missing EOS is a GGUF-builder bug that should surface during
    // conversion, not hide behind a runtime fallback in production.
    if (m->hparams.eos_token_id < 0) {
        std::fprintf(stderr,
                     "cohere: GGUF tokenizer has no eos_token_id -- "
                     "regenerate with an up-to-date converter\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Mel frontend.
    {
        transcribe::MelConfig cfg {};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;

        // Load checkpoint filterbank/window from GGUF if present.
        // These small tensors are read directly from the file using
        // the gguf index — avoids recomputing from scratch and
        // matches the exact values the model was trained with.
        //
        // read_f32_tensor_checked validates type (must be F32), byte
        // alignment, expected element count, and read completeness.
        // Absent → compute from hparams (the non-GGUF path).
        // BadType/BadSize/ReadErr → hard fail with TRANSCRIBE_ERR_GGUF.
        {
            using R = load_common::ReadF32Result;

            const size_t fb_elems = static_cast<size_t>(cfg.num_mels)
                                  * static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto fb_rc = load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.mel_filterbank", fb_elems,
                "cohere", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }

            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto win_rc = load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.window", win_elems,
                "cohere", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
    }

    // Stage 2: reopen with no_alloc.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st =
            build_cohere_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // Resolve the backend plan from the caller request. See
    // transcribe-backend.h + transcribe-load-common.h for the
    // semantics (AUTO / CPU / METAL / VULKAN). The important case
    // for cohere is strict CPU: the F16→F32 conv pointwise
    // promotion below depends on `plan.primary_kind ==
    // BackendKind::Cpu`, which only a TRANSCRIBE_BACKEND_CPU
    // request reliably produces.
    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;

    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "cohere", m->plan);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    m->backend = ggml_backend_name(m->plan.primary);

    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        std::fprintf(stderr,
                     "cohere: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Stream tensor data from the GGUF file into the backend buffer
    // slots. See transcribe-load-common.h for the shared loop.
    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "cohere");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    gguf_free(gguf_data);

    // Fuse BatchNorm.
    if (const transcribe_status st = fuse_batch_norm(*m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Fuse encoder Q bias into pos_bias_u/v (drops one add per layer).
    if (const transcribe_status st = fuse_encoder_q_bias(*m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // On CPU backend, dequantize conv pointwise weights back to F32 —
    // Zen 2 class CPUs don't have native F16 compute and the upconvert
    // cost per matmul outweighs the bandwidth savings from the smaller
    // weight. No-op on GPU backends.
    if (const transcribe_status st = promote_conv_pw_to_f32_on_cpu(*m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(
    transcribe_model *                model,
    const transcribe_context_params * params,
    transcribe_context **             out_ctx)
{
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc = std::make_unique<CohereContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    // Flash-attention policy -- see CohereContext (cohere.h) for why
    // this is split into encoder vs decoder. The short version:
    // encoder dk=160 is unsupported by upstream ggml Metal flash, so
    // the encoder auto-disables on Metal; decoder dk=128 works on
    // every backend we ship, so the decoder defaults on. The env-var
    // overrides (TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH) are
    // applied globally in transcribe::flash::apply_env_overrides.
    //
    // We check the classified primary BackendKind here instead of
    // string-matching on ggml_backend_name — the plan already has
    // the kind ready.
    auto * cm = static_cast<CohereModel *>(model);
    const bool is_metal =
        (cm->plan.primary_kind == transcribe::BackendKind::Metal);

    cc->encoder_use_flash = !is_metal;
    cc->decoder_use_flash = true;

    transcribe::flash::apply_env_overrides(
        cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

transcribe_status run(
    transcribe_context *      ctx,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<CohereContext *>(ctx);
    auto * cm = static_cast<CohereModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-run abort check. Cohere ASR is single-chunk today; this is
    // the single observation point.
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    transcribe::debug::init();

    // ----- Mel front-end -------------------------------------------
    if (!cm->mel.has_value()) {
        std::fprintf(stderr,
                     "cohere run: model has no MelFrontend (load skipped?)\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t t_mel_start = ggml_time_us();
    int mel_n_mels   = 0;
    int mel_n_frames = 0;
    if (const transcribe_status mst = cm->mel->compute(
            pcm, static_cast<size_t>(n_samples),
            cc->mel_buf, mel_n_mels, mel_n_frames);
        mst != TRANSCRIBE_OK)
    {
        std::fprintf(stderr,
                     "cohere run: MelFrontend::compute failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // ----- Reset per-call compute state ----------------------------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    cc->encoder_out = nullptr;

    // ----- Build encoder graph -------------------------------------
    {
        // 48 encoder blocks + decoder = large graph. 8 MB metadata arena.
        ggml_init_params init_params {};
        init_params.mem_size   = 8 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx = ggml_init(init_params);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr,
                         "cohere run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, cm->hparams, mel_n_frames,
        resolved_kv, cc->encoder_use_flash, cm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Allocate + compute encoder graph -------------------------
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr,
                         "cohere run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "cohere run: ggml_backend_sched_alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload mel.
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(),
                            0, cc->mel_buf.size() * sizeof(float));

    transcribe::debug::dump_tensor(
        "enc.mel.in", eb.mel_in, "encoder.mel");

    // Sinusoidal positional embedding (same as Parakeet).
    if (eb.pos_emb_in != nullptr) {
        const int d_model = cm->hparams.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
        const int T_enc   = (pos_len + 1) / 2;

        cc->pos_buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);

        cc->pos_div_term.resize(static_cast<size_t>(d_model / 2));
        const float ln_10000 = std::log(10000.0f);
        for (int k = 0; k < d_model / 2; ++k) {
            cc->pos_div_term[static_cast<size_t>(k)] =
                std::exp(static_cast<float>(2 * k) *
                         (-ln_10000 / static_cast<float>(d_model)));
        }

        for (int i = 0; i < pos_len; ++i) {
            const float pos = static_cast<float>((T_enc - 1) - i);
            float * row = cc->pos_buf.data() + static_cast<size_t>(i) * d_model;
            for (int k = 0; k < d_model / 2; ++k) {
                const float div = cc->pos_div_term[static_cast<size_t>(k)];
                row[2 * k]     = std::sin(pos * div);
                row[2 * k + 1] = std::cos(pos * div);
            }
        }

        ggml_backend_tensor_set(eb.pos_emb_in, cc->pos_buf.data(),
                                0, cc->pos_buf.size() * sizeof(float));

        transcribe::debug::dump_tensor(
            "enc.pos_emb", eb.pos_emb_in, "encoder.pos_emb");
    }

    // Set thread count.
    {
        int n_threads = cc->n_threads;
        if (n_threads <= 0) {
            n_threads = std::min(8, std::max(1, static_cast<int>(
                std::thread::hardware_concurrency())));
        }
        for (int i = 0; i < ggml_backend_sched_get_n_backends(cc->sched); ++i) {
            ggml_backend_t be = ggml_backend_sched_get_backend(cc->sched, i);
            ggml_backend_dev_t dev = ggml_backend_get_device(be);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg == nullptr) continue;
            auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
            if (fn != nullptr) {
                fn(be, n_threads);
            }
        }
    }

    // Compute encoder graph.
    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "cohere run: encoder graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // Dump encoder intermediates.
    auto try_dump = [](const char * name, ggml_tensor * t,
                       const char * stage)
    {
        if (t != nullptr) {
            transcribe::debug::dump_tensor(name, t, stage);
        }
    };

    // Pre-encode sub-stage dumps for debugging.
    try_dump("enc.pre_encode.out",   eb.dumps.pre_encode_out,   "encoder.pre_encode");
    try_dump("enc.block.0.out",      eb.dumps.block0_out,       "encoder.block0.out");
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", cm->hparams.enc_n_layers / 2 - 1);
        try_dump(bname,  eb.dumps.block_mid_out,  "encoder.block_mid.out");
    }
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", cm->hparams.enc_n_layers - 1);
        try_dump(bname,  eb.dumps.block_last_out, "encoder.block_last.out");
    }
    try_dump("enc.final",            eb.dumps.final_out,        "encoder.final");
    try_dump("enc_dec_proj.out",     eb.dumps.enc_dec_proj_out, "encoder.enc_dec_proj");

    cc->encoder_out = eb.out;

    // Read encoder output (after enc-dec projection) to host.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_enc <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "cohere run: encoder output has degenerate shape "
                     "[%d, %d]\n", d_enc, T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ----- Decoder with KV cache ------------------------------------
    //
    // Steps:
    //   1. Initialize KV cache (if not already done for this T_enc).
    //   2. Compute cross-attention K/V once from encoder output.
    //   3. Prompt pass: process all prompt tokens, populate self-attn
    //      KV cache, get first predicted token.
    //   4. Autoregressive loop: single-token step passes using cached KV.

    // Build the prompt tokens from tokenizer vocabulary.
    //
    // Prompt structure (matches Transformers get_decoder_prompt_ids):
    //   ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
    //   <|{lang}|> <|{lang}|>
    //   <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    //
    const char * lang = (params && params->language) ? params->language : "en";

    // Language validation lives in the central dispatcher
    // (transcribe_run), which rejects unsupported caller-provided
    // languages with TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE before
    // reaching this handler. The NULL → "en" default above applies
    // only when the caller did not specify a language; "en" is
    // always in the model's list for every published Cohere ASR
    // checkpoint. If that ever ceases to hold, the dispatcher's
    // check still covers non-NULL cases and we'd revisit the
    // default here.

    const std::string lang_token = std::string("<|") + lang + "|>";
    const std::vector<std::string> prompt_pieces = {
        "\xe2\x96\x81",  // ▁ (U+2581, sentencepiece space prefix)
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        lang_token,
        lang_token,
        "<|pnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    };

    std::vector<int32_t> prompt_ids;
    prompt_ids.reserve(prompt_pieces.size());
    for (const auto & piece : prompt_pieces) {
        const int id = cm->tok.find(piece);
        if (id < 0) {
            std::fprintf(stderr,
                         "cohere run: unknown prompt token '%s'\n",
                         piece.c_str());
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        prompt_ids.push_back(id);
    }
    const int prompt_len = static_cast<int>(prompt_ids.size());

    // --- Step 1: Initialize KV cache --------------------------------
    {
        // Free any existing cache if T_enc changed.
        if (cc->kv_cache.buffer != nullptr &&
            cc->kv_cache.T_enc != T_enc) {
            cc->kv_cache.free();
        }

        if (cc->kv_cache.buffer == nullptr) {
            const int n_ctx = cm->hparams.dec_max_seq > 0
                            ? cm->hparams.dec_max_seq : 1024;
            // Decoder cache dtype: honor user override, else default to
            // F16 (weights are bf16, so F16 KV is lossless-enough and
            // halves autoregressive memory bandwidth).
            ggml_type cache_type = resolved_kv;
            if (cache_type == GGML_TYPE_COUNT) cache_type = GGML_TYPE_F16;
            if (!kv_cache_init(cc->kv_cache,
                               cm->plan.primary,
                               n_ctx, T_enc,
                               static_cast<int>(cm->hparams.dec_hidden),
                               cm->hparams.dec_n_layers,
                               cache_type))
            {
                std::fprintf(stderr,
                             "cohere run: KV cache init failed\n");
                return TRANSCRIBE_ERR_BACKEND;
            }
        } else {
            // Cache exists with same T_enc; just reset self-attn state.
            cc->kv_cache.n    = 0;
            cc->kv_cache.head = 0;
            cc->kv_cache.cross_populated = false;
        }
    }

    // Helper to create a fresh compute context.
    //
    // Any ggml_tensor pointer we hold that was allocated inside the
    // previous compute_ctx becomes dangling the instant we ggml_free
    // it below, so null them out here. Today cc->encoder_out is the
    // only such pointer -- its *data* was already copied to
    // cc->enc_host above, and cross-attn graphs below re-declare a
    // fresh encoder_out input tensor -- but keeping a stale pointer
    // around invites a future edit to use-after-free it.
    auto new_compute_ctx = [&](size_t mem_size) -> bool {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        cc->encoder_out = nullptr;
        ggml_init_params init_params {};
        init_params.mem_size   = mem_size;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx = ggml_init(init_params);
        return cc->compute_ctx != nullptr;
    };

    // Helper to find the causal mask input tensor by name.
    auto find_mask_input = [&]() -> ggml_tensor * {
        for (ggml_tensor * t = ggml_get_first_tensor(cc->compute_ctx);
             t != nullptr;
             t = ggml_get_next_tensor(cc->compute_ctx, t))
        {
            if (std::strcmp(t->name, "dec.causal_mask") == 0 &&
                t->type == GGML_TYPE_F32)
            {
                return t;
            }
        }
        return nullptr;
    };

    // --- Step 2: Compute cross-attention K/V -------------------------
    const int64_t t_dec_start = ggml_time_us();
    {
        if (!new_compute_ctx(4 * 1024 * 1024)) {
            std::fprintf(stderr,
                         "cohere run: ggml_init for cross_kv failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        DecoderBuild cross_db = build_cross_kv_graph(
            cc->compute_ctx, cm->weights, cm->hparams,
            cc->kv_cache, T_enc);
        if (cross_db.graph == nullptr) {
            std::fprintf(stderr,
                         "cohere run: build_cross_kv_graph failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
            std::fprintf(stderr,
                         "cohere run: alloc_graph failed (cross_kv)\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        // Upload encoder output.
        ggml_backend_tensor_set(cross_db.encoder_out_in,
                                cc->enc_host.data(), 0,
                                cc->enc_host.size() * sizeof(float));

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, cross_db.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "cohere run: cross_kv compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    // --- Step 3: Prompt pass with KV cache ---------------------------
    {
        if (!new_compute_ctx(4 * 1024 * 1024)) {
            std::fprintf(stderr,
                         "cohere run: ggml_init for decoder prompt failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        // Prompt pass: skip log_softmax and emit a GPU argmax over the
        // last position. Debug dumps still need the pre-head hidden
        // state; when dumping is enabled we take the slower path.
        const bool prompt_skip_softmax = !transcribe::debug::enabled();
        DecoderBuild db = build_decoder_graph_kv(
            cc->compute_ctx, cm->weights, cm->hparams,
            cc->kv_cache,
            prompt_len, /*n_past=*/0, T_enc,
            /*skip_log_softmax=*/prompt_skip_softmax,
            cc->decoder_use_flash);
        if (db.out == nullptr || db.graph == nullptr) {
            std::fprintf(stderr,
                         "cohere run: build_decoder_graph_kv (prompt) failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
            std::fprintf(stderr,
                         "cohere run: alloc_graph failed (decoder prompt)\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        // Upload token IDs and position IDs.
        ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(),
                                0, prompt_ids.size() * sizeof(int32_t));
        std::vector<int32_t> pos_ids(prompt_len);
        for (int i = 0; i < prompt_len; ++i) pos_ids[i] = i;
        ggml_backend_tensor_set(db.pos_ids_in, pos_ids.data(),
                                0, pos_ids.size() * sizeof(int32_t));

        // Causal mask for prompt pass: [n_kv=prompt_len, prompt_len].
        if (prompt_len > 1) {
            ggml_tensor * mask_input = find_mask_input();
            if (mask_input != nullptr) {
                const int n_kv = prompt_len;
                std::vector<float> mask_data(
                    static_cast<size_t>(n_kv) * prompt_len);
                for (int q = 0; q < prompt_len; ++q) {
                    for (int k = 0; k < n_kv; ++k) {
                        mask_data[static_cast<size_t>(q) * n_kv + k] =
                            (k <= q) ? 0.0f : -1e9f;
                    }
                }
                ggml_backend_tensor_set(mask_input, mask_data.data(),
                                        0, mask_data.size() * sizeof(float));
            }
        }

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, db.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "cohere run: decoder prompt compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }

        // Dump decoder intermediates (prompt pass only).
        try_dump("dec.token_emb",       db.dumps.token_emb,       "decoder.embedding");
        try_dump("dec.pos_emb",         db.dumps.pos_emb,         "decoder.position_embedding");
        try_dump("dec.embed_norm",      db.dumps.embed_norm,      "decoder.embed_norm");
        for (int i = 0; i < cm->hparams.dec_n_layers; ++i) {
            char bname[32], stage[48];
            std::snprintf(bname, sizeof(bname), "dec.block.%d.out", i);
            std::snprintf(stage, sizeof(stage), "decoder.block%d.out", i);
            try_dump(bname, db.dumps.block_out[i], stage);
        }
        try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.out_before_head");
        try_dump("dec.logits_raw",      db.dumps.logits_raw,      "decoder.logits_raw");
        try_dump("dec.logits",          db.dumps.logits,          "decoder.logits");

        // Update KV cache state: prompt_len tokens are now cached.
        cc->kv_cache.n    = prompt_len;
        cc->kv_cache.head = prompt_len;

        // Greedy decode from last prompt position.
        cc->clear_result();

        // Load-time validation guarantees eos_token_id >= 0; no
        // fallback is needed here. See the tokenizer.eos_id() check
        // in cohere::load() at the top of this file.
        const int eos_id = cm->hparams.eos_token_id;
        const int max_tokens = std::min(512,
                                        cc->kv_cache.n_ctx - prompt_len);

        // Pick the first generated token. Fast path reads a single
        // int32 argmax that the GPU computed; debug path reads the
        // full log_softmax'd logits for dumping and argmaxes on host.
        int next_token = 0;
        if (prompt_skip_softmax && db.argmax_out != nullptr) {
            int32_t argmax_id = 0;
            ggml_backend_tensor_get(db.argmax_out, &argmax_id,
                                    0, sizeof(int32_t));
            next_token = argmax_id;
        } else {
            const int64_t vocab_size = db.out->ne[0];
            std::vector<float> logits_host(
                static_cast<size_t>(vocab_size) * prompt_len);
            ggml_backend_tensor_get(db.out, logits_host.data(), 0,
                                    logits_host.size() * sizeof(float));
            const float * last_logits = logits_host.data() +
                                        static_cast<size_t>(prompt_len - 1) *
                                        vocab_size;
            float best = last_logits[0];
            for (int j = 1; j < static_cast<int>(vocab_size); ++j) {
                if (last_logits[j] > best) {
                    best = last_logits[j];
                    next_token = j;
                }
            }
        }

        std::vector<int> generated_ids;
        if (next_token != eos_id) {
            generated_ids.push_back(next_token);
        }

        // Commit accumulated generated_ids as the run's segment + full
        // text. Called both on normal loop exit and on abort so the
        // public contract (partial result on TRANSCRIBE_ERR_ABORTED)
        // holds. Mirrors the result-shape rationale below: cohere
        // advertises max_timestamp_kind == NONE so we expose a single
        // text-only segment with zeroed timings and no token/word
        // substructure.
        auto commit_result = [&]() {
            cc->t_decode_us = ggml_time_us() - t_dec_start;

            if (generated_ids.empty()) return;

            const transcribe::Tokenizer & tok = cm->tok;

            std::string full = tok.decode(generated_ids.data(),
                                          static_cast<int>(generated_ids.size()));
            if (!full.empty() && full.front() == ' ') {
                full.erase(full.begin());
            }

            transcribe_context::SegmentEntry seg;
            seg.t0_ms       = 0;
            seg.t1_ms       = 0;
            seg.first_token = 0;
            seg.n_tokens    = 0;
            seg.first_word  = 0;
            seg.n_words     = 0;
            seg.text        = full;

            cc->segments.push_back(std::move(seg));
            cc->full_text   = std::move(full);
            cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
            cc->has_result  = true;
        };

        // --- Step 4: Autoregressive step passes ----------------------
        //
        // Two decoder loop variants; pick by primary backend kind.
        //
        //   GPU (Vulkan/Metal/CUDA/SYCL): build_step_graph — one static-
        //     topology graph for the whole utterance. KV writes go via
        //     ggml_set_rows at runtime kv_idx; flash-attn reads a fixed
        //     max_n_kv window with a runtime mask. Removes per-step
        //     graph_build + sched_alloc, which dominate dispatch overhead
        //     on GPUs.
        //
        //   CPU (incl. accelerator host-memory backends): build_decoder_
        //     graph_kv per step — n_kv grows with n_past, attention only
        //     reads the populated prefix. CPU has no dispatch overhead
        //     to amortize, so the static-graph bandwidth tax (reading
        //     max_n_kv KV slots when avg n_past is small) is a net loss.
        //
        // The prompt pass uses build_decoder_graph_kv on both paths
        // (already executed above) — only the per-token loop branches.
        int n_past = prompt_len;

        const bool primary_is_gpu =
            cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
            cm->plan.primary_kind != transcribe::BackendKind::Accel &&
            cm->plan.primary_kind != transcribe::BackendKind::Unknown;

        if (primary_is_gpu) {
            // ---------- Static-graph step path (GPU) ----------
            // max_n_kv: pad to next power of two with a 1024 floor.
            // Vulkan/Metal flash-attn dispatches faster on pow2 ne[1];
            // the slight bandwidth cost is amortized by removing
            // per-step graph_build + sched_alloc.
            int max_n_kv = 1024;
            while (max_n_kv < prompt_len + max_tokens) max_n_kv *= 2;
            if (max_n_kv > cc->kv_cache.n_ctx) max_n_kv = cc->kv_cache.n_ctx;

            if (!new_compute_ctx(8 * 1024 * 1024)) {
                std::fprintf(stderr,
                             "cohere run: new_compute_ctx failed (step)\n");
                commit_result();
                return TRANSCRIBE_ERR_GGUF;
            }
            StepBuild sb = build_step_graph(
                cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                max_n_kv, T_enc, cc->decoder_use_flash);
            if (sb.graph == nullptr || sb.argmax_out == nullptr) {
                std::fprintf(stderr,
                             "cohere run: build_step_graph failed\n");
                commit_result();
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
                std::fprintf(stderr,
                             "cohere run: sched_alloc_graph failed (step)\n");
                commit_result();
                return TRANSCRIBE_ERR_GGUF;
            }

            // Mask buffer: full max_n_kv span, reused host-side. Positions
            // already populated by the prompt pass [0, prompt_len) start
            // attendable; remaining slots are -inf until each step flips
            // its newly-written position to attendable.
            const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);
            for (int p = 0; p < prompt_len; ++p) step_mask[p] = mask_zero;

            for (int step = 1; step < max_tokens && next_token != eos_id;
                 ++step)
            {
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }
                if (n_past + 1 > max_n_kv) {
                    std::fprintf(stderr,
                                 "cohere run: hit max_n_kv=%d at n_past=%d\n",
                                 max_n_kv, n_past);
                    break;
                }

                int32_t token_val = next_token;
                int32_t pos_val   = n_past;
                int64_t kv_val    = n_past;
                ggml_backend_tensor_set(sb.token_id_in, &token_val, 0, sizeof(int32_t));
                ggml_backend_tensor_set(sb.pos_id_in,   &pos_val,   0, sizeof(int32_t));
                ggml_backend_tensor_set(sb.kv_idx_in,   &kv_val,    0, sizeof(int64_t));

                step_mask[n_past] = mask_zero;
                ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                        static_cast<size_t>(max_n_kv) *
                                        sizeof(ggml_fp16_t));

                if (const ggml_status gs =
                        ggml_backend_sched_graph_compute(cc->sched, sb.graph);
                    gs != GGML_STATUS_SUCCESS)
                {
                    std::fprintf(stderr,
                                 "cohere run: step compute failed (%d, n_past=%d)\n",
                                 static_cast<int>(gs), n_past);
                    commit_result();
                    return TRANSCRIBE_ERR_GGUF;
                }

                n_past += 1;
                cc->kv_cache.n    = n_past;
                cc->kv_cache.head = n_past;

                int32_t argmax_id = 0;
                ggml_backend_tensor_get(sb.argmax_out, &argmax_id, 0, sizeof(int32_t));
                next_token = argmax_id;

                if (next_token != eos_id) generated_ids.push_back(next_token);
            }
        } else {
            // ---------- Dynamic-graph step path (CPU) ----------
            // Reserve scheduler buffers with a worst-case single-token
            // graph (maximum n_past = n_ctx - 1). This ensures that
            // alloc_graph during the loop never triggers reallocation.
            if (new_compute_ctx(4 * 1024 * 1024)) {
                const int worst_n_past = cc->kv_cache.n_ctx - 1;
                DecoderBuild db_reserve = build_decoder_graph_kv(
                    cc->compute_ctx, cm->weights, cm->hparams,
                    cc->kv_cache,
                    /*n_tokens=*/1, worst_n_past, T_enc,
                    /*skip_log_softmax=*/true,
                    cc->decoder_use_flash);
                if (db_reserve.graph != nullptr) {
                    ggml_backend_sched_reserve(cc->sched, db_reserve.graph);
                }
            }

            for (int step = 1; step < max_tokens && next_token != eos_id;
                 ++step)
            {
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }

                if (n_past + 1 > cc->kv_cache.n_ctx) {
                    std::fprintf(stderr,
                                 "cohere run: KV cache full at n_past=%d, "
                                 "n_ctx=%d\n", n_past, cc->kv_cache.n_ctx);
                    break;
                }

                if (!new_compute_ctx(4 * 1024 * 1024)) break;

                DecoderBuild db_step = build_decoder_graph_kv(
                    cc->compute_ctx, cm->weights, cm->hparams,
                    cc->kv_cache,
                    /*n_tokens=*/1, n_past, T_enc,
                    /*skip_log_softmax=*/true,
                    cc->decoder_use_flash);
                if (db_step.out == nullptr || db_step.graph == nullptr) break;

                ggml_backend_sched_reset(cc->sched);
                if (!ggml_backend_sched_alloc_graph(cc->sched, db_step.graph))
                    break;

                int32_t token_id = next_token;
                int32_t pos_id   = n_past;
                ggml_backend_tensor_set(db_step.token_ids_in,
                                        &token_id, 0, sizeof(int32_t));
                ggml_backend_tensor_set(db_step.pos_ids_in,
                                        &pos_id, 0, sizeof(int32_t));

                if (const ggml_status gs =
                        ggml_backend_sched_graph_compute(cc->sched,
                                                         db_step.graph);
                    gs != GGML_STATUS_SUCCESS)
                {
                    break;
                }

                n_past += 1;
                cc->kv_cache.n    = n_past;
                cc->kv_cache.head = n_past;

                int32_t argmax_id = 0;
                ggml_backend_tensor_get(db_step.argmax_out, &argmax_id,
                                        0, sizeof(int32_t));
                next_token = argmax_id;

                if (next_token != eos_id) {
                    generated_ids.push_back(next_token);
                }
            }
        }

        // Build result hierarchy. Cohere advertises
        // max_timestamp_kind == NONE: "text but no alignment data".
        // That means no per-token, per-word, or per-segment timing
        // surface at all — not even a zero-timed placeholder. The
        // honest output shape is:
        //
        //   full_text   - the whole transcript
        //   segments[0] - single segment whose text == full_text,
        //                 with zeroed timings and zero first/n
        //                 counts (the caller asked for NONE, so
        //                 they should not see a token or word
        //                 structure)
        //   tokens      - empty
        //   words       - empty
        //
        // The previous implementation populated cc->tokens with
        // zero-timed entries and set seg.n_tokens to the token
        // count, which made transcribe_n_tokens > 0 and let a
        // caller enumerate token_text / token_t0_ms for a model
        // that is not supposed to expose alignment data. That
        // undercut the NONE contract. Drop them.
        commit_result();
    } // end Step 3 block

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name             = */ "cohere_asr",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::cohere
