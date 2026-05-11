// arch/parakeet/model.cpp - Parakeet family handler.
//
// Phase 4 step 1: load() now binds a runtime backend (Metal on Apple
// Silicon, CPU elsewhere or on Metal init failure). The second-stage
// gguf_init_from_file uses no_alloc=true so ggml builds the tensor
// catalog without touching the data section; we then allocate a
// backend buffer for every tensor via ggml_backend_alloc_ctx_tensors
// and stream the GGUF data section directly into it via
// ggml_backend_tensor_set. After load returns, every borrowed
// ggml_tensor* in `weights` has its data pointer rebound to backend
// memory.
//
// Stage 1 (header-only via transcribe::Loader, done by the dispatcher
// before we get here) is unchanged. Stage 2's gguf_context is freed
// before load() returns; the model's persistent state is ctx_meta +
// backend_buffer + backend_handle, freed in that order in the
// destructor.
//
// Phase 4 step 3a: run() now computes the mel front-end and runs
// the pre_encode subsampling stack of the encoder, with the result
// dumped via TRANSCRIBE_DUMP_DIR for the numerical accuracy harness.
// The remaining sub-stages (3b conformer FF1, 3c MHSA, 3d conv
// module, 3e FF2/norm_out, 3f loop over 24 blocks) progressively
// extend the encoder graph in arch/parakeet/encoder.cpp; the
// run-driver wiring here doesn't need to change. The decoder
// (predictor + joint + TDT) lands in phase 5. run() returns OK
// today, but every result accessor (full_text / segment / word /
// token) still returns its safe sentinel because nothing has
// populated the result yet.
//
// What's still missing relative to a fully-functional v1:
//   - Conformer blocks (3b-3f).
//   - Predictor + joint + TDT decode driver (phase 5).
//   - Quantization. fp32 only today.
//   - ggml_gallocr_t for repeat-call buffer reuse (phase 4 step 5).
//   - Public timings API. t_load_us is captured on the model but
//     not surfaced; phase 4 step 6 adds transcribe_timings.

#include "parakeet.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
// ggml-cpu.h no longer needed — threading is set via the registry
// proc address pattern, not ggml_backend_cpu_set_n_threads directly.
#include "gguf.h"

// No backend-specific #includes needed — ggml's device registry in
// ggml-backend.h discovers Metal/Vulkan/CUDA/BLAS at runtime.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

// Forward declaration so the anonymous-namespace functions below can
// take the address of the registry entry. The full definition lives at
// the bottom of the file (after the per-call function definitions it
// references).
extern const Arch arch;

// Cheap insurance against a future contributor accidentally dropping the
// inheritance relationship. Cost: zero. Caught at compile time.
static_assert(std::is_base_of_v<transcribe_model,   ParakeetModel>);
static_assert(std::is_base_of_v<transcribe_context, ParakeetContext>);

ParakeetContext::~ParakeetContext() {
    // Tear down per-call compute state. Order matters: the scheduler
    // owns the compute buffers, the context owns the tensor metadata,
    // both must be freed before the model's backend plan (which
    // outlives the context, owned by ParakeetModel).
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

ParakeetModel::~ParakeetModel() {
    // Teardown order: ctx_meta → backend_buffer → plan backends
    // (reverse init order).
    //
    // ctx_meta owns the tensor metadata (ggml_tensor structs); the
    // tensors' data pointers reference backend_buffer's interior.
    // Freeing ctx_meta drops the metadata only — the buffer is
    // independent. backend_buffer must be freed before the backends
    // because the buffer was allocated against a backend and may
    // hold a reference to it. Backends freed in reverse init order.
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        ggml_backend_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
    }
    // The CPU conv_pw F32 promotion ctx + buffer (no-op on GPU primary
    // backends; non-null only when promote_conv_pw_to_f32_on_cpu ran).
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

// Fuse inference-time BatchNorm into precomputed scale + bias tensors.
// BN eval: y = (x - mean) / sqrt(var + eps) * weight + bias
// Fused:   y = x * scale + shift
//   where scale = weight / sqrt(var + eps)
//         shift = bias - mean * scale
//
// Allocates fused tensors in a separate ggml context + CPU buffer on
// the model. Called once at load time; the fused tensors are then
// referenced by the encoder graph instead of the raw BN parameters.
transcribe_status fuse_batch_norm(ParakeetModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t d = m.hparams.enc_d_model;
    const size_t tensor_bytes = static_cast<size_t>(d) * sizeof(float);

    // Create a context for the fused tensors (2 per block).
    const size_t ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params = {ctx_size, nullptr, /*no_alloc=*/true};
    m.bn_fused_ctx = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) return TRANSCRIBE_ERR_BACKEND;

    // Create all tensors first, then allocate a buffer.
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
    }

    // Allocate on the CPU backend. The plan's scheduler list always
    // has CPU last as the fallback; for strict-CPU loads it is the
    // only entry.
    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(
        m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) return TRANSCRIBE_ERR_BACKEND;

    // Compute fused values from the raw BN tensors.
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

// On a CPU primary backend, dequantize the conformer 1×1 pointwise
// conv weights (pw1, pw2) from F16 back to F32. The shared quantizer
// (post commit 6fee9b9) routes Conformer ConvPw to F16 across all
// presets so GPU backends with native F16 compute (Metal, Vulkan,
// CUDA) get the bandwidth halving for free; on CPUs without native
// F16 arithmetic (Zen 2 and earlier) the per-element F16→F32
// upconvert per matmul erases the bandwidth win and outweighs the
// original cost.
//
// Today's parakeet GGUFs predate the universal F16 conv_pw policy
// and ship F32 conv_pw weights, so this function is a no-op against
// the current on-disk artifacts. The wiring is in place so the next
// regen does the right thing automatically. See the same comment
// block on the cohere side for the cost trade and the ~235 MB of
// "wasted" originals that stay resident in the main weight buffer
// (ggml's backend buffer model does not support freeing individual
// tensors).
//
// The machinery itself lives in transcribe::load_common — this
// function is just the parakeet-specific weight walk.
transcribe_status promote_conv_pw_to_f32_on_cpu(ParakeetModel & m) {
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
        m.plan, slots, "parakeet",
        &m.conv_pw_f32_ctx, &m.conv_pw_f32_buffer);
}

// Default variant string when the GGUF did not carry stt.variant.
// Defaulting belongs in the family handler, not the loader: each
// family knows its own canonical default and the loader does not.
constexpr const char k_default_variant[] = "tdt-0.6b-v2";

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
    // The dispatcher has already verified out_model is non-null and
    // the loader has a valid gguf_context with general.architecture
    // set. *out_model is currently null (the dispatcher cleared it).
    // params->backend is consumed below in the backend init block;
    // params->gpu_device is reserved per the public header contract
    // and is not yet honored (multi-device selection is a future
    // release).

    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<ParakeetModel>();
    m->arch      = &arch;
    m->t_load_us = 0; // will be set just before the successful return

    // Variant defaulting belongs here, not in the loader: each family
    // knows its own canonical default. We accept the GGUF's stt.variant
    // verbatim if present and only fall back when it is empty.
    //
    // The variant string is descriptive metadata for users — it
    // surfaces through transcribe_model_variant_string but does NOT
    // drive any behavior decision. Capability differences between
    // Parakeet variants (v2 English vs v3 multilingual, etc.) are
    // expressed as stt.capability.* and general.languages KV, read
    // below. The only PLAN.md-blessed exception is the decoder-kind
    // prefix dispatch for hybrid models, which Parakeet v1 does not
    // exercise (only TDT ships).
    if (loader.variant().empty()) {
        m->variant = k_default_variant;
    } else {
        m->variant = loader.variant();
    }

    // backend is set below, after the runtime backend is initialized.
    // Until then m->backend stays empty per the public ABI semantic
    // for "no backend currently bound".
    m->backend.clear();

    // Capability resolution, KV-driven. Order matters:
    //
    //   1. apply_family_invariants populates the family defaults
    //      (native_sample_rate, supports_translate). These are the
    //      "the architecture supplies a default" half of PLAN.md's
    //      capability precedence rule.
    //   2. read_capability_kv overlays any stt.capability.* KV the
    //      converter wrote. This is the "GGUF KV is authoritative"
    //      half. Absent KV leaves the family default in place;
    //      present-but-wrong-type KV is a converter bug and propagates
    //      as TRANSCRIBE_ERR_GGUF.
    //
    // Information-gap default for the languages chain. read_languages_kv
    // below leaves these in place if general.languages is absent — the
    // result is the public ABI's "n_languages == 0, languages == nullptr"
    // observation, documented in include/transcribe.h as "we don't
    // know" rather than "the model has no languages". A v3 multilingual
    // GGUF supplies the real list via general.languages and
    // read_languages_kv overwrites the defaults via set_languages.
    apply_family_invariants(m->caps);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Languages from general.languages, installed via set_languages
    // (which keeps the language pointer chain in caps in sync). The
    // strings are copied into the model so the loader's gguf_context
    // can be freed after this point without invalidating them.
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Tokenizer ingest. Reads tokenizer.ggml.* keys from the loader's
    // gguf_context and copies the strings / scores / token types into
    // the Tokenizer's own storage. Like read_languages_kv above, this
    // does not retain a borrow into the gguf_context.
    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }

    // Architecture KV (encoder / predictor / joint / frontend dims).
    // Read directly from the loader's gguf_context — these are
    // hparams, not tensor data, so we can do this before the second
    // open. read_parakeet_hparams enforces cross-field invariants
    // (d_model divisible by n_heads, etc.) so we know the shapes
    // we're about to validate against are themselves consistent.
    if (const transcribe_status st = read_parakeet_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Construct the mel front-end now that hparams are available.
    // The MelFrontend constructor precomputes the periodic Hann
    // window + Slaney mel filterbank (~50 ms on jfk-sized configs);
    // doing it here amortizes the cost across every transcribe_run
    // on every context derived from this model. The instance is
    // const-after-construction and thread-safe per its contract.
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
        cfg.normalize    = m->hparams.fe_normalize;
        m->mel.emplace(cfg);
    }

    // Stage 2: reopen the file with no_alloc=true + ctx=&ctx_meta.
    // ggml builds the tensor catalog (metadata only — no data
    // buffers); we then bind a runtime backend, allocate a backend
    // buffer for every tensor in ctx_meta, and stream the GGUF data
    // section directly into that backend buffer. After this
    // sequence, ggml_get_tensor(ctx_meta, name)->data points at
    // backend memory, not host malloc.
    //
    // The first gguf_context held by the loader stays alive until
    // load() returns (the dispatcher owns its lifetime); we don't
    // touch it here. The second gguf_context (gguf_data below) is
    // freed before load() returns.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) {
        // ggml has already logged a structured error. The first-stage
        // open succeeded so this is genuinely unexpected — most
        // likely a race where the file changed between the two opens,
        // or a permissions transition. Surface as ERR_GGUF.
        return TRANSCRIBE_ERR_GGUF;
    }

    // Validate every tensor against the canonical catalog. find_tensor
    // only inspects type + shape (not data pointers), so this works
    // before the backend buffer is bound.
    if (const transcribe_status st =
            build_parakeet_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // Resolve the backend plan via ggml's device registry. The
    // caller's requested backend (AUTO, CPU, METAL, VULKAN) is
    // honored here; see transcribe-backend.h + transcribe-load-common.h
    // for the full semantic. This is runtime-dynamic: the library
    // code has no compile-time #if TRANSCRIBE_METAL / #if
    // TRANSCRIBE_VULKAN guards. Whatever backends ggml was compiled
    // with are discovered here.
    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;

    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "parakeet", m->plan);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // Label for the public API: report the primary backend.
    m->backend = ggml_backend_name(m->plan.primary);

    // Allocate a backend buffer for every tensor in ctx_meta on the
    // primary backend. After this returns, each ggml_tensor in
    // ctx_meta has its `buffer` and `data` slot bound to the backend
    // allocation. We still need to upload the actual weight bytes
    // from the GGUF data section.
    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        std::fprintf(stderr,
                     "parakeet: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Stream tensor data from the GGUF file into the backend buffer
    // slots. See transcribe-load-common.h for the shared loop; it
    // uses ggml_backend_tensor_set so this works for both host-
    // memory backends (CPU, Metal on Apple Silicon) and discrete
    // GPUs.
    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "parakeet");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // gguf_data only carried tensor offsets + names; both have been
    // consumed by build_parakeet_weights and the streaming loop
    // above. m->ctx_meta + m->backend_buffer + m->backend_handle +
    // m->weights are now the model's entire state.
    gguf_free(gguf_data);

    // Fuse BatchNorm parameters into scale + bias for the encoder
    // graph. This replaces 4 elementwise ops per block with 2. Skipped
    // for streaming variants whose conv module uses LayerNorm (raw
    // bn.weight / bn.bias travel through the graph as LN scale/bias,
    // and there are no running stats to fuse against).
    if (m->hparams.enc_conv_norm_type
            == ParakeetHParams::ConvNormType::BatchNorm)
    {
        if (const transcribe_status st = fuse_batch_norm(*m);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
    }

    // On CPU primary backend, dequantize conv pointwise weights back
    // to F32 — Zen 2 class CPUs don't have native F16 compute and the
    // upconvert cost per matmul outweighs the bandwidth savings from
    // the smaller weight. No-op on GPU backends and on parakeet GGUFs
    // that predate the universal F16 conv_pw quantizer policy (today,
    // all of them).
    if (const transcribe_status st = promote_conv_pw_to_f32_on_cpu(*m);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Phase 5: build the host mirror of the predictor + joint
    // weights. This is a one-shot copy from backend memory into
    // std::vector<float>s on the model. The decoder runs on host
    // (see decoder.h for the rationale), and centralizing the
    // backend → host copy here means the per-call decode loop
    // pays no readback cost.
    if (const transcribe_status st = build_host_decoder_weights(*m, m->host_decoder);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Capture wall-clock load time on the model. The accumulator
    // is on the base transcribe_model so the public timings API
    // can read it without per-family casts.
    m->t_load_us = ggml_time_us() - t_load_start;

    // Hand off to the caller. From here on the caller owns the model
    // and must call transcribe_model_free.
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(
    transcribe_model *                model,
    const transcribe_context_params * params,
    transcribe_context **             out_ctx)
{
    // The central dispatcher has already null-checked model, params,
    // and out_ctx. We still want to be sure the model we received is
    // actually a ParakeetModel — `arch` is the discriminator. In debug
    // builds an assert would be louder, but a routing mistake should
    // not silently corrupt anything in release either, so we
    // defensively check.
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto pc = std::make_unique<ParakeetContext>();
    pc->model     = model;
    pc->n_threads = params->n_threads;
    pc->kv_type   = params->kv_type;

    *out_ctx = pc.release();
    return TRANSCRIBE_OK;
}

transcribe_status run(
    transcribe_context *      ctx,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    // The dispatcher (transcribe.cpp) has already enum-range
    // validated params->task / params->timestamps, rejected
    // TRANSLATE against this family's supports_translate=false, and
    // rejected any timestamp request finer than our advertised
    // max_timestamp_kind=TOKEN. What arrives here is guaranteed
    // sane — we only need to resolve AUTO and downcast finer-grained
    // output to the requested ceiling when the result is built.
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * pc = static_cast<ParakeetContext *>(ctx);
    auto * pm = static_cast<ParakeetModel *>(pc->model);
    if (pm == nullptr || pm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-run abort check. Parakeet is non-chunked today; this is the
    // single observation point. A caller that wants to veto a run
    // without paying encoder cost flips the callback's state and the
    // next transcribe_run short-circuits here.
    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    // Initialize the debug dumper from TRANSCRIBE_DUMP_DIR. Idempotent
    // — only the first call has effect.
    transcribe::debug::init();

    // ----- Mel front-end -------------------------------------------
    //
    // The MelFrontend was constructed once at load() time and lives
    // on the model. compute() is documented thread-safe across
    // contexts since the instance is const-after-construction.
    if (!pm->mel.has_value()) {
        std::fprintf(stderr,
                     "parakeet run: model has no MelFrontend (load skipped?)\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t t_mel_start = ggml_time_us();
    int mel_n_mels   = 0;
    int mel_n_frames = 0;
    if (const transcribe_status mst = pm->mel->compute(
            pcm, static_cast<size_t>(n_samples),
            pc->mel_buf, mel_n_mels, mel_n_frames);
        mst != TRANSCRIBE_OK)
    {
        std::fprintf(stderr,
                     "parakeet run: MelFrontend::compute failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    pc->t_mel_us = ggml_time_us() - t_mel_start;

    // ----- Reset per-call compute state ----------------------------
    //
    // Free the previous run's compute_ctx (tensor metadata + cgraph).
    // The gallocr persists across calls — it reuses its internal
    // buffer when the graph topology is unchanged (same audio length)
    // and reallocates automatically on topology change. The
    // encoder_out borrowed pointer is invalidated here.
    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    pc->encoder_out = nullptr;

    // ----- Build the compute context + encoder graph ---------------
    //
    // The compute context only holds tensor metadata + the cgraph;
    // tensor data is allocated separately by
    // ggml_backend_alloc_ctx_tensors below. The full 24-block
    // encoder graph has ~2200 tensors plus an 8192-slot cgraph
    // (see encoder.cpp's ggml_new_graph_custom call). Empirically
    // this uses ~1.23 MB of metadata arena (logged once during
    // step 5 polish). 4 MB gives ~3x headroom for future per-block
    // op additions (BN fold, gallocr metadata, etc.).
    {
        ggml_init_params init_params {};
        init_params.mem_size   = 4 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        pc->compute_ctx = ggml_init(init_params);
        if (pc->compute_ctx == nullptr) {
            std::fprintf(stderr,
                         "parakeet run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Resolve kv_type from the public enum to ggml_type.
    // GGML_TYPE_COUNT is the sentinel for "auto" inside the encoder.
    ggml_type resolved_kv = GGML_TYPE_COUNT; // auto
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    EncoderBuild eb = build_encoder_graph(
        pc->compute_ctx, pm->weights, pm->hparams, mel_n_frames,
        resolved_kv, pm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        // build_encoder_graph already logged the diagnostic.
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Allocate compute tensors via scheduler --------------------
    //
    // The multi-backend scheduler dispatches ops to the best backend
    // (GPU for matmuls if available, BLAS for CPU matmuls, CPU for
    // the rest). It also manages compute buffer allocation with
    // live-range packing. Created lazily, persists across calls.
    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(
            pm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(pm->plan.scheduler_list.size()),
            /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            std::fprintf(stderr,
                         "parakeet run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "parakeet run: ggml_backend_sched_alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload the mel into the input tensor. The C++ MelFrontend
    // produces a row-major [num_mels, n_frames] buffer, which is
    // byte-identical to the ggml ne=[n_frames, num_mels, 1, 1]
    // layout the encoder builder created (see encoder.cpp's layout
    // cheat sheet for the derivation).
    ggml_backend_tensor_set(eb.mel_in, pc->mel_buf.data(),
                            0, pc->mel_buf.size() * sizeof(float));

    // Optional dump of the mel input for the numerical accuracy
    // harness. Gated on TRANSCRIBE_DUMP_DIR; zero-cost when unset.
    transcribe::debug::dump_tensor(
        "enc.mel.in", eb.mel_in, "encoder.mel");

    // ----- Host-side sinusoidal positional embedding ---------------
    //
    // Sub-stage 3c+: the attention sub-block needs a precomputed
    // sin/cos pos_emb input tensor. The encoder graph builder
    // allocated `eb.pos_emb_in` with ne=[d_model, pos_len, 1, 1]
    // (after it learned T_enc from pre_encode); we fill it here.
    //
    // Full attention (att_context_left == -1):
    //   positions[i] = (T_enc - 1) - i  for i in [0, 2*T_enc-1)
    // Local attention (NeMo LocalAttRelPositionalEncoding):
    //   positions[i] = W_left - i      for i in [0, W_left+W_right+1)
    // In both cases the per-row sinusoid is identical:
    //   div_term[k]  = exp(2k * -ln(10000) / d_model)
    //   pe[i, 2k]    = sin(positions[i] * div_term[k])
    //   pe[i, 2k+1]  = cos(positions[i] * div_term[k])
    //
    // The on-disk row-major shape is (pos_len, d_model). In ggml ne
    // (fast-to-slow) that's [d_model, pos_len, 1, 1] which the
    // encoder builder created.
    if (eb.pos_emb_in != nullptr) {
        const int d_model = pm->hparams.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);

        // Recover the position of "relative offset 0" inside the buffer.
        // - Full attention / chunked_limited streaming: pos_len = 2*T_enc - 1,
        //   zero index = T_enc - 1 (= (pos_len-1)/2).
        // - Regular local attention: pos_len = W_left + W_right + 1,
        //   zero index = W_left. Only applies to the Regular style — the
        //   ChunkedLimited path always uses the full RelPositionalEncoding.
        const bool is_chunked =
            (pm->hparams.enc_att_context_style ==
                 ParakeetHParams::AttContextStyle::ChunkedLimited);
        const bool is_local_pe =
            (!is_chunked) &&
            (pm->hparams.enc_att_context_left >= 0 &&
             pm->hparams.enc_att_context_right >= 0);
        const int zero_index = is_local_pe
            ? pm->hparams.enc_att_context_left
            : (pos_len - 1) / 2;

        pc->pos_buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);

        // div_term[k] = exp(2k * -ln(10000) / d_model) for k in [0, d_model/2)
        pc->pos_div_term.resize(static_cast<size_t>(d_model / 2));
        const float ln_10000 = std::log(10000.0f);
        for (int k = 0; k < d_model / 2; ++k) {
            pc->pos_div_term[static_cast<size_t>(k)] =
                std::exp(static_cast<float>(2 * k) *
                         (-ln_10000 / static_cast<float>(d_model)));
        }

        for (int i = 0; i < pos_len; ++i) {
            const float pos = static_cast<float>(zero_index - i);
            float * row = pc->pos_buf.data() + static_cast<size_t>(i) * d_model;
            for (int k = 0; k < d_model / 2; ++k) {
                const float div = pc->pos_div_term[static_cast<size_t>(k)];
                row[2 * k]     = std::sin(pos * div);
                row[2 * k + 1] = std::cos(pos * div);
            }
        }

        ggml_backend_tensor_set(eb.pos_emb_in, pc->pos_buf.data(),
                                0, pc->pos_buf.size() * sizeof(float));

        transcribe::debug::dump_tensor(
            "enc.pos_emb", eb.pos_emb_in, "encoder.pos_emb");
    }

    // ChunkedLimited attention mask (streaming variants). pos_emb above
    // stays at the full 2T-1 length and rel_pos contributes its usual
    // (q-k) bias; this mask adds 0 on (q, k) pairs whose chunk indices
    // are in [q_chunk - left_chunks, q_chunk] and -INF elsewhere, so
    // softmax zeroes everything outside the cache-aware band. NeMo
    // semantics: chunk_size = att_context_right + 1, left_chunks =
    // att_context_left / chunk_size.
    if (eb.chunked_mask_in != nullptr) {
        const int T_enc       = static_cast<int>(eb.chunked_mask_in->ne[0]);
        const int chunk_size  = pm->hparams.enc_att_context_right + 1;
        const int left_chunks = (chunk_size > 0)
            ? (pm->hparams.enc_att_context_left / chunk_size)
            : 0;

        std::vector<float> mask_buf(
            static_cast<size_t>(T_enc) * static_cast<size_t>(T_enc));
        // ggml ne = [T_k, T_q, 1, 1], row-major in the host buffer is
        // contiguous along T_k (ne[0]). Indexing: mask[q, k] lives at
        // offset (q * T_enc + k).
        for (int q = 0; q < T_enc; ++q) {
            const int q_chunk = (chunk_size > 0) ? (q / chunk_size) : 0;
            const int k_min_chunk =
                (q_chunk - left_chunks > 0) ? (q_chunk - left_chunks) : 0;
            const int k_min = k_min_chunk * chunk_size;
            const int k_max = (q_chunk + 1) * chunk_size; // exclusive
            float * row = mask_buf.data() + static_cast<size_t>(q) * T_enc;
            for (int k = 0; k < T_enc; ++k) {
                row[k] = (k >= k_min && k < k_max)
                    ? 0.0f
                    : -std::numeric_limits<float>::infinity();
            }
        }

        ggml_backend_tensor_set(eb.chunked_mask_in, mask_buf.data(),
                                0, mask_buf.size() * sizeof(float));
        transcribe::debug::dump_tensor(
            "enc.attn.chunked_mask",
            eb.chunked_mask_in,
            "encoder.attn.chunked_mask");
    }

    // ----- Set thread count on all backends --------------------------
    //
    // Whisper.cpp pattern: iterate the scheduler's backends and set
    // n_threads via the registry proc address. GPU backends ignore
    // this; CPU and BLAS backends use it.
    {
        int n_threads = pc->n_threads;
        if (n_threads <= 0) {
            n_threads = std::min(8, std::max(1, static_cast<int>(
                std::thread::hardware_concurrency())));
        }
        for (int i = 0; i < ggml_backend_sched_get_n_backends(pc->sched); ++i) {
            ggml_backend_t be = ggml_backend_sched_get_backend(pc->sched, i);
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

    // ----- Compute --------------------------------------------------
    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(pc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "parakeet run: ggml_backend_sched_graph_compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    pc->t_encode_us = ggml_time_us() - t_enc_start;
    pc->t_decode_us = 0;

    // ----- Dump intermediates (debug only) -------------------------
    //
    // Gated on TRANSCRIBE_DUMP_DIR via transcribe::debug::dump_tensor;
    // when the env var is unset these are zero-cost. Each field on
    // EncoderDumps is a borrowed pointer into the compute_ctx;
    // nullptr fields mean "this sub-stage isn't wired yet".
    auto try_dump = [](const char * name, ggml_tensor * t,
                       const char * stage)
    {
        if (t != nullptr) {
            transcribe::debug::dump_tensor(name, t, stage);
        }
    };
    try_dump("enc.pre_encode.out",   eb.dumps.pre_encode_out,   "encoder.pre_encode");
    try_dump("enc.block.0.ff1",      eb.dumps.block0_after_ff1, "encoder.block0.ff1");
    try_dump("enc.block.0.attn",     eb.dumps.block0_after_attn,"encoder.block0.attn");
    try_dump("enc.block.0.conv",     eb.dumps.block0_after_conv,"encoder.block0.conv");
    try_dump("enc.block.0.ff2",      eb.dumps.block0_after_ff2, "encoder.block0.ff2");
    try_dump("enc.block.0.out",      eb.dumps.block0_out,       "encoder.block0.out");
    // Mid- and last-block spot-check dumps. File name encodes the
    // actual block index (scales with n_layers): 0/12/23 for 24-layer,
    // 0/21/41 for 42-layer, 0/8/16 for 17-layer. last_block_out
    // aliases final_out (same ggml pointer); we dump it under the
    // per-variant block name AND under "enc.final" so both validation
    // entries find data.
    if (eb.dumps.mid_block_out != nullptr && eb.dumps.mid_block_idx >= 0) {
        char name[64];
        std::snprintf(name, sizeof(name), "enc.block.%d.out", eb.dumps.mid_block_idx);
        transcribe::debug::dump_tensor(name, eb.dumps.mid_block_out,
                                       "encoder.block.mid.out");
    }
    if (eb.dumps.last_block_out != nullptr && eb.dumps.last_block_idx >= 0) {
        char name[64];
        std::snprintf(name, sizeof(name), "enc.block.%d.out", eb.dumps.last_block_idx);
        transcribe::debug::dump_tensor(name, eb.dumps.last_block_out,
                                       "encoder.block.last.out");
    }
    // Per-block dump for the layer-by-layer divergence bisect (gated
    // by TRANSCRIBE_DUMP_ALL_BLOCKS env var; mark_tensor_for_dump was
    // applied per-block in encoder.cpp's loop, so the data is
    // preserved across the scheduler's compute. The block 0 / mid /
    // last names are still emitted above; this just adds the
    // remaining blocks under "enc.block.<i>.out" so they collide
    // (overwrite) consistently with the named dumps.
    if (std::getenv("TRANSCRIBE_DUMP_ALL_BLOCKS") != nullptr) {
        for (size_t i = 0; i < eb.dumps.all_block_outs.size(); ++i) {
            ggml_tensor * t = eb.dumps.all_block_outs[i];
            if (t == nullptr) continue;
            char name[64];
            std::snprintf(name, sizeof(name), "enc.block.%zu.out", i);
            transcribe::debug::dump_tensor(name, t,
                                           "encoder.block.bisect");
        }
    }
    // Sub-block intermediates for blocks listed in
    // TRANSCRIBE_DUMP_SUB_BLOCKS. Each entry was populated by the
    // sub-block observer (see encoder.cpp) at graph-build time and
    // passed through the scheduler intact via mark_tensor_for_dump.
    for (const auto & p : eb.dumps.sub_block_dumps) {
        if (p.second == nullptr) continue;
        transcribe::debug::dump_tensor(p.first.c_str(), p.second,
                                       "encoder.block.subblock");
    }
    try_dump("enc.final",            eb.dumps.final_out,        "encoder.final");

    // Stash the encoder output for the accuracy test (it reaches in
    // via the ParakeetContext view) and as a borrowed reference for
    // the readback below.
    pc->encoder_out = eb.out;

    // ----- TDT decode --------------------------------------------
    //
    // Read the encoder output to host, run the decoder, then build
    // the result hierarchy (segments / words / tokens / full text)
    // for the public accessors. The encoder activation has ne=
    // [d_enc, T_enc, 1, 1] in fast-to-slow ggml order, so the
    // contiguous byte range is row-major [T_enc, d_enc] from the
    // host's POV — exactly what decode_tdt_greedy expects.
    pc->clear_result();
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_enc <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "parakeet run: encoder output has degenerate shape "
                     "[%d, %d]\n", d_enc, T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }
    pc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, pc->enc_host.data(), 0,
                            pc->enc_host.size() * sizeof(float));

    // Optional dump of the encoder output as the decoder sees it,
    // so the bring-up loop can verify the readback is faithful
    // before chasing decoder bugs.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, d_enc };
        transcribe::debug::dump_host_f32(
            "dec.enc_out", pc->enc_host.data(),
            static_cast<long long>(pc->enc_host.size()),
            shape, 2, "decoder.enc_out");
    }

    pc->raw_tokens.clear();
    const int64_t t_dec_start = ggml_time_us();
    {
        transcribe_status st = TRANSCRIBE_OK;
        switch (pm->host_decoder.head_kind) {
            case HostHeadKind::TDT:
                st = decode_tdt_greedy(pm->host_decoder, pc->enc_host.data(),
                                       T_enc, d_enc, pc->raw_tokens);
                break;
            case HostHeadKind::RNNT:
                st = decode_rnnt_greedy(pm->host_decoder, pc->enc_host.data(),
                                        T_enc, d_enc, pc->raw_tokens);
                break;
            case HostHeadKind::CTC:
                st = decode_ctc_greedy(pm->host_decoder, pc->enc_host.data(),
                                       T_enc, d_enc, pc->raw_tokens);
                break;
        }
        if (st != TRANSCRIBE_OK) return st;
    }
    pc->t_decode_us = ggml_time_us() - t_dec_start;

    // ----- Build the public result hierarchy ----------------------
    //
    // Conversion: each emitted token's `step_at_emit` is an encoder
    // frame index. The encoder downsamples by `subsampling_factor`
    // relative to the mel hop, so one encoder frame corresponds to
    // `subsampling_factor * hop_length / sample_rate` seconds of
    // audio. For Parakeet 0.6B v2/v3 that's 8 * 160 / 16000 = 0.08 s
    // per frame, i.e. 80 ms.
    const double frame_to_ms =
        1000.0 *
        static_cast<double>(pm->hparams.enc_subsampling_factor) *
        static_cast<double>(pm->hparams.fe_hop_length) /
        static_cast<double>(pm->hparams.fe_sample_rate);

    // SentencePiece word-boundary marker U+2581 ("▁"). UTF-8 bytes
    // 0xE2 0x96 0x81. A token whose decoded text starts with the
    // marker opens a new word; otherwise it extends the current
    // word. We use the raw NeMo piece (not the post-decode "▁ →
    // space" form) because the post-decode form leads with a space
    // and that's a fragile thing to compare against — the raw
    // piece's leading 3 UTF-8 bytes are unambiguous.
    constexpr const char k_sp_marker[] = "\xE2\x96\x81";
    constexpr int        k_sp_marker_len = 3;

    const transcribe::Tokenizer & tok = pm->tok;

    pc->tokens.reserve(pc->raw_tokens.size());
    for (const TdtToken & rt : pc->raw_tokens) {
        transcribe_context::TokenEntry te;
        te.id    = rt.id;
        te.p     = rt.p;
        te.t0_ms = static_cast<int64_t>(
            std::llround(frame_to_ms * static_cast<double>(rt.step_at_emit)));
        // Per-token duration: clamp the duration_frames=0 case to a
        // visually-distinct minimum of 0 ms (== t0). The result is a
        // "point in time" token with no width.
        const double end_frame =
            static_cast<double>(rt.step_at_emit) +
            static_cast<double>(rt.duration_frames);
        te.t1_ms = static_cast<int64_t>(std::llround(frame_to_ms * end_frame));
        // Decode just this single token to get its visible text
        // fragment. The decoder strips the SentencePiece marker via
        // Tokenizer::decode (▁ → space), so the leading character
        // for word-starting tokens is an ASCII space.
        te.text  = tok.decode(&rt.id, 1);
        // seg_index / word_index filled below.
        pc->tokens.push_back(std::move(te));
    }

    // Word + segment construction. v1 produces a single segment
    // covering the entire clip; words are split on SentencePiece
    // marker boundaries. Empty result → no segment, no words, no
    // text — exactly what the public accessors return as their
    // safe-sentinel state when there are no tokens.
    if (!pc->tokens.empty()) {
        // Single segment.
        transcribe_context::SegmentEntry seg;
        seg.t0_ms       = pc->tokens.front().t0_ms;
        seg.t1_ms       = pc->tokens.back().t1_ms;
        seg.first_token = 0;
        seg.n_tokens    = static_cast<int>(pc->tokens.size());
        seg.first_word  = 0;
        // n_words filled after we count words below.

        // Walk the raw token ids again to detect word boundaries.
        // The first token always opens word 0; a token whose raw
        // piece begins with the SentencePiece marker opens a new
        // word. Continuation tokens (no marker) extend the current
        // word.
        transcribe_context::WordEntry  cur_word;
        bool                        cur_word_open = false;

        auto open_new_word = [&](int token_index, const transcribe_context::TokenEntry & tk) {
            if (cur_word_open) {
                cur_word.t1_ms   = pc->tokens[static_cast<size_t>(token_index - 1)].t1_ms;
                cur_word.n_tokens = token_index - cur_word.first_token;
                pc->words.push_back(std::move(cur_word));
                cur_word = transcribe_context::WordEntry{};
            }
            cur_word.t0_ms       = tk.t0_ms;
            cur_word.first_token = token_index;
            cur_word.seg_index   = 0;
            cur_word_open        = true;
        };

        for (size_t i = 0; i < pc->tokens.size(); ++i) {
            const auto & tk        = pc->tokens[i];
            const auto & raw_piece = tok.token(tk.id); // empty if id OOR
            const bool starts_word =
                (i == 0) ||
                (raw_piece.size() >= static_cast<size_t>(k_sp_marker_len) &&
                 std::memcmp(raw_piece.data(), k_sp_marker, k_sp_marker_len) == 0);
            if (starts_word) {
                open_new_word(static_cast<int>(i), tk);
            }
            // Whether we just opened a new word or not, the current
            // word now contains this token. seg/word back-pointers:
            pc->tokens[i].seg_index  = 0;
            pc->tokens[i].word_index = static_cast<int>(pc->words.size());
        }
        // Close out the trailing word.
        if (cur_word_open) {
            cur_word.t1_ms    = pc->tokens.back().t1_ms;
            cur_word.n_tokens = static_cast<int>(pc->tokens.size()) - cur_word.first_token;
            pc->words.push_back(std::move(cur_word));
        }

        // Materialize each word's text via the tokenizer (so the
        // SentencePiece "▁ → space" substitution runs on the whole
        // span at once, including any mid-word continuation
        // pieces). The leading space from a word-opener token is
        // trimmed: a "word" should not include its own leading
        // space.
        std::vector<int> id_buf;
        for (auto & wd : pc->words) {
            id_buf.clear();
            id_buf.reserve(static_cast<size_t>(wd.n_tokens));
            for (int j = 0; j < wd.n_tokens; ++j) {
                id_buf.push_back(pc->tokens[static_cast<size_t>(wd.first_token + j)].id);
            }
            std::string text = tok.decode(id_buf.data(), wd.n_tokens);
            if (!text.empty() && text.front() == ' ') {
                text.erase(text.begin());
            }
            wd.text = std::move(text);
        }

        seg.n_words = static_cast<int>(pc->words.size());

        // Build the full text for the segment via the tokenizer
        // (one decode call over every id, so the SentencePiece
        // substitution sees the whole sequence).
        std::vector<int> all_ids;
        all_ids.reserve(pc->tokens.size());
        for (const auto & tk : pc->tokens) all_ids.push_back(tk.id);
        std::string full = tok.decode(all_ids.data(),
                                      static_cast<int>(all_ids.size()));
        if (!full.empty() && full.front() == ' ') {
            full.erase(full.begin());
        }
        seg.text       = full;
        pc->full_text  = std::move(full);
        pc->segments.push_back(std::move(seg));

        // Parakeet TDT produces token-level timestamps from the
        // encoder frame indices (each emitted token's
        // step_at_emit). Word and segment timestamps are derived
        // by aggregating contained tokens. TOKEN is therefore the
        // family's max_timestamp_kind, and everything above has
        // already been populated at TOKEN granularity.
        //
        // Clamp to the caller's requested ceiling. AUTO resolves to
        // the family max (TOKEN). A coarser request elides the
        // finer levels: WORD drops the token list, SEGMENT drops
        // tokens+words, NONE drops tokens+words and zeros the
        // segment's t0/t1 so nothing dresses up as alignment data
        // the caller did not ask for.
        //
        // The dispatcher has already rejected any request finer
        // than TOKEN, so after the AUTO resolution below, eff is in
        // {NONE, SEGMENT, WORD, TOKEN}.
        transcribe_timestamp_kind eff = params->timestamps;
        if (eff == TRANSCRIBE_TIMESTAMPS_AUTO) {
            eff = pm->caps.max_timestamp_kind; // = TOKEN for parakeet
        }
        if (eff == TRANSCRIBE_TIMESTAMPS_NONE) {
            // Keep the segment and its text, but drop alignment
            // data. Tokens + words are a finer granularity than
            // the caller asked for; segment timings are alignment
            // too, so zero them.
            pc->tokens.clear();
            pc->words.clear();
            auto & s = pc->segments.back();
            s.t0_ms      = 0;
            s.t1_ms      = 0;
            s.first_word = 0;
            s.n_words    = 0;
            s.first_token = 0;
            s.n_tokens    = 0;
        } else if (eff == TRANSCRIBE_TIMESTAMPS_SEGMENT) {
            // Keep segment timings, drop token + word tables.
            pc->tokens.clear();
            pc->words.clear();
            auto & s = pc->segments.back();
            s.first_word  = 0;
            s.n_words     = 0;
            s.first_token = 0;
            s.n_tokens    = 0;
        } else if (eff == TRANSCRIBE_TIMESTAMPS_WORD) {
            // Keep segment + word timings, drop the token table.
            // Every back-reference into the cleared token table
            // must also be zeroed so a caller iterating words or
            // the segment can never index into a now-empty
            // pc->tokens. The word_* / segment_* accessors return
            // safe sentinels when n_tokens == 0, which is the
            // documented behavior for "coarser than requested."
            pc->tokens.clear();
            for (auto & w : pc->words) {
                w.first_token = 0;
                w.n_tokens    = 0;
            }
            auto & s = pc->segments.back();
            s.first_token = 0;
            s.n_tokens    = 0;
        }
        // TRANSCRIBE_TIMESTAMPS_TOKEN: nothing to elide.

        pc->result_kind = eff;
        pc->has_result  = true;
    }

    return TRANSCRIBE_OK;
}

} // namespace

// `extern const` to force external linkage. Without `extern`, a
// namespace-scope `const` object has internal linkage in C++ and the
// forward declaration in transcribe-arch.cpp would not resolve.
extern const Arch arch = {
    /* .name         = */ "parakeet",
    /* .load         = */ load,
    /* .init_context = */ init_context,
    /* .run          = */ run,
};

} // namespace transcribe::parakeet
