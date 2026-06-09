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

#include "transcribe/parakeet.h"

#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"
#include "transcribe-log.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
// ggml-cpu.h no longer needed — threading is set via the registry
// proc address pattern, not ggml_backend_cpu_set_n_threads directly.
#include "gguf.h"

// No backend-specific #includes needed — ggml's device registry in
// ggml-backend.h discovers Metal/Vulkan/CUDA/BLAS at runtime.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
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
static_assert(std::is_base_of_v<transcribe_session, ParakeetSession>);

ParakeetSession::~ParakeetSession() {
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

    // Streaming cache tensors live in their own ggml_context with
    // their own backend buffer (allocated lazily on first stream_begin
    // for ChunkedLimited variants). Free in the same order as the
    // weights buffer/session in ParakeetModel: backend buffer first (it
    // may hold a backend ref), then the ggml_context that owned the
    // tensor metadata.
    if (stream_caches.buffer != nullptr) {
        ggml_backend_buffer_free(stream_caches.buffer);
        stream_caches.buffer = nullptr;
    }
    if (stream_caches.ctx != nullptr) {
        ggml_free(stream_caches.ctx);
        stream_caches.ctx = nullptr;
    }
    stream_caches.last_channel.clear();
    stream_caches.last_time.clear();
    stream_caches.initialized = false;
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
    // The CPU conv_pw F32 promotion session + buffer (no-op on GPU primary
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

// Resolve the runtime language hint to a prompt-table index for the
// multilingual prompt-conditioned variants
// (NeMo's EncDecRNNTBPEModelWithPrompt). The dictionary carries both
// canonical BCP-47 codes ("en-US") and short aliases ("en") that map
// to the same index — we just do an exact-string lookup.
//
// Empty / null hint maps to the dictionary's `auto` slot (i.e.
// `prompt.auto_id`). When the model exposes language detection (its
// hparams declare auto_id), this is the "let the model emit a
// <lang-XX> tag" path. Unknown hints return -1 (the caller should
// surface this as TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE — capability
// validation in transcribe.cpp will already have screened most
// invalid hints; this is the prompt-dictionary-vs-caps mismatch case).
int32_t resolve_prompt_id(const ParakeetHParams & hp,
                          const char *            language_hint)
{
    if (!hp.has_prompt) return -1;
    const bool empty_hint =
        (language_hint == nullptr) || (*language_hint == '\0');
    if (empty_hint) {
        // Auto-language slot. Models without an explicit auto entry
        // leave auto_id at -1; in that case the dictionary's first
        // entry is the conservative default.
        if (hp.prompt_auto_id >= 0) return hp.prompt_auto_id;
        if (!hp.prompt_dictionary_indices.empty()) {
            return hp.prompt_dictionary_indices.front();
        }
        return -1;
    }
    for (size_t i = 0; i < hp.prompt_dictionary_locales.size(); ++i) {
        if (hp.prompt_dictionary_locales[i] == language_hint) {
            return hp.prompt_dictionary_indices[i];
        }
    }
    return -1;
}

// Fill a [num_prompts, T_enc, 1, n_batch] float buffer with a one-hot
// vector at column `prompt_id` of each utterance, replicated across
// the T_enc axis. The host-side replication keeps the in-graph
// step a single concat + two matmuls, with no in-graph one_hot /
// broadcast machinery. `prompt_ids` carries one id per utterance.
//
// Returns false if any per-utterance prompt_id is out of range.
bool fill_prompt_one_hot(std::vector<float> &        out,
                         int                         num_prompts,
                         int                         T_enc,
                         int                         n_batch,
                         const std::vector<int32_t> & prompt_ids)
{
    const size_t total = static_cast<size_t>(num_prompts) *
                         static_cast<size_t>(T_enc) *
                         static_cast<size_t>(n_batch);
    out.assign(total, 0.0f);
    for (int b = 0; b < n_batch; ++b) {
        const int32_t pid =
            (b < static_cast<int>(prompt_ids.size()))
                ? prompt_ids[static_cast<size_t>(b)]
                : (prompt_ids.empty() ? -1 : prompt_ids.front());
        if (pid < 0 || pid >= num_prompts) return false;
        const size_t per_utt = static_cast<size_t>(num_prompts) *
                               static_cast<size_t>(T_enc);
        const size_t utt_off = static_cast<size_t>(b) * per_utt;
        for (int t = 0; t < T_enc; ++t) {
            const size_t row_off = utt_off +
                static_cast<size_t>(t) * static_cast<size_t>(num_prompts);
            out[row_off + static_cast<size_t>(pid)] = 1.0f;
        }
    }
    return true;
}

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

// Allocate the streaming encoder caches (cache_last_channel,
// cache_last_time). Called lazily on the first
// stream_begin for ChunkedLimited variants. Layout matches NeMo's
// get_initial_cache_state exactly: see ParakeetStreamingCaches in
// parakeet.h for shapes and the per-variant sizing rule.
//
// The cache tensors live in their own ggml_context with a dedicated
// backend buffer, allocated on the model's primary backend so reads
// and writes inside the encoder graph stay backend-local. The buffer
// is freed in ~ParakeetSession.
//
// Initialization is idempotent — calling on an already-initialized
// caches struct is a no-op. To zero contents on a fresh stream, call
// zero_streaming_caches separately.
transcribe_status init_streaming_caches(ParakeetSession * pc,
                                        ParakeetModel *   pm)
{
    if (pc->stream_caches.initialized) return TRANSCRIBE_OK;

    const auto & hp        = pm->hparams;
    const int    n_layer   = static_cast<int>(pm->weights.blocks.size());
    const int    d_model   = hp.enc_d_model;
    const int    T_cache   = hp.enc_att_context_left;
    const int    k_minus_1 = hp.enc_conv_context_left >= 0
                             ? hp.enc_conv_context_left
                             : (hp.enc_conv_kernel - 1);

    if (n_layer <= 0 || d_model <= 0 || T_cache <= 0 || k_minus_1 <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet stream init_caches: degenerate sizes "
                     "(n_layer=%d, d_model=%d, T_cache=%d, k-1=%d)",
                     n_layer, d_model, T_cache, k_minus_1);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // 2 tensors per layer (last_channel + last_time) plus headroom
    // for descriptor / view tensors created during graph build.
    const size_t ctx_size =
        static_cast<size_t>(2 * n_layer + 8) * ggml_tensor_overhead();
    ggml_init_params ip {};
    ip.mem_size   = ctx_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    pc->stream_caches.ctx = ggml_init(ip);
    if (pc->stream_caches.ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet stream init_caches: ggml_init failed");
        return TRANSCRIBE_ERR_OOM;
    }

    pc->stream_caches.last_channel.assign(n_layer, nullptr);
    pc->stream_caches.last_time.assign(n_layer, nullptr);
    for (int i = 0; i < n_layer; ++i) {
        ggml_tensor * lc = ggml_new_tensor_2d(pc->stream_caches.ctx,
                                              GGML_TYPE_F32,
                                              d_model, T_cache);
        ggml_tensor * lt = ggml_new_tensor_2d(pc->stream_caches.ctx,
                                              GGML_TYPE_F32,
                                              k_minus_1, d_model);
        if (lc == nullptr || lt == nullptr) {
            ggml_free(pc->stream_caches.ctx);
            pc->stream_caches.ctx = nullptr;
            return TRANSCRIBE_ERR_OOM;
        }
        char name[64];
        std::snprintf(name, sizeof(name), "stream.cache.last_channel.%d", i);
        ggml_set_name(lc, name);
        std::snprintf(name, sizeof(name), "stream.cache.last_time.%d", i);
        ggml_set_name(lt, name);
        pc->stream_caches.last_channel[i] = lc;
        pc->stream_caches.last_time[i]    = lt;
    }

    pc->stream_caches.buffer =
        ggml_backend_alloc_ctx_tensors(pc->stream_caches.ctx, pm->plan.primary);
    if (pc->stream_caches.buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet stream init_caches: backend buffer alloc failed");
        ggml_free(pc->stream_caches.ctx);
        pc->stream_caches.ctx = nullptr;
        return TRANSCRIBE_ERR_BACKEND;
    }

    pc->stream_caches.channel_len         = 0;
    pc->stream_caches.mel_frames_consumed = 0;

    pc->stream_caches.initialized = true;
    return TRANSCRIBE_OK;
}

// Zero the streaming caches and reset cursors. Called at the start of
// every stream_begin so each utterance starts from a clean slate
// (NeMo's get_initial_cache_state returns zeros every time).
//
// The backend buffer survives — only the contents are cleared.
void zero_streaming_caches(ParakeetSession * pc) {
    if (pc->stream_caches.buffer != nullptr) {
        ggml_backend_buffer_clear(pc->stream_caches.buffer, 0);
    }
    pc->stream_caches.channel_len         = 0;
    pc->stream_caches.mel_frames_consumed = 0;
    pc->stream_caches.pcm_start_sample    = 0;
}

// Reset the streaming decoder state (LSTM h/c, prev token, frame
// offset). Sized to the model's predictor on first call; resized
// in-place on subsequent calls if the predictor reshape ever happens
// (it won't, but the guard is cheap).
void reset_streaming_decoder_state(ParakeetSession * pc,
                                   const ParakeetModel * pm)
{
    const auto & pred = pm->host_decoder.predictor;
    const int    n_layers    = static_cast<int>(pred.lstm.size());
    const int    pred_hidden = pred.pred_hidden;

    pc->stream_dec_state.lstm_state.reset(n_layers, pred_hidden);
    pc->stream_dec_state.prev_token_id = -1;
    pc->stream_dec_state.frame_offset  = 0;
    pc->stream_dec_state.initialized   = true;
}

// Forward declarations for the Arch trait below.
extern transcribe_status load        (Loader &, const transcribe_model_load_params *,
                                      transcribe_model **);
extern transcribe_status init_context(transcribe_model *, const transcribe_session_params *,
                                      transcribe_session **);
extern transcribe_status run         (transcribe_session *, const float *, int,
                                      const transcribe_run_params *);

transcribe_status load(
    Loader &                          loader,
    const transcribe_model_load_params *   params,
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
    apply_family_invariants(*m);
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

    // Derive supports_streaming from hparams.
    //
    //   ChunkedLimited + (L, R) >= 0   — cache-aware streaming
    //     (nemotron-speech-streaming-en-0.6b). The cache_last_channel /
    //     cache_last_time path is engaged at stream_begin.
    //   ChunkedLimitedWithRc + non-empty (L, C, R) menu — buffered
    //     streaming (parakeet-unified-en-0.6b). The buffered driver
    //     re-runs the offline encoder per chunk with a 3-tuple chunk
    //     mask; no per-layer cache.
    //
    // Offline parakeet variants (Regular full attention or no chunking)
    // stay non-streaming regardless of any KV claim.
    const bool cache_aware_streaming =
        (m->hparams.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimited) &&
        m->hparams.enc_att_context_left  >= 0 &&
        m->hparams.enc_att_context_right >= 0;
    const bool buffered_streaming =
        (m->hparams.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimitedWithRc) &&
        !m->hparams.enc_att_chunk_left_choices.empty() &&
        !m->hparams.enc_att_chunk_chunk_choices.empty() &&
        !m->hparams.enc_att_chunk_right_choices.empty();
    // supports_streaming is the generic gate; the configurable
    // streaming geometry (the (left, chunk, right) menu for buffered,
    // the att_context_right menu for cache-aware) is reached via the
    // parakeet stream extensions, not advertised as flat caps fields.
    if (buffered_streaming || cache_aware_streaming) {
        m->caps.supports_streaming = true;
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
        // NeMo's FilterbankFeatures.stft uses pad_mode="constant" for
        // the STFT center-pad (see features.py line 385). The cpp
        // MelConfig default is "reflect" — matching cpp's behavior to
        // NeMo here. The two differ in the first/last few STFT frames:
        // reflect pads with mirrored audio, constant pads with zeros.
        // Offline this is one boundary per utterance and the residual
        // washes it out; streaming sees boundary effects every chunk
        // and the per-feature normalize amplifies the divergence into
        // the encoder.
        cfg.pad_mode     = "constant";
        m->mel.emplace(cfg);
    }

    // Stage 2: reopen the file with no_alloc=true + session=&ctx_meta.
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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet: ggml_backend_alloc_ctx_tensors failed");
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
    const transcribe_session_params * params,
    transcribe_session **             out_ctx)
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

    auto pc = std::make_unique<ParakeetSession>();
    pc->model     = model;
    pc->n_threads = params->n_threads;
    pc->kv_type   = params->kv_type;

    *out_ctx = pc.release();
    return TRANSCRIBE_OK;
}

// Internal inference helper. Assumes pc/pm are non-null and the
// scheduler plan is populated; callers (run() and stream_finalize)
// validate that up front. n_samples must be > 0; pcm must be non-null.
//
// Does NOT call pc->clear_result(); callers handle result-snapshot
// management. (The public run() entry clears once before reaching here;
// stream_finalize relies on the dispatcher's clear at stream_begin time
// and does not re-clear, so its accumulated stream cursors survive.)
//
// Phase 4a — run/stream parity:
//
//   For cache-aware-streaming Parakeet variants (today: nemotron-
//   speech-streaming-en-0.6b), transcribe_run and the streaming hooks
//   are siblings of this helper; neither owns the inference. That
//   gives the streaming-of-whole path guaranteed-by-construction
//   parity with one-shot run on the same audio. When real per-chunk
//   encoder feeding lands (Phase 4b: cache_last_channel /
//   cache_last_time / RNNT predictor carry), the streaming hooks
//   will diverge from this helper and an empirical stream-vs-batch
//   parity test becomes meaningful.
// Host-side TDT/RNNT/CTC decode + public result-hierarchy build for a
// single utterance's encoder output. Shared by the single-shot path
// True for a multilingual language-tag SentencePiece piece of the form
// "<ll-RR>" (lowercase 2-3 letter language, '-', 2-4 letter region), e.g.
// "<en-US>", "<zh-CN>", "<nb-NO>". The nemotron-3.5 multilingual vocab emits
// one of these per segment to mark the (detected or requested) language. We
// drop them from the public token/word/segment/text result by default so the
// transcript stays clean and word/timestamp aggregation isn't polluted by a
// zero-width tag "word". Requiring the '-REGION' part avoids matching control
// pieces like "<unk>". No-op for the English parakeets (no such pieces in
// their 1024-token vocab). Preserved when keep_special_tags is set
// (CLI --raw-tokens).
static bool is_lang_tag_piece(const std::string & p) {
    const size_t n = p.size();
    if (n < 7 || p.front() != '<' || p.back() != '>') return false; // min "<aa-AA>"
    size_t i = 1;
    const size_t end = n - 1;
    const size_t lang0 = i;
    while (i < end && p[i] >= 'a' && p[i] <= 'z') ++i;
    const size_t lang_len = i - lang0;
    if (lang_len < 2 || lang_len > 3) return false;
    if (i >= end || p[i] != '-') return false;
    ++i;
    const size_t reg0 = i;
    while (i < end && std::isalpha(static_cast<unsigned char>(p[i]))) ++i;
    const size_t reg_len = i - reg0;
    if (reg_len < 2 || reg_len > 4) return false;
    return i == end; // interior consumed exactly up to '>'
}

// (run_one_shot_inner) and the batched path (run_batch_inner). `enc` is
// the row-major [T_enc, d_enc] encoder activation for ONE utterance;
// utt_index >= 0 tags the optional tensor dump per utterance, -1 for the
// single-shot name. Writes the session scratch result slot (tokens /
// words / segments / full_text / result_kind / has_result).
static transcribe_status decode_and_populate(
    ParakeetSession *             pc,
    ParakeetModel *               pm,
    const transcribe_run_params * params,
    const float *                 enc,
    int                           T_enc,
    int                           d_enc,
    int                           utt_index,
    const char *                  enc_dump_name_override = nullptr)
{
    // Default dump name is "dec.enc_out"; the prompt-conditioned path
    // overrides to "dec.enc_out_prompted" so the comparator sees the
    // post-prompt tensor under its expected filename.
    std::string dump_name = (enc_dump_name_override != nullptr &&
                             *enc_dump_name_override != '\0')
        ? std::string(enc_dump_name_override)
        : std::string("dec.enc_out");
    if (utt_index >= 0) {
        dump_name += ".b" + std::to_string(utt_index);
    }
    // Optional dump of the encoder output as the decoder sees it,
    // so the bring-up loop can verify the readback is faithful
    // before chasing decoder bugs.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, d_enc };
        const char * stage = (enc_dump_name_override != nullptr &&
                              *enc_dump_name_override != '\0')
            ? "decoder.enc_out_prompted"
            : "decoder.enc_out";
        transcribe::debug::dump_host_f32(
            dump_name.c_str(), enc,
            static_cast<long long>(T_enc) * static_cast<long long>(d_enc),
            shape, 2, stage);
    }

    pc->raw_tokens.clear();
    const int64_t t_dec_start = ggml_time_us();
    {
        transcribe_status st = TRANSCRIBE_OK;
        switch (pm->host_decoder.head_kind) {
            case HostHeadKind::TDT:
                st = decode_tdt_greedy(pm->host_decoder, enc,
                                       T_enc, d_enc, pc->n_threads, pc->raw_tokens);
                break;
            case HostHeadKind::RNNT:
                st = decode_rnnt_greedy(pm->host_decoder, enc,
                                        T_enc, d_enc, pc->n_threads, pc->raw_tokens);
                break;
            case HostHeadKind::CTC:
                st = decode_ctc_greedy(pm->host_decoder, enc,
                                       T_enc, d_enc, pc->n_threads, pc->raw_tokens);
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
    // Strip multilingual <ll-RR> language tags from the public result by
    // default (clean transcript + uncorrupted word/timestamp aggregation).
    // Gated on the standard keep_special_tags run param (CLI --raw-tokens) —
    // the same switch canary/sensevoice use for their <|...|> tags.
    //
    // Primary detection is Tokenizer::is_control: convert-parakeet.py marks the
    // tag pieces CONTROL in the GGUF token_type (read from the vocab shape, not
    // a hard-coded list). The is_lang_tag_piece() pattern is a transitional
    // fallback for GGUFs produced before that converter change; it can be
    // dropped once all shipped artifacts carry the metadata.
    const bool strip_tags = (params == nullptr) ? true : !params->keep_special_tags;
    for (const TdtToken & rt : pc->raw_tokens) {
        if (strip_tags &&
            (tok.is_control(rt.id) || is_lang_tag_piece(tok.token(rt.id)))) {
            continue;
        }
        transcribe_session::TokenEntry te;
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
        transcribe_session::SegmentEntry seg;
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
        transcribe_session::WordEntry  cur_word;
        bool                        cur_word_open = false;

        auto open_new_word = [&](int token_index, const transcribe_session::TokenEntry & tk) {
            if (cur_word_open) {
                cur_word.t1_ms   = pc->tokens[static_cast<size_t>(token_index - 1)].t1_ms;
                cur_word.n_tokens = token_index - cur_word.first_token;
                pc->words.push_back(std::move(cur_word));
                cur_word = transcribe_session::WordEntry{};
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
        // Normalize whitespace: removing a <ll-RR> language tag from the id
        // sequence leaves its neighbours' spaces adjacent (double space), and
        // a stripped trailing tag leaves a trailing space. Collapse runs of
        // spaces to one and trim both ends — an ASR transcript never carries
        // meaningful double/edge whitespace. No-op when nothing was stripped.
        {
            std::string norm;
            norm.reserve(full.size());
            bool prev_space = false;
            for (char ch : full) {
                const bool is_space = (ch == ' ');
                if (is_space && prev_space) continue;
                norm.push_back(ch);
                prev_space = is_space;
            }
            while (!norm.empty() && norm.front() == ' ') norm.erase(norm.begin());
            while (!norm.empty() && norm.back()  == ' ') norm.pop_back();
            full.swap(norm);
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

static transcribe_status run_one_shot_inner(
    ParakeetSession *         pc,
    ParakeetModel *           pm,
    const float *             pcm,
    int                       n_samples,
    const transcribe_run_params * params)
{
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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet run: model has no MelFrontend (load skipped?)");
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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet run: MelFrontend::compute failed (%s)",
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet run: ggml_init for compute_ctx failed");
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
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet run: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, eb.graph)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet run: ggml_backend_sched_alloc_graph failed");
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

    // Prompt one-hot input (multilingual variants only). The graph
    // builder allocates `prompt_one_hot_in` of shape
    // [num_prompts, T_enc, 1, 1] when hp.has_prompt is true; the
    // host fills it with the resolved language's one-hot replicated
    // across T_enc frames. See resolve_prompt_id for the language
    // string → index lookup.
    if (eb.prompt_one_hot_in != nullptr) {
        const int P     = static_cast<int>(eb.prompt_one_hot_in->ne[0]);
        const int T_oh  = static_cast<int>(eb.prompt_one_hot_in->ne[1]);
        const char * lang_hint =
            (params != nullptr) ? params->language : nullptr;
        const int32_t pid = resolve_prompt_id(pm->hparams, lang_hint);
        if (pid < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet run: language %s%s%s not in prompt "
                         "dictionary",
                         lang_hint ? "\"" : "",
                         lang_hint ? lang_hint : "<null>",
                         lang_hint ? "\"" : "");
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
        std::vector<float> one_hot_buf;
        if (!fill_prompt_one_hot(one_hot_buf, P, T_oh, /*n_batch=*/1,
                                 {pid}))
        {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet run: prompt_id %d out of range "
                         "[0, %d)", pid, P);
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(eb.prompt_one_hot_in, one_hot_buf.data(),
                                0, one_hot_buf.size() * sizeof(float));
    }

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
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet run: ggml_backend_sched_graph_compute failed (%d)",
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
    try_dump("enc.prompted",         eb.dumps.prompted_out,     "encoder.prompted");

    // Stash the encoder output for the accuracy test (it reaches in
    // via the ParakeetSession view) and as a borrowed reference for
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
    //
    // The result snapshot has already been cleared by the caller
    // (public run() at its entry, or the streaming dispatcher at
    // stream_begin). Re-clearing here would also reset the audio
    // cursors stream_finalize is about to commit, so it stays out.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_enc <= 0 || T_enc <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet run: encoder output has degenerate shape "
                     "[%d, %d]", d_enc, T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }
    pc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, pc->enc_host.data(), 0,
                            pc->enc_host.size() * sizeof(float));

    // For prompt-conditioned models the pre-prompt encoder output is
    // a distinct buffer from `eb.out` (which is the post-prompt
    // tensor the decoder consumes). The reference dumper labels the
    // pre-prompt tensor "dec.enc_out" and the post-prompt tensor
    // "dec.enc_out_prompted"; read both back so the validate.py
    // compare names match. decode_and_populate handles the prompted
    // dump via the override below.
    if (pm->hparams.has_prompt && eb.dumps.final_out != nullptr &&
        transcribe::debug::enabled())
    {
        std::vector<float> unprompted(pc->enc_host.size());
        ggml_backend_tensor_get(eb.dumps.final_out, unprompted.data(), 0,
                                unprompted.size() * sizeof(float));
        const long long shape[2] = { T_enc, d_enc };
        transcribe::debug::dump_host_f32(
            "dec.enc_out", unprompted.data(),
            static_cast<long long>(T_enc) * static_cast<long long>(d_enc),
            shape, 2, "decoder.enc_out");
    }

    const char * enc_dump_name =
        pm->hparams.has_prompt ? "dec.enc_out_prompted" : nullptr;
    return decode_and_populate(pc, pm, params, pc->enc_host.data(),
                               T_enc, d_enc, /*utt_index=*/-1,
                               enc_dump_name);
}

// One-shot entry point. Validates session/pm, clears the previous result
// snapshot, then forwards to the shared inference helper. The
// streaming hooks reuse the same helper so a finalize on the
// accumulated buffer produces identical results to a direct run() of
// the same audio.
transcribe_status run(
    transcribe_session *      session,
    const float *             pcm,
    int                       n_samples,
    const transcribe_run_params * params)
{
    // The dispatcher (transcribe.cpp) has already enum-range
    // validated params->task / params->timestamps, rejected
    // TRANSLATE against this family's supports_translate=false, and
    // rejected any timestamp request finer than our advertised
    // max_timestamp_kind=TOKEN. What arrives here is guaranteed
    // sane — we only need to resolve AUTO and downcast finer-grained
    // output to the requested ceiling when the result is built.
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * pc = static_cast<ParakeetSession *>(session);
    auto * pm = static_cast<ParakeetModel *>(pc->model);
    if (pm == nullptr || pm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    pc->clear_result();
    return run_one_shot_inner(pc, pm, pcm, n_samples, params);
}

// ---------------------------------------------------------------------------
// Offline batch (transcribe_run_batch)
// ---------------------------------------------------------------------------
//
// Batches B same-length utterances through ONE encoder graph (n_batch == B,
// the batch riding the activation's ne[2] axis — see the conformer helpers)
// in a single device dispatch, then host-decodes each utterance's encoder
// slice. The mel front-end, decoder, and result-build are identical to the
// single-shot path; only the encoder is fused. Utterances of differing
// length fall back to the per-utterance path (variable-length batching pads
// + masks the overhang — a separate change). Either way every utterance's
// result is captured into session->batch_results in order.

// Build + compute the batched encoder for `n` utterances, then decode each
// into session->batch_results. Every entry of `mels` is the raw mel-major
// [n_mels, nf[b]] buffer; this packs them into a [T_max, n_mels, 1, n] graph
// input, zero-padding each along time to the batch's T_max. When utterances
// differ in length it also builds + fills the variable-length masks (attn
// key-padding + conv valid-frame) so each utterance's padded tail cannot
// corrupt its real frames, and decodes each at its own T_enc. Same-length
// batches (every nf == T_max) skip the masks and are bit-identical to
// single-shot.
static int pre_encode_t_out(int in) {
    // One stride-2, kernel-3, pad-1 conv: floor((in + 2 - 3)/2) + 1.
    return (in - 1) / 2 + 1;
}

static transcribe_status run_batch_encode(
    ParakeetSession *                       pc,
    ParakeetModel *                         pm,
    const std::vector<std::vector<float>> & mels,
    const std::vector<int> &                nf,
    int                                     n_mels,
    int                                     T_max,
    int64_t                                 total_mel_us,
    const transcribe_run_params *           params)
{
    const int n = static_cast<int>(mels.size());
    bool var_len = false;
    for (int b = 0; b < n; ++b) {
        if (nf[b] != T_max) { var_len = true; break; }
    }

    // Pack mels into [T_max, n_mels, 1, n], zero-padding each along time
    // (channel-major source [n_mels, nf[b]] -> per-utterance [T_max, n_mels]).
    transcribe::pack_pad_channel_major(pc->mel_buf, mels, nf, n_mels, T_max);

    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    pc->encoder_out = nullptr;
    {
        ggml_init_params init_params {};
        init_params.mem_size   = 4 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        pc->compute_ctx = ggml_init(init_params);
        if (pc->compute_ctx == nullptr) return TRANSCRIBE_ERR_GGUF;
    }

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    EncoderBuild eb = build_encoder_graph(
        pc->compute_ctx, pm->weights, pm->hparams, T_max,
        resolved_kv, pm->backend.c_str(), /*buf_mask=*/nullptr,
        /*n_batch=*/n, /*batch_var_len=*/var_len);
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(
            pm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(pm->plan.scheduler_list.size()),
            /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, eb.graph)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(eb.mel_in, pc->mel_buf.data(),
                            0, pc->mel_buf.size() * sizeof(float));

    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_enc <= 0 || T_enc <= 0) return TRANSCRIBE_ERR_GGUF;

    // Per-utterance valid encoder-frame count (the same subsample the conv
    // stack applies: 3 stride-2 convs on the time axis).
    std::vector<int> real_tenc(static_cast<size_t>(n), T_enc);
    if (var_len) {
        for (int b = 0; b < n; ++b) {
            int t = nf[b];
            t = pre_encode_t_out(t);
            t = pre_encode_t_out(t);
            t = pre_encode_t_out(t);
            real_tenc[b] = std::min(t, T_enc);
        }
        // Attention key-padding mask [T_enc, 1, 1, n] (0 real / -INF padded)
        // and conv valid-frame mask [T_enc, 1, n, 1] (1 real / 0 padded).
        transcribe::fill_keypad_mask(eb.attn_pad_mask_in, real_tenc, T_enc, n);
        transcribe::fill_valid_frame_mask(eb.conv_pad_mask_in, real_tenc, T_enc, n);

        // Pre-encode valid-frame masks (masked subsampling). One per ReLU
        // stage; the valid time length at stage k is the per-utterance mel
        // length downsampled k times by the (in-1)/2+1 conv formula. Each
        // mask is ne=[1, H_stage, 1, n] -> host index b*H_stage + h.
        auto fill_pe_mask = [&](ggml_tensor * mask, int n_down) {
            if (mask == nullptr) return;
            const int H = static_cast<int>(mask->ne[1]);
            std::vector<float> mb(static_cast<size_t>(H) * n, 0.0f);
            for (int b = 0; b < n; ++b) {
                int v = nf[b];
                for (int d = 0; d < n_down; ++d) v = pre_encode_t_out(v);
                if (v > H) v = H;
                for (int h = 0; h < v; ++h) {
                    mb[static_cast<size_t>(b) * H + h] = 1.0f;
                }
            }
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        };
        fill_pe_mask(eb.pre_encode_mask_s1_in, 1);  // after relu0
        fill_pe_mask(eb.pre_encode_mask_s2_in, 2);  // after relu3
        fill_pe_mask(eb.pre_encode_mask_s3_in, 3);  // after relu6
    }

    // Positional embedding (batch-independent; depends only on T_enc).
    if (eb.pos_emb_in != nullptr) {
        const int d_model = pm->hparams.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
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
    }

    // Prompt one-hot upload (multilingual variants). Resolves params->language
    // ONCE for the whole batch (the public ABI doesn't expose a per-utterance
    // language array yet — every utterance in a batched call shares the same
    // language hint) and replicates the one-hot across all (T_enc, B) slots.
    if (eb.prompt_one_hot_in != nullptr) {
        const int P    = static_cast<int>(eb.prompt_one_hot_in->ne[0]);
        const int T_oh = static_cast<int>(eb.prompt_one_hot_in->ne[1]);
        const char * lang_hint =
            (params != nullptr) ? params->language : nullptr;
        const int32_t pid = resolve_prompt_id(pm->hparams, lang_hint);
        if (pid < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet run_batch: language %s%s%s not in prompt "
                         "dictionary",
                         lang_hint ? "\"" : "",
                         lang_hint ? lang_hint : "<null>",
                         lang_hint ? "\"" : "");
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
        std::vector<int32_t> pids(static_cast<size_t>(n), pid);
        std::vector<float> one_hot_buf;
        if (!fill_prompt_one_hot(one_hot_buf, P, T_oh, /*n_batch=*/n, pids)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet run_batch: prompt_id %d out of range "
                         "[0, %d)", pid, P);
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(eb.prompt_one_hot_in, one_hot_buf.data(),
                                0, one_hot_buf.size() * sizeof(float));
    }

    // ChunkedLimited attention mask (cache-aware variants). Allocated by
    // build_encoder_graph when the model declares chunked attention; the
    // single-shot path fills it and the batched path must too (otherwise the
    // input is uninitialized and corrupts every attention score). The pattern
    // depends only on T_enc, which the whole batch shares, so one mask
    // broadcasts across the batch. Mirrors run_one_shot_inner.
    if (eb.chunked_mask_in != nullptr) {
        const int Tk          = static_cast<int>(eb.chunked_mask_in->ne[0]);
        const int chunk_size  = pm->hparams.enc_att_context_right + 1;
        const int left_chunks = (chunk_size > 0)
            ? (pm->hparams.enc_att_context_left / chunk_size) : 0;
        std::vector<float> mask_buf(static_cast<size_t>(Tk) * Tk);
        for (int q = 0; q < Tk; ++q) {
            const int q_chunk = (chunk_size > 0) ? (q / chunk_size) : 0;
            const int k_min_chunk =
                (q_chunk - left_chunks > 0) ? (q_chunk - left_chunks) : 0;
            const int k_min = k_min_chunk * chunk_size;
            const int k_max = (q_chunk + 1) * chunk_size;
            float * row = mask_buf.data() + static_cast<size_t>(q) * Tk;
            for (int k = 0; k < Tk; ++k) {
                row[k] = (k >= k_min && k < k_max)
                    ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
        ggml_backend_tensor_set(eb.chunked_mask_in, mask_buf.data(),
                                0, mask_buf.size() * sizeof(float));
    }

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
            if (fn != nullptr) fn(be, n_threads);
        }
    }

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(pc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet run_batch: graph_compute failed (%d)",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    pc->t_encode_us = ggml_time_us() - t_enc_start;

    // Bisect dump (debug only): the batched encoder intermediates as full
    // [d_model, T_max, n] tensors, so a batched-vs-single divergence can be
    // located per stage. Gated on TRANSCRIBE_DUMP_ALL_BLOCKS.
    if (transcribe::debug::enabled() &&
        std::getenv("TRANSCRIBE_DUMP_ALL_BLOCKS") != nullptr)
    {
        if (eb.dumps.pre_encode_out != nullptr) {
            transcribe::debug::dump_tensor("enc.pre_encode.out",
                                           eb.dumps.pre_encode_out,
                                           "encoder.pre_encode");
        }
        for (size_t i = 0; i < eb.dumps.all_block_outs.size(); ++i) {
            ggml_tensor * t = eb.dumps.all_block_outs[i];
            if (t == nullptr) continue;
            char name[64];
            std::snprintf(name, sizeof(name), "enc.block.%zu.out", i);
            transcribe::debug::dump_tensor(name, t, "encoder.block.bisect");
        }
    }

    const size_t utt_elems = static_cast<size_t>(d_enc) * static_cast<size_t>(T_enc);
    pc->enc_host.resize(utt_elems * static_cast<size_t>(n));
    ggml_backend_tensor_get(eb.out, pc->enc_host.data(), 0,
                            pc->enc_host.size() * sizeof(float));

    // Prompt-conditioned variants: eb.out is the POST-prompt tensor (what
    // the decoder consumes); the single-shot path dumps the PRE-prompt
    // tensor as "dec.enc_out" and the POST-prompt as "dec.enc_out_prompted"
    // so the comparator sees both under their reference names. Mirror that
    // in the batched path with per-utterance .b{i} suffixes.
    const bool has_prompt = pm->hparams.has_prompt &&
                            eb.dumps.final_out != nullptr;
    std::vector<float> unprompted_host;
    if (has_prompt && transcribe::debug::enabled()) {
        unprompted_host.resize(pc->enc_host.size());
        ggml_backend_tensor_get(eb.dumps.final_out, unprompted_host.data(), 0,
                                unprompted_host.size() * sizeof(float));
        for (int b = 0; b < n; ++b) {
            const long long shape[2] = { real_tenc[b], d_enc };
            char namebuf[64];
            std::snprintf(namebuf, sizeof(namebuf), "dec.enc_out.b%d", b);
            transcribe::debug::dump_host_f32(
                namebuf,
                unprompted_host.data() + static_cast<size_t>(b) * utt_elems,
                static_cast<long long>(real_tenc[b]) *
                    static_cast<long long>(d_enc),
                shape, 2, "decoder.enc_out");
        }
    }
    const char * enc_dump_name =
        has_prompt ? "dec.enc_out_prompted" : nullptr;

    // Host-slice the shared encoder output and decode each utterance. The
    // encoder is ONE shared dispatch, so decode_batch_slices amortizes its
    // total compute + the total mel cost across the batch (the per-utt sum then
    // equals the real batch time); decode is genuinely per-utterance.
    return transcribe::decode_batch_slices(
        pc, n, pc->enc_host.data(), utt_elems, pc->t_encode_us, total_mel_us,
        [&](int b, const float * enc_b) {
            return decode_and_populate(pc, pm, params, enc_b, real_tenc[b],
                                       d_enc, /*utt_index=*/b,
                                       enc_dump_name);
        });
}

transcribe_status run_batch(
    transcribe_session *          session,
    const float * const *         pcm,
    const int *                   n_samples,
    int                           n,
    const transcribe_run_params * params)
{
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * pc = static_cast<ParakeetSession *>(session);
    auto * pm = static_cast<ParakeetModel *>(pc->model);
    if (pm == nullptr || pm->plan.scheduler_list.empty() || !pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe::debug::init();

    // Compute each utterance's mel. A malformed utterance (null pcm /
    // non-positive samples / mel failure) means we cannot pack a clean batch
    // tensor, so we fall back to the per-utterance path for the whole call
    // (rare; keeps the batch tensor rectangular and the masks well-defined).
    // The mel front-end is pure host code with no cross-utterance state, so
    // the B extractions run in parallel across CPU workers (see
    // transcribe::parallel_for_all) — frequently the dominant wall cost once
    // the encoder is on a fast accelerator. n_mels is constant across
    // utterances; collect it per-index to avoid a shared write.
    std::vector<std::vector<float>> mels(static_cast<size_t>(n));
    std::vector<int>                nf(static_cast<size_t>(n), 0);
    std::vector<int>                n_mels_per(static_cast<size_t>(n), 0);
    const int64_t t_mel_start = ggml_time_us();
    const bool all_ok = transcribe::parallel_for_all(
        n, pc->n_threads, [&](int i) -> bool {
            if (pcm[i] == nullptr || n_samples[i] <= 0) return false;
            int this_mels = 0, this_frames = 0;
            const transcribe_status st = pm->mel->compute(
                pcm[i], static_cast<size_t>(n_samples[i]),
                mels[i], this_mels, this_frames);
            if (st != TRANSCRIBE_OK || this_frames <= 0) return false;
            nf[i]         = this_frames;
            n_mels_per[i] = this_mels;
            return true;
        });
    const int64_t total_mel_us = ggml_time_us() - t_mel_start;

    if (all_ok) {
        int T_max = 0, n_mels = 0;
        for (int i = 0; i < n; ++i) {
            T_max  = std::max(T_max, nf[i]);
            n_mels = std::max(n_mels, n_mels_per[i]);
        }
        return run_batch_encode(pc, pm, mels, nf, n_mels, T_max,
                                total_mel_us, params);
    }

    // Per-utterance fallback (also the malformed-input path).
    for (int i = 0; i < n; ++i) {
        if (pc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
        if (pcm[i] == nullptr || n_samples[i] <= 0) {
            transcribe_session::ResultSet rs;
            rs.status = TRANSCRIBE_ERR_INVALID_ARG;
            pc->batch_results.push_back(std::move(rs));
            continue;
        }
        pc->clear_result();
        const transcribe_status st =
            run_one_shot_inner(pc, pm, pcm[i], n_samples[i], params);
        pc->batch_results.push_back(pc->capture_result(st));
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Streaming hooks (Phase 4a: stream-of-whole)
// ---------------------------------------------------------------------------
//
// Phase 4a buffers the entire stream in stream_pcm_buffer and runs
// the existing one-shot inference path at finalize. This makes the
// streaming API observable end-to-end and trivially parity-equivalent
// to transcribe_run on the same audio. Real incremental encoder /
// RNNT-predictor streaming via NeMo's cache_last_channel /
// cache_last_time tensors is Phase 4b.
//
// Only cache-aware streaming variants (ChunkedLimited attention) take
// this path; offline parakeet variants never reach the hook because
// load() leaves caps.supports_streaming = false for them and the
// streaming dispatcher rejects begin with NOT_IMPLEMENTED. The defensive
// gate in stream_begin below is defense in depth.
//
// The dispatcher handles the IDLE/ACTIVE/FINISHED/FAILED lifecycle
// and the result-snapshot counters (revision, committed counts, audio
// cursors). Family hooks own the per-utterance audio scratch.

constexpr int64_t k_sample_rate_hz = 16000;

static int64_t samples_to_us(int64_t n_samples) {
    return (n_samples * 1000000) / k_sample_rate_hz;
}

static int64_t us_to_ms(int64_t us) {
    return us / 1000;
}

// ---------------------------------------------------------------------------
// M2 streaming-encoder helpers
// ---------------------------------------------------------------------------

// Per-chunk shape constants for nemotron-speech-streaming-en-0.6b
// (att_context_size=[70,13], subsampling_factor=8). These match NeMo's
// Streaming chunk geometry now lives on ParakeetStreamingCaches and is
// resolved per stream from ParakeetHParams + the caller-selected
// att_context_right. The values used here in stream_feed /
// emit_streaming_chunk are pc->stream_caches.{mel_chunk_total,
// mel_new_per_chunk, drop_extra_pre_encoded}; pm->hparams.
// enc_stream_pre_encode_cache_size is used directly for the
// history-frame count (= mel_chunk_total - mel_new_per_chunk).
//
// Simplification vs NeMo (M2): we always use the "subsequent" mode
// (with history-frame mel prepend) for every chunk including the
// first; the first chunk's history is zero-initialized at
// stream_begin. The downside is a few frames of output are wasted at
// the start; the upside is uniform driver logic. Phase B will fix
// the first-chunk semantics to match NeMo's chunk_size[0] /
// pre_encode_cache_size[0] = 0.

// Fill the sinusoidal pos_emb buffer for a streaming chunk. Layout is
// identical to the offline ChunkedLimited path (RelPositionalEncoding):
// pos_len = 2*T_virtual - 1; positions[i] = (T_virtual - 1) - i; per-row
// sinusoid uses div_term[k] = exp(2k * -ln(10000) / d_model).
// `pos_emb_in` shape: ne=[d_model, pos_len, 1, 1] f32 (the encoder
// graph builder allocated it). Side-effect: pc->pos_buf and
// pc->pos_div_term are resized in place.
void fill_streaming_pos_emb(ParakeetSession * pc,
                            ggml_tensor *     pos_emb_in,
                            int               d_model)
{
    const int pos_len = static_cast<int>(pos_emb_in->ne[1]);
    const int zero_index = (pos_len - 1) / 2;

    pc->pos_buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);
    pc->pos_div_term.resize(static_cast<size_t>(d_model / 2));
    const float ln_10000 = std::log(10000.0f);
    for (int k = 0; k < d_model / 2; ++k) {
        pc->pos_div_term[static_cast<size_t>(k)] =
            std::exp(static_cast<float>(2 * k) *
                     (-ln_10000 / static_cast<float>(d_model)));
    }
    for (int i = 0; i < pos_len; ++i) {
        const float pos = static_cast<float>(zero_index - i);
        float * row = pc->pos_buf.data() +
            static_cast<size_t>(i) * d_model;
        for (int k = 0; k < d_model / 2; ++k) {
            const float div = pc->pos_div_term[static_cast<size_t>(k)];
            row[2 * k]     = std::sin(pos * div);
            row[2 * k + 1] = std::cos(pos * div);
        }
    }
    ggml_backend_tensor_set(pos_emb_in, pc->pos_buf.data(),
                            0, pc->pos_buf.size() * sizeof(float));
}

// Fill the streaming attention mask. Square shape [T_virtual,T_virtual]
// in ggml ne (broadcasts across n_heads inside rel_pos_mhsa). Rules:
//   - Within the chunked-limited band (k in [q-att_left..q+att_right] in
//     chunk-aligned terms): 0
//   - Outside the band: -INF
//   - Additionally for any query: cache-prefix keys whose slot has not
//     been written yet (k < T_cache - channel_len): -INF (NeMo's
//     cache_last_channel_len-driven masking).
void fill_streaming_chunked_mask(ggml_tensor * mask_in,
                                 int           T_virtual,
                                 int           T_cache,
                                 int           channel_len,
                                 int           att_context_left,
                                 int           att_context_right)
{
    const int chunk_size  = att_context_right + 1;
    const int left_chunks = (chunk_size > 0)
        ? (att_context_left / chunk_size)
        : 0;
    const int invalid_cache_threshold = T_cache - channel_len;

    std::vector<float> mask_buf(
        static_cast<size_t>(T_virtual) * static_cast<size_t>(T_virtual));
    for (int q = 0; q < T_virtual; ++q) {
        const int q_chunk     = (chunk_size > 0) ? (q / chunk_size) : 0;
        const int k_min_chunk = (q_chunk - left_chunks > 0)
            ? (q_chunk - left_chunks) : 0;
        const int k_min = k_min_chunk * chunk_size;
        const int k_max = (q_chunk + 1) * chunk_size; // exclusive
        float * row = mask_buf.data() +
            static_cast<size_t>(q) * T_virtual;
        for (int k = 0; k < T_virtual; ++k) {
            const bool in_band      = (k >= k_min && k < k_max);
            const bool cache_unfilled = (k < invalid_cache_threshold);
            row[k] = (in_band && !cache_unfilled)
                ? 0.0f
                : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_backend_tensor_set(mask_in, mask_buf.data(),
                            0, mask_buf.size() * sizeof(float));
}

} // namespace

// Already inside namespace transcribe::parakeet { ... }. The function is
// declared in parakeet.h so external callers (unit tests, the buffered
// streaming driver) can link against it.
void compute_chunked_limited_with_rc_mask(
    float * out_buf,
    int     T,
    int     left_context_frames,
    int     chunk_size_frames,
    int     right_context_frames,
    int     pad_length)
{
    assert(out_buf != nullptr);
    assert(T >= 1);
    assert(chunk_size_frames >= 1);
    assert(left_context_frames >= 0);
    assert(right_context_frames >= 0);
    assert(pad_length >= 0);

    const int L = left_context_frames;
    const int C = chunk_size_frames;
    const int R = right_context_frames;
    // Clamp pad_length to [0, T]. pad_length >= T means "no pad mask" —
    // every frame is valid.
    const int P = pad_length > T ? T : pad_length;

    for (int q = 0; q < T; ++q) {
        const int c_q                    = q / C;
        const int window_start_unclamped = c_q * C - L;
        const int window_end_unclamped   = c_q * C + C - 1 + R;
        const int window_start = window_start_unclamped > 0 ? window_start_unclamped : 0;
        const int window_end   = window_end_unclamped < (T - 1) ? window_end_unclamped : (T - 1);
        const bool q_padded    = q >= P;

        float * row = out_buf + static_cast<size_t>(q) * static_cast<size_t>(T);
        for (int k = 0; k < T; ++k) {
            const bool k_padded = k >= P;
            const bool allowed  = (k >= window_start && k <= window_end)
                                  && !q_padded && !k_padded;
            row[k] = allowed
                ? 0.0f
                : -std::numeric_limits<float>::infinity();
        }
    }
}

namespace {

// Build, run, and post-process a single streaming encoder chunk.
//
// Inputs:
//   pc, pm                      context + model
//   mel_chunk_data              row-major [n_mels, n_mel_chunk_frames]
//                               f32 (matches ggml ne=[n_mel_chunk_frames,
//                               n_mels, 1, 1] for upload).
//   n_mel_chunk_frames          mel frames in this chunk (e.g. 121 for
//                               nemotron subsequent chunks = 9 cache +
//                               112 new; 105 for first chunk = no
//                               cache prepend).
//   drop_extra_pre_encoded      0 for first chunk, 2 for subsequent on
//                               nemotron-streaming.
//   mel_frames_advance          how many mel frames this chunk consumes
//                               from the input buffer (chunk_size_first
//                               or chunk_size_subsequent). NOT the same
//                               as n_mel_chunk_frames when a pre-encode
//                               cache prepend is in play.
//
// On success, appends new TdtTokens to pc->raw_tokens and advances
// the persistent cache + decoder state.
transcribe_status emit_streaming_chunk(
    ParakeetSession * pc,
    ParakeetModel *   pm,
    const float *     mel_chunk_data,
    int               n_mel_chunk_frames,
    int               drop_extra_pre_encoded,
    int               mel_frames_advance)
{
    const auto & hp = pm->hparams;
    const int n_layers = static_cast<int>(pm->weights.blocks.size());

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    // Tear down any previous compute_ctx (one per chunk; matches the
    // offline run() lifecycle and keeps scratch bounded).
    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    ggml_init_params ip {};
    ip.mem_size   = 16 * 1024 * 1024;  // 16 MB; same as offline run path
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    pc->compute_ctx = ggml_init(ip);
    if (pc->compute_ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet stream: ggml_init compute_ctx failed");
        return TRANSCRIBE_ERR_OOM;
    }

    const int step_num = pc->stream_caches.chunk_step;
    const bool dump_on = transcribe::debug::enabled();

    // Per-step dump layer selection: match the Python dumper's default
    // ({0, n/2, n-1}). The selected layers are read AFTER graph_compute
    // for cache_out and BEFORE for cache_in.
    auto sel_layers = [&]() {
        std::vector<int> v;
        if (n_layers <= 0) return v;
        std::set<int> s = {0, n_layers / 2, n_layers - 1};
        for (int i : s) if (i >= 0 && i < n_layers) v.push_back(i);
        std::sort(v.begin(), v.end());
        return v;
    }();

    // Dump mel_in + cache_in (snapshot of inputs to this chunk).
    if (dump_on) {
        char namebuf[128];
        std::snprintf(namebuf, sizeof(namebuf),
                      "stream.chunk.%d.mel_in", step_num);
        const long long mel_shape[2] = {hp.fe_num_mels, n_mel_chunk_frames};
        transcribe::debug::dump_host_f32(
            namebuf, mel_chunk_data,
            static_cast<long long>(hp.fe_num_mels) * n_mel_chunk_frames,
            mel_shape, 2, "streaming.mel_in");
        for (int L : sel_layers) {
            std::snprintf(namebuf, sizeof(namebuf),
                          "stream.chunk.%d.cache_lc_in_%d", step_num, L);
            transcribe::debug::dump_tensor(
                namebuf, pc->stream_caches.last_channel[L],
                "streaming.cache_in");
            std::snprintf(namebuf, sizeof(namebuf),
                          "stream.chunk.%d.cache_lt_in_%d", step_num, L);
            transcribe::debug::dump_tensor(
                namebuf, pc->stream_caches.last_time[L],
                "streaming.cache_in");
        }
    }

    StreamingEncoderCacheIO cache_io;
    cache_io.channel_in = pc->stream_caches.last_channel;
    cache_io.time_in    = pc->stream_caches.last_time;

    EncoderBuild eb = build_encoder_graph_streaming(
        pc->compute_ctx, pm->weights, hp,
        n_mel_chunk_frames, drop_extra_pre_encoded,
        cache_io, resolved_kv, pm->backend.c_str());
    if (eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(
            pm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(pm->plan.scheduler_list.size()),
            /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet stream: sched_new failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, eb.graph)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet stream: alloc_graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Upload mel chunk. Row-major [n_mels, n_mel_chunk_frames] is
    // byte-identical to ggml ne=[n_mel_chunk_frames, n_mels, 1, 1].
    ggml_backend_tensor_set(
        eb.mel_in, mel_chunk_data, 0,
        static_cast<size_t>(n_mel_chunk_frames) *
            static_cast<size_t>(hp.fe_num_mels) * sizeof(float));

    // Build & upload pos_emb (sized for T_virtual).
    fill_streaming_pos_emb(pc, eb.pos_emb_in, hp.enc_d_model);

    // Build & upload chunked mask with cache-unfilled prefix masking.
    // The mask band is a function of the (att_context_left, att_context_right)
    // RESOLVED for this stream from the caller's latency selection, not the
    // model-default hparams. They coincide today (left is invariant across
    // the menu, and the per-chunk geometry derived from chosen_right keeps
    // the default-R chunking accidentally equivalent on the new-frame rows),
    // but reading the resolved values makes the mask correct by construction
    // for any future variant whose menu breaks those invariants.
    const int T_virtual = static_cast<int>(eb.chunked_mask_in->ne[0]);
    const int T_cache   = hp.enc_att_context_left;
    const int T_q_new   = T_virtual - T_cache;
    fill_streaming_chunked_mask(
        eb.chunked_mask_in, T_virtual, T_cache,
        pc->stream_caches.channel_len,
        pc->stream_caches.att_context_left,
        pc->stream_caches.att_context_right);

    // Prompt one-hot upload (multilingual variants only). Same shape
    // contract as the offline path: [num_prompts, T_q_new, 1, 1] holds
    // a single one-hot column at the resolved language's index,
    // replicated across T_q_new. The streaming language hint lives in
    // pc->stream_run_params (captured at stream_begin).
    if (eb.prompt_one_hot_in != nullptr) {
        const int P    = static_cast<int>(eb.prompt_one_hot_in->ne[0]);
        const int T_oh = static_cast<int>(eb.prompt_one_hot_in->ne[1]);
        const char * lang_hint = pc->stream_run_params.language;
        const int32_t pid = resolve_prompt_id(pm->hparams, lang_hint);
        if (pid < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet stream: language %s%s%s not in prompt "
                         "dictionary",
                         lang_hint ? "\"" : "",
                         lang_hint ? lang_hint : "<null>",
                         lang_hint ? "\"" : "");
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
        std::vector<float> one_hot_buf;
        if (!fill_prompt_one_hot(one_hot_buf, P, T_oh, /*n_batch=*/1,
                                 {pid}))
        {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet stream: prompt_id %d out of range "
                         "[0, %d)", pid, P);
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(eb.prompt_one_hot_in, one_hot_buf.data(),
                                0, one_hot_buf.size() * sizeof(float));
    }

    // Thread count (same recipe as offline run()).
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
            if (fn != nullptr) fn(be, n_threads);
        }
    }

    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(pc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet stream: graph_compute failed (%d)",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }

    // Read encoder output back to host.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    pc->enc_host.resize(
        static_cast<size_t>(d_enc) * static_cast<size_t>(T_q_new));
    ggml_backend_tensor_get(eb.out, pc->enc_host.data(), 0,
                            pc->enc_host.size() * sizeof(float));

    // Dump enc_out + cache_out (snapshot of outputs from this chunk).
    // Done BEFORE the cache rotation so we read the freshly-computed
    // cache_out tensors. channel_len is dumped post-rotation since
    // that matches NeMo's "cache_last_channel_len after chunk" return.
    if (dump_on) {
        char namebuf[128];
        std::snprintf(namebuf, sizeof(namebuf),
                      "stream.chunk.%d.enc_out", step_num);
        // Match the Python dumper's to_np() squeeze: when T_q_new == 1
        // (R=0 at the lowest-latency setting), drop the leading size-1
        // dim so the on-disk shape becomes [d_enc] not [1, d_enc].
        // compare_tensors.py treats shape mismatches as failures
        // regardless of element values.
        if (T_q_new == 1) {
            const long long enc_shape[1] = {d_enc};
            transcribe::debug::dump_host_f32(
                namebuf, pc->enc_host.data(), d_enc,
                enc_shape, 1, "streaming.enc_out");
        } else {
            const long long enc_shape[2] = {T_q_new, d_enc};
            transcribe::debug::dump_host_f32(
                namebuf, pc->enc_host.data(),
                static_cast<long long>(T_q_new) * d_enc,
                enc_shape, 2, "streaming.enc_out");
        }
        for (int L : sel_layers) {
            std::snprintf(namebuf, sizeof(namebuf),
                          "stream.chunk.%d.cache_lc_out_%d", step_num, L);
            transcribe::debug::dump_tensor(
                namebuf, cache_io.channel_out[L],
                "streaming.cache_out");
            std::snprintf(namebuf, sizeof(namebuf),
                          "stream.chunk.%d.cache_lt_out_%d", step_num, L);
            transcribe::debug::dump_tensor(
                namebuf, cache_io.time_out[L],
                "streaming.cache_out");
        }
    }

    // Rotate per-layer caches: cache_out → persistent cache_in. Both
    // tensors live on the model's primary backend, so a backend-side
    // copy avoids host roundtripping. The null check is defense in
    // depth — every supported R produces a fully-allocated cache_out
    // (the conv_module cache_next logic in conformer/conformer.cpp
    // covers T_q_new < pad_left explicitly), so reaching this branch
    // indicates a builder bug.
    for (int i = 0; i < n_layers; ++i) {
        if (cache_io.channel_out[i] == nullptr || cache_io.channel_out[i]->buffer == nullptr ||
            cache_io.time_out[i]    == nullptr || cache_io.time_out[i]->buffer    == nullptr)
        {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet stream: cache_out unallocated at layer %d "
                "(att_context_right=%d) — builder bug",
                i, pc->stream_caches.att_context_right);
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_copy(cache_io.channel_out[i],
                                 pc->stream_caches.last_channel[i]);
        ggml_backend_tensor_copy(cache_io.time_out[i],
                                 pc->stream_caches.last_time[i]);
    }
    pc->stream_caches.channel_len = std::min(
        T_cache, pc->stream_caches.channel_len + T_q_new);

    if (dump_on) {
        char namebuf[128];
        std::snprintf(namebuf, sizeof(namebuf),
                      "stream.chunk.%d.channel_len", step_num);
        const float channel_len_f = static_cast<float>(pc->stream_caches.channel_len);
        const long long len_shape[1] = {1};
        transcribe::debug::dump_host_f32(
            namebuf, &channel_len_f, 1, len_shape, 1,
            "streaming.channel_len");
    }
    pc->stream_caches.chunk_step += 1;

    // Run streaming RNN-T decoder on the new encoder frames.
    if (const transcribe_status st = decode_rnnt_greedy_streaming(
            pm->host_decoder, pc->enc_host.data(),
            T_q_new, d_enc,
            pc->stream_dec_state.lstm_state,
            pc->stream_dec_state.prev_token_id,
            static_cast<int>(pc->stream_dec_state.frame_offset),
            pc->n_threads, pc->raw_tokens);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    pc->stream_dec_state.frame_offset += T_q_new;
    // Cursor advances by mel_frames_advance (caller computes:
    // chunk_size_first on the first chunk, chunk_size_subsequent
    // afterward). This is the same as NeMo's shift_size: it is
    // INDEPENDENT of the chunk we fed (which may include a
    // pre-encode-cache prepend).
    pc->stream_caches.mel_frames_consumed += mel_frames_advance;
    return TRANSCRIBE_OK;
}

// Rebuild the public result vectors (tokens, full_text) from the
// committed raw_tokens. Simple and idempotent: clears the vectors and
// re-populates from scratch every call. Segments / words are NOT
// populated during streaming — those derive from richer logic in the
// offline run() result builder and are deferred to stream_finalize
// (or omitted entirely for the first M2 cut).
//
// result_kind is TOKEN whenever any tokens are present: each TokenEntry
// gets real t0_ms / t1_ms from step_at_emit, so the snapshot is honest
// at token granularity. Words/segments stay empty until finalize, but
// a caller that asked for WORD or SEGMENT timestamps already passed the
// max_timestamp_kind gate at begin time (Parakeet advertises TOKEN as
// its max), so TOKEN here is the strongest honest answer we can give.
void rebuild_streaming_result_text(ParakeetSession * pc,
                                   const ParakeetModel * pm)
{
    pc->tokens.clear();
    pc->words.clear();
    pc->segments.clear();
    pc->full_text.clear();

    if (pc->raw_tokens.empty()) {
        pc->has_result  = false;
        pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        return;
    }

    const auto & hp = pm->hparams;
    // Frame-to-ms ratio: encoder frame at index f spans
    // f * (hop * subsampling) / sample_rate seconds.
    const int64_t ms_per_frame =
        static_cast<int64_t>(hp.fe_hop_length) *
        static_cast<int64_t>(hp.enc_subsampling_factor) *
        1000 / static_cast<int64_t>(hp.fe_sample_rate);

    pc->tokens.reserve(pc->raw_tokens.size());
    std::vector<int32_t> all_ids;
    all_ids.reserve(pc->raw_tokens.size());
    for (const auto & tk : pc->raw_tokens) {
        transcribe_session::TokenEntry te;
        te.id           = tk.id;
        te.p            = tk.p;
        te.t0_ms        = tk.step_at_emit * ms_per_frame;
        te.t1_ms        = (tk.step_at_emit + tk.duration_frames) * ms_per_frame;
        te.seg_index    = 0;
        te.word_index   = -1;
        te.text         = pm->tok.decode(&tk.id, 1);
        pc->tokens.push_back(std::move(te));
        all_ids.push_back(tk.id);
    }
    pc->full_text = pm->tok.decode(all_ids.data(),
                                      static_cast<int>(all_ids.size()));
    if (!pc->full_text.empty() && pc->full_text.front() == ' ') {
        pc->full_text.erase(pc->full_text.begin());
    }
    pc->has_result  = true;
    pc->result_kind = TRANSCRIBE_TIMESTAMPS_TOKEN;
}

// ----- Buffered streaming (parakeet-unified-en-0.6b) -----------------
//
// Mirrors NeMo's `speech_to_text_streaming_infer_rnnt.py` reference at
// the per-chunk granularity. The variable-stride algorithm:
//   - Step 0 (initial fill): num_new = samples_chunk + samples_right.
//   - Steady state: num_new = samples_chunk.
//   - Final step (from finalize): num_new = remaining audio; the
//     trailing right context slot is folded into chunk via
//     `add_frames_get_removed_(is_last=true)`.
// Per step:
//   - Update the buffer's internal ContextSize (buf_ctx_*) via the
//     same accounting NeMo's StreamingBatchedAudioBuffer uses.
//   - Slice the last session.total() samples from stream_pcm_buffer as
//     the encoder window.
//   - Compute mel over the full window.
//   - Build the encoder graph with a BufferedStreamMaskOverride to
//     engage the chunked_limited_with_rc attention mask sized for
//     the expected (L, C, R).
//   - Slice off the first (ctx_left / samples_per_frame) encoder
//     frames and decode (ctx_chunk / samples_per_frame) on non-last
//     or (T_enc_full - ctx_left_frames) on last with greedy RNN-T
//     plus carried LstmState. Matches the reference's
//     encoder_context_batch.chunk vs
//     encoder_output_len - encoder_context_batch.left dispatch.
static void buf_ctx_add_frames(
    ParakeetSession * pc, int64_t num_new, bool is_last)
{
    pc->buf_ctx_left  += pc->buf_ctx_chunk;
    pc->buf_ctx_chunk  = 0;
    pc->buf_ctx_right += num_new;
    if (is_last) {
        pc->buf_ctx_chunk = pc->buf_ctx_right;
        pc->buf_ctx_right = 0;
    } else {
        pc->buf_ctx_chunk  = pc->buf_samples_chunk;
        pc->buf_ctx_right -= static_cast<int64_t>(pc->buf_samples_chunk);
    }
    const int64_t total_now = pc->buf_ctx_left + pc->buf_ctx_chunk + pc->buf_ctx_right;
    const int64_t expected  = static_cast<int64_t>(pc->buf_samples_left) +
                              static_cast<int64_t>(pc->buf_samples_chunk) +
                              static_cast<int64_t>(pc->buf_samples_right);
    const int64_t extra = std::max<int64_t>(total_now - expected, 0);
    pc->buf_ctx_left -= extra;
}

transcribe_status emit_buffered_chunk(
    ParakeetSession * pc,
    ParakeetModel *   pm,
    int64_t           num_new_samples,
    bool              is_last_chunk)
{
    if (pc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    const auto & hp = pm->hparams;
    if (!pm->mel.has_value()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet buffered: model has no MelFrontend");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int samples_per_frame =
        hp.enc_subsampling_factor * hp.fe_hop_length;
    if (samples_per_frame <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

    // ----- Update buffer ContextSize (mirrors NeMo's add_frames_get_removed_) -----
    buf_ctx_add_frames(pc, num_new_samples, is_last_chunk);

    // ----- Build the [left | chunk | right] PCM window from absolute coords -----
    const int64_t end_abs   = pc->buf_next_audio_read + num_new_samples;
    const int64_t total_now =
        pc->buf_ctx_left + pc->buf_ctx_chunk + pc->buf_ctx_right;
    const int effective_T = static_cast<int>(total_now / samples_per_frame);
    const int64_t start_abs = end_abs - total_now;

    std::vector<float> window_pcm(
        static_cast<size_t>(total_now), 0.0f);
    const int64_t buf_size =
        static_cast<int64_t>(pc->stream_pcm_buffer.size());
    for (int64_t i = 0; i < total_now; ++i) {
        const int64_t src = start_abs + i;
        if (src >= 0 && src < buf_size) {
            window_pcm[static_cast<size_t>(i)] =
                pc->stream_pcm_buffer[static_cast<size_t>(src)];
        }
    }

    // ----- Per-chunk dumps (TRANSCRIBE_DUMP_DIR; mirror NeMo names) -----
    //
    // Push a `stream.chunk.<N>.` prefix so every dump_tensor inside the
    // encoder graph builder gets scoped per chunk (otherwise the block
    // outputs `enc.block.<L>.out` would overwrite across chunks). The
    // prefix is popped at function exit via the guard below.
    const int chunk_step = pc->buf_chunk_step;
    char chunk_prefix[64];
    std::snprintf(chunk_prefix, sizeof(chunk_prefix),
                  "stream.chunk.%d.", chunk_step);
    transcribe::debug::push_name_prefix(chunk_prefix);
    struct PrefixPopGuard {
        ~PrefixPopGuard() { transcribe::debug::pop_name_prefix(); }
    } _prefix_pop_guard;
    if (transcribe::debug::enabled()) {
        const long long shape[1] = { static_cast<long long>(window_pcm.size()) };
        transcribe::debug::dump_host_f32(
            "audio_in", window_pcm.data(),
            static_cast<long long>(window_pcm.size()),
            shape, 1, "buffered_streaming.audio_in");
    }

    // ----- Mel -----
    const int64_t t_mel_start = ggml_time_us();
    int mel_n_mels   = 0;
    int mel_n_frames = 0;
    if (const transcribe_status mst = pm->mel->compute(
            window_pcm.data(), window_pcm.size(),
            pc->mel_buf, mel_n_mels, mel_n_frames);
        mst != TRANSCRIBE_OK)
    {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet buffered: MelFrontend::compute failed (%s)",
                     transcribe_status_string(mst));
        return mst;
    }
    pc->t_mel_us += ggml_time_us() - t_mel_start;

    // ----- Reset per-chunk compute state -----
    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    pc->encoder_out = nullptr;
    {
        ggml_init_params init_params {};
        init_params.mem_size   = 4 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        pc->compute_ctx = ggml_init(init_params);
        if (pc->compute_ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet buffered: ggml_init for compute_ctx failed");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (pc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    BufferedStreamMaskOverride buf_mask {};
    buf_mask.left_frames  = pc->buf_left_frames;
    buf_mask.chunk_frames = pc->buf_chunk_frames;
    buf_mask.right_frames = pc->buf_right_frames;
    buf_mask.valid_frames = effective_T;

    EncoderBuild eb = build_encoder_graph(
        pc->compute_ctx, pm->weights, pm->hparams, mel_n_frames,
        resolved_kv, pm->backend.c_str(), &buf_mask);
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Scheduler alloc + mel upload -----
    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(
            pm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(pm->plan.scheduler_list.size()),
            /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, eb.graph)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_tensor_set(eb.mel_in, pc->mel_buf.data(),
                            0, pc->mel_buf.size() * sizeof(float));

    // ----- Pos_emb fill (full 2T-1 layout; matches offline ChunkedLimited path) -----
    if (eb.pos_emb_in != nullptr) {
        const int d_model = hp.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
        const int zero_index = (pos_len - 1) / 2;

        pc->pos_buf.assign(
            static_cast<size_t>(pos_len) * d_model, 0.0f);
        pc->pos_div_term.resize(static_cast<size_t>(d_model / 2));
        const float ln_10000 = std::log(10000.0f);
        for (int k = 0; k < d_model / 2; ++k) {
            pc->pos_div_term[static_cast<size_t>(k)] =
                std::exp(static_cast<float>(2 * k) *
                         (-ln_10000 / static_cast<float>(d_model)));
        }
        for (int i = 0; i < pos_len; ++i) {
            const float pos = static_cast<float>(zero_index - i);
            float * row = pc->pos_buf.data() +
                static_cast<size_t>(i) * d_model;
            for (int k = 0; k < d_model / 2; ++k) {
                const float div = pc->pos_div_term[static_cast<size_t>(k)];
                row[2 * k]     = std::sin(pos * div);
                row[2 * k + 1] = std::cos(pos * div);
            }
        }
        ggml_backend_tensor_set(eb.pos_emb_in, pc->pos_buf.data(),
                                0, pc->pos_buf.size() * sizeof(float));
    }

    // ----- Conformer conv pad mask -----
    //
    // NeMo passes pad_mask into every Conformer conv module and zeros
    // post-GLU activations where pre_encode emitted padded overhang
    // frames. The attention mask below already excludes those frames
    // from MHA; this mask prevents them from leaking backward through
    // the depthwise conv's right context.
    if (eb.conv_pad_mask_in != nullptr) {
        const int T_enc =
            static_cast<int>(eb.conv_pad_mask_in->ne[0]);
        const int P = std::max(0, std::min(effective_T, T_enc));
        std::vector<float> mask_buf(static_cast<size_t>(T_enc), 1.0f);
        for (int t = P; t < T_enc; ++t) {
            mask_buf[static_cast<size_t>(t)] = 0.0f;
        }
        ggml_backend_tensor_set(eb.conv_pad_mask_in, mask_buf.data(),
                                0, mask_buf.size() * sizeof(float));
        transcribe::debug::dump_tensor(
            "enc.conv.pad_mask",
            eb.conv_pad_mask_in,
            "encoder.conv.pad_mask");
    }

    // ----- Chunked_limited_with_rc mask fill -----
    //
    // Pad-mask: NeMo's encoder ANDs a conv-overhang pad_mask onto the
    // chunked mask before MHA — frames past `padding_length` (post
    // pre-encode) are masked out. The conv subsampling can emit T_enc
    // > session.total/samples_per_frame when the input doesn't tile the
    // subsampling stride exactly; without the pad mask, those trailing
    // frames contaminate every other frame's attention scores. We
    // mirror NeMo by passing `effective_T = session.total /
    // samples_per_frame` so the mask compute folds in the pad_mask.
    // Critical at low-C/low-R configs where one extra frame of
    // contamination tips many greedy-emission decisions; barely
    // noticeable at (70,13,13).
    if (eb.chunked_mask_in != nullptr) {
        const int T_enc =
            static_cast<int>(eb.chunked_mask_in->ne[0]);
        std::vector<float> mask_buf(
            static_cast<size_t>(T_enc) * static_cast<size_t>(T_enc));
        compute_chunked_limited_with_rc_mask(
            mask_buf.data(), T_enc,
            pc->buf_left_frames,
            pc->buf_chunk_frames,
            pc->buf_right_frames,
            effective_T);
        ggml_backend_tensor_set(eb.chunked_mask_in, mask_buf.data(),
                                0, mask_buf.size() * sizeof(float));
    }

    // ----- Threading -----
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
            if (fn != nullptr) fn(be, n_threads);
        }
    }

    // ----- Compute -----
    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(pc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet buffered: sched_graph_compute failed (%d)",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    pc->t_encode_us += ggml_time_us() - t_enc_start;

    pc->encoder_out = eb.out;

    // Dump cpp's mel input alongside the per-block intermediates so the
    // ref-vs-cpp bisect can also localize drift to the mel frontend.
    if (transcribe::debug::enabled()) {
        transcribe::debug::dump_tensor("enc.mel.in", eb.mel_in, "encoder.mel");
    }

    // ----- Per-chunk intermediate dumps (debug only) -----
    //
    // Same set of intermediates the offline path dumps via run(); used
    // for layer-by-layer divergence bisect against NeMo's per-block
    // outputs at the streaming geometry. Gated on TRANSCRIBE_DUMP_DIR
    // and TRANSCRIBE_DUMP_ALL_BLOCKS env vars. The active per-chunk
    // name prefix (push_name_prefix above) scopes these names to this
    // chunk's directory: `stream.chunk.<N>.enc.block.<L>.out` etc.
    if (transcribe::debug::enabled()) {
        auto try_dump = [](const char * name, ggml_tensor * t,
                           const char * stage) {
            if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
        };
        try_dump("enc.pre_encode.out",   eb.dumps.pre_encode_out,    "encoder.pre_encode");
        try_dump("enc.block.0.ff1",      eb.dumps.block0_after_ff1,  "encoder.block0.ff1");
        try_dump("enc.block.0.attn",     eb.dumps.block0_after_attn, "encoder.block0.attn");
        try_dump("enc.block.0.conv",     eb.dumps.block0_after_conv, "encoder.block0.conv");
        try_dump("enc.block.0.ff2",      eb.dumps.block0_after_ff2,  "encoder.block0.ff2");
        try_dump("enc.block.0.out",      eb.dumps.block0_out,        "encoder.block0.out");
        if (eb.dumps.mid_block_out != nullptr && eb.dumps.mid_block_idx >= 0) {
            char name[64];
            std::snprintf(name, sizeof(name), "enc.block.%d.out",
                          eb.dumps.mid_block_idx);
            transcribe::debug::dump_tensor(name, eb.dumps.mid_block_out,
                                           "encoder.block.mid.out");
        }
        if (eb.dumps.last_block_out != nullptr && eb.dumps.last_block_idx >= 0) {
            char name[64];
            std::snprintf(name, sizeof(name), "enc.block.%d.out",
                          eb.dumps.last_block_idx);
            transcribe::debug::dump_tensor(name, eb.dumps.last_block_out,
                                           "encoder.block.last.out");
        }
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
    }

    // ----- Readback encoder output, slice off left frames -----
    const int d_enc      = static_cast<int>(eb.out->ne[0]);
    const int T_enc_full = static_cast<int>(eb.out->ne[1]);
    if (T_enc_full <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet buffered: encoder produced %d frames",
                     T_enc_full);
        return TRANSCRIBE_ERR_GGUF;
    }
    pc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc_full));
    ggml_backend_tensor_get(eb.out, pc->enc_host.data(), 0,
                            pc->enc_host.size() * sizeof(float));

    transcribe::debug::dump_tensor(
        "enc.final", eb.out, "encoder.final");

    // Subsample session into encoder frames — mirrors NeMo's
    // buffer.context_size.subsample(factor=encoder_frame2audio_samples).
    const int ctx_left_frames  = static_cast<int>(
        pc->buf_ctx_left  / samples_per_frame);
    const int ctx_chunk_frames = static_cast<int>(
        pc->buf_ctx_chunk / samples_per_frame);

    auto advance_cursor = [&]() {
        pc->buf_next_audio_read += num_new_samples;
        pc->buf_chunk_step      += 1;
        pc->buf_initialized      = true;
    };

    if (T_enc_full <= ctx_left_frames) {
        // Window was too small to produce any chunk-region frames
        // (e.g. degenerate audio shorter than the left context). Skip
        // emission for this chunk.
        advance_cursor();
        return TRANSCRIBE_OK;
    }

    // Decode length matches NeMo's dispatch:
    //   non-final:   encoder_context_batch.chunk
    //   final:       encoder_output_len - encoder_context_batch.left
    const int T_chunk_avail = T_enc_full - ctx_left_frames;
    const int T_to_decode = is_last_chunk
        ? T_chunk_avail
        : std::min(ctx_chunk_frames, T_chunk_avail);
    if (T_to_decode <= 0) {
        advance_cursor();
        return TRANSCRIBE_OK;
    }

    const float * enc_chunk = pc->enc_host.data() +
        static_cast<size_t>(ctx_left_frames) * static_cast<size_t>(d_enc);

    // Per-chunk encoder output dump. Mirrors NeMo's reference: emit
    // the FULL post-slice encoder output (chunk + right context),
    // not just the frames that get decoded this step. The decoder
    // gates on T_to_decode below; the dump captures the entire
    // chunk-aligned region so the parity harness can compare both
    // the decoded and lookahead portions.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = {
            static_cast<long long>(T_chunk_avail),
            static_cast<long long>(d_enc),
        };
        transcribe::debug::dump_host_f32(
            "enc_out", enc_chunk,
            static_cast<long long>(T_chunk_avail) *
                static_cast<long long>(d_enc),
            shape, 2, "buffered_streaming.enc_out");
    }

    // ----- Decode with carried RNN-T state -----
    const int64_t t_dec_start = ggml_time_us();
    if (const transcribe_status st = decode_rnnt_greedy_streaming(
            pm->host_decoder, enc_chunk, T_to_decode, d_enc,
            pc->stream_dec_state.lstm_state,
            pc->stream_dec_state.prev_token_id,
            static_cast<int>(pc->stream_dec_state.frame_offset),
            pc->n_threads, pc->raw_tokens);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    pc->t_decode_us += ggml_time_us() - t_dec_start;

    pc->stream_dec_state.frame_offset += T_to_decode;
    advance_cursor();
    return TRANSCRIBE_OK;
}

// Pure validation of the buffered-stream extension. Resolves (L, C, R)
// to encoder frames using the same sentinel semantics stream_begin uses
// to configure the buffered path; emits the same stderr diagnostics on
// rejection. Does NOT touch pc->buf_*. Output pointers are only written
// on TRANSCRIBE_OK return.
transcribe_status resolve_buffered_stream_geom(
    const ParakeetModel *             pm,
    const transcribe_stream_params *  stream_params,
    int *                             out_L_frames,
    int *                             out_C_frames,
    int *                             out_R_frames)
{
    const int frame_ms =
        (pm->hparams.enc_subsampling_factor * pm->hparams.fe_hop_length * 1000) /
        std::max(pm->hparams.fe_sample_rate, 1);

    auto max_in = [](const std::vector<int32_t> & v) -> int32_t {
        int32_t best = 0;
        for (auto x : v) if (x > best) best = x;
        return best;
    };
    const int default_L = static_cast<int>(max_in(pm->hparams.enc_att_chunk_left_choices));
    const int default_C = static_cast<int>(max_in(pm->hparams.enc_att_chunk_chunk_choices));
    const int default_R = static_cast<int>(max_in(pm->hparams.enc_att_chunk_right_choices));

    int req_L_ms = -1, req_C_ms = -1, req_R_ms = -1;
    const transcribe_ext * family = stream_params != nullptr ? stream_params->family : nullptr;
    if (const transcribe_status st = transcribe_ext_check(
            family,
            TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM,
            sizeof(struct transcribe_parakeet_buffered_stream_ext));
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (family != nullptr) {
        const auto * px = reinterpret_cast<const transcribe_parakeet_buffered_stream_ext *>(family);
        req_L_ms = px->left_ms;
        req_C_ms = px->chunk_ms;
        req_R_ms = px->right_ms;
    }
    const int safe_frame_ms = std::max(frame_ms, 1);
    auto ms_to_exact_frames = [&](int req_ms, int default_frames,
                                  int * out_frames) -> bool {
        if (req_ms == -1) { *out_frames = default_frames; return true; }
        if (req_ms <  -1) return false;
        if (req_ms % safe_frame_ms != 0) return false;
        *out_frames = req_ms / safe_frame_ms;
        return true;
    };
    int L_frames = 0, C_frames = 0, R_frames = 0;
    if (!ms_to_exact_frames(req_L_ms, default_L, &L_frames) ||
        !ms_to_exact_frames(req_C_ms, default_C, &C_frames) ||
        !ms_to_exact_frames(req_R_ms, default_R, &R_frames))
    {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                     "parakeet buffered: requested (L, C, R)_ms = "
                     "(%d, %d, %d) is invalid. Use -1 on any field to "
                     "select the model default; otherwise the value "
                     "must be 0 or a positive exact multiple of the "
                     "%d ms encoder frame.",
                     req_L_ms, req_C_ms, req_R_ms, safe_frame_ms);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto contains = [](const std::vector<int32_t> & v, int x) -> bool {
        for (auto y : v) if (y == x) return true;
        return false;
    };
    if (!contains(pm->hparams.enc_att_chunk_left_choices, L_frames) ||
        !contains(pm->hparams.enc_att_chunk_chunk_choices, C_frames) ||
        !contains(pm->hparams.enc_att_chunk_right_choices, R_frames))
    {
        std::string allowed = "L=";
        for (auto v : pm->hparams.enc_att_chunk_left_choices)  allowed += std::to_string(v) + ",";
        allowed += " C=";
        for (auto v : pm->hparams.enc_att_chunk_chunk_choices) allowed += std::to_string(v) + ",";
        allowed += " R=";
        for (auto v : pm->hparams.enc_att_chunk_right_choices) allowed += std::to_string(v) + ",";
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet buffered: requested (L, C, R) = (%d, %d, %d) "
                "encoder frames not in model menu. Allowed %s",
                L_frames, C_frames, R_frames, allowed.c_str());
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (C_frames < 1) return TRANSCRIBE_ERR_INVALID_ARG;

    *out_L_frames = L_frames;
    *out_C_frames = C_frames;
    *out_R_frames = R_frames;
    return TRANSCRIBE_OK;
}

// Pure validation of the cache-aware-stream extension. Resolves the
// active (att_context_left, att_context_right) for the stream from the
// model's training menu. Does NOT touch pc->*. Output pointers are only
// written on TRANSCRIBE_OK return.
transcribe_status resolve_cache_aware_stream_geom(
    const ParakeetModel *             pm,
    const transcribe_stream_params *  stream_params,
    int *                             out_chosen_left,
    int *                             out_chosen_right)
{
    int chosen_right = pm->hparams.enc_att_context_right;
    int chosen_left  = pm->hparams.enc_att_context_left;

    const transcribe_ext * family = stream_params != nullptr ? stream_params->family : nullptr;
    if (const transcribe_status st = transcribe_ext_check(
            family,
            TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
            sizeof(struct transcribe_parakeet_stream_ext));
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (family != nullptr) {
        const auto * px = reinterpret_cast<const transcribe_parakeet_stream_ext *>(family);
        const int requested = px->att_context_right;
        if (requested < -1) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                         "parakeet: att_context_right=%d is invalid; "
                         "use -1 for the model default or >=0 to pick "
                         "an entry from the model's training menu",
                         requested);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        if (requested >= 0) {
            bool matched = false;
            for (const auto & p : pm->hparams.enc_att_context_size_choices) {
                if (p.second == requested) {
                    chosen_right = p.second;
                    chosen_left  = p.first;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                std::string available;
                for (const auto & p : pm->hparams.enc_att_context_size_choices) {
                    available += std::to_string(p.second) + " ";
                }
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                        "parakeet: requested att_context_right=%d "
                        "not in model's training menu; available: %s",
                        requested, available.c_str());
                return TRANSCRIBE_ERR_INVALID_ARG;
            }
        }
    }

    *out_chosen_left  = chosen_left;
    *out_chosen_right = chosen_right;
    return TRANSCRIBE_OK;
}

// Pre-flight: validate caller-supplied extension fields without mutating
// session or model state. Called by the dispatcher BEFORE clear_result,
// so a rejection here leaves the previous utterance's snapshot intact.
// stream_begin re-runs the same resolvers as defense in depth and uses
// their parsed outputs to configure per-stream state.
transcribe_status stream_validate(
    const transcribe_session *        session,
    const transcribe_run_params *     /*run_params*/,
    const transcribe_stream_params *  stream_params)
{
    const auto * pc = static_cast<const ParakeetSession *>(session);
    const auto * pm = static_cast<const ParakeetModel *>(pc->model);
    if (pm == nullptr || pm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const bool is_chunked_limited =
        (pm->hparams.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimited);
    const bool is_chunked_with_rc =
        (pm->hparams.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimitedWithRc);
    if (!is_chunked_limited && !is_chunked_with_rc) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    if (is_chunked_with_rc) {
        int L = 0, C = 0, R = 0;
        return resolve_buffered_stream_geom(pm, stream_params, &L, &C, &R);
    }

    int left = 0, right = 0;
    return resolve_cache_aware_stream_geom(pm, stream_params, &left, &right);
}

transcribe_status stream_begin(
    transcribe_session *              session,
    const transcribe_run_params *         run_params,
    const transcribe_stream_params *  stream_params)
{
    auto * pc = static_cast<ParakeetSession *>(session);
    auto * pm = static_cast<ParakeetModel *>(pc->model);
    if (pm == nullptr || pm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Initialize the debug dumper from TRANSCRIBE_DUMP_DIR. Idempotent.
    // Streaming may run without ever going through run() (the offline
    // path), so this is needed in addition to the run() callsite.
    transcribe::debug::init();

    // Defense in depth: the dispatcher should have already rejected
    // begin via the supports_streaming gate for non-streaming variants,
    // but reject again here in case a future refactor lets a Regular
    // (offline) variant slip through.
    const bool is_chunked_limited =
        (pm->hparams.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimited);
    const bool is_chunked_with_rc =
        (pm->hparams.enc_att_context_style ==
             ParakeetHParams::AttContextStyle::ChunkedLimitedWithRc);
    if (!is_chunked_limited && !is_chunked_with_rc) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    // -------- Buffered streaming path (parakeet-unified-en-0.6b) --------
    //
    // The unified model declares chunked_limited_with_rc and ships a
    // 3-tuple training menu. Buffered streaming re-runs the offline
    // encoder on a sliding [left | chunk | right] PCM window per chunk;
    // there's no per-layer cache to maintain. The RNN-T predictor +
    // joint reuse the existing greedy-streaming decoder so the only
    // new state is the runtime (L, C, R) geometry and the LSTM carry.
    if (is_chunked_with_rc) {
        // Resolve and validate (L, C, R) in encoder frames. The same
        // helper runs in stream_validate, so the dispatcher has
        // already rejected bad caller input before clearing the
        // previous result; the call here is defense in depth and
        // also produces the parsed values we configure state from.
        int L_frames = 0, C_frames = 0, R_frames = 0;
        if (const transcribe_status st = resolve_buffered_stream_geom(
                pm, stream_params, &L_frames, &C_frames, &R_frames);
            st != TRANSCRIBE_OK)
        {
            return st;
        }

        const int subsampling_factor = pm->hparams.enc_subsampling_factor;
        const int hop                = pm->hparams.fe_hop_length;
        const int samples_per_frame  = subsampling_factor * hop;

        pc->buf_left_frames     = L_frames;
        pc->buf_chunk_frames    = C_frames;
        pc->buf_right_frames    = R_frames;
        pc->buf_samples_left    = L_frames * samples_per_frame;
        pc->buf_samples_chunk   = C_frames * samples_per_frame;
        pc->buf_samples_right   = R_frames * samples_per_frame;
        pc->buf_next_audio_read = 0;
        pc->buf_ctx_left        = 0;
        pc->buf_ctx_chunk       = 0;
        pc->buf_ctx_right       = 0;
        pc->buf_initialized     = false;
        pc->buf_chunk_step      = 0;
        pc->buf_active          = true;

        pc->stream_pcm_buffer.clear();
        pc->raw_tokens.clear();
        pc->stream_run_params = *run_params;
        reset_streaming_decoder_state(pc, pm);
        // For buffered streaming the predictor state's frame_offset
        // starts at 0; we advance it per-chunk by T_to_decode so token
        // step_at_emit lands in stream-wide encoder-frame coordinates.
        pc->stream_dec_state.frame_offset = 0;

        return TRANSCRIBE_OK;
    }

    // -------- Cache-aware streaming path (nemotron-speech-streaming) --------
    pc->buf_active = false;

    // Resolve the active (att_context_left, att_context_right) for this
    // stream. Same helper runs in stream_validate; rerunning here is
    // defense in depth and also extracts the values we use to configure
    // the per-stream caches.
    int chosen_left = 0, chosen_right = 0;
    if (const transcribe_status st = resolve_cache_aware_stream_geom(
            pm, stream_params, &chosen_left, &chosen_right);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    pc->stream_pcm_buffer.clear();
    pc->raw_tokens.clear();  // The dispatcher clears session->tokens / words /
                             // segments / full_text via clear_result, but
                             // raw_tokens is parakeet-internal and
                             // accumulates per-chunk; without this reset
                             // a back-to-back stream_begin (batch WER) on
                             // the same context would carry the previous
                             // utterance's tokens into the next stream.
    pc->stream_run_params = *run_params;

    // M2: allocate streaming caches on first stream_begin for this
    // context (idempotent — survives across utterances). Zero the
    // contents and reset cursors on every begin so each stream starts
    // from NeMo's get_initial_cache_state (zeros).
    if (const transcribe_status st = init_streaming_caches(pc, pm);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    zero_streaming_caches(pc);
    reset_streaming_decoder_state(pc, pm);

    // Resolve chunking geometry for this stream. Matches NeMo's
    // setup_streaming_params formulas exactly:
    //
    //   chunk_size_subsequent = subsampling_factor * (1 + R)
    //   chunk_size_first      = sampling_frames_first + subsampling_factor * R
    //                         = chunk_size_subsequent - (subsampling_factor
    //                           - sampling_frames_first)
    //   shift_size_subsequent = chunk_size_subsequent (cache_drop_size=0
    //                           for chunked_limited)
    //   mel_fed_first         = chunk_size_first  (pre_encode_cache_size[0] = 0)
    //   mel_fed_subsequent    = chunk_size_subsequent + pre_encode_cache_size
    //   drop_extra_first      = 0
    //   drop_extra_subsequent = streaming_cfg.drop_extra_pre_encoded
    //
    // Fallbacks for legacy streaming GGUFs (converted before the new
    // streaming.* KVs landed): derive pre_encode_cache_size from
    // subsampling_factor+1, drop_extra_pre_encoded via NeMo's formula,
    // and sampling_frames_first defaults to subsampling_factor (which
    // collapses chunk_size_first == chunk_size_subsequent — i.e. no
    // first-chunk special case, matching pre-Phase-B behavior).
    const int subsampling_factor = pm->hparams.enc_subsampling_factor;
    int pre_encode_cache_size = pm->hparams.enc_stream_pre_encode_cache_size;
    if (pre_encode_cache_size <= 0) {
        pre_encode_cache_size = subsampling_factor + 1;
    }
    int drop_extra_subsequent = pm->hparams.enc_stream_drop_extra_pre_encoded;
    if (drop_extra_subsequent <= 0 && pre_encode_cache_size >= 1) {
        drop_extra_subsequent = 1 + (pre_encode_cache_size - 1) /
                                    std::max(subsampling_factor, 1);
    }
    int sampling_frames_first = pm->hparams.enc_stream_sampling_frames_first;
    if (sampling_frames_first <= 0) {
        sampling_frames_first = subsampling_factor;
    }

    const int chunk_size_subsequent = subsampling_factor * (1 + chosen_right);
    const int chunk_size_first = sampling_frames_first + subsampling_factor * chosen_right;

    pc->stream_caches.att_context_right     = chosen_right;
    pc->stream_caches.att_context_left      = chosen_left;
    pc->stream_caches.chunk_size_first      = chunk_size_first;
    pc->stream_caches.chunk_size_subsequent = chunk_size_subsequent;
    pc->stream_caches.mel_fed_first         = chunk_size_first;
    pc->stream_caches.mel_fed_subsequent    = chunk_size_subsequent + pre_encode_cache_size;
    pc->stream_caches.drop_extra_first      = 0;
    pc->stream_caches.drop_extra_subsequent = drop_extra_subsequent;
    pc->stream_caches.is_first_chunk        = true;
    pc->stream_caches.chunk_step            = 0;

    return TRANSCRIBE_OK;
}

transcribe_status stream_feed(
    transcribe_session *        session,
    const float *               pcm,
    int                         n_samples,
    transcribe_stream_update *  update)
{
    auto * pc = static_cast<ParakeetSession *>(session);
    auto * pm = static_cast<ParakeetModel *>(pc->model);
    if (pc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    pc->stream_pcm_buffer.insert(
        pc->stream_pcm_buffer.end(), pcm, pcm + n_samples);
    pc->stream_audio_input_us += samples_to_us(n_samples);

    const int prev_n_tokens = static_cast<int>(pc->raw_tokens.size());

    if (pm == nullptr || pm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (!pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // -------- Buffered streaming path --------
    //
    // Emit one chunk at a time while we have enough audio in
    // stream_pcm_buffer to cover the next non-final [chunk | right]
    // window. The left context is pulled from already-buffered audio
    // (zero-padded at the start of the stream).
    if (pc->buf_active) {
        // Variable-stride loop: step 0 requires (samples_chunk +
        // samples_right) of new audio; steady-state needs samples_chunk.
        // We only emit non-last chunks here; the last chunk (and its
        // ragged-tail folding) is the responsibility of stream_finalize.
        while (true) {
            if (pc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
            const int64_t num_new = pc->buf_initialized
                ? static_cast<int64_t>(pc->buf_samples_chunk)
                : static_cast<int64_t>(pc->buf_samples_chunk) +
                  static_cast<int64_t>(pc->buf_samples_right);
            const int64_t need_end = pc->buf_next_audio_read + num_new;
            if (need_end > static_cast<int64_t>(pc->stream_pcm_buffer.size())) {
                break;
            }
            if (const transcribe_status st = emit_buffered_chunk(
                    pc, pm, num_new, /*is_last_chunk=*/false);
                st != TRANSCRIBE_OK)
            {
                return st;
            }
        }
        const bool tokens_changed =
            static_cast<int>(pc->raw_tokens.size()) != prev_n_tokens;
        if (tokens_changed) {
            rebuild_streaming_result_text(pc, pm);
            pc->n_committed_tokens   = static_cast<int>(pc->tokens.size());
            pc->n_committed_words    = 0;
            pc->n_committed_segments = 0;
            pc->stream_revision     += 1;
            pc->stream_audio_committed_us =
                pc->buf_next_audio_read * 1000000LL /
                std::max<int64_t>(pm->hparams.fe_sample_rate, 1);
        }
        if (update != nullptr) {
            update->result_changed     = tokens_changed;
            update->revision           = pc->stream_revision;
            update->input_received_ms  = us_to_ms(pc->stream_audio_input_us);
            update->audio_committed_ms = us_to_ms(pc->stream_audio_committed_us);
            update->buffered_ms        = us_to_ms(
                pc->stream_audio_input_us - pc->stream_audio_committed_us);
        }
        return TRANSCRIBE_OK;
    }

    // -------- Cache-aware streaming path --------
    //
    // Recompute mel from the (sliding) accumulated PCM buffer.
    //
    // The buffer is trimmed after each emit to retain only the prefix
    // samples that future mel frames will still need (the STFT window
    // of the next-unemitted frame extends back by `n_fft/2` samples).
    // That keeps mel cost bounded per feed regardless of stream length.
    //
    // pcm_start_sample is the absolute sample index of buffer[0] and
    // is always hop-aligned, so the absolute → buffer-relative
    // mel-frame translation is the integer pcm_start_sample / hop.
    // The first few buffer-relative mel frames (where the STFT window
    // would extend left of buffer[0]) are reflect-pad-contaminated
    // garbage; they correspond to already-emitted absolute frames and
    // are never read.
    const int64_t t_mel_start = ggml_time_us();
    int mel_n_mels   = 0;
    int mel_n_frames = 0;
    if (const transcribe_status mst = pm->mel->compute(
            pc->stream_pcm_buffer.data(),
            pc->stream_pcm_buffer.size(),
            pc->mel_buf, mel_n_mels, mel_n_frames,
            pc->n_threads);
        mst != TRANSCRIBE_OK)
    {
        // For very short buffers (< 2 frames worth) compute returns
        // INVALID_ARG; that's a "not enough audio yet" condition for
        // streaming, not a fatal error. Treat as a no-op feed.
        if (mst == TRANSCRIBE_ERR_INVALID_ARG) {
            if (update != nullptr) {
                update->result_changed     = false;
                update->revision           = pc->stream_revision;
                update->input_received_ms  = us_to_ms(pc->stream_audio_input_us);
                update->audio_committed_ms = us_to_ms(pc->stream_audio_committed_us);
                update->buffered_ms        = us_to_ms(
                    pc->stream_audio_input_us - pc->stream_audio_committed_us);
            }
            return TRANSCRIBE_OK;
        }
        return mst;
    }
    pc->t_mel_us += ggml_time_us() - t_mel_start;

    // Emit as many chunks as we have new mel frames for. NeMo
    // distinguishes first vs subsequent: first chunk has chunk_size_first
    // NEW mel frames (no pre-encode-cache prepend, drop_extra=0);
    // subsequent chunks have chunk_size_subsequent NEW frames plus a
    // pre_encode_cache_size left-prepend (drop_extra=drop_extra_subsequent).
    const int pre_encode_cache_size = pm->hparams.enc_stream_pre_encode_cache_size > 0
        ? pm->hparams.enc_stream_pre_encode_cache_size
        : pm->hparams.enc_subsampling_factor + 1;
    // pcm_start_frame is the absolute mel-frame index of buffer-relative
    // frame 0. Hop-aligned trim keeps this an integer.
    const int hop = pm->hparams.fe_hop_length;
    const int64_t pcm_start_frame =
        pc->stream_caches.pcm_start_sample / hop;
    while (true) {
        if (pc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

        const int64_t consumed = pc->stream_caches.mel_frames_consumed;
        const int64_t avail    = pcm_start_frame +
                                 static_cast<int64_t>(mel_n_frames);
        const bool    is_first = pc->stream_caches.is_first_chunk;

        const int chunk_advance = is_first
            ? pc->stream_caches.chunk_size_first
            : pc->stream_caches.chunk_size_subsequent;
        const int mel_fed = is_first
            ? pc->stream_caches.mel_fed_first
            : pc->stream_caches.mel_fed_subsequent;
        const int drop_extra = is_first
            ? pc->stream_caches.drop_extra_first
            : pc->stream_caches.drop_extra_subsequent;
        const int prepend_frames = mel_fed - chunk_advance;

        // Need (consumed + chunk_advance + right_edge_margin) total
        // mel frames available before we can emit. The +margin is the
        // right-edge reflect-pad zone of the mel transform: the last
        // ceil(n_fft/2 / hop_length) frames computed at any moment
        // have their windows reflect-padded against the buffer end,
        // producing slightly wrong values until more audio arrives.
        // Without this margin, chunk N's last few mel frames would
        // get values from a "pre-mature" mel computation and never
        // match NeMo's full-audio mel reference. The cost is a few
        // mel frames of extra latency per chunk (40ms for nemotron's
        // n_fft=512/hop=160).
        //
        // stream_finalize doesn't gate on this margin because by then
        // we've received the EOF signal and the right-edge reflect-pad
        // is "real" — it's the end of the stream.
        const int right_edge_margin = (pm->hparams.fe_n_fft / 2 +
                                       pm->hparams.fe_hop_length - 1) /
                                      pm->hparams.fe_hop_length;
        if (avail - consumed < chunk_advance + right_edge_margin) break;

        std::vector<float> chunk(
            static_cast<size_t>(mel_n_mels) *
            static_cast<size_t>(mel_fed));

        // mel_buf layout: row-major [n_mels, mel_n_frames]; entry
        // for absolute frame `abs_t` at offset
        // (m * mel_n_frames + (abs_t - pcm_start_frame)).
        auto mel_at = [&](int m, int64_t abs_t) -> float {
            const int64_t rel = abs_t - pcm_start_frame;
            return pc->mel_buf[static_cast<size_t>(m) * mel_n_frames +
                               static_cast<size_t>(rel)];
        };

        for (int m = 0; m < mel_n_mels; ++m) {
            // Pre-encode-cache prepend: [consumed - prepend_frames,
            // consumed). On the first chunk prepend_frames == 0 so the
            // loop is skipped entirely. On subsequent chunks consumed
            // >= chunk_advance >= prepend_frames so the indices are
            // never negative.
            for (int h = 0; h < prepend_frames; ++h) {
                const int64_t frame_idx = consumed - prepend_frames + h;
                const float val = (frame_idx < 0)
                    ? 0.0f
                    : mel_at(m, frame_idx);
                chunk[static_cast<size_t>(m) * mel_fed + h] = val;
            }
            // New frames: [consumed, consumed + chunk_advance).
            for (int n = 0; n < chunk_advance; ++n) {
                const float val = mel_at(m, consumed + n);
                chunk[static_cast<size_t>(m) * mel_fed +
                      prepend_frames + n] = val;
            }
        }

        // Flip is_first_chunk BEFORE the emit so the dump hooks see the
        // correct step_num and drop semantics. The caller (emit) uses
        // the parameters we already extracted.
        pc->stream_caches.is_first_chunk = false;
        (void)pre_encode_cache_size; // kept for future first-chunk diagnostics

        const int64_t t_enc_start = ggml_time_us();
        if (const transcribe_status st = emit_streaming_chunk(
                pc, pm, chunk.data(),
                mel_fed,
                drop_extra,
                chunk_advance);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        pc->t_encode_us += ggml_time_us() - t_enc_start;
    }

    // Rebuild the partial transcript from committed tokens.
    const bool tokens_changed =
        static_cast<int>(pc->raw_tokens.size()) != prev_n_tokens;
    if (tokens_changed) {
        rebuild_streaming_result_text(pc, pm);
        pc->n_committed_tokens   = static_cast<int>(pc->tokens.size());
        pc->n_committed_words    = 0;
        pc->n_committed_segments = 0;
        pc->stream_revision     += 1;
    }

    // Trim already-consumed PCM. The next chunk's mel_at() reads back
    // by `prepend_max` frames (the pre-encode-cache history) AND each
    // mel-frame STFT window extends another `pad = n_fft/2` samples
    // left of its center. Keep both margins or the prepend frames'
    // windows reflect-pad against a truncated buffer start, silently
    // corrupting the pre-encode-cache input on every subsequent chunk
    // (observed as deletions at chunk boundaries in WER).
    // Round the drop down to a hop boundary to keep pcm_start_sample
    // aligned with mel-frame index 0.
    //
    // First-feed safety: with mel_frames_consumed == 0 the target
    // keep_from is negative, the drop_aligned guard rejects it, and
    // the buffer stays intact.
    {
        const int pad = pm->hparams.fe_n_fft / 2;
        const int prepend_max =
            pc->stream_caches.mel_fed_subsequent -
            pc->stream_caches.chunk_size_subsequent;
        const int64_t earliest_frame_needed =
            pc->stream_caches.mel_frames_consumed -
            static_cast<int64_t>(prepend_max);
        const int64_t earliest_sample_needed =
            earliest_frame_needed * static_cast<int64_t>(hop);
        const int64_t keep_from =
            earliest_sample_needed - static_cast<int64_t>(pad);
        if (keep_from > pc->stream_caches.pcm_start_sample) {
            const int64_t drop_raw =
                keep_from - pc->stream_caches.pcm_start_sample;
            const int64_t drop_aligned =
                (drop_raw / static_cast<int64_t>(hop)) *
                static_cast<int64_t>(hop);
            if (drop_aligned > 0 &&
                drop_aligned <=
                static_cast<int64_t>(pc->stream_pcm_buffer.size()))
            {
                pc->stream_pcm_buffer.erase(
                    pc->stream_pcm_buffer.begin(),
                    pc->stream_pcm_buffer.begin() +
                        static_cast<ptrdiff_t>(drop_aligned));
                pc->stream_caches.pcm_start_sample += drop_aligned;
            }
        }
    }

    // Audio "committed" cursor tracks the amount of audio we've fully
    // consumed through the encoder (one mel frame = hop_length /
    // sample_rate seconds). Using mel_frames_consumed (vs encoder
    // output frame count) avoids the right-context overshoot that
    // makes "encoder frames * 80 ms" exceed the actual audio time
    // by a few ms per chunk on streaming variants.
    pc->stream_audio_committed_us =
        pc->stream_caches.mel_frames_consumed *
        static_cast<int64_t>(pm->hparams.fe_hop_length) *
        1000000 /
        static_cast<int64_t>(pm->hparams.fe_sample_rate);

    if (update != nullptr) {
        update->result_changed     = tokens_changed;
        update->revision           = pc->stream_revision;
        update->input_received_ms  = us_to_ms(pc->stream_audio_input_us);
        update->audio_committed_ms = us_to_ms(pc->stream_audio_committed_us);
        update->buffered_ms        =
            us_to_ms(pc->stream_audio_input_us - pc->stream_audio_committed_us);
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_finalize(
    transcribe_session *        session,
    transcribe_stream_update *  update)
{
    auto * pc = static_cast<ParakeetSession *>(session);
    auto * pm = static_cast<ParakeetModel *>(pc->model);
    if (pm == nullptr || pm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (!pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int prev_n_tokens = static_cast<int>(pc->raw_tokens.size());

    // -------- Buffered streaming finalize --------
    //
    // Mirrors NeMo's reference loop tail exactly: one final emit that
    // consumes all remaining audio (num_new = audio_len - next_audio_read)
    // with is_last_chunk=true. add_frames_get_removed_ folds the existing
    // samples_right slot plus this final num_new into the chunk slot, so
    // the decoder gets every encoder frame past ctx_left. No zero-pad,
    // no fixed-stride trailing chunks.
    if (pc->buf_active) {
        const int64_t total =
            static_cast<int64_t>(pc->stream_pcm_buffer.size());
        if (pc->buf_next_audio_read < total) {
            if (pc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
            const int64_t num_new = total - pc->buf_next_audio_read;
            if (const transcribe_status st = emit_buffered_chunk(
                    pc, pm, num_new, /*is_last_chunk=*/true);
                st != TRANSCRIBE_OK)
            {
                return st;
            }
        }
        const bool tokens_changed =
            static_cast<int>(pc->raw_tokens.size()) != prev_n_tokens;
        if (tokens_changed || !pc->has_result) {
            rebuild_streaming_result_text(pc, pm);
        }
        pc->stream_audio_committed_us = pc->stream_audio_input_us;
        pc->n_committed_tokens        = static_cast<int>(pc->tokens.size());
        pc->n_committed_words         = static_cast<int>(pc->words.size());
        pc->n_committed_segments      = static_cast<int>(pc->segments.size());
        pc->stream_revision          += 1;
        if (update != nullptr) {
            update->result_changed     = pc->has_result;
            update->revision           = pc->stream_revision;
            update->input_received_ms  = us_to_ms(pc->stream_audio_input_us);
            update->audio_committed_ms = us_to_ms(pc->stream_audio_committed_us);
            update->buffered_ms        = 0;
        }
        return TRANSCRIBE_OK;
    }

    // -------- Cache-aware streaming finalize --------
    //
    // Recompute mel one last time and check for a sub-chunk tail.
    // If there are any unprocessed mel frames left over (i.e. the
    // stream ended mid-chunk), zero-pad to chunk_size_total and run a
    // final chunk so we don't drop the tail audio. Sub-chunk tails
    // produce a smaller post-subsample/drop output, but the encoder
    // graph handles arbitrary mel chunk sizes.
    if (!pc->stream_pcm_buffer.empty()) {
        int mel_n_mels   = 0;
        int mel_n_frames = 0;
        const transcribe_status mst = pm->mel->compute(
            pc->stream_pcm_buffer.data(),
            pc->stream_pcm_buffer.size(),
            pc->mel_buf, mel_n_mels, mel_n_frames,
            pc->n_threads);
        if (mst != TRANSCRIBE_OK && mst != TRANSCRIBE_ERR_INVALID_ARG) {
            return mst;
        }

        if (mst == TRANSCRIBE_OK) {
            const int hop = pm->hparams.fe_hop_length;
            const int64_t pcm_start_frame =
                pc->stream_caches.pcm_start_sample / hop;
            const int64_t consumed = pc->stream_caches.mel_frames_consumed;
            const int64_t avail    = pcm_start_frame +
                                     static_cast<int64_t>(mel_n_frames);
            const int64_t remaining = avail - consumed;
            if (remaining > 0) {
                const bool is_first = pc->stream_caches.is_first_chunk;
                const int drop_extra = is_first
                    ? pc->stream_caches.drop_extra_first
                    : pc->stream_caches.drop_extra_subsequent;
                const int mel_fed_full = is_first
                    ? pc->stream_caches.mel_fed_first
                    : pc->stream_caches.mel_fed_subsequent;
                const int chunk_advance_full = is_first
                    ? pc->stream_caches.chunk_size_first
                    : pc->stream_caches.chunk_size_subsequent;
                const int prepend_frames = mel_fed_full - chunk_advance_full;
                // Feed the natural partial-chunk size — no silence pad.
                // Matches NeMo's last-chunk behavior (the streaming buffer
                // gives whatever's left; conformer_stream_step happily
                // accepts a shorter chunk). The encoder graph rebuilds
                // each call so it's fine to vary the shape. The decoder
                // sees fewer enc_out frames but the trailing-audio
                // tokens (e.g. the closing punctuation) survive instead
                // of getting masked by zero-pad frames.
                const int new_take = static_cast<int>(remaining);
                const int mel_fed = prepend_frames + new_take;
                std::vector<float> chunk(
                    static_cast<size_t>(mel_n_mels) *
                    static_cast<size_t>(mel_fed),
                    0.0f);
                // mel_buf indexed by ABSOLUTE frame; translate to the
                // buffer-relative offset via pcm_start_frame.
                auto mel_at = [&](int m, int64_t abs_t) -> float {
                    const int64_t rel = abs_t - pcm_start_frame;
                    return pc->mel_buf[static_cast<size_t>(m) * mel_n_frames +
                                       static_cast<size_t>(rel)];
                };
                for (int m = 0; m < mel_n_mels; ++m) {
                    for (int h = 0; h < prepend_frames; ++h) {
                        const int64_t frame_idx = consumed - prepend_frames + h;
                        const float val = (frame_idx < 0)
                            ? 0.0f
                            : mel_at(m, frame_idx);
                        chunk[static_cast<size_t>(m) * mel_fed + h] = val;
                    }
                    for (int n = 0; n < new_take; ++n) {
                        const float val = mel_at(m, consumed + n);
                        chunk[static_cast<size_t>(m) * mel_fed +
                              prepend_frames + n] = val;
                    }
                }
                pc->stream_caches.is_first_chunk = false;
                if (const transcribe_status st = emit_streaming_chunk(
                        pc, pm, chunk.data(),
                        mel_fed,
                        drop_extra,
                        new_take);
                    st != TRANSCRIBE_OK)
                {
                    return st;
                }
            }
        }
    }

    const bool tokens_changed =
        static_cast<int>(pc->raw_tokens.size()) != prev_n_tokens;
    if (tokens_changed || !pc->has_result) {
        rebuild_streaming_result_text(pc, pm);
    }

    // Commit everything: at finalize all emitted tokens are final.
    pc->stream_audio_committed_us = pc->stream_audio_input_us;
    pc->n_committed_tokens        = static_cast<int>(pc->tokens.size());
    pc->n_committed_words         = static_cast<int>(pc->words.size());
    pc->n_committed_segments      = static_cast<int>(pc->segments.size());
    pc->stream_revision          += 1;

    if (update != nullptr) {
        update->result_changed     = pc->has_result;
        update->revision           = pc->stream_revision;
        update->input_received_ms  = us_to_ms(pc->stream_audio_input_us);
        update->audio_committed_ms = us_to_ms(pc->stream_audio_committed_us);
        update->buffered_ms        = 0;
    }
    return TRANSCRIBE_OK;
}

void stream_reset(transcribe_session * session) {
    auto * pc = static_cast<ParakeetSession *>(session);
    // Drop the buffered audio contents but keep the allocation for
    // the next stream.
    pc->stream_pcm_buffer.clear();
}

// Kind+slot probe. Parakeet ships no run-slot extensions, so the _RUN
// slot is always false. On the _STREAM slot, cache-aware variants
// (ChunkedLimited) take the PARAKEET_STREAM kind; chunked-attention
// variants (ChunkedLimitedWithRc) take the PARAKEET_BUFFERED_STREAM
// kind. Offline-only Parakeet variants accept neither.
bool accepts_ext_kind(const transcribe_model * model,
                      transcribe_ext_slot      slot,
                      uint32_t                 kind) {
    if (model == nullptr) return false;
    if (slot != TRANSCRIBE_EXT_SLOT_STREAM) return false;
    const auto * pm = static_cast<const ParakeetModel *>(model);
    switch (pm->hparams.enc_att_context_style) {
        case ParakeetHParams::AttContextStyle::ChunkedLimited:
            return kind == TRANSCRIBE_EXT_KIND_PARAKEET_STREAM;
        case ParakeetHParams::AttContextStyle::ChunkedLimitedWithRc:
            return kind == TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM;
        case ParakeetHParams::AttContextStyle::Regular:
            return false;
    }
    return false;
}

} // namespace

// `extern const` to force external linkage. Without `extern`, a
// namespace-scope `const` object has internal linkage in C++ and the
// forward declaration in transcribe-arch.cpp would not resolve.
extern const Arch arch = {
    /* .name             = */ "parakeet",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ run_batch,
    /* .stream_validate  = */ stream_validate,
    /* .stream_begin     = */ stream_begin,
    /* .stream_feed      = */ stream_feed,
    /* .stream_finalize  = */ stream_finalize,
    /* .stream_reset     = */ stream_reset,
    /* .accepts_ext_kind = */ accepts_ext_kind,
};

} // namespace transcribe::parakeet

// ---------------------------------------------------------------------------
// Public parakeet extension init functions (global scope, C linkage).
// Defined here in family source so transcribe.cpp stays family-agnostic.
// Each stamps the transcribe_ext header (size + kind) and the field
// defaults; they are the single source of truth now that the INIT macros
// are gone.
// ---------------------------------------------------------------------------

extern "C" void transcribe_parakeet_stream_ext_init(
    struct transcribe_parakeet_stream_ext * p)
{
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->ext.size          = sizeof(*p);
    p->ext.kind          = TRANSCRIBE_EXT_KIND_PARAKEET_STREAM;
    p->att_context_right = -1;  // model default (max accuracy / max latency)
}

extern "C" void transcribe_parakeet_buffered_stream_ext_init(
    struct transcribe_parakeet_buffered_stream_ext * p)
{
    if (p == nullptr) { return; }
    std::memset(p, 0, sizeof(*p));
    p->ext.size  = sizeof(*p);
    p->ext.kind  = TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM;
    p->left_ms   = -1;  // model default
    p->chunk_ms  = -1;
    p->right_ms  = -1;
}
