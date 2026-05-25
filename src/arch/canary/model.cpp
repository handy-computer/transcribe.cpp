// arch/canary/model.cpp - Canary multitask AED family handler.
//
// Load / init_context / run lifecycle for the four canary variants
// (180m-flash, 1b-flash, 1b-v2, 1b). Encoder is FastConformer in the
// parakeet shape, but unlike parakeet every linear (Q/K/V/out, both
// macaron FFs, attention-pos projection, conv pointwise pair) carries
// a bias term — see weights.cpp for the full set. Decoder is an
// autoregressive Transformer (cohere-shape, but with canary-specific
// tensor names and an UNTIED LM head). The multitask prompt is a
// 4-slot or 5-slot positional sequence read from the GGUF's
// stt.canary.special.* / stt.canary.tokenizer.prompt_format KV.

#include "canary.h"

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
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::canary {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   CanaryModel>);
static_assert(std::is_base_of_v<transcribe_context, CanaryContext>);

CanaryContext::~CanaryContext() {
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

bool kv_cache_init(CanaryKvCache & cache,
                   ggml_backend_t  backend,
                   int             n_ctx,
                   int             T_enc,
                   int             n_state,
                   int             n_layer,
                   ggml_type       kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr, "canary kv_cache: unsupported kv_type=%d\n",
                     static_cast<int>(kv_type));
        return false;
    }

    const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;

    ggml_init_params params {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "canary kv_cache: ggml_init failed\n");
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
        std::fprintf(stderr, "canary kv_cache: buffer alloc failed\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }

    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx           = n_ctx;
    cache.T_enc           = T_enc;
    cache.n               = 0;
    cache.head            = 0;
    cache.cross_populated = false;

    const size_t total_bytes =
        ggml_nbytes(cache.self_k) + ggml_nbytes(cache.self_v) +
        ggml_nbytes(cache.cross_k) + ggml_nbytes(cache.cross_v);
    std::fprintf(stderr,
                 "canary kv_cache: allocated %.1f MB (%s) "
                 "(self: %d ctx x %d layers, cross: %d T_enc x %d layers)\n",
                 static_cast<double>(total_bytes) / (1024.0 * 1024.0),
                 ggml_type_name(kv_type),
                 n_ctx, n_layer, T_enc, n_layer);

    return true;
}

CanaryModel::~CanaryModel() {
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

// Fuse inference-time BatchNorm into precomputed scale + bias. Same as
// parakeet/cohere — see those files for the math.
transcribe_status fuse_batch_norm(CanaryModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t d = m.hparams.enc_d_model;
    const size_t  tensor_bytes = static_cast<size_t>(d) * sizeof(float);

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

// On CPU primary backend, dequantize 1x1 conformer pointwise convs
// from F16 to F32. Same rationale as parakeet/cohere.
transcribe_status promote_conv_pw_to_f32_on_cpu(CanaryModel & m) {
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
        m.plan, slots, "canary",
        &m.conv_pw_f32_ctx, &m.conv_pw_f32_buffer);
}

constexpr const char k_default_variant[] = "canary";

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
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<CanaryModel>();
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

    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = read_canary_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Cross-check tokenizer and KV-declared vocab sizes match.
    {
        const int tok_vocab = m->tok.n_tokens();
        if (tok_vocab > 0 && tok_vocab != m->hparams.dec_vocab_size) {
            std::fprintf(stderr,
                         "canary: tokenizer vocab (%d) != stt.canary.decoder.vocab_size (%d)\n",
                         tok_vocab, m->hparams.dec_vocab_size);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();
    if (m->hparams.eos_token_id < 0) {
        // Fall back to canary's named special: <|endoftext|>.
        m->hparams.eos_token_id = m->hparams.endoftext_id;
    }
    if (m->hparams.eos_token_id < 0) {
        std::fprintf(stderr,
                     "canary: GGUF tokenizer has no eos_token_id and no "
                     "stt.canary.special.endoftext_id — regenerate GGUF\n");
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
        // window_type defaults to "hann_symmetric" (NeMo behavior); the
        // GGUF stt.frontend.window value is currently always "hann"
        // and is treated as the family of Hann window only — the
        // periodic/symmetric distinction is not carried in the GGUF
        // today and the reference uses the symmetric variant.

        {
            using R = load_common::ReadF32Result;

            const size_t fb_elems = static_cast<size_t>(cfg.num_mels)
                                  * static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto fb_rc = load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.mel_filterbank", fb_elems,
                "canary", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }

            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto win_rc = load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.window", win_elems,
                "canary", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
    }

    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st =
            build_canary_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;

    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "canary", m->plan);
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
        std::fprintf(stderr, "canary: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "canary");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    gguf_free(gguf_data);

    if (const transcribe_status st = fuse_batch_norm(*m); st != TRANSCRIBE_OK) {
        return st;
    }
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

    auto cc = std::make_unique<CanaryContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    auto * cm = static_cast<CanaryModel *>(model);
    const bool is_metal =
        (cm->plan.primary_kind == transcribe::BackendKind::Metal);

    cc->encoder_use_flash = !is_metal;
    cc->decoder_use_flash = true;

    transcribe::flash::apply_env_overrides(
        cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Build the multitask prompt token sequence.
//
// canary  (4-slot, used by canary-1b):
//   <|startoftranscript|> <|src_lang|> <|task|> <|target_lang|> <|nopnc|or|pnc|>
//
// canary2 (5-slot, used by canary-1b-flash, canary-1b-v2, canary-180m-flash):
//   <|startofcontext|> <|startoftranscript|> <|src_lang|> <|target_lang|>
//   <|task|> <|pnc|or|nopnc|> <|notimestamp|or|timestamp|>
//
// The actual prompt format used by NeMo is determined by the model's
// CanaryTokenizer / CanaryBPETokenizer implementation. For v1 we
// hardcode pnc=yes and timestamps=no (the first port focuses on
// transcription correctness; pnc/timestamp toggles land in a follow-up).
//
// Each slot's value is selected from the GGUF's stt.canary.special.*_id
// catalog at load time. Returns the assembled prompt or an empty
// vector + error message if any required slot is missing for the
// detected variant.
std::vector<int32_t> build_prompt_canary(const CanaryHParams & hp,
                                         int                   src_lang_id,
                                         int                   tgt_lang_id,
                                         const char *          task,
                                         bool                  pnc)
{
    // canary-1 5-slot prompt template (from
    // nemo.collections.common.prompts.canary.CanaryPromptFormatter):
    //   <|startoftranscript|> <|source_lang|> <|task|> <|target_lang|> <|pnc|>
    // where <|task|> is one of <|transcribe|> / <|translate|> (canary-1
    // ships explicit task tokens, unlike canary2 which infers from
    // src/tgt). We pick <|translate|> when src != tgt (and translate_id
    // is available), else <|transcribe|>.
    std::vector<int32_t> ids;
    ids.reserve(8);

    if (hp.startoftranscript_id < 0 || src_lang_id < 0 || tgt_lang_id < 0) {
        return {};
    }
    ids.push_back(hp.startoftranscript_id);
    ids.push_back(src_lang_id);

    const bool is_translate =
        (task != nullptr && std::strcmp(task, "translate") == 0) ||
        (src_lang_id != tgt_lang_id);
    int task_id = -1;
    if (is_translate && hp.translate_id >= 0) {
        task_id = hp.translate_id;
    } else if (hp.transcribe_id >= 0) {
        task_id = hp.transcribe_id;
    } else {
        return {};
    }
    ids.push_back(task_id);
    ids.push_back(tgt_lang_id);

    const int pnc_slot = pnc ? hp.pnc_id : hp.nopnc_id;
    if (pnc_slot < 0) return {};
    ids.push_back(pnc_slot);
    return ids;
}

std::vector<int32_t> build_prompt_canary2(const CanaryModel &   cm,
                                          const CanaryHParams & hp,
                                          int                   src_lang_id,
                                          int                   tgt_lang_id,
                                          const char *          /*task*/,
                                          bool                  pnc)
{
    // canary2 prompt template (from nemo.collections.common.prompts.canary2):
    //   <|startofcontext|> [decodercontext] <|startoftranscript|>
    //   <|emo:?|> <|src_lang|> <|tgt_lang|> <|pnc|> <|itn|> <|timestamp|> <|diarize|>
    //
    // For ASR with empty decoder context the realized sequence is 9
    // tokens. itn / timestamp / diarize stay hardwired to their off
    // tokens — only pnc is user-toggleable (see transcribe_params::pnc).
    std::vector<int32_t> ids;
    ids.reserve(9);

    if (hp.startofcontext_id < 0 || hp.startoftranscript_id < 0 ||
        src_lang_id < 0 || tgt_lang_id < 0)
    {
        return {};
    }

    // Emotion is not in the named-special KV catalog; look it up via
    // the tokenizer (every canary2 SP vocab carries <|emo:undefined|>).
    const int emo_id = cm.tok.find("<|emo:undefined|>");
    if (emo_id < 0) return {};

    // Single-SP canary2 (canary-1b-v2): NeMo's prompt formatter renders
    // the empty `decodercontext` slot as one SP whitespace marker (`▁`).
    // Aggregate-tokenizer variants drop the empty slot entirely.
    if (hp.tokenizer_single_sp) {
        const int sp_space_id = cm.tok.find("\xe2\x96\x81");  // U+2581 LOWER ONE EIGHTH BLOCK = ▁
        if (sp_space_id < 0) return {};
        ids.push_back(sp_space_id);
    }

    ids.push_back(hp.startofcontext_id);
    ids.push_back(hp.startoftranscript_id);
    ids.push_back(emo_id);
    ids.push_back(src_lang_id);
    ids.push_back(tgt_lang_id);

    const int pnc_slot = pnc ? hp.pnc_id : hp.nopnc_id;
    if (pnc_slot < 0) return {};
    ids.push_back(pnc_slot);

    if (hp.noitn_id < 0) return {};
    ids.push_back(hp.noitn_id);

    if (hp.notimestamp_id < 0) return {};
    ids.push_back(hp.notimestamp_id);

    if (hp.nodiarize_id < 0) return {};
    ids.push_back(hp.nodiarize_id);

    return ids;
}

int find_language_id(const CanaryHParams & hp, const char * lang) {
    if (lang == nullptr) return -1;
    for (size_t i = 0; i < hp.languages.size() && i < hp.language_ids.size(); ++i) {
        if (hp.languages[i] == lang) return hp.language_ids[i];
    }
    return -1;
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

    auto * cc = static_cast<CanaryContext *>(ctx);
    auto * cm = static_cast<CanaryModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();

    // ----- Mel front-end -------------------------------------------
    if (!cm->mel.has_value()) {
        std::fprintf(stderr, "canary run: model has no MelFrontend\n");
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
        std::fprintf(stderr, "canary run: MelFrontend::compute failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // ----- Reset compute state --------------------------------------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    cc->encoder_out = nullptr;

    // ----- Build encoder graph --------------------------------------
    {
        ggml_init_params init_params {};
        init_params.mem_size   = 8 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx = ggml_init(init_params);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr, "canary run: ggml_init for compute_ctx failed\n");
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

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr, "canary run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr, "canary run: alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(),
                            0, cc->mel_buf.size() * sizeof(float));
    transcribe::debug::dump_tensor("enc.mel.in", eb.mel_in, "encoder.mel");

    // Sinusoidal relative positional embedding.
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
        transcribe::debug::dump_tensor("enc.pos_emb", eb.pos_emb_in, "encoder.pos_emb");
    }

    // Thread count.
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
            if (fn != nullptr) fn(be, n_threads);
        }
    }

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "canary run: encoder compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };

    try_dump("enc.pre_encode.out", eb.dumps.pre_encode_out, "encoder.pre_encode");
    try_dump("enc.block.0.out",    eb.dumps.block0_out,     "encoder.block0.out");
    {
        char bname[64], stage[64];
        const int mid = cm->hparams.enc_n_layers / 2;
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", mid);
        std::snprintf(stage, sizeof(stage), "encoder.block%d.out", mid);
        try_dump(bname, eb.dumps.block_mid_out, stage);
    }
    {
        char bname[64], stage[64];
        const int last = cm->hparams.enc_n_layers - 1;
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", last);
        std::snprintf(stage, sizeof(stage), "encoder.block%d.out", last);
        try_dump(bname, eb.dumps.block_last_out, stage);
    }
    // Native (pre-projection) and final (post-projection) encoder outs.
    try_dump("enc.native", eb.dumps.native_out, "encoder.native_out");
    try_dump("enc.final",  eb.dumps.final_out,  "encoder.final");

    cc->encoder_out = eb.out;

    // Pull encoder output to host so the cross-attn graph can consume it.
    const int d_dec = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_dec <= 0 || T_enc <= 0) {
        std::fprintf(stderr,
                     "canary run: encoder output has degenerate shape [%d, %d]\n",
                     d_dec, T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->enc_host.resize(static_cast<size_t>(d_dec) *
                        static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ----- Build multitask prompt -----------------------------------
    const char * lang = (params && params->language) ? params->language : "en";
    const bool is_translate =
        (params != nullptr && params->task == TRANSCRIBE_TASK_TRANSLATE);
    const char * tgt_lang = lang;
    if (is_translate && params && params->target_language) {
        tgt_lang = params->target_language;
    }
    const char * task = is_translate ? "translate" : "asr";

    const int src_id = find_language_id(cm->hparams, lang);
    const int tgt_id = find_language_id(cm->hparams, tgt_lang);
    if (src_id < 0) {
        std::fprintf(stderr,
                     "canary run: source language '%s' has no token id "
                     "(model supports %zu langs)\n",
                     lang, cm->hparams.languages.size());
        return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
    }
    if (tgt_id < 0) {
        std::fprintf(stderr,
                     "canary run: target language '%s' has no token id "
                     "(model supports %zu langs)\n",
                     tgt_lang, cm->hparams.languages.size());
        return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
    }

    // Generic transcribe_params::pnc routes here. DEFAULT maps to the
    // model's shipped behavior (pnc=on; matches the upstream model card's
    // published WER numbers). OFF / ON override explicitly. Non-DEFAULT
    // requests have already passed the dispatcher's advisory-warn gate
    // (which only warns when supports_pnc == false; canary advertises
    // supports_pnc = true so no warn fires here).
    bool pnc = true;
    if (params != nullptr) {
        switch (params->pnc) {
            case TRANSCRIBE_PNC_MODE_DEFAULT: pnc = true;  break;
            case TRANSCRIBE_PNC_MODE_OFF:     pnc = false; break;
            case TRANSCRIBE_PNC_MODE_ON:      pnc = true;  break;
        }
    }

    std::vector<int32_t> prompt_ids;
    if (cm->hparams.prompt_format == "canary2") {
        prompt_ids = build_prompt_canary2(*cm, cm->hparams, src_id, tgt_id,
                                          task, pnc);
    } else if (cm->hparams.prompt_format == "canary") {
        prompt_ids = build_prompt_canary(cm->hparams, src_id, tgt_id,
                                         task, pnc);
    }
    if (prompt_ids.empty()) {
        std::fprintf(stderr,
                     "canary run: failed to build prompt for format='%s'\n",
                     cm->hparams.prompt_format.c_str());
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int prompt_len = static_cast<int>(prompt_ids.size());

    // ----- Init KV cache --------------------------------------------
    {
        if (cc->kv_cache.buffer != nullptr && cc->kv_cache.T_enc != T_enc) {
            cc->kv_cache.free();
        }
        if (cc->kv_cache.buffer == nullptr) {
            const int n_ctx = cm->hparams.dec_max_position > 0
                            ? cm->hparams.dec_max_position : 1024;
            ggml_type cache_type = resolved_kv;
            if (cache_type == GGML_TYPE_COUNT) cache_type = GGML_TYPE_F16;
            if (!kv_cache_init(cc->kv_cache, cm->plan.primary,
                               n_ctx, T_enc,
                               cm->hparams.dec_d_model,
                               cm->hparams.dec_n_layers,
                               cache_type))
            {
                std::fprintf(stderr, "canary run: KV cache init failed\n");
                return TRANSCRIBE_ERR_BACKEND;
            }
        } else {
            cc->kv_cache.n    = 0;
            cc->kv_cache.head = 0;
            cc->kv_cache.cross_populated = false;
        }
    }

    auto new_compute_ctx = [&](size_t mem_size) -> bool {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        cc->encoder_out = nullptr;
        ggml_init_params ip {};
        ip.mem_size   = mem_size;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        return cc->compute_ctx != nullptr;
    };

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

    // ----- Cross-attention KV pre-compute ---------------------------
    const int64_t t_dec_start = ggml_time_us();
    {
        if (!new_compute_ctx(4 * 1024 * 1024)) {
            std::fprintf(stderr, "canary run: ggml_init for cross_kv failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        DecoderBuild cross_db = build_cross_kv_graph(
            cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, T_enc);
        if (cross_db.graph == nullptr) {
            std::fprintf(stderr, "canary run: build_cross_kv_graph failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
            std::fprintf(stderr, "canary run: alloc_graph failed (cross_kv)\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_tensor_set(cross_db.encoder_out_in,
                                cc->enc_host.data(), 0,
                                cc->enc_host.size() * sizeof(float));

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, cross_db.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "canary run: cross_kv compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    // ----- Prompt pass + autoregressive decode ---------------------
    {
        if (!new_compute_ctx(4 * 1024 * 1024)) {
            std::fprintf(stderr, "canary run: ggml_init for decoder prompt failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        // When debug-dumping, we need full softmax'd logits for the
        // pre-head dump pipeline (matches cohere's pattern). Otherwise
        // skip log_softmax and read a GPU argmax over the last position.
        const bool prompt_skip_softmax = !transcribe::debug::enabled();
        DecoderBuild db = build_decoder_graph_kv(
            cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
            prompt_len, /*n_past=*/0, T_enc,
            /*skip_log_softmax=*/prompt_skip_softmax,
            cc->decoder_use_flash);
        if (db.out == nullptr || db.graph == nullptr) {
            std::fprintf(stderr, "canary run: build_decoder_graph_kv (prompt) failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
            std::fprintf(stderr, "canary run: alloc_graph failed (decoder prompt)\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(),
                                0, prompt_ids.size() * sizeof(int32_t));
        std::vector<int32_t> pos_ids(prompt_len);
        for (int i = 0; i < prompt_len; ++i) pos_ids[i] = i;
        ggml_backend_tensor_set(db.pos_ids_in, pos_ids.data(),
                                0, pos_ids.size() * sizeof(int32_t));

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
            std::fprintf(stderr, "canary run: decoder prompt compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }

        // Per-sublayer dumps at layers {0, n_layers/2, n_layers-1}.
        for (int i = 0; i < cm->hparams.dec_n_layers; ++i) {
            const bool dump_this =
                (i == 0) || (i == cm->hparams.dec_n_layers / 2) ||
                (i == cm->hparams.dec_n_layers - 1);
            if (!dump_this || i >= 64) continue;
            char bname[64], stage[64];
            std::snprintf(bname, sizeof(bname), "dec.layer.%d.self_attn.out", i);
            std::snprintf(stage, sizeof(stage), "decoder.layer%d.self_attn", i);
            try_dump(bname, db.dumps.sub_self[i], stage);
            std::snprintf(bname, sizeof(bname), "dec.layer.%d.cross_attn.out", i);
            std::snprintf(stage, sizeof(stage), "decoder.layer%d.cross_attn", i);
            try_dump(bname, db.dumps.sub_cross[i], stage);
            std::snprintf(bname, sizeof(bname), "dec.layer.%d.ffn.out", i);
            std::snprintf(stage, sizeof(stage), "decoder.layer%d.ffn", i);
            try_dump(bname, db.dumps.sub_ffn[i], stage);
        }

        cc->kv_cache.n    = prompt_len;
        cc->kv_cache.head = prompt_len;

        cc->clear_result();

        const int eos_id = cm->hparams.eos_token_id;
        const int max_tokens = std::min(512, cc->kv_cache.n_ctx - prompt_len);

        int next_token = 0;
        if (prompt_skip_softmax && db.argmax_out != nullptr) {
            int32_t argmax_id = 0;
            ggml_backend_tensor_get(db.argmax_out, &argmax_id, 0, sizeof(int32_t));
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
        if (next_token != eos_id) generated_ids.push_back(next_token);

        int n_past = prompt_len;

        // Commit accumulated generated_ids as the run's segment + full
        // text. Called both on normal loop exit and on abort so the
        // public contract (partial result on TRANSCRIBE_ERR_ABORTED)
        // holds.
        auto commit_result = [&]() {
            cc->t_decode_us = ggml_time_us() - t_dec_start;

            if (generated_ids.empty()) return;

            const transcribe::Tokenizer & tok = cm->tok;

            // strip_special_tags (default true): canary's vocab carries
            // language/task/PNC control tokens at CONTROL type. They
            // shouldn't appear after the prompt is consumed, but the
            // strip is defensive — and only applied when the caller
            // wants clean text. --raw-tokens / strip_special_tags=false
            // exposes whatever the decoder emitted.
            const bool strip =
                (params == nullptr) ? true : params->strip_special_tags;
            std::vector<int> text_ids;
            if (strip) {
                text_ids.reserve(generated_ids.size());
                for (int id : generated_ids) {
                    if (tok.is_control(id)) continue;
                    text_ids.push_back(id);
                }
            } else {
                text_ids.assign(generated_ids.begin(), generated_ids.end());
            }

            std::string full = tok.decode(text_ids.data(),
                                          static_cast<int>(text_ids.size()));
            if (!full.empty() && full.front() == ' ') full.erase(full.begin());

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

        // Two decoder loop variants; pick by primary backend kind.
        //
        //   GPU (Vulkan/Metal/CUDA/SYCL): build_step_graph — one static-
        //     topology graph for the whole utterance. KV writes go via
        //     ggml_set_rows at runtime kv_idx; flash-attn reads a fixed
        //     max_n_kv window with a runtime mask. Removes per-step
        //     graph_build + sched_alloc, which dominate dispatch overhead
        //     on GPUs (~2 ms/tok on Vulkan-Renoir for canary-1b).
        //
        //   CPU (incl. accelerator host-memory backends): build_decoder_
        //     graph_kv per step — n_kv grows with n_past, attention only
        //     reads the populated prefix. CPU has no dispatch overhead
        //     to amortize, so the static-graph bandwidth tax (reading
        //     max_n_kv KV slots when avg n_past is ~25) is a net loss
        //     on the deep-decoder variants.
        //
        // The prompt pass uses build_decoder_graph_kv on both paths
        // (already executed above) — only the per-token loop branches.
        const bool primary_is_gpu =
            cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
            cm->plan.primary_kind != transcribe::BackendKind::Accel &&
            cm->plan.primary_kind != transcribe::BackendKind::Unknown;

        if (primary_is_gpu) {
            // ---------- Static-graph step path (GPU) ----------
            // max_n_kv: pad to next power of two with a 1024 floor.
            // Vulkan/Metal flash-attn dispatches faster on pow2 ne[1]
            // (~30% on M4 Max per qwen3_asr); the slight bandwidth cost
            // is amortized by removing per-step graph_build + sched_alloc.
            int max_n_kv = 1024;
            while (max_n_kv < prompt_len + max_tokens) max_n_kv *= 2;
            if (max_n_kv > cc->kv_cache.n_ctx) max_n_kv = cc->kv_cache.n_ctx;

            if (!new_compute_ctx(8 * 1024 * 1024)) {
                std::fprintf(stderr, "canary run: new_compute_ctx failed (step)\n");
                commit_result();
                return TRANSCRIBE_ERR_GGUF;
            }
            StepBuild sb = build_step_graph(
                cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                max_n_kv, T_enc, cc->decoder_use_flash);
            if (sb.graph == nullptr || sb.argmax_out == nullptr) {
                std::fprintf(stderr, "canary run: build_step_graph failed\n");
                commit_result();
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
                std::fprintf(stderr,
                             "canary run: sched_alloc_graph failed (step)\n");
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

            for (int step = 1; step < max_tokens && next_token != eos_id; ++step) {
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }
                if (n_past + 1 > max_n_kv) {
                    std::fprintf(stderr,
                                 "canary run: hit max_n_kv=%d at n_past=%d\n",
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
                    break;
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
            // Reserve the worst-case step graph so per-step alloc_graph
            // doesn't grow the scheduler's memory pool.
            if (new_compute_ctx(4 * 1024 * 1024)) {
                const int worst_n_past = cc->kv_cache.n_ctx - 1;
                DecoderBuild db_reserve = build_decoder_graph_kv(
                    cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                    /*n_tokens=*/1, worst_n_past, T_enc,
                    /*skip_log_softmax=*/true,
                    cc->decoder_use_flash);
                if (db_reserve.graph != nullptr) {
                    ggml_backend_sched_reserve(cc->sched, db_reserve.graph);
                }
            }

            for (int step = 1; step < max_tokens && next_token != eos_id; ++step) {
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }
                if (n_past + 1 > cc->kv_cache.n_ctx) {
                    std::fprintf(stderr,
                                 "canary run: KV cache full at n_past=%d\n",
                                 n_past);
                    break;
                }

                if (!new_compute_ctx(4 * 1024 * 1024)) break;

                DecoderBuild db_step = build_decoder_graph_kv(
                    cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                    /*n_tokens=*/1, n_past, T_enc,
                    /*skip_log_softmax=*/true,
                    cc->decoder_use_flash);
                if (db_step.out == nullptr || db_step.graph == nullptr) break;

                ggml_backend_sched_reset(cc->sched);
                if (!ggml_backend_sched_alloc_graph(cc->sched, db_step.graph)) break;

                int32_t token_id = next_token;
                int32_t pos_id   = n_past;
                ggml_backend_tensor_set(db_step.token_ids_in, &token_id, 0, sizeof(int32_t));
                ggml_backend_tensor_set(db_step.pos_ids_in,   &pos_id,   0, sizeof(int32_t));

                if (const ggml_status gs =
                        ggml_backend_sched_graph_compute(cc->sched, db_step.graph);
                    gs != GGML_STATUS_SUCCESS)
                {
                    break;
                }

                n_past += 1;
                cc->kv_cache.n    = n_past;
                cc->kv_cache.head = n_past;

                int32_t argmax_id = 0;
                ggml_backend_tensor_get(db_step.argmax_out, &argmax_id, 0, sizeof(int32_t));
                next_token = argmax_id;

                if (next_token != eos_id) generated_ids.push_back(next_token);
            }
        }

        commit_result();
    }

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name             = */ "canary",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::canary
