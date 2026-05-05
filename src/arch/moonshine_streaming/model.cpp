// arch/moonshine_streaming/model.cpp - Moonshine-Streaming family handler.
//
// Lifecycle:
//
//   1. load              — read GGUF, populate hparams, wire weight slots.
//   2. init_context      — allocate scheduler / context.
//   3. run               — encoder → adapter → cross_kv precompute →
//                          autoregressive decode.
//
// The adapter (pos_emb add + optional proj) runs once per session in a
// separate compute_ctx, and its output is read back to host. The
// cross_kv precompute graph re-uploads the host adapter buffer, which
// matches the pattern moonshine uses for its encoder output. This
// keeps the decoder graphs free of the in-place adapter mutation that
// HF's `decoder.forward` performs.

#include "moonshine_streaming.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::moonshine_streaming {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   MoonshineStreamingModel>);
static_assert(std::is_base_of_v<transcribe_context, MoonshineStreamingContext>);

MoonshineStreamingContext::~MoonshineStreamingContext() {
    kv_cache.free();
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

MoonshineStreamingModel::~MoonshineStreamingModel() {
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

bool kv_cache_init(MoonshineStreamingKvCache & cache,
                   ggml_backend_t              backend,
                   int                         n_ctx,
                   int                         T_enc,
                   int                         d_model,
                   int                         n_layer,
                   ggml_type                   kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
                     "moonshine_streaming kv_cache: unsupported kv_type=%d "
                     "(only F16/F32)\n", static_cast<int>(kv_type));
        return false;
    }

    const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;
    ggml_init_params params { ctx_size, nullptr, /*no_alloc=*/true };
    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "moonshine_streaming kv_cache: ggml_init failed\n");
        return false;
    }

    const int64_t self_elements  = static_cast<int64_t>(d_model) * n_layer * n_ctx;
    const int64_t cross_elements = static_cast<int64_t>(d_model) * n_layer * T_enc;

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
        std::fprintf(stderr, "moonshine_streaming kv_cache: buffer alloc failed\n");
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

    return true;
}

namespace {

constexpr const char k_default_variant[] = "moonshine-streaming";

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

    auto m = std::make_unique<MoonshineStreamingModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    m->variant = loader.variant().empty() ? k_default_variant
                                          : loader.variant();
    m->backend.clear();

    apply_family_invariants(m->caps);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) return st;
    if (auto st = read_languages_kv (loader.gguf(), *m);       st != TRANSCRIBE_OK) return st;

    if (auto st = m->tok.load(loader.gguf());                              st != TRANSCRIBE_OK) return st;
    if (auto st = read_moonshine_streaming_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) return st;

    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) return TRANSCRIBE_ERR_GGUF;

    if (auto st = build_moonshine_streaming_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(
            backend_req, "moonshine_streaming", m->plan); st != TRANSCRIBE_OK)
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
                     "moonshine_streaming: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "moonshine_streaming");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

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
    auto cc = std::make_unique<MoonshineStreamingContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    cc->encoder_use_flash = false;   // encoder uses sliding-window mask;
                                     // non-flash path is the safer default.
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(
        cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

void apply_thread_policy(MoonshineStreamingContext * cc) {
    int n_threads = cc->n_threads;
    if (n_threads <= 0) {
        n_threads = std::min(8, std::max(1, static_cast<int>(
            std::thread::hardware_concurrency())));
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(cc->sched); ++i) {
        ggml_backend_t be       = ggml_backend_sched_get_backend(cc->sched, i);
        ggml_backend_dev_t dev  = ggml_backend_get_device(be);
        ggml_backend_reg_t reg  = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) continue;
        auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (fn != nullptr) fn(be, n_threads);
    }
}

bool ensure_compute_ctx(MoonshineStreamingContext * cc, size_t mem_size) {
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    ggml_init_params init_params {};
    init_params.mem_size   = mem_size;
    init_params.mem_buffer = nullptr;
    init_params.no_alloc   = true;
    cc->compute_ctx = ggml_init(init_params);
    return cc->compute_ctx != nullptr;
}

ggml_tensor * find_tensor_by_name(ggml_context * ctx, const char * name) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx, t))
    {
        if (std::strcmp(t->name, name) == 0) return t;
    }
    return nullptr;
}

// Pad PCM up to a multiple of frame_len with zeros (matches HF's
// Wav2Vec2FeatureExtractor pad_to_multiple_of=80 behavior). The padded
// tail contributes silent CMVN frames that the encoder's first
// causal-conv stride+windowed-attention naturally folds into the
// post-stride T_enc count. We do NOT truncate: dropping samples would
// silently change the transcript.
std::vector<float> right_pad_pcm(const float * pcm, int n_samples, int frame_len) {
    if (frame_len <= 0) {
        return std::vector<float>(pcm, pcm + n_samples);
    }
    const int rem = n_samples % frame_len;
    const int pad = (rem == 0) ? 0 : (frame_len - rem);
    std::vector<float> out;
    out.resize(static_cast<size_t>(n_samples) + pad, 0.0f);
    std::memcpy(out.data(), pcm, static_cast<size_t>(n_samples) * sizeof(float));
    return out;
}

transcribe_status run(
    transcribe_context *      ctx,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    (void)params;

    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<MoonshineStreamingContext *>(ctx);
    auto * cm = static_cast<MoonshineStreamingModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();
    cc->clear_result();

    const auto & hp = cm->hparams;

    cc->t_mel_us    = 0;
    cc->t_encode_us = 0;
    cc->t_decode_us = 0;
    const int64_t t_encode_start = ggml_time_us();

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    // ----- right-pad PCM to a multiple of frame_len -----
    std::vector<float> pcm_padded = right_pad_pcm(pcm, n_samples, hp.enc_frame_len);
    const int n_samples_padded = static_cast<int>(pcm_padded.size());

    // ----- Encoder graph -----
    if (!ensure_compute_ctx(cc, 16 * 1024 * 1024)) {
        std::fprintf(stderr, "moonshine_streaming run: ensure_compute_ctx failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, hp,
        n_samples_padded, cc->encoder_use_flash);
    if (eb.audio_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    const int T_enc = eb.T_enc;

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr, "moonshine_streaming run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "moonshine_streaming run: alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload PCM.
    ggml_backend_tensor_set(eb.audio_in, pcm_padded.data(),
                            0, static_cast<size_t>(n_samples_padded) * sizeof(float));
    // Upload per-layer sliding-window masks.
    {
        std::vector<float> mask_buf(static_cast<size_t>(T_enc) * T_enc);
        for (int i = 0; i < hp.enc_n_layers; ++i) {
            const int L = hp.enc_sliding_windows[2 * i + 0];
            const int R = hp.enc_sliding_windows[2 * i + 1];
            build_sliding_window_mask(T_enc, L, R, mask_buf.data());
            ggml_backend_tensor_set(eb.per_layer_masks[i], mask_buf.data(),
                                    0, mask_buf.size() * sizeof(float));
        }
    }

    apply_thread_policy(cc);

    if (ggml_backend_sched_graph_compute(cc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "moonshine_streaming run: encoder compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("enc.audio.in",            eb.dumps.audio_in,   "encoder.audio.in");
    try_dump("enc.embedder.cmvn.out",   eb.dumps.cmvn_out,   "encoder.embedder.cmvn");
    try_dump("enc.embedder.comp.out",   eb.dumps.comp_out,   "encoder.embedder.comp");
    try_dump("enc.embedder.linear.out", eb.dumps.linear_out, "encoder.embedder.linear");
    try_dump("enc.embedder.conv1.out",  eb.dumps.conv1_out,  "encoder.embedder.conv1");
    try_dump("enc.embedder.conv2.out",  eb.dumps.conv2_out,  "encoder.embedder.conv2");
    for (size_t i = 0; i < eb.dumps.block_outs.size(); ++i) {
        char bname[64], stage[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
        std::snprintf(stage, sizeof(stage), "encoder.block%zu.out", i);
        try_dump(bname, eb.dumps.block_outs[i], stage);
    }
    try_dump("enc.final", eb.dumps.final_out, "encoder.final");

    // Read encoder hidden to host so we can re-upload to fresh compute_ctx.
    const int enc_h = hp.enc_d_model;
    const int dec_h = hp.dec_d_model;
    std::vector<float> enc_host(static_cast<size_t>(enc_h) *
                                static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, enc_host.data(), 0,
                            enc_host.size() * sizeof(float));

    // ----- Adapter graph (pos_emb + optional proj) -----
    {
        if (!ensure_compute_ctx(cc, 4 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        AdapterBuild ab = build_adapter_graph(cc->compute_ctx, cm->weights, hp, T_enc);
        if (ab.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, ab.graph)) {
            std::fprintf(stderr,
                         "moonshine_streaming run: alloc_graph failed (adapter)\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(ab.encoder_out_in, enc_host.data(), 0,
                                enc_host.size() * sizeof(float));
        std::vector<int32_t> pos_ids(static_cast<size_t>(T_enc));
        for (int i = 0; i < T_enc; ++i) pos_ids[i] = i;
        ggml_backend_tensor_set(ab.pos_ids_in, pos_ids.data(), 0,
                                pos_ids.size() * sizeof(int32_t));

        if (ggml_backend_sched_graph_compute(cc->sched, ab.graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "moonshine_streaming run: adapter compute failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        try_dump("adapter.pos_emb", ab.pos_emb_out, "adapter.pos_emb");
        try_dump("adapter.out",     ab.out,         "adapter.out");

        // Read adapted encoder hidden to host buffer.
        cc->enc_T = T_enc;
        cc->adapter_host.assign(static_cast<size_t>(dec_h) *
                                static_cast<size_t>(T_enc), 0.0f);
        ggml_backend_tensor_get(ab.out, cc->adapter_host.data(), 0,
                                cc->adapter_host.size() * sizeof(float));
    }

    // ----- KV cache init -----
    {
        if (cc->kv_cache.buffer != nullptr && cc->kv_cache.T_enc != T_enc) {
            cc->kv_cache.free();
        }
        if (cc->kv_cache.buffer == nullptr) {
            const int n_ctx = hp.dec_max_position_embeddings > 0
                            ? hp.dec_max_position_embeddings : 512;
            ggml_type cache_type = resolved_kv;
            if (cache_type == GGML_TYPE_COUNT) cache_type = GGML_TYPE_F32;
            if (!kv_cache_init(cc->kv_cache, cm->plan.primary,
                               n_ctx, T_enc, dec_h, hp.dec_n_layers,
                               cache_type))
            {
                return TRANSCRIBE_ERR_BACKEND;
            }
        } else {
            cc->kv_cache.n               = 0;
            cc->kv_cache.head            = 0;
            cc->kv_cache.cross_populated = false;
        }
    }

    // ----- Cross-KV precompute (from adapter output) -----
    {
        if (!ensure_compute_ctx(cc, 4 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        DecoderBuild cross_db = build_cross_kv_graph(
            cc->compute_ctx, cm->weights, hp, cc->kv_cache, T_enc);
        if (cross_db.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(cross_db.encoder_out_in,
                                cc->adapter_host.data(), 0,
                                cc->adapter_host.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(cc->sched, cross_db.graph)
            != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "moonshine_streaming run: cross_kv compute failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    cc->t_encode_us = ggml_time_us() - t_encode_start;

    // ----- Greedy decoder loop -----
    const int64_t t_decode_start = ggml_time_us();
    const int decoder_start = hp.decoder_start_token_id;   // 1
    const int eos           = hp.eos_token_id;             // 2
    const int max_pos       = hp.dec_max_position_embeddings;

    std::vector<int32_t> generated_ids;
    int next_token = -1;
    int n_past     = 0;

    auto run_step = [&](int n_tokens, int n_past_in,
                        int token_id_first, bool dump_prompt,
                        const char * mid_gen_dump_name)
                       -> transcribe_status {
        if (!ensure_compute_ctx(cc, 4 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        const bool skip_log_softmax = !dump_prompt;
        DecoderBuild db = build_decoder_graph_kv(
            cc->compute_ctx, cm->weights, hp, cc->kv_cache,
            n_tokens, n_past_in, T_enc,
            skip_log_softmax, cc->decoder_use_flash);
        if (db.out == nullptr || db.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        std::vector<int32_t> token_ids(static_cast<size_t>(n_tokens));
        std::vector<int32_t> pos_ids  (static_cast<size_t>(n_tokens));
        for (int i = 0; i < n_tokens; ++i) {
            token_ids[i] = (i == 0) ? token_id_first : 0;
            pos_ids  [i] = n_past_in + i;
        }
        ggml_backend_tensor_set(db.token_ids_in, token_ids.data(),
                                0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(db.pos_ids_in, pos_ids.data(),
                                0, pos_ids.size() * sizeof(int32_t));
        if (n_tokens > 1) {
            ggml_tensor * m = find_tensor_by_name(cc->compute_ctx, "dec.causal_mask");
            if (m != nullptr) {
                const int n_kv = n_past_in + n_tokens;
                std::vector<float> mask(static_cast<size_t>(n_kv) * n_tokens,
                                        -1e9f);
                for (int q = 0; q < n_tokens; ++q) {
                    for (int k = 0; k < n_kv; ++k) {
                        if (k <= q + n_past_in) {
                            mask[static_cast<size_t>(q) * n_kv + k] = 0.0f;
                        }
                    }
                }
                ggml_backend_tensor_set(m, mask.data(), 0,
                                        mask.size() * sizeof(float));
            }
        }

        if (ggml_backend_sched_graph_compute(cc->sched, db.graph)
            != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "moonshine_streaming run: decoder compute failed (n_past=%d)\n",
                         n_past_in);
            return TRANSCRIBE_ERR_GGUF;
        }

        if (dump_prompt) {
            try_dump("dec.token_emb",       db.dumps.token_emb,       "decoder.embedding");
            try_dump("dec.embed_sum",       db.dumps.embed_sum,       "decoder.embed_sum");
            for (size_t i = 0; i < db.dumps.block_outs.size(); ++i) {
                char bname[64], stage[64];
                std::snprintf(bname, sizeof(bname), "dec.block.%zu.out", i);
                std::snprintf(stage, sizeof(stage), "decoder.block%zu.out", i);
                try_dump(bname, db.dumps.block_outs[i], stage);
            }
            try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.output_before_head");
            try_dump("dec.logits_raw",      db.dumps.logits_raw,      "decoder.logits_raw");
            try_dump("dec.logits",          db.dumps.logits,          "decoder.logits");
        } else if (mid_gen_dump_name != nullptr) {
            try_dump(mid_gen_dump_name, db.out, "decoder.logits_raw.gen");
        }

        const int64_t vocab_size = db.out->ne[0];
        const int64_t last_T     = (ggml_n_dims(db.out) == 1) ? 1 : db.out->ne[1];
        std::vector<float> logits_host(static_cast<size_t>(vocab_size) *
                                       static_cast<size_t>(last_T));
        ggml_backend_tensor_get(db.out, logits_host.data(), 0,
                                logits_host.size() * sizeof(float));
        const float * last_logits = logits_host.data() +
                                    static_cast<size_t>(last_T - 1) * vocab_size;
        int best_idx = 0;
        float best_v = last_logits[0];
        for (int j = 1; j < static_cast<int>(vocab_size); ++j) {
            if (last_logits[j] > best_v) { best_v = last_logits[j]; best_idx = j; }
        }
        next_token = best_idx;

        cc->kv_cache.n    = n_past_in + n_tokens;
        cc->kv_cache.head = cc->kv_cache.n;
        n_past            = cc->kv_cache.n;
        return TRANSCRIBE_OK;
    };

    if (auto st = run_step(/*n_tokens=*/1, /*n_past=*/0,
                           /*token_id_first=*/decoder_start,
                           /*dump_prompt=*/true,
                           /*mid_gen_dump_name=*/nullptr);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (next_token != eos) {
        generated_ids.push_back(next_token);
    }

    // The reference dumper's `dec.logits_raw.gen20` is the logits that
    // PREDICT the (gen_step_n)-th decoded token. After `gen_step_n - 1`
    // emissions, the cache holds 1 (bos) + (gen_step_n - 1) tokens =
    // n_past = gen_step_n. The subsequent step graph runs with that
    // n_past and emits the dump tensor. Convention mirrors moonshine.
    constexpr int k_mid_gen_step = 20;
    while (next_token != eos && n_past < max_pos) {
        if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

        const bool is_mid_gen = (n_past == k_mid_gen_step);
        const char * dump_name = is_mid_gen ? "dec.logits_raw.gen20" : nullptr;

        if (auto st = run_step(/*n_tokens=*/1, /*n_past=*/n_past,
                               /*token_id_first=*/next_token,
                               /*dump_prompt=*/false,
                               /*mid_gen_dump_name=*/dump_name);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        if (next_token != eos) {
            generated_ids.push_back(next_token);
        }
    }

    cc->t_decode_us = ggml_time_us() - t_decode_start;

    if (!generated_ids.empty()) {
        std::string full = cm->tok.decode(generated_ids.data(),
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
    }

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name         = */ "moonshine_streaming",
    /* .load         = */ load,
    /* .init_context = */ init_context,
    /* .run          = */ run,
};

} // namespace transcribe::moonshine_streaming
