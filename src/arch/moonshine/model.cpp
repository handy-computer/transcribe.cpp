// arch/moonshine/model.cpp - Moonshine ASR family handler.
//
// Load / init_context / run lifecycle for the Moonshine encoder-decoder
// model. Greedy generation only: the upstream generation_config is
// `do_sample=False, num_beams=1, max_length=194`; there are no
// suppress_tokens, no language tokens, no temperature fallback.

#include "moonshine.h"

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

namespace transcribe::moonshine {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   MoonshineModel>);
static_assert(std::is_base_of_v<transcribe_context, MoonshineContext>);

MoonshineContext::~MoonshineContext() {
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

MoonshineModel::~MoonshineModel() {
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

// ---------------------------------------------------------------------------
// KV cache initialization
// ---------------------------------------------------------------------------

bool kv_cache_init(MoonshineKvCache & cache,
                   ggml_backend_t     backend,
                   int                n_ctx,
                   int                T_enc,
                   int                d_model,
                   int                n_layer,
                   ggml_type          kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
                     "moonshine kv_cache: unsupported kv_type=%d "
                     "(only F16/F32)\n", static_cast<int>(kv_type));
        return false;
    }

    const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;
    ggml_init_params params { ctx_size, nullptr, /*no_alloc=*/true };
    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "moonshine kv_cache: ggml_init failed\n");
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
        std::fprintf(stderr, "moonshine kv_cache: buffer alloc failed\n");
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

constexpr const char k_default_variant[] = "moonshine";

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

    auto m = std::make_unique<MoonshineModel>();
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

    if (auto st = m->tok.load(loader.gguf());                 st != TRANSCRIBE_OK) return st;
    if (auto st = read_moonshine_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) return st;

    // Stage 2: reopen GGUF with no_alloc to wire weight slots.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) return TRANSCRIBE_ERR_GGUF;

    if (auto st = build_moonshine_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(
            backend_req, "moonshine", m->plan); st != TRANSCRIBE_OK)
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
                     "moonshine: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "moonshine");
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
    auto cc = std::make_unique<MoonshineContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    // See MoonshineContext for why decoder FA is Metal-only-off.
    auto * cm = static_cast<MoonshineModel *>(model);
    const bool is_metal =
        (cm->plan.primary_kind == transcribe::BackendKind::Metal);
    cc->encoder_use_flash = true;
    cc->decoder_use_flash = !is_metal;
    transcribe::flash::apply_env_overrides(
        cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Set per-backend thread count from cc->n_threads (with a sensible default).
void apply_thread_policy(MoonshineContext * cc) {
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

bool ensure_compute_ctx(MoonshineContext * cc, size_t mem_size) {
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

// Search compute_ctx for a tensor by name. Used to locate the causal
// mask input tensor when the prompt pass needs one. Cohere uses the
// same trick.
ggml_tensor * find_tensor_by_name(ggml_context * ctx, const char * name) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx, t))
    {
        if (std::strcmp(t->name, name) == 0) return t;
    }
    return nullptr;
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
    auto * cc = static_cast<MoonshineContext *>(ctx);
    auto * cm = static_cast<MoonshineModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();
    cc->clear_result();

    const auto & hp = cm->hparams;

    // Phase timers. Moonshine has no mel/STFT (raw PCM passthrough), so
    // t_mel_us stays 0; t_encode_us covers encoder graph + cross-KV
    // precompute, t_decode_us covers prompt pass + step loop.
    cc->t_mel_us    = 0;
    cc->t_encode_us = 0;
    cc->t_decode_us = 0;
    const int64_t t_encode_start = ggml_time_us();

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    // ----- Encoder: build + compute -----
    if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        std::fprintf(stderr, "moonshine run: ensure_compute_ctx failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, hp, n_samples, cc->encoder_use_flash);
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
            std::fprintf(stderr, "moonshine run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "moonshine run: alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload PCM and encoder pos_ids.
    ggml_backend_tensor_set(eb.audio_in, pcm,
                            0, static_cast<size_t>(n_samples) * sizeof(float));
    {
        std::vector<int32_t> enc_pos_ids(static_cast<size_t>(T_enc));
        for (int i = 0; i < T_enc; ++i) enc_pos_ids[i] = i;
        ggml_backend_tensor_set(eb.pos_ids_in, enc_pos_ids.data(), 0,
                                enc_pos_ids.size() * sizeof(int32_t));
    }

    apply_thread_policy(cc);

    if (ggml_backend_sched_graph_compute(cc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "moonshine run: encoder compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Dump encoder intermediates.
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("enc.audio.in",      eb.dumps.audio_in,      "encoder.audio.in");
    try_dump("enc.conv1.out",     eb.dumps.conv1_out,     "encoder.conv1");
    try_dump("enc.groupnorm.out", eb.dumps.groupnorm_out, "encoder.groupnorm");
    try_dump("enc.conv2.out",     eb.dumps.conv2_out,     "encoder.conv2");
    try_dump("enc.conv3.out",     eb.dumps.conv3_out,     "encoder.conv3");
    for (size_t i = 0; i < eb.dumps.block_outs.size(); ++i) {
        char bname[64], stage[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
        std::snprintf(stage, sizeof(stage), "encoder.block%zu.out", i);
        try_dump(bname, eb.dumps.block_outs[i], stage);
    }
    try_dump("enc.final", eb.dumps.final_out, "encoder.final");

    // Read encoder output to host buffer (for re-upload into the
    // cross_kv graph's input tensor — it lives in a fresh compute_ctx).
    const int d_model = hp.dec_d_model;
    cc->enc_T = T_enc;
    cc->enc_host.assign(static_cast<size_t>(d_model) *
                        static_cast<size_t>(T_enc), 0.0f);
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ----- KV cache init -----
    {
        if (cc->kv_cache.buffer != nullptr && cc->kv_cache.T_enc != T_enc) {
            cc->kv_cache.free();
        }
        if (cc->kv_cache.buffer == nullptr) {
            const int n_ctx = hp.dec_max_position_embeddings > 0
                            ? hp.dec_max_position_embeddings : 512;
            ggml_type cache_type = resolved_kv;
            // For F32 reference dtype, default the cache to F32 too —
            // matching the reference numerical regime in Stage 4.
            if (cache_type == GGML_TYPE_COUNT) cache_type = GGML_TYPE_F32;
            if (!kv_cache_init(cc->kv_cache, cm->plan.primary,
                               n_ctx, T_enc, d_model, hp.dec_n_layers,
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

    // ----- Cross-KV precompute -----
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
                                cc->enc_host.data(), 0,
                                cc->enc_host.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(cc->sched, cross_db.graph)
            != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "moonshine run: cross_kv compute failed\n");
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
        // Build a fresh compute_ctx + step graph. With dump_prompt=true
        // the graph emits the full log_softmax tensor so the dumper can
        // capture dec.logits; otherwise we skip log_softmax (argmax is
        // invariant) and read raw logits.
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
        // Upload tokens + position ids.
        std::vector<int32_t> token_ids(static_cast<size_t>(n_tokens));
        std::vector<int32_t> pos_ids  (static_cast<size_t>(n_tokens));
        for (int i = 0; i < n_tokens; ++i) {
            token_ids[i] = (i == 0) ? token_id_first
                                    : 0 /* unused: caller batches one token */;
            pos_ids  [i] = n_past_in + i;
        }
        ggml_backend_tensor_set(db.token_ids_in, token_ids.data(),
                                0, token_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(db.pos_ids_in, pos_ids.data(),
                                0, pos_ids.size() * sizeof(int32_t));
        // Causal mask only when n_tokens > 1 (not exercised on this
        // family today — every step is single-token).
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
                         "moonshine run: decoder compute failed (n_past=%d)\n",
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
            // Reference dumper writes pre-softmax logits at gen step 20
            // under the name dec.logits_raw.gen20. The C++ step graph's
            // out tensor IS the raw logits when skip_log_softmax=true.
            try_dump(mid_gen_dump_name, db.out, "decoder.logits_raw.gen");
        }

        // Pick next_token. Fast path: argmax tensor computed on-device
        // (every step pass uses skip_log_softmax=true, so argmax_out is
        // populated). Slow path: prompt pass dumps full log_softmax;
        // argmax in C against the downloaded vocab row.
        if (db.argmax_out != nullptr) {
            int32_t argmax_id = 0;
            ggml_backend_tensor_get(db.argmax_out, &argmax_id,
                                    0, sizeof(int32_t));
            next_token = argmax_id;
        } else {
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
        }

        cc->kv_cache.n    = n_past_in + n_tokens;
        cc->kv_cache.head = cc->kv_cache.n;
        n_past            = cc->kv_cache.n;
        return TRANSCRIBE_OK;
    };

    // Prompt pass: a single decoder_start_token_id (=1, <s>).
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

    // Step loop.
    //
    // Two variants:
    //   GPU (Vulkan/Metal/CUDA/SYCL): build_step_graph — one static-
    //     topology graph for the whole utterance. KV writes via
    //     ggml_set_rows at runtime kv_idx; flash-attn reads a fixed
    //     max_n_kv window with a runtime mask. Removes per-step
    //     graph_build + sched_alloc.
    //   CPU (or debug, which needs the mid-gen dump path): per-step
    //     run_step with build_decoder_graph_kv.
    constexpr int k_mid_gen_step = 20;
    const bool primary_is_gpu =
        cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
        cm->plan.primary_kind != transcribe::BackendKind::Accel &&
        cm->plan.primary_kind != transcribe::BackendKind::Unknown;
    const bool use_step_graph = primary_is_gpu &&
                                !transcribe::debug::enabled();

    if (use_step_graph) {
        // ---------- Static-graph step path (GPU) ----------
        // max_n_kv: pad max_pos to next power of two (no fixed floor —
        // moonshine's max_pos is ~194, so a 1024 floor would waste 5×
        // attention bandwidth per step).
        int max_n_kv = 64;
        while (max_n_kv < max_pos) max_n_kv *= 2;
        if (max_n_kv > cc->kv_cache.n_ctx) max_n_kv = cc->kv_cache.n_ctx;

        if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
            std::fprintf(stderr,
                         "moonshine run: ensure_compute_ctx failed (step)\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        StepBuild sb = build_step_graph(
            cc->compute_ctx, cm->weights, hp, cc->kv_cache,
            max_n_kv, T_enc, cc->decoder_use_flash);
        if (sb.graph == nullptr || sb.argmax_out == nullptr) {
            std::fprintf(stderr,
                         "moonshine run: build_step_graph failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
            std::fprintf(stderr,
                         "moonshine run: sched_alloc_graph failed (step)\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        // Mask buffer: [0, n_past) populated by the prompt pass start
        // attendable; remaining slots -inf until each step flips its
        // newly-written position to attendable.
        const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);
        for (int p = 0; p < n_past; ++p) step_mask[p] = mask_zero;

        while (next_token != eos && n_past < max_pos) {
            if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
            if (n_past + 1 > max_n_kv) {
                std::fprintf(stderr,
                             "moonshine run: hit max_n_kv=%d at n_past=%d\n",
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

            if (ggml_backend_sched_graph_compute(cc->sched, sb.graph)
                != GGML_STATUS_SUCCESS)
            {
                std::fprintf(stderr,
                             "moonshine run: step compute failed (n_past=%d)\n",
                             n_past);
                return TRANSCRIBE_ERR_GGUF;
            }

            n_past += 1;
            cc->kv_cache.n    = n_past;
            cc->kv_cache.head = n_past;

            int32_t argmax_id = 0;
            ggml_backend_tensor_get(sb.argmax_out, &argmax_id, 0, sizeof(int32_t));
            next_token = argmax_id;

            if (next_token != eos) generated_ids.push_back(next_token);
        }
    } else {
        // ---------- Dynamic-graph step path (CPU / debug) ----------
        while (next_token != eos && n_past < max_pos) {
            if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

            // The mid-generation dump fires when the cache holds 20 tokens
            // (= prompt + 19 generated, so this step predicts the 21st
            // token = generated_ids[19]) — matching the reference dumper's
            // gen_step_n=20 contract.
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
    }

    cc->t_decode_us = ggml_time_us() - t_decode_start;

    // Decode IDs to text.
    if (!generated_ids.empty()) {
        std::string full = cm->tok.decode(generated_ids.data(),
                                          static_cast<int>(generated_ids.size()));
        // SentencePiece-style BPE leading ▁ → ' '; the decode helper
        // already replaces those, but the very first token typically
        // produces a leading space. Trim it.
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
    /* .name             = */ "moonshine",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::moonshine
