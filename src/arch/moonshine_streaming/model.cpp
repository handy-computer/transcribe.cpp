// arch/moonshine_streaming/model.cpp - Moonshine-Streaming family handler.
//
// Lifecycle:
//
//   1. load              — read GGUF, populate hparams, wire weight slots.
//   2. init_context      — allocate scheduler / context.
//   3. run               — encoder → adapter → cross_kv precompute →
//                          autoregressive decode.
//
// One-shot path: the adapter (pos_emb add + optional proj) runs once
// per session in a separate compute_ctx, its output is read back to
// host, and the cross_kv precompute graph uploads that buffer to
// project K/V projections directly into the persistent kv_cache.
//
// Streaming path: encoder, adapter, and cross-KV K/V projection all
// run incrementally per stream_feed. The encoder is ergodic (no
// positional encoding on encoder self-attn), the adapter's pos_emb is
// a get_rows indexed by absolute frame, and the cross-KV K/V
// projections are per-frame linear, so each stage can run on a slice
// and the slice outputs concatenate across feeds to produce the same
// values a one-shot pass would. Per-feed work is bounded by the slice
// size; the only thing that runs at finalize is one bulk upload of
// the accumulated host K/V buffers into the kv_cache plus the AR
// decoder loop. PCM trimming bounds memory: anything older than
// (T_emitted - L_total - frontend_pad) encoder frames is no longer
// reachable by any future encoder window and is dropped from the
// per-utterance buffer.
//
// AR decoding still happens once at finalize because the model is
// not trained for partial cross-attention; partial transcripts
// mid-stream (re-decode per feed) is a Phase 4b-full follow-up.

#include "moonshine_streaming.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "transcribe/moonshine_streaming.h"

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
static_assert(std::is_base_of_v<transcribe_session, MoonshineStreamingSession>);

MoonshineStreamingSession::~MoonshineStreamingSession() {
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
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<MoonshineStreamingModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    m->variant = loader.variant().empty() ? k_default_variant
                                          : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
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
    const transcribe_session_params * params,
    transcribe_session **             out_ctx)
{
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto cc = std::make_unique<MoonshineStreamingSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    cc->encoder_use_flash = true;    // sliding-window mask is uploaded as
                                     // F32 and cast to F16 inside the graph,
                                     // which is the format flash_attn_ext
                                     // expects. Validated under tolerances.
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(
        cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

void apply_thread_policy(MoonshineStreamingSession * cc) {
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

bool ensure_compute_ctx(MoonshineStreamingSession * cc, size_t mem_size) {
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

ggml_tensor * find_tensor_by_name(ggml_context * gctx, const char * name) {
    for (ggml_tensor * t = ggml_get_first_tensor(gctx); t != nullptr;
         t = ggml_get_next_tensor(gctx, t))
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

// Ensure the per-context backend scheduler is allocated. Idempotent.
transcribe_status ensure_sched(MoonshineStreamingSession * cc,
                               MoonshineStreamingModel *   cm)
{
    if (cc->sched != nullptr) return TRANSCRIBE_OK;
    cc->sched = ggml_backend_sched_new(
        cm->plan.scheduler_list.data(), nullptr,
        static_cast<int>(cm->plan.scheduler_list.size()),
        16384, false, true);
    if (cc->sched == nullptr) {
        std::fprintf(stderr,
                     "moonshine_streaming: ggml_backend_sched_new failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

// Cumulative attention right-context across cascaded encoder layers,
// in conv-stack output frames. A layer with sliding window (L, R)
// reads up to R-1 keys ahead of each query, so the layer's output at
// frame t requires its INPUT (= the previous layer's output) up to
// frame t + (R - 1). Stacked layers compound: layer N's output at
// frame t requires the conv-stack output up to frame t + sum_i(R_i - 1).
// Returns 0 if no layers have R > 0 (strictly causal model).
int cumulative_right_context(const MoonshineStreamingHParams & hp) {
    int total = 0;
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        const int R = hp.enc_sliding_windows[2 * i + 1];
        if (R > 0) total += (R - 1);
    }
    return total;
}

// Mirror of cumulative_right_context for the left side. Used for
// sizing the streaming window so emitted frames see the same left
// context they would in a one-shot pass.
int cumulative_left_context(const MoonshineStreamingHParams & hp) {
    int total = 0;
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        const int L = hp.enc_sliding_windows[2 * i + 0];
        if (L > 1) total += (L - 1);
    }
    return total;
}

// Frontend conv-stack left-pad slack in *encoder output frames*. Each
// causal conv is left-padded by k-1 input frames; we need enough PCM
// history that the conv pad slots are filled with real samples rather
// than implicit zeros, otherwise streaming chunks downstream of the
// first one would silently differ from one-shot for the first conv
// output frame of every chunk. Frame algebra (tiny defaults, k=5, two
// stride-2 layers):
//
//   conv1 left-pad: 4 embedder frames
//   conv2 left-pad: 4 conv1-output frames → needs 8 embedder frames of
//                   stride history + 4 left-pad embedder frames = 12.
//   Total embedder history needed: 4 + 12 = 16 embedder frames
//                                 = 4 encoder output frames.
//
// One encoder frame == subsampling_factor (4) * frame_len (80) PCM samples.
constexpr int k_frontend_pad_enc_frames = 4;

// Number of PCM samples represented by one encoder output frame for the
// streaming-tiny / -small / -medium architecture: two stride-2 causal
// convs over an embedder running at `frame_len` samples/frame.
int samples_per_encoder_frame(const MoonshineStreamingHParams & hp) {
    return 4 * hp.enc_frame_len;
}

// Encoder helper: build the encoder graph for `n_samples` PCM,
// upload PCM + per-layer sliding-window masks, compute, and read the
// final-LN output into the caller-provided host vector. Updates
// cc->t_encode_us (additive). When emit_dumps is true the standard
// encoder.* dump points fire; the streaming feed path passes false so
// per-feed runs don't clobber the reference-dump artifacts that
// validate.py compares.
//
// n_samples must be > 0 and a multiple of hp.enc_frame_len.
transcribe_status encode_window_to_host(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    const float *               pcm,
    int                         n_samples,
    bool                        emit_dumps,
    std::vector<float> &        out_enc_host,
    int &                       out_T_enc)
{
    const auto & hp = cm->hparams;
    if (n_samples <= 0 || hp.enc_frame_len <= 0 ||
        n_samples % hp.enc_frame_len != 0)
    {
        std::fprintf(stderr,
                     "moonshine_streaming encode_window: invalid n_samples=%d "
                     "(frame_len=%d)\n", n_samples, hp.enc_frame_len);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t t_start = ggml_time_us();

    if (!ensure_compute_ctx(cc, 16 * 1024 * 1024)) {
        std::fprintf(stderr,
                     "moonshine_streaming encode_window: ensure_compute_ctx failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, hp,
        n_samples, cc->encoder_use_flash);
    if (eb.audio_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    const int T_enc = eb.T_enc;

    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) return st;
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "moonshine_streaming encode_window: alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(eb.audio_in, pcm,
                            0, static_cast<size_t>(n_samples) * sizeof(float));
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
        std::fprintf(stderr,
                     "moonshine_streaming encode_window: encoder compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    if (emit_dumps) {
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
            if (!dump_block_index(static_cast<int>(i), hp.enc_n_layers)) continue;
            char bname[64], stage[64];
            std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
            std::snprintf(stage, sizeof(stage), "encoder.block%zu.out", i);
            try_dump(bname, eb.dumps.block_outs[i], stage);
        }
        try_dump("enc.final", eb.dumps.final_out, "encoder.final");
    }

    const int enc_h = hp.enc_d_model;
    out_enc_host.assign(static_cast<size_t>(enc_h) *
                        static_cast<size_t>(T_enc), 0.0f);
    ggml_backend_tensor_get(eb.out, out_enc_host.data(), 0,
                            out_enc_host.size() * sizeof(float));
    out_T_enc = T_enc;

    cc->t_encode_us += ggml_time_us() - t_start;
    return TRANSCRIBE_OK;
}

// Apply the adapter (pos_emb get_rows + add, plus optional proj) to an
// encoder slice. The pos_ids supplied to the graph are absolute frame
// positions starting at `abs_frame_offset`, so the slice can sit
// anywhere along the stream. Wall-clock time accumulates into
// cc->t_decode_us (the adapter is conceptually part of the decoder
// pipeline). emit_dumps gates the `adapter.*` dump points.
//
// out_adapter is resized to [dec_d_model, n_frames] f32 row-major.
transcribe_status apply_adapter_window(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    const float *               enc_data,
    int                         n_frames,
    int                         abs_frame_offset,
    bool                        emit_dumps,
    std::vector<float> &        out_adapter)
{
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
    if (n_frames <= 0 || abs_frame_offset < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const auto & hp = cm->hparams;
    const int enc_h = hp.enc_d_model;
    const int dec_h = hp.dec_d_model;

    const int64_t t_start = ggml_time_us();

    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) return st;
    if (!ensure_compute_ctx(cc, 4 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    AdapterBuild ab = build_adapter_graph(cc->compute_ctx, cm->weights, hp, n_frames);
    if (ab.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, ab.graph)) {
        std::fprintf(stderr,
                     "moonshine_streaming adapter: alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_tensor_set(ab.encoder_out_in, enc_data, 0,
                            static_cast<size_t>(enc_h) *
                                static_cast<size_t>(n_frames) * sizeof(float));
    std::vector<int32_t> pos_ids(static_cast<size_t>(n_frames));
    for (int i = 0; i < n_frames; ++i) pos_ids[i] = abs_frame_offset + i;
    ggml_backend_tensor_set(ab.pos_ids_in, pos_ids.data(), 0,
                            pos_ids.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(cc->sched, ab.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "moonshine_streaming adapter: compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    if (emit_dumps) {
        auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
            if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
        };
        try_dump("adapter.pos_emb", ab.pos_emb_out, "adapter.pos_emb");
        try_dump("adapter.out",     ab.out,         "adapter.out");
    }

    out_adapter.assign(static_cast<size_t>(dec_h) *
                       static_cast<size_t>(n_frames), 0.0f);
    ggml_backend_tensor_get(ab.out, out_adapter.data(), 0,
                            out_adapter.size() * sizeof(float));

    cc->t_decode_us += ggml_time_us() - t_start;
    return TRANSCRIBE_OK;
}

// Project an adapter slice through every decoder layer's cross-attn
// k_proj / v_proj, leaving the per-layer K and V as host buffers
// indexed by layer. Each output buffer is appended to with this
// slice's contribution: out_per_layer_k[il].insert(...) of
// [dec_d_model, n_frames] floats. Caller is responsible for sizing
// the outer vectors to n_layers before the first call.
//
// Wall-clock accumulates into cc->t_decode_us (cross-KV projection is
// part of the decoder pipeline).
transcribe_status project_cross_kv_window(
    MoonshineStreamingSession *           cc,
    MoonshineStreamingModel *             cm,
    const float *                         adapter_data,
    int                                   n_frames,
    std::vector<std::vector<float>> &     out_per_layer_k,
    std::vector<std::vector<float>> &     out_per_layer_v)
{
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
    if (n_frames <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

    const auto & hp = cm->hparams;
    const int dec_h    = hp.dec_d_model;
    const int n_layers = hp.dec_n_layers;
    if (static_cast<int>(out_per_layer_k.size()) != n_layers ||
        static_cast<int>(out_per_layer_v.size()) != n_layers)
    {
        std::fprintf(stderr,
                     "moonshine_streaming cross_kv_proj: per-layer buffer "
                     "size mismatch (got K=%zu V=%zu, expected %d)\n",
                     out_per_layer_k.size(), out_per_layer_v.size(), n_layers);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t t_start = ggml_time_us();

    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) return st;
    if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    CrossKVProjectionBuild pb = build_cross_kv_projection_graph(
        cc->compute_ctx, cm->weights, hp, n_frames);
    if (pb.graph == nullptr) return TRANSCRIBE_ERR_GGUF;

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        std::fprintf(stderr,
                     "moonshine_streaming cross_kv_proj: alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(pb.encoder_out_in, adapter_data, 0,
                            static_cast<size_t>(dec_h) *
                                static_cast<size_t>(n_frames) * sizeof(float));

    if (ggml_backend_sched_graph_compute(cc->sched, pb.graph)
        != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "moonshine_streaming cross_kv_proj: compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    const size_t per_slice_floats =
        static_cast<size_t>(dec_h) * static_cast<size_t>(n_frames);
    std::vector<float> scratch_k(per_slice_floats);
    std::vector<float> scratch_v(per_slice_floats);
    for (int il = 0; il < n_layers; ++il) {
        ggml_backend_tensor_get(pb.per_layer_k[il], scratch_k.data(), 0,
                                per_slice_floats * sizeof(float));
        ggml_backend_tensor_get(pb.per_layer_v[il], scratch_v.data(), 0,
                                per_slice_floats * sizeof(float));
        out_per_layer_k[il].insert(out_per_layer_k[il].end(),
                                   scratch_k.begin(), scratch_k.end());
        out_per_layer_v[il].insert(out_per_layer_v[il].end(),
                                   scratch_v.begin(), scratch_v.end());
    }

    cc->t_decode_us += ggml_time_us() - t_start;
    return TRANSCRIBE_OK;
}

// Allocate (or reuse / resize) the persistent KV cache for the
// supplied T_enc. The self-attention cache uses
// dec_max_position_embeddings as its capacity; the cross-attention
// cache is sized exactly at T_enc. Marks cross_populated = false so a
// downstream commit / direct-write graph fills it.
transcribe_status ensure_kv_cache_for_T(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    int                         T_enc)
{
    if (T_enc <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

    const auto & hp = cm->hparams;

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) resolved_kv = GGML_TYPE_F32;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) resolved_kv = GGML_TYPE_F16;

    if (cc->kv_cache.buffer != nullptr && cc->kv_cache.T_enc != T_enc) {
        cc->kv_cache.free();
    }
    if (cc->kv_cache.buffer == nullptr) {
        const int n_ctx = hp.dec_max_position_embeddings > 0
                        ? hp.dec_max_position_embeddings : 512;
        ggml_type cache_type = resolved_kv;
        if (cache_type == GGML_TYPE_COUNT) cache_type = GGML_TYPE_F32;
        if (!kv_cache_init(cc->kv_cache, cm->plan.primary,
                           n_ctx, T_enc, hp.dec_d_model, hp.dec_n_layers,
                           cache_type))
        {
            return TRANSCRIBE_ERR_BACKEND;
        }
    } else {
        cc->kv_cache.n               = 0;
        cc->kv_cache.head            = 0;
        cc->kv_cache.cross_populated = false;
    }
    return TRANSCRIBE_OK;
}

// Upload per-layer host K/V buffers into the persistent kv_cache via a
// small commit graph. ggml_cpy inside the graph handles any F32→F16
// conversion the cache's storage dtype requires. Accumulates into
// cc->t_decode_us.
transcribe_status commit_cross_kv_from_host(
    MoonshineStreamingSession *               cc,
    MoonshineStreamingModel *                 cm,
    int                                       T_enc,
    const std::vector<std::vector<float>> &   per_layer_k,
    const std::vector<std::vector<float>> &   per_layer_v)
{
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
    if (T_enc <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

    const auto & hp = cm->hparams;
    const int dec_h    = hp.dec_d_model;
    const int n_layers = hp.dec_n_layers;
    const size_t expected_floats =
        static_cast<size_t>(dec_h) * static_cast<size_t>(T_enc);

    if (static_cast<int>(per_layer_k.size()) != n_layers ||
        static_cast<int>(per_layer_v.size()) != n_layers)
    {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    for (int il = 0; il < n_layers; ++il) {
        if (per_layer_k[il].size() != expected_floats ||
            per_layer_v[il].size() != expected_floats)
        {
            std::fprintf(stderr,
                         "moonshine_streaming cross_kv_commit: layer %d size "
                         "mismatch (K=%zu V=%zu, expected %zu)\n",
                         il, per_layer_k[il].size(), per_layer_v[il].size(),
                         expected_floats);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }

    const int64_t t_start = ggml_time_us();

    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) return st;
    if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    CrossKVCommitBuild cb = build_cross_kv_commit_graph(
        cc->compute_ctx, hp, cc->kv_cache, T_enc);
    if (cb.graph == nullptr) return TRANSCRIBE_ERR_GGUF;

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, cb.graph)) {
        std::fprintf(stderr,
                     "moonshine_streaming cross_kv_commit: alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    for (int il = 0; il < n_layers; ++il) {
        ggml_backend_tensor_set(cb.per_layer_k_in[il], per_layer_k[il].data(),
                                0, expected_floats * sizeof(float));
        ggml_backend_tensor_set(cb.per_layer_v_in[il], per_layer_v[il].data(),
                                0, expected_floats * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(cc->sched, cb.graph)
        != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "moonshine_streaming cross_kv_commit: compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->kv_cache.cross_populated = true;

    cc->t_decode_us += ggml_time_us() - t_start;
    return TRANSCRIBE_OK;
}

// Greedy AR decoder loop. Assumes the cross-attention slots of the
// kv_cache are already populated (cross_populated == true). Mutates
// the context result vectors and accumulates wall-clock time into
// cc->t_decode_us.
transcribe_status decode_from_kv_cache(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    int                         T_enc,
    const transcribe_run_params *   params,
    bool                        emit_dumps)
{
    (void)params;

    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
    if (T_enc <= 0) return TRANSCRIBE_ERR_INVALID_ARG;
    if (!cc->kv_cache.cross_populated) return TRANSCRIBE_ERR_INVALID_ARG;

    if (auto st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) return st;

    const auto & hp = cm->hparams;
    const int64_t t_decode_start = ggml_time_us();

    auto try_dump = [emit_dumps](const char * name, ggml_tensor * t,
                                 const char * stage) {
        if (!emit_dumps || t == nullptr) return;
        transcribe::debug::dump_tensor(name, t, stage);
    };

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
                         "moonshine_streaming decode: decoder compute failed (n_past=%d)\n",
                         n_past_in);
            return TRANSCRIBE_ERR_GGUF;
        }

        if (dump_prompt) {
            try_dump("dec.token_emb",       db.dumps.token_emb,       "decoder.embedding");
            try_dump("dec.embed_sum",       db.dumps.embed_sum,       "decoder.embed_sum");
            for (size_t i = 0; i < db.dumps.block_outs.size(); ++i) {
                if (!dump_block_index(static_cast<int>(i), hp.dec_n_layers)) continue;
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

    if (auto st = run_step(/*n_tokens=*/1, /*n_past=*/0,
                           /*token_id_first=*/decoder_start,
                           /*dump_prompt=*/emit_dumps,
                           /*mid_gen_dump_name=*/nullptr);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (next_token != eos) {
        generated_ids.push_back(next_token);
    }

    // dec.logits_raw.gen20 dumps the logits that predict the 20th
    // emitted token (n_past == 20 at that step). Matches moonshine.
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

    cc->t_decode_us += ggml_time_us() - t_decode_start;

    if (!generated_ids.empty()) {
        std::string full = cm->tok.decode(generated_ids.data(),
                                          static_cast<int>(generated_ids.size()));
        if (!full.empty() && full.front() == ' ') {
            full.erase(full.begin());
        }

        const int n_generated = static_cast<int>(generated_ids.size());

        // Per-token entries. Moonshine-Streaming doesn't carry
        // per-token timestamps or probabilities — leave them at the
        // zero sentinel. seg_index = 0 because every decode emits a
        // single segment.
        cc->tokens.reserve(cc->tokens.size() +
                           static_cast<size_t>(n_generated));
        for (int i = 0; i < n_generated; ++i) {
            transcribe_session::TokenEntry tok;
            tok.id         = generated_ids[static_cast<size_t>(i)];
            tok.p          = 0.0f;
            tok.t0_ms      = 0;
            tok.t1_ms      = 0;
            tok.seg_index  = 0;
            tok.word_index = -1;
            cc->tokens.push_back(std::move(tok));
        }

        transcribe_session::SegmentEntry seg;
        seg.t0_ms       = 0;
        seg.t1_ms       = 0;
        seg.first_token = 0;
        seg.n_tokens    = n_generated;
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

// Take a host-side [enc_d_model, T_enc] encoder buffer and run the
// full adapter + cross-KV precompute + AR decode pipeline. Used by
// the one-shot path; streaming finalize uses its own incremental
// adapter + commit path so the cross-KV K/V projections never need
// to round-trip through host adapter at finalize.
transcribe_status decode_from_committed_enc(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    const float *               enc_host_data,
    int                         T_enc,
    const transcribe_run_params *   params,
    bool                        emit_dumps)
{
    if (T_enc <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

    const auto & hp = cm->hparams;

    // 1. Adapter on the full encoder buffer (single window).
    std::vector<float> adapter_host;
    if (auto st = apply_adapter_window(
            cc, cm, enc_host_data, T_enc, /*abs_frame_offset=*/0,
            emit_dumps, adapter_host);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    cc->enc_T = T_enc;

    // 2. Allocate / resize the kv_cache for T_enc.
    if (auto st = ensure_kv_cache_for_T(cc, cm, T_enc); st != TRANSCRIBE_OK) {
        return st;
    }

    // 3. Direct-write cross-KV graph. One-shot keeps using the
    //    existing builder (project + write into cache in one graph)
    //    to avoid a host round-trip that streaming has to pay.
    {
        if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
        const int64_t t_xkv_start = ggml_time_us();
        if (!ensure_compute_ctx(cc, 4 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        DecoderBuild cross_db = build_cross_kv_graph(
            cc->compute_ctx, cm->weights, hp, cc->kv_cache, T_enc);
        if (cross_db.graph == nullptr) return TRANSCRIBE_ERR_GGUF;
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(cross_db.encoder_out_in,
                                adapter_host.data(), 0,
                                adapter_host.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(cc->sched, cross_db.graph)
            != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "moonshine_streaming decode: cross_kv compute failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
        cc->t_decode_us += ggml_time_us() - t_xkv_start;
    }

    // 4. AR decoder loop.
    return decode_from_kv_cache(cc, cm, T_enc, params, emit_dumps);
}

// Internal one-shot inference helper. Encoder over the full PCM, then
// adapter + cross-KV + decoder. Does NOT call session->clear_result();
// callers handle result-snapshot management.
//
// The streaming hooks share encode_window_to_host + apply_adapter_window
// + project_cross_kv_window with this path. They diverge at finalize:
// streaming uses the accumulated host K/V buffers + commit_cross_kv_from_host,
// whereas one-shot's decode_from_committed_enc fuses the cross-KV
// projection and cache write into a single graph for minimum
// host↔backend traffic.
transcribe_status run_one_shot_inner(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    const float *               pcm,
    int                         n_samples,
    const transcribe_run_params *   params)
{
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();

    const auto & hp = cm->hparams;

    cc->t_mel_us    = 0;
    cc->t_encode_us = 0;
    cc->t_decode_us = 0;

    std::vector<float> pcm_padded = right_pad_pcm(pcm, n_samples, hp.enc_frame_len);
    const int n_samples_padded = static_cast<int>(pcm_padded.size());

    std::vector<float> enc_host;
    int T_enc = 0;
    if (auto st = encode_window_to_host(
            cc, cm, pcm_padded.data(), n_samples_padded,
            /*emit_dumps=*/true, enc_host, T_enc);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    return decode_from_committed_enc(
        cc, cm, enc_host.data(), T_enc, params,
        /*emit_dumps=*/true);
}

// One-shot entry point. Validates session/cm, clears the previous result
// snapshot, then forwards to the shared inference helper.
transcribe_status run(
    transcribe_session *      session,
    const float *             pcm,
    int                       n_samples,
    const transcribe_run_params * params)
{
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<MoonshineStreamingSession *>(session);
    auto * cm = static_cast<MoonshineStreamingModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    cc->clear_result();
    return run_one_shot_inner(cc, cm, pcm, n_samples, params);
}

// ---------------------------------------------------------------------------
// Streaming hooks
// ---------------------------------------------------------------------------
//
// Per-feed: encode a sliding window, apply adapter on the new emit
// slice, project K/V for every decoder layer, append everything to
// the per-utterance host buffers, then trim PCM that no future
// encoder window will read. Then upload the accumulated host K/V to
// the kv_cache and re-decode from BOS over the current cross-KV so
// the caller sees a live partial transcript.
//
// At finalize: top up the tail (no R_total margin needed since audio
// is done), optionally re-decode if frames advanced since the last
// feed's decode, then mark everything committed. Finalize cost
// bounded by transcript length (just an AR decoder run).
//
// PCM trimming retains samples back to
// (T_emitted - L_total - frontend_pad) encoder frames. That window
// covers every future encoder-graph slice (the longest reach is at
// finalize, where the encoder slice may extend back by L_total +
// frontend_pad to seed the conv frontend and self-attn masks). For
// the tiny variant L_total = 90 enc frames = 1.8 s of audio.
//
// Commit semantics:
//
//   The per-feed decode runs from BOS each time over the growing
//   cross-KV. Tokens emitted on feed N may shift on feed N+1 as more
//   audio arrives (the model isn't trained for partial cross-attn).
//   We mark as committed the longest token-id prefix that's IDENTICAL
//   to the previous feed's decode — that's the substring the model
//   has now produced consistently across two cross-KV sizes and is
//   unlikely to revise. Tokens past the divergence point are
//   tentative; consumers should treat them as preview-only.
//
//   n_committed_tokens advances monotonically. n_committed_words /
//   n_committed_segments stay at 0 mid-stream (we don't track
//   per-word boundaries, and there's exactly one segment per decode).
//   At finalize, the full transcript is committed: n_committed_* are
//   set to the full counts.

constexpr int64_t k_sample_rate_hz = 16000;

int64_t samples_to_us(int64_t n_samples) {
    return (n_samples * 1000000) / k_sample_rate_hz;
}

int64_t us_to_ms(int64_t us) {
    return us / 1000;
}

// Clear ONLY the result-text fields (tokens, words, segments,
// full_text, has_result, result_kind). Leaves stream snapshot
// counters (stream_revision, n_committed_*, audio cursors, etc.)
// intact — those are managed explicitly by the streaming dispatcher
// and the feed/finalize callbacks.
//
// Used by the per-feed partial decode, which overwrites the result
// text from scratch every call: re-decoding from BOS produces a full
// transcript that replaces the previous one, but the snapshot
// counters need to keep moving forward, not reset.
void reset_result_text_only(MoonshineStreamingSession * cc) {
    cc->tokens.clear();
    cc->words.clear();
    cc->segments.clear();
    cc->full_text.clear();
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = false;
}

// Run one partial-decode cycle: allocate / reuse the kv_cache for
// the current stream_T_emitted, upload the accumulated host K/V via
// the commit graph, and run the AR decoder from BOS. Mutates the
// result text fields on cc; bumps `stream_last_decoded_T` on success
// so subsequent calls can skip when no new frames have committed.
transcribe_status decode_partial(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    const transcribe_run_params *   params,
    bool                        emit_dumps)
{
    const int T_enc = cc->stream_T_emitted;
    if (T_enc <= 0) return TRANSCRIBE_ERR_INVALID_ARG;

    reset_result_text_only(cc);

    if (auto st = ensure_kv_cache_for_T(cc, cm, T_enc); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = commit_cross_kv_from_host(
            cc, cm, T_enc,
            cc->stream_cross_k_committed,
            cc->stream_cross_v_committed);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (auto st = decode_from_kv_cache(
            cc, cm, T_enc, params, emit_dumps);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    cc->stream_last_decoded_T = T_enc;
    cc->enc_T                 = T_enc;
    return TRANSCRIBE_OK;
}

// Compute how many leading token ids of the new decode agree with the
// previous decode. The matching prefix is considered "committed" — it
// reproduced under a larger cross-KV, so it's unlikely to change
// again. Returns the prefix length in tokens.
int longest_common_prefix_in_tokens(
    const std::vector<transcribe_session::TokenEntry> & new_tokens,
    const std::vector<int32_t> &                        prev_ids)
{
    const int n_new  = static_cast<int>(new_tokens.size());
    const int n_prev = static_cast<int>(prev_ids.size());
    const int bound  = std::min(n_new, n_prev);
    int prefix = 0;
    while (prefix < bound &&
           new_tokens[static_cast<size_t>(prefix)].id ==
               prev_ids[static_cast<size_t>(prefix)])
    {
        ++prefix;
    }
    return prefix;
}

// Default decode-interval when the caller passes -1 (or omits the
// moonshine_streaming stream params block entirely). Chosen to match
// one cumulative right-context window (R_total = 12 frames × 20 ms),
// giving ~4 partial-transcript updates per second after the initial
// warmup. Compute cost at this setting is roughly 2–3× one-shot on
// the tiny variant; smaller values (e.g. 80 ms) push it to 5×+.
constexpr int k_default_min_decode_interval_ms = 240;

transcribe_status stream_begin(
    transcribe_session *              session,
    const transcribe_run_params *         run_params,
    const transcribe_stream_params *  stream_params)
{
    auto * cc = static_cast<MoonshineStreamingSession *>(session);
    auto * cm = static_cast<MoonshineStreamingModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    transcribe::debug::init();

    const auto & hp = cm->hparams;

    cc->stream_pcm_buffer.clear();
    cc->stream_pcm_start_sample       = 0;
    cc->stream_adapter_committed.clear();
    cc->stream_cross_k_committed.assign(static_cast<size_t>(hp.dec_n_layers),
                                        std::vector<float>{});
    cc->stream_cross_v_committed.assign(static_cast<size_t>(hp.dec_n_layers),
                                        std::vector<float>{});
    cc->stream_T_emitted              = 0;
    cc->stream_last_decoded_T         = 0;
    cc->stream_prev_token_ids.clear();
    cc->stream_L_total_frames         = cumulative_left_context(hp);
    cc->stream_R_total_frames         = cumulative_right_context(hp);
    cc->stream_frontend_pad_frames    = k_frontend_pad_enc_frames;
    cc->stream_samples_per_enc_frame  = samples_per_encoder_frame(hp);
    cc->stream_run_params             = *run_params;

    // Resolve the per-stream decode-throttle knob from the Moonshine-
    // Streaming family extension. Sentinel -1 selects the family default.
    // Convert milliseconds to encoder frames using the model's actual
    // frame rate (samples_per_enc_frame / sample_rate).
    int32_t min_decode_ms = k_default_min_decode_interval_ms;
    const transcribe_ext * family =
        stream_params != nullptr ? stream_params->family : nullptr;
    if (const transcribe_status st = transcribe_ext_check(
            family,
            TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM,
            sizeof(struct transcribe_moonshine_streaming_stream_ext));
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (family != nullptr) {
        const auto * mx =
            reinterpret_cast<const transcribe_moonshine_streaming_stream_ext *>(family);
        if (mx->min_decode_interval_ms < -1) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        if (mx->min_decode_interval_ms >= 0) {
            min_decode_ms = mx->min_decode_interval_ms;
        }
    }
    const int64_t spf = cc->stream_samples_per_enc_frame > 0
                          ? cc->stream_samples_per_enc_frame : 1;
    const int64_t min_decode_samples =
        static_cast<int64_t>(min_decode_ms) * k_sample_rate_hz;
    cc->stream_min_decode_frames = static_cast<int32_t>(
        (min_decode_samples + 1000LL * spf - 1) / (1000LL * spf));
    if (min_decode_ms >= 0 && cc->stream_min_decode_frames < 1) {
        cc->stream_min_decode_frames = 1;
    }

    cc->kv_cache.n               = 0;
    cc->kv_cache.head            = 0;
    cc->kv_cache.cross_populated = false;

    cc->t_mel_us    = 0;
    cc->t_encode_us = 0;
    cc->t_decode_us = 0;

    return TRANSCRIBE_OK;
}

// Run the encoder over the slice of stream_pcm_buffer covering the
// absolute encoder-frame range [win_start, win_end), then apply the
// adapter and project cross-KV K/V on the emit slice
// [emit_start, emit_end) and append the slice outputs to the
// per-utterance host buffers.
//
// All frame indices are ABSOLUTE (relative to stream origin). The
// PCM buffer is sliding — `pcm_start_sample` tracks the absolute
// sample index of buffer[0] so absolute frame indices translate to
// buffer-relative offsets via the subtraction.
transcribe_status flush_stable_frames(
    MoonshineStreamingSession * cc,
    MoonshineStreamingModel *   cm,
    int                         win_start,
    int                         win_end,
    int                         emit_start,
    int                         emit_end)
{
    if (win_start < 0 || emit_start < win_start ||
        emit_end > win_end || emit_start >= emit_end)
    {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const auto & hp = cm->hparams;
    const int spf   = cc->stream_samples_per_enc_frame;
    const int enc_h = hp.enc_d_model;
    const int dec_h = hp.dec_d_model;

    // Extend the slice leftward by frontend_pad encoder frames so the
    // conv stack's left-pad is filled with real PCM (or with implicit
    // zero-pad at stream origin, which matches one-shot).
    const int pad_left_frames = std::min(cc->stream_frontend_pad_frames, win_start);
    const int slice_start_frame = win_start - pad_left_frames;
    const int slice_end_frame   = win_end;

    const int64_t abs_pcm_start = static_cast<int64_t>(slice_start_frame) * spf;
    const int64_t abs_pcm_end   = static_cast<int64_t>(slice_end_frame)   * spf;
    if (abs_pcm_start < cc->stream_pcm_start_sample) {
        std::fprintf(stderr,
                     "moonshine_streaming flush: slice underruns trimmed "
                     "PCM (slice_start_abs=%lld, pcm_start=%lld)\n",
                     static_cast<long long>(abs_pcm_start),
                     static_cast<long long>(cc->stream_pcm_start_sample));
        return TRANSCRIBE_ERR_GGUF;
    }
    const int64_t buf_pcm_start = abs_pcm_start - cc->stream_pcm_start_sample;
    const int64_t buf_pcm_end   = abs_pcm_end   - cc->stream_pcm_start_sample;
    if (buf_pcm_end > static_cast<int64_t>(cc->stream_pcm_buffer.size())) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int n_samples = static_cast<int>(buf_pcm_end - buf_pcm_start);
    if (n_samples <= 0 || n_samples % hp.enc_frame_len != 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    std::vector<float> enc_host;
    int slice_T_enc = 0;
    if (auto st = encode_window_to_host(
            cc, cm,
            cc->stream_pcm_buffer.data() + buf_pcm_start,
            n_samples,
            /*emit_dumps=*/false,
            enc_host, slice_T_enc);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    const int expected_T = slice_end_frame - slice_start_frame;
    if (slice_T_enc != expected_T) {
        std::fprintf(stderr,
                     "moonshine_streaming flush: slice T_enc mismatch "
                     "(got %d, expected %d)\n", slice_T_enc, expected_T);
        return TRANSCRIBE_ERR_GGUF;
    }

    // Extract emit-slice rows from the encoder output, then run the
    // adapter on the slice and project K/V.
    const int rel_emit_start = emit_start - slice_start_frame;
    const int rel_emit_end   = emit_end   - slice_start_frame;
    const int n_emit         = rel_emit_end - rel_emit_start;

    const size_t emit_floats =
        static_cast<size_t>(enc_h) * static_cast<size_t>(n_emit);
    const size_t emit_off =
        static_cast<size_t>(enc_h) * static_cast<size_t>(rel_emit_start);
    std::vector<float> enc_emit_slice(emit_floats);
    std::memcpy(enc_emit_slice.data(), enc_host.data() + emit_off,
                emit_floats * sizeof(float));

    std::vector<float> adapter_slice;
    if (auto st = apply_adapter_window(
            cc, cm, enc_emit_slice.data(), n_emit,
            /*abs_frame_offset=*/emit_start,
            /*emit_dumps=*/false, adapter_slice);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (static_cast<int>(adapter_slice.size()) != dec_h * n_emit) {
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->stream_adapter_committed.insert(
        cc->stream_adapter_committed.end(),
        adapter_slice.begin(), adapter_slice.end());

    if (auto st = project_cross_kv_window(
            cc, cm, adapter_slice.data(), n_emit,
            cc->stream_cross_k_committed,
            cc->stream_cross_v_committed);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    cc->stream_T_emitted = emit_end;
    return TRANSCRIBE_OK;
}

// Drop PCM samples no future encoder window will read. The leftmost
// frame the next stream_feed (or stream_finalize) can possibly need
// is (T_emitted - L_total - frontend_pad); samples before that are
// unreachable. Quantize the trim to samples_per_enc_frame so
// pcm_start_sample stays frame-aligned and absolute↔buffer-relative
// translation is integer.
void trim_pcm_buffer(MoonshineStreamingSession * cc) {
    const int spf = cc->stream_samples_per_enc_frame;
    const int64_t keep_from_frame = std::max<int64_t>(
        0,
        static_cast<int64_t>(cc->stream_T_emitted) -
            static_cast<int64_t>(cc->stream_L_total_frames) -
            static_cast<int64_t>(cc->stream_frontend_pad_frames));
    const int64_t keep_from_sample = keep_from_frame * spf;
    if (keep_from_sample <= cc->stream_pcm_start_sample) return;

    const int64_t drop = keep_from_sample - cc->stream_pcm_start_sample;
    if (drop <= 0 ||
        drop >= static_cast<int64_t>(cc->stream_pcm_buffer.size()))
    {
        return;
    }
    cc->stream_pcm_buffer.erase(
        cc->stream_pcm_buffer.begin(),
        cc->stream_pcm_buffer.begin() + drop);
    cc->stream_pcm_start_sample = keep_from_sample;
}

transcribe_status stream_feed(
    transcribe_session *        session,
    const float *               pcm,
    int                         n_samples,
    transcribe_stream_update *  update)
{
    auto * cc = static_cast<MoonshineStreamingSession *>(session);
    auto * cm = static_cast<MoonshineStreamingModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    cc->stream_pcm_buffer.insert(
        cc->stream_pcm_buffer.end(), pcm, pcm + n_samples);
    cc->stream_audio_input_us += samples_to_us(n_samples);

    const int spf  = cc->stream_samples_per_enc_frame;
    // total_pcm and available_frames are computed in ABSOLUTE frames
    // (the buffer holds samples [pcm_start_sample, ...)).
    const int64_t total_pcm_abs =
        cc->stream_pcm_start_sample +
        static_cast<int64_t>(cc->stream_pcm_buffer.size());
    const int available_frames = static_cast<int>(total_pcm_abs / spf);
    const int R = cc->stream_R_total_frames;
    const int L = cc->stream_L_total_frames;

    const int stable_frames = available_frames - R;
    bool result_changed = false;
    if (stable_frames > cc->stream_T_emitted) {
        const int emit_start = cc->stream_T_emitted;
        const int emit_end   = stable_frames;
        const int win_start  = std::max(0, emit_start - L);
        const int win_end    = emit_end + R;
        if (auto st = flush_stable_frames(cc, cm, win_start, win_end,
                                          emit_start, emit_end);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        cc->stream_audio_committed_us =
            static_cast<int64_t>(cc->stream_T_emitted) *
            static_cast<int64_t>(spf) * 1000000LL /
            k_sample_rate_hz;

        trim_pcm_buffer(cc);

        // Per-feed partial decode: re-run the AR decoder from BOS
        // over the now-extended cross-KV. Skipped when:
        //   - no new encoder frames have committed since the prior
        //     decode (nothing to update); OR
        //   - fewer than stream_min_decode_frames have committed
        //     since the prior decode (caller asked for a throttle).
        //
        // Frames not yet decoded by this feed are still extended in
        // the host adapter + cross-KV buffers; they get picked up by
        // the next feed that crosses the throttle, or by the
        // stream_finalize decode regardless.
        const int frames_since_last_decode =
            cc->stream_T_emitted - cc->stream_last_decoded_T;
        const bool throttle_allows_decode =
            frames_since_last_decode >= cc->stream_min_decode_frames;
        if (cc->stream_T_emitted > cc->stream_last_decoded_T &&
            throttle_allows_decode)
        {
            // Snapshot prior text so we can detect "decode produced
            // the same transcript again" and avoid bumping revision
            // unnecessarily. Cheap — full_text is short.
            const std::string prev_full_text = cc->full_text;

            if (auto st = decode_partial(
                    cc, cm, &cc->stream_run_params,
                    /*emit_dumps=*/false);
                st != TRANSCRIBE_OK)
            {
                return st;
            }

            // Commit prefix: tokens that re-appeared identically in
            // this decode vs the previous one are stable enough to
            // mark as committed. Tokens past the divergence point
            // remain tentative.
            const int commit_prefix = longest_common_prefix_in_tokens(
                cc->tokens, cc->stream_prev_token_ids);
            if (commit_prefix > cc->n_committed_tokens) {
                cc->n_committed_tokens = commit_prefix;
            }

            // Cache this decode's ids for the next feed's prefix
            // comparison. n_committed_words / n_committed_segments
            // stay at 0 mid-stream — we don't model per-word
            // boundaries, and there's exactly one segment per
            // decode; both are committed at finalize.
            cc->stream_prev_token_ids.clear();
            cc->stream_prev_token_ids.reserve(cc->tokens.size());
            for (const auto & tok : cc->tokens) {
                cc->stream_prev_token_ids.push_back(tok.id);
            }

            if (cc->full_text != prev_full_text) {
                cc->stream_revision += 1;
                result_changed = true;
            }
        }
    }

    if (update != nullptr) {
        update->result_changed     = result_changed;
        update->revision           = cc->stream_revision;
        update->input_received_ms  = us_to_ms(cc->stream_audio_input_us);
        update->audio_committed_ms = us_to_ms(cc->stream_audio_committed_us);
        update->buffered_ms        =
            us_to_ms(cc->stream_audio_input_us - cc->stream_audio_committed_us);
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_finalize(
    transcribe_session *        session,
    transcribe_stream_update *  update)
{
    auto * cc = static_cast<MoonshineStreamingSession *>(session);
    auto * cm = static_cast<MoonshineStreamingModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const auto & hp = cm->hparams;

    // Empty stream: produce an empty result. Mirrors transcribe_run on
    // zero-sample input.
    if (cc->stream_pcm_buffer.empty() && cc->stream_pcm_start_sample == 0) {
        cc->stream_audio_committed_us = cc->stream_audio_input_us;
        cc->stream_revision += 1;
        if (update != nullptr) {
            update->result_changed     = false;
            update->revision           = cc->stream_revision;
            update->input_received_ms  = us_to_ms(cc->stream_audio_input_us);
            update->audio_committed_ms = us_to_ms(cc->stream_audio_committed_us);
            update->buffered_ms        = 0;
        }
        return TRANSCRIBE_OK;
    }

    // Right-pad the trailing buffer to a multiple of enc_frame_len so
    // the encoder reshape divides evenly (matches the one-shot
    // right_pad_pcm). pcm_start_sample is already frame-aligned, so a
    // local pad here keeps the buffer-relative offsets clean.
    const int orig_pcm_size = static_cast<int>(cc->stream_pcm_buffer.size());
    {
        const int rem = orig_pcm_size % hp.enc_frame_len;
        const int pad = (rem == 0) ? 0 : (hp.enc_frame_len - rem);
        if (pad > 0) {
            cc->stream_pcm_buffer.resize(orig_pcm_size + pad, 0.0f);
        }
    }

    const int spf = cc->stream_samples_per_enc_frame;
    const int64_t total_pcm_abs =
        cc->stream_pcm_start_sample +
        static_cast<int64_t>(cc->stream_pcm_buffer.size());
    const int T_total = static_cast<int>(total_pcm_abs / spf);

    auto write_update = [&](transcribe_status st) {
        if (update == nullptr) return;
        update->result_changed     = cc->has_result;
        update->revision           = cc->stream_revision;
        update->input_received_ms  = us_to_ms(cc->stream_audio_input_us);
        update->audio_committed_ms = us_to_ms(cc->stream_audio_committed_us);
        update->buffered_ms        = (st == TRANSCRIBE_OK)
            ? 0
            : us_to_ms(cc->stream_audio_input_us -
                       cc->stream_audio_committed_us);
    };

    if (T_total > cc->stream_T_emitted) {
        const int emit_start = cc->stream_T_emitted;
        const int emit_end   = T_total;
        const int win_start  = std::max(0, emit_start - cc->stream_L_total_frames);
        const int win_end    = emit_end;
        if (auto st = flush_stable_frames(cc, cm, win_start, win_end,
                                          emit_start, emit_end);
            st != TRANSCRIBE_OK)
        {
            write_update(st);
            return st;
        }
    }

    const int T_enc = cc->stream_T_emitted;
    if (T_enc <= 0) {
        cc->stream_audio_committed_us = cc->stream_audio_input_us;
        cc->stream_revision += 1;
        if (update != nullptr) {
            update->result_changed     = false;
            update->revision           = cc->stream_revision;
            update->input_received_ms  = us_to_ms(cc->stream_audio_input_us);
            update->audio_committed_ms = us_to_ms(cc->stream_audio_committed_us);
            update->buffered_ms        = 0;
        }
        return TRANSCRIBE_OK;
    }

    // Re-decode if frames advanced since the most recent per-feed
    // partial decode (typically true when there's tail audio that
    // was waiting for right-context). When no new frames have
    // arrived, the last feed already produced the final transcript
    // and we just commit it as-is.
    const std::string prev_full_text = cc->full_text;
    if (T_enc > cc->stream_last_decoded_T || !cc->has_result) {
        if (auto st = decode_partial(
                cc, cm, &cc->stream_run_params,
                /*emit_dumps=*/true);
            st != TRANSCRIBE_OK)
        {
            write_update(st);
            return st;
        }
    }

    // Commit the entire result at finalize: tokens, words, and
    // segments all become committed. The previous-token-ids cache
    // is now consumed and can be discarded.
    cc->stream_audio_committed_us = cc->stream_audio_input_us;
    cc->n_committed_tokens        = static_cast<int>(cc->tokens.size());
    cc->n_committed_words         = static_cast<int>(cc->words.size());
    cc->n_committed_segments      = static_cast<int>(cc->segments.size());
    cc->stream_prev_token_ids.clear();
    if (cc->full_text != prev_full_text) {
        cc->stream_revision += 1;
    } else {
        // Finalize is a state transition; bump revision even if text
        // matches so callers can distinguish the FINAL snapshot from
        // the previous tentative one (committed counts moved even if
        // text didn't).
        cc->stream_revision += 1;
    }

    if (update != nullptr) {
        update->result_changed     = cc->has_result;
        update->revision           = cc->stream_revision;
        update->input_received_ms  = us_to_ms(cc->stream_audio_input_us);
        update->audio_committed_ms = us_to_ms(cc->stream_audio_committed_us);
        update->buffered_ms        = 0;
    }
    return TRANSCRIBE_OK;
}

void stream_reset(transcribe_session * session) {
    auto * cc = static_cast<MoonshineStreamingSession *>(session);
    cc->stream_pcm_buffer.clear();
    cc->stream_pcm_start_sample = 0;
    cc->stream_adapter_committed.clear();
    for (auto & v : cc->stream_cross_k_committed) v.clear();
    for (auto & v : cc->stream_cross_v_committed) v.clear();
    cc->stream_T_emitted      = 0;
    cc->stream_last_decoded_T = 0;
    cc->stream_prev_token_ids.clear();
}

bool accepts_ext_kind(const transcribe_model * model, uint32_t kind) {
    (void) model;
    return kind == TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM;
}

} // namespace

extern const Arch arch = {
    /* .name             = */ "moonshine_streaming",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .stream_begin     = */ stream_begin,
    /* .stream_feed      = */ stream_feed,
    /* .stream_finalize  = */ stream_finalize,
    /* .stream_reset     = */ stream_reset,
    /* .accepts_ext_kind = */ accepts_ext_kind,
};

} // namespace transcribe::moonshine_streaming
