// arch/granite_nar/model.cpp - IBM Granite Speech NLE (NAR) family handler.
//
// Architecture:
//   - Conformer encoder (16 layers, hidden=1024) with self-conditioned
//     CTC bypass at layer N/2 and a BPE-CTC head (1024 -> 100353)
//   - EncoderProjectorQFormer (per-encoder-layer LN over the cat of 4
//     captured layers, layer_projector 4096->2048, learned query +
//     window_positions, 2 cross-attn+MLP layers, out_norm + out_linear)
//   - Bidirectional Granite-4 1b base LM (40 layers, GQA 16/4, single
//     forward pass — no KV cache, is_causal=False)
//
// The forward composes in one run() call as:
//   PCM -> mel -> 2-frame stack -> encoder graph (emit cat_out,
//     ctc_logits, ctc_bpe_logits) -> host CTC pool + greedy decode of
//     the BPE head -> initial hypothesis -> add_insertion_slots ->
//     text_ids -> decoder forward graph (with projector audio embeds
//     concatenated to the front of inputs_embeds) -> text_logits ->
//     argmax + collapse + drop EOS -> final transcript.

#include "granite_nar.h"

#include "decoder.h"
#include "encoder.h"
#include "projector.h"
#include "weights.h"

#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace transcribe::granite_nar {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   GraniteNarModel>);
static_assert(std::is_base_of_v<transcribe_context, GraniteNarContext>);

GraniteNarContext::~GraniteNarContext() {
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

GraniteNarModel::~GraniteNarModel() {
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        ggml_backend_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
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

constexpr const char k_default_variant[] = "granite-speech-nar";
constexpr float kBnEps = 1e-5f;

transcribe_status fuse_batch_norm(GraniteNarModel & m) {
    const size_t n_blocks = m.weights.enc_blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t inner = static_cast<int64_t>(m.hparams.enc_hidden) *
                          m.hparams.enc_conv_expansion;
    const size_t tensor_bytes = static_cast<size_t>(inner) * sizeof(float);

    const size_t ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    m.bn_fused_ctx = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) {
        std::fprintf(stderr, "granite_nar: bn_fused ggml_init failed\n");
        return TRANSCRIBE_ERR_BACKEND;
    }

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.enc_blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx,
                                                   GGML_TYPE_F32, inner);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx,
                                                   GGML_TYPE_F32, inner);
    }

    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(
        m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) {
        std::fprintf(stderr, "granite_nar: bn_fused alloc failed\n");
        return TRANSCRIBE_ERR_BACKEND;
    }

    std::vector<float> bn_w(inner), bn_b(inner), rm(inner), rv(inner);
    std::vector<float> fused_s(inner), fused_b(inner);
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.enc_blocks[i];
        ggml_backend_tensor_get(b.conv_bn_w,    bn_w.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_b,    bn_b.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_mean, rm.data(),   0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_var,  rv.data(),   0, tensor_bytes);
        for (int64_t c = 0; c < inner; ++c) {
            const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
            fused_s[c] = s;
            fused_b[c] = bn_b[c] - rm[c] * s;
        }
        ggml_backend_tensor_set(b.conv_bn_fused_scale, fused_s.data(),
                                0, tensor_bytes);
        ggml_backend_tensor_set(b.conv_bn_fused_bias,  fused_b.data(),
                                0, tensor_bytes);
    }
    return TRANSCRIBE_OK;
}

transcribe_status load(
    Loader &                         loader,
    const transcribe_model_params *  params,
    transcribe_model **              out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<GraniteNarModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(m->caps);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps);
        st != TRANSCRIBE_OK) return st;
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m);
        st != TRANSCRIBE_OK) return st;

    if (const transcribe_status st = m->tok.load(loader.gguf());
        st != TRANSCRIBE_OK) return st;

    if (const transcribe_status st = read_granite_nar_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK) return st;

    m->hparams.dec_bos_id = m->tok.bos_id();
    m->hparams.dec_eos_id = m->tok.eos_id();

    if (m->hparams.dec_eos_id < 0) {
        std::fprintf(stderr, "granite_nar: tokenizer has no eos_token_id\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->tok.n_tokens() != m->hparams.dec_vocab_size) {
        std::fprintf(stderr,
                     "granite_nar: tokenizer vocab (%d) != dec_vocab_size (%d)\n",
                     m->tok.n_tokens(), m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---------- Mel frontend ----------
    // The NLE shares the whisper-mode feature extractor with AR Granite
    // but the NAR GGUF doesn't ship pre_emphasis / f_min / f_max KVs (the
    // upstream WhisperFeatureExtractor doesn't expose them; we hardcode
    // the equivalent defaults here).
    {
        transcribe::MelConfig cfg {};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = 0.0f;
        cfg.f_min        = 0.0f;
        cfg.f_max        = static_cast<float>(m->hparams.fe_sample_rate) / 2.0f;
        cfg.pad_mode     = m->hparams.fe_pad_mode;
        cfg.window_type  = m->hparams.fe_window;
        cfg.normalize    = m->hparams.fe_normalize;

        // Frontend buffers baked by the converter.
        {
            using R = transcribe::load_common::ReadF32Result;
            const size_t fb_elems =
                static_cast<size_t>(cfg.num_mels) *
                static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto fb_rc = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.mel_filterbank", fb_elems,
                "granite_nar", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto win_rc = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.window", win_elems,
                "granite_nar", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
    }

    // ---------- Tensor catalog ----------
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) return TRANSCRIBE_ERR_GGUF;

    if (const transcribe_status st = build_granite_nar_weights(
            m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "granite_nar", m->plan);
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
        std::fprintf(stderr, "granite_nar: alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "granite_nar");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    if (const transcribe_status st = fuse_batch_norm(*m); st != TRANSCRIBE_OK) {
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
    if (model->arch != &arch) return TRANSCRIBE_ERR_INVALID_ARG;

    auto cc = std::make_unique<GraniteNarContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;
    cc->encoder_use_flash = false;
    cc->decoder_use_flash = false;  // bidirectional path doesn't use KV cache
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash,
                                           cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

namespace {

void apply_thread_count(ggml_backend_sched_t sched, int n_threads) {
    if (n_threads <= 0) {
        n_threads = std::min(8, std::max(1, static_cast<int>(
            std::thread::hardware_concurrency())));
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t be  = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) continue;
        auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (fn != nullptr) fn(be, n_threads);
    }
}

} // namespace

transcribe_status run(
    transcribe_context *      ctx_base,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    (void)params;  // NLE has no language / task knobs

    if (ctx_base == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<GraniteNarContext *>(ctx_base);
    auto * cm = static_cast<GraniteNarModel *>(cc->model);

    transcribe::debug::init();

    if (!cm->mel.has_value()) {
        std::fprintf(stderr, "granite_nar run: model has no MelFrontend\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // ---------- Mel + 2-frame stack ----------
    const int64_t t_mel_start = ggml_time_us();
    int t_enc = 0;
    if (const transcribe_status mst = compute_mel_encoder_input(
            *cm->mel, pcm, n_samples, cc->n_threads, cc->mel_buf, t_enc);
        mst != TRANSCRIBE_OK)
    {
        std::fprintf(stderr,
                     "granite_nar run: mel/stack failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    cc->t_enc = t_enc;
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    if (transcribe::debug::enabled()) {
        const long long shape[2] = { t_enc, cm->hparams.enc_input_dim };
        transcribe::debug::dump_host_f32(
            "enc.mel.in", cc->mel_buf.data(),
            static_cast<long long>(cc->mel_buf.size()),
            shape, 2, "encoder");
    }

    // ---------- Reset compute state ----------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr, "granite_nar run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ---------- Encoder graph ----------
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights,
                                          cm->hparams, t_enc,
                                          cc->encoder_use_flash);
    if (eb.graph == nullptr || eb.cat_out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            32768, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr, "granite_nar run: sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr, "granite_nar run: sched_alloc_graph (encoder) failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload encoder inputs.
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(), 0,
                            cc->mel_buf.size() * sizeof(float));
    {
        std::vector<int32_t> dists = precompute_attention_dists(
            cm->hparams.enc_context_size, cm->hparams.enc_max_pos_emb);
        ggml_backend_tensor_set(eb.attention_dists, dists.data(), 0,
                                dists.size() * sizeof(int32_t));
    }
    {
        const int ctx_size = cm->hparams.enc_context_size;
        const size_t plane = static_cast<size_t>(ctx_size) * ctx_size;
        std::vector<float> mask(plane * eb.n_blocks_local, 0.0f);
        const int rem = eb.last_block_rem;
        if (rem > 0 && rem < ctx_size) {
            std::vector<float> last = precompute_last_block_mask(ctx_size, rem);
            std::memcpy(mask.data() + plane * (eb.n_blocks_local - 1),
                        last.data(), plane * sizeof(float));
        }
        ggml_backend_tensor_set(eb.last_block_mask, mask.data(), 0,
                                mask.size() * sizeof(float));
    }
    if (eb.zero_pad != nullptr) {
        const size_t n_elems = static_cast<size_t>(eb.zero_pad->ne[0]) *
                               eb.zero_pad->ne[1];
        std::vector<float> zeros(n_elems, 0.0f);
        ggml_backend_tensor_set(eb.zero_pad, zeros.data(), 0,
                                zeros.size() * sizeof(float));
    }

    apply_thread_count(cc->sched, cc->n_threads);

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "granite_nar run: encoder compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("enc.input_linear.out", eb.dumps.input_linear_out, "encoder");
    try_dump("enc.block.0.out",      eb.dumps.block_0_out,      "encoder");
    if (eb.dumps.block_mid_pre)  try_dump(eb.dumps.block_mid_pre->name,
                                          eb.dumps.block_mid_pre,  "encoder");
    if (eb.dumps.block_mid_post) try_dump(eb.dumps.block_mid_post->name,
                                          eb.dumps.block_mid_post, "encoder");
    if (eb.dumps.block_last_out) try_dump(eb.dumps.block_last_out->name,
                                          eb.dumps.block_last_out, "encoder");
    try_dump("enc.ctc_logits", eb.dumps.ctc_logits, "encoder");

    // ---------- Read encoder outputs to host ----------
    const int64_t cat_h = eb.cat_out->ne[0];
    const int64_t cat_T = eb.cat_out->ne[1];
    cc->enc_cat_host.resize(static_cast<size_t>(cat_h) * cat_T);
    ggml_backend_tensor_get(eb.cat_out, cc->enc_cat_host.data(), 0,
                            cc->enc_cat_host.size() * sizeof(float));

    const int n_ctc_vocab = static_cast<int>(eb.ctc_logits->ne[0]);
    cc->ctc_logits_host.resize(static_cast<size_t>(n_ctc_vocab) * cat_T);
    ggml_backend_tensor_get(eb.ctc_logits, cc->ctc_logits_host.data(), 0,
                            cc->ctc_logits_host.size() * sizeof(float));

    int n_bpe_vocab = 0;
    if (eb.ctc_bpe_logits != nullptr) {
        n_bpe_vocab = static_cast<int>(eb.ctc_bpe_logits->ne[0]);
        cc->ctc_bpe_logits_host.resize(static_cast<size_t>(n_bpe_vocab) * cat_T);
        ggml_backend_tensor_get(eb.ctc_bpe_logits, cc->ctc_bpe_logits_host.data(), 0,
                                cc->ctc_bpe_logits_host.size() * sizeof(float));
    }

    std::vector<float> mid_non_blank;
    if (eb.mid_blank_probs != nullptr) {
        std::vector<float> mid_blank(t_enc);
        ggml_backend_tensor_get(eb.mid_blank_probs, mid_blank.data(), 0,
                                mid_blank.size() * sizeof(float));
        mid_non_blank.resize(t_enc);
        for (int i = 0; i < t_enc; ++i) {
            mid_non_blank[i] = 1.0f - mid_blank[i];
        }
    }

    // ggml stores ne[0]=F as the innermost (fastest-varying) axis. For
    // a tensor with ne=[F, T_enc], the host buffer layout in memory is
    //   buf[t * F + f] = element at ggml indices (f, t)
    // which is the SAME as `[T_enc, F]` numpy row-major (T outer, F
    // inner). No transpose needed: the host-side BPE pool reads
    //   ctc_bpe_logits_host[t * V + v]
    // and gets the v-th logit at frame t correctly.
    (void)n_ctc_vocab;

    // ---------- Initial BPE hypothesis ----------
    std::vector<int32_t> hyp_ids;
    if (n_bpe_vocab > 0 && !mid_non_blank.empty()) {
        compute_bpe_ctc_initial_hypothesis(
            mid_non_blank,
            cc->ctc_bpe_logits_host, n_bpe_vocab,
            t_enc, cm->hparams.enc_bpe_pool_window, /*blank_id=*/0, hyp_ids);
    }
    if (hyp_ids.empty()) {
        // No tokens — emit empty transcript and return.
        cc->full_text.clear();
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        cc->has_result  = true;
        transcribe_context::SegmentEntry seg {};
        seg.text  = cc->full_text;
        seg.t0_ms = 0;
        seg.t1_ms = static_cast<int64_t>(n_samples) * 1000
                  / static_cast<int64_t>(cm->hparams.fe_sample_rate);
        cc->segments.push_back(std::move(seg));
        return TRANSCRIBE_OK;
    }

    // ---------- Projector graph ----------
    ggml_context * proj_ctx = nullptr;
    {
        ggml_init_params ip {};
        ip.mem_size   = 16 * 1024 * 1024;
        ip.no_alloc   = true;
        proj_ctx = ggml_init(ip);
        if (proj_ctx == nullptr) {
            std::fprintf(stderr, "granite_nar run: proj_ctx init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ProjectorBuild pb = build_projector_graph(proj_ctx, cm->weights,
                                              cm->hparams, t_enc);
    if (pb.graph == nullptr || pb.out == nullptr) {
        ggml_free(proj_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        std::fprintf(stderr, "granite_nar run: sched_alloc_graph (projector) failed\n");
        ggml_free(proj_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    // The encoder output sits as [cat_h, T_enc] in ggml ne order on the
    // host (i.e., enc_cat_host is feature-major in row-major bytes).
    // ggml expects the same layout for pb.enc_in — pass directly.
    ggml_backend_tensor_set(pb.enc_in, cc->enc_cat_host.data(), 0,
                            cc->enc_cat_host.size() * sizeof(float));
    if (pb.enc_pad != nullptr) {
        const size_t pad_n = static_cast<size_t>(pb.enc_pad->ne[0]) *
                             pb.enc_pad->ne[1];
        std::vector<float> zeros(pad_n, 0.0f);
        ggml_backend_tensor_set(pb.enc_pad, zeros.data(), 0,
                                zeros.size() * sizeof(float));
    }

    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, pb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "granite_nar run: projector compute failed (%d)\n",
                     static_cast<int>(gs));
        ggml_free(proj_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    try_dump("proj.qformer.out", pb.dumps.qformer_out, "projector");
    try_dump("proj.out",         pb.dumps.proj_out,    "projector");

    // Reference projector emits nblocks*n_query audio tokens (here 37*3
    // = 111 for T_enc=550) but the LLM forward only consumes
    // `projected_length = T_enc / downsample_rate` of them (110), dropping
    // the trailing tokens that came from the zero-padded window. Mirror
    // that truncation so the audio-side n matches the reference.
    const int n_audio_full   = pb.n_audio_tokens;
    const int n_audio_used   = t_enc / cm->hparams.prj_downsample_rate;
    const int n_audio_clipped = std::min(n_audio_used, n_audio_full);
    cc->n_audio_tokens = n_audio_clipped;

    const int64_t llm_dim_runtime = pb.out->ne[0];
    cc->proj_out_host.resize(static_cast<size_t>(llm_dim_runtime) *
                              n_audio_clipped);
    ggml_backend_tensor_get(pb.out, cc->proj_out_host.data(), 0,
                            cc->proj_out_host.size() * sizeof(float));
    ggml_free(proj_ctx);

    // ---------- add_insertion_slots ----------
    std::vector<int32_t> text_ids;
    add_insertion_slots(hyp_ids, cm->hparams.dec_eos_id, text_ids);
    const int n_text = static_cast<int>(text_ids.size());

    // Divide projector output by embedding_multiplier so the post-
    // multiplier audio rows round-trip to the original values. We
    // could do this in the graph but keeping it host-side avoids one
    // extra scale op per audio token.
    const float emb_mul = cm->hparams.dec_embedding_multiplier;
    if (emb_mul > 0.0f) {
        const float inv = 1.0f / emb_mul;
        for (auto & v : cc->proj_out_host) v *= inv;
    }

    // ---------- Decoder graph ----------
    ggml_context * dec_ctx = nullptr;
    {
        ggml_init_params ip {};
        ip.mem_size   = 128 * 1024 * 1024;  // 40 layers @ bidirectional
        ip.no_alloc   = true;
        dec_ctx = ggml_init(ip);
        if (dec_ctx == nullptr) {
            std::fprintf(stderr, "granite_nar run: dec_ctx init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ForwardBuild fb = build_forward_graph(dec_ctx, cm->weights, cm->hparams,
                                          cc->n_audio_tokens, n_text);
    if (fb.graph == nullptr || fb.out == nullptr) {
        ggml_free(dec_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, fb.graph)) {
        std::fprintf(stderr, "granite_nar run: sched_alloc_graph (decoder) failed\n");
        ggml_free(dec_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(fb.audio_in, cc->proj_out_host.data(), 0,
                            cc->proj_out_host.size() * sizeof(float));
    ggml_backend_tensor_set(fb.text_ids_in, text_ids.data(), 0,
                            text_ids.size() * sizeof(int32_t));
    {
        std::vector<int32_t> positions(fb.T_total);
        for (int i = 0; i < fb.T_total; ++i) positions[i] = i;
        ggml_backend_tensor_set(fb.positions_in, positions.data(), 0,
                                positions.size() * sizeof(int32_t));
    }

    const int64_t t_dec_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, fb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "granite_nar run: decoder compute failed (%d)\n",
                     static_cast<int>(gs));
        ggml_free(dec_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_decode_us = ggml_time_us() - t_dec_start;

    try_dump("dec.flat_embeds", fb.dumps.flat_embeds, "decoder");
    try_dump("dec.text_logits", fb.dumps.text_logits, "decoder");

    // Read text-portion logits. ggml stores ne[0]=vocab as the innermost
    // axis, which means the host buffer layout is [t * vocab + v] —
    // i.e. `[n_text, vocab]` numpy row-major already. No transpose.
    const int64_t vocab = fb.out->ne[0];
    std::vector<float> logits_host(static_cast<size_t>(vocab) * n_text);
    ggml_backend_tensor_get(fb.out, logits_host.data(), 0,
                            logits_host.size() * sizeof(float));
    ggml_free(dec_ctx);

    std::vector<int32_t> final_ids;
    argmax_collapse_drop_eos(logits_host, static_cast<int>(vocab), n_text,
                             cm->hparams.dec_eos_id, final_ids);

    cc->full_text = cm->tok.decode(final_ids.data(),
                                   static_cast<int>(final_ids.size()));

    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;
    transcribe_context::SegmentEntry seg {};
    seg.text  = cc->full_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000
              / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name         = */ "granite_speech_nar",
    /* .load         = */ load,
    /* .init_context = */ init_context,
    /* .run          = */ run,
};

} // namespace transcribe::granite_nar
