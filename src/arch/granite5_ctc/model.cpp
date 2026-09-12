// arch/granite5_ctc/model.cpp - Granite Speech 5.0 TurboCTC family handler.
//
// load() reads hparams, validates the tensor catalog, streams weights
// into a backend buffer, pre-fuses BatchNorm and builds the mel
// frontend. run() drives frontend -> encoder graph -> host CTC greedy
// decode. There is no decoder graph and no KV cache.

#include "decoder.h"
#include "encoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "granite5_ctc.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace transcribe::granite5_ctc {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, Granite5CtcModel>);
static_assert(std::is_base_of_v<transcribe_session, Granite5CtcSession>);

Granite5CtcSession::~Granite5CtcSession() = default;

Granite5CtcModel::~Granite5CtcModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        safe_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
    }
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

namespace {

constexpr const char k_default_variant[] = "granite-speech-5.0-470m-turboctc";

// nn.BatchNorm1d default eps.
constexpr float kBnEps = 1e-5f;

// Input-length contract (see docs/input-limits.md). granite5_ctc is an
// UNBOUNDED family: block-local attention costs O(T * context_size), not
// O(T^2), and there is no decoder context to exhaust — memory is linear
// in audio length and no published cap exists. So nothing is rejected
// and nothing is truncated; `n_ctx` has no meaning here and is ignored.
// The only floor is the frontend's: a clip must survive the two
// subsampling blocks with at least one frame left.

// Pre-fuse BatchNorm per encoder block:
//   scale = gamma / sqrt(var + eps)
//   bias  = beta - mean * scale
// Mirrors parakeet / granite. The reference keeps `conv.norm` in fp32
// (`_keep_in_fp32_modules_strict`), and the converter stores all four BN
// tensors F32, so the fusion happens in exactly the reference's dtype.
transcribe_status fuse_batch_norm(Granite5CtcModel & m) {
    const size_t n_blocks = m.weights.enc_blocks.size();
    if (n_blocks == 0) {
        return TRANSCRIBE_OK;
    }

    const int64_t inner        = static_cast<int64_t>(m.hparams.enc_hidden) * m.hparams.enc_conv_expansion;
    const size_t  tensor_bytes = static_cast<size_t>(inner) * sizeof(float);

    const size_t     ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params   = { ctx_size, nullptr, /*no_alloc=*/true };
    m.bn_fused_ctx            = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: BatchNorm-fusion context allocation failed — out of memory.");
        return TRANSCRIBE_ERR_OOM;
    }

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b              = m.weights.enc_blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, inner);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, inner);
    }

    // CPU (the scheduler list's fallback): tiny, model-lifetime buffers
    // that would only take GPU memory away from activations.
    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: BatchNorm-fusion buffer allocation failed — out of memory.");
        return TRANSCRIBE_ERR_OOM;
    }

    std::vector<float> bn_w(inner), bn_b(inner), rm(inner), rv(inner);
    std::vector<float> fused_s(inner), fused_b(inner);
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.enc_blocks[i];
        ggml_backend_tensor_get(b.conv_bn_w, bn_w.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_b, bn_b.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_mean, rm.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_var, rv.data(), 0, tensor_bytes);
        for (int64_t c = 0; c < inner; ++c) {
            const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
            fused_s[c]    = s;
            fused_b[c]    = bn_b[c] - rm[c] * s;
        }
        ggml_backend_tensor_set(b.conv_bn_fused_scale, fused_s.data(), 0, tensor_bytes);
        ggml_backend_tensor_set(b.conv_bn_fused_bias, fused_b.data(), 0, tensor_bytes);
    }

    return TRANSCRIBE_OK;
}

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<Granite5CtcModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_granite5_ctc_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    // Mel frontend. The converter bakes the torchaudio htk filterbank and
    // the periodic Hann window into the GGUF, so the C++ frontend runs on
    // bit-identical buffers rather than recomputed ones.
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = 0.0f;
        cfg.f_min        = 0.0f;
        cfg.f_max        = static_cast<float>(m->hparams.fe_sample_rate) / 2.0f;
        cfg.pad_mode     = m->hparams.fe_pad_mode;    // "reflect"
        cfg.window_type  = m->hparams.fe_window;      // "hann_periodic"
        cfg.normalize    = m->hparams.fe_normalize;   // "per_utterance"

        using R               = transcribe::load_common::ReadF32Result;
        const size_t fb_elems = static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(cfg.n_fft / 2 + 1);
        if (transcribe::load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.mel_filterbank",
                                                             fb_elems, "granite5_ctc", cfg.filterbank) != R::Ok) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (transcribe::load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.window",
                                                             static_cast<size_t>(cfg.win_length), "granite5_ctc",
                                                             cfg.window) != R::Ok) {
            return TRANSCRIBE_ERR_GGUF;
        }
        m->mel.emplace(cfg);
    }

    // Reopen for tensor metadata.
    gguf_init_params init_params{};
    init_params.no_alloc     = true;
    init_params.ctx          = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (auto st = build_granite5_ctc_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(backend_req, (params != nullptr) ? params->device : nullptr,
                                                         "granite5_ctc", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "granite5_ctc");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    if (auto st = fuse_batch_norm(*m); st != TRANSCRIBE_OK) {
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

    auto cc       = std::make_unique<Granite5CtcSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    // Recorded for transcribe_session_get_limits only: there is no KV
    // cache in this family, so the value never reaches a graph.
    cc->kv_type   = params->kv_type;

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Run the encoder graph for `n` utterances (n == 1 is the single-shot
// path) whose stacked frontend features are already packed into
// `cc->feats_buf` as [input_dim, T_max, n], and leave the CTC logits in
// `eb.ctc_logits`. `real_lens` carries each utterance's valid stacked
// frame count; every entry equal to T_max builds the mask-free graph.
transcribe_status run_encoder(Granite5CtcSession *     cc,
                              Granite5CtcModel *       cm,
                              int                      T_max,
                              const std::vector<int> & real_lens,
                              EncoderBuild &           out_eb) {
    const auto & hp = cm->hparams;

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip{};
        ip.mem_size     = 16 * 1024 * 1024;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "granite5_ctc: compute context allocation failed — out of memory. "
                    "Split long audio into shorter segments, or use a smaller batch.");
            return TRANSCRIBE_ERR_OOM;
        }
    }

    out_eb = build_encoder_graph(cc->compute_ctx, cm->weights, hp, T_max, real_lens);
    if (out_eb.graph == nullptr || out_eb.out == nullptr || out_eb.ctc_logits == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()),
                                           /*graph_size=*/16384, /*parallel=*/false, /*op_offload=*/true);
        if (cc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "granite5_ctc: scheduler allocation failed — out of memory. "
                    "Split long audio into shorter segments, or use a smaller batch.");
            return TRANSCRIBE_ERR_OOM;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, out_eb.graph)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "granite5_ctc: encoder graph allocation failed — out of memory. "
                "Split long audio into shorter segments, or use a smaller batch.");
        return TRANSCRIBE_ERR_OOM;
    }

    ggml_backend_tensor_set(out_eb.feats_in, cc->feats_buf.data(), 0, cc->feats_buf.size() * sizeof(float));
    {
        const std::vector<int32_t> rows = precompute_pos_rows(hp.enc_context_size, hp.enc_max_pos_emb);
        ggml_backend_tensor_set(out_eb.pos_rows, rows.data(), 0, rows.size() * sizeof(int32_t));
    }
    for (size_t si = 0; si < out_eb.stages.size(); ++si) {
        const AttnStage & stage = out_eb.stages[si];
        std::vector<int>  lens(static_cast<size_t>(out_eb.n_batch), stage.t_len);
        if (!real_lens.empty()) {
            // Recover this stage's per-utterance lengths by replaying the
            // halving down to the stage's own t_len.
            lens = real_lens;
            int t = T_max;
            while (t > stage.t_len) {
                for (int & v : lens) {
                    v /= 2;
                }
                t /= 2;
            }
        }
        const std::vector<float> mask = precompute_pad_mask(hp.enc_context_size, stage.t_len, lens);
        ggml_backend_tensor_set(stage.pad_mask, mask.data(), 0, mask.size() * sizeof(float));
        if (stage.zero_pad != nullptr) {
            const std::vector<float> zeros(ggml_nelements(stage.zero_pad), 0.0f);
            ggml_backend_tensor_set(stage.zero_pad, zeros.data(), 0, zeros.size() * sizeof(float));
        }
        if (stage.frame_mask != nullptr) {
            const std::vector<float> fm = precompute_frame_mask(stage.t_len, lens);
            ggml_backend_tensor_set(stage.frame_mask, fm.data(), 0, fm.size() * sizeof(float));
        }
    }

    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);

    const int64_t t_enc_start = ggml_time_us();
    if (ggml_backend_sched_graph_compute(cc->sched, out_eb.graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: graph_compute failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // Dumps, driven by the builder's explicit (name, tensor) list.
    if (transcribe::debug::enabled()) {
        for (const auto & entry : out_eb.dump_list) {
            transcribe::debug::dump_tensor(entry.first.c_str(), entry.second, "encoder");
        }
    }
    return TRANSCRIBE_OK;
}

// Host CTC greedy decode of one utterance's logits slice, publishing into
// the session's scratch result slot.
// `utt_index` tags the per-utterance logits dump for the batch parity
// gate: -1 (single-shot) writes `dec.ctc_logits`, b >= 0 writes
// `dec.ctc_logits.b{b}`. The slice is cut to the utterance's own valid
// frame count, so a batched dump is shape-comparable with its
// single-shot counterpart.
void decode_and_populate(Granite5CtcSession * cc,
                         Granite5CtcModel *   cm,
                         const float *        logits,
                         int                  t_valid,
                         int                  vocab,
                         int64_t              clip_ms,
                         int                  utt_index) {
    if (transcribe::debug::enabled()) {
        std::string name = "dec.ctc_logits";
        if (utt_index >= 0) {
            name += ".b" + std::to_string(utt_index);
        }
        const long long shape[2] = { t_valid, vocab };
        transcribe::debug::dump_host_f32(name.c_str(), logits, static_cast<long long>(t_valid) * vocab, shape, 2,
                                         "decoder.ctc_logits");
    }
    const int64_t         t_dec_start = ggml_time_us();
    std::vector<CtcToken> toks;
    ctc_greedy_collapse(logits, t_valid, vocab, cm->hparams.blank_id, toks);
    build_result(*cc, cm->tok, cm->hparams, toks, clip_ms);
    cc->t_decode_us = ggml_time_us() - t_dec_start;
}

int64_t clip_ms_for(const Granite5CtcHParams & hp, int n_samples) {
    return (hp.fe_sample_rate > 0) ? (static_cast<int64_t>(n_samples) * 1000 / hp.fe_sample_rate) : 0;
}

transcribe_status encode_and_decode(Granite5CtcSession * cc,
                                    Granite5CtcModel *   cm,
                                    const float *        pcm,
                                    int                  n_samples) {
    const auto & hp = cm->hparams;

    const int64_t t_mel_start = ggml_time_us();
    int           t_enc       = 0;
    if (auto st = compute_encoder_input(*cm->mel, hp, pcm, n_samples, cc->n_threads, cc->feats_buf, t_enc);
        st != TRANSCRIBE_OK) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "granite5_ctc run: frontend failed on %d samples (%.2f s) — the clip is "
                "too short to produce an encoder frame.",
                n_samples, static_cast<double>(n_samples) / std::max(1, hp.fe_sample_rate));
        return st;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    EncoderBuild eb{};
    if (auto st = run_encoder(cc, cm, t_enc, /*real_lens=*/{}, eb); st != TRANSCRIBE_OK) {
        return st;
    }

    const int t_out = static_cast<int>(eb.ctc_logits->ne[1]);
    const int vocab = static_cast<int>(eb.ctc_logits->ne[0]);
    cc->logits_buf.assign(static_cast<size_t>(t_out) * vocab, 0.0f);
    ggml_backend_tensor_get(eb.ctc_logits, cc->logits_buf.data(), 0, cc->logits_buf.size() * sizeof(float));

    decode_and_populate(cc, cm, cc->logits_buf.data(), t_out, vocab, clip_ms_for(hp, n_samples),
                        /*utt_index=*/-1);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status run(transcribe_session * session, const float * pcm, int n_samples, const transcribe_run_params *) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<Granite5CtcSession *>(session);
    auto * cm = static_cast<Granite5CtcModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    transcribe::debug::init();

    // The model is monolingual English and carries no language
    // conditioning anywhere in the graph, so an explicit `--language`
    // hint and auto both run the identical forward. Nothing to branch on.
    return encode_and_decode(cc, cm, pcm, n_samples);
}

// Offline batch (transcribe_run_batch). The whole model is the encoder,
// so batching means one encoder dispatch over B utterances on the
// activation's ne[2] axis, then a host CTC decode per utterance. Frontend
// work is pure host code with no cross-utterance state, so it runs in
// parallel first.
//
// Same-length batches take the mask-free graph (bit-identical to
// single-shot per utterance). Uneven batches pad to T_max and apply the
// reference's three masks: zeroed pad frames after input_linear, masked
// pad key columns in every block-local attention, and zeroed pad frames
// before each depthwise conv. Without that last one a short utterance's
// padding would reach its own real frames through the kernel-7 receptive
// field.
transcribe_status run_batch(transcribe_session *          session,
                            const float * const *         pcm,
                            const int *                   n_samples,
                            int                           n,
                            const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<Granite5CtcSession *>(session);
    auto * cm = static_cast<Granite5CtcModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe::debug::init();

    // Per-utterance frontend, in parallel. One bounded thread budget is
    // shared between utterance-level and frame-level parallelism so a
    // batch does not nest n_threads mel pools.
    const int total_threads = cc->n_threads > 0 ? cc->n_threads : transcribe::default_n_threads();
    const int outer_threads = std::max(1, std::min(n, total_threads));
    const int mel_threads   = std::max(1, total_threads / outer_threads);

    std::vector<std::vector<float>> feats(static_cast<size_t>(n));
    std::vector<int>                lens(static_cast<size_t>(n), 0);

    const int64_t t_mel_start = ggml_time_us();
    const bool    all_ok      = transcribe::parallel_for_all(n, outer_threads, [&](int i) -> bool {
        if (pcm[i] == nullptr || n_samples[i] <= 0) {
            return false;
        }
        int t_enc = 0;
        if (compute_encoder_input(*cm->mel, cm->hparams, pcm[i], n_samples[i], mel_threads, feats[i], t_enc) !=
                TRANSCRIBE_OK ||
            t_enc <= 0) {
            return false;
        }
        lens[i] = t_enc;
        return true;
    });
    const int64_t total_mel_us = ggml_time_us() - t_mel_start;

    if (all_ok) {
        int T_max = 0;
        for (int i = 0; i < n; ++i) {
            T_max = std::max(T_max, lens[i]);
        }

        // Pack to [input_dim, T_max, n], zero-padding each utterance's tail.
        const int    input_dim = cm->hparams.enc_input_dim;
        const size_t per_utt   = static_cast<size_t>(input_dim) * T_max;
        cc->feats_buf.assign(per_utt * static_cast<size_t>(n), 0.0f);
        for (int i = 0; i < n; ++i) {
            std::memcpy(cc->feats_buf.data() + per_utt * static_cast<size_t>(i), feats[i].data(),
                        feats[i].size() * sizeof(float));
        }

        EncoderBuild eb{};
        if (auto st = run_encoder(cc, cm, T_max, lens, eb); st == TRANSCRIBE_OK) {
            const int    t_out    = static_cast<int>(eb.ctc_logits->ne[1]);
            const int    vocab    = static_cast<int>(eb.ctc_logits->ne[0]);
            const size_t utt_elems = static_cast<size_t>(t_out) * vocab;
            // Full read then host-slice: non-zero-offset backend reads are
            // not reliable across every backend.
            cc->logits_buf.assign(utt_elems * static_cast<size_t>(n), 0.0f);
            ggml_backend_tensor_get(eb.ctc_logits, cc->logits_buf.data(), 0,
                                    cc->logits_buf.size() * sizeof(float));

            return transcribe::decode_batch_slices(
                cc, n, cc->logits_buf.data(), utt_elems, cc->t_encode_us, total_mel_us,
                [&](int b, const float * logits_b) -> transcribe_status {
                    const int t_valid = std::min(eb.real_lens_out[static_cast<size_t>(b)], t_out);
                    decode_and_populate(cc, cm, logits_b, t_valid, vocab, clip_ms_for(cm->hparams, n_samples[b]),
                                        /*utt_index=*/b);
                    return TRANSCRIBE_OK;
                });
        }
        // Fall through to the serial path on an encoder failure (usually
        // OOM on a large batch), which retries one utterance at a time.
        cc->batch_results.clear();
    }

    // Serial fallback: also the malformed-input path.
    for (int i = 0; i < n; ++i) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (pcm[i] == nullptr || n_samples[i] <= 0) {
            transcribe_session::ResultSet rs;
            rs.status = TRANSCRIBE_ERR_INVALID_ARG;
            cc->batch_results.push_back(std::move(rs));
            continue;
        }
        cc->clear_result();
        const transcribe_status st = run(session, pcm[i], n_samples[i], params);
        cc->batch_results.push_back(cc->capture_result(st));
    }
    return TRANSCRIBE_OK;
}

extern const Arch arch = {
    /* .name             = */ "granite_speech5_ctc",
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

}  // namespace transcribe::granite5_ctc
