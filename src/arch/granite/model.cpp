// arch/granite/model.cpp - Granite Speech family handler.
//
// Wiring skeleton (Phase 1 of porting-4-cpp).
//
//   load()         is real: reads the GGUF KV, resolves the tokenizer,
//                  configures the mel frontend, builds the tensor
//                  catalog, allocates the backend buffer, and streams
//                  tensor data. This lets `transcribe-cli -m <gguf> ...`
//                  load every granite variant and print its hparams
//                  before any encode/decode code lands.
//   init_context() returns TRANSCRIBE_ERR_NOT_IMPLEMENTED.
//   run()          returns TRANSCRIBE_ERR_NOT_IMPLEMENTED.
//
// Phase 2-5 fill in the encoder graph, projector, decoder LM, and
// end-to-end run(). The Phase 1 contract is "loader smoke succeeds";
// we deliberately error out on context creation so a caller cannot
// accidentally invoke an empty encode pipeline.
//
// The mel frontend currently builds with the existing MelFrontend
// surface. Granite's torchaudio melspec needs htk-mel norm, no log,
// no per-utterance norm, and a 2-frame stack — those knobs are added
// in Phase 2b. For Phase 1 we just record the cfg and accept whatever
// the existing frontend produces; the encoder isn't wired up yet so
// the discrepancy is not user-visible.

#include "granite.h"

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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace transcribe::granite {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   GraniteModel>);
static_assert(std::is_base_of_v<transcribe_context, GraniteContext>);

GraniteContext::~GraniteContext() {
    kv.free();
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

GraniteModel::~GraniteModel() {
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        ggml_backend_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
    }
    packed_gate_up.free();
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

constexpr const char k_default_variant[] = "granite-speech";
constexpr float kBnEps = 1e-5f;

// Pre-fuse BatchNorm scale/bias per encoder block. Mirrors parakeet's
// fuse_batch_norm: allocates a separate ggml context + CPU buffer for
// the [inner_dim] fused tensors, copies the raw BN tensors out,
// computes  scale = gamma / sqrt(var + eps) and bias = beta - mean * scale,
// and uploads.
transcribe_status fuse_batch_norm(GraniteModel & m) {
    const size_t n_blocks = m.weights.enc_blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t inner = static_cast<int64_t>(m.hparams.enc_hidden) *
                          m.hparams.enc_conv_expansion;
    const size_t tensor_bytes = static_cast<size_t>(inner) * sizeof(float);

    const size_t ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    m.bn_fused_ctx = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) {
        std::fprintf(stderr, "granite: bn_fused ggml_init failed\n");
        return TRANSCRIBE_ERR_BACKEND;
    }

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.enc_blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx,
                                                   GGML_TYPE_F32, inner);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx,
                                                   GGML_TYPE_F32, inner);
    }

    // The plan's scheduler_list always has CPU last as fallback. We
    // could pin to the primary (Metal/Vulkan) but the fused buffers
    // are tiny and live the whole life of the model — putting them on
    // CPU keeps GPU memory free for KV cache + activations.
    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(
        m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) {
        std::fprintf(stderr, "granite: bn_fused alloc_ctx_tensors failed\n");
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

// Resolve the chat-template pieces we may need across the two
// templates (bare-USER/ASSISTANT for 1b/2b, granite-4 system-role for
// -plus). The audio token must exist on every variant; the
// start_of_role / end_of_role pieces only exist on the -plus vocab,
// so they're allowed to be absent.
//
// Hard-fails for the must-have pieces (audio, end_of_text, pad) so a
// future vocab drift surfaces at load rather than mid-decode.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok,
                                      ChatTokens &                  out)
{
    // Must-have on every granite variant.
    const struct { const char * piece; int32_t * slot; } required[] = {
        { "<|audio|>",       &out.audio       },
        { "<|end_of_text|>", &out.end_of_text },
        { "<|pad|>",         &out.pad         },
    };
    for (const auto & p : required) {
        const int id = tok.find(p.piece);
        if (id < 0) {
            std::fprintf(stderr,
                         "granite: tokenizer missing required piece \"%s\"\n",
                         p.piece);
            return TRANSCRIBE_ERR_GGUF;
        }
        *p.slot = id;
    }
    // Optional: present in the -plus vocab, absent in 1b/2b. We resolve
    // anyway so the plus prompt builder can use them without re-looking.
    out.start_of_role = tok.find("<|start_of_role|>");   // may be -1
    out.end_of_role   = tok.find("<|end_of_role|>");     // may be -1
    return TRANSCRIBE_OK;
}

transcribe_status load(
    Loader &                         loader,
    const transcribe_model_params *  params,
    transcribe_model **              out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<GraniteModel>();
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

    // Tokenizer (byte-level BPE; loader handles the "gpt2" model tag).
    if (const transcribe_status st = m->tok.load(loader.gguf());
        st != TRANSCRIBE_OK) return st;

    // Chat template (required at run time but the runtime accepts an
    // empty string today and the prompt builder reports a clear error
    // later). We read it eagerly so a converter regression that drops
    // the template surfaces at load.
    (void)read_optional_string_kv(
        loader.gguf(), "tokenizer.chat_template", "granite",
        "", m->chat_template);

    if (const transcribe_status st = resolve_chat_tokens(m->tok, m->chat_tokens);
        st != TRANSCRIBE_OK) return st;

    // Hparams.
    if (const transcribe_status st = read_granite_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK) return st;

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();
    // The Tokenizer class doesn't track pad; chat_tokens.pad was
    // resolved above by direct piece lookup.
    m->hparams.pad_token_id = m->chat_tokens.pad;

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        std::fprintf(stderr,
                     "granite: tokenizer vocab (%d) != decoder vocab_size (%d)\n",
                     m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        std::fprintf(stderr, "granite: tokenizer has no eos_token_id\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Mel frontend. Phase 1: we configure with the granite KV but the
    // existing MelFrontend doesn't yet honour htk-norm / no-log /
    // power=2 / per-utterance-stack. Phase 2b extends the frontend
    // surface. Since the encoder graph isn't wired up in Phase 1, the
    // frontend's actual output is not observed by any code path that
    // matters yet — but we still construct it so the loader smoke
    // exercises the same code path the eventual run() will use.
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
        cfg.window_type  = m->hparams.fe_window;     // "hann_periodic"
        cfg.normalize    = m->hparams.fe_normalize;  // "none"

        // Frontend buffers baked by the converter: librosa htk mel
        // filterbank + periodic Hann window. We pick them up here so
        // the eventual frontend bring-up uses bit-identical values.
        {
            using R = transcribe::load_common::ReadF32Result;
            const size_t fb_elems =
                static_cast<size_t>(cfg.num_mels) *
                static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto fb_rc = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.mel_filterbank", fb_elems,
                "granite", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto win_rc = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.window", win_elems,
                "granite", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
    }

    // Stage 2: reopen with no_alloc to build the tensor catalog.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_granite_weights(
            m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // Backend plan.
    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "granite", m->plan);
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
                     "granite: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "granite");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // BatchNorm fusion. Computed once at load using the streamed-in
    // gamma/beta + running_mean/var; the encoder graph references the
    // fused [inner_dim] scale/bias tensors per block.
    if (const transcribe_status st = fuse_batch_norm(*m); st != TRANSCRIBE_OK) {
        return st;
    }

    // Gate+Up fusion: shared qwen3_lm packer. Drops one FFN mul_mat per
    // block and lets the graph use ggml_swiglu(gate_up) instead of
    // explicit silu(gate)*up.
    {
        std::vector<transcribe::qwen3_lm::GateUpEntry> entries;
        entries.reserve(m->weights.dec_blocks.size());
        for (auto & b : m->weights.dec_blocks) {
            entries.push_back({b.ffn_gate_w, b.ffn_up_w, &b.ffn_gate_up_w});
        }
        if (!transcribe::qwen3_lm::pack_gate_up(
                m->plan.primary,
                m->hparams.dec_hidden,
                m->hparams.dec_intermediate,
                entries,
                m->packed_gate_up,
                "granite"))
        {
            m->packed_gate_up.free();
            return TRANSCRIBE_ERR_GGUF;
        }
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

    auto cc = std::make_unique<GraniteContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    // Encoder uses manual mul_mat + soft_max (no flash) because the
    // Shaw bias requires a per-(head, block) additive term that
    // flash_attn_ext doesn't broadcast cleanly yet. Decoder flash is
    // wired in Phase 4.
    cc->encoder_use_flash = false;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash,
                                           cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Phase 2 run(): mel + 2-frame stack -> encoder graph -> dump enc.*
// tensors. The projector + LM are still NOT_IMPLEMENTED, so we set
// has_result/full_text to "" so validate.py picks up a `text:` line
// and the per-tensor compare can run against enc.* dumps. Phase 3-5
// replaces this body with the full pipeline.
// Map a BCP-47 language code or English name to the language name the
// granite-speech instruction expects. Returns nullptr for unsupported.
// granite-speech (1b/2b/-plus) advertises fr / de / es / pt / ja.
static const char * granite_target_language_name(const char * code_or_name) {
    if (code_or_name == nullptr || *code_or_name == '\0') return nullptr;
    std::string s = code_or_name;
    for (auto & c : s) c = static_cast<char>(std::tolower(c));
    if (s == "de" || s == "ger" || s == "deu" || s == "german")     return "German";
    if (s == "fr" || s == "fra" || s == "fre" || s == "french")     return "French";
    if (s == "es" || s == "spa" || s == "spanish")                   return "Spanish";
    if (s == "pt" || s == "por" || s == "portuguese")                return "Portuguese";
    if (s == "ja" || s == "jpn" || s == "japanese")                  return "Japanese";
    if (s == "en" || s == "eng" || s == "english")                   return "English";
    return nullptr;
}

transcribe_status run(
    transcribe_context *      ctx_base,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * params)
{
    if (ctx_base == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<GraniteContext *>(ctx_base);
    auto * cm = static_cast<GraniteModel *>(cc->model);

    transcribe::debug::init();

    if (!cm->mel.has_value()) {
        std::fprintf(stderr, "granite run: model has no MelFrontend\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // ----- Mel + 2-frame stack (host side) -----
    const int64_t t_mel_start = ggml_time_us();
    int t_enc = 0;
    if (const transcribe_status mst = compute_mel_encoder_input(
            *cm->mel, pcm, n_samples, cc->n_threads, cc->mel_buf, t_enc);
        mst != TRANSCRIBE_OK)
    {
        std::fprintf(stderr,
                     "granite run: mel/stack failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // Dump enc.mel.in in the reference's [T_enc, input_dim] row-major
    // shape. The graph input is the same data, so this is the natural
    // comparison point on the host.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { t_enc, cm->hparams.enc_input_dim };
        transcribe::debug::dump_host_f32(
            "enc.mel.in", cc->mel_buf.data(),
            static_cast<long long>(cc->mel_buf.size()),
            shape, 2, "encoder");
    }

    // ----- Reset compute state -----
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        // 16 conformer blocks * many ops. 32 MB is generous headroom.
        ip.mem_size   = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr,
                         "granite run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ----- Build encoder graph -----
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights,
                                          cm->hparams, t_enc,
                                          cc->encoder_use_flash);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Encoder operates at T_enc throughout (no host-side padding of
    // the mel buffer required). The Shaw-attention helper pads to
    // T_pad internally using the eb.zero_pad graph input.

    // ----- Scheduler -----
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            32768, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr,
                         "granite run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "granite run: sched_alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Upload inputs -----
    // mel_in: contiguous [T_enc, input_dim] row-major matches ggml ne
    // [input_dim, T_enc] (ne[0]=input_dim is innermost).
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(), 0,
                            cc->mel_buf.size() * sizeof(float));

    // Shaw attention_dists. Row-major over (c, r) with int32 indices.
    std::vector<int32_t> dists = precompute_attention_dists(
        cm->hparams.enc_context_size, cm->hparams.enc_max_pos_emb);
    ggml_backend_tensor_set(eb.attention_dists, dists.data(), 0,
                            dists.size() * sizeof(int32_t));

    // last_block_mask: [context_size, context_size, n_blocks_local].
    // All zeros except the last slice when t_enc is not a multiple of
    // context_size.
    {
        const int ctx_size = cm->hparams.enc_context_size;
        const size_t plane = static_cast<size_t>(ctx_size) * ctx_size;
        std::vector<float> mask(plane * eb.n_blocks_local, 0.0f);
        const int rem = eb.last_block_rem;
        if (rem > 0 && rem < ctx_size) {
            std::vector<float> last = precompute_last_block_mask(ctx_size, rem);
            // Last slice index = n_blocks_local - 1. ne[2] is the slice
            // axis (row-major plane order). Offset = (n_blocks_local-1)*plane.
            std::memcpy(mask.data() + plane * (eb.n_blocks_local - 1),
                        last.data(), plane * sizeof(float));
        }
        ggml_backend_tensor_set(eb.last_block_mask, mask.data(), 0,
                                mask.size() * sizeof(float));
    }

    // zero_pad: only present when t_enc is not aligned to context_size.
    if (eb.zero_pad != nullptr) {
        const size_t n_elems = static_cast<size_t>(eb.zero_pad->ne[0]) *
                               eb.zero_pad->ne[1];
        std::vector<float> zeros(n_elems, 0.0f);
        ggml_backend_tensor_set(eb.zero_pad, zeros.data(), 0,
                                zeros.size() * sizeof(float));
    }

    // Thread count.
    {
        int n_threads = cc->n_threads;
        if (n_threads <= 0) {
            n_threads = std::min(8, std::max(1, static_cast<int>(
                std::thread::hardware_concurrency())));
        }
        for (int i = 0; i < ggml_backend_sched_get_n_backends(cc->sched); ++i) {
            ggml_backend_t be  = ggml_backend_sched_get_backend(cc->sched, i);
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
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "granite run: encoder graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // ----- Dump intermediates -----
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("enc.input_linear.out", eb.dumps.input_linear_out, "encoder");
    try_dump("enc.block.0.out",      eb.dumps.block_0_out,      "encoder");
    {
        const int mid_idx = cm->hparams.enc_n_layers / 2;
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", mid_idx);
        try_dump(bname, eb.dumps.block_mid_out, "encoder");
    }
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out",
                      cm->hparams.enc_n_layers - 1);
        try_dump(bname, eb.dumps.block_last_out, "encoder");
    }
    try_dump("enc.out", eb.dumps.out_named, "encoder");

    // ----- Read encoder output to host for the projector pass -----
    // Phase 3 uses a SEPARATE compute graph for the projector so the
    // per-stage validation harness can compare proj.* dumps without
    // any encoder graph state lingering. Phase 5 may fuse encoder +
    // projector into one compute when end-to-end latency matters.
    const int64_t enc_hidden_runtime = eb.out->ne[0];
    const int64_t enc_T_runtime      = eb.out->ne[1];
    std::vector<float> enc_host(static_cast<size_t>(enc_hidden_runtime) *
                                 enc_T_runtime);
    ggml_backend_tensor_get(eb.out, enc_host.data(), 0,
                            enc_host.size() * sizeof(float));

    // ----- Cat-hidden-layers handling (-plus variant) -----
    // The -plus checkpoint sets enc.cat_hidden_layers = [3]: the
    // encoder graph (see encoder.cpp) channel-concatenates block[K-1].out
    // captures with the final hidden along dim=0 BEFORE returning eb.out,
    // so the projector input dimension widens from 1024 to 2048
    // automatically. Nothing variant-specific is needed here.

    // ----- Projector graph -----
    ggml_context * proj_ctx = nullptr;
    {
        ggml_init_params ip {};
        ip.mem_size   = 16 * 1024 * 1024;  // 2 Q-Former layers — small graph
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        proj_ctx = ggml_init(ip);
        if (proj_ctx == nullptr) {
            std::fprintf(stderr, "granite run: ggml_init for proj_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ProjectorBuild pb = build_projector_graph(proj_ctx, cm->weights,
                                              cm->hparams,
                                              static_cast<int>(enc_T_runtime));
    if (pb.graph == nullptr || pb.out == nullptr) {
        ggml_free(proj_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        std::fprintf(stderr,
                     "granite run: sched_alloc_graph failed (projector)\n");
        ggml_free(proj_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload encoder output to the projector's enc_in input.
    ggml_backend_tensor_set(pb.enc_in, enc_host.data(), 0,
                            enc_host.size() * sizeof(float));
    if (pb.enc_pad != nullptr) {
        const size_t pad_bytes =
            static_cast<size_t>(pb.enc_pad->ne[0]) * pb.enc_pad->ne[1] *
            sizeof(float);
        std::vector<float> pad(pad_bytes / sizeof(float), 0.0f);
        ggml_backend_tensor_set(pb.enc_pad, pad.data(), 0, pad_bytes);
    }

    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, pb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "granite run: projector graph compute failed (%d)\n",
                     static_cast<int>(gs));
        ggml_free(proj_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    try_dump("proj.qformer.out", pb.dumps.qformer_out, "projector");
    try_dump("proj.out",         pb.dumps.proj_out,    "projector");

    // Read projector output (the audio tokens in LM hidden space).
    cc->n_audio_tokens = pb.n_audio_tokens;
    cc->audio_tokens_host.resize(
        static_cast<size_t>(pb.out->ne[0]) * pb.out->ne[1]);
    ggml_backend_tensor_get(pb.out, cc->audio_tokens_host.data(), 0,
                            cc->audio_tokens_host.size() * sizeof(float));

    ggml_free(proj_ctx);

    // ----- Decoder prefill pass -----
    // Build the prompt as `prefix_text | <|audio|>×n | suffix_text`.
    // Two templates ship today:
    //
    //   1b / 2b  (chat_tokens.start_of_role < 0):
    //     "USER: <|audio|>{instruction}\n ASSISTANT:"
    //   -plus    (chat_tokens.start_of_role >= 0):
    //     granite-4 system-role chat. The system message and assistant
    //     start-of-role are appended via special-token id splices so the
    //     BPE encoder only sees plain text fragments — special tokens
    //     are emitted directly rather than tokenized from their literal
    //     piece strings.
    //
    // Granite-speech instructions. Each variant's published WER on
    // LibriSpeech test-clean was measured with a specific prompt; using
    // a different prompt produces a measurable WER drift (typically
    // 0.02-0.04pp). Match the model card per variant.
    //
    //   granite-4.0-1b-speech            : "can you transcribe the speech into a written format?"
    //   granite-speech-4.1-2b            : "transcribe the speech with proper punctuation and capitalization."
    //   granite-speech-4.1-2b-plus       : " can you transcribe the speech into a written format?"
    //                                       (leading space matters: BPE token IDs differ)
    //                                       PLUS a Granite-specific system prompt (see use_granite4_chat below).
    //   word-timestamps task             : "transcribe the speech with timestamps in [SS:MS] format"
    //                                       (only -plus actually emits the [SS:N] markers; 1b/2b
    //                                       fall back to plain transcript, which Stage-4 Capability
    //                                       Validation records as SKIP rather than PASS)
    //   translate task                   : "can you translate the speech into <Language>?"
    std::string instruction;
    if (cm->hparams.variant == "granite-speech-4.1-2b") {
        instruction = "transcribe the speech with proper punctuation and capitalization.";
    } else if (cm->hparams.variant == "granite-speech-4.1-2b-plus") {
        instruction = " can you transcribe the speech into a written format?";
    } else {
        // granite-4.0-1b-speech and any future variant default to the
        // 1b prompt unless we add a per-variant override above.
        instruction = "can you transcribe the speech into a written format?";
    }
    bool want_timestamps = false;
    if (params != nullptr) {
        if (params->task == TRANSCRIBE_TASK_TRANSLATE) {
            if (params->target_language == nullptr ||
                params->target_language[0] == '\0')
            {
                std::fprintf(stderr,
                             "granite run: translate task requires --target-language\n");
                return TRANSCRIBE_ERR_INVALID_ARG;
            }
            const char * lang_name =
                granite_target_language_name(params->target_language);
            if (lang_name == nullptr) {
                std::fprintf(stderr,
                             "granite run: target_language '%s' is not advertised "
                             "by granite-speech (supported: de fr es pt ja en)\n",
                             params->target_language);
                return TRANSCRIBE_ERR_INVALID_ARG;
            }
            instruction = std::string("can you translate the speech into ")
                        + lang_name + "?";
        } else if (params->timestamps == TRANSCRIBE_TIMESTAMPS_WORD ||
                   params->timestamps == TRANSCRIBE_TIMESTAMPS_AUTO)
        {
            instruction = "transcribe the speech with timestamps in [SS:MS] format";
            want_timestamps = true;
        }
    }
    (void)want_timestamps;  // reserved for future per-word segment emission

    std::vector<int32_t> prefix_ids;
    std::vector<int32_t> suffix_ids;
    // The granite-4 system-role template is identified by the chat
    // template KV containing "<|start_of_role|>". 1b / 2b vocabs DO
    // carry the start_of_role / end_of_role added tokens (their ids are
    // resolvable via tok.find), but their chat template is the bare
    // USER:/ASSISTANT: Jinja, so token presence alone is not a usable
    // selector.
    const bool use_granite4_chat =
        cm->chat_template.find("<|start_of_role|>") != std::string::npos
        && cm->chat_tokens.start_of_role >= 0
        && cm->chat_tokens.end_of_role >= 0;
    if (use_granite4_chat) {
        // prefix:
        //   <|start_of_role|> system <|end_of_role|>
        //   {system_content}
        //   <|end_of_text|> \n <|start_of_role|> user <|end_of_role|>
        //   <|audio|>...
        // We assemble token ids manually so the special pieces are
        // exact, not BPE'd.
        //
        // Per-variant system content (the granite-speech-4.1-2b-plus
        // model card publishes the Granite-style system prompt below;
        // future granite4-chat variants would slot in here):
        const char * system_content =
            (cm->hparams.variant == "granite-speech-4.1-2b-plus")
                ? "Knowledge Cutoff Date: April 2024.\n"
                  "Today's Date: December 19, 2024.\n"
                  "You are Granite, developed by IBM. You are a helpful AI assistant"
                : "You are a helpful assistant. Please ensure responses "
                  "are professional, accurate, and safe.";
        std::vector<int32_t> text_a, text_b;
        if (const transcribe_status st = cm->tok.encode("system", text_a);
            st != TRANSCRIBE_OK) return st;
        if (const transcribe_status st = cm->tok.encode(system_content, text_b);
            st != TRANSCRIBE_OK) return st;
        prefix_ids.push_back(cm->chat_tokens.start_of_role);
        prefix_ids.insert(prefix_ids.end(), text_a.begin(), text_a.end());
        prefix_ids.push_back(cm->chat_tokens.end_of_role);
        prefix_ids.insert(prefix_ids.end(), text_b.begin(), text_b.end());
        prefix_ids.push_back(cm->chat_tokens.end_of_text);

        std::vector<int32_t> nl_ids, user_ids;
        if (const transcribe_status st = cm->tok.encode("\n", nl_ids);
            st != TRANSCRIBE_OK) return st;
        if (const transcribe_status st = cm->tok.encode("user", user_ids);
            st != TRANSCRIBE_OK) return st;
        prefix_ids.insert(prefix_ids.end(), nl_ids.begin(), nl_ids.end());
        prefix_ids.push_back(cm->chat_tokens.start_of_role);
        prefix_ids.insert(prefix_ids.end(), user_ids.begin(), user_ids.end());
        prefix_ids.push_back(cm->chat_tokens.end_of_role);
        // (audio tokens emit between prefix and suffix.)

        // suffix:  {instruction} <|end_of_text|> \n
        //          <|start_of_role|> assistant <|end_of_role|>
        // The trailing assistant-role marker is equivalent to HF's
        // `add_generation_prompt=True`. We deliberately diverge from the
        // Stage-2 dumper (which used False to match a literal
        // apply_chat_template call) because the model otherwise emits
        // immediate <|end_of_text|> on short / ambiguous clips (27/512
        // empties on test-clean with False, 0 with True). The Stage-4
        // tensor dumps were captured WITHOUT this marker — see
        // tests/tolerances/granite.json _comment for the regime note —
        // but production transcription uses it because it is the
        // user-visible chat protocol the model was trained on.
        if (const transcribe_status st = cm->tok.encode(instruction, suffix_ids);
            st != TRANSCRIBE_OK) return st;
        suffix_ids.push_back(cm->chat_tokens.end_of_text);
        std::vector<int32_t> nl_ids2;
        if (const transcribe_status st = cm->tok.encode("\n", nl_ids2);
            st != TRANSCRIBE_OK) return st;
        suffix_ids.insert(suffix_ids.end(), nl_ids2.begin(), nl_ids2.end());
        suffix_ids.push_back(cm->chat_tokens.start_of_role);
        std::vector<int32_t> asst_ids;
        if (const transcribe_status st = cm->tok.encode("assistant", asst_ids);
            st != TRANSCRIBE_OK) return st;
        suffix_ids.insert(suffix_ids.end(), asst_ids.begin(), asst_ids.end());
        suffix_ids.push_back(cm->chat_tokens.end_of_role);
    } else {
        // bare USER:/ASSISTANT: template (1b / 2b).
        const std::string prefix_text = "USER: ";
        const std::string suffix_text =
            instruction + "\n ASSISTANT:";
        if (const transcribe_status st = cm->tok.encode(prefix_text, prefix_ids);
            st != TRANSCRIBE_OK)
        {
            std::fprintf(stderr, "granite run: tokenize(prefix) failed\n");
            return st;
        }
        if (const transcribe_status st = cm->tok.encode(suffix_text, suffix_ids);
            st != TRANSCRIBE_OK)
        {
            std::fprintf(stderr, "granite run: tokenize(suffix) failed\n");
            return st;
        }
    }

    const int n_audio_tokens = cc->n_audio_tokens;
    const int prefix_len     = static_cast<int>(prefix_ids.size());
    const int suffix_len     = static_cast<int>(suffix_ids.size());
    const int T_prompt       = prefix_len + n_audio_tokens + suffix_len;

    // Reference quirk: HF's GraniteSpeechForConditionalGeneration replaces
    // audio_token_id with 0 BEFORE the embed_tokens lookup (the
    // resulting rows are scattered over with audio_features moments
    // later). We mirror that masking so dec.token_emb matches at the
    // audio positions — the forward result is identical either way
    // because those rows are overwritten by the audio scatter.
    std::vector<int32_t> input_ids;
    input_ids.reserve(T_prompt);
    input_ids.insert(input_ids.end(), prefix_ids.begin(), prefix_ids.end());
    for (int i = 0; i < n_audio_tokens; ++i) {
        input_ids.push_back(0);
    }
    input_ids.insert(input_ids.end(), suffix_ids.begin(), suffix_ids.end());

    // Size the KV cache dynamically: T_prompt + room for the longest
    // generation we'll emit. Matches the HF reference's DynamicCache
    // semantics (grows as needed) without paying for a worst-case
    // pre-alloc. If the existing cache is too small for this prompt,
    // free and re-allocate. Round up to a 256-row bucket so back-to-back
    // runs of similar audio lengths don't keep re-allocating.
    constexpr int kStepBudget = 256;
    constexpr int kKvBucket   = 256;
    const int needed_raw = T_prompt + kStepBudget;
    const int needed_n_ctx =
        ((needed_raw + kKvBucket - 1) / kKvBucket) * kKvBucket;

    if (cc->kv.self_k != nullptr && cc->kv.n_ctx < needed_n_ctx) {
        cc->kv.free();
    }
    if (cc->kv.self_k == nullptr) {
        ggml_type kv_t = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) kv_t = GGML_TYPE_F32;
        if (!transcribe::qwen3_lm::kv_init(
                cc->kv, cm->plan.primary, needed_n_ctx,
                cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                cm->hparams.dec_n_layers, kv_t))
        {
            std::fprintf(stderr,
                         "granite run: kv_init failed (n_ctx=%d)\n",
                         needed_n_ctx);
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    // After dynamic sizing this should be unreachable; keep as a
    // belt-and-braces invariant check.
    if (T_prompt > cc->kv.n_ctx) {
        std::fprintf(stderr,
                     "granite run: T_prompt=%d exceeds kv.n_ctx=%d\n",
                     T_prompt, cc->kv.n_ctx);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Prefill graph in its own compute context.
    ggml_context * dec_ctx = nullptr;
    {
        ggml_init_params ip {};
        ip.mem_size   = 64 * 1024 * 1024;  // 40 layers — generous headroom
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        dec_ctx = ggml_init(ip);
        if (dec_ctx == nullptr) {
            std::fprintf(stderr, "granite run: ggml_init for dec_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    PrefillBuild dec = build_prefill_graph(
        dec_ctx, cm->weights, cm->hparams, cc->kv,
        T_prompt, n_audio_tokens, prefix_len, suffix_len,
        cc->decoder_use_flash, /*slice_last=*/false);
    if (dec.graph == nullptr || dec.out == nullptr) {
        ggml_free(dec_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, dec.graph)) {
        std::fprintf(stderr,
                     "granite run: sched_alloc_graph failed (decoder)\n");
        ggml_free(dec_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload decoder inputs.
    ggml_backend_tensor_set(dec.input_ids_in, input_ids.data(), 0,
                            input_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(dec.audio_feats_in, cc->audio_tokens_host.data(),
                            0, cc->audio_tokens_host.size() * sizeof(float));

    {
        std::vector<int32_t> positions(T_prompt);
        for (int i = 0; i < T_prompt; ++i) positions[i] = i;
        ggml_backend_tensor_set(dec.positions_in, positions.data(), 0,
                                positions.size() * sizeof(int32_t));
    }

    // Causal mask in F16: zero on/below diagonal, -inf above. f16 -inf
    // is the bit pattern 0xFC00.
    {
        const uint16_t f16_zero    = 0x0000;
        const uint16_t f16_neg_inf = 0xFC00;
        std::vector<uint16_t> mask(
            static_cast<size_t>(T_prompt) * T_prompt, f16_zero);
        for (int q = 0; q < T_prompt; ++q) {
            for (int k = q + 1; k < T_prompt; ++k) {
                mask[static_cast<size_t>(q) * T_prompt + k] = f16_neg_inf;
            }
        }
        ggml_backend_tensor_set(dec.mask_in, mask.data(), 0,
                                mask.size() * sizeof(uint16_t));
    }

    const int64_t t_dec_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, dec.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "granite run: decoder prefill compute failed (%d)\n",
                     static_cast<int>(gs));
        ggml_free(dec_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_decode_us = ggml_time_us() - t_dec_start;

    cc->kv.n    = T_prompt;
    cc->kv.head = T_prompt;

    try_dump("dec.token_emb",       dec.dumps.token_emb,       "decoder");
    try_dump("dec.audio_injected",  dec.dumps.audio_injected,  "decoder");
    try_dump("dec.block.0.out",     dec.dumps.block_0_out,     "decoder");
    if (dec.dumps.block_mid_out != nullptr) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out",
                      cm->hparams.dec_n_layers / 2);
        try_dump(nm, dec.dumps.block_mid_out, "decoder");
    }
    if (dec.dumps.block_last_out != nullptr) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out",
                      cm->hparams.dec_n_layers - 1);
        try_dump(nm, dec.dumps.block_last_out, "decoder");
    }
    try_dump("dec.out_before_head", dec.dumps.out_before_head, "decoder");
    try_dump("dec.logits_raw",      dec.dumps.logits_raw,      "decoder");

    // Read final-position logits to compute the first generated token.
    std::vector<float> last_logits(cm->hparams.dec_vocab_size);
    ggml_backend_tensor_get(dec.out, last_logits.data(), 0,
                            last_logits.size() * sizeof(float));
    ggml_free(dec_ctx);

    auto argmax_logits = [](const std::vector<float> & v) -> int32_t {
        int32_t best = 0;
        float   best_v = v[0];
        for (int32_t i = 1; i < static_cast<int32_t>(v.size()); ++i) {
            if (v[i] > best_v) { best_v = v[i]; best = i; }
        }
        return best;
    };

    int32_t next_id = argmax_logits(last_logits);

    // ----- Greedy step loop -----
    // Build one reusable step graph sized for the whole run. The
    // n_ctx of the KV cache bounds the max generation length we can
    // attend over.
    const int max_n_kv  = cc->kv.n_ctx;
    const int max_steps = std::min(256, max_n_kv - T_prompt);

    ggml_context * step_ctx = nullptr;
    {
        ggml_init_params ip {};
        ip.mem_size   = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        step_ctx = ggml_init(ip);
    }
    StepBuild step = build_step_graph(step_ctx, cm->weights, cm->hparams,
                                      cc->kv, max_n_kv,
                                      cc->decoder_use_flash);
    if (step.graph == nullptr) {
        ggml_free(step_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, step.graph)) {
        std::fprintf(stderr,
                     "granite run: sched_alloc_graph failed (step)\n");
        ggml_free(step_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    std::vector<int32_t> gen_ids;
    gen_ids.reserve(max_steps);

    const int32_t eos_id = cm->hparams.eos_token_id;
    // Step-loop mask sized for max_n_kv. We pre-fill -inf and let
    // valid positions get zeroed per step.
    std::vector<uint16_t> step_mask(max_n_kv, 0xFC00);

    for (int step_i = 0; step_i < max_steps; ++step_i) {
        if (next_id == eos_id) break;
        gen_ids.push_back(next_id);

        const int32_t pos = T_prompt + step_i;          // RoPE position
        const int64_t kv_idx = pos;                     // KV write row
        const int32_t input_id = next_id;

        // Zero mask up through `pos` (inclusive); the new K/V row sits
        // at index `pos`, so attention can see it.
        std::fill(step_mask.begin(), step_mask.end(),
                  static_cast<uint16_t>(0xFC00));
        for (int i = 0; i <= pos; ++i) step_mask[i] = 0x0000;

        ggml_backend_tensor_set(step.input_id_in, &input_id, 0, sizeof(int32_t));
        ggml_backend_tensor_set(step.position_in, &pos,      0, sizeof(int32_t));
        ggml_backend_tensor_set(step.kv_idx_in,   &kv_idx,   0, sizeof(int64_t));
        ggml_backend_tensor_set(step.mask_in, step_mask.data(), 0,
                                step_mask.size() * sizeof(uint16_t));

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, step.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "granite run: step compute failed (%d)\n",
                         static_cast<int>(gs));
            ggml_free(step_ctx);
            return TRANSCRIBE_ERR_GGUF;
        }

        int32_t amax = 0;
        ggml_backend_tensor_get(step.out, &amax, 0, sizeof(int32_t));
        cc->kv.n    = pos + 1;
        cc->kv.head = pos + 1;
        next_id = amax;
    }

    ggml_free(step_ctx);

    // ----- Detokenize -----
    cc->full_text = cm->tok.decode(gen_ids.data(),
                                   static_cast<int>(gen_ids.size()));

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
    /* .name             = */ "granite_speech",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::granite
