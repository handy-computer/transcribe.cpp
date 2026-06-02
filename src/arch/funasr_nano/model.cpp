// arch/funasr_nano/model.cpp - FunASR-Nano family handler.

#include "funasr_nano.h"

#include "adaptor.h"
#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "qwen3_lm/qwen3_lm.h"
#include "sanm/sanm.h"
#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-kaldi-fbank.h"
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
#include <vector>

namespace transcribe::funasr_nano {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   FunAsrNanoModel>);
static_assert(std::is_base_of_v<transcribe_session, FunAsrNanoSession>);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

FunAsrNanoSession::~FunAsrNanoSession() {
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

FunAsrNanoModel::~FunAsrNanoModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    packed_gate_up.free();
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

constexpr const char k_default_variant[] = "fun-asr-nano-2512";

// Resolve chat-template special-token ids at load time.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok,
                                      ChatTokens & out)
{
    struct PieceSlot { const char * piece; int32_t * slot; };
    const PieceSlot pieces[] = {
        { "<|im_start|>",  &out.im_start       },
        { "<|im_end|>",    &out.im_end         },
        { "\xC4\x8A",      &out.newline        },   // byte-level "\n"
        { "system",        &out.role_system    },
        { "user",          &out.role_user      },
        { "assistant",     &out.role_assistant },
    };
    for (const auto & p : pieces) {
        const int id = tok.find(p.piece);
        if (id < 0) {
            std::fprintf(stderr,
                         "funasr_nano: chat-template piece \"%s\" not in tokenizer\n",
                         p.piece);
            return TRANSCRIBE_ERR_GGUF;
        }
        *p.slot = id;
    }
    return TRANSCRIBE_OK;
}

// Build the language/itn prompt text that the reference's
// FunASRNano.get_prompt produces. hotwords-empty path only.
std::string build_funasr_prompt_text(const char * lang, bool use_itn) {
    std::string out;
    if (lang != nullptr && lang[0] != '\0') {
        // 语音转写成 = "transcribe to" / "transcribe into"
        out = "\xE8\xAF\xAD\xE9\x9F\xB3\xE8\xBD\xAC\xE5\x86\x99\xE6\x88\x90";
        out += lang;
    } else {
        out = "\xE8\xAF\xAD\xE9\x9F\xB3\xE8\xBD\xAC\xE5\x86\x99";
    }
    if (!use_itn) {
        // ，不进行文本规整 = "; do not apply text normalization"
        out += "\xEF\xBC\x8C\xE4\xB8\x8D\xE8\xBF\x9B\xE8\xA1\x8C"
               "\xE6\x96\x87\xE6\x9C\xAC\xE8\xA7\x84\xE6\x95\xB4";
    }
    out += "\xEF\xBC\x9A";   // 全角冒号 = "："
    return out;
}

// Encode a chat-template text fragment, pushing the literal id for
// `<|im_start|>` / `<|im_end|>` markers and BPE-encoding the rest. The
// reference Python `tokenizer.encode(...)` recognizes these as added
// tokens; our internal Tokenizer::encode does not, so we split here.
transcribe_status encode_with_chat_specials(const transcribe::Tokenizer & tok,
                                            const ChatTokens &            ct,
                                            const std::string &           text,
                                            std::vector<int32_t> &        out_ids)
{
    static const std::string k_im_start = "<|im_start|>";
    static const std::string k_im_end   = "<|im_end|>";

    size_t cursor = 0;
    while (cursor < text.size()) {
        size_t s_idx = text.find(k_im_start, cursor);
        size_t e_idx = text.find(k_im_end,   cursor);
        size_t next  = std::min(s_idx, e_idx);
        if (next == std::string::npos) {
            // Tail: BPE-encode whatever remains.
            const std::string tail = text.substr(cursor);
            if (!tail.empty()) {
                std::vector<int32_t> ids;
                if (const transcribe_status st = tok.encode(tail, ids);
                    st != TRANSCRIBE_OK) return st;
                out_ids.insert(out_ids.end(), ids.begin(), ids.end());
            }
            return TRANSCRIBE_OK;
        }
        if (next > cursor) {
            const std::string seg = text.substr(cursor, next - cursor);
            std::vector<int32_t> ids;
            if (const transcribe_status st = tok.encode(seg, ids);
                st != TRANSCRIBE_OK) return st;
            out_ids.insert(out_ids.end(), ids.begin(), ids.end());
        }
        if (next == s_idx) {
            out_ids.push_back(ct.im_start);
            cursor = next + k_im_start.size();
        } else {
            out_ids.push_back(ct.im_end);
            cursor = next + k_im_end.size();
        }
    }
    return TRANSCRIBE_OK;
}

// Build the FunASRNano prompt mirroring data_load_speech.
//
// source_input pattern (i=0, sys_prompt=True, do_think=True default):
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
//   <|im_start|>user\n{prompt}<|startofspeech|>!!<|endofspeech|><|im_end|>\n
//   <|im_start|>assistant\n
//
// The reference's pattern.split() splits at <|startofspeech|>...<|endofspeech|>;
// for each text segment it calls tokenizer.encode(...). We mirror that
// boundary exactly; encode_with_chat_specials handles the
// <|im_start|>/<|im_end|> within each segment.
transcribe_status build_funasr_nano_prompt(const transcribe::Tokenizer & tok,
                                           const ChatTokens &            ct,
                                           const char *                  language,
                                           bool                          use_itn,
                                           int                           fake_token_len,
                                           std::vector<int32_t> &        out_ids,
                                           int &                         out_fbank_beg)
{
    out_ids.clear();
    out_fbank_beg = 0;

    const std::string prompt_text = build_funasr_prompt_text(language, use_itn);

    std::string seg_a =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n";
    seg_a += prompt_text;
    if (const transcribe_status st = encode_with_chat_specials(
            tok, ct, seg_a, out_ids);
        st != TRANSCRIBE_OK) return st;

    out_fbank_beg = static_cast<int>(out_ids.size());
    for (int i = 0; i < fake_token_len; ++i) out_ids.push_back(0);

    const std::string seg_c =
        "<|im_end|>\n<|im_start|>assistant\n";
    if (const transcribe_status st = encode_with_chat_specials(
            tok, ct, seg_c, out_ids);
        st != TRANSCRIBE_OK) return st;

    return TRANSCRIBE_OK;
}

void apply_thread_policy(FunAsrNanoSession * cc) {
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

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

transcribe_status load(
    Loader &                          loader,
    const transcribe_model_load_params *   params,
    transcribe_model **               out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<FunAsrNanoModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) return st;
    if (auto st = read_languages_kv (loader.gguf(), *m);       st != TRANSCRIBE_OK) return st;

    // Tokenizer (gpt2 byte-level BPE; pretokenizer "qwen2").
    if (auto st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) return st;

    // Chat template — required; absence is a hard fail because the
    // prompt builder needs the structure.
    (void)read_optional_string_kv(
        loader.gguf(), "tokenizer.chat_template", "funasr_nano",
        "", m->chat_template);

    // Resolve special token ids.
    if (auto st = resolve_chat_tokens(m->tok, m->chat_tokens); st != TRANSCRIBE_OK) return st;

    if (auto st = read_funasr_nano_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK) return st;

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        std::fprintf(stderr,
                     "funasr_nano: tokenizer vocab (%d) != decoder vocab_size (%d)\n",
                     m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        std::fprintf(stderr,
                     "funasr_nano: GGUF tokenizer has no eos_token_id\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Stage 2: reopen with no_alloc to bind the tensor catalog.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (auto st = build_funasr_nano_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(
            backend_req, "funasr_nano", m->plan); st != TRANSCRIBE_OK)
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
                     "funasr_nano: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "funasr_nano");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // Pack gate+up into one tensor per layer so the FFN runs as a
    // single mul_mat. Owned by qwen3_lm::pack_gate_up.
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
                "funasr_nano"))
        {
            m->packed_gate_up.free();
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    {
        const auto & hp_ = m->hparams;
        transcribe::KaldiFbankParams fe_params;
        fe_params.n_mels          = hp_.fe_num_mels;
        fe_params.sample_rate     = hp_.fe_sample_rate;
        fe_params.win_length      = hp_.fe_win_length;
        fe_params.hop_length      = hp_.fe_hop_length;
        fe_params.lfr_m           = hp_.fe_lfr_m;
        fe_params.lfr_n           = hp_.fe_lfr_n;
        fe_params.d_input         = hp_.enc_d_input;
        fe_params.upscale_samples = hp_.fe_upscale_samples;
        fe_params.apply_cmvn      = hp_.fe_apply_cmvn;
        // Fun-ASR-Nano trains on raw LFR features (apply_cmvn=false), so
        // the cmvn_shift / cmvn_scale buffers are intentionally empty.
        m->frontend = std::make_unique<transcribe::KaldiFbankFrontend>(
            std::move(fe_params));
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(
    transcribe_model *                model,
    const transcribe_session_params * params,
    transcribe_session **             out_ctx)
{
    if (model->arch != &arch) return TRANSCRIBE_ERR_INVALID_ARG;

    auto cc = std::make_unique<FunAsrNanoSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    cc->encoder_use_flash = true;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    auto * cm = static_cast<FunAsrNanoModel *>(model);
    {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) kv_type = GGML_TYPE_F32;
        if (!transcribe::qwen3_lm::kv_init(cc->kv_cache, cm->plan.primary,
                                           /*n_ctx=*/2048,
                                           cm->hparams.dec_n_kv_heads,
                                           cm->hparams.dec_head_dim,
                                           cm->hparams.dec_n_layers,
                                           kv_type))
        {
            std::fprintf(stderr,
                         "funasr_nano init_context: kv_init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

transcribe_status run(
    transcribe_session *      session,
    const float *             pcm,
    int                       n_samples,
    const transcribe_run_params * params)
{
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<FunAsrNanoSession *>(session);
    auto * cm = static_cast<FunAsrNanoModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() ||
        cm->frontend == nullptr)
    {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();
    cc->clear_result();

    const auto & hp = cm->hparams;

    // ---- Frontend (host-side) ----
    const int64_t t_mel_start = ggml_time_us();
    const int T_lfr = cm->frontend->compute(
        pcm, static_cast<size_t>(n_samples), cc->frontend_buf);
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    if (T_lfr <= 0) {
        std::fprintf(stderr,
                     "funasr_nano run: input too short (n_samples=%d → T_lfr=0)\n",
                     n_samples);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (transcribe::debug::enabled()) {
        // dump_host_f32's `shape` arg is numpy/row-major (slow-to-fast).
        // The frontend buffer is [T_lfr, d_input] row-major (T outer).
        const long long shape[2] = { T_lfr, hp.enc_d_input };
        transcribe::debug::dump_host_f32(
            "frontend.fbank.lfr.cmvn.out", cc->frontend_buf.data(),
            static_cast<long long>(cc->frontend_buf.size()),
            shape, 2, "frontend.lfr.cmvn");
    }

    // ---- Reset compute context ----
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 32 * 1024 * 1024;  // generous for 70 SAN-M blocks + adaptor + 28 LM blocks
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr,
                         "funasr_nano run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ---- Build encoder graph ----
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights,
                                          hp, T_lfr);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, /*parallel=*/false, /*op_offload=*/true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr, "funasr_nano run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "funasr_nano run: sched_alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload frontend output and PE.
    ggml_backend_tensor_set(eb.frontend_in, cc->frontend_buf.data(),
                            0, cc->frontend_buf.size() * sizeof(float));
    transcribe::sanm::build_sinusoidal_pe(cc->pe_buf, hp.enc_d_input, T_lfr);
    ggml_backend_tensor_set(eb.pe_in, cc->pe_buf.data(),
                            0, cc->pe_buf.size() * sizeof(float));

    apply_thread_policy(cc);

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "funasr_nano run: encoder graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // Dump encoder intermediates.
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("enc.embed.out",        eb.dumps.embed_out,        "encoder.embed.pos_added");
    try_dump("enc.encoders0.0.out",  eb.dumps.encoders0_0_out,  "encoder.encoders0.0");
    if (eb.dumps.encoders_first) try_dump("enc.encoders.0.out", eb.dumps.encoders_first, "encoder.encoders.0");
    if (eb.dumps.encoders_mid) {
        const char * nm = eb.dumps.encoders_mid->name;
        try_dump(nm, eb.dumps.encoders_mid, "encoder.encoders.mid");
    }
    if (eb.dumps.encoders_last) {
        const char * nm = eb.dumps.encoders_last->name;
        try_dump(nm, eb.dumps.encoders_last, "encoder.encoders.last");
    }
    try_dump("enc.after_norm.out",   eb.dumps.after_norm_out,   "encoder.after_norm");
    if (eb.dumps.tp_encoders_first) try_dump("enc.tp_encoders.0.out", eb.dumps.tp_encoders_first, "encoder.tp_encoders.0");
    if (eb.dumps.tp_encoders_mid) {
        const char * nm = eb.dumps.tp_encoders_mid->name;
        try_dump(nm, eb.dumps.tp_encoders_mid, "encoder.tp_encoders.mid");
    }
    if (eb.dumps.tp_encoders_last) {
        const char * nm = eb.dumps.tp_encoders_last->name;
        try_dump(nm, eb.dumps.tp_encoders_last, "encoder.tp_encoders.last");
    }
    try_dump("enc.tp_norm.out",      eb.dumps.tp_norm_out,      "encoder.tp_norm");

    // Read encoder output to host.
    cc->enc_host.resize(static_cast<size_t>(hp.enc_d_model) *
                        static_cast<size_t>(T_lfr));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ---- Build adaptor graph ----
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 8 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
    }

    AdaptorBuild ab = build_adaptor_graph(cc->compute_ctx, cm->weights, hp, T_lfr);
    if (ab.graph == nullptr || ab.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, ab.graph)) {
        std::fprintf(stderr,
                     "funasr_nano run: sched_alloc_graph failed (adaptor)\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_tensor_set(ab.enc_in, cc->enc_host.data(),
                            0, cc->enc_host.size() * sizeof(float));

    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, ab.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "funasr_nano run: adaptor graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }

    try_dump("adaptor.linear1.out",  ab.dumps.linear1_out,  "adaptor.linear1");
    try_dump("adaptor.linear2.out",  ab.dumps.linear2_out,  "adaptor.linear2");
    try_dump("adaptor.blocks.0.out", ab.dumps.block0_out,   "adaptor.blocks.0");
    try_dump("adaptor.out",          ab.dumps.adaptor_out,  "adaptor.out");

    cc->adaptor_host.resize(static_cast<size_t>(hp.adaptor_llm_dim) *
                            static_cast<size_t>(T_lfr));
    ggml_backend_tensor_get(ab.out, cc->adaptor_host.data(), 0,
                            cc->adaptor_host.size() * sizeof(float));

    // ---- Build prompt + audio splice ----
    const int fake_token_len =
        compute_fake_token_len(T_lfr, hp.adaptor_use_low_frame_rate);

    const char * lang = (params != nullptr) ? params->language : nullptr;
    // Generic transcribe_run_params::itn routes here. DEFAULT maps to the
    // family's shipping default (use_itn=false; matches the upstream
    // `itn=False` Python path). OFF / ON override explicitly. The
    // dispatcher's advisory WARN only fires when supports_itn == false;
    // funasr_nano advertises supports_itn = true, so no WARN here.
    bool use_itn = false;
    if (params != nullptr) {
        switch (params->itn) {
            case TRANSCRIBE_ITN_MODE_DEFAULT: use_itn = false; break;
            case TRANSCRIBE_ITN_MODE_OFF:     use_itn = false; break;
            case TRANSCRIBE_ITN_MODE_ON:      use_itn = true;  break;
        }
    }

    std::vector<int32_t> prompt_ids;
    int fbank_beg = 0;
    if (const transcribe_status st = build_funasr_nano_prompt(
            cm->tok, cm->chat_tokens, lang, use_itn, fake_token_len,
            prompt_ids, fbank_beg);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    const int T_prompt   = static_cast<int>(prompt_ids.size());
    const int T_audio    = fake_token_len;
    const int prefix_len = fbank_beg;
    const int suffix_len = T_prompt - prefix_len - T_audio;

    if (T_prompt > cc->kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "funasr_nano run: prompt len %d exceeds kv_n_ctx %d\n",
                     T_prompt, cc->kv_cache.n_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---- Reset KV cache + build prefill ----
    if (cc->kv_cache.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
    }
    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;

    const int64_t t_dec_start = ggml_time_us();

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 16 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
    }

    const bool dumps_on  = transcribe::debug::enabled();
    const bool slice_last = !dumps_on;
    PrefillBuild pb = build_prefill_graph(
        cc->compute_ctx, cm->weights, hp,
        cc->kv_cache, T_prompt, T_audio, prefix_len, suffix_len,
        cc->decoder_use_flash, slice_last);
    if (pb.graph == nullptr || pb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        std::fprintf(stderr,
                     "funasr_nano run: sched_alloc_graph failed (prefill)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(),
                            0, prompt_ids.size() * sizeof(int32_t));

    if (T_audio > 0 && pb.audio_in != nullptr) {
        // Upload first fake_token_len rows of adaptor_out. adaptor_host
        // is row-major [T_lfr, llm_dim]; upload llm_dim * fake_token_len
        // floats.
        const size_t bytes =
            static_cast<size_t>(T_audio) * hp.adaptor_llm_dim * sizeof(float);
        ggml_backend_tensor_set(pb.audio_in, cc->adaptor_host.data(),
                                0, bytes);
    }

    {
        std::vector<int32_t> positions(T_prompt);
        for (int i = 0; i < T_prompt; ++i) positions[i] = i;
        ggml_backend_tensor_set(pb.positions_in, positions.data(),
                                0, positions.size() * sizeof(int32_t));
    }
    {
        const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask(
            static_cast<size_t>(T_prompt) * T_prompt, mask_neg_inf);
        for (int r = 0; r < T_prompt; ++r) {
            for (int c = 0; c <= r; ++c) {
                mask[static_cast<size_t>(r) * T_prompt + c] = mask_zero;
            }
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(),
                                0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, pb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "funasr_nano run: prefill graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }

    cc->kv_cache.n    = T_prompt;
    cc->kv_cache.head = T_prompt;

    try_dump("dec.token_emb",                 pb.dumps.token_emb,       "dec.token_emb");
    try_dump("dec.inputs_embeds.with_audio",  pb.dumps.audio_injected,  "dec.audio_injected");
    try_dump("dec.block.0.out",               pb.dumps.block_0_out,     "dec.block.0");
    if (pb.dumps.block_last_out) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out", hp.dec_n_layers - 1);
        try_dump(nm, pb.dumps.block_last_out, "dec.block.last");
    }
    try_dump("dec.out_before_head",           pb.dumps.out_before_head, "dec.out_before_head");
    try_dump("dec.logits_raw.prefill",        pb.dumps.logits_raw,      "decoder.logits.prefill");

    const int vocab = hp.dec_vocab_size;
    std::vector<float> logits(vocab);
    ggml_backend_tensor_get(pb.out, logits.data(), 0,
                            logits.size() * sizeof(float));

    auto argmax = [&](const std::vector<float> & v) -> int32_t {
        int32_t best = 0;
        float   best_v = v[0];
        for (int32_t i = 1; i < static_cast<int32_t>(v.size()); ++i) {
            if (v[i] > best_v) { best_v = v[i]; best = i; }
        }
        return best;
    };

    std::vector<int32_t> generated_ids;
    int32_t next_tok = argmax(logits);
    generated_ids.push_back(next_tok);

    // ---- Step loop ----
    const int32_t eos_id = hp.eos_token_id;
    const int max_new = 256;
    int cur_past = T_prompt;

    int max_n_kv = 1024;
    while (max_n_kv < T_prompt + max_new) max_n_kv *= 2;
    if (max_n_kv > cc->kv_cache.n_ctx) max_n_kv = cc->kv_cache.n_ctx;

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 8 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
    }
    StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, hp,
                                    cc->kv_cache, max_n_kv,
                                    cc->decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
        std::fprintf(stderr,
                     "funasr_nano run: sched_alloc_graph failed (step)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);

    // Mid-generation logits dump. The reference dumper captures
    // `lm_head_h.values[8]` — the 9th lm_head call (0-indexed, where
    // call 0 is the prefill pass). In our step loop, the 9th lm_head
    // call corresponds to iter 7 of the loop (prefill = 1st call,
    // iter K = (K+2)th call), so we dump when n_steps == 7. Misnamed
    // historically as gen_dump_step=8 — that captured the 10th call
    // and produced an off-by-one against the reference; the bug
    // happened to pass nano's tolerances because nano's adjacent-step
    // logits are similar in magnitude, but blew up on MLT where the
    // tied-lm_head distribution is heavily shifted.
    const int gen_dump_step = 7;
    int n_steps = 0;
    while (next_tok != eos_id &&
           static_cast<int32_t>(generated_ids.size()) < max_new &&
           cur_past + 1 <= max_n_kv)
    {
        ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
        const int32_t pos_val = cur_past;
        ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
        const int64_t kv_idx_val = cur_past;
        ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));

        if (cur_past == T_prompt) {
            std::fill(step_mask.begin(),
                      step_mask.begin() + cur_past + 1, mask_zero);
        } else {
            step_mask[cur_past] = mask_zero;
        }
        ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                static_cast<size_t>(max_n_kv) *
                                sizeof(ggml_fp16_t));

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, sb.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "funasr_nano run: step graph compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }

        int32_t argmax_tok = 0;
        ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
        next_tok = argmax_tok;
        generated_ids.push_back(next_tok);

        // Mid-generation tensor coverage: `dec.logits_raw.gen8` is the
        // logits for the 9th lm_head call (= step 8 of our step loop,
        // when we've already emitted prefill + 7 generated tokens and
        // are emitting the 8th from gen step 8). The reference dumper
        // calls this same tensor `dec.logits_raw.gen8`; capture here so
        // validate.py can compare the n_past>0 step-graph code path.
        if (n_steps == gen_dump_step && transcribe::debug::enabled()) {
            try_dump("dec.logits_raw.gen8", sb.logits, "decoder.logits.gen8");
        }

        cur_past += 1;
        n_steps += 1;
    }
    (void)n_steps;

    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    std::string transcript = cm->tok.decode(
        generated_ids.data(), static_cast<int>(generated_ids.size()));

    cc->t_decode_us = ggml_time_us() - t_dec_start;

    cc->full_text = transcript;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result = true;

    transcribe_session::SegmentEntry seg {};
    seg.text  = transcript;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000
              / static_cast<int64_t>(hp.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name             = */ "funasr_nano",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::funasr_nano
