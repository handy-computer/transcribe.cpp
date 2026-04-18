// arch/qwen3_asr/model.cpp - Qwen3-ASR family handler.
//
// Wiring-only skeleton. `load()` is real — it reads the GGUF KV,
// resolves the tokenizer + frontend, and builds the tensor catalog —
// so that GGUF conversion round-trips can be verified before inference
// code lands. `init_context()` and `run()` currently return
// TRANSCRIBE_ERR_NOT_IMPLEMENTED and will be filled in once the
// encoder and LM graph builders exist.

#include "qwen3_asr.h"

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
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace transcribe::qwen3_asr {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   QwenAsrModel>);
static_assert(std::is_base_of_v<transcribe_context, QwenAsrContext>);

QwenAsrContext::~QwenAsrContext() {
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

QwenAsrModel::~QwenAsrModel() {
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

bool kv_cache_init(QwenAsrKvCache & cache,
                   ggml_backend_t   backend,
                   int              n_ctx,
                   int              n_kv_heads,
                   int              head_dim,
                   int              n_layer,
                   ggml_type        kv_type)
{
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        std::fprintf(stderr,
                     "qwen3_asr kv_cache: unsupported kv_type=%d "
                     "(only F16/F32)\n", static_cast<int>(kv_type));
        return false;
    }

    const size_t ctx_size = 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        std::fprintf(stderr, "qwen3_asr kv_cache: ggml_init failed\n");
        return false;
    }

    const int64_t elems =
        static_cast<int64_t>(n_kv_heads) * head_dim * n_ctx * n_layer;

    cache.self_k = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
    cache.self_v = ggml_new_tensor_1d(cache.ctx, kv_type, elems);
    ggml_set_name(cache.self_k, "kv_self_k");
    ggml_set_name(cache.self_v, "kv_self_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        std::fprintf(stderr, "qwen3_asr kv_cache: buffer alloc failed\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx = n_ctx;
    cache.n     = 0;
    cache.head  = 0;
    return true;
}

namespace {

constexpr const char k_default_variant[] = "qwen3-asr";

transcribe_status load(
    Loader &                         loader,
    const transcribe_model_params *  params,
    transcribe_model **              out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<QwenAsrModel>();
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

    // Chat template (optional, but required at run time — we read it
    // here so the absence surfaces at load rather than mid-decode).
    (void)read_optional_string_kv(
        loader.gguf(), "tokenizer.chat_template", "qwen3_asr",
        "", m->chat_template);

    // Hparams.
    if (const transcribe_status st = read_qwen3_asr_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK) return st;

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        std::fprintf(stderr,
                     "qwen3_asr: tokenizer vocab (%d) != decoder vocab_size (%d)\n",
                     m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        std::fprintf(stderr,
                     "qwen3_asr: GGUF tokenizer has no eos_token_id\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Mel frontend (Whisper-style 128-bin log-mel at 16 kHz, 30 s max).
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
        cfg.normalize    = m->hparams.fe_normalize;  // "per_utterance" (Whisper)

        // Optional filterbank + window buffers baked by the converter
        // from librosa / Whisper. If present, MelFrontend uses them
        // instead of reconstructing from hparams. Removes filterbank/
        // window as a variable during numerical bring-up. Absent falls
        // back to the in-code builders.
        {
            using R = transcribe::load_common::ReadF32Result;
            const size_t fb_elems =
                static_cast<size_t>(cfg.num_mels) *
                static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto fb_rc = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.mel_filterbank", fb_elems,
                "qwen3_asr", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto win_rc = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(),
                "frontend.window", win_elems,
                "qwen3_asr", cfg.window);
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

    if (const transcribe_status st = build_qwen3_asr_weights(
            gguf_data, m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // Backend plan.
    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, "qwen3_asr", m->plan);
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
                     "qwen3_asr: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "qwen3_asr");
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

    auto cc = std::make_unique<QwenAsrContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    cc->decoder_use_flash = true;
    // No encoder-side flash to toggle (bidirectional encoder is
    // graph-built without a flash_attn_ext call in the first pass).
    bool encoder_noop = false;
    transcribe::flash::apply_env_overrides(encoder_noop, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Qwen3 tokenizer special-token ids that aren't covered by the
// standard GGUF bos/eos slots. Hardcoded because we don't yet ship a
// BPE encoder in C++; the chat template is short and fixed across the
// 0.6B and 1.7B variants.
struct ChatTokens {
    int32_t im_start       = 151644;
    int32_t im_end         = 151645;
    int32_t newline        = 198;
    int32_t role_system    = 8948;
    int32_t role_user      = 872;
    int32_t role_assistant = 77091;
};

// Build the prompt token sequence + audio-position list for a single
// utterance. Mirrors the Qwen3-ASR chat template at the token level:
//
//   <|im_start|>system\n<|im_end|>\n
//   <|im_start|>user\n<|audio_start|><|audio_pad|>*T_enc<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n
//
// (with empty system content; language/context hinting is not yet
// implemented in the first port.) Returns total prompt length.
void build_prompt_tokens(const QwenAsrHParams &  hp,
                         int                     T_enc,
                         std::vector<int32_t> &  out_ids,
                         std::vector<int64_t> &  out_audio_positions)
{
    const ChatTokens ct {};
    out_ids.clear();
    out_audio_positions.clear();

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_system);
    out_ids.push_back(ct.newline);
    out_ids.push_back(ct.im_end);
    out_ids.push_back(ct.newline);

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_user);
    out_ids.push_back(ct.newline);

    out_ids.push_back(hp.audio_start_token_id);
    const int64_t audio_start_pos = static_cast<int64_t>(out_ids.size());
    for (int i = 0; i < T_enc; ++i) {
        out_ids.push_back(hp.audio_token_id);
        out_audio_positions.push_back(audio_start_pos + i);
    }
    out_ids.push_back(hp.audio_end_token_id);

    out_ids.push_back(ct.im_end);
    out_ids.push_back(ct.newline);

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_assistant);
    out_ids.push_back(ct.newline);
}

// GPT2/Qwen2 byte-level BPE "bytes_to_unicode" map (Python reference
// at transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode).
// Built once at first call; thread-safe via std::call_once.
static std::string byte_from_unicode_index(int cp) {
    // Inverse of bytes_to_unicode: return single-byte string or empty.
    static std::unordered_map<int, uint8_t> * g_map = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        auto * m = new std::unordered_map<int, uint8_t>();
        std::vector<int> bs;
        for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
        for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
        for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (int v : bs) if (v == b) { found = true; break; }
            if (!found) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            (*m)[cs[i]] = static_cast<uint8_t>(bs[i]);
        }
        g_map = m;
    });
    auto it = g_map->find(cp);
    if (it == g_map->end()) return {};
    return std::string(1, static_cast<char>(it->second));
}

// Decode a UTF-8 token string (GPT2 byte-level encoded) back to the
// raw UTF-8 bytes the tokenizer represents. Iterates codepoints and
// looks up each in the byte-to-unicode inverse.
static std::string gpt2_token_to_bytes(const std::string & tok) {
    std::string out;
    out.reserve(tok.size());
    size_t i = 0;
    while (i < tok.size()) {
        // Decode one UTF-8 codepoint from tok[i..].
        const unsigned char c = static_cast<unsigned char>(tok[i]);
        int cp = 0; int w = 1;
        if (c < 0x80) { cp = c; w = 1; }
        else if ((c & 0xE0) == 0xC0) {
            cp = ((c & 0x1F) << 6) |
                 (static_cast<unsigned char>(tok[i + 1]) & 0x3F);
            w = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = ((c & 0x0F) << 12) |
                 ((static_cast<unsigned char>(tok[i + 1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(tok[i + 2]) & 0x3F);
            w = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = ((c & 0x07) << 18) |
                 ((static_cast<unsigned char>(tok[i + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(tok[i + 2]) & 0x3F) <<  6) |
                 (static_cast<unsigned char>(tok[i + 3]) & 0x3F);
            w = 4;
        } else { ++i; continue; }  // malformed; skip

        std::string b = byte_from_unicode_index(cp);
        if (!b.empty()) out.append(b);
        i += w;
    }
    return out;
}

// Host-side pack [n_mels, T_mel] mel into batched chunks
// [mel_per_chunk, n_mels, 1, n_chunks]. Chunks shorter than
// mel_per_chunk are zero-padded.
void pack_mel_chunks(const float *         mel,       // [n_mels, T_mel]
                     int                   n_mels,
                     int                   n_mel_frames,
                     const EncoderTiming & t,
                     std::vector<float> &  out)
{
    const size_t per_chunk_elems = static_cast<size_t>(t.mel_per_chunk) * n_mels;
    out.assign(per_chunk_elems * t.n_chunks, 0.0f);

    for (int c = 0; c < t.n_chunks; ++c) {
        const int tail = (c == t.n_chunks - 1)
                       ? t.last_chunk_real_mel : t.mel_per_chunk;
        for (int m = 0; m < n_mels; ++m) {
            const float * src = mel
                + static_cast<size_t>(m) * n_mel_frames
                + static_cast<size_t>(c) * t.mel_per_chunk;
            float * dst = out.data()
                + static_cast<size_t>(c) * per_chunk_elems
                + static_cast<size_t>(m) * t.mel_per_chunk;
            std::memcpy(dst, src, tail * sizeof(float));
            // trailing frames in the tail chunk stay 0.0 (zero pad).
        }
    }
}

transcribe_status run(
    transcribe_context *      ctx,
    const float *             pcm,
    int                       n_samples,
    const transcribe_params * /*params*/)
{
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<QwenAsrContext *>(ctx);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    transcribe::debug::init();

    // ----- Mel front-end -------------------------------------------
    if (!cm->mel.has_value()) {
        std::fprintf(stderr,
                     "qwen3_asr run: model has no MelFrontend\n");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t t_mel_start = ggml_time_us();
    int mel_n_mels   = 0;
    int mel_n_frames = 0;
    if (const transcribe_status mst = cm->mel->compute(
            pcm, static_cast<size_t>(n_samples),
            cc->mel_buf, mel_n_mels, mel_n_frames,
            cc->n_threads);
        mst != TRANSCRIBE_OK)
    {
        std::fprintf(stderr,
                     "qwen3_asr run: MelFrontend::compute failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // Dump the post-frontend mel in the reference's contract shape
    // [n_mels, T_mel]. The batched graph input is a reshaped view of
    // the same data, so the comparison point lives on the host.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { mel_n_mels, mel_n_frames };
        transcribe::debug::dump_host_f32(
            "enc.mel.in", cc->mel_buf.data(),
            static_cast<long long>(cc->mel_buf.size()),
            shape, 2, "frontend.mel.norm");
    }

    // ----- Compute encoder timing + reject unsupported shapes ------
    EncoderTiming timing = compute_encoder_timing(mel_n_frames, cm->hparams);
    if (timing.n_chunks <= 0) {
        std::fprintf(stderr,
                     "qwen3_asr run: encoder timing is degenerate "
                     "(n_mel_frames=%d)\n", mel_n_frames);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Reset per-call compute state ----------------------------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }

    {
        ggml_init_params ip {};
        ip.mem_size   = 16 * 1024 * 1024;  // 18 blocks + subsample + head
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr,
                         "qwen3_asr run: ggml_init for compute_ctx failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ----- Build encoder graph -------------------------------------
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights,
                                          cm->hparams, timing);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Allocate + compute encoder graph ------------------------
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, false, true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr,
                         "qwen3_asr run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "qwen3_asr run: sched_alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Pack + upload mel.
    std::vector<float> mel_batched;
    pack_mel_chunks(cc->mel_buf.data(), mel_n_mels, mel_n_frames,
                    timing, mel_batched);
    ggml_backend_tensor_set(eb.mel_in, mel_batched.data(),
                            0, mel_batched.size() * sizeof(float));

    // Positional embedding.
    {
        std::vector<float> pe = build_sinusoid_pe(
            cm->hparams.enc_d_model, timing.per_chunk_aftercnn);
        ggml_backend_tensor_set(eb.pos_emb_in, pe.data(),
                                0, pe.size() * sizeof(float));
    }

    // Attention mask (block-diagonal from cu_seqlens).
    {
        std::vector<float> mask = build_cu_seqlens_mask(timing, cm->hparams);
        ggml_backend_tensor_set(eb.mask_in, mask.data(),
                                0, mask.size() * sizeof(float));
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

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "qwen3_asr run: encoder graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // Dump encoder intermediates.
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) {
            transcribe::debug::dump_tensor(name, t, stage);
        }
    };
    try_dump("enc.subsample.out",  eb.dumps.subsample_out,  "enc.subsample");
    try_dump("enc.pos_add.out",    eb.dumps.pos_add_out,    "enc.pos_add");
    try_dump("enc.block.0.out",    eb.dumps.block_0_out,    "enc.block.0");
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out",
                      cm->hparams.enc_n_layers - 1);
        try_dump(bname, eb.dumps.block_last_out, "enc.block.last");
    }
    try_dump("enc.ln_post.out",    eb.dumps.ln_post_out,    "enc.ln_post");
    try_dump("enc.proj.out",       eb.dumps.proj_out,       "enc.proj");

    // Read encoder output to host for use by the LM prefill.
    // NOTE: ne[1] is T_enc_padded (n_chunks * per_chunk_aftercnn); for
    // multi-chunk audio with a ragged tail this slightly overshoots
    // the reference's ragged T_enc. The padded rows carry near-zero
    // activations (from zero-padded mel tails) and the LM tolerates
    // the extra audio_pad tokens without derailing the transcript.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    cc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ----- KV cache init -----
    // Size for a generous default context; the step loop caps
    // max_new_tokens so the cache is never touched past T_prompt +
    // max_new_tokens. 2048 * 28 * 8 * 128 * 2 bytes (f16) = ~117 MiB.
    const int kv_n_ctx = 2048;
    if (cc->kv_cache.ctx == nullptr) {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) kv_type = GGML_TYPE_F32;
        if (!kv_cache_init(cc->kv_cache, cm->plan.primary,
                           kv_n_ctx,
                           cm->hparams.dec_n_kv_heads,
                           cm->hparams.dec_head_dim,
                           cm->hparams.dec_n_layers,
                           kv_type))
        {
            std::fprintf(stderr, "qwen3_asr run: kv_cache_init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    } else {
        // Clear stale positions for a fresh prefill.
        if (cc->kv_cache.buffer != nullptr) {
            ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
        }
        cc->kv_cache.n    = 0;
        cc->kv_cache.head = 0;
    }

    // ----- Prompt construction -----
    std::vector<int32_t> prompt_ids;
    std::vector<int64_t> audio_positions;
    build_prompt_tokens(cm->hparams, T_enc, prompt_ids, audio_positions);
    const int T_prompt   = static_cast<int>(prompt_ids.size());
    const int prefix_len = audio_positions.empty()
                         ? 0 : static_cast<int>(audio_positions.front());
    const int suffix_len = T_prompt - prefix_len - T_enc;
    (void)audio_positions;  // no longer fed to the graph
    if (T_prompt > kv_n_ctx) {
        std::fprintf(stderr,
                     "qwen3_asr run: prompt len %d exceeds kv_n_ctx %d\n",
                     T_prompt, kv_n_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Prefill graph -----
    PrefillBuild pb = build_prefill_graph(
        cc->compute_ctx, cm->weights, cm->hparams,
        cc->kv_cache, T_prompt, T_enc, prefix_len, suffix_len);
    if (pb.graph == nullptr || pb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Allocate + compute prefill on the same scheduler.
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        std::fprintf(stderr,
                     "qwen3_asr run: sched_alloc_graph failed (prefill)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload prefill inputs.
    ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(),
                            0, prompt_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pb.enc_out_in, cc->enc_host.data(),
                            0, cc->enc_host.size() * sizeof(float));

    {
        std::vector<int32_t> positions(T_prompt);
        for (int i = 0; i < T_prompt; ++i) positions[i] = i;
        ggml_backend_tensor_set(pb.positions_in, positions.data(),
                                0, positions.size() * sizeof(int32_t));
    }

    {
        // Causal mask: [T_prompt, T_prompt]; row r, col c: 0 if c <= r,
        // else finfo.min. Row-major buffer (ne[0]=T_prompt fastest).
        std::vector<float> mask(static_cast<size_t>(T_prompt) * T_prompt,
                                std::numeric_limits<float>::lowest());
        for (int r = 0; r < T_prompt; ++r) {
            for (int c = 0; c <= r; ++c) {
                mask[static_cast<size_t>(r) * T_prompt + c] = 0.0f;
            }
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(),
                                0, mask.size() * sizeof(float));
    }

    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, pb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "qwen3_asr run: prefill graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }

    cc->kv_cache.n    = T_prompt;
    cc->kv_cache.head = T_prompt;

    // Dump dec.* intermediates.
    try_dump("dec.token_emb",       pb.dumps.token_emb,       "dec.token_emb");
    try_dump("dec.audio_injected",  pb.dumps.audio_injected,  "dec.audio_injected");
    try_dump("dec.block.0.out",     pb.dumps.block_0_out,     "dec.block.0");
    {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out",
                      cm->hparams.dec_n_layers - 1);
        try_dump(nm,                pb.dumps.block_last_out,  "dec.block.last");
    }
    try_dump("dec.out_before_head", pb.dumps.out_before_head, "dec.out_before_head");
    try_dump("dec.logits_raw",      pb.dumps.logits_raw,      "dec.logits_raw");

    // ----- Decode phase (prefill logits + step loop) -----
    const int64_t t_dec_start = ggml_time_us();

    // ----- Read prefill logits + first argmax -----
    const int vocab = cm->hparams.dec_vocab_size;
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

    // ----- Step loop -----
    const int32_t eos_id = cm->hparams.eos_token_id;
    const int32_t max_new = 256;  // matches reference dumper default
    int cur_past = T_prompt;

    // Reserve the scheduler once against a worst-case single-token
    // step graph at n_past = kv_n_ctx - 1. Subsequent sched_alloc_graph
    // calls in the loop reuse the already-reserved allocation rather
    // than growing internal buffers for each new step size.
    {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        ggml_init_params ip {};
        ip.mem_size   = 8 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx != nullptr) {
            StepBuild sb_reserve = build_step_graph(
                cc->compute_ctx, cm->weights, cm->hparams,
                cc->kv_cache, /*n_past=*/kv_n_ctx - 1);
            if (sb_reserve.graph != nullptr) {
                ggml_backend_sched_reserve(cc->sched, sb_reserve.graph);
            }
        }
    }

    while (next_tok != eos_id &&
           static_cast<int32_t>(generated_ids.size()) < max_new &&
           cur_past + 1 <= kv_n_ctx)
    {
        // Fresh compute_ctx per step: step graphs are small, freeing
        // and re-initing keeps memory low without scheduler plumbing.
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
            if (cc->compute_ctx == nullptr) {
                std::fprintf(stderr,
                             "qwen3_asr step: ggml_init failed\n");
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights,
                                        cm->hparams, cc->kv_cache, cur_past);
        if (sb.graph == nullptr || sb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
            std::fprintf(stderr,
                         "qwen3_asr step: sched_alloc_graph failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_tensor_set(sb.input_id_in, &next_tok,
                                0, sizeof(int32_t));
        const int32_t pos_val = cur_past;
        ggml_backend_tensor_set(sb.position_in, &pos_val,
                                0, sizeof(int32_t));

        // Causal mask for the step: a single query row attending to
        // all n_past + 1 keys (all zeros — no masking needed since
        // this is the newest token).
        {
            std::vector<float> sm(cur_past + 1, 0.0f);
            ggml_backend_tensor_set(sb.mask_in, sm.data(),
                                    0, sm.size() * sizeof(float));
        }

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, sb.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "qwen3_asr step: graph compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }

        // GPU argmax: 4-byte readback instead of vocab_size*4.
        int32_t argmax_tok = 0;
        ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
        next_tok = argmax_tok;
        generated_ids.push_back(next_tok);
        cur_past += 1;
        cc->kv_cache.n    = cur_past + 1;
        cc->kv_cache.head = cur_past + 1;
    }

    // Strip trailing EOS if present (match the reference transcript).
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    // Decode generated ids to text via GPT2 byte-level BPE inverse.
    std::string raw_text;
    for (int32_t id : generated_ids) {
        const std::string & s = cm->tok.token(id);
        if (s.empty()) continue;
        raw_text += gpt2_token_to_bytes(s);
    }

    // Parse Qwen3-ASR output format: "language X<asr_text>actual_text"
    // The <asr_text> separator marks the transcript; everything before
    // it is the language tag.
    std::string transcript_text = raw_text;
    if (auto sep = raw_text.find("<asr_text>"); sep != std::string::npos) {
        transcript_text = raw_text.substr(sep + std::strlen("<asr_text>"));
    }

    // Write full_text + a single segment (no timestamps; Qwen3-ASR's
    // ASR head emits TIMESTAMPS_NONE per the family capability).
    cc->t_decode_us = ggml_time_us() - t_dec_start;

    cc->full_text = transcript_text;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;
    transcribe_context::SegmentEntry seg {};
    seg.text  = transcript_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000
              / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name         = */ "qwen3_asr",
    /* .load         = */ load,
    /* .init_context = */ init_context,
    /* .run          = */ run,
};

} // namespace transcribe::qwen3_asr
