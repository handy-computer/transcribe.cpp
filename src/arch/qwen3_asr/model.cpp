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

#include "qwen3_lm/qwen3_lm.h"
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
#include <cmath>
#include <limits>
#include <memory>
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

constexpr const char k_default_variant[] = "qwen3-asr";

// Forward declarations for helpers defined further down in this file.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok,
                                      ChatTokens &                  out);

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

    // Resolve the chat-template special-token ids at load so a vocab
    // drift (e.g. a future fine-tune that renames or reorders the
    // role tokens) surfaces here instead of silently producing a
    // wrong prompt at decode time.
    if (const transcribe_status st = resolve_chat_tokens(m->tok, m->chat_tokens);
        st != TRANSCRIBE_OK) return st;

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

    // Pack gate+up into a separate ctx + backend buffer so the FFN
    // can run a single mul_mat instead of two. ctx_meta is sized
    // exactly for GGUF file tensors with no headroom, so packed
    // tensors live in their own context owned by `qwen3_lm::pack_gate_up`.
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
                "qwen3_asr"))
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

    auto cc = std::make_unique<QwenAsrContext>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    cc->encoder_use_flash = false;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    // Pre-allocate KV cache at context creation so the first run
    // doesn't pay the allocation cost inside the decode phase.
    auto * cm = static_cast<QwenAsrModel *>(model);
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
            std::fprintf(stderr, "qwen3_asr init_context: kv_init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Resolve the narrow set of chat-template piece strings against the
// loaded tokenizer, filling `out`. Hard-fails with TRANSCRIBE_ERR_GGUF
// if any piece is missing — a future checkpoint that reorders the
// vocab will surface here at load time, before any prompt is built.
//
// The newline piece is stored under its GPT-2 byte-level-encoded form
// (\n = 0x0A → Unicode codepoint U+010A "Ċ", UTF-8 \xC4\x8A), same as
// every other non-printable byte in the Qwen3 vocab. The role names
// and the two <|im_*|> special tokens are stored verbatim.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok,
                                      ChatTokens &                  out)
{
    struct PieceSlot {
        const char * piece;
        int32_t *    slot;
    };
    const PieceSlot pieces[] = {
        { "<|im_start|>",  &out.im_start       },
        { "<|im_end|>",    &out.im_end         },
        { "\xC4\x8A",      &out.newline        },   // "Ċ" = byte-level \n
        { "system",        &out.role_system    },
        { "user",          &out.role_user      },
        { "assistant",     &out.role_assistant },
    };
    for (const auto & p : pieces) {
        const int id = tok.find(p.piece);
        if (id < 0) {
            std::fprintf(stderr,
                         "qwen3_asr: chat-template piece \"%s\" not in tokenizer\n",
                         p.piece);
            return TRANSCRIBE_ERR_GGUF;
        }
        *p.slot = id;
    }
    return TRANSCRIBE_OK;
}

// Build the prompt token sequence + audio-position list for a single
// utterance. Mirrors the Qwen3-ASR chat template at the token level:
//
//   <|im_start|>system\n<|im_end|>\n
//   <|im_start|>user\n<|audio_start|><|audio_pad|>*T_enc<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n[language {Name}<asr_text>]?
//
// The system prompt is empty (we do not yet thread a caller-supplied
// context through). When `lang_prefix_ids` is non-null its contents
// are appended verbatim after the trailing newline, which is how the
// reference forces a specific output language: the LM continues from
// "<asr_text>" with pure transcript text rather than emitting its own
// "language X<asr_text>" prefix. Callers resolve the prefix via
// encode_language_prefix() below; it stays out of this function so
// build_prompt_tokens can remain a pure token-id assembler.
void build_prompt_tokens(const QwenAsrHParams &           hp,
                         const ChatTokens &               ct,
                         int                              T_enc,
                         const std::vector<int32_t> *     lang_prefix_ids,
                         std::vector<int32_t> &           out_ids,
                         std::vector<int64_t> &           out_audio_positions)
{
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

    if (lang_prefix_ids != nullptr && !lang_prefix_ids->empty()) {
        out_ids.insert(out_ids.end(),
                       lang_prefix_ids->begin(),
                       lang_prefix_ids->end());
    }
}

} // namespace  (close anon temporarily; the two helpers below have
  // external linkage — one because the public-header declaration in
  // qwen3_asr.h needs a matching definition, the other because the
  // function lives just above and wants to name the data table.)

// BCP-47 → publisher canonical name. The Qwen3-ASR prompt renders the
// publisher's canonical string ("English", "Chinese", ...) rather
// than the BCP-47 code, and the list is frozen per Qwen3-ASR release
// (qwen_asr.inference.utils.SUPPORTED_LANGUAGES in the publisher's
// Python package). Keeping the map here tracks the publisher's list
// directly; if a future Qwen3-ASR variant changes it, this table is
// the update point and the model will error cleanly for languages
// that haven't been added yet.
struct LangNameEntry {
    const char * bcp47;
    const char * pub_name;
};
constexpr LangNameEntry k_qwen3_asr_language_names[] = {
    {"zh",  "Chinese"},
    {"en",  "English"},
    {"yue", "Cantonese"},
    {"ar",  "Arabic"},
    {"de",  "German"},
    {"fr",  "French"},
    {"es",  "Spanish"},
    {"pt",  "Portuguese"},
    {"id",  "Indonesian"},
    {"it",  "Italian"},
    {"ko",  "Korean"},
    {"ru",  "Russian"},
    {"th",  "Thai"},
    {"vi",  "Vietnamese"},
    {"ja",  "Japanese"},
    {"tr",  "Turkish"},
    {"hi",  "Hindi"},
    {"ms",  "Malay"},
    {"nl",  "Dutch"},
    {"sv",  "Swedish"},
    {"da",  "Danish"},
    {"fi",  "Finnish"},
    {"pl",  "Polish"},
    {"cs",  "Czech"},
    {"fil", "Filipino"},
    {"fa",  "Persian"},
    {"el",  "Greek"},
    {"ro",  "Romanian"},
    {"hu",  "Hungarian"},
    {"mk",  "Macedonian"},
};

// Resolve a caller-supplied BCP-47 language code to the token-id
// sequence the chat template expects: the BPE-encoded bytes of
// "language {Name}" followed by the `<asr_text>` special-token id.
//
// We split the prefix around the special token because
// Tokenizer::encode() does not partition special tokens out before
// BPE — it would run a byte-level merge loop over the literal
// "<asr_text>" string and produce wrong ids. The Hugging Face
// tokenizer's "added_tokens" path is what recognizes the tag and
// emits its single id; we replicate that by looking up the id
// directly from the vocab and appending it by hand. The id is
// checkpoint-specific (151704 on Qwen3-ASR 0.6B / 1.7B) but we never
// hardcode it — `tok.find("<asr_text>")` reads the actual vocab.
//
// The dispatcher validates `bcp47` against caps.languages before we
// get here, so an unknown code at this layer is either a converter
// drift (language written into caps that our static map doesn't
// know about) or a future Qwen3-ASR variant we haven't caught up to.
// Either way, surface as UNSUPPORTED_LANGUAGE so the caller sees the
// right error.
transcribe_status encode_language_prefix(const transcribe::Tokenizer & tok,
                                         const char *                  bcp47,
                                         std::vector<int32_t> &        out_ids)
{
    out_ids.clear();
    if (bcp47 == nullptr || bcp47[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const char * pub_name = nullptr;
    for (const auto & e : k_qwen3_asr_language_names) {
        if (std::strcmp(e.bcp47, bcp47) == 0) {
            pub_name = e.pub_name;
            break;
        }
    }
    if (pub_name == nullptr) {
        std::fprintf(stderr,
                     "qwen3_asr: no canonical publisher name for "
                     "language=\"%s\"\n", bcp47);
        return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
    }
    if (!tok.has_encoder()) {
        std::fprintf(stderr,
                     "qwen3_asr: tokenizer missing encoder (merges "
                     "unavailable); cannot render language hint\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int asr_text_id = tok.find("<asr_text>");
    if (asr_text_id < 0) {
        std::fprintf(stderr,
                     "qwen3_asr: tokenizer vocab missing <asr_text> "
                     "special token\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    std::string text = "language ";
    text += pub_name;
    if (const transcribe_status st = tok.encode(text, out_ids);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    out_ids.push_back(asr_text_id);
    return TRANSCRIBE_OK;
}

namespace {  // reopen anon for the rest of the file's helpers.

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
    const transcribe_params * params)
{
    if (ctx == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<QwenAsrContext *>(ctx);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-run abort check. Qwen3-ASR is single-shot today; this is
    // the single observation point. Stage 2's long-form loop will add
    // per-chunk polling for Whisper; the other autoregressive families
    // can wire per-step polling when they grow their own long-form
    // paths.
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    // Language hint handling. Null / empty == auto-detect (the LM
    // emits its own "language X<asr_text>" prefix which the output
    // parser below strips). A non-null code is resolved to the
    // token-id sequence for "language {Name}<asr_text>" via the
    // tokenizer; we seed the assistant turn with those tokens so the
    // LM continues straight into pure transcript text. Dispatcher
    // already validated `params->language` against caps.languages,
    // so if encode_language_prefix reports "no canonical name" the
    // static publisher map and the converter's BCP-47 list have
    // drifted — surface that as UNSUPPORTED_LANGUAGE instead of
    // silently falling back to auto-detect.
    std::vector<int32_t> lang_prefix_ids;
    const std::vector<int32_t> * lang_prefix_ptr = nullptr;
    if (params != nullptr && params->language != nullptr &&
        params->language[0] != '\0')
    {
        if (const transcribe_status st = encode_language_prefix(
                cm->tok, params->language, lang_prefix_ids);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        lang_prefix_ptr = &lang_prefix_ids;
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
    // Phase timers. Internal diagnostic: the public
    // transcribe_timings only exposes mel/encode/decode, but a lot of
    // per-run cost sits in graph build, scheduler alloc, host uploads,
    // and prefill compute. These locals break that out so we can print
    // a full breakdown under TRANSCRIBE_PERF_DEBUG without touching
    // the public API.
    const int64_t t_enc_build_start = ggml_time_us();
    int64_t t_enc_build_us       = 0;
    int64_t t_enc_d2h_us         = 0;
    int64_t t_prefill_build_us   = 0;
    int64_t t_prefill_compute_us = 0;
    int64_t t_prefill_logits_us  = 0;
    int64_t t_step_loop_us       = 0;
    int     n_steps              = 0;

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
                                          cm->hparams, timing,
                                          cc->encoder_use_flash);
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
        if (cc->encoder_use_flash) {
            std::vector<ggml_fp16_t> mask_f16(mask.size());
            for (size_t i = 0; i < mask.size(); ++i)
                mask_f16[i] = ggml_fp32_to_fp16(mask[i]);
            ggml_backend_tensor_set(eb.mask_in, mask_f16.data(),
                                    0, mask_f16.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(eb.mask_in, mask.data(),
                                    0, mask.size() * sizeof(float));
        }
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
    t_enc_build_us = t_enc_start - t_enc_build_start;
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

    // Read encoder output to host for use by the LM prefill. The graph
    // already dropped the aftercnn pad rows before the encoder blocks
    // (see encoder.cpp), so eb.out is exactly [d_enc, T_enc] — matching
    // the reference's `padded_embed[padded_mask_after_cnn]`-selected
    // shape.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    cc->enc_host.resize(static_cast<size_t>(d_enc) *
                        static_cast<size_t>(T_enc));
    const int64_t t_d2h_start = ggml_time_us();
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));
    t_enc_d2h_us = ggml_time_us() - t_d2h_start;

    // ----- KV cache init -----
    // Size for a generous default context; the step loop caps
    // max_new_tokens so the cache is never touched past T_prompt +
    // max_new_tokens. 2048 * 28 * 8 * 128 * 2 bytes (f16) = ~117 MiB.
    //
    // t_dec_start covers the whole decode phase (KV init + prompt +
    // prefill build/compute + step loop). Earlier versions started it
    // after prefill compute, which bucketed prefill into the wall/phase
    // gap. Prefill is part of "decode" as users understand it.
    const int64_t t_dec_start = ggml_time_us();
    const int64_t t_prefill_build_start = t_dec_start;
    const int kv_n_ctx = 2048;
    if (cc->kv_cache.ctx == nullptr) {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) kv_type = GGML_TYPE_F32;
        if (!transcribe::qwen3_lm::kv_init(cc->kv_cache, cm->plan.primary,
                                           kv_n_ctx,
                                           cm->hparams.dec_n_kv_heads,
                                           cm->hparams.dec_head_dim,
                                           cm->hparams.dec_n_layers,
                                           kv_type))
        {
            std::fprintf(stderr, "qwen3_asr run: kv_init failed\n");
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
    build_prompt_tokens(cm->hparams, cm->chat_tokens, T_enc,
                        lang_prefix_ptr, prompt_ids, audio_positions);
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
    // slice_last: when false, the last block's FFN + final norm run
    // on every position (needed for debug dump parity). When true, we
    // slice to just the final position before the last FFN — same
    // optimization llama.cpp's inp_out_ids gives, worth ~25 ms here.
    const bool dumps_on  = transcribe::debug::enabled();
    const bool slice_last = !dumps_on;
    PrefillBuild pb = build_prefill_graph(
        cc->compute_ctx, cm->weights, cm->hparams,
        cc->kv_cache, T_prompt, T_enc, prefix_len, suffix_len,
        /*use_flash=*/cc->decoder_use_flash, slice_last);
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
        // Causal mask in F16 (matches pb.mask_in's type). Row r, col c:
        // +0 if c <= r, else -inf. Row-major (ne[0]=T_prompt fastest).
        // F16 direct upload avoids a per-layer ggml_cast in the graph.
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

    const int64_t t_prefill_compute_start = ggml_time_us();
    t_prefill_build_us = t_prefill_compute_start - t_prefill_build_start;
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, pb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "qwen3_asr run: prefill graph compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    t_prefill_compute_us = ggml_time_us() - t_prefill_compute_start;

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

    // ----- Read prefill logits + first argmax -----
    const int64_t t_prefill_logits_start = ggml_time_us();
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
    t_prefill_logits_us = ggml_time_us() - t_prefill_logits_start;

    // ----- Step loop -----
    const int32_t eos_id = cm->hparams.eos_token_id;
    const int32_t max_new = 256;  // matches reference dumper default
    int cur_past = T_prompt;

    // ---------- Build step graph ONCE and reuse every step ----------
    // Static shape sized for the actual workload: T_prompt audio/prompt
    // tokens already written + up to max_new generated tokens.
    // Metal's flash-attn kernels dispatch much faster (~30% on M4 Max)
    // when the K/V ne[1] is a power of 2. Round up accordingly, with a
    // floor of 1024 since smaller values just move us into the "slow
    // misaligned" branch without saving meaningful bandwidth.
    int max_n_kv = 1024;
    while (max_n_kv < T_prompt + max_new) max_n_kv *= 2;
    if (max_n_kv > kv_n_ctx) max_n_kv = kv_n_ctx;
    const int64_t t_step_build_start = ggml_time_us();
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
                                    cm->hparams, cc->kv_cache, max_n_kv,
                                    /*use_flash=*/cc->decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
        std::fprintf(stderr,
                     "qwen3_asr step: sched_alloc_graph failed (build-once)\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int64_t t_step_build_once_us = ggml_time_us() - t_step_build_start;

    // Mask buffer: reused host-side across steps. Starts all -inf;
    // per step we zero positions [0, cur_past] (attend) and leave
    // [cur_past+1, max_n_kv) as -inf (ignore).
    const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);

    // Per-step timers. Now much smaller — no per-step graph work.
    int64_t t_step_set_us  = 0;
    int64_t t_step_comp_us = 0;
    int64_t t_step_get_us  = 0;
    const int64_t t_step_loop_start = ggml_time_us();
    while (next_tok != eos_id &&
           static_cast<int32_t>(generated_ids.size()) < max_new &&
           cur_past + 1 <= max_n_kv)
    {
        const int64_t t_set0 = ggml_time_us();

        ggml_backend_tensor_set(sb.input_id_in, &next_tok,
                                0, sizeof(int32_t));
        const int32_t pos_val = cur_past;
        ggml_backend_tensor_set(sb.position_in, &pos_val,
                                0, sizeof(int32_t));
        const int64_t kv_idx_val = cur_past;
        ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val,
                                0, sizeof(int64_t));

        // Mask: mark the newly-added position as attendable. Positions
        // [0, cur_past) were zeroed in prior iterations; just set the
        // new one. (On iter 0, zero everything in [0, cur_past].)
        if (cur_past == T_prompt) {
            std::fill(step_mask.begin(),
                      step_mask.begin() + cur_past + 1, mask_zero);
        } else {
            step_mask[cur_past] = mask_zero;
        }
        ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                static_cast<size_t>(max_n_kv) *
                                sizeof(ggml_fp16_t));

        const int64_t t_set1 = ggml_time_us();
        t_step_set_us += t_set1 - t_set0;

        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, sb.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "qwen3_asr step: graph compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        const int64_t t_comp1 = ggml_time_us();
        t_step_comp_us += t_comp1 - t_set1;

        int32_t argmax_tok = 0;
        ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
        next_tok = argmax_tok;
        generated_ids.push_back(next_tok);
        cur_past += 1;
        cc->kv_cache.n    = cur_past + 1;
        cc->kv_cache.head = cur_past + 1;
        t_step_get_us += ggml_time_us() - t_comp1;
    }
    t_step_loop_us = ggml_time_us() - t_step_loop_start;
    n_steps = static_cast<int>(generated_ids.size()) - 1;

    // Map the granular diagnostic counters to the shape the debug
    // print expects. With graph reuse all per-step overhead collapses
    // to tensor_set; build/alloc/ctx_reset are amortized in the
    // t_step_build_once_us bucket printed separately.
    int64_t t_step_ctx_us   = 0;
    int64_t t_step_build_us = t_step_build_once_us;
    int64_t t_step_alloc_us = 0;

    // Strip trailing EOS if present (match the reference transcript).
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    // Decode generated ids to text. Tokenizer::decode handles the
    // "gpt2" byte-level inversion natively — no more per-family
    // byte-to-unicode walker.
    std::string raw_text = cm->tok.decode(
        generated_ids.data(), static_cast<int>(generated_ids.size()));

    // Parse Qwen3-ASR output format:
    //   auto-detect: "language X<asr_text>actual_text"  (strip prefix)
    //   forced:      "actual_text"                      (no prefix -
    //                we seeded the assistant turn with "language
    //                X<asr_text>" already, so the LM's generated
    //                continuation is pure transcript.)
    //
    // We unconditionally try the split — if the prefix is absent (the
    // forced case, or a model that just didn't emit it this run), the
    // find() returns npos and transcript_text stays as raw_text.
    std::string transcript_text = raw_text;
    if (auto sep = raw_text.find("<asr_text>"); sep != std::string::npos) {
        // Auto-detect path: the prefix carries the language name the
        // model picked. Surface it as detected_language (reverse-map
        // the human-readable name to its BCP-47 code via the same
        // table encode_language_prefix uses), but only when the caller
        // did NOT supply a hint — the public field's contract is "what
        // the model told us," not "what we told the model."
        if (lang_prefix_ptr == nullptr) {
            constexpr const char k_prefix[] = "language ";
            const size_t name_start = raw_text.find(k_prefix);
            if (name_start != std::string::npos && name_start < sep) {
                const size_t ns = name_start + (sizeof(k_prefix) - 1);
                std::string name = raw_text.substr(ns, sep - ns);
                while (!name.empty() && (name.back() == ' ' ||
                       name.back() == '\t' || name.back() == '\n'))
                {
                    name.pop_back();
                }
                for (const auto & e : k_qwen3_asr_language_names) {
                    if (name == e.pub_name) {
                        cc->detected_language = e.bcp47;
                        break;
                    }
                }
            }
        }
        transcript_text = raw_text.substr(sep + std::strlen("<asr_text>"));
    }

    // Write full_text + a single segment (no timestamps; Qwen3-ASR's
    // ASR head emits TIMESTAMPS_NONE per the family capability).
    cc->t_decode_us = ggml_time_us() - t_dec_start;

    // Optional perf breakdown. Public transcribe_timings only has
    // mel/encode/decode — this dumps the finer split we need to
    // reason about bench wall-vs-phase gaps without changing the
    // public API or the bench JSON schema. Gated on env var so it
    // doesn't spam normal runs.
    if (const char * e = std::getenv("TRANSCRIBE_PERF_DEBUG");
        e != nullptr && *e != '\0' && *e != '0')
    {
        const double ms = 1.0 / 1000.0;
        const double sum_ms = (cc->t_mel_us + t_enc_build_us +
                               cc->t_encode_us + t_enc_d2h_us +
                               t_prefill_build_us + t_prefill_compute_us +
                               t_prefill_logits_us + t_step_loop_us) * ms;
        const double per_step_ms = (n_steps > 0)
            ? (t_step_loop_us * ms / n_steps) : 0.0;
        std::fprintf(stderr,
            "qwen3_asr perf breakdown:\n"
            "  mel              %8.2f ms\n"
            "  enc_build        %8.2f ms  (graph + sched + uploads)\n"
            "  enc_compute      %8.2f ms\n"
            "  enc_d2h          %8.2f ms  (%d floats)\n"
            "  prefill_build    %8.2f ms  (kv_init + prompt + graph + sched + uploads)\n"
            "  prefill_compute  %8.2f ms  (T_prompt=%d)\n"
            "  prefill_logits   %8.2f ms  (readback + argmax, vocab=%d)\n"
            "  step_loop        %8.2f ms  (%d steps, %.2f ms/step)\n"
            "    ctx_reset    %8.2f ms  (%.3f ms/step)\n"
            "    graph_build  %8.2f ms  (%.3f ms/step)\n"
            "    sched_alloc  %8.2f ms  (%.3f ms/step)\n"
            "    tensor_set   %8.2f ms  (%.3f ms/step)\n"
            "    compute      %8.2f ms  (%.3f ms/step)\n"
            "    tensor_get   %8.2f ms  (%.3f ms/step)\n"
            "  ---\n"
            "  sum              %8.2f ms\n",
            cc->t_mel_us         * ms,
            t_enc_build_us       * ms,
            cc->t_encode_us      * ms,
            t_enc_d2h_us         * ms, static_cast<int>(cc->enc_host.size()),
            t_prefill_build_us   * ms,
            t_prefill_compute_us * ms, T_prompt,
            t_prefill_logits_us  * ms, vocab,
            t_step_loop_us       * ms, n_steps, per_step_ms,
            t_step_ctx_us   * ms, (n_steps > 0) ? (t_step_ctx_us   * ms / n_steps) : 0.0,
            t_step_build_us * ms, (n_steps > 0) ? (t_step_build_us * ms / n_steps) : 0.0,
            t_step_alloc_us * ms, (n_steps > 0) ? (t_step_alloc_us * ms / n_steps) : 0.0,
            t_step_set_us   * ms, (n_steps > 0) ? (t_step_set_us   * ms / n_steps) : 0.0,
            t_step_comp_us  * ms, (n_steps > 0) ? (t_step_comp_us  * ms / n_steps) : 0.0,
            t_step_get_us   * ms, (n_steps > 0) ? (t_step_get_us   * ms / n_steps) : 0.0,
            sum_ms);
    }

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
    /* .name            = */ "qwen3_asr",
    /* .load            = */ load,
    /* .init_context    = */ init_context,
    /* .run             = */ run,
    /* .stream_begin    = */ nullptr,
    /* .stream_feed     = */ nullptr,
    /* .stream_finalize = */ nullptr,
    /* .stream_reset    = */ nullptr,
};

} // namespace transcribe::qwen3_asr
