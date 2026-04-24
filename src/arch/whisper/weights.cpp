// arch/whisper/weights.cpp - implementation of read_whisper_hparams
// and build_whisper_weights.
//
// Pattern mirrors cohere/weights.cpp: read every required KV
// explicitly, validate cross-field invariants, then resolve each
// tensor slot against the GGUF with expected shape.
//
// Whisper-specific notes:
//
//   - Attention projections: q/v/out carry bias; k does NOT. The
//     tensor catalog reflects this — no "attn.k.bias" slot exists
//     (self-attn and cross-attn follow the same rule).
//
//   - Logits head: no separate lm_head weight or bias. The converter
//     does not emit dec.lm_head.weight; the decoder graph reuses
//     dec.token_embd.weight via ggml_mul_mat at the final step.
//
//   - suppress_tokens / begin_suppress_tokens are int32 arrays under
//     the stt.whisper.* namespace. Empty vectors are the Absent case
//     — the .en variants legitimately ship without these keys.
//
//   - The converter tags normalization as "whisper_logmel", not one of
//     the stat-based names (per_feature, global). This is the
//     per-utterance log10(max(mel, 1e-10)) → clamp(max-8) → (+4)/4
//     dynamic-range compression. Any future frontend shared-helper
//     must understand this tag.

#include "weights.h"

#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace transcribe::whisper {

namespace {

constexpr const char * kFamilyTag = "whisper";

// Optional u32 KV. Absent leaves `out` untouched; BadType is fatal.
transcribe_status read_optional_u32_kv(const gguf_context * gguf,
                                       const char *         key,
                                       const char *         error_tag,
                                       int32_t &            out)
{
    uint32_t value = 0;
    const auto st = transcribe::read_uint32_kv(gguf, key, value);
    if (st == transcribe::KvResult::Ok) {
        out = static_cast<int32_t>(value);
        return TRANSCRIBE_OK;
    }
    if (st == transcribe::KvResult::Absent) {
        return TRANSCRIBE_OK;
    }
    std::fprintf(stderr, "%s: KV %s has wrong type\n", error_tag, key);
    return TRANSCRIBE_ERR_GGUF;
}

// Optional int32 array KV. Absent leaves `out` untouched.
transcribe_status read_optional_i32_array_kv(const gguf_context * gguf,
                                             const char *         key,
                                             const char *         error_tag,
                                             std::vector<int32_t> & out)
{
    std::vector<int32_t> tmp;
    const auto st = transcribe::read_int32_array_kv(gguf, key, tmp);
    if (st == transcribe::KvResult::Ok) {
        out = std::move(tmp);
        return TRANSCRIBE_OK;
    }
    if (st == transcribe::KvResult::Absent) {
        return TRANSCRIBE_OK;
    }
    std::fprintf(stderr, "%s: KV %s has wrong type (expected int32 array)\n",
                 error_tag, key);
    return TRANSCRIBE_ERR_GGUF;
}

// Optional float32 KV.
transcribe_status read_optional_f32_kv(const gguf_context * gguf,
                                       const char *         key,
                                       const char *         error_tag,
                                       float                default_value,
                                       float &              out)
{
    float value = default_value;
    const auto st = transcribe::read_float32_kv(gguf, key, value);
    if (st == transcribe::KvResult::Ok) {
        out = value;
        return TRANSCRIBE_OK;
    }
    if (st == transcribe::KvResult::Absent) {
        out = default_value;
        return TRANSCRIBE_OK;
    }
    std::fprintf(stderr, "%s: KV %s has wrong type\n", error_tag, key);
    return TRANSCRIBE_ERR_GGUF;
}

} // namespace

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------

transcribe_status read_whisper_hparams(const gguf_context * gguf,
                                       WhisperHParams &     hp)
{
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Encoder.
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.encoder.n_layers",             kFamilyTag, hp.enc_n_layers);             st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.encoder.d_model",              kFamilyTag, hp.enc_d_model);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.encoder.n_heads",              kFamilyTag, hp.enc_n_heads);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.encoder.ffn_dim",              kFamilyTag, hp.enc_ffn_dim);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.encoder.num_mel_bins",         kFamilyTag, hp.enc_num_mel_bins);         st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.encoder.max_source_positions", kFamilyTag, hp.enc_max_source_positions); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.whisper.encoder.activation",        kFamilyTag, hp.enc_activation);           st != TRANSCRIBE_OK) return st;

    // Decoder.
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder.n_layers",             kFamilyTag, hp.dec_n_layers);             st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder.d_model",              kFamilyTag, hp.dec_d_model);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder.n_heads",              kFamilyTag, hp.dec_n_heads);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder.ffn_dim",              kFamilyTag, hp.dec_ffn_dim);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder.max_target_positions", kFamilyTag, hp.dec_max_target_positions); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder.vocab_size",           kFamilyTag, hp.dec_vocab_size);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.whisper.decoder.activation",        kFamilyTag, hp.dec_activation);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv(gguf, "stt.whisper.decoder.tie_word_embeddings", kFamilyTag, true, hp.dec_tie_word_embeddings); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv(gguf, "stt.whisper.decoder.scale_embedding",     kFamilyTag, false, hp.dec_scale_embedding);   st != TRANSCRIBE_OK) return st;

    // Whisper generation contract.
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.decoder_start_token_id", kFamilyTag, hp.decoder_start_token_id); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.whisper.no_timestamps_token_id", kFamilyTag, hp.no_timestamps_token_id); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv(gguf, "stt.whisper.sot_token_id",        kFamilyTag, hp.sot_token_id);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv(gguf, "stt.whisper.transcribe_token_id", kFamilyTag, hp.transcribe_token_id); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv(gguf, "stt.whisper.translate_token_id",  kFamilyTag, hp.translate_token_id);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv(gguf, "stt.whisper.prev_sot_token_id",   kFamilyTag, hp.prev_sot_token_id);   st != TRANSCRIBE_OK) return st;

    // sot falls back to decoder_start when unspecified — whisper-multilingual
    // uses the same id for both.
    if (hp.sot_token_id < 0) {
        hp.sot_token_id = hp.decoder_start_token_id;
    }

    if (auto st = read_optional_i32_array_kv(gguf, "stt.whisper.suppress_tokens",       kFamilyTag, hp.suppress_tokens);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_i32_array_kv(gguf, "stt.whisper.begin_suppress_tokens", kFamilyTag, hp.begin_suppress_tokens); st != TRANSCRIBE_OK) return st;

    // Frontend.
    if (auto st = read_required_string_kv(gguf, "stt.frontend.type",        kFamilyTag, hp.fe_type);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv   (gguf, "stt.frontend.num_mels",    kFamilyTag, hp.fe_num_mels);    st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv   (gguf, "stt.frontend.sample_rate", kFamilyTag, hp.fe_sample_rate); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv   (gguf, "stt.frontend.n_fft",       kFamilyTag, hp.fe_n_fft);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv   (gguf, "stt.frontend.win_length",  kFamilyTag, hp.fe_win_length);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv   (gguf, "stt.frontend.hop_length",  kFamilyTag, hp.fe_hop_length);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.frontend.window",      kFamilyTag, hp.fe_window);      st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.frontend.normalize",   kFamilyTag, hp.fe_normalize);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_f32_kv   (gguf, "stt.frontend.dither",       kFamilyTag, 0.0f, hp.fe_dither);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_f32_kv   (gguf, "stt.frontend.pre_emphasis", kFamilyTag, 0.0f, hp.fe_pre_emphasis); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_f32_kv   (gguf, "stt.frontend.f_min",        kFamilyTag, 0.0f, hp.fe_f_min);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_f32_kv   (gguf, "stt.frontend.f_max",        kFamilyTag, 8000.0f, hp.fe_f_max);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.pad_mode",     kFamilyTag, "reflect", hp.fe_pad_mode); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv  (gguf, "stt.frontend.center",       kFamilyTag, true, hp.fe_center);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.mel_norm",     kFamilyTag, "slaney", hp.fe_mel_norm);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv   (gguf, "stt.frontend.chunk_length", kFamilyTag, hp.fe_chunk_length);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv   (gguf, "stt.frontend.n_samples",    kFamilyTag, hp.fe_n_samples);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_u32_kv   (gguf, "stt.frontend.nb_max_frames", kFamilyTag, hp.fe_nb_max_frames);      st != TRANSCRIBE_OK) return st;

    // Capability flags. stt.capability.* is ecosystem-wide; the shared
    // read_capability_kv() (called later in the load path) re-reads
    // these into the public transcribe_capabilities struct, but we also
    // cache them here for easy access in decoder logic (supports_translate
    // gates the <|translate|> token availability etc).
    if (auto st = read_optional_bool_kv(gguf, "stt.capability.lang_detect", kFamilyTag, false, hp.cap_lang_detect); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv(gguf, "stt.capability.translate",   kFamilyTag, false, hp.cap_translate);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv(gguf, "stt.capability.timestamps",  kFamilyTag, false, hp.cap_timestamps);  st != TRANSCRIBE_OK) return st;

    // Cross-field invariants.
    if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 ||
        hp.enc_ffn_dim <= 0 || hp.enc_num_mel_bins <= 0 ||
        hp.enc_max_source_positions <= 0)
    {
        std::fprintf(stderr, "whisper: encoder hparams must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model % hp.enc_n_heads != 0) {
        std::fprintf(stderr,
                     "whisper: encoder d_model (%d) not divisible by n_heads (%d)\n",
                     hp.enc_d_model, hp.enc_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_n_layers <= 0 || hp.dec_d_model <= 0 || hp.dec_n_heads <= 0 ||
        hp.dec_ffn_dim <= 0 || hp.dec_max_target_positions <= 0 ||
        hp.dec_vocab_size <= 0)
    {
        std::fprintf(stderr, "whisper: decoder hparams must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_d_model % hp.dec_n_heads != 0) {
        std::fprintf(stderr,
                     "whisper: decoder d_model (%d) not divisible by n_heads (%d)\n",
                     hp.dec_d_model, hp.dec_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model != hp.dec_d_model) {
        // Upstream whisper always uses d_model=d_model; a mismatch would
        // break the cross-attention K/V dim contract.
        std::fprintf(stderr,
                     "whisper: encoder d_model (%d) != decoder d_model (%d); "
                     "cross-attention would mismatch\n",
                     hp.enc_d_model, hp.dec_d_model);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_activation != "gelu" || hp.dec_activation != "gelu") {
        std::fprintf(stderr,
                     "whisper: only \"gelu\" activation is supported "
                     "(enc=\"%s\", dec=\"%s\")\n",
                     hp.enc_activation.c_str(), hp.dec_activation.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!hp.dec_tie_word_embeddings) {
        // Whisper always ties lm_head to token_embedding; a false here
        // would mean the converter shipped a separate head tensor we
        // don't currently load.
        std::fprintf(stderr,
                     "whisper: stt.whisper.decoder.tie_word_embeddings=false "
                     "is not supported (no separate lm_head tensor in the "
                     "catalog)\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_scale_embedding) {
        // HF WhisperDecoder multiplies inputs_embeds by sqrt(d_model) when
        // config.scale_embedding=True. Upstream Whisper always ships False;
        // if this ever flips we need to add the scale to the decoder graph
        // (both prefill and KV paths, after get_rows(token_embd, ...)).
        std::fprintf(stderr,
                     "whisper: stt.whisper.decoder.scale_embedding=true is "
                     "not supported (decoder graph does not apply the "
                     "sqrt(d_model) embed scale)\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.decoder_start_token_id < 0 || hp.no_timestamps_token_id < 0) {
        std::fprintf(stderr,
                     "whisper: decoder_start_token_id / no_timestamps_token_id "
                     "must be set\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_num_mels <= 0 || hp.fe_sample_rate <= 0 ||
        hp.fe_n_fft <= 0 || hp.fe_win_length <= 0 || hp.fe_hop_length <= 0)
    {
        std::fprintf(stderr, "whisper: frontend dimensions must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_win_length != hp.fe_n_fft) {
        // Whisper uses a full-length periodic Hann window (win=n_fft=400).
        // A mismatch would require a windowed-shorter-than-FFT code path
        // that we don't implement.
        std::fprintf(stderr,
                     "whisper: frontend win_length (%d) != n_fft (%d); "
                     "only full-length window is supported\n",
                     hp.fe_win_length, hp.fe_n_fft);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_num_mels != hp.enc_num_mel_bins) {
        std::fprintf(stderr,
                     "whisper: frontend num_mels (%d) != encoder num_mel_bins (%d)\n",
                     hp.fe_num_mels, hp.enc_num_mel_bins);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_type != "mel") {
        std::fprintf(stderr,
                     "whisper: unsupported frontend type \"%s\" (only \"mel\")\n",
                     hp.fe_type.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_window != "hann" && hp.fe_window != "hann_periodic") {
        std::fprintf(stderr,
                     "whisper: unsupported frontend window \"%s\" "
                     "(only \"hann\"/\"hann_periodic\" is supported)\n",
                     hp.fe_window.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_normalize != "whisper_logmel") {
        std::fprintf(stderr,
                     "whisper: unsupported frontend normalize \"%s\" "
                     "(only \"whisper_logmel\" is supported)\n",
                     hp.fe_normalize.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_mel_norm != "slaney") {
        std::fprintf(stderr,
                     "whisper: unsupported frontend mel_norm \"%s\" "
                     "(only \"slaney\" is supported)\n",
                     hp.fe_mel_norm.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_pad_mode != "reflect") {
        std::fprintf(stderr,
                     "whisper: unsupported frontend pad_mode \"%s\" "
                     "(only \"reflect\" is supported)\n",
                     hp.fe_pad_mode.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!hp.fe_center) {
        std::fprintf(stderr,
                     "whisper: non-centered STFT is not supported\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_dither != 0.0f) {
        // HF WhisperFeatureExtractor defaults dither=0.0 and the C++ mel
        // frontend never applies dither (transcribe-mel.cpp's contract).
        // A non-zero value in the GGUF would be silently dropped, so
        // reject it at load time rather than producing a silently-wrong
        // spectrogram.
        std::fprintf(stderr,
                     "whisper: stt.frontend.dither=%g is not supported "
                     "(frontend is deterministic; expected 0.0)\n",
                     hp.fe_dither);
        return TRANSCRIBE_ERR_GGUF;
    }

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

namespace {

using transcribe::weights::lname;

constexpr const char * kTag = kFamilyTag;

#define GET_F32(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {GGML_TYPE_F32}, {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_CONV(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {TRANSCRIBE_QUANT_CONV_TYPES}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_LIN(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {TRANSCRIBE_QUANT_LINEAR_TYPES}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

} // namespace

transcribe_status build_whisper_weights(const gguf_context *    gguf,
                                        ggml_context *          ctx_meta,
                                        const WhisperHParams &  hp,
                                        WhisperWeights &        weights)
{
    if (gguf == nullptr || ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t d_model     = hp.enc_d_model;
    const int64_t n_mel       = hp.enc_num_mel_bins;
    const int64_t enc_ff      = hp.enc_ffn_dim;
    const int64_t src_pos     = hp.enc_max_source_positions;
    const int64_t dec_h       = hp.dec_d_model;
    const int64_t dec_ff      = hp.dec_ffn_dim;
    const int64_t tgt_pos     = hp.dec_max_target_positions;
    const int64_t vocab_size  = hp.dec_vocab_size;
    const int64_t n_fft_half  = hp.fe_n_fft / 2 + 1;

    // ----- frontend (mel filterbank + hann window) -----
    GET_F32(weights.frontend.mel_filterbank, "frontend.mel_filterbank", n_fft_half, n_mel);
    GET_F32(weights.frontend.window,         "frontend.window",         hp.fe_n_fft);

    // ----- encoder conv stem -----
    // Conv1d kernels: PyTorch [out, in, K] -> ggml ne=[K, in, out].
    GET_CONV(weights.enc_stem.conv0_w, "enc.conv.0.weight", 3, n_mel,   d_model);
    GET_F32 (weights.enc_stem.conv0_b, "enc.conv.0.bias",   d_model);
    GET_CONV(weights.enc_stem.conv1_w, "enc.conv.1.weight", 3, d_model, d_model);
    GET_F32 (weights.enc_stem.conv1_b, "enc.conv.1.bias",   d_model);

    // ----- encoder top (pos emb + final LN) -----
    // pos_emb is nn.Embedding weight [max_source_positions, d_model] ->
    // ggml ne = [d_model, max_source_positions].
    GET_F32(weights.enc_top.pos_emb_w,    "enc.pos_emb.weight",    d_model, src_pos);
    GET_F32(weights.enc_top.final_norm_w, "enc.final_norm.weight", d_model);
    GET_F32(weights.enc_top.final_norm_b, "enc.final_norm.bias",   d_model);

    // ----- encoder blocks -----
    weights.enc_blocks.assign(hp.enc_n_layers, WhisperEncBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = weights.enc_blocks[i];

        GET_F32(b.norm_attn_w, lname("enc.blocks.%d.norm_attn.weight", i), d_model);
        GET_F32(b.norm_attn_b, lname("enc.blocks.%d.norm_attn.bias",   i), d_model);

        GET_LIN(b.attn_q_w,    lname("enc.blocks.%d.attn.q.weight",   i), d_model, d_model);
        GET_F32(b.attn_q_b,    lname("enc.blocks.%d.attn.q.bias",     i), d_model);
        GET_LIN(b.attn_k_w,    lname("enc.blocks.%d.attn.k.weight",   i), d_model, d_model);
        // attn.k has NO bias.
        GET_LIN(b.attn_v_w,    lname("enc.blocks.%d.attn.v.weight",   i), d_model, d_model);
        GET_F32(b.attn_v_b,    lname("enc.blocks.%d.attn.v.bias",     i), d_model);
        GET_LIN(b.attn_out_w,  lname("enc.blocks.%d.attn.out.weight", i), d_model, d_model);
        GET_F32(b.attn_out_b,  lname("enc.blocks.%d.attn.out.bias",   i), d_model);

        GET_F32(b.norm_ffn_w,  lname("enc.blocks.%d.norm_ffn.weight", i), d_model);
        GET_F32(b.norm_ffn_b,  lname("enc.blocks.%d.norm_ffn.bias",   i), d_model);
        GET_LIN(b.ffn_fc1_w,   lname("enc.blocks.%d.ffn.fc1.weight",  i), d_model, enc_ff);
        GET_F32(b.ffn_fc1_b,   lname("enc.blocks.%d.ffn.fc1.bias",    i), enc_ff);
        GET_LIN(b.ffn_fc2_w,   lname("enc.blocks.%d.ffn.fc2.weight",  i), enc_ff,  d_model);
        GET_F32(b.ffn_fc2_b,   lname("enc.blocks.%d.ffn.fc2.bias",    i), d_model);
    }

    // ----- decoder top -----
    // Token embedding: PyTorch [vocab, dec_h] -> ggml ne = [dec_h, vocab].
    // This tensor doubles as the lm_head weight (tied), so it must
    // accept the full quant allowlist (Embed bucket gets bumped to Q6_K
    // under _M presets, F16 under F16, etc).
    GET_LIN(weights.dec_top.token_embd_w, "dec.token_embd.weight", dec_h, vocab_size);
    GET_F32(weights.dec_top.pos_emb_w,    "dec.pos_emb.weight",    dec_h, tgt_pos);
    GET_F32(weights.dec_top.final_norm_w, "dec.final_norm.weight", dec_h);
    GET_F32(weights.dec_top.final_norm_b, "dec.final_norm.bias",   dec_h);

    // ----- decoder blocks -----
    weights.dec_blocks.assign(hp.dec_n_layers, WhisperDecBlock{});
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        auto & b = weights.dec_blocks[i];

        // Self-attention (k has no bias).
        GET_F32(b.norm_self_w,  lname("dec.blocks.%d.norm_self.weight",    i), dec_h);
        GET_F32(b.norm_self_b,  lname("dec.blocks.%d.norm_self.bias",      i), dec_h);
        GET_LIN(b.self_q_w,     lname("dec.blocks.%d.self_attn.q.weight",  i), dec_h, dec_h);
        GET_F32(b.self_q_b,     lname("dec.blocks.%d.self_attn.q.bias",    i), dec_h);
        GET_LIN(b.self_k_w,     lname("dec.blocks.%d.self_attn.k.weight",  i), dec_h, dec_h);
        GET_LIN(b.self_v_w,     lname("dec.blocks.%d.self_attn.v.weight",  i), dec_h, dec_h);
        GET_F32(b.self_v_b,     lname("dec.blocks.%d.self_attn.v.bias",    i), dec_h);
        GET_LIN(b.self_out_w,   lname("dec.blocks.%d.self_attn.out.weight",i), dec_h, dec_h);
        GET_F32(b.self_out_b,   lname("dec.blocks.%d.self_attn.out.bias",  i), dec_h);

        // Cross-attention (k has no bias).
        GET_F32(b.norm_cross_w, lname("dec.blocks.%d.norm_cross.weight",    i), dec_h);
        GET_F32(b.norm_cross_b, lname("dec.blocks.%d.norm_cross.bias",      i), dec_h);
        GET_LIN(b.cross_q_w,    lname("dec.blocks.%d.cross_attn.q.weight",  i), dec_h, dec_h);
        GET_F32(b.cross_q_b,    lname("dec.blocks.%d.cross_attn.q.bias",    i), dec_h);
        GET_LIN(b.cross_k_w,    lname("dec.blocks.%d.cross_attn.k.weight",  i), dec_h, dec_h);
        GET_LIN(b.cross_v_w,    lname("dec.blocks.%d.cross_attn.v.weight",  i), dec_h, dec_h);
        GET_F32(b.cross_v_b,    lname("dec.blocks.%d.cross_attn.v.bias",    i), dec_h);
        GET_LIN(b.cross_out_w,  lname("dec.blocks.%d.cross_attn.out.weight",i), dec_h, dec_h);
        GET_F32(b.cross_out_b,  lname("dec.blocks.%d.cross_attn.out.bias",  i), dec_h);

        // FFN.
        GET_F32(b.norm_ffn_w,   lname("dec.blocks.%d.norm_ffn.weight", i), dec_h);
        GET_F32(b.norm_ffn_b,   lname("dec.blocks.%d.norm_ffn.bias",   i), dec_h);
        GET_LIN(b.ffn_fc1_w,    lname("dec.blocks.%d.ffn.fc1.weight",  i), dec_h,  dec_ff);
        GET_F32(b.ffn_fc1_b,    lname("dec.blocks.%d.ffn.fc1.bias",    i), dec_ff);
        GET_LIN(b.ffn_fc2_w,    lname("dec.blocks.%d.ffn.fc2.weight",  i), dec_ff, dec_h);
        GET_F32(b.ffn_fc2_b,    lname("dec.blocks.%d.ffn.fc2.bias",    i), dec_h);
    }

    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_CONV
#undef GET_LIN

} // namespace transcribe::whisper
