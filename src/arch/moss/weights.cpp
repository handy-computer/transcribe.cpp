// arch/moss/weights.cpp - read_moss_hparams + build_moss_weights.
//
// Mirrors arch/qwen3_asr/weights.cpp: every required hparam is read explicitly
// (BadType fatal), and the tensor catalog is a sequence of typed get_tensor()
// calls with expected shapes. The lm_head slot does not exist — MOSS ties the
// output projection to dec.token_embd.weight.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cstdio>

namespace transcribe::moss {

namespace {
constexpr const char * kFamilyTag = "moss";

// Read a required non-empty INT32 array KV into `out`.
transcribe_status read_required_i32_array(const gguf_context * gguf, const char * key, std::vector<int32_t> & out) {
    switch (read_int32_array_kv(gguf, key, out)) {
        case KvResult::Ok:
            if (out.empty()) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: KV %s is an empty array", key);
                return TRANSCRIBE_ERR_GGUF;
            }
            return TRANSCRIBE_OK;
        case KvResult::Absent:
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: missing required KV %s", key);
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::BadType:
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: KV %s has wrong type (want i32 array)", key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF;  // unreachable
}
}  // namespace

transcribe_status read_moss_hparams(const gguf_context * gguf, MossHParams & hp) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Encoder (Whisper).
    if (auto st = read_required_u32_kv(gguf, "stt.moss.encoder.n_layers", kFamilyTag, hp.enc_n_layers);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.encoder.d_model", kFamilyTag, hp.enc_d_model);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.encoder.n_heads", kFamilyTag, hp.enc_n_heads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.encoder.ffn_dim", kFamilyTag, hp.enc_ffn_dim);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.encoder.num_mel_bins", kFamilyTag, hp.enc_num_mel_bins);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.encoder.max_source_positions", kFamilyTag,
                                       hp.enc_max_source_positions);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.moss.encoder.activation", kFamilyTag, hp.enc_activation);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Adaptor.
    if (auto st = read_required_u32_kv(gguf, "stt.moss.adaptor.input_dim", kFamilyTag, hp.adaptor_input_dim);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.audio_merge_size", kFamilyTag, hp.audio_merge_size);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Decoder (Qwen3).
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.n_layers", kFamilyTag, hp.dec_n_layers);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.hidden_size", kFamilyTag, hp.dec_hidden);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.intermediate_size", kFamilyTag, hp.dec_intermediate);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.n_heads", kFamilyTag, hp.dec_n_heads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.n_kv_heads", kFamilyTag, hp.dec_n_kv_heads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.head_dim", kFamilyTag, hp.dec_head_dim);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.moss.decoder.hidden_act", kFamilyTag, hp.dec_hidden_act);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.moss.decoder.rms_norm_eps", kFamilyTag, hp.dec_rms_norm_eps);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.moss.decoder.rope_theta", kFamilyTag, hp.dec_rope_theta);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.max_position_embeddings", kFamilyTag,
                                       hp.dec_max_position_embeddings);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_bool_kv(gguf, "stt.moss.decoder.tie_word_embeddings", kFamilyTag, true,
                                        hp.dec_tie_word_embeddings);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.decoder.vocab_size", kFamilyTag, hp.dec_vocab_size);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Audio injection + time-marker span.
    if (auto st = read_required_u32_kv(gguf, "stt.moss.audio_token_id", kFamilyTag, hp.audio_token_id);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.moss.audio_tokens_per_second", kFamilyTag, hp.audio_tokens_per_second);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.moss.time_marker_every_seconds", kFamilyTag,
                                       hp.time_marker_every_seconds);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_bool_kv(gguf, "stt.moss.enable_time_marker", kFamilyTag, true, hp.enable_time_marker);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Baked prompt tokens.
    if (auto st = read_required_i32_array(gguf, "stt.moss.prompt_prefix_tokens", hp.prompt_prefix_tokens);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_i32_array(gguf, "stt.moss.prompt_suffix_tokens", hp.prompt_suffix_tokens);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_i32_array(gguf, "stt.moss.digit_tokens", hp.digit_tokens); st != TRANSCRIBE_OK) {
        return st;
    }
    if (hp.digit_tokens.size() != 10) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: stt.moss.digit_tokens must have 10 entries, got %zu",
                hp.digit_tokens.size());
        return TRANSCRIBE_ERR_GGUF;
    }

    // Frontend.
    if (auto st = read_required_string_kv(gguf, "stt.frontend.type", kFamilyTag, hp.fe_type); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.num_mels", kFamilyTag, hp.fe_num_mels);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.sample_rate", kFamilyTag, hp.fe_sample_rate);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.n_fft", kFamilyTag, hp.fe_n_fft); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.win_length", kFamilyTag, hp.fe_win_length);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.hop_length", kFamilyTag, hp.fe_hop_length);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.frontend.window", kFamilyTag, hp.fe_window); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.frontend.normalize", kFamilyTag, hp.fe_normalize);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.dither", kFamilyTag, hp.fe_dither); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.pre_emphasis", kFamilyTag, hp.fe_pre_emphasis);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.f_min", kFamilyTag, hp.fe_f_min); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.f_max", kFamilyTag, hp.fe_f_max); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.pad_mode", kFamilyTag, "reflect", hp.fe_pad_mode);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.mel_norm", kFamilyTag, "slaney", hp.fe_mel_norm);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_bool_kv(gguf, "stt.frontend.center", kFamilyTag, true, hp.fe_center);
        st != TRANSCRIBE_OK) {
        return st;
    }
    {
        const auto read_optional = [&](const char * key, int32_t & dst) -> transcribe_status {
            uint32_t tmp = 0;
            switch (read_uint32_kv(gguf, key, tmp)) {
                case KvResult::Ok:
                    dst = static_cast<int32_t>(tmp);
                    return TRANSCRIBE_OK;
                case KvResult::Absent:
                    return TRANSCRIBE_OK;
                case KvResult::BadType:
                    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: \"%s\" has wrong type", key);
                    return TRANSCRIBE_ERR_GGUF;
            }
            return TRANSCRIBE_ERR_GGUF;
        };
        if (auto st = read_optional("stt.frontend.chunk_length", hp.fe_chunk_length); st != TRANSCRIBE_OK) {
            return st;
        }
        if (auto st = read_optional("stt.frontend.n_samples", hp.fe_n_samples); st != TRANSCRIBE_OK) {
            return st;
        }
        if (auto st = read_optional("stt.frontend.nb_max_frames", hp.fe_nb_max_frames); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    // Cross-field invariants.
    if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 || hp.enc_ffn_dim <= 0 ||
        hp.enc_num_mel_bins <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: encoder hparams must be positive");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model % hp.enc_n_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: encoder d_model (%d) not divisible by n_heads (%d)", hp.enc_d_model,
                hp.enc_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_activation != "gelu") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: unsupported encoder activation \"%s\" (only gelu)",
                hp.enc_activation.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.audio_merge_size <= 0 || hp.adaptor_input_dim != hp.dec_hidden * hp.audio_merge_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: adaptor_input_dim (%d) != dec_hidden (%d) * merge (%d)",
                hp.adaptor_input_dim, hp.dec_hidden, hp.audio_merge_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_n_layers <= 0 || hp.dec_hidden <= 0 || hp.dec_n_heads <= 0 || hp.dec_n_kv_heads <= 0 ||
        hp.dec_head_dim <= 0 || hp.dec_intermediate <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: decoder hparams must be positive");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: n_heads (%d) not divisible by n_kv_heads (%d)", hp.dec_n_heads,
                hp.dec_n_kv_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_hidden_act != "silu" && hp.dec_hidden_act != "swish") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: unsupported decoder hidden_act \"%s\" (only silu/swish)",
                hp.dec_hidden_act.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_type != "mel") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: unsupported frontend type \"%s\"", hp.fe_type.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!hp.dec_tie_word_embeddings) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "moss: decoder.tie_word_embeddings=false is not supported (graph assumes tied lm_head)");
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

#define GET_F32(slot, name, ...)                                                                          \
    do {                                                                                                  \
        ggml_tensor * _t =                                                                                \
            transcribe::weights::find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr)                                                                                \
            return TRANSCRIBE_ERR_GGUF;                                                                   \
        (slot) = _t;                                                                                      \
    } while (0)

#define GET_CONV(slot, name, ...)                                                                              \
    do {                                                                                                       \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_CONV_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                            \
        if (_t == nullptr)                                                                                     \
            return TRANSCRIBE_ERR_GGUF;                                                                        \
        (slot) = _t;                                                                                           \
    } while (0)

#define GET_LIN(slot, name, ...)                                                                                 \
    do {                                                                                                         \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                              \
        if (_t == nullptr)                                                                                       \
            return TRANSCRIBE_ERR_GGUF;                                                                          \
        (slot) = _t;                                                                                             \
    } while (0)

}  // namespace

transcribe_status build_moss_weights(ggml_context * ctx_meta, const MossHParams & hp, MossWeights & weights) {
    if (ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t d_model = hp.enc_d_model;
    const int64_t ffn_dim = hp.enc_ffn_dim;
    const int64_t n_mels  = hp.enc_num_mel_bins;
    const int64_t adp_in  = hp.adaptor_input_dim;

    // ----- encoder: conv stem -----
    GET_CONV(weights.enc_stem.conv0_w, "enc.conv.0.weight", 3, n_mels, d_model);
    GET_F32(weights.enc_stem.conv0_b, "enc.conv.0.bias", d_model);
    GET_CONV(weights.enc_stem.conv1_w, "enc.conv.1.weight", 3, d_model, d_model);
    GET_F32(weights.enc_stem.conv1_b, "enc.conv.1.bias", d_model);

    // ----- encoder: top (pos emb + final LN) -----
    GET_F32(weights.enc_top.pos_emb_w, "enc.pos_emb.weight", d_model, hp.enc_max_source_positions);
    GET_F32(weights.enc_top.final_norm_w, "enc.final_norm.weight", d_model);
    GET_F32(weights.enc_top.final_norm_b, "enc.final_norm.bias", d_model);

    // ----- encoder: blocks (q/v/out bias; k none) -----
    weights.enc_blocks.assign(hp.enc_n_layers, MossEncBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = weights.enc_blocks[i];
        GET_F32(b.norm_attn_w, lname("enc.blocks.%d.norm_attn.weight", i), d_model);
        GET_F32(b.norm_attn_b, lname("enc.blocks.%d.norm_attn.bias", i), d_model);
        GET_LIN(b.attn_q_w, lname("enc.blocks.%d.attn.q.weight", i), d_model, d_model);
        GET_F32(b.attn_q_b, lname("enc.blocks.%d.attn.q.bias", i), d_model);
        GET_LIN(b.attn_k_w, lname("enc.blocks.%d.attn.k.weight", i), d_model, d_model);
        GET_LIN(b.attn_v_w, lname("enc.blocks.%d.attn.v.weight", i), d_model, d_model);
        GET_F32(b.attn_v_b, lname("enc.blocks.%d.attn.v.bias", i), d_model);
        GET_LIN(b.attn_out_w, lname("enc.blocks.%d.attn.out.weight", i), d_model, d_model);
        GET_F32(b.attn_out_b, lname("enc.blocks.%d.attn.out.bias", i), d_model);
        GET_F32(b.norm_ffn_w, lname("enc.blocks.%d.norm_ffn.weight", i), d_model);
        GET_F32(b.norm_ffn_b, lname("enc.blocks.%d.norm_ffn.bias", i), d_model);
        GET_LIN(b.ffn_fc1_w, lname("enc.blocks.%d.ffn.fc1.weight", i), d_model, ffn_dim);
        GET_F32(b.ffn_fc1_b, lname("enc.blocks.%d.ffn.fc1.bias", i), ffn_dim);
        GET_LIN(b.ffn_fc2_w, lname("enc.blocks.%d.ffn.fc2.weight", i), ffn_dim, d_model);
        GET_F32(b.ffn_fc2_b, lname("enc.blocks.%d.ffn.fc2.bias", i), d_model);
    }

    // ----- adaptor -----
    GET_LIN(weights.adaptor.fc1_w, "adaptor.fc1.weight", adp_in, hp.dec_hidden);
    GET_F32(weights.adaptor.fc1_b, "adaptor.fc1.bias", hp.dec_hidden);
    GET_LIN(weights.adaptor.fc2_w, "adaptor.fc2.weight", hp.dec_hidden, hp.dec_hidden);
    GET_F32(weights.adaptor.fc2_b, "adaptor.fc2.bias", hp.dec_hidden);
    GET_F32(weights.adaptor.norm_out_w, "adaptor.norm_out.weight", hp.dec_hidden);
    GET_F32(weights.adaptor.norm_out_b, "adaptor.norm_out.bias", hp.dec_hidden);

    // ----- decoder: embedding (tied output) -----
    {
        ggml_tensor * tw = ggml_get_tensor(ctx_meta, "dec.token_embd.weight");
        if (tw == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: missing tensor \"dec.token_embd.weight\"");
            return TRANSCRIBE_ERR_GGUF;
        }
        if (tw->ne[0] != hp.dec_hidden || tw->ne[1] != hp.dec_vocab_size) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: dec.token_embd.weight shape [%lld,%lld], expected [%lld,%lld]",
                    static_cast<long long>(tw->ne[0]), static_cast<long long>(tw->ne[1]),
                    static_cast<long long>(hp.dec_hidden), static_cast<long long>(hp.dec_vocab_size));
            return TRANSCRIBE_ERR_GGUF;
        }
        weights.dec_embed.token_w = tw;
    }

    // ----- decoder: blocks -----
    const int64_t dec_h  = hp.dec_hidden;
    const int64_t dec_hd = hp.dec_head_dim;
    const int64_t dec_im = hp.dec_intermediate;
    const int64_t q_out  = static_cast<int64_t>(hp.dec_n_heads) * dec_hd;
    const int64_t kv_out = static_cast<int64_t>(hp.dec_n_kv_heads) * dec_hd;

    weights.dec_blocks.assign(hp.dec_n_layers, MossDecBlock{});
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        auto & b = weights.dec_blocks[i];
        GET_F32(b.norm_attn_w, lname("dec.blocks.%d.norm_attn.weight", i), dec_h);
        GET_F32(b.norm_ffn_w, lname("dec.blocks.%d.norm_ffn.weight", i), dec_h);
        GET_LIN(b.attn_q_w, lname("dec.blocks.%d.attn.q.weight", i), dec_h, q_out);
        GET_LIN(b.attn_k_w, lname("dec.blocks.%d.attn.k.weight", i), dec_h, kv_out);
        GET_LIN(b.attn_v_w, lname("dec.blocks.%d.attn.v.weight", i), dec_h, kv_out);
        GET_LIN(b.attn_o_w, lname("dec.blocks.%d.attn.o.weight", i), q_out, dec_h);
        GET_F32(b.attn_q_norm, lname("dec.blocks.%d.attn.q_norm.weight", i), dec_hd);
        GET_F32(b.attn_k_norm, lname("dec.blocks.%d.attn.k_norm.weight", i), dec_hd);
        GET_LIN(b.ffn_gate_w, lname("dec.blocks.%d.ffn.gate.weight", i), dec_h, dec_im);
        GET_LIN(b.ffn_up_w, lname("dec.blocks.%d.ffn.up.weight", i), dec_h, dec_im);
        GET_LIN(b.ffn_down_w, lname("dec.blocks.%d.ffn.down.weight", i), dec_im, dec_h);
        if (b.ffn_gate_w->type != b.ffn_up_w->type) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "moss: ffn gate/up dtype mismatch at layer %d", i);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    GET_F32(weights.dec_final.norm_w, "dec.output_norm.weight", dec_h);
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::moss
