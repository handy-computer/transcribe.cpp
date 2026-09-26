// arch/nemotron3_diar/weights.cpp - Nemotron-3-Diarization hparam KV
// reader + weight catalog.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cstdio>
#include <string>

namespace transcribe::nemotron3_diar {

namespace {

transcribe_status kv_u32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

transcribe_status kv_f32(const gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_f32(g, id);
    return TRANSCRIBE_OK;
}

transcribe_status kv_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_str(g, id);
    return TRANSCRIBE_OK;
}

// Look up a tensor by name and validate its ne (fast-to-slow; -1 skips).
ggml_tensor * get_checked(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1, int64_t ne2 = -1) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing tensor %s", name);
        return nullptr;
    }
    if ((ne0 >= 0 && t->ne[0] != ne0) || (ne1 >= 0 && t->ne[1] != ne1) || (ne2 >= 0 && t->ne[2] != ne2)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "nemotron3_diar: tensor %s shape mismatch: have [%lld,%lld,%lld] want [%lld,%lld,%lld]", name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) ne0, (long long) ne1,
                (long long) ne2);
        return nullptr;
    }
    return t;
}

}  // namespace

transcribe_status read_hparams(const gguf_context * g, Nemotron3DiarHParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

#define RD_U32(key, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_F32(key, field)                                                          \
    if (const transcribe_status st = kv_f32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_STR(key, field)                                                          \
    if (const transcribe_status st = kv_str(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("stt.nemotron3_diar.max_speakers", max_speakers);
    RD_U32("stt.nemotron3_diar.frame_hop", frame_hop);
    RD_U32("stt.nemotron3_diar.output_hop", output_hop);
    RD_U32("stt.nemotron3_diar.upsample_factor", upsample_factor);

    RD_U32("stt.nemotron3_diar.encoder.n_layers", enc_n_layers);
    RD_U32("stt.nemotron3_diar.encoder.d_model", enc_d_model);
    RD_U32("stt.nemotron3_diar.encoder.n_heads", enc_n_heads);
    RD_U32("stt.nemotron3_diar.encoder.d_ff", enc_d_ff);
    RD_U32("stt.nemotron3_diar.encoder.feat_in", enc_feat_in);
    RD_U32("stt.nemotron3_diar.encoder.subsampling_factor", enc_subsampling_factor);
    RD_STR("stt.nemotron3_diar.encoder.subsampling", enc_subsampling);
    RD_F32("stt.nemotron3_diar.encoder.rope_base", enc_rope_base);
    RD_F32("stt.nemotron3_diar.encoder.rotary_fraction", enc_rotary_fraction);
    RD_F32("stt.nemotron3_diar.encoder.layer_norm_eps", enc_ln_eps);
    RD_STR("stt.nemotron3_diar.encoder.activation", enc_activation);
    RD_U32("stt.nemotron3_diar.head.d_model", head_d_model);

    RD_U32("stt.nemotron3_diar.aosc.spkcache_sil_frames_per_spk", spkcache_sil_frames_per_spk);
    RD_F32("stt.nemotron3_diar.aosc.pred_score_threshold", pred_score_threshold);
    RD_F32("stt.nemotron3_diar.aosc.scores_boost_latest", scores_boost_latest);
    RD_F32("stt.nemotron3_diar.aosc.sil_threshold", sil_threshold);
    RD_F32("stt.nemotron3_diar.aosc.strong_boost_rate", strong_boost_rate);
    RD_F32("stt.nemotron3_diar.aosc.weak_boost_rate", weak_boost_rate);
    RD_F32("stt.nemotron3_diar.aosc.min_pos_scores_rate", min_pos_scores_rate);
    RD_U32("stt.nemotron3_diar.aosc.max_index", max_index);

    RD_U32("stt.nemotron3_diar.stream.spkcache_len", stream_spkcache_len);
    RD_U32("stt.nemotron3_diar.stream.fifo_len", stream_fifo_len);
    RD_U32("stt.nemotron3_diar.stream.chunk_len", stream_chunk_len);
    RD_U32("stt.nemotron3_diar.stream.chunk_right_context", stream_chunk_right_context);
    RD_U32("stt.nemotron3_diar.stream.spkcache_update_period", stream_spkcache_update_period);

    RD_U32("stt.frontend.num_mels", fe_num_mels);
    RD_U32("stt.frontend.sample_rate", fe_sample_rate);
    RD_U32("stt.frontend.n_fft", fe_n_fft);
    RD_U32("stt.frontend.win_length", fe_win_length);
    RD_U32("stt.frontend.hop_length", fe_hop_length);
    RD_STR("stt.frontend.window", fe_window);
    RD_STR("stt.frontend.normalize", fe_normalize);
    RD_F32("stt.frontend.dither", fe_dither);
    RD_F32("stt.frontend.pre_emphasis", fe_pre_emphasis);

#undef RD_U32
#undef RD_F32
#undef RD_STR

    if (hp.enc_n_heads <= 0 || hp.enc_d_model % hp.enc_n_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: d_model %d not divisible by n_heads %d", hp.enc_d_model,
                hp.enc_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_subsampling != "feature_stacking" || hp.enc_activation != "gelu" || hp.enc_rotary_fraction != 1.0f) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "nemotron3_diar: unsupported encoder config (subsampling=%s activation=%s rotary_fraction=%g)",
                hp.enc_subsampling.c_str(), hp.enc_activation.c_str(), static_cast<double>(hp.enc_rotary_fraction));
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_subsampling_factor <= 0 || hp.upsample_factor <= 0 ||
        hp.frame_hop != hp.fe_hop_length * hp.enc_subsampling_factor ||
        hp.output_hop * hp.upsample_factor != hp.frame_hop) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "nemotron3_diar: inconsistent frame geometry (hop=%d sub=%d frame_hop=%d output_hop=%d upsample=%d)",
                hp.fe_hop_length, hp.enc_subsampling_factor, hp.frame_hop, hp.output_hop, hp.upsample_factor);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status build_weights(ggml_context * ctx, const Nemotron3DiarHParams & hp, Nemotron3DiarWeights & w) {
    const int64_t d   = hp.enc_d_model;
    const int64_t dff = hp.enc_d_ff;
    const int64_t hd  = hp.head_d_model;
    const int64_t spk = hp.max_speakers;
    const int64_t up  = hp.upsample_factor;
    const int64_t nfr = hp.fe_n_fft / 2 + 1;
    const int64_t fin = static_cast<int64_t>(hp.enc_feat_in) * hp.enc_subsampling_factor;
    char          name[96];

#define GET(dst, nm, ...)                            \
    do {                                             \
        (dst) = get_checked(ctx, (nm), __VA_ARGS__); \
        if ((dst) == nullptr)                        \
            return TRANSCRIBE_ERR_GGUF;              \
    } while (0)

    GET(w.fe_window, "frontend.window", hp.fe_win_length, -1);
    GET(w.fe_filterbank, "frontend.mel_filterbank", nfr, hp.fe_num_mels);

    GET(w.pre_encode_proj_w, "enc.pre_encode.proj.weight", fin, d);
    GET(w.embed_norm_w, "enc.embed.norm.weight", d, -1);
    GET(w.embed_norm_b, "enc.embed.norm.bias", d, -1);

    w.blocks.resize(static_cast<size_t>(hp.enc_n_layers));
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        Nemotron3DiarBlock & b = w.blocks[static_cast<size_t>(i)];
#define GETB(dst, suffix, ...)                                            \
    do {                                                                  \
        std::snprintf(name, sizeof(name), "enc.blocks.%d.%s", i, suffix); \
        GET(dst, name, __VA_ARGS__);                                      \
    } while (0)
        GETB(b.norm1_w, "norm_1.weight", d, -1);
        GETB(b.norm1_b, "norm_1.bias", d, -1);
        GETB(b.attn_qkv_w, "attn.qkv.weight", d, 3 * d);
        GETB(b.attn_out_w, "attn.out.weight", d, d);
        GETB(b.attn_out_b, "attn.out.bias", d, -1);
        GETB(b.norm2_w, "norm_2.weight", d, -1);
        GETB(b.norm2_b, "norm_2.bias", d, -1);
        GETB(b.ff_in_w, "ff.in.weight", d, dff);
        GETB(b.ff_in_b, "ff.in.bias", dff, -1);
        GETB(b.ff_out_w, "ff.out.weight", dff, d);
        GETB(b.ff_out_b, "ff.out.bias", d, -1);
#undef GETB
    }

    GET(w.final_norm_w, "enc.final_norm.weight", d, -1);
    GET(w.final_norm_b, "enc.final_norm.bias", d, -1);

    GET(w.enc_proj_w, "diar.encoder_proj.weight", d, hd);
    GET(w.enc_proj_b, "diar.encoder_proj.bias", hd, -1);
    GET(w.upsample_w, "diar.upsample.conv.weight", 3, hd, hd * up);
    GET(w.upsample_b, "diar.upsample.conv.bias", hd * up, -1);
    GET(w.fc1_w, "diar.fc1.weight", hd, hd);
    GET(w.fc1_b, "diar.fc1.bias", hd, -1);
    GET(w.spk_head_w, "diar.single_spk_head.weight", hd, spk);
    GET(w.spk_head_b, "diar.single_spk_head.bias", spk, -1);
    GET(w.sil_emb, "diar.sil_emb", d, -1);

#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::nemotron3_diar
