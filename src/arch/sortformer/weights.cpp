// arch/sortformer/weights.cpp - Sortformer hparam KV reader + weight
// catalog for the Sortformer-specific tensors (encoder_proj + transformer
// + diar head). The conformer weights are loaded separately (parakeet
// reuse); see model.cpp::load.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cstdio>
#include <string>

namespace transcribe::sortformer {

namespace {

transcribe_status kv_u32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

transcribe_status kv_f32(const gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_f32(g, id);
    return TRANSCRIBE_OK;
}

transcribe_status kv_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_str(g, id);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status read_sortformer_hparams(const gguf_context * g, SortformerHParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

#define RD_U32(key, field)                                              \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) return st
#define RD_F32(key, field)                                              \
    if (const transcribe_status st = kv_f32(g, key, hp.field); st != TRANSCRIBE_OK) return st
#define RD_STR(key, field)                                              \
    if (const transcribe_status st = kv_str(g, key, hp.field); st != TRANSCRIBE_OK) return st

    RD_U32("stt.sortformer.max_speakers", max_speakers);
    RD_U32("stt.sortformer.frame_hop", frame_hop);

    RD_U32("stt.sortformer.encoder.n_layers", enc_n_layers);
    RD_U32("stt.sortformer.encoder.d_model", enc_d_model);
    RD_U32("stt.sortformer.encoder.n_heads", enc_n_heads);
    RD_U32("stt.sortformer.encoder.d_ff", enc_d_ff);
    RD_U32("stt.sortformer.encoder.conv_kernel", enc_conv_kernel);
    RD_U32("stt.sortformer.encoder.subsampling_factor", enc_subsampling_factor);
    RD_U32("stt.sortformer.encoder.subsampling_channels", enc_subsampling_channels);
    RD_U32("stt.sortformer.encoder.feat_in", enc_feat_in);
    RD_U32("stt.sortformer.encoder.pos_emb_max_len", enc_pos_emb_max_len);
    RD_STR("stt.sortformer.encoder.conv_norm_type", enc_conv_norm_type);

    RD_U32("stt.sortformer.transformer.n_layers", tf_n_layers);
    RD_U32("stt.sortformer.transformer.d_model", tf_d_model);
    RD_U32("stt.sortformer.transformer.n_heads", tf_n_heads);
    RD_U32("stt.sortformer.transformer.d_ff", tf_d_ff);
    RD_STR("stt.sortformer.transformer.activation", tf_activation);
    {
        const int64_t id = gguf_find_key(g, "stt.sortformer.transformer.pre_ln");
        hp.tf_pre_ln     = (id >= 0) && gguf_get_val_bool(g, id);
    }

    RD_U32("stt.frontend.num_mels", fe_num_mels);
    RD_U32("stt.frontend.sample_rate", fe_sample_rate);
    RD_U32("stt.frontend.n_fft", fe_n_fft);
    RD_U32("stt.frontend.win_length", fe_win_length);
    RD_U32("stt.frontend.hop_length", fe_hop_length);
    RD_STR("stt.frontend.window", fe_window);
    RD_STR("stt.frontend.normalize", fe_normalize);
    RD_F32("stt.frontend.dither", fe_dither);
    RD_F32("stt.frontend.pre_emphasis", fe_pre_emphasis);

    RD_U32("stt.sortformer.stream.chunk_len", stream_chunk_len);
    RD_U32("stt.sortformer.stream.spkcache_len", stream_spkcache_len);
    RD_U32("stt.sortformer.stream.fifo_len", stream_fifo_len);
    RD_U32("stt.sortformer.stream.spkcache_update_period", stream_spkcache_update_period);

#undef RD_U32
#undef RD_F32
#undef RD_STR

    if (hp.tf_n_heads == 0 || hp.tf_d_model % hp.tf_n_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: tf_d_model %d not divisible by tf_n_heads %d",
                hp.tf_d_model, hp.tf_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.tf_pre_ln) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: pre_ln=true transformer not supported (post-LN only)");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

namespace {

// Look up a tensor by name and validate its ne against the expected dims
// (fast-to-slow; pass -1 to skip a dim). Returns null + logs on failure.
ggml_tensor * get_checked(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing tensor %s", name);
        return nullptr;
    }
    if ((ne0 >= 0 && t->ne[0] != ne0) || (ne1 >= 0 && t->ne[1] != ne1)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "sortformer: tensor %s shape mismatch: have [%lld,%lld] want [%lld,%lld]", name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) ne0, (long long) ne1);
        return nullptr;
    }
    return t;
}

}  // namespace

transcribe_status build_sortformer_weights(ggml_context * ctx, const SortformerHParams & hp, SortformerWeights & w) {
    const int64_t d    = hp.tf_d_model;
    const int64_t dff  = hp.tf_d_ff;
    const int64_t ed   = hp.enc_d_model;
    const int64_t spk  = hp.max_speakers;
    char          name[128];

#define GET(dst, nm, e0, e1)                                     \
    do {                                                         \
        (dst) = get_checked(ctx, (nm), (e0), (e1));              \
        if ((dst) == nullptr) return TRANSCRIBE_ERR_GGUF;        \
    } while (0)

    GET(w.enc_proj_w, "diar.encoder_proj.weight", ed, d);
    GET(w.enc_proj_b, "diar.encoder_proj.bias", d, -1);

    w.tf_blocks.resize(static_cast<size_t>(hp.tf_n_layers));
    for (int i = 0; i < hp.tf_n_layers; ++i) {
        SortformerTfBlock & b = w.tf_blocks[static_cast<size_t>(i)];
#define GETB(dst, suffix, e0, e1)                                \
        do {                                                     \
            std::snprintf(name, sizeof(name), "tf.blocks.%d.%s", i, suffix);   \
            GET(dst, name, e0, e1);                              \
        } while (0)
        GETB(b.norm1_w, "norm_1.weight", d, -1);
        GETB(b.norm1_b, "norm_1.bias", d, -1);
        GETB(b.attn_q_w, "attn.q.weight", d, d);
        GETB(b.attn_q_b, "attn.q.bias", d, -1);
        GETB(b.attn_k_w, "attn.k.weight", d, d);
        GETB(b.attn_k_b, "attn.k.bias", d, -1);
        GETB(b.attn_v_w, "attn.v.weight", d, d);
        GETB(b.attn_v_b, "attn.v.bias", d, -1);
        GETB(b.attn_o_w, "attn.out.weight", d, d);
        GETB(b.attn_o_b, "attn.out.bias", d, -1);
        GETB(b.norm2_w, "norm_2.weight", d, -1);
        GETB(b.norm2_b, "norm_2.bias", d, -1);
        GETB(b.ff_in_w, "ff.in.weight", d, dff);
        GETB(b.ff_in_b, "ff.in.bias", dff, -1);
        GETB(b.ff_out_w, "ff.out.weight", dff, d);
        GETB(b.ff_out_b, "ff.out.bias", d, -1);
#undef GETB
    }

    GET(w.fc1_w, "diar.fc1.weight", d, d);
    GET(w.fc1_b, "diar.fc1.bias", d, -1);
    GET(w.single_spk_head_w, "diar.single_spk_head.weight", d, spk);
    GET(w.single_spk_head_b, "diar.single_spk_head.bias", spk, -1);
    GET(w.spk_head_w, "diar.spk_head.weight", 2 * d, spk);
    GET(w.spk_head_b, "diar.spk_head.bias", spk, -1);

#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::sortformer
