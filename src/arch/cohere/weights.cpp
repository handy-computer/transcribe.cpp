// arch/cohere/weights.cpp - implementation of read_cohere_hparams
// and build_cohere_weights.
//
// Pattern follows parakeet/weights.cpp exactly:
//   - read every required hparam from KV explicitly
//   - build the tensor catalog as a sequence of get_tensor() calls
//     with explicit expected shapes
//   - missing tensor or shape mismatch -> TRANSCRIBE_ERR_GGUF

#include "weights.h"

#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace transcribe::cohere {

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------

namespace {

transcribe_status read_required_u32(const gguf_context * gguf,
                                    const char *         key,
                                    int32_t &            out)
{
    uint32_t v = 0;
    switch (read_uint32_kv(gguf, key, v)) {
        case KvResult::Ok:
            out = static_cast<int32_t>(v);
            return TRANSCRIBE_OK;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "cohere: required KV \"%s\" missing or wrong type\n",
                         key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF;
}

transcribe_status read_required_f32(const gguf_context * gguf,
                                    const char *         key,
                                    float &              out)
{
    float v = 0.0f;
    switch (read_float32_kv(gguf, key, v)) {
        case KvResult::Ok:
            out = v;
            return TRANSCRIBE_OK;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "cohere: required KV \"%s\" missing or wrong type\n",
                         key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF;
}

transcribe_status read_required_string(const gguf_context * gguf,
                                       const char *         key,
                                       std::string &        out)
{
    std::string v;
    switch (read_string_kv(gguf, key, v)) {
        case KvResult::Ok:
            out = std::move(v);
            return TRANSCRIBE_OK;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "cohere: required KV \"%s\" missing or wrong type\n",
                         key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF;
}

transcribe_status read_optional_bool(const gguf_context * gguf,
                                     const char *         key,
                                     bool                 default_value,
                                     bool &               out)
{
    bool tmp = default_value;
    switch (read_bool_kv(gguf, key, tmp)) {
        case KvResult::Absent:
            out = default_value;
            return TRANSCRIBE_OK;
        case KvResult::Ok:
            out = tmp;
            return TRANSCRIBE_OK;
        case KvResult::BadType:
            std::fprintf(stderr,
                         "cohere: optional KV \"%s\" has wrong type\n", key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF;
}

transcribe_status read_optional_string(const gguf_context * gguf,
                                       const char *         key,
                                       const char *         default_value,
                                       std::string &        out)
{
    std::string v;
    switch (read_string_kv(gguf, key, v)) {
        case KvResult::Absent:
            out = default_value;
            return TRANSCRIBE_OK;
        case KvResult::Ok:
            out = std::move(v);
            return TRANSCRIBE_OK;
        case KvResult::BadType:
            std::fprintf(stderr,
                         "cohere: optional KV \"%s\" has wrong type\n", key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF;
}

} // namespace

transcribe_status read_cohere_hparams(const gguf_context * gguf,
                                      CohereHParams &      hp)
{
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Encoder.
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.n_layers",           hp.enc_n_layers);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.d_model",            hp.enc_d_model);            st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.n_heads",            hp.enc_n_heads);            st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.d_ff",               hp.enc_d_ff);               st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.conv_kernel",        hp.enc_conv_kernel);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.subsampling_factor", hp.enc_subsampling_factor); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.subsampling_channels", hp.enc_subsampling_channels); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.encoder.pos_emb_max_len",    hp.enc_pos_emb_max_len);    st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool(gguf, "stt.cohere.encoder.use_bias", true, hp.enc_use_bias); st != TRANSCRIBE_OK) return st;

    // Decoder.
    if (auto st = read_required_u32(gguf, "stt.cohere.decoder.n_layers",      hp.dec_n_layers); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.decoder.hidden_size",   hp.dec_hidden);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.decoder.n_heads",       hp.dec_n_heads);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.decoder.inner_size",    hp.dec_inner);    st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.cohere.decoder.max_seq_len",   hp.dec_max_seq);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string(gguf, "stt.cohere.decoder.activation", hp.dec_activation); st != TRANSCRIBE_OK) return st;

    // Token IDs. vocab_size comes from the tokenizer; decoder_start_token_id is explicit.
    if (auto st = read_required_u32(gguf, "stt.cohere.decoder_start_token_id", hp.decoder_start_token_id); st != TRANSCRIBE_OK) return st;

    // Head.
    if (auto st = read_optional_bool(gguf, "stt.cohere.head.log_softmax", true, hp.head_log_softmax); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool(gguf, "stt.cohere.head.tied_weights", true, hp.head_tied_weights); st != TRANSCRIBE_OK) return st;

    // Frontend.
    if (auto st = read_required_string(gguf, "stt.frontend.type",         hp.fe_type);         st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.num_mels",     hp.fe_num_mels);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.sample_rate",  hp.fe_sample_rate);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.n_fft",        hp.fe_n_fft);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.win_length",   hp.fe_win_length);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.hop_length",   hp.fe_hop_length);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string(gguf, "stt.frontend.window",       hp.fe_window);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string(gguf, "stt.frontend.normalize",    hp.fe_normalize);    st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.dither",       hp.fe_dither);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.pre_emphasis", hp.fe_pre_emphasis); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.f_min",        hp.fe_f_min);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.f_max",        hp.fe_f_max);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_string(gguf, "stt.frontend.pad_mode", "reflect", hp.fe_pad_mode); st != TRANSCRIBE_OK) return st;

    // Cross-field invariants.
    if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 ||
        hp.enc_d_ff <= 0 || hp.enc_conv_kernel <= 0 ||
        hp.enc_subsampling_factor <= 0 || hp.enc_subsampling_channels <= 0)
    {
        std::fprintf(stderr, "cohere: encoder hparams must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model % hp.enc_n_heads != 0) {
        std::fprintf(stderr,
                     "cohere: encoder d_model (%d) not divisible by n_heads (%d)\n",
                     hp.enc_d_model, hp.enc_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_n_layers <= 0 || hp.dec_hidden <= 0 || hp.dec_n_heads <= 0 ||
        hp.dec_inner <= 0 || hp.dec_max_seq <= 0)
    {
        std::fprintf(stderr, "cohere: decoder hparams must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_hidden % hp.dec_n_heads != 0) {
        std::fprintf(stderr,
                     "cohere: decoder hidden (%d) not divisible by n_heads (%d)\n",
                     hp.dec_hidden, hp.dec_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_activation != "relu" && hp.dec_activation != "silu" &&
        hp.dec_activation != "swish")
    {
        std::fprintf(stderr,
                     "cohere: unsupported decoder activation \"%s\" "
                     "(only relu, silu, swish are implemented)\n",
                     hp.dec_activation.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_num_mels <= 0 || hp.fe_sample_rate <= 0 ||
        hp.fe_n_fft <= 0 || hp.fe_win_length <= 0 || hp.fe_hop_length <= 0)
    {
        std::fprintf(stderr, "cohere: frontend dimensions must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_win_length > hp.fe_n_fft) {
        std::fprintf(stderr,
                     "cohere: frontend win_length (%d) > n_fft (%d)\n",
                     hp.fe_win_length, hp.fe_n_fft);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_f_min < 0.0f || hp.fe_f_max <= hp.fe_f_min) {
        std::fprintf(stderr,
                     "cohere: frontend mel band invalid: f_min=%f f_max=%f\n",
                     hp.fe_f_min, hp.fe_f_max);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_type != "mel") {
        std::fprintf(stderr,
                     "cohere: unsupported frontend type \"%s\" (only \"mel\")\n",
                     hp.fe_type.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_window != "hann") {
        std::fprintf(stderr,
                     "cohere: unsupported frontend window \"%s\" "
                     "(only \"hann\" is implemented)\n",
                     hp.fe_window.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_normalize != "per_feature") {
        std::fprintf(stderr,
                     "cohere: unsupported frontend normalize \"%s\" "
                     "(only \"per_feature\" is implemented)\n",
                     hp.fe_normalize.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if ((hp.fe_n_fft & (hp.fe_n_fft - 1)) != 0) {
        std::fprintf(stderr,
                     "cohere: frontend n_fft (%d) must be a power of 2\n",
                     hp.fe_n_fft);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_pad_mode != "reflect" && hp.fe_pad_mode != "constant") {
        std::fprintf(stderr,
                     "cohere: unsupported frontend pad_mode \"%s\" "
                     "(only \"reflect\" and \"constant\" are implemented)\n",
                     hp.fe_pad_mode.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

namespace {

const char * lname(const char * fmt, int layer_idx) {
    thread_local char buf[128];
    std::snprintf(buf, sizeof(buf), fmt, layer_idx);
    return buf;
}

// The canonical find_tensor() helper lives in
// src/transcribe-weights-util.{h,cpp}; see that header for rationale.
// It is shared between every per-family weights.cpp. The GET_*
// macros below still live here because their type allowlists encode
// a per-family quantization policy, not a shared convention.
constexpr const char * kTag = "cohere";

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
            {GGML_TYPE_F32, GGML_TYPE_F16}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_LIN(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {GGML_TYPE_F32, GGML_TYPE_F16, \
             GGML_TYPE_BF16, \
             GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, \
             GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, \
             GGML_TYPE_Q8_0, \
             GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, \
             GGML_TYPE_Q6_K}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

} // namespace

transcribe_status build_cohere_weights(const gguf_context *   gguf,
                                       ggml_context *         ctx_meta,
                                       const CohereHParams &  hp,
                                       CohereWeights &        weights)
{
    if (gguf == nullptr || ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t channels = hp.enc_subsampling_channels;
    const int64_t d_model  = hp.enc_d_model;
    const int64_t d_ff     = hp.enc_d_ff;
    const int64_t n_heads  = hp.enc_n_heads;
    const int64_t head_dim = hp.enc_head_dim();
    const int64_t k        = hp.enc_conv_kernel;
    const int64_t dec_h    = hp.dec_hidden;
    const int64_t dec_nh   = hp.dec_n_heads;
    const int64_t dec_hd   = hp.dec_head_dim();
    const int64_t dec_in   = hp.dec_inner;

    (void)dec_nh;
    (void)dec_hd;

    const int64_t pre_encode_freq = channels * (hp.fe_num_mels / hp.enc_subsampling_factor);
    const int64_t pre_encode_in   = pre_encode_freq;

    // ----- pre_encode -----
    GET_CONV(weights.pre_encode.conv0_w, "enc.pre_encode.conv.0.weight", 3, 3, 1,        channels);
    GET_F32 (weights.pre_encode.conv0_b, "enc.pre_encode.conv.0.bias", channels);
    GET_CONV(weights.pre_encode.conv2_w, "enc.pre_encode.conv.2.weight", 3, 3, 1,        channels);
    GET_F32 (weights.pre_encode.conv2_b, "enc.pre_encode.conv.2.bias", channels);
    GET_CONV(weights.pre_encode.conv3_w, "enc.pre_encode.conv.3.weight", 1, 1, channels, channels);
    GET_F32 (weights.pre_encode.conv3_b, "enc.pre_encode.conv.3.bias", channels);
    GET_CONV(weights.pre_encode.conv5_w, "enc.pre_encode.conv.5.weight", 3, 3, 1,        channels);
    GET_F32 (weights.pre_encode.conv5_b, "enc.pre_encode.conv.5.bias", channels);
    GET_CONV(weights.pre_encode.conv6_w, "enc.pre_encode.conv.6.weight", 1, 1, channels, channels);
    GET_F32 (weights.pre_encode.conv6_b, "enc.pre_encode.conv.6.bias", channels);
    GET_LIN (weights.pre_encode.out_w,   "enc.pre_encode.out.weight", pre_encode_in, d_model);
    GET_F32 (weights.pre_encode.out_b,   "enc.pre_encode.out.bias", d_model);

    // ----- encoder blocks -----
    weights.blocks.assign(hp.enc_n_layers, CohereBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = weights.blocks[i];

        // Macaron FF1 (with bias).
        GET_F32(b.norm_ff1_w, lname("enc.blocks.%d.norm_ff1.weight", i), d_model);
        GET_F32(b.norm_ff1_b, lname("enc.blocks.%d.norm_ff1.bias",   i), d_model);
        GET_LIN(b.ff1_lin1_w, lname("enc.blocks.%d.ff1.linear1.weight", i), d_model, d_ff);
        GET_F32(b.ff1_lin1_b, lname("enc.blocks.%d.ff1.linear1.bias",   i), d_ff);
        GET_LIN(b.ff1_lin2_w, lname("enc.blocks.%d.ff1.linear2.weight", i), d_ff,    d_model);
        GET_F32(b.ff1_lin2_b, lname("enc.blocks.%d.ff1.linear2.bias",   i), d_model);

        // Self-attention with relative position (with bias on Q, K, V, out).
        GET_F32(b.norm_attn_w, lname("enc.blocks.%d.norm_attn.weight", i), d_model);
        GET_F32(b.norm_attn_b, lname("enc.blocks.%d.norm_attn.bias",   i), d_model);
        GET_LIN(b.attn_q_w,    lname("enc.blocks.%d.attn.linear_q.weight",   i), d_model, d_model);
        GET_F32(b.attn_q_b,    lname("enc.blocks.%d.attn.linear_q.bias",     i), d_model);
        GET_LIN(b.attn_k_w,    lname("enc.blocks.%d.attn.linear_k.weight",   i), d_model, d_model);
        GET_F32(b.attn_k_b,    lname("enc.blocks.%d.attn.linear_k.bias",     i), d_model);
        GET_LIN(b.attn_v_w,    lname("enc.blocks.%d.attn.linear_v.weight",   i), d_model, d_model);
        GET_F32(b.attn_v_b,    lname("enc.blocks.%d.attn.linear_v.bias",     i), d_model);
        GET_LIN(b.attn_out_w,  lname("enc.blocks.%d.attn.linear_out.weight", i), d_model, d_model);
        GET_F32(b.attn_out_b,  lname("enc.blocks.%d.attn.linear_out.bias",   i), d_model);
        GET_LIN(b.attn_pos_w,  lname("enc.blocks.%d.attn.linear_pos.weight", i), d_model, d_model);
        GET_F32(b.attn_pos_u,  lname("enc.blocks.%d.attn.pos_bias_u",        i), head_dim, n_heads);
        GET_F32(b.attn_pos_v,  lname("enc.blocks.%d.attn.pos_bias_v",        i), head_dim, n_heads);

        // Conv module (with bias on pointwise and depthwise).
        GET_F32 (b.norm_conv_w, lname("enc.blocks.%d.norm_conv.weight", i), d_model);
        GET_F32 (b.norm_conv_b, lname("enc.blocks.%d.norm_conv.bias",   i), d_model);
        GET_CONV(b.conv_pw1_w,  lname("enc.blocks.%d.conv.pointwise1.weight", i), 1, d_model, 2 * d_model);
        GET_F32 (b.conv_pw1_b,  lname("enc.blocks.%d.conv.pointwise1.bias",   i), 2 * d_model);
        GET_CONV(b.conv_dw_w,   lname("enc.blocks.%d.conv.depthwise.weight",  i), k, 1,       d_model);
        GET_F32 (b.conv_dw_b,   lname("enc.blocks.%d.conv.depthwise.bias",    i), d_model);
        GET_CONV(b.conv_pw2_w,  lname("enc.blocks.%d.conv.pointwise2.weight", i), 1, d_model, d_model);
        GET_F32 (b.conv_pw2_b,  lname("enc.blocks.%d.conv.pointwise2.bias",   i), d_model);
        GET_F32 (b.conv_bn_w,   lname("enc.blocks.%d.conv.bn.weight",       i), d_model);
        GET_F32 (b.conv_bn_b,   lname("enc.blocks.%d.conv.bn.bias",         i), d_model);
        GET_F32 (b.conv_bn_rm,  lname("enc.blocks.%d.conv.bn.running_mean", i), d_model);
        GET_F32 (b.conv_bn_rv,  lname("enc.blocks.%d.conv.bn.running_var",  i), d_model);

        // Macaron FF2 (with bias).
        GET_F32(b.norm_ff2_w, lname("enc.blocks.%d.norm_ff2.weight", i), d_model);
        GET_F32(b.norm_ff2_b, lname("enc.blocks.%d.norm_ff2.bias",   i), d_model);
        GET_LIN(b.ff2_lin1_w, lname("enc.blocks.%d.ff2.linear1.weight", i), d_model, d_ff);
        GET_F32(b.ff2_lin1_b, lname("enc.blocks.%d.ff2.linear1.bias",   i), d_ff);
        GET_LIN(b.ff2_lin2_w, lname("enc.blocks.%d.ff2.linear2.weight", i), d_ff,    d_model);
        GET_F32(b.ff2_lin2_b, lname("enc.blocks.%d.ff2.linear2.bias",   i), d_model);

        // Final per-block layer norm.
        GET_F32(b.norm_out_w, lname("enc.blocks.%d.norm_out.weight", i), d_model);
        GET_F32(b.norm_out_b, lname("enc.blocks.%d.norm_out.bias",   i), d_model);
    }

    // ----- encoder-decoder projection -----
    GET_LIN(weights.enc_dec_proj.weight, "enc_dec_proj.weight", d_model, dec_h);
    GET_F32(weights.enc_dec_proj.bias,   "enc_dec_proj.bias",   dec_h);

    // ----- decoder embedding -----
    // Token embedding: ne=[dec_hidden, vocab_size] in ggml (PyTorch [vocab, hidden]).
    // We read vocab_size from the tensor shape since we don't have it as a separate KV.
    {
        ggml_tensor * tw = ggml_get_tensor(ctx_meta, "dec.embed.token.weight");
        if (tw == nullptr) {
            std::fprintf(stderr, "cohere: missing tensor \"dec.embed.token.weight\"\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        // The embedding table: ne[0]=dec_hidden, ne[1]=vocab_size.
        if (tw->ne[0] != dec_h) {
            std::fprintf(stderr,
                         "cohere: dec.embed.token.weight ne[0]=%lld, expected %lld\n",
                         static_cast<long long>(tw->ne[0]),
                         static_cast<long long>(dec_h));
            return TRANSCRIBE_ERR_GGUF;
        }
        weights.dec_embed.token_w = tw;
    }

    // Read vocab_size from the token embedding shape.
    const int64_t vocab_size = weights.dec_embed.token_w->ne[1];

    // Positional encoding: [dec_hidden, dec_max_seq].
    GET_F32(weights.dec_embed.pos_enc, "dec.embed.pos_enc", dec_h, hp.dec_max_seq);
    GET_F32(weights.dec_embed.norm_w,  "dec.embed.norm.weight", dec_h);
    GET_F32(weights.dec_embed.norm_b,  "dec.embed.norm.bias",   dec_h);

    // ----- decoder blocks -----
    weights.dec_blocks.assign(hp.dec_n_layers, CohereDecBlock{});
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        auto & db = weights.dec_blocks[i];

        // Self-attention.
        GET_F32(db.norm_self_w, lname("dec.blocks.%d.norm_self.weight", i), dec_h);
        GET_F32(db.norm_self_b, lname("dec.blocks.%d.norm_self.bias",   i), dec_h);
        GET_LIN(db.self_q_w,   lname("dec.blocks.%d.self_attn.q.weight", i), dec_h, dec_h);
        GET_F32(db.self_q_b,   lname("dec.blocks.%d.self_attn.q.bias",   i), dec_h);
        GET_LIN(db.self_k_w,   lname("dec.blocks.%d.self_attn.k.weight", i), dec_h, dec_h);
        GET_F32(db.self_k_b,   lname("dec.blocks.%d.self_attn.k.bias",   i), dec_h);
        GET_LIN(db.self_v_w,   lname("dec.blocks.%d.self_attn.v.weight", i), dec_h, dec_h);
        GET_F32(db.self_v_b,   lname("dec.blocks.%d.self_attn.v.bias",   i), dec_h);
        GET_LIN(db.self_out_w, lname("dec.blocks.%d.self_attn.out.weight", i), dec_h, dec_h);
        GET_F32(db.self_out_b, lname("dec.blocks.%d.self_attn.out.bias",   i), dec_h);

        // Cross-attention.
        GET_F32(db.norm_cross_w, lname("dec.blocks.%d.norm_cross.weight", i), dec_h);
        GET_F32(db.norm_cross_b, lname("dec.blocks.%d.norm_cross.bias",   i), dec_h);
        GET_LIN(db.cross_q_w,   lname("dec.blocks.%d.cross_attn.q.weight", i), dec_h, dec_h);
        GET_F32(db.cross_q_b,   lname("dec.blocks.%d.cross_attn.q.bias",   i), dec_h);
        GET_LIN(db.cross_k_w,   lname("dec.blocks.%d.cross_attn.k.weight", i), dec_h, dec_h);
        GET_F32(db.cross_k_b,   lname("dec.blocks.%d.cross_attn.k.bias",   i), dec_h);
        GET_LIN(db.cross_v_w,   lname("dec.blocks.%d.cross_attn.v.weight", i), dec_h, dec_h);
        GET_F32(db.cross_v_b,   lname("dec.blocks.%d.cross_attn.v.bias",   i), dec_h);
        GET_LIN(db.cross_out_w, lname("dec.blocks.%d.cross_attn.out.weight", i), dec_h, dec_h);
        GET_F32(db.cross_out_b, lname("dec.blocks.%d.cross_attn.out.bias",   i), dec_h);

        // FFN.
        GET_F32(db.norm_ff_w, lname("dec.blocks.%d.norm_ff.weight", i), dec_h);
        GET_F32(db.norm_ff_b, lname("dec.blocks.%d.norm_ff.bias",   i), dec_h);
        GET_LIN(db.ff_in_w,  lname("dec.blocks.%d.ff.dense_in.weight",  i), dec_h, dec_in);
        GET_F32(db.ff_in_b,  lname("dec.blocks.%d.ff.dense_in.bias",    i), dec_in);
        GET_LIN(db.ff_out_w, lname("dec.blocks.%d.ff.dense_out.weight", i), dec_in, dec_h);
        GET_F32(db.ff_out_b, lname("dec.blocks.%d.ff.dense_out.bias",   i), dec_h);
    }

    // ----- decoder final norm -----
    GET_F32(weights.dec_final.norm_w, "dec.final_norm.weight", dec_h);
    GET_F32(weights.dec_final.norm_b, "dec.final_norm.bias",   dec_h);

    // ----- head -----
    GET_F32(weights.head.bias, "head.bias", vocab_size);

    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_CONV
#undef GET_LIN

} // namespace transcribe::cohere
