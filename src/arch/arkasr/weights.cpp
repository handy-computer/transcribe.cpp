#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

namespace transcribe::arkasr {

namespace {
constexpr const char * kTag = "arkasr";

#define READ_U32(key, field)                                                                        \
    do {                                                                                            \
        if (const auto st = read_required_u32_kv(gguf, (key), kTag, hp.field); st != TRANSCRIBE_OK) \
            return st;                                                                              \
    } while (0)
#define READ_F32(key, field)                                                                        \
    do {                                                                                            \
        if (const auto st = read_required_f32_kv(gguf, (key), kTag, hp.field); st != TRANSCRIBE_OK) \
            return st;                                                                              \
    } while (0)

#define GET_F32(slot, name, ...)                                                                            \
    do {                                                                                                    \
        auto * t = transcribe::weights::find_tensor(ctx, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (t == nullptr)                                                                                   \
            return TRANSCRIBE_ERR_GGUF;                                                                     \
        (slot) = t;                                                                                         \
    } while (0)
#define GET_CONV(slot, name, ...)                                                                                  \
    do {                                                                                                           \
        auto * t =                                                                                                 \
            transcribe::weights::find_tensor(ctx, (name), { TRANSCRIBE_QUANT_CONV_TYPES }, { __VA_ARGS__ }, kTag); \
        if (t == nullptr)                                                                                          \
            return TRANSCRIBE_ERR_GGUF;                                                                            \
        (slot) = t;                                                                                                \
    } while (0)
#define GET_LIN(slot, name, ...)                                                                                     \
    do {                                                                                                             \
        auto * t =                                                                                                   \
            transcribe::weights::find_tensor(ctx, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, { __VA_ARGS__ }, kTag); \
        if (t == nullptr)                                                                                            \
            return TRANSCRIBE_ERR_GGUF;                                                                              \
        (slot) = t;                                                                                                  \
    } while (0)
}  // namespace

transcribe_status read_arkasr_hparams(const gguf_context * gguf, ArkAsrHParams & hp) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    READ_U32("arkasr.llm.hidden_size", dec_hidden);
    READ_U32("arkasr.llm.num_layers", dec_n_layers);
    READ_U32("arkasr.llm.num_heads", dec_n_heads);
    READ_U32("arkasr.llm.num_kv_heads", dec_n_kv_heads);
    READ_U32("arkasr.llm.head_dim", dec_head_dim);
    READ_U32("arkasr.llm.intermediate_size", dec_intermediate);
    READ_U32("arkasr.llm.vocab_size", dec_vocab_size);
    READ_U32("arkasr.llm.max_position_embeddings", dec_max_position);
    READ_F32("arkasr.llm.rope_theta", dec_rope_theta);
    READ_F32("arkasr.llm.rms_norm_eps", dec_rms_eps);

    READ_U32("arkasr.whisper.d_model", enc_d_model);
    READ_U32("arkasr.whisper.num_layers", enc_n_layers);
    READ_U32("arkasr.whisper.num_heads", enc_n_heads);
    READ_U32("arkasr.whisper.head_dim", enc_head_dim);
    READ_U32("arkasr.whisper.ffn_dim", enc_ffn_dim);
    READ_U32("arkasr.whisper.num_mel_bins", enc_num_mels);
    READ_U32("arkasr.whisper.max_source_positions", enc_max_position);
    READ_U32("arkasr.whisper.rot_dim", enc_rot_dim);
    READ_F32("arkasr.whisper.rope_theta", enc_rope_theta);
    READ_F32("arkasr.whisper.ln_eps", enc_ln_eps);

    READ_U32("arkasr.adapter.merge_factor", merge_factor);
    READ_U32("arkasr.audio_token_id", audio_token_id);
    READ_U32("arkasr.bos_token_id", bos_token_id);
    READ_U32("arkasr.eos_token_id", eos_token_id);
    READ_U32("arkasr.pad_token_id", pad_token_id);
    READ_U32("arkasr.n_fft", n_fft);
    READ_U32("arkasr.hop_length", hop_length);
    READ_U32("arkasr.sample_rate", sample_rate);

    if (hp.dec_hidden <= 0 || hp.dec_n_layers <= 0 || hp.dec_n_heads <= 0 || hp.dec_n_kv_heads <= 0 ||
        hp.dec_head_dim <= 0 || hp.dec_intermediate <= 0 || hp.dec_vocab_size <= 0 || hp.dec_max_position <= 0 ||
        hp.dec_hidden != hp.dec_n_heads * hp.dec_head_dim || hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: invalid Qwen2.5 decoder dimensions");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model <= 0 || hp.enc_n_layers <= 0 || hp.enc_n_heads <= 0 || hp.enc_head_dim <= 0 ||
        hp.enc_ffn_dim <= 0 || hp.enc_num_mels <= 0 || hp.enc_d_model != hp.enc_n_heads * hp.enc_head_dim ||
        hp.enc_rot_dim <= 0 || hp.enc_rot_dim > hp.enc_head_dim || (hp.enc_rot_dim % 2) != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: invalid Whisper encoder dimensions");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.merge_factor <= 0 || hp.sample_rate != 16000 || hp.n_fft <= 0 || hp.hop_length <= 0 ||
        hp.audio_token_id < 0 || hp.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: invalid adapter, frontend, or token metadata");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status build_arkasr_weights(ggml_context * ctx, const ArkAsrHParams & hp, ArkAsrWeights & w) {
    if (ctx == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t ed = hp.enc_d_model;
    const int64_t ef = hp.enc_ffn_dim;
    const int64_t dh = hp.dec_hidden;
    const int64_t di = hp.dec_intermediate;
    const int64_t qd = static_cast<int64_t>(hp.dec_n_heads) * hp.dec_head_dim;
    const int64_t kd = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;
    const int64_t ah = static_cast<int64_t>(hp.dec_hidden) * 2;

    GET_CONV(w.conv1_w, "enc.conv1.weight", 3, hp.enc_num_mels, ed);
    GET_F32(w.conv1_b, "enc.conv1.bias", ed);
    GET_CONV(w.conv2_w, "enc.conv2.weight", 3, ed, ed);
    GET_F32(w.conv2_b, "enc.conv2.bias", ed);

    w.enc_blocks.assign(hp.enc_n_layers, ArkAsrEncBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = w.enc_blocks[i];
        GET_F32(b.attn_ln_w, transcribe::weights::lname("enc.blk.%d.attn_ln.weight", i), ed);
        GET_F32(b.attn_ln_b, transcribe::weights::lname("enc.blk.%d.attn_ln.bias", i), ed);
        GET_LIN(b.q_w, transcribe::weights::lname("enc.blk.%d.attn.q.weight", i), ed, ed);
        GET_F32(b.q_b, transcribe::weights::lname("enc.blk.%d.attn.q.bias", i), ed);
        GET_LIN(b.k_w, transcribe::weights::lname("enc.blk.%d.attn.k.weight", i), ed, ed);
        GET_LIN(b.v_w, transcribe::weights::lname("enc.blk.%d.attn.v.weight", i), ed, ed);
        GET_F32(b.v_b, transcribe::weights::lname("enc.blk.%d.attn.v.bias", i), ed);
        GET_LIN(b.o_w, transcribe::weights::lname("enc.blk.%d.attn.o.weight", i), ed, ed);
        GET_F32(b.o_b, transcribe::weights::lname("enc.blk.%d.attn.o.bias", i), ed);
        GET_F32(b.ffn_ln_w, transcribe::weights::lname("enc.blk.%d.ffn_ln.weight", i), ed);
        GET_F32(b.ffn_ln_b, transcribe::weights::lname("enc.blk.%d.ffn_ln.bias", i), ed);
        GET_LIN(b.fc1_w, transcribe::weights::lname("enc.blk.%d.fc1.weight", i), ed, ef);
        GET_F32(b.fc1_b, transcribe::weights::lname("enc.blk.%d.fc1.bias", i), ef);
        GET_LIN(b.fc2_w, transcribe::weights::lname("enc.blk.%d.fc2.weight", i), ef, ed);
        GET_F32(b.fc2_b, transcribe::weights::lname("enc.blk.%d.fc2.bias", i), ed);
    }

    GET_F32(w.adapter_ln_w, "adapter.ln.weight", ed);
    GET_F32(w.adapter_ln_b, "adapter.ln.bias", ed);
    GET_LIN(w.adapter_fc1_w, "adapter.fc1.weight", ed * hp.merge_factor, ah);
    GET_F32(w.adapter_fc1_b, "adapter.fc1.bias", ah);
    GET_LIN(w.adapter_fc2_w, "adapter.fc2.weight", ah, dh);
    GET_F32(w.adapter_fc2_b, "adapter.fc2.bias", dh);

    GET_LIN(w.dec_embed_w, "dec.embed.weight", dh, hp.dec_vocab_size);
    GET_F32(w.dec_norm_w, "dec.norm.weight", dh);
    w.dec_blocks.assign(hp.dec_n_layers, ArkAsrDecBlock{});
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        auto & b = w.dec_blocks[i];
        GET_F32(b.attn_norm_w, transcribe::weights::lname("dec.blk.%d.attn_norm.weight", i), dh);
        GET_LIN(b.q_w, transcribe::weights::lname("dec.blk.%d.attn.q.weight", i), dh, qd);
        GET_F32(b.q_b, transcribe::weights::lname("dec.blk.%d.attn.q.bias", i), qd);
        GET_LIN(b.k_w, transcribe::weights::lname("dec.blk.%d.attn.k.weight", i), dh, kd);
        GET_F32(b.k_b, transcribe::weights::lname("dec.blk.%d.attn.k.bias", i), kd);
        GET_LIN(b.v_w, transcribe::weights::lname("dec.blk.%d.attn.v.weight", i), dh, kd);
        GET_F32(b.v_b, transcribe::weights::lname("dec.blk.%d.attn.v.bias", i), kd);
        GET_LIN(b.o_w, transcribe::weights::lname("dec.blk.%d.attn.o.weight", i), qd, dh);
        GET_F32(b.ffn_norm_w, transcribe::weights::lname("dec.blk.%d.ffn_norm.weight", i), dh);
        GET_LIN(b.ffn_gate_w, transcribe::weights::lname("dec.blk.%d.ffn.gate.weight", i), dh, di);
        GET_LIN(b.ffn_up_w, transcribe::weights::lname("dec.blk.%d.ffn.up.weight", i), dh, di);
        GET_LIN(b.ffn_down_w, transcribe::weights::lname("dec.blk.%d.ffn.down.weight", i), di, dh);
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::arkasr
