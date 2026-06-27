// arch/vibevoice/weights.cpp - read_vibevoice_hparams + build_vibevoice_weights.
//
// Mirrors arch/qwen3_asr/weights.cpp: every required hparam is read
// explicitly (BadType fatal); the tensor catalog is a sequence of
// find_tensor() calls with expected shapes. The converter's tensor names
// (scripts/convert-vibevoice.py) are the contract.
//
// VAE conv kernels: stride is NOT stored — it is derived from the kernel at
// graph-build time (kernel = 2*stride for the strided downsample convs). The
// loader therefore validates only the channel dims (ne[1], ne[2]) against the
// known stage channels and accepts the kernel (ne[0]) the file carries.

#include "vibevoice.h"

#include "transcribe-meta.h"
#include "transcribe-weights-util.h"
#include "transcribe-log.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace transcribe::vibevoice {

using transcribe::weights::find_tensor;
using transcribe::weights::lname;

namespace {
constexpr const char * kFamilyTag = "vibevoice";

// Split an "a-b-c" depths string into ints. Returns false on a malformed
// entry (empty / non-numeric).
bool parse_depths(const std::string & s, std::vector<int32_t> & out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find('-', i);
        if (j == std::string::npos) j = s.size();
        const std::string tok = s.substr(i, j - i);
        if (tok.empty()) return false;
        char * end = nullptr;
        const long v = std::strtol(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || v <= 0) return false;
        out.push_back(static_cast<int32_t>(v));
        i = j + 1;
    }
    return !out.empty();
}

transcribe_status read_vae_hparams(const gguf_context * gguf,
                                   const char * stream, VaeHParams & v) {
    char key[128];
    auto K = [&](const char * suffix) -> const char * {
        std::snprintf(key, sizeof key, "stt.vibevoice.%s.%s", stream, suffix);
        return key;
    };
    if (auto st = read_required_u32_kv(gguf, K("vae_dim"),   kFamilyTag, v.vae_dim);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, K("n_filters"), kFamilyTag, v.n_filters); st != TRANSCRIBE_OK) return st;

    std::string depths_str;
    if (auto st = read_required_string_kv(gguf, K("encoder_depths"), kFamilyTag, depths_str); st != TRANSCRIBE_OK) return st;
    if (!parse_depths(depths_str, v.depths)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "%s: malformed encoder_depths '%s' for %s", kFamilyTag,
                depths_str.c_str(), stream);
        return TRANSCRIBE_ERR_GGUF;
    }

    if (auto st = read_required_string_kv(gguf, K("mixer_layer"),     kFamilyTag, v.mixer);          st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, K("layernorm"),       kFamilyTag, v.layernorm);      st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv   (gguf, K("layernorm_eps"),   kFamilyTag, v.layernorm_eps);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv   (gguf, K("layer_scale_init"),kFamilyTag, v.layer_scale_init); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv   (gguf, K("fix_std"),         kFamilyTag, v.fix_std);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, K("std_dist_type"),   kFamilyTag, v.std_dist_type);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, K("pad_mode"),        kFamilyTag, v.pad_mode);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv  (gguf, K("causal"),            kFamilyTag, true, v.causal);            st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv  (gguf, K("disable_last_norm"), kFamilyTag, true, v.disable_last_norm); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv  (gguf, K("conv_bias"),         kFamilyTag, true, v.conv_bias);         st != TRANSCRIBE_OK) return st;
    return TRANSCRIBE_OK;
}
}  // namespace

transcribe_status read_vibevoice_hparams(const gguf_context * gguf,
                                         VibeVoiceHParams &   hp) {
    if (gguf == nullptr) return TRANSCRIBE_ERR_INVALID_ARG;

    // Qwen2.5 LM.
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.n_layers",          kFamilyTag, hp.dec_n_layers);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.hidden_size",       kFamilyTag, hp.dec_hidden);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.intermediate_size", kFamilyTag, hp.dec_intermediate); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.n_heads",           kFamilyTag, hp.dec_n_heads);      st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.n_kv_heads",        kFamilyTag, hp.dec_n_kv_heads);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.head_dim",          kFamilyTag, hp.dec_head_dim);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.vibevoice.decoder.hidden_act",     kFamilyTag, hp.dec_hidden_act);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv(gguf, "stt.vibevoice.decoder.rms_norm_eps",      kFamilyTag, hp.dec_rms_norm_eps); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv(gguf, "stt.vibevoice.decoder.rope_theta",        kFamilyTag, hp.dec_rope_theta);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.max_position_embeddings", kFamilyTag, hp.dec_max_position_embeddings); st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_bool_kv(gguf, "stt.vibevoice.decoder.tie_word_embeddings", kFamilyTag, false, hp.dec_tie_word_embeddings); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.decoder.vocab_size",        kFamilyTag, hp.dec_vocab_size);   st != TRANSCRIBE_OK) return st;

    // VAE encoders.
    if (auto st = read_vae_hparams(gguf, "acoustic", hp.acoustic); st != TRANSCRIBE_OK) return st;
    if (auto st = read_vae_hparams(gguf, "semantic", hp.semantic); st != TRANSCRIBE_OK) return st;

    // Speech fusion / prompt metadata.
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.speech_start_token_id", kFamilyTag, hp.speech_start_token_id); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.speech_end_token_id",   kFamilyTag, hp.speech_end_token_id);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.speech_pad_token_id",   kFamilyTag, hp.speech_pad_token_id);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.im_start_token_id",     kFamilyTag, hp.im_start_token_id);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.im_end_token_id",       kFamilyTag, hp.im_end_token_id);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.vibevoice.speech_tok_compress_ratio", kFamilyTag, hp.speech_tok_compress_ratio); st != TRANSCRIBE_OK) return st;

    // Frontend (raw waveform).
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.sample_rate", kFamilyTag, hp.frontend_sample_rate); st != TRANSCRIBE_OK) return st;

    // Sanity.
    if (hp.dec_n_layers <= 0 || hp.dec_hidden <= 0 || hp.dec_n_heads <= 0 ||
        hp.dec_n_kv_heads <= 0 || hp.dec_head_dim <= 0 || hp.dec_intermediate <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid decoder hparams", kFamilyTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.dec_n_heads % hp.dec_n_kv_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "%s: n_heads (%d) not divisible by n_kv_heads (%d)",
                kFamilyTag, hp.dec_n_heads, hp.dec_n_kv_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.acoustic.depths.size() != hp.semantic.depths.size()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: acoustic/semantic stage count mismatch", kFamilyTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Tensor catalog
// ---------------------------------------------------------------------------

#define GET_F32(slot, name, ...)                                               \
    do {                                                                       \
        ggml_tensor * _t = find_tensor(ctx_meta, (name),                       \
            {GGML_TYPE_F32}, {__VA_ARGS__}, kFamilyTag);                       \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF;                         \
        (slot) = _t;                                                           \
    } while (0)

#define GET_LIN(slot, name, ...)                                              \
    do {                                                                       \
        ggml_tensor * _t = find_tensor(ctx_meta, (name),                       \
            {TRANSCRIBE_QUANT_LINEAR_TYPES}, {__VA_ARGS__}, kFamilyTag);       \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF;                         \
        (slot) = _t;                                                           \
    } while (0)

// Conv kernel: validate channel dims, accept the file's kernel size (ne[0]).
#define GET_CONV(slot, name, in_ch, out_ch)                                   \
    do {                                                                       \
        ggml_tensor * _p = ggml_get_tensor(ctx_meta, (name));                  \
        if (_p == nullptr) {                                                   \
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: missing tensor %s",       \
                    kFamilyTag, (name));                                       \
            return TRANSCRIBE_ERR_GGUF;                                        \
        }                                                                      \
        ggml_tensor * _t = find_tensor(ctx_meta, (name),                       \
            {TRANSCRIBE_QUANT_CONV_TYPES},                                     \
            {_p->ne[0], (int64_t)(in_ch), (int64_t)(out_ch)}, kFamilyTag);     \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF;                         \
        (slot) = _t;                                                           \
    } while (0)

namespace {

transcribe_status build_vae_encoder(ggml_context * ctx_meta, const char * stream,
                                    const VaeHParams & v, VaeEncoderWeights & w) {
    char nm[176];
    const int n_stage = static_cast<int>(v.depths.size());
    const int nf = v.n_filters;
    auto chan = [&](int i) -> int64_t { return static_cast<int64_t>(nf) << i; };  // nf * 2^i

    // Downsample layers: [0] stem (1 -> nf, stride 1); [i] strided (C[i-1] -> C[i]).
    w.downsample.resize(n_stage);
    for (int i = 0; i < n_stage; ++i) {
        const int64_t in_ch  = (i == 0) ? 1 : chan(i - 1);
        const int64_t out_ch = chan(i);
        std::snprintf(nm, sizeof nm, "enc.%s.downsample_layers.%d.0.conv.conv.weight", stream, i);
        GET_CONV(w.downsample[i].w, nm, in_ch, out_ch);
        std::snprintf(nm, sizeof nm, "enc.%s.downsample_layers.%d.0.conv.conv.bias", stream, i);
        GET_F32(w.downsample[i].b, nm, out_ch);
    }

    // Stages: per stage i (channels C[i]), v.depths[i] Block1D blocks.
    w.stages.resize(n_stage);
    for (int i = 0; i < n_stage; ++i) {
        const int64_t C = chan(i);
        const int depth = v.depths[i];
        w.stages[i].resize(depth);
        for (int j = 0; j < depth; ++j) {
            VaeBlock & b = w.stages[i][j];
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.norm.weight", stream, i, j);                 GET_F32(b.norm_w, nm, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.mixer.conv.conv.conv.weight", stream, i, j); GET_CONV(b.mixer.w, nm, 1, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.mixer.conv.conv.conv.bias", stream, i, j);   GET_F32(b.mixer.b, nm, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.gamma", stream, i, j);                       GET_F32(b.gamma, nm, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.ffn_norm.weight", stream, i, j);             GET_F32(b.ffn_norm_w, nm, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.ffn.linear1.weight", stream, i, j);          GET_LIN(b.ffn_lin1_w, nm, C, 4 * C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.ffn.linear1.bias", stream, i, j);            GET_F32(b.ffn_lin1_b, nm, 4 * C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.ffn.linear2.weight", stream, i, j);          GET_LIN(b.ffn_lin2_w, nm, 4 * C, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.ffn.linear2.bias", stream, i, j);            GET_F32(b.ffn_lin2_b, nm, C);
            std::snprintf(nm, sizeof nm, "enc.%s.stages.%d.%d.ffn_gamma", stream, i, j);                   GET_F32(b.ffn_gamma, nm, C);
        }
    }

    // Head: SConv1d C_last -> vae_dim.
    const int64_t C_last = chan(n_stage - 1);
    std::snprintf(nm, sizeof nm, "enc.%s.head.conv.conv.weight", stream); GET_CONV(w.head.w, nm, C_last, v.vae_dim);
    std::snprintf(nm, sizeof nm, "enc.%s.head.conv.conv.bias", stream);   GET_F32(w.head.b, nm, v.vae_dim);
    return TRANSCRIBE_OK;
}

transcribe_status build_connector(ggml_context * ctx_meta, const char * stream,
                                  int64_t vae_dim, int64_t hidden, ConnectorWeights & c) {
    char nm[128];
    std::snprintf(nm, sizeof nm, "conn.%s.fc1.weight", stream);  GET_LIN(c.fc1_w, nm, vae_dim, hidden);
    std::snprintf(nm, sizeof nm, "conn.%s.fc1.bias", stream);    GET_F32(c.fc1_b, nm, hidden);
    std::snprintf(nm, sizeof nm, "conn.%s.norm.weight", stream); GET_F32(c.norm_w, nm, hidden);
    std::snprintf(nm, sizeof nm, "conn.%s.fc2.weight", stream);  GET_LIN(c.fc2_w, nm, hidden, hidden);
    std::snprintf(nm, sizeof nm, "conn.%s.fc2.bias", stream);    GET_F32(c.fc2_b, nm, hidden);
    return TRANSCRIBE_OK;
}
}  // namespace

transcribe_status build_vibevoice_weights(ggml_context *           ctx_meta,
                                          const VibeVoiceHParams & hp,
                                          VibeVoiceWeights &       weights) {
    if (ctx_meta == nullptr) return TRANSCRIBE_ERR_INVALID_ARG;

    // VAE encoders + connectors.
    if (auto st = build_vae_encoder(ctx_meta, "acoustic", hp.acoustic, weights.enc_acoustic); st != TRANSCRIBE_OK) return st;
    if (auto st = build_vae_encoder(ctx_meta, "semantic", hp.semantic, weights.enc_semantic); st != TRANSCRIBE_OK) return st;
    if (auto st = build_connector(ctx_meta, "acoustic", hp.acoustic.vae_dim, hp.dec_hidden, weights.conn_acoustic); st != TRANSCRIBE_OK) return st;
    if (auto st = build_connector(ctx_meta, "semantic", hp.semantic.vae_dim, hp.dec_hidden, weights.conn_semantic); st != TRANSCRIBE_OK) return st;

    // Qwen2.5 LM.
    const int64_t hidden = hp.dec_hidden;
    const int64_t vocab  = hp.dec_vocab_size;
    const int64_t q_out  = static_cast<int64_t>(hp.dec_n_heads)    * hp.dec_head_dim;
    const int64_t kv_out = static_cast<int64_t>(hp.dec_n_kv_heads) * hp.dec_head_dim;
    const int64_t ffn    = hp.dec_intermediate;

    GET_LIN(weights.dec_token_embd, "dec.token_embd.weight", hidden, vocab);

    weights.dec_blocks.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        VibeVoiceDecBlock & b = weights.dec_blocks[i];
        GET_F32(b.norm_attn_w, lname("dec.blocks.%d.norm_attn.weight", i), hidden);
        GET_LIN(b.attn_q_w,    lname("dec.blocks.%d.attn.q.weight", i), hidden, q_out);
        GET_F32(b.attn_q_b,    lname("dec.blocks.%d.attn.q.bias", i), q_out);
        GET_LIN(b.attn_k_w,    lname("dec.blocks.%d.attn.k.weight", i), hidden, kv_out);
        GET_F32(b.attn_k_b,    lname("dec.blocks.%d.attn.k.bias", i), kv_out);
        GET_LIN(b.attn_v_w,    lname("dec.blocks.%d.attn.v.weight", i), hidden, kv_out);
        GET_F32(b.attn_v_b,    lname("dec.blocks.%d.attn.v.bias", i), kv_out);
        GET_LIN(b.attn_o_w,    lname("dec.blocks.%d.attn.o.weight", i), q_out, hidden);
        GET_F32(b.norm_ffn_w,  lname("dec.blocks.%d.norm_ffn.weight", i), hidden);
        GET_LIN(b.ffn_gate_w,  lname("dec.blocks.%d.ffn.gate.weight", i), hidden, ffn);
        GET_LIN(b.ffn_up_w,    lname("dec.blocks.%d.ffn.up.weight", i), hidden, ffn);
        GET_LIN(b.ffn_down_w,  lname("dec.blocks.%d.ffn.down.weight", i), ffn, hidden);
    }

    GET_F32(weights.dec_output_norm, "dec.output_norm.weight", hidden);
    GET_LIN(weights.dec_output,      "dec.output.weight", hidden, vocab);
    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_LIN
#undef GET_CONV

}  // namespace transcribe::vibevoice
