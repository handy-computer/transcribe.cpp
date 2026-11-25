// arch/sava/weights.cpp - hparams and weight binding.

#include "ggml.h"
#include "gguf.h"
#include "sava.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::sava {

namespace {

constexpr const char * kTag = "sava";

// Bounds on metadata-derived sizes; the real model is far inside them.
constexpr int32_t kMaxDim = 1 << 16;

#define REQ(expr)                             \
    do {                                      \
        const transcribe_status _st = (expr); \
        if (_st != TRANSCRIBE_OK) {           \
            return _st;                       \
        }                                     \
    } while (0)

transcribe_status read_required_f64(const gguf_context * gguf, const char * key, double & out) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_FLOAT64) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV \"%s\" missing or not float64", kTag, key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_f64(gguf, id);
    return TRANSCRIBE_OK;
}

transcribe_status read_required_bool(const gguf_context * gguf, const char * key, bool & out) {
    if (read_bool_kv(gguf, key, out) != KvResult::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV \"%s\" missing or not a bool", kTag, key);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status read_attn(const gguf_context * gguf, const std::string & prefix, AttnHParams & a) {
    REQ(read_required_u32_kv(gguf, (prefix + ".num_heads").c_str(), kTag, a.n_heads));
    REQ(read_required_u32_kv(gguf, (prefix + ".num_kv_heads").c_str(), kTag, a.n_kv_heads));
    REQ(read_required_u32_kv(gguf, (prefix + ".head_dim").c_str(), kTag, a.head_dim));
    REQ(read_required_u32_kv(gguf, (prefix + ".sliding_window").c_str(), kTag, a.window));
    REQ(read_required_f32_kv(gguf, (prefix + ".rope_theta").c_str(), kTag, a.rope_theta));
    REQ(read_required_bool(gguf, (prefix + ".xsa").c_str(), a.xsa));
    REQ(read_required_f32_kv(gguf, (prefix + ".xsa_alpha").c_str(), kTag, a.xsa_alpha));
    return TRANSCRIBE_OK;
}

bool attn_ok(const AttnHParams & a) {
    return a.n_heads > 0 && a.n_kv_heads > 0 && a.n_heads % a.n_kv_heads == 0 && a.head_dim > 0 &&
           a.head_dim % 2 == 0 && a.head_dim <= kMaxDim && a.n_heads <= kMaxDim && a.window > 0 &&
           a.window <= kMaxDim && std::isfinite(a.xsa_alpha) &&
           // qkv rows in int64: the product of two bounded factors may not fit int32
           (static_cast<int64_t>(a.n_heads) + 2 * static_cast<int64_t>(a.n_kv_heads)) * a.head_dim <= kMaxDim;
}

}  // namespace

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp) {
    REQ(read_required_u32_kv(gguf, "stt.frontend.sample_rate", kTag, hp.sample_rate));
    REQ(read_required_u32_kv(gguf, "stt.frontend.n_fft", kTag, hp.n_fft));
    REQ(read_required_u32_kv(gguf, "stt.frontend.hop_length", kTag, hp.hop));
    REQ(read_required_u32_kv(gguf, "stt.frontend.num_mels", kTag, hp.n_mels));
    REQ(read_required_u32_kv(gguf, "stt.sava.frame_samples", kTag, hp.frame_samples));
    REQ(read_required_u32_kv(gguf, "stt.sava.context_samples", kTag, hp.context_samples));
    REQ(read_required_f32_kv(gguf, "stt.sava.mel_floor", kTag, hp.mel_floor));
    REQ(read_required_f32_kv(gguf, "stt.sava.log_energy_floor", kTag, hp.log_energy_floor));

    REQ(read_required_u32_kv(gguf, "stt.sava.hidden_size", kTag, hp.hidden));
    REQ(read_required_u32_kv(gguf, "stt.sava.ffn_size", kTag, hp.ffn));
    REQ(read_required_f32_kv(gguf, "stt.sava.rms_eps", kTag, hp.rms_eps));
    REQ(read_required_u32_kv(gguf, "stt.sava.mixer_context_frames", kTag, hp.mixer_context_frames));
    std::vector<std::string> layers;
    // At most 64 trunk layers bounds the step graph (see build_graph).
    if (read_string_array_kv(gguf, "stt.sava.layers", layers) != KvResult::Ok || layers.empty() || layers.size() > 64) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV \"stt.sava.layers\" missing or invalid", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    hp.layer_is_ssm.clear();
    for (const std::string & kind : layers) {
        if (kind != "mamba" && kind != "attention") {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: unknown trunk layer kind \"%s\"", kTag, kind.c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
        hp.layer_is_ssm.push_back(kind == "mamba");
    }

    REQ(read_required_u32_kv(gguf, "stt.sava.mamba.expand", kTag, hp.ssm_expand));
    REQ(read_required_u32_kv(gguf, "stt.sava.mamba.head_dim", kTag, hp.ssm_head_dim));
    REQ(read_required_u32_kv(gguf, "stt.sava.mamba.state_size", kTag, hp.ssm_state));
    REQ(read_required_u32_kv(gguf, "stt.sava.mamba.num_groups", kTag, hp.ssm_groups));
    REQ(read_required_u32_kv(gguf, "stt.sava.mamba.conv_kernel", kTag, hp.ssm_conv_kernel));
    REQ(read_attn(gguf, "stt.sava.attention", hp.attn));

    REQ(read_required_u32_kv(gguf, "stt.sava.vad_branch.hidden_size", kTag, hp.vad_hidden));
    REQ(read_required_u32_kv(gguf, "stt.sava.vad_branch.ffn_size", kTag, hp.vad_ffn));
    REQ(read_attn(gguf, "stt.sava.vad_branch.attention", hp.vad_attn));
    REQ(read_required_u32_kv(gguf, "stt.sava.turn_branch.hidden_size", kTag, hp.turn_hidden));
    REQ(read_required_u32_kv(gguf, "stt.sava.turn_branch.ffn_size", kTag, hp.turn_ffn));
    REQ(read_attn(gguf, "stt.sava.turn_branch.attention", hp.turn_attn));

    std::vector<int32_t> horizons;
    if (read_int32_array_kv(gguf, "stt.sava.turn_horizons", horizons) != KvResult::Ok || horizons.empty() ||
        horizons.size() > 64 || horizons[0] != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV \"stt.sava.turn_horizons\" missing or invalid", kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    hp.n_horizons = static_cast<int32_t>(horizons.size());
    REQ(read_required_u32_kv(gguf, "stt.sava.vad_embed_size", kTag, hp.vad_embed));
    REQ(read_required_u32_kv(gguf, "stt.sava.lookahead_frames", kTag, hp.lookahead_frames));
    REQ(read_required_f32_kv(gguf, "stt.sava.no_speech_logit", kTag, hp.no_speech_logit));

    REQ(read_required_f64(gguf, "stt.sava.threshold", hp.threshold));

    auto bad = [](const char * what) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid hparams: %s", kTag, what);
        return TRANSCRIBE_ERR_GGUF;
    };
    if (hp.sample_rate != 16000) {
        return bad("stt.frontend.sample_rate must be 16000");
    }
    if (hp.n_fft <= 0 || hp.n_fft > 4096 || hp.hop <= 0 || hp.hop > hp.n_fft || hp.n_mels <= 0 || hp.n_mels > 1024) {
        return bad("front end sizes");
    }
    if (hp.frame_samples <= 0 || hp.frame_samples > 16000 || hp.frame_samples % hp.hop != 0 ||
        hp.context_samples != hp.n_fft - hp.hop) {
        return bad("frame_samples must be a multiple of hop_length; context_samples = n_fft - hop_length");
    }
    if (!(hp.mel_floor > 0.0f) || !(hp.log_energy_floor > 0.0f) || !(hp.rms_eps > 0.0f)) {
        return bad("floors and rms_eps must be positive");
    }
    if (hp.hidden <= 0 || hp.hidden > kMaxDim || hp.ffn <= 0 || hp.ffn > kMaxDim) {
        return bad("hidden_size / ffn_size");
    }
    if (hp.mixer_context_frames != 2) {
        return bad("mixer_context_frames must be 2");
    }
    if (hp.ssm_expand <= 0 || hp.ssm_expand > 64 || hp.ssm_head_dim <= 0 || hp.ssm_head_dim > kMaxDim ||
        hp.ssm_state <= 0 || hp.ssm_state > kMaxDim || hp.ssm_groups <= 0 || hp.ssm_groups > kMaxDim ||
        hp.ssm_conv_kernel < 2 || hp.ssm_conv_kernel > 64) {
        return bad("mamba sizes");
    }
    // The derived widths in int64, then range-checked, so a malformed GGUF
    // cannot overflow int32.
    const int64_t ssm_inner    = static_cast<int64_t>(hp.ssm_expand) * hp.hidden;
    const int64_t ssm_conv_dim = ssm_inner + 2 * static_cast<int64_t>(hp.ssm_groups) * hp.ssm_state;
    if (ssm_inner > kMaxDim || ssm_conv_dim > 4 * static_cast<int64_t>(kMaxDim)) {
        return bad("mamba inner / conv widths");
    }
    hp.ssm_inner = static_cast<int32_t>(ssm_inner);
    if (hp.ssm_inner % hp.ssm_head_dim != 0) {
        return bad("expand * hidden must divide by mamba head_dim");
    }
    hp.ssm_heads = hp.ssm_inner / hp.ssm_head_dim;
    if (hp.ssm_heads % hp.ssm_groups != 0 || hp.ssm_inner % hp.ssm_groups != 0) {
        return bad("mamba heads must divide by num_groups");
    }
    hp.ssm_conv_dim = static_cast<int32_t>(ssm_conv_dim);
    if (!attn_ok(hp.attn) || !attn_ok(hp.vad_attn) || !attn_ok(hp.turn_attn)) {
        return bad("attention sizes");
    }
    // One RoPE table (rope.freqs) serves every attention layer.
    if (hp.vad_attn.head_dim != hp.attn.head_dim || hp.turn_attn.head_dim != hp.attn.head_dim ||
        hp.vad_attn.rope_theta != hp.attn.rope_theta || hp.turn_attn.rope_theta != hp.attn.rope_theta) {
        return bad("every attention layer must share head_dim and rope_theta");
    }
    if (hp.vad_hidden <= 0 || hp.vad_hidden > kMaxDim || hp.turn_hidden <= 0 || hp.turn_hidden > kMaxDim ||
        hp.vad_ffn <= 0 || hp.vad_ffn > kMaxDim || hp.turn_ffn <= 0 || hp.turn_ffn > kMaxDim || hp.vad_embed <= 0 ||
        hp.vad_embed > kMaxDim) {
        return bad("branch sizes");
    }
    if (hp.lookahead_frames < 0 || hp.lookahead_frames > 64) {
        return bad("lookahead_frames");
    }
    if (!(hp.threshold >= 0.0 && hp.threshold <= 1.0)) {
        return bad("stt.sava.threshold must be 0 to 1");
    }

    hp.hops_per_frame = hp.frame_samples / hp.hop;
    hp.n_bins         = hp.n_fft / 2 + 1;
    hp.mel_width      = hp.hops_per_frame * hp.n_mels;
    hp.row_width      = hp.mel_width + 1;
    return TRANSCRIBE_OK;
}

#define GET_F32(slot, name, ...)                                                                          \
    do {                                                                                                  \
        ggml_tensor * _t =                                                                                \
            transcribe::weights::find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr)                                                                                \
            return TRANSCRIBE_ERR_GGUF;                                                                   \
        (slot) = _t;                                                                                      \
    } while (0)

#define GET_LIN(slot, name, ...)                                                                                 \
    do {                                                                                                         \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                              \
        if (_t == nullptr)                                                                                       \
            return TRANSCRIBE_ERR_GGUF;                                                                          \
        (slot) = _t;                                                                                             \
    } while (0)

namespace {

transcribe_status bind_attn(ggml_context *      ctx_meta,
                            const std::string & p,
                            int64_t             d,
                            const AttnHParams & a,
                            AttnWeights &       w) {
    const int64_t hd = a.head_dim;
    GET_F32(w.norm, (p + ".norm_attn.weight").c_str(), d);
    GET_LIN(w.qkv, (p + ".attn.qkv.weight").c_str(), d, (a.n_heads + 2 * a.n_kv_heads) * hd);
    GET_F32(w.q_norm, (p + ".attn.q_norm.weight").c_str(), hd);
    GET_F32(w.k_norm, (p + ".attn.k_norm.weight").c_str(), hd);
    GET_LIN(w.out, (p + ".attn.out.weight").c_str(), a.n_heads * hd, d);
    return TRANSCRIBE_OK;
}

transcribe_status bind_mlp(ggml_context * ctx_meta, const std::string & p, int64_t d, int64_t ffn, MlpWeights & w) {
    GET_F32(w.norm, (p + ".norm_mlp.weight").c_str(), d);
    GET_LIN(w.gate, (p + ".mlp.gate.weight").c_str(), d, ffn);
    GET_LIN(w.up, (p + ".mlp.up.weight").c_str(), d, ffn);
    GET_LIN(w.down, (p + ".mlp.down.weight").c_str(), ffn, d);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status bind_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w) {
    const int64_t d = hp.hidden;
    GET_LIN(w.mixer_proj, "mixer.proj.weight", static_cast<int64_t>(hp.mixer_context_frames) * hp.row_width, d);
    GET_F32(w.mixer_norm, "mixer.norm_out.weight", d);

    w.trunk.assign(hp.layer_is_ssm.size(), Block{});
    for (size_t i = 0; i < hp.layer_is_ssm.size(); ++i) {
        Block &           b = w.trunk[i];
        const std::string p = "trunk." + std::to_string(i);
        b.is_ssm            = hp.layer_is_ssm[i];
        if (b.is_ssm) {
            SsmWeights & s = b.ssm;
            GET_F32(s.norm, (p + ".norm_mix.weight").c_str(), d);
            GET_LIN(s.in, (p + ".ssm.in.weight").c_str(), d, hp.ssm_inner + hp.ssm_conv_dim + hp.ssm_heads);
            GET_F32(s.conv_w, (p + ".ssm.conv1d.weight").c_str(), hp.ssm_conv_kernel, hp.ssm_conv_dim);
            GET_F32(s.conv_b, (p + ".ssm.conv1d.bias").c_str(), hp.ssm_conv_dim);
            GET_F32(s.dt_b, (p + ".ssm.dt.bias").c_str(), hp.ssm_heads);
            GET_F32(s.a, (p + ".ssm.a").c_str(), 1, hp.ssm_heads);
            GET_F32(s.d, (p + ".ssm.d").c_str(), hp.ssm_heads);
            GET_F32(s.norm_gated, (p + ".ssm.norm_gated.weight").c_str(), hp.ssm_inner);
            GET_LIN(s.out, (p + ".ssm.out.weight").c_str(), hp.ssm_inner, d);
        } else {
            REQ(bind_attn(ctx_meta, p, d, hp.attn, b.attn));
        }
        REQ(bind_mlp(ctx_meta, p, d, hp.ffn, b.mlp));
    }
    GET_F32(w.trunk_norm, "trunk.norm_out.weight", d);

    GET_F32(w.vad_norm_in, "head.vad.norm_in.weight", d);
    GET_LIN(w.vad_down, "head.vad.down.weight", d, hp.vad_hidden);
    REQ(bind_attn(ctx_meta, "head.vad.blk", hp.vad_hidden, hp.vad_attn, w.vad_attn));
    REQ(bind_mlp(ctx_meta, "head.vad.blk", hp.vad_hidden, hp.vad_ffn, w.vad_mlp));
    GET_F32(w.vad_norm_out, "head.vad.norm_out.weight", hp.vad_hidden);
    GET_LIN(w.vad_ro_w, "head.vad.readout.weight", hp.vad_hidden, 1);
    GET_F32(w.vad_ro_b, "head.vad.readout.bias", 1);
    GET_LIN(w.vad_la_w, "head.vad.readout_la.weight", hp.vad_hidden, 1);
    GET_F32(w.vad_la_b, "head.vad.readout_la.bias", 1);

    GET_F32(w.turn_norm_in, "head.turn.norm_in.weight", d);
    GET_F32(w.turn_emb_w, "head.turn.speech_embed.weight", hp.vad_embed);
    GET_F32(w.turn_emb_b, "head.turn.speech_embed.bias", hp.vad_embed);
    GET_LIN(w.turn_down, "head.turn.down.weight", d + hp.vad_embed, hp.turn_hidden);
    REQ(bind_attn(ctx_meta, "head.turn.blk", hp.turn_hidden, hp.turn_attn, w.turn_attn));
    REQ(bind_mlp(ctx_meta, "head.turn.blk", hp.turn_hidden, hp.turn_ffn, w.turn_mlp));
    GET_F32(w.turn_norm_out, "head.turn.norm_out.weight", hp.turn_hidden);
    GET_LIN(w.turn_ro_w, "head.turn.readout.weight", hp.turn_hidden, hp.n_horizons);
    GET_F32(w.turn_ro_b, "head.turn.readout.bias", hp.n_horizons);
    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_LIN
#undef REQ

}  // namespace transcribe::sava
