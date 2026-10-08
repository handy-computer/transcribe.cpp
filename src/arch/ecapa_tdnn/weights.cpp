// arch/ecapa_tdnn/weights.cpp - read_hparams + build_weights.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cstdio>
#include <string>

namespace transcribe::ecapa_tdnn {

namespace {

constexpr const char * kTag = "ecapa_tdnn";

// Required int32 array of exactly `n` entries.
transcribe_status read_required_i32_array_kv(const gguf_context *   gguf,
                                             const char *           key,
                                             size_t                 n,
                                             std::vector<int32_t> & out) {
    if (read_int32_array_kv(gguf, key, out) != KvResult::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: required KV \"%s\" missing or not an int32 array", kTag, key);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (out.size() != n) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV \"%s\" has %zu entries, expected %zu", kTag, key, out.size(), n);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

#define REQ(expr)                             \
    do {                                      \
        const transcribe_status _st = (expr); \
        if (_st != TRANSCRIBE_OK) {           \
            return _st;                       \
        }                                     \
    } while (0)

    REQ(read_required_u32_kv(gguf, "stt.frontend.sample_rate", kTag, hp.sample_rate));
    REQ(read_required_u32_kv(gguf, "stt.frontend.n_fft", kTag, hp.mel_n_fft));
    REQ(read_required_u32_kv(gguf, "stt.frontend.hop_length", kTag, hp.mel_hop));
    REQ(read_required_u32_kv(gguf, "stt.frontend.win_length", kTag, hp.mel_win));
    REQ(read_required_u32_kv(gguf, "stt.frontend.num_mels", kTag, hp.mel_n_mels));
    REQ(read_required_string_kv(gguf, "stt.frontend.window", kTag, hp.mel_window));
    REQ(read_required_string_kv(gguf, "stt.frontend.pad_mode", kTag, hp.mel_pad_mode));
    REQ(read_required_f32_kv(gguf, "stt.frontend.log_clamp_min", kTag, hp.mel_log_floor));
    REQ(read_required_f32_kv(gguf, "stt.frontend.top_db", kTag, hp.mel_top_db));
    REQ(read_required_string_kv(gguf, "stt.frontend.normalize", kTag, hp.mel_normalize));

    REQ(read_required_i32_array_kv(gguf, "stt.ecapa_tdnn.channels", kNumStages, hp.channels));
    REQ(read_required_i32_array_kv(gguf, "stt.ecapa_tdnn.kernel_sizes", kNumStages, hp.kernel_sizes));
    REQ(read_required_i32_array_kv(gguf, "stt.ecapa_tdnn.dilations", kNumStages, hp.dilations));
    REQ(read_required_u32_kv(gguf, "stt.ecapa_tdnn.res2net_scale", kTag, hp.res2net_scale));
    REQ(read_required_u32_kv(gguf, "stt.ecapa_tdnn.se_channels", kTag, hp.se_channels));
    REQ(read_required_u32_kv(gguf, "stt.ecapa_tdnn.attention_channels", kTag, hp.attention_channels));
    REQ(read_required_f32_kv(gguf, "stt.ecapa_tdnn.asp_eps", kTag, hp.asp_eps));
    REQ(read_required_u32_kv(gguf, "stt.ecapa_tdnn.embedding_dim", kTag, hp.embedding_dim));
    REQ(read_required_u32_kv(gguf, "stt.ecapa_tdnn.classifier_hidden", kTag, hp.classifier_hidden));
    REQ(read_required_f32_kv(gguf, "stt.ecapa_tdnn.classifier_leaky_slope", kTag, hp.leaky_slope));

#undef REQ

    if (hp.sample_rate != 16000) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: stt.frontend.sample_rate is %d; only 16000 is supported", kTag,
                hp.sample_rate);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.mel_n_fft <= 0 || hp.mel_hop <= 0 || hp.mel_win <= 0 || hp.mel_n_mels <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "%s: frontend dimensions must be positive (n_fft=%d hop_length=%d win_length=%d num_mels=%d)", kTag,
                hp.mel_n_fft, hp.mel_hop, hp.mel_win, hp.mel_n_mels);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.mel_win > hp.mel_n_fft) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: frontend win_length (%d) > n_fft (%d)", kTag, hp.mel_win,
                hp.mel_n_fft);
        return TRANSCRIBE_ERR_GGUF;
    }
    // MelFrontend falls back silently on unknown values.
    if (hp.mel_window != "hamming_periodic" || hp.mel_pad_mode != "constant" || hp.mel_normalize != "sentence_mean") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: unsupported front end (window=%s pad_mode=%s normalize=%s)", kTag,
                hp.mel_window.c_str(), hp.mel_pad_mode.c_str(), hp.mel_normalize.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    for (int i = 0; i < kNumStages; ++i) {
        const int32_t k = hp.kernel_sizes[static_cast<size_t>(i)];
        const int32_t d = hp.dilations[static_cast<size_t>(i)];
        if (k <= 0 || k % 2 == 0 || d <= 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: stage %d needs an odd kernel and positive dilation (k=%d d=%d)",
                    kTag, i, k, d);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (hp.kernel_sizes[kNumStages - 1] != 1 || hp.dilations[kNumStages - 1] != 1) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: MFA stage must be k=1 d=1, got k=%d d=%d", kTag,
                hp.kernel_sizes[kNumStages - 1], hp.dilations[kNumStages - 1]);
        return TRANSCRIBE_ERR_GGUF;
    }
    // SERes2Net blocks have no shortcut projection: every block width matches.
    for (int i = 1; i < kNumStages - 1; ++i) {
        if (hp.channels[static_cast<size_t>(i)] != hp.channels[0]) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: channels[%d] = %d differs from channels[0] = %d", kTag, i,
                    hp.channels[static_cast<size_t>(i)], hp.channels[0]);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (hp.channels[kNumStages - 1] != (kNumSeBlocks * hp.channels[0])) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: MFA channels %d must equal %d x block channels %d", kTag,
                hp.channels[kNumStages - 1], kNumSeBlocks, hp.channels[0]);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.res2net_scale != kRes2NetScale || hp.channels[0] <= 0 || hp.channels[0] % hp.res2net_scale != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: res2net_scale must be %d and divide channels[0] (got %d, %d)", kTag,
                kRes2NetScale, hp.res2net_scale, hp.channels[0]);
        return TRANSCRIBE_ERR_GGUF;
    }

    return TRANSCRIBE_OK;
}

namespace {

using transcribe::weights::find_tensor;

#define GET_F32(slot, name, ...)                                                                    \
    do {                                                                                            \
        ggml_tensor * _t = find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr) {                                                                        \
            return TRANSCRIBE_ERR_GGUF;                                                             \
        }                                                                                           \
        (slot) = _t;                                                                                \
    } while (0)

#define GET_CONV(slot, name, ...)                                                                                 \
    do {                                                                                                          \
        ggml_tensor * _t = find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_CONV_TYPES }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr) {                                                                                      \
            return TRANSCRIBE_ERR_GGUF;                                                                           \
        }                                                                                                         \
        (slot) = _t;                                                                                              \
    } while (0)

#define GET_LIN(slot, name, ...)                                                                                    \
    do {                                                                                                            \
        ggml_tensor * _t = find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr) {                                                                                        \
            return TRANSCRIBE_ERR_GGUF;                                                                             \
        }                                                                                                           \
        (slot) = _t;                                                                                                \
    } while (0)

std::string cat(const std::string & prefix, const char * suffix) {
    return prefix + suffix;
}

// A k=1 TDNNBlock: [in, out] weight, [out] bias, [out] BN scale/shift.
transcribe_status load_tdnn(ggml_context *      ctx_meta,
                            const std::string & prefix,
                            int64_t             in,
                            int64_t             out,
                            TdnnLayer &         t) {
    const std::string n_w     = cat(prefix, ".weight");
    const std::string n_b     = cat(prefix, ".bias");
    const std::string n_scale = cat(prefix, ".bn.scale");
    const std::string n_shift = cat(prefix, ".bn.shift");

    GET_LIN(t.w, n_w.c_str(), in, out);
    GET_F32(t.b, n_b.c_str(), out);
    GET_F32(t.bn.scale, n_scale.c_str(), out);
    GET_F32(t.bn.shift, n_shift.c_str(), out);
    return TRANSCRIBE_OK;
}

// One dilated k-tap Res2Net sub-convolution, tap-major ne=[c, c, k].
transcribe_status load_res2_sub(ggml_context *      ctx_meta,
                                const std::string & prefix,
                                int64_t             c,
                                int64_t             k,
                                Res2Sub &           s) {
    const std::string n_w     = cat(prefix, ".conv.weight");
    const std::string n_b     = cat(prefix, ".conv.bias");
    const std::string n_scale = cat(prefix, ".bn.scale");
    const std::string n_shift = cat(prefix, ".bn.shift");

    GET_CONV(s.w, n_w.c_str(), c, c, k);
    GET_F32(s.b, n_b.c_str(), c);
    GET_F32(s.bn.scale, n_scale.c_str(), c);
    GET_F32(s.bn.shift, n_shift.c_str(), c);
    return TRANSCRIBE_OK;
}

transcribe_status load_se(ggml_context * ctx_meta, const std::string & prefix, int64_t c, int64_t se, SeBlock & s) {
    const std::string n_c1_w = cat(prefix, ".c1.weight");
    const std::string n_c1_b = cat(prefix, ".c1.bias");
    const std::string n_c2_w = cat(prefix, ".c2.weight");
    const std::string n_c2_b = cat(prefix, ".c2.bias");

    GET_LIN(s.c1_w, n_c1_w.c_str(), c, se);
    GET_F32(s.c1_b, n_c1_b.c_str(), se);
    GET_LIN(s.c2_w, n_c2_w.c_str(), se, c);
    GET_F32(s.c2_b, n_c2_b.c_str(), c);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status build_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w) {
    if (ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t n_mels = hp.mel_n_mels;
    const int64_t n_freq = hp.n_freq();
    const int64_t C      = hp.c_block();
    const int64_t Cm     = hp.c_mfa();
    const int64_t chunk  = hp.c_chunk();
    const int64_t se     = hp.se_channels;
    const int64_t att    = hp.attention_channels;
    const int64_t emb    = hp.embedding_dim;
    const int64_t hid    = hp.classifier_hidden;
    const int64_t n_lab  = hp.n_labels;
    const int64_t k_blk0 = hp.kernel_sizes[0];

    // ---- front end ------------------------------------------------------
    // Mel-major: each filter's n_freq weights contiguous, so ne[0] = n_freq.
    GET_F32(w.mel_filters, "frontend.mel_filterbank", n_freq, n_mels);

    // ---- stage 0 --------------------------------------------------------
    GET_CONV(w.blk0_w, "blk.0.conv.weight", n_mels, C, k_blk0);
    GET_F32(w.blk0_b, "blk.0.conv.bias", C);
    GET_F32(w.blk0_bn.scale, "blk.0.bn.scale", C);
    GET_F32(w.blk0_bn.shift, "blk.0.bn.shift", C);

    // ---- stages 1..3 (SERes2Net) ----------------------------------------
    for (int i = 0; i < kNumSeBlocks; ++i) {
        SeRes2NetBlock &  blk    = w.blocks[i];
        const std::string base   = "blk." + std::to_string(i + 1);
        const int64_t     k_res2 = hp.kernel_sizes[static_cast<size_t>(i + 1)];

        if (auto st = load_tdnn(ctx_meta, base + ".tdnn1", C, C, blk.tdnn1); st != TRANSCRIBE_OK) {
            return st;
        }
        for (int j = 0; j < kRes2NetSubs; ++j) {
            const std::string p = base + ".res2." + std::to_string(j);
            if (auto st = load_res2_sub(ctx_meta, p, chunk, k_res2, blk.res2[j]); st != TRANSCRIBE_OK) {
                return st;
            }
        }
        if (auto st = load_tdnn(ctx_meta, base + ".tdnn2", C, C, blk.tdnn2); st != TRANSCRIBE_OK) {
            return st;
        }
        if (auto st = load_se(ctx_meta, base + ".se", C, se, blk.se); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    // ---- MFA -------------------------------------------------------------
    GET_LIN(w.mfa_w[0], "mfa.w1.weight", C, Cm);
    GET_LIN(w.mfa_w[1], "mfa.w2.weight", C, Cm);
    GET_LIN(w.mfa_w[2], "mfa.w3.weight", C, Cm);
    GET_F32(w.mfa_b, "mfa.bias", Cm);
    GET_F32(w.mfa_bn.scale, "mfa.bn.scale", Cm);
    GET_F32(w.mfa_bn.shift, "mfa.bn.shift", Cm);

    // ---- attentive statistics pooling -------------------------------------
    GET_LIN(w.asp_wx, "asp.tdnn.x.weight", Cm, att);
    GET_LIN(w.asp_wm, "asp.tdnn.mean.weight", Cm, att);
    GET_LIN(w.asp_ws, "asp.tdnn.std.weight", Cm, att);
    GET_F32(w.asp_b, "asp.tdnn.bias", att);
    GET_F32(w.asp_bn.scale, "asp.tdnn.bn.scale", att);
    GET_F32(w.asp_bn.shift, "asp.tdnn.bn.shift", att);
    GET_LIN(w.asp_attn_w, "asp.attn.weight", att, Cm);
    GET_F32(w.asp_attn_b, "asp.attn.bias", Cm);

    // ---- embedding head ---------------------------------------------------
    GET_LIN(w.fc_w, "fc.weight", 2 * Cm, emb);
    GET_F32(w.fc_b, "fc.bias", emb);

    // ---- classifier -------------------------------------------------------
    GET_LIN(w.cls_l1_w, "cls.l1.weight", emb, hid);
    GET_F32(w.cls_l1_b, "cls.l1.bias", hid);
    GET_LIN(w.cls_out_w, "cls.out.weight", hid, n_lab);
    GET_F32(w.cls_out_b, "cls.out.bias", n_lab);

    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_CONV
#undef GET_LIN

}  // namespace transcribe::ecapa_tdnn
