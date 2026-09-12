// arch/granite5_ctc/weights.cpp - read_granite5_ctc_hparams +
// build_granite5_ctc_weights.
//
// Pattern mirrors arch/granite/weights.cpp. Every required KV is read
// explicitly; BadType is always fatal. The tensor catalog is a sequence
// of find_tensor() calls with expected shapes; a missing tensor or a
// shape mismatch returns TRANSCRIBE_ERR_GGUF.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <algorithm>
#include <cstdio>

namespace transcribe::granite5_ctc {

namespace {
constexpr const char * kFamilyTag = "granite5_ctc";
}

transcribe_status read_granite5_ctc_hparams(const gguf_context * gguf, Granite5CtcHParams & hp) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (auto st = read_optional_string_kv(gguf, "stt.variant", kFamilyTag, "", hp.variant); st != TRANSCRIBE_OK) {
        return st;
    }

    // Encoder.
    struct U32Field {
        const char * key;
        int32_t *    slot;
    };
    const U32Field enc_fields[] = {
        { "stt.granite5_ctc.encoder.n_layers",         &hp.enc_n_layers         },
        { "stt.granite5_ctc.encoder.hidden",           &hp.enc_hidden           },
        { "stt.granite5_ctc.encoder.n_heads",          &hp.enc_n_heads          },
        { "stt.granite5_ctc.encoder.head_dim",         &hp.enc_head_dim         },
        { "stt.granite5_ctc.encoder.intermediate",     &hp.enc_intermediate     },
        { "stt.granite5_ctc.encoder.input_dim",        &hp.enc_input_dim        },
        { "stt.granite5_ctc.encoder.output_dim",       &hp.enc_output_dim       },
        { "stt.granite5_ctc.encoder.conv_kernel_size", &hp.enc_conv_kernel_size },
        { "stt.granite5_ctc.encoder.conv_expansion",   &hp.enc_conv_expansion   },
        { "stt.granite5_ctc.encoder.context_size",     &hp.enc_context_size     },
        { "stt.granite5_ctc.encoder.max_pos_emb",      &hp.enc_max_pos_emb      },
        { "stt.granite5_ctc.encoder.self_cond_layer",  &hp.enc_self_cond_layer  },
    };
    for (const auto & f : enc_fields) {
        if (auto st = read_required_u32_kv(gguf, f.key, kFamilyTag, *f.slot); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    // subsample_layers is a (possibly empty) uint32 array. Absent means
    // "no in-block subsampling", which is a legal architecture even
    // though no published variant uses it.
    {
        std::vector<int32_t> tmp;
        switch (read_int32_array_kv(gguf, "stt.granite5_ctc.encoder.subsample_layers", tmp)) {
            case KvResult::Ok:
                hp.enc_subsample_layers = std::move(tmp);
                break;
            case KvResult::Absent:
                hp.enc_subsample_layers.clear();
                break;
            case KvResult::BadType:
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                        "granite5_ctc: stt.granite5_ctc.encoder.subsample_layers has wrong type");
                return TRANSCRIBE_ERR_GGUF;
        }
    }
    std::sort(hp.enc_subsample_layers.begin(), hp.enc_subsample_layers.end());
    for (int32_t idx : hp.enc_subsample_layers) {
        if (idx < 0 || idx >= hp.enc_n_layers) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: subsample_layers entry %d out of range [0, %d)", idx,
                    hp.enc_n_layers);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    if (auto st = read_required_u32_kv(gguf, "stt.granite5_ctc.blank_id", kFamilyTag, hp.blank_id);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Frontend.
    if (auto st = read_required_string_kv(gguf, "stt.frontend.type", kFamilyTag, hp.fe_type); st != TRANSCRIBE_OK) {
        return st;
    }
    const U32Field fe_fields[] = {
        { "stt.frontend.sample_rate",       &hp.fe_sample_rate      },
        { "stt.frontend.num_mels",          &hp.fe_num_mels         },
        { "stt.frontend.n_fft",             &hp.fe_n_fft            },
        { "stt.frontend.win_length",        &hp.fe_win_length       },
        { "stt.frontend.hop_length",        &hp.fe_hop_length       },
        { "stt.frontend.delta_win_length",  &hp.fe_delta_win_length },
        { "stt.frontend.stack_factor",      &hp.fe_stack_factor     },
    };
    for (const auto & f : fe_fields) {
        if (auto st = read_required_u32_kv(gguf, f.key, kFamilyTag, *f.slot); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (auto st = read_required_string_kv(gguf, "stt.frontend.window", kFamilyTag, hp.fe_window);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_string_kv(gguf, "stt.frontend.normalize", kFamilyTag, hp.fe_normalize);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.pad_mode", kFamilyTag, "reflect", hp.fe_pad_mode);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_string_kv(gguf, "stt.frontend.mel_norm", kFamilyTag, "htk", hp.fe_mel_norm);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.dither", kFamilyTag, hp.fe_dither); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.logmel_floor_db", kFamilyTag, hp.fe_logmel_floor_db);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_bool_kv(gguf, "stt.frontend.deltas", kFamilyTag, true, hp.fe_deltas);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Invariants. Each of these is silently load-bearing: getting one
    // wrong produces a plausible-looking but incorrect forward rather
    // than a crash, so fail loudly at load instead.
    if (hp.fe_normalize != "per_utterance") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "granite5_ctc: unsupported frontend normalize \"%s\" (only \"per_utterance\")",
                hp.fe_normalize.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_window != "hann_periodic") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: unsupported frontend window \"%s\" (only \"hann_periodic\")",
                hp.fe_window.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_mel_norm != "htk") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: unsupported frontend mel_norm \"%s\" (only \"htk\")",
                hp.fe_mel_norm.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_stack_factor != 2) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: unsupported frontend stack_factor %d (only 2)",
                hp.fe_stack_factor);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_delta_win_length != 3) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "granite5_ctc: unsupported frontend delta_win_length %d (only 3; the "
                "host delta kernel is the win_length=3 special case)",
                hp.fe_delta_win_length);
        return TRANSCRIBE_ERR_GGUF;
    }
    {
        const int32_t mel_ch   = hp.fe_num_mels * (hp.fe_deltas ? 2 : 1);
        const int32_t expected = mel_ch * hp.fe_stack_factor;
        if (expected != hp.enc_input_dim) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "granite5_ctc: frontend produces %d features "
                    "(%d mel x %d delta x %d stack) but encoder.input_dim is %d",
                    expected, hp.fe_num_mels, hp.fe_deltas ? 2 : 1, hp.fe_stack_factor, hp.enc_input_dim);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if ((hp.enc_conv_kernel_size % 2) == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "granite5_ctc: conv_kernel_size %d is even; SAME padding requires an odd kernel",
                hp.enc_conv_kernel_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_context_size <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: context_size must be positive, got %d",
                hp.enc_context_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_self_cond_layer < 0 || hp.enc_self_cond_layer > hp.enc_n_layers) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "granite5_ctc: self_cond_layer %d out of range [0, %d]",
                hp.enc_self_cond_layer, hp.enc_n_layers);
        return TRANSCRIBE_ERR_GGUF;
    }

    return TRANSCRIBE_OK;
}

// Weights.

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
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_CONV_TYPES },  \
                                                            { __VA_ARGS__ }, kTag);                            \
        if (_t == nullptr)                                                                                      \
            return TRANSCRIBE_ERR_GGUF;                                                                         \
        (slot) = _t;                                                                                            \
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

transcribe_status build_granite5_ctc_weights(ggml_context *             ctx_meta,
                                             const Granite5CtcHParams & hp,
                                             Granite5CtcWeights &       weights) {
    if (ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t enc_h       = hp.enc_hidden;
    const int64_t enc_in      = hp.enc_input_dim;
    const int64_t enc_out     = hp.enc_output_dim;
    const int64_t enc_inner   = static_cast<int64_t>(hp.enc_n_heads) * hp.enc_head_dim;
    const int64_t enc_ffn     = hp.enc_intermediate;
    const int64_t conv_inner  = enc_h * hp.enc_conv_expansion;
    const int64_t conv_up_out = conv_inner * 2;  // pre-GLU
    const int64_t conv_k      = hp.enc_conv_kernel_size;
    const int64_t head_dim    = hp.enc_head_dim;
    const int64_t rel_pos_len = 2LL * hp.enc_max_pos_emb + 1;

    // Encoder: top-level. ctc_proj is the tied head — the same tensor
    // feeds the mid-layer self-conditioning logits and the final CTC
    // logits, which is why there is no separate `ctc_head.*` slot.
    GET_LIN(weights.enc_top.input_linear_w, "enc.input_linear.weight", enc_in, enc_h);
    GET_F32(weights.enc_top.input_linear_b, "enc.input_linear.bias", enc_h);
    GET_LIN(weights.enc_top.ctc_proj_w, "enc.ctc_proj.weight", enc_h, enc_out);
    GET_F32(weights.enc_top.ctc_proj_b, "enc.ctc_proj.bias", enc_out);
    GET_LIN(weights.enc_top.ctc_bypass_w, "enc.ctc_bypass.weight", enc_out, enc_h);
    GET_F32(weights.enc_top.ctc_bypass_b, "enc.ctc_bypass.bias", enc_h);

    // Encoder: per-block.
    weights.enc_blocks.assign(hp.enc_n_layers, Granite5CtcEncBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = weights.enc_blocks[i];

        // FF1 (first macaron half). Both linears carry a bias
        // (config.attention_bias=true reaches the FFN, not q/k/v).
        GET_F32(b.norm_ff1_w, lname("enc.blocks.%d.norm_ff1.weight", i), enc_h);
        GET_F32(b.norm_ff1_b, lname("enc.blocks.%d.norm_ff1.bias", i), enc_h);
        GET_LIN(b.ff1_lin1_w, lname("enc.blocks.%d.ff1.linear1.weight", i), enc_h, enc_ffn);
        GET_F32(b.ff1_lin1_b, lname("enc.blocks.%d.ff1.linear1.bias", i), enc_ffn);
        GET_LIN(b.ff1_lin2_w, lname("enc.blocks.%d.ff1.linear2.weight", i), enc_ffn, enc_h);
        GET_F32(b.ff1_lin2_b, lname("enc.blocks.%d.ff1.linear2.bias", i), enc_h);

        // Block-local Shaw self-attention. q has no bias; kv is the
        // converter-fused K|V (no bias); out has a bias.
        GET_F32(b.norm_attn_w, lname("enc.blocks.%d.norm_attn.weight", i), enc_h);
        GET_F32(b.norm_attn_b, lname("enc.blocks.%d.norm_attn.bias", i), enc_h);
        GET_LIN(b.attn_q_w, lname("enc.blocks.%d.attn.q.weight", i), enc_h, enc_inner);
        GET_LIN(b.attn_kv_w, lname("enc.blocks.%d.attn.kv.weight", i), enc_h, 2 * enc_inner);
        GET_LIN(b.attn_out_w, lname("enc.blocks.%d.attn.out.weight", i), enc_inner, enc_h);
        GET_F32(b.attn_out_b, lname("enc.blocks.%d.attn.out.bias", i), enc_h);
        GET_LIN(b.attn_rel_pos_emb, lname("enc.blocks.%d.attn.rel_pos_emb.weight", i), head_dim, rel_pos_len);

        // Conv module. pointwise1/2 are nn.Linear upstream, so they are
        // 2-D here — granite 4.x's 3-D [1, in, out] layout does NOT apply.
        // They are plain mul_mat operands (encoder.cpp), hence GET_LIN:
        // the quantizer classifies a 2-D pointwise as Linear, so these
        // arrive block-quantized in the derived presets. The reference
        // GGUF still stores them F16, which the linear allowlist covers.
        GET_F32(b.norm_conv_w, lname("enc.blocks.%d.norm_conv.weight", i), enc_h);
        GET_F32(b.norm_conv_b, lname("enc.blocks.%d.norm_conv.bias", i), enc_h);
        GET_LIN(b.conv_pointwise1_w, lname("enc.blocks.%d.conv.pointwise1.weight", i), enc_h, conv_up_out);
        GET_F32(b.conv_pointwise1_b, lname("enc.blocks.%d.conv.pointwise1.bias", i), conv_up_out);
        GET_CONV(b.conv_depthwise_w, lname("enc.blocks.%d.conv.depthwise.weight", i), conv_k, 1, conv_inner);
        GET_F32(b.conv_bn_w, lname("enc.blocks.%d.conv.bn.weight", i), conv_inner);
        GET_F32(b.conv_bn_b, lname("enc.blocks.%d.conv.bn.bias", i), conv_inner);
        GET_F32(b.conv_bn_mean, lname("enc.blocks.%d.conv.bn.running_mean", i), conv_inner);
        GET_F32(b.conv_bn_var, lname("enc.blocks.%d.conv.bn.running_var", i), conv_inner);
        GET_LIN(b.conv_pointwise2_w, lname("enc.blocks.%d.conv.pointwise2.weight", i), conv_inner, enc_h);
        GET_F32(b.conv_pointwise2_b, lname("enc.blocks.%d.conv.pointwise2.bias", i), enc_h);

        // FF2 (second macaron half).
        GET_F32(b.norm_ff2_w, lname("enc.blocks.%d.norm_ff2.weight", i), enc_h);
        GET_F32(b.norm_ff2_b, lname("enc.blocks.%d.norm_ff2.bias", i), enc_h);
        GET_LIN(b.ff2_lin1_w, lname("enc.blocks.%d.ff2.linear1.weight", i), enc_h, enc_ffn);
        GET_F32(b.ff2_lin1_b, lname("enc.blocks.%d.ff2.linear1.bias", i), enc_ffn);
        GET_LIN(b.ff2_lin2_w, lname("enc.blocks.%d.ff2.linear2.weight", i), enc_ffn, enc_h);
        GET_F32(b.ff2_lin2_b, lname("enc.blocks.%d.ff2.linear2.bias", i), enc_h);

        // Per-block final LayerNorm.
        GET_F32(b.norm_out_w, lname("enc.blocks.%d.norm_out.weight", i), enc_h);
        GET_F32(b.norm_out_b, lname("enc.blocks.%d.norm_out.bias", i), enc_h);
    }

    // Frontend buffers. The converter bakes both straight out of
    // torchaudio (melscale_fbanks htk/norm=None and hann_window(periodic)),
    // so they are bit-exact against the reference rather than recomputed
    // here. Required: silently recomputing them would trade a verified
    // buffer for an approximation.
    const int64_t n_freq = static_cast<int64_t>(hp.fe_n_fft) / 2 + 1;
    GET_F32(weights.frontend_mel_filterbank, "frontend.mel_filterbank", n_freq, hp.fe_num_mels);
    GET_F32(weights.frontend_window, "frontend.window", hp.fe_win_length);

    return TRANSCRIBE_OK;
}

}  // namespace transcribe::granite5_ctc
