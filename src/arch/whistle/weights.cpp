// arch/whistle/weights.cpp - read_whistle_hparams / build_whistle_weights.
//
// Pattern follows cohere/weights.cpp: every hparam is read explicitly from
// stt.whistle.* / stt.frontend.*, and every tensor is fetched with its
// expected ggml shape. Missing tensor or shape mismatch -> TRANSCRIBE_ERR_GGUF.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cstdio>
#include <initializer_list>

namespace transcribe::whistle {

namespace {

constexpr const char * kFamilyTag = "whistle";

transcribe_status read_required_i32_array(const gguf_context * gguf, const char * key, std::vector<int32_t> & out) {
    const KvResult r = read_int32_array_kv(gguf, key, out);
    if (r != KvResult::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: missing or mistyped int32 array KV \"%s\"", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status read_whistle_hparams(const gguf_context * gguf, WhistleHParams & hp) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    struct U32Kv {
        const char * key;
        int32_t *    out;
    };

    const U32Kv u32s[] = {
        { "stt.whistle.d_model",               &hp.d_model              },
        { "stt.whistle.n_heads",               &hp.n_heads              },
        { "stt.whistle.n_kv_heads",            &hp.n_kv_heads           },
        { "stt.whistle.qk_head_dim",           &hp.qk_head_dim          },
        { "stt.whistle.v_head_dim",            &hp.v_head_dim           },
        { "stt.whistle.mhc_lanes",             &hp.mhc_lanes            },
        { "stt.whistle.hada_n",                &hp.hada_n               },
        { "stt.whistle.vocab_size",            &hp.vocab_size           },
        { "stt.whistle.encoder.n_layers",      &hp.enc_n_layers         },
        { "stt.whistle.encoder.conv_kernel",   &hp.enc_conv_kernel      },
        { "stt.whistle.stem.channels",         &hp.stem_channels        },
        { "stt.whistle.stem.stages",           &hp.stem_stages          },
        { "stt.whistle.decoder.n_layers",      &hp.dec_n_layers         },
        { "stt.whistle.decoder.max_seq_len",   &hp.dec_max_seq          },
        { "stt.whistle.decoder.qkv_conv_taps", &hp.dec_taps             },
        { "stt.whistle.engram.slots",          &hp.engram_slots         },
        { "stt.whistle.engram.sub_dim",        &hp.engram_sub_dim       },
        { "stt.whistle.engram.n_tables",       &hp.engram_n_tables      },
        { "stt.whistle.engram.conv_taps",      &hp.engram_conv_taps     },
        { "stt.whistle.engram.conv_dilation",  &hp.engram_conv_dilation },
        { "stt.whistle.engram.seed_heads",     &hp.engram_seed_heads    },
        { "stt.whistle.lang_token_base",       &hp.lang_token_base      },
        { "stt.whistle.max_audio_samples",     &hp.max_audio_samples    },
        { "stt.frontend.sample_rate",          &hp.fe_sample_rate       },
        { "stt.frontend.num_mels",             &hp.fe_num_mels          },
        { "stt.frontend.n_fft",                &hp.fe_n_fft             },
        { "stt.frontend.win_length",           &hp.fe_win_length        },
        { "stt.frontend.hop_length",           &hp.fe_hop_length        },
    };
    for (const U32Kv & kv : u32s) {
        if (auto st = read_required_u32_kv(gguf, kv.key, kFamilyTag, *kv.out); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (auto st = read_required_f32_kv(gguf, "stt.whistle.rope_theta", kFamilyTag, hp.rope_theta);
        st != TRANSCRIBE_OK) {
        return st;
    }
    {
        uint32_t       g = 0;
        const KvResult r = read_uint32_kv(gguf, "stt.whistle.hadamard_group", g);
        if (r == KvResult::BadType || (r == KvResult::Ok && g != 128)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: stt.whistle.hadamard_group must be a u32 equal to 128");
            return TRANSCRIBE_ERR_GGUF;
        }
        hp.hadamard_group = static_cast<int32_t>(g);
    }
    if (auto st = read_required_i32_array(gguf, "stt.whistle.engram.orders", hp.engram_orders); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_i32_array(gguf, "stt.whistle.engram.sites", hp.engram_sites); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_i32_array(gguf, "stt.whistle.lang_token_ids", hp.lang_token_ids); st != TRANSCRIBE_OK) {
        return st;
    }
    if (read_string_array_kv(gguf, "stt.whistle.languages", hp.languages) != KvResult::Ok) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: missing stt.whistle.languages");
        return TRANSCRIBE_ERR_GGUF;
    }
    std::string fe_type;
    if (auto st = read_required_string_kv(gguf, "stt.frontend.type", kFamilyTag, fe_type); st != TRANSCRIBE_OK) {
        return st;
    }

    // Cross-field invariants. The graph code hard-wires the shapes it was
    // validated on; reject anything else rather than run a guessed graph.
    if (fe_type != "mel") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: unsupported frontend type \"%s\"", fe_type.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.d_model <= 0 || hp.n_heads <= 0 || hp.n_kv_heads <= 0 || hp.n_heads % hp.n_kv_heads != 0 ||
        hp.qk_head_dim <= 0 || hp.v_head_dim <= 0 || hp.n_heads * hp.v_head_dim != hp.d_model) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: inconsistent attention geometry");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.hada_n != hp.d_model || hp.d_model != 512) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: HadamardMLP is implemented for d_model == hada_n == 512 only");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.mhc_lanes <= 0 || hp.enc_n_layers <= 0 || hp.dec_n_layers <= 0 || hp.dec_max_seq <= 2 || hp.dec_taps <= 0 ||
        hp.enc_conv_kernel <= 0 || hp.enc_conv_kernel % 2 == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: invalid layer / lane / tap counts");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.stem_stages != 3 || hp.stem_channels <= 0 || (hp.fe_num_mels % (1 << hp.stem_stages)) != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: stem implemented for 3 stride-2 stages over num_mels %% 8 == 0");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_n_fft != 512 || hp.fe_win_length != 400 || hp.fe_hop_length != 160 || hp.fe_sample_rate != 16000) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: frontend implemented for 16 kHz, n_fft 512, win 400, hop 160");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.engram_orders.empty() || hp.engram_n_tables % static_cast<int32_t>(hp.engram_orders.size()) != 0 ||
        hp.engram_n_tables * hp.engram_sub_dim != hp.d_model || hp.engram_slots <= 0 || hp.engram_conv_taps <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: inconsistent Engram geometry");
        return TRANSCRIBE_ERR_GGUF;
    }
    for (int32_t s : hp.engram_sites) {
        if (s < 0 || s >= hp.dec_n_layers) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: Engram site %d out of range", s);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (hp.languages.empty() || hp.languages.size() != hp.lang_token_ids.size()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: languages / lang_token_ids mismatch");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

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

// GET_LIN also accepts Q2_K, outside the shared allowlist: the Hadamard-domain
// Q2_K_HR preset (docs/tools/quantization.md) is Whistle-only.
#define GET_LIN(slot, name, ...)                                                                         \
    do {                                                                                                 \
        ggml_tensor * _t = transcribe::weights::find_tensor(                                             \
            ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES, GGML_TYPE_Q2_K }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr)                                                                               \
            return TRANSCRIBE_ERR_GGUF;                                                                  \
        (slot) = _t;                                                                                     \
    } while (0)

transcribe_status get_hmlp(ggml_context * ctx_meta, const char * prefix, int n, WhistleHmlp & h) {
    char name[160];
    auto nm = [&](const char * leaf) {
        std::snprintf(name, sizeof(name), "%s.%s", prefix, leaf);
        return name;
    };
    const int64_t ba = 16;
    const int64_t bb = n / ba;
    GET_F32(h.d1, nm("d1"), n);
    GET_F32(h.d2, nm("d2"), n);
    GET_F32(h.b2, nm("b2"), n);
    GET_F32(h.d3, nm("d3"), n);
    GET_F32(h.d4, nm("d4"), n);
    GET_F32(h.wa[0], nm("w1a"), ba, ba);
    GET_F32(h.wb[0], nm("w1b"), bb, bb);
    GET_F32(h.wa[1], nm("w2a"), ba, ba);
    GET_F32(h.wb[1], nm("w2b"), bb, bb);
    GET_F32(h.wa[2], nm("w3a"), ba, ba);
    GET_F32(h.wb[2], nm("w3b"), bb, bb);
    GET_F32(h.cond_v, nm("cond_v"), 8, n);
    GET_F32(h.cond_u, nm("cond_u"), n, 8);
    return TRANSCRIBE_OK;
}

transcribe_status get_mhc(ggml_context * ctx_meta, const char * prefix, int L, int lanes, int d, WhistleMhc & m) {
    char name[96];
    auto nm = [&](const char * leaf) {
        std::snprintf(name, sizeof(name), "%s.%s", prefix, leaf);
        return name;
    };
    GET_F32(m.a_pre, nm("a_pre"), L);
    GET_F32(m.a_post, nm("a_post"), L);
    GET_F32(m.a_res, nm("a_res"), L);
    GET_F32(m.b_pre, nm("b_pre"), lanes, L);
    GET_F32(m.b_post, nm("b_post"), lanes, L);
    GET_F32(m.b_res, nm("b_res"), lanes, lanes, L);
    GET_LIN(m.phi_pre, nm("phi_pre.weight"), (int64_t) lanes * d, (int64_t) L * lanes);
    GET_LIN(m.phi_post, nm("phi_post.weight"), (int64_t) lanes * d, (int64_t) L * lanes);
    GET_LIN(m.phi_res, nm("phi_res.weight"), (int64_t) lanes * d, (int64_t) L * lanes * lanes);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status build_whistle_weights(ggml_context * ctx_meta, const WhistleHParams & hp, WhistleWeights & w) {
    if (ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t d     = hp.d_model;
    const int64_t C     = hp.stem_channels;
    const int64_t qk    = hp.qk_head_dim;
    const int64_t nh    = hp.n_heads;
    const int64_t nkv   = hp.n_kv_heads;
    const int64_t vd    = hp.v_head_dim;
    const int     lanes = hp.mhc_lanes;
    char          pfx[96];

    // ----- stem -----
    GET_CONV(w.stem.conv_w, "enc.stem.conv.w.weight", C, 9);
    GET_CONV(w.stem.dw_1, "enc.stem.conv.dw_1.weight", C, 9);
    GET_LIN(w.stem.pw_1, "enc.stem.pw_1.weight", C, C);
    GET_CONV(w.stem.dw_2, "enc.stem.conv.dw_2.weight", C, 9);
    GET_LIN(w.stem.pw_2, "enc.stem.pw_2.weight", C, C);
    GET_LIN(w.stem.out, "enc.stem.out.weight", C * hp.stem_freq_out(), d);

    // ----- encoder blocks -----
    w.enc_blocks.assign(hp.enc_n_layers, WhistleEncBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        WhistleEncBlock & b = w.enc_blocks[i];
        GET_F32(b.norm_hmlp_0, lname("enc.blocks.%d.norm_hmlp_0.weight", i), d);
        GET_F32(b.norm_in, lname("enc.blocks.%d.norm_in.weight", i), d);
        GET_F32(b.norm_post_attn, lname("enc.blocks.%d.norm_post_attn.weight", i), d);
        GET_F32(b.norm_conv, lname("enc.blocks.%d.norm_conv.weight", i), d);
        GET_F32(b.norm_conv_out, lname("enc.blocks.%d.norm_conv_out.weight", i), d);
        GET_F32(b.norm_hmlp, lname("enc.blocks.%d.norm_hmlp.weight", i), d);
        GET_LIN(b.attn.q, lname("enc.blocks.%d.attn.q.weight", i), d, nh * qk);
        GET_LIN(b.attn.k, lname("enc.blocks.%d.attn.k.weight", i), d, nkv * qk);
        GET_LIN(b.attn.v, lname("enc.blocks.%d.attn.v.weight", i), d, nkv * vd);
        GET_LIN(b.attn.gate, lname("enc.blocks.%d.attn.gate.weight", i), d, nh * vd);
        GET_LIN(b.attn.out, lname("enc.blocks.%d.attn.out.weight", i), nh * vd, d);
        GET_F32(b.attn.q_norm, lname("enc.blocks.%d.attn.q_norm.weight", i), qk);
        GET_F32(b.attn.k_norm, lname("enc.blocks.%d.attn.k_norm.weight", i), qk);
        GET_F32(b.attn_gate, lname("enc.blocks.%d.attn_gate", i), 1);
        GET_LIN(b.conv_pw1, lname("enc.blocks.%d.conv_pw1.weight", i), d, 2 * d);
        GET_LIN(b.conv_pw2, lname("enc.blocks.%d.conv_pw2.weight", i), d, d);
        GET_CONV(b.conv_dw, lname("enc.blocks.%d.conv.dw.weight", i), d, hp.enc_conv_kernel);
        std::snprintf(pfx, sizeof(pfx), "enc.blocks.%d.hmlp_0", i);
        if (auto st = get_hmlp(ctx_meta, pfx, hp.hada_n, b.hmlp_0); st != TRANSCRIBE_OK) {
            return st;
        }
        std::snprintf(pfx, sizeof(pfx), "enc.blocks.%d.hmlp", i);
        if (auto st = get_hmlp(ctx_meta, pfx, hp.hada_n, b.hmlp); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (auto st = get_mhc(ctx_meta, "enc.mhc", hp.enc_n_layers, lanes, hp.d_model, w.enc_mhc); st != TRANSCRIBE_OK) {
        return st;
    }
    GET_F32(w.enc_final_norm, "enc.final_norm.weight", d);

    // ----- decoder -----
    GET_LIN(w.token_embd, "dec.token_embd.weight", d, hp.vocab_size);
    w.dec_blocks.assign(hp.dec_n_layers, WhistleDecBlock{});
    for (int i = 0; i < hp.dec_n_layers; ++i) {
        WhistleDecBlock & b = w.dec_blocks[i];
        GET_F32(b.norm_in, lname("dec.blocks.%d.norm_in.weight", i), d);
        GET_LIN(b.attn.q, lname("dec.blocks.%d.attn.q.weight", i), d, nh * qk);
        GET_LIN(b.attn.k, lname("dec.blocks.%d.attn.k.weight", i), d, nkv * qk);
        GET_LIN(b.attn.v, lname("dec.blocks.%d.attn.v.weight", i), d, nkv * vd);
        GET_F32(b.attn.q_taps, lname("dec.blocks.%d.attn.q_taps", i), nh * qk, hp.dec_taps);
        GET_F32(b.attn.k_taps, lname("dec.blocks.%d.attn.k_taps", i), nkv * qk, hp.dec_taps);
        GET_F32(b.attn.v_taps, lname("dec.blocks.%d.attn.v_taps", i), nkv * vd, hp.dec_taps);
        GET_F32(b.attn.q_norm, lname("dec.blocks.%d.attn.q_norm.weight", i), qk);
        GET_F32(b.attn.k_norm, lname("dec.blocks.%d.attn.k_norm.weight", i), qk);
        GET_LIN(b.attn.gate, lname("dec.blocks.%d.attn.gate.weight", i), d, nh * vd);
        GET_LIN(b.attn.out, lname("dec.blocks.%d.attn.out.weight", i), nh * vd, d);
        GET_F32(b.norm_post_attn, lname("dec.blocks.%d.norm_post_attn.weight", i), d);
        GET_F32(b.attn_gate, lname("dec.blocks.%d.attn_gate", i), 1);
        GET_F32(b.norm_hmlp, lname("dec.blocks.%d.norm_hmlp.weight", i), d);
        std::snprintf(pfx, sizeof(pfx), "dec.blocks.%d.hmlp", i);
        if (auto st = get_hmlp(ctx_meta, pfx, hp.hada_n, b.hmlp); st != TRANSCRIBE_OK) {
            return st;
        }
        // Cross-attention is full multi-head (n_heads K/V heads).
        GET_F32(b.norm_cross, lname("dec.blocks.%d.norm_cross.weight", i), d);
        GET_LIN(b.cross.q, lname("dec.blocks.%d.cross.q.weight", i), d, nh * qk);
        GET_LIN(b.cross.k, lname("dec.blocks.%d.cross.k.weight", i), d, nh * qk);
        GET_LIN(b.cross.v, lname("dec.blocks.%d.cross.v.weight", i), d, nh * vd);
        GET_LIN(b.cross.gate, lname("dec.blocks.%d.cross.gate.weight", i), d, nh * vd);
        GET_LIN(b.cross.out, lname("dec.blocks.%d.cross.out.weight", i), nh * vd, d);
        GET_F32(b.cross.q_norm, lname("dec.blocks.%d.cross.q_norm.weight", i), qk);
        GET_F32(b.cross.k_norm, lname("dec.blocks.%d.cross.k_norm.weight", i), qk);
        GET_F32(b.norm_post_cross, lname("dec.blocks.%d.norm_post_cross.weight", i), d);
        GET_F32(b.cross_gate, lname("dec.blocks.%d.cross_gate", i), 1);
    }
    if (auto st = get_mhc(ctx_meta, "dec.mhc", hp.dec_n_layers, lanes, hp.d_model, w.dec_mhc); st != TRANSCRIBE_OK) {
        return st;
    }
    w.engrams.assign(hp.engram_sites.size(), WhistleEngram{});
    for (size_t s = 0; s < hp.engram_sites.size(); ++s) {
        WhistleEngram &     e  = w.engrams[s];
        const int           si = static_cast<int>(s);
        const char *        tn = lname("dec.engram.%d.tables.weight", si);
        const ggml_tensor * tm = ggml_get_tensor(ctx_meta, tn);
        e.paired               = tm != nullptr && tm->ne[0] == 2 * (int64_t) hp.engram_sub_dim;
        if (s > 0 && e.paired != w.engrams[0].paired) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "whistle: Engram sites mix paired and plain table layouts");
            return TRANSCRIBE_ERR_GGUF;
        }
        const int64_t pr = e.paired ? 2 : 1;
        GET_LIN(e.tables, tn, pr * hp.engram_sub_dim, (int64_t) hp.engram_n_tables * hp.engram_slots / pr);
        GET_LIN(e.key_proj, lname("dec.engram.%d.key_proj.weight", si),
                (int64_t) hp.engram_n_tables * hp.engram_sub_dim, d);
        GET_LIN(e.value_proj, lname("dec.engram.%d.value_proj.weight", si),
                (int64_t) hp.engram_n_tables * hp.engram_sub_dim, d);
        GET_F32(e.conv_taps, lname("dec.engram.%d.conv_taps", si), d, hp.engram_conv_taps);
    }
    GET_F32(w.dec_final_norm, "dec.final_norm.weight", d);
    GET_F32(w.pe_gate, "dec.pe_gate", 1);
    GET_F32(w.hada_perm1, "hada.perm1", hp.hada_n);
    GET_F32(w.hada_perm2, "hada.perm2", hp.hada_n);
    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_CONV
#undef GET_LIN

}  // namespace transcribe::whistle
