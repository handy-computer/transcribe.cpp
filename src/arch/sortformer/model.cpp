// arch/sortformer/model.cpp - Sortformer (encoder-diarizer) load / run /
// Arch instance. The 17-layer NEST FastConformer encoder is the same NeMo
// ConformerEncoder Parakeet ports, so it reuses parakeet::build_encoder_graph
// verbatim (constructed from a ParakeetWeights populated with only the
// pre_encode + block slots). This file adds the Sortformer-specific graph:
// encoder_proj (512->192) -> 18x post-LN Transformer -> diar sigmoid head,
// producing a T x 4 speaker-activity matrix.

#include "../parakeet/encoder.h"
#include "../parakeet/weights.h"
#include "conformer/conformer.h"
#include "ggml.h"
#include "gguf.h"
#include "sortformer.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace transcribe::sortformer {

extern const Arch arch;

namespace pk   = transcribe::parakeet;
namespace conf = transcribe::conformer;

static constexpr float kBnEps = 1e-5f;
static constexpr char  k_default_variant[] = "diar_streaming_sortformer_4spk-v2.1";

SortformerModel::~SortformerModel() {
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
    }
    if (bn_fused_buffer != nullptr) {
        transcribe::safe_buffer_free(bn_fused_buffer);
    }
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    if (backend_buffer != nullptr) {
        transcribe::safe_buffer_free(backend_buffer);
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        transcribe::safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

SortformerSession::~SortformerSession() {
    if (sched != nullptr) {
        transcribe::safe_sched_free(sched);
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
    }
}

namespace {

// Map the Sortformer hparams onto the parakeet encoder hparams that
// build_encoder_graph reads. The FastConformer is NeMo's ConformerEncoder
// with xscaling=True, rel_pos self-attention, batch_norm conv, use_bias,
// full (offline) attention.
void fill_conformer_hp(const SortformerHParams & s, pk::ParakeetHParams & p) {
    p.enc_n_layers             = s.enc_n_layers;
    p.enc_d_model              = s.enc_d_model;
    p.enc_n_heads              = s.enc_n_heads;
    p.enc_d_ff                 = s.enc_d_ff;
    p.enc_conv_kernel          = s.enc_conv_kernel;
    p.enc_subsampling_factor   = s.enc_subsampling_factor;
    p.enc_subsampling_channels = s.enc_subsampling_channels;
    p.enc_pos_emb_max_len      = s.enc_pos_emb_max_len;
    p.enc_use_bias             = true;
    p.enc_xscaling             = true;  // NEST FastConformer uses xscaling
    p.enc_att_context_left     = -1;
    p.enc_att_context_right    = -1;
    p.enc_att_context_style    = pk::ParakeetHParams::AttContextStyle::Regular;
    p.enc_conv_context_left    = -1;
    p.enc_conv_context_right   = -1;
    p.enc_conv_norm_type       = pk::ParakeetHParams::ConvNormType::BatchNorm;

    // Frontend fields build_encoder_graph's pre_encode geometry reads.
    p.fe_num_mels    = s.fe_num_mels;
    p.fe_sample_rate = s.fe_sample_rate;
    p.fe_n_fft       = s.fe_n_fft;
    p.fe_win_length  = s.fe_win_length;
    p.fe_hop_length  = s.fe_hop_length;
    p.fe_normalize   = s.fe_normalize;
    p.fe_dither      = s.fe_dither;
    p.fe_pre_emphasis = s.fe_pre_emphasis;
}

// Populate ONLY the pre_encode + block slots of a ParakeetWeights from the
// GGUF (the conformer). Predictor / joint / head stay empty. Mirrors the
// encoder portion of build_parakeet_weights (identical tensor names).
transcribe_status load_conformer_weights(ggml_context * ctx, const pk::ParakeetHParams & hp, pk::ParakeetWeights & w) {
    char name[128];
    auto get = [&](const char * nm) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx, nm);
        if (t == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing conformer tensor %s", nm);
        }
        return t;
    };
#define G(dst, nm)                              \
    do {                                        \
        (dst) = get(nm);                        \
        if ((dst) == nullptr) return TRANSCRIBE_ERR_GGUF; \
    } while (0)
#define GL(dst, fmt, i)                         \
    do {                                        \
        std::snprintf(name, sizeof(name), fmt, i); \
        G(dst, name);                           \
    } while (0)

    auto & pe = w.pre_encode;
    G(pe.conv0_w, "enc.pre_encode.conv.0.weight");
    G(pe.conv0_b, "enc.pre_encode.conv.0.bias");
    G(pe.conv2_w, "enc.pre_encode.conv.2.weight");
    G(pe.conv2_b, "enc.pre_encode.conv.2.bias");
    G(pe.conv3_w, "enc.pre_encode.conv.3.weight");
    G(pe.conv3_b, "enc.pre_encode.conv.3.bias");
    G(pe.conv5_w, "enc.pre_encode.conv.5.weight");
    G(pe.conv5_b, "enc.pre_encode.conv.5.bias");
    G(pe.conv6_w, "enc.pre_encode.conv.6.weight");
    G(pe.conv6_b, "enc.pre_encode.conv.6.bias");
    G(pe.out_w, "enc.pre_encode.out.weight");
    G(pe.out_b, "enc.pre_encode.out.bias");

    w.blocks.assign(static_cast<size_t>(hp.enc_n_layers), pk::ParakeetBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = w.blocks[static_cast<size_t>(i)];
        GL(b.norm_ff1_w, "enc.blocks.%d.norm_ff1.weight", i);
        GL(b.norm_ff1_b, "enc.blocks.%d.norm_ff1.bias", i);
        GL(b.ff1_lin1_w, "enc.blocks.%d.ff1.linear1.weight", i);
        GL(b.ff1_lin2_w, "enc.blocks.%d.ff1.linear2.weight", i);
        GL(b.norm_attn_w, "enc.blocks.%d.norm_attn.weight", i);
        GL(b.norm_attn_b, "enc.blocks.%d.norm_attn.bias", i);
        GL(b.attn_q_w, "enc.blocks.%d.attn.linear_q.weight", i);
        GL(b.attn_k_w, "enc.blocks.%d.attn.linear_k.weight", i);
        GL(b.attn_v_w, "enc.blocks.%d.attn.linear_v.weight", i);
        GL(b.attn_out_w, "enc.blocks.%d.attn.linear_out.weight", i);
        GL(b.attn_pos_w, "enc.blocks.%d.attn.linear_pos.weight", i);
        GL(b.attn_pos_u, "enc.blocks.%d.attn.pos_bias_u", i);
        GL(b.attn_pos_v, "enc.blocks.%d.attn.pos_bias_v", i);
        GL(b.norm_conv_w, "enc.blocks.%d.norm_conv.weight", i);
        GL(b.norm_conv_b, "enc.blocks.%d.norm_conv.bias", i);
        GL(b.conv_pw1_w, "enc.blocks.%d.conv.pointwise1.weight", i);
        GL(b.conv_dw_w, "enc.blocks.%d.conv.depthwise.weight", i);
        GL(b.conv_pw2_w, "enc.blocks.%d.conv.pointwise2.weight", i);
        GL(b.conv_bn_w, "enc.blocks.%d.conv.bn.weight", i);
        GL(b.conv_bn_b, "enc.blocks.%d.conv.bn.bias", i);
        GL(b.conv_bn_rm, "enc.blocks.%d.conv.bn.running_mean", i);
        GL(b.conv_bn_rv, "enc.blocks.%d.conv.bn.running_var", i);
        GL(b.norm_ff2_w, "enc.blocks.%d.norm_ff2.weight", i);
        GL(b.norm_ff2_b, "enc.blocks.%d.norm_ff2.bias", i);
        GL(b.ff2_lin1_w, "enc.blocks.%d.ff2.linear1.weight", i);
        GL(b.ff2_lin2_w, "enc.blocks.%d.ff2.linear2.weight", i);
        GL(b.norm_out_w, "enc.blocks.%d.norm_out.weight", i);
        GL(b.norm_out_b, "enc.blocks.%d.norm_out.bias", i);
        // use_bias=true: linear + conv biases (linear_pos stays bias-free).
        GL(b.ff1_lin1_b, "enc.blocks.%d.ff1.linear1.bias", i);
        GL(b.ff1_lin2_b, "enc.blocks.%d.ff1.linear2.bias", i);
        GL(b.attn_q_b, "enc.blocks.%d.attn.linear_q.bias", i);
        GL(b.attn_k_b, "enc.blocks.%d.attn.linear_k.bias", i);
        GL(b.attn_v_b, "enc.blocks.%d.attn.linear_v.bias", i);
        GL(b.attn_out_b, "enc.blocks.%d.attn.linear_out.bias", i);
        GL(b.conv_pw1_b, "enc.blocks.%d.conv.pointwise1.bias", i);
        GL(b.conv_dw_b, "enc.blocks.%d.conv.depthwise.bias", i);
        GL(b.conv_pw2_b, "enc.blocks.%d.conv.pointwise2.bias", i);
        GL(b.ff2_lin1_b, "enc.blocks.%d.ff2.linear1.bias", i);
        GL(b.ff2_lin2_b, "enc.blocks.%d.ff2.linear2.bias", i);
    }
#undef G
#undef GL
    return TRANSCRIBE_OK;
}

// Fuse the conformer BatchNorm into scale + bias tensors (parakeet's
// build_encoder_graph consumes conv_bn_fused_scale/bias, not the raw BN).
transcribe_status fuse_conformer_bn(SortformerModel & m) {
    const size_t n_blocks = m.conformer.blocks.size();
    if (n_blocks == 0) {
        return TRANSCRIBE_OK;
    }
    const int64_t d            = m.conformer_hp.enc_d_model;
    const size_t  tensor_bytes = static_cast<size_t>(d) * sizeof(float);

    const size_t     ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params   = { ctx_size, nullptr, /*no_alloc=*/true };
    m.bn_fused_ctx            = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b              = m.conformer.blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
    }
    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    std::vector<float> bn_w(d), bn_b(d), rm(d), rv(d), fused_s(d), fused_b(d);
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.conformer.blocks[i];
        ggml_backend_tensor_get(b.conv_bn_w, bn_w.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_b, bn_b.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rm, rm.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rv, rv.data(), 0, tensor_bytes);
        for (int64_t c = 0; c < d; ++c) {
            const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
            fused_s[c]    = s;
            fused_b[c]    = bn_b[c] - rm[c] * s;
        }
        ggml_backend_tensor_set(b.conv_bn_fused_scale, fused_s.data(), 0, tensor_bytes);
        ggml_backend_tensor_set(b.conv_bn_fused_bias, fused_b.data(), 0, tensor_bytes);
    }
    return TRANSCRIBE_OK;
}

// ---- diar-head graph helpers ----

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * W, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    if (b != nullptr) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// One post-LN Transformer encoder block (no positional encoding, full
// attention). x ne = [d, T].
ggml_tensor * tf_block(ggml_context * ctx, const SortformerTfBlock & b, ggml_tensor * x, int d, int n_head, int T) {
    const int   head_dim  = d / n_head;
    const float attn_scale = std::sqrt(std::sqrt(static_cast<float>(head_dim)));
    const float inv_scale  = 1.0f / attn_scale;

    ggml_tensor * residual = x;

    ggml_tensor * q = linear(ctx, b.attn_q_w, x, b.attn_q_b);
    ggml_tensor * k = linear(ctx, b.attn_k_w, x, b.attn_k_b);
    ggml_tensor * v = linear(ctx, b.attn_v_w, x, b.attn_v_b);

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, n_head, T), 0, 2, 1, 3));  // [hd,T,H]
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, n_head, T), 0, 2, 1, 3));  // [hd,T,H]
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, n_head, T), 0, 2, 1, 3));  // [hd,T,H]

    // NeMo pre-divides q and k by sqrt(sqrt(head_dim)); scores = q.k^T / sqrt(head_dim).
    q = ggml_scale(ctx, q, inv_scale);
    k = ggml_scale(ctx, k, inv_scale);

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);   // [T_k, T_q, H]
    kq               = ggml_soft_max(ctx, kq);    // softmax over keys (ne0)

    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [T, hd, H]
    ggml_tensor * ctx_out = ggml_mul_mat(ctx, v_t, kq);                    // [hd, T_q, H]
    ctx_out = ggml_cont(ctx, ggml_permute(ctx, ctx_out, 0, 2, 1, 3));      // [hd, H, T]
    ctx_out = ggml_reshape_2d(ctx, ctx_out, d, T);                         // [d, T]

    ggml_tensor * attn_out = linear(ctx, b.attn_o_w, ctx_out, b.attn_o_b);

    // residual + LN1 (post-LN)
    x = ggml_add(ctx, attn_out, residual);
    x = layer_norm(ctx, x, b.norm1_w, b.norm1_b);

    // FFN: dense_in -> relu -> dense_out, residual + LN2
    ggml_tensor * res2 = x;
    ggml_tensor * f    = linear(ctx, b.ff_in_w, x, b.ff_in_b);
    f                  = ggml_relu(ctx, f);
    f                  = linear(ctx, b.ff_out_w, f, b.ff_out_b);
    x                  = ggml_add(ctx, f, res2);
    x                  = layer_norm(ctx, x, b.norm2_w, b.norm2_b);
    return x;
}

// ---- Streaming two-phase graph helpers ----
//
// The FastConformer is reused via the family-agnostic conf:: primitives so
// the streaming path can split pre_encode (Graph A) from the block stack
// (Graph B) without touching parakeet. sf_*_view mirror the static
// to_view() helpers in parakeet/encoder.cpp; sf_conv_policy mirrors the
// policy build in build_encoder_graph (offline subsample, no causal).

conf::ConvPolicy sf_conv_policy(const char * backend) {
    conf::ConvPolicy policy{};
    policy.direct_pw = conf::detect_direct_pw(backend);
    // Parakeet block/pre_encode depthwise defaults (encoder.cpp:50-66).
    policy.direct_dw_in_block = conf::resolve_conv_direct("TRANSCRIBE_CONV_DIRECT_DW", "TRANSCRIBE_CONV_NO_DIRECT_DW",
                                                          /*backend_default=*/true);
    const bool is_metal = backend != nullptr &&
                          (std::strstr(backend, "Metal") != nullptr || std::strstr(backend, "metal") != nullptr);
    policy.direct_dw_in_pre_encode = conf::resolve_conv_direct(
        "TRANSCRIBE_CONV_DIRECT_DW", "TRANSCRIBE_CONV_NO_DIRECT_DW", /*backend_default=*/!is_metal);
    policy.causal_pre_encode = false;  // NEST offline subsample (Regular att)
    return policy;
}

conf::PreEncodeView sf_pre_view(const pk::ParakeetPreEncode & pe) {
    conf::PreEncodeView v;
    v.conv0_w = pe.conv0_w;  v.conv0_b = pe.conv0_b;
    v.conv2_w = pe.conv2_w;  v.conv2_b = pe.conv2_b;
    v.conv3_w = pe.conv3_w;  v.conv3_b = pe.conv3_b;
    v.conv5_w = pe.conv5_w;  v.conv5_b = pe.conv5_b;
    v.conv6_w = pe.conv6_w;  v.conv6_b = pe.conv6_b;
    v.out_w   = pe.out_w;    v.out_b   = pe.out_b;
    return v;
}

conf::BlockView sf_block_view(const pk::ParakeetBlock & b) {
    conf::BlockView v;
    v.norm_ff1_w = b.norm_ff1_w;  v.norm_ff1_b = b.norm_ff1_b;
    v.ff1_lin1_w = b.ff1_lin1_w;  v.ff1_lin1_b = b.ff1_lin1_b;
    v.ff1_lin2_w = b.ff1_lin2_w;  v.ff1_lin2_b = b.ff1_lin2_b;
    v.norm_attn_w = b.norm_attn_w;  v.norm_attn_b = b.norm_attn_b;
    v.attn_q_w = b.attn_q_w;  v.attn_q_b = b.attn_q_b;
    v.attn_k_w = b.attn_k_w;  v.attn_k_b = b.attn_k_b;
    v.attn_v_w = b.attn_v_w;  v.attn_v_b = b.attn_v_b;
    v.attn_out_w = b.attn_out_w;  v.attn_out_b = b.attn_out_b;
    v.attn_pos_w = b.attn_pos_w;  v.attn_pos_u = b.attn_pos_u;  v.attn_pos_v = b.attn_pos_v;
    v.norm_conv_w = b.norm_conv_w;  v.norm_conv_b = b.norm_conv_b;
    v.conv_pw1_w = b.conv_pw1_w;  v.conv_pw1_b = b.conv_pw1_b;
    v.conv_dw_w = b.conv_dw_w;  v.conv_dw_b = b.conv_dw_b;
    v.conv_pw2_w = b.conv_pw2_w;  v.conv_pw2_b = b.conv_pw2_b;
    v.conv_bn_fused_scale = b.conv_bn_fused_scale;  v.conv_bn_fused_bias = b.conv_bn_fused_bias;
    v.conv_ln_w = b.conv_bn_w;  v.conv_ln_b = b.conv_bn_b;
    v.norm_ff2_w = b.norm_ff2_w;  v.norm_ff2_b = b.norm_ff2_b;
    v.ff2_lin1_w = b.ff2_lin1_w;  v.ff2_lin1_b = b.ff2_lin1_b;
    v.ff2_lin2_w = b.ff2_lin2_w;  v.ff2_lin2_b = b.ff2_lin2_b;
    v.norm_out_w = b.norm_out_w;  v.norm_out_b = b.norm_out_b;
    return v;
}

// Host sinusoidal rel-pos table [pos_len, d_model] (same as the offline
// builder in run(); factored for per-chunk reuse in the streaming path).
void fill_rel_pos_emb(std::vector<float> & buf, std::vector<float> & div_term, int pos_len, int d_model) {
    const int   zero_index = (pos_len - 1) / 2;
    const float ln_10000   = std::log(10000.0f);
    buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);
    div_term.resize(static_cast<size_t>(d_model / 2));
    for (int kk = 0; kk < d_model / 2; ++kk) {
        div_term[static_cast<size_t>(kk)] =
            std::exp(static_cast<float>(2 * kk) * (-ln_10000 / static_cast<float>(d_model)));
    }
    for (int i = 0; i < pos_len; ++i) {
        const float pos = static_cast<float>(zero_index - i);
        float *     row = buf.data() + static_cast<size_t>(i) * d_model;
        for (int kk = 0; kk < d_model / 2; ++kk) {
            const float div = div_term[static_cast<size_t>(kk)];
            row[2 * kk]     = std::sin(pos * div);
            row[2 * kk + 1] = std::cos(pos * div);
        }
    }
}

// Graph A: pre_encode over one mel window [n_mels, M] -> [enc_d_model, T_diar].
struct PreEncodeBuild {
    ggml_cgraph * graph  = nullptr;
    ggml_tensor * mel_in = nullptr;
    ggml_tensor * out    = nullptr;
};

PreEncodeBuild build_pre_encode_graph(ggml_context * ctx, SortformerModel & m, int M) {
    PreEncodeBuild b{};
    const int     n_mels = m.conformer_hp.fe_num_mels;
    ggml_tensor * mel_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, M, n_mels, 1, 1);
    ggml_set_name(mel_in, "chunk.mel.in");
    ggml_set_input(mel_in);
    const conf::ConvPolicy policy = sf_conv_policy(m.backend.c_str());
    ggml_tensor * out = conf::build_pre_encode(ctx, sf_pre_view(m.conformer.pre_encode), mel_in, policy,
                                               /*name_prefix=*/"chunk.pre_encode", /*error_tag=*/"sortformer", nullptr);
    if (out == nullptr) {
        return b;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, out);
    b.graph  = graph;
    b.mel_in = mel_in;
    b.out    = out;
    return b;
}

// Graph B: [enc_d_model, T_concat] concat -> xscale -> 17 conformer blocks ->
// encoder_proj -> 18 post-LN transformer -> diar head -> preds [n_spk, T_concat].
struct StreamInferBuild {
    ggml_cgraph * graph      = nullptr;
    ggml_tensor * concat_in  = nullptr;
    ggml_tensor * pos_emb_in = nullptr;
    ggml_tensor * preds      = nullptr;
};

StreamInferBuild build_stream_infer_graph(ggml_context * ctx, SortformerModel & m, int T_concat) {
    StreamInferBuild b{};
    const int     ed        = m.conformer_hp.enc_d_model;
    ggml_tensor * concat_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ed, T_concat);
    ggml_set_name(concat_in, "stream.concat.in");
    ggml_set_input(concat_in);

    // xscaling (NEST FastConformer, applied on the concat, not stored in cache).
    ggml_tensor * x = ggml_scale(ctx, concat_in, std::sqrt(static_cast<float>(ed)));

    const int     pos_len    = 2 * T_concat - 1;
    ggml_tensor * pos_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ed, pos_len);
    ggml_set_name(pos_emb_in, "stream.pos_emb.in");
    ggml_set_input(pos_emb_in);

    conf::BlockParams bp{};
    bp.d_model     = ed;
    bp.n_head      = m.conformer_hp.enc_n_heads;
    bp.conv_kernel = m.conformer_hp.enc_conv_kernel;
    bp.kv_type     = GGML_TYPE_F32;
    bool enc_use_flash = true, dec_use_flash = false;
    transcribe::flash::apply_env_overrides(enc_use_flash, dec_use_flash);
    bp.use_flash          = enc_use_flash;
    bp.policy             = sf_conv_policy(m.backend.c_str());
    bp.att_context_left   = -1;  // full (offline) attention over the concat
    bp.att_context_right  = -1;
    bp.att_context_style  = conf::BlockParams::AttContextStyle::Regular;
    bp.conv_context_left  = -1;
    bp.conv_context_right = -1;
    bp.conv_norm_type     = conf::BlockParams::ConvNormType::BatchNorm;

    for (int i = 0; i < m.conformer_hp.enc_n_layers; ++i) {
        x = conf::build_conformer_block(ctx, x, pos_emb_in, sf_block_view(m.conformer.blocks[static_cast<size_t>(i)]),
                                        bp);
    }

    // encoder_proj (enc_d_model -> tf_d_model) then 18x post-LN transformer.
    ggml_tensor * t = linear(ctx, m.weights.enc_proj_w, x, m.weights.enc_proj_b);
    const int     d = m.hparams.tf_d_model;
    for (int i = 0; i < m.hparams.tf_n_layers; ++i) {
        t = tf_block(ctx, m.weights.tf_blocks[static_cast<size_t>(i)], t, d, m.hparams.tf_n_heads, T_concat);
    }

    // Diar head (forward_speaker_sigmoids). All frames valid in sync mode, so
    // the encoder_mask multiply is a no-op.
    ggml_tensor * h = ggml_relu(ctx, t);
    h               = linear(ctx, m.weights.fc1_w, h, m.weights.fc1_b);
    h               = ggml_relu(ctx, h);
    ggml_tensor * s     = linear(ctx, m.weights.single_spk_head_w, h, m.weights.single_spk_head_b);
    ggml_tensor * preds = ggml_sigmoid(ctx, s);  // [n_spk, T_concat]

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, preds);
    b.graph      = graph;
    b.concat_in  = concat_in;
    b.pos_emb_in = pos_emb_in;
    b.preds      = preds;
    return b;
}

}  // namespace

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<SortformerModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_sortformer_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    fill_conformer_hp(m->hparams, m->conformer_hp);

    // Mel front-end (NeMo AudioToMelSpectrogramPreprocessor; normalize=none).
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.normalize    = m->hparams.fe_normalize;
        cfg.pad_mode     = "constant";
        // NeMo AudioToMelSpectrogramPreprocessor uses ceil(n/hop) framing;
        // the mel tensor is pad_to=16 padded but forward() trims it back to
        // the real length before the encoder, so feeding the ceil length is
        // the faithful path (see forward-map Frontend note).
        cfg.nemo_seq_len_ceil = true;
        m->mel.emplace(cfg);
    }

    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = load_conformer_weights(m->ctx_meta, m->conformer_hp, m->conformer);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    if (const transcribe_status st = build_sortformer_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->gpu_device : 0, "sortformer", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "sortformer");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    if (const transcribe_status st = fuse_conformer_bn(*m); st != TRANSCRIBE_OK) {
        return st;
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model * model, const transcribe_session_params * params,
                               transcribe_session ** out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto pc       = std::make_unique<SortformerSession>();
    pc->model     = model;
    pc->n_threads = params->n_threads;
    pc->kv_type   = params->kv_type;
    *out_ctx = pc.release();
    return TRANSCRIBE_OK;
}

// Threshold-based probs -> speaker segments (simple offline segmentation;
// the DER-grade dihard3 post-processing is a later, streaming-checkpoint
// concern). probs is row-major [T, n_spk]. Emits one contiguous run per
// speaker as a speaker_segment row.
static void probs_to_speaker_segments(transcribe_session *   pc,
                                      const std::vector<float> & probs,
                                      int                        T,
                                      int                        n_spk,
                                      double                     ms_per_frame,
                                      float                      threshold) {
    for (int s = 0; s < n_spk; ++s) {
        int run_start = -1;
        for (int t = 0; t <= T; ++t) {
            const bool active = (t < T) && (probs[static_cast<size_t>(t) * n_spk + s] > threshold);
            if (active && run_start < 0) {
                run_start = t;
            } else if (!active && run_start >= 0) {
                transcribe_session::SpeakerSegmentEntry row;
                row.t0_ms      = static_cast<int64_t>(std::llround(run_start * ms_per_frame));
                row.t1_ms      = static_cast<int64_t>(std::llround(t * ms_per_frame));
                row.speaker_id = s + 1;  // 1-based
                row.p          = std::numeric_limits<float>::quiet_NaN();
                pc->speaker_segments.push_back(row);
                run_start = -1;
            }
        }
    }
}

// Lazily create the persistent multi-backend scheduler shared by the offline
// and streaming graphs.
static transcribe_status ensure_sched(SortformerSession * pc, SortformerModel * pm) {
    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    return TRANSCRIBE_OK;
}

// Offline full-context forward. Builds the whole encoder + transformer + diar
// head over the entire mel and dumps the Stage-4 parity tensors (enc.* +
// diar.preds_offline). Invoked only when tensor dumping is active (the
// streaming path is the product); keeps the offline tensor gate green.
static transcribe_status run_offline_forward(SortformerSession * pc, SortformerModel * pm, int mel_n_frames) {
    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip{};
        ip.mem_size     = 16 * 1024 * 1024;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        pc->compute_ctx = ggml_init(ip);
        if (pc->compute_ctx == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_context * ctx = pc->compute_ctx;

    pk::EncoderBuild eb =
        pk::build_encoder_graph(ctx, pm->conformer, pm->conformer_hp, mel_n_frames, GGML_TYPE_F32, pm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    const int T = static_cast<int>(eb.out->ne[1]);
    const int d = pm->hparams.tf_d_model;

    ggml_tensor * proj = linear(ctx, pm->weights.enc_proj_w, eb.out, pm->weights.enc_proj_b);
    ggml_tensor * x    = proj;
    for (int i = 0; i < pm->hparams.tf_n_layers; ++i) {
        x = tf_block(ctx, pm->weights.tf_blocks[static_cast<size_t>(i)], x, d, pm->hparams.tf_n_heads, T);
    }
    ggml_tensor * tf_out = x;

    ggml_tensor * h = ggml_relu(ctx, tf_out);
    h               = linear(ctx, pm->weights.fc1_w, h, pm->weights.fc1_b);
    h               = ggml_relu(ctx, h);
    ggml_tensor * s     = linear(ctx, pm->weights.single_spk_head_w, h, pm->weights.single_spk_head_b);
    ggml_tensor * preds = ggml_sigmoid(ctx, s);

    ggml_build_forward_expand(eb.graph, preds);
    transcribe::debug::mark_tensor_for_dump(proj);
    transcribe::debug::mark_tensor_for_dump(tf_out);
    transcribe::debug::mark_tensor_for_dump(preds);

    if (const transcribe_status st = ensure_sched(pc, pm); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, eb.graph)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(eb.mel_in, pc->mel_buf.data(), 0, pc->mel_buf.size() * sizeof(float));
    transcribe::debug::dump_tensor("enc.mel.in", eb.mel_in, "frontend");

    if (eb.pos_emb_in != nullptr) {
        const int d_model = pm->conformer_hp.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
        fill_rel_pos_emb(pc->pos_buf, pc->pos_div_term, pos_len, d_model);
        ggml_backend_tensor_set(eb.pos_emb_in, pc->pos_buf.data(), 0, pc->pos_buf.size() * sizeof(float));
    }

    transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);
    if (ggml_backend_sched_graph_compute(pc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer offline forward: graph_compute failed");
        return TRANSCRIBE_ERR_GGUF;
    }

    transcribe::debug::dump_tensor("enc.fastconformer.out", eb.out, "encoder");
    transcribe::debug::dump_tensor("enc.encoder_proj.out", proj, "encoder");
    transcribe::debug::dump_tensor("enc.transformer.out", tf_out, "encoder");
    transcribe::debug::dump_tensor("diar.preds_offline", preds, "encoder");
    return TRANSCRIBE_OK;
}

// Streaming AOSC/FIFO forward (the product path). Chunks the mel per NeMo's
// streaming_feat_loader, runs Graph A (pre_encode) then Graph B (blocks +
// proj + transformer + head) over the [spkcache|fifo|chunk] concat, and drives
// the host streaming_update_sync / _compress_spkcache state machine. Emits the
// accumulated T x n_spk probs as diar.probs and the speaker segments.
static transcribe_status run_streaming(SortformerSession * pc, SortformerModel * pm, int mel_n_mels, int mel_n_frames,
                                       double ms_per_frame, transcribe_sortformer_preset preset) {
    const SortformerStreamParams P     = resolve_stream_params(pm->hparams, preset);
    const int                    ed    = pm->conformer_hp.enc_d_model;
    const int                    n_spk = pm->hparams.max_speakers;
    const int                    sub   = pm->hparams.enc_subsampling_factor;
    const int                    feat_len = mel_n_frames;

    if (sub <= 0 || P.chunk_len <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    pc->stream.reset(ed);
    if (const transcribe_status st = ensure_sched(pc, pm); st != TRANSCRIBE_OK) {
        return st;
    }

    const int64_t t_stream_start = ggml_time_us();
    int           stt = 0, end = 0, chunk_idx = 0;
    while (end < feat_len) {
        if (pc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int left_offset  = std::min(P.chunk_left_context * sub, stt);
        end                    = std::min(stt + P.chunk_len * sub, feat_len);
        const int right_offset = std::min(P.chunk_right_context * sub, feat_len - end);
        const int win_lo       = stt - left_offset;
        const int win_hi       = end + right_offset;
        const int M            = win_hi - win_lo;
        // Diar-frame context: left_offset is a multiple of sub -> round is
        // exact; right_offset may be short on the final chunk -> ceil.
        const int lc = (left_offset + sub / 2) / sub;
        const int rc = (right_offset + sub - 1) / sub;

        if (pc->compute_ctx != nullptr) {
            ggml_free(pc->compute_ctx);
            pc->compute_ctx = nullptr;
        }
        {
            ggml_init_params ip{};
            ip.mem_size     = 32 * 1024 * 1024;
            ip.mem_buffer   = nullptr;
            ip.no_alloc     = true;
            pc->compute_ctx = ggml_init(ip);
            if (pc->compute_ctx == nullptr) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }
        ggml_context * ctx = pc->compute_ctx;
        transcribe::debug::push_name_prefix("stream.chunk");

        // ---- Graph A: pre_encode over the mel window ----
        PreEncodeBuild A = build_pre_encode_graph(ctx, *pm, M);
        if (A.graph == nullptr) {
            transcribe::debug::pop_name_prefix();
            return TRANSCRIBE_ERR_GGUF;
        }
        const int T_diar = static_cast<int>(A.out->ne[1]);

        // Mel window [n_mels, M] from full mel [n_mels, feat_len], cols
        // [win_lo, win_hi). ggml ne=[M, n_mels] -> offset mel*M + col.
        pc->chunk_mel_buf.resize(static_cast<size_t>(mel_n_mels) * M);
        for (int mel = 0; mel < mel_n_mels; ++mel) {
            const float * src = pc->mel_buf.data() + static_cast<size_t>(mel) * feat_len + win_lo;
            std::copy(src, src + M, pc->chunk_mel_buf.data() + static_cast<size_t>(mel) * M);
        }

        ggml_backend_sched_reset(pc->sched);
        if (!ggml_backend_sched_alloc_graph(pc->sched, A.graph)) {
            transcribe::debug::pop_name_prefix();
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(A.mel_in, pc->chunk_mel_buf.data(), 0, pc->chunk_mel_buf.size() * sizeof(float));
        transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);
        if (ggml_backend_sched_graph_compute(pc->sched, A.graph) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer streaming: pre_encode compute failed");
            transcribe::debug::pop_name_prefix();
            return TRANSCRIBE_ERR_GGUF;
        }
        pc->chunk_embs_host.resize(static_cast<size_t>(T_diar) * ed);
        ggml_backend_tensor_get(A.out, pc->chunk_embs_host.data(), 0, pc->chunk_embs_host.size() * sizeof(float));

        // ---- host concat [spkcache | fifo | chunk_embs] ----
        const int S        = pc->stream.spkcache_n;
        const int F        = pc->stream.fifo_n;
        const int T_concat = S + F + T_diar;
        pc->concat_host.resize(static_cast<size_t>(T_concat) * ed);
        std::copy(pc->stream.spkcache.begin(), pc->stream.spkcache.begin() + static_cast<size_t>(S) * ed,
                  pc->concat_host.begin());
        std::copy(pc->stream.fifo.begin(), pc->stream.fifo.begin() + static_cast<size_t>(F) * ed,
                  pc->concat_host.begin() + static_cast<size_t>(S) * ed);
        std::copy(pc->chunk_embs_host.begin(), pc->chunk_embs_host.end(),
                  pc->concat_host.begin() + static_cast<size_t>(S + F) * ed);

        // ---- Graph B: xscale + 17 blocks + proj + 18 tf + head ----
        StreamInferBuild B = build_stream_infer_graph(ctx, *pm, T_concat);
        if (B.graph == nullptr) {
            transcribe::debug::pop_name_prefix();
            return TRANSCRIBE_ERR_GGUF;
        }
        fill_rel_pos_emb(pc->pos_buf, pc->pos_div_term, 2 * T_concat - 1, ed);

        ggml_backend_sched_reset(pc->sched);
        if (!ggml_backend_sched_alloc_graph(pc->sched, B.graph)) {
            transcribe::debug::pop_name_prefix();
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_set(B.concat_in, pc->concat_host.data(), 0, pc->concat_host.size() * sizeof(float));
        ggml_backend_tensor_set(B.pos_emb_in, pc->pos_buf.data(), 0, pc->pos_buf.size() * sizeof(float));
        transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);
        if (ggml_backend_sched_graph_compute(pc->sched, B.graph) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer streaming: infer compute failed");
            transcribe::debug::pop_name_prefix();
            return TRANSCRIBE_ERR_GGUF;
        }
        pc->stream_preds_host.resize(static_cast<size_t>(T_concat) * n_spk);
        ggml_backend_tensor_get(B.preds, pc->stream_preds_host.data(), 0,
                                pc->stream_preds_host.size() * sizeof(float));

        transcribe::debug::pop_name_prefix();

        // ---- host streaming update (FIFO + AOSC compress) ----
        streaming_update_sync(pc->stream, P, n_spk, ed, pc->chunk_embs_host, T_diar, pc->stream_preds_host, T_concat,
                              lc, rc);

        stt = end;
        ++chunk_idx;
    }
    pc->t_encode_us = ggml_time_us() - t_stream_start;

    // Trim padding tail: NeMo total_preds[:, :ceil(feat_len/sub)].
    const int n_frames = (feat_len + sub - 1) / sub;
    const int used_n   = std::min(pc->stream.total_n, n_frames);

    if (transcribe::debug::enabled()) {
        const long long shape[2] = { used_n, n_spk };
        transcribe::debug::dump_host_f32("diar.probs", pc->stream.total_preds.data(),
                                         static_cast<long long>(used_n) * n_spk, shape, 2, "diarize");
    }

    probs_to_speaker_segments(pc, pc->stream.total_preds, used_n, n_spk, ms_per_frame, /*threshold=*/0.5f);
    (void) chunk_idx;
    return TRANSCRIBE_OK;
}

transcribe_status run(transcribe_session * session, const float * pcm, int n_samples,
                      const transcribe_run_params * params) {
    auto * pc = static_cast<SortformerSession *>(session);
    auto * pm = static_cast<SortformerModel *>(session->model);

    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    // Streaming operating-point run ext (kind/size/range already validated
    // by the dispatcher + run_validate pre-clear; re-check is belt-and-braces).
    transcribe_sortformer_preset preset = TRANSCRIBE_SORTFORMER_PRESET_DEFAULT;
    if (params != nullptr && params->family != nullptr) {
        if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM,
                                                              sizeof(struct transcribe_sortformer_stream_ext));
            st != TRANSCRIBE_OK) {
            return st;
        }
        preset = reinterpret_cast<const transcribe_sortformer_stream_ext *>(params->family)->preset;
    }
    pc->clear_result();
    transcribe::debug::init();

    if (!pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    int mel_n_mels = 0, mel_n_frames = 0;
    if (const transcribe_status mst =
            pm->mel->compute(pcm, static_cast<size_t>(n_samples), pc->mel_buf, mel_n_mels, mel_n_frames);
        mst != TRANSCRIBE_OK) {
        return mst;
    }

    const double ms_per_frame = 1000.0 * static_cast<double>(pm->hparams.frame_hop) /
                                static_cast<double>(pm->hparams.fe_sample_rate);

    // Offline forward: parity dumps only. Gated behind an explicit env
    // (set by validate.py cmd_cpp) because it runs full-context O(T^2)
    // attention over the WHOLE clip — fine for the short oracle, but it would
    // OOM on the many-minute audio the DER runner feeds (which also enables
    // dumping, to read diar.probs). The streaming path is always the product.
    if (transcribe::debug::enabled() && std::getenv("TRANSCRIBE_SORTFORMER_OFFLINE_DUMP") != nullptr) {
        if (const transcribe_status st = run_offline_forward(pc, pm, mel_n_frames); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    // Streaming AOSC/FIFO forward drives the product output + diar.probs.
    if (const transcribe_status st = run_streaming(pc, pm, mel_n_mels, mel_n_frames, ms_per_frame, preset);
        st != TRANSCRIBE_OK) {
        return st;
    }

    pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    pc->has_result  = true;
    return TRANSCRIBE_OK;
}

// Kind+slot probe. Sortformer ships one RUN-slot extension (the streaming
// operating-point preset); there is no STREAM-slot surface (no push-audio
// entry point yet — a future one registers a separate kind).
static bool accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    if (model == nullptr) {
        return false;
    }
    if (slot != TRANSCRIBE_EXT_SLOT_RUN) {
        return false;
    }
    return kind == TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM;
}

// Pre-clear validation for the _RUN slot (see Arch::run_validate): reject a
// malformed ext or an out-of-range preset before the previous result
// snapshot is destroyed.
static transcribe_status run_validate(const transcribe_session * /*ctx*/, const transcribe_run_params * params) {
    if (params == nullptr || params->family == nullptr) {
        return TRANSCRIBE_OK;  // NULL ext -> family defaults
    }
    if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM,
                                                          sizeof(struct transcribe_sortformer_stream_ext));
        st != TRANSCRIBE_OK) {
        return st;
    }
    const auto * ext = reinterpret_cast<const transcribe_sortformer_stream_ext *>(params->family);
    switch (ext->preset) {
        case TRANSCRIBE_SORTFORMER_PRESET_DEFAULT:
        case TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY:
        case TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY:
        case TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY:
            return TRANSCRIBE_OK;
    }
    return TRANSCRIBE_ERR_INVALID_ARG;
}

extern const Arch arch = {
    /* .name             = */ "sortformer",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ accepts_ext_kind,
    /* .run_validate     = */ run_validate,
};

}  // namespace transcribe::sortformer

// ---------------------------------------------------------------------------
// Public sortformer extension init function (global scope, C linkage).
// Defined here so transcribe.cpp stays family-agnostic; stamps the
// transcribe_ext header (size + kind) and the preset default.
// ---------------------------------------------------------------------------

extern "C" void transcribe_sortformer_stream_ext_init(struct transcribe_sortformer_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM;
    p->preset   = TRANSCRIBE_SORTFORMER_PRESET_DEFAULT;
}
