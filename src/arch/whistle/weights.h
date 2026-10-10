// arch/whistle/weights.h - Whistle tensor catalog and hyperparameters.
//
// INTERNAL to src/arch/whistle/. Whistle (Cactus Compute) is an
// encoder-decoder speech model built from Needle-3 parts:
//
//   log-mel (80) -> conv stem (3x stride-2 DW/PW, 128 ch) -> Linear 1280->512
//   -> 8 encoder blocks inside a 4-lane mHC residual stream
//      {HadamardMLP_0 (x0.5) -> GQA self-attn (RoPE) -> conv module -> HadamardMLP (x0.5)}
//   -> lane mean -> ZCRMSNorm
//   decoder: 8 Needle-3 blocks inside a 4-lane mHC stream
//      {Engram (layers 3, 7) -> causal-tap GQA self-attn (RoPE) -> gated cross-attn -> HadamardMLP}
//   -> lane mean -> ZCRMSNorm -> tied head
//
// See reports/porting/whistle/forward-map.md for the provenance of every op.
// Linear weights are stored [in, out] in ggml ne order (PyTorch [out, in]);
// no tensor has a bias.

#pragma once

#include "transcribe.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::whistle {

struct WhistleHParams {
    int32_t d_model     = 0;
    int32_t n_heads     = 0;
    int32_t n_kv_heads  = 0;
    int32_t qk_head_dim = 0;
    int32_t v_head_dim  = 0;
    int32_t mhc_lanes   = 0;
    int32_t hada_n      = 0;
    float   rope_theta  = 0.0f;
    int32_t vocab_size  = 0;

    int32_t enc_n_layers    = 0;
    int32_t enc_conv_kernel = 0;
    int32_t stem_channels   = 0;
    int32_t stem_stages     = 0;

    // stt.whistle.hadamard_group: every CQ-sourced matrix is stored in its
    // Walsh-Hadamard domain, C = W blockdiag(H_g) along the input axis, so
    // W x = C (H_g x) (convert-whistle.py --hadamard-domain). 0 = plain weights.
    int32_t hadamard_group = 0;

    int32_t dec_n_layers = 0;
    int32_t dec_max_seq  = 0;
    int32_t dec_taps     = 0;

    int32_t              engram_slots         = 0;
    int32_t              engram_sub_dim       = 0;
    int32_t              engram_n_tables      = 0;
    int32_t              engram_conv_taps     = 0;
    int32_t              engram_conv_dilation = 0;
    int32_t              engram_seed_heads    = 0;
    std::vector<int32_t> engram_orders;
    std::vector<int32_t> engram_sites;

    int32_t                  lang_token_base   = 0;
    int32_t                  max_audio_samples = 0;
    std::vector<std::string> languages;
    std::vector<int32_t>     lang_token_ids;

    // Frontend.
    int32_t fe_sample_rate = 0;
    int32_t fe_num_mels    = 0;
    int32_t fe_n_fft       = 0;
    int32_t fe_win_length  = 0;
    int32_t fe_hop_length  = 0;

    int32_t stem_freq_out() const { return fe_num_mels >> stem_stages; }

    int32_t engram_heads() const {
        return engram_orders.empty() ? 0 : engram_n_tables / static_cast<int32_t>(engram_orders.size());
    }
};

transcribe_status read_whistle_hparams(const gguf_context * gguf, WhistleHParams & hp);

// HadamardMLP: Kronecker Walsh factors (a: [ba, ba], b: [bb, bb]) per stage,
// per-channel scales, and a rank-8 softmax conditioning.
struct WhistleHmlp {
    ggml_tensor *                d1 = nullptr;
    ggml_tensor *                d2 = nullptr;
    ggml_tensor *                b2 = nullptr;
    ggml_tensor *                d3 = nullptr;
    ggml_tensor *                d4 = nullptr;
    std::array<ggml_tensor *, 3> wa{};                // [ba, ba]
    std::array<ggml_tensor *, 3> wb{};                // [bb, bb]
    ggml_tensor *                cond_v   = nullptr;  // ne [8, d_model]
    ggml_tensor *                cond_u   = nullptr;  // ne [d_model, 8]
    // Transposed copies built once at load (model.cpp build_aux).
    ggml_tensor *                cond_v_t = nullptr;  // ne [d_model, 8]
    ggml_tensor *                cond_u_t = nullptr;  // ne [8, d_model]
    std::array<ggml_tensor *, 3> wa_t{};
    std::array<ggml_tensor *, 3> wb_t{};
    // Host weight pointers for the fused CPU op (set only when every graph
    // runs on the CPU backend; see WhistleAux::cpu_fused_ops).
    const struct HmlpHost *      host = nullptr;
};

// Host-memory view of one HadamardMLP for the fused CPU custom op.
struct HmlpHost {
    int             d = 0, ba = 0, bb = 0;
    const float *   d1 = nullptr;
    const float *   d2 = nullptr;
    const float *   b2 = nullptr;
    const float *   d3 = nullptr;
    const float *   d4 = nullptr;
    const float *   a[3]{};            // [ba][ba] row-major (row i, column k)
    const float *   b[3]{};            // [bb][bb] row-major (row j, column l)
    const float *   cond_v = nullptr;  // [d][8]
    const float *   cond_u = nullptr;  // [8][d]
    const int32_t * perm1  = nullptr;
    const int32_t * perm2  = nullptr;
};

struct WhistleAttn {
    ggml_tensor * q      = nullptr;  // [d_model, n_heads*qk]
    ggml_tensor * k      = nullptr;  // [d_model, n_kv*qk]
    ggml_tensor * v      = nullptr;  // [d_model, n_kv*v]
    ggml_tensor * gate   = nullptr;  // [d_model, n_heads*v]
    ggml_tensor * out    = nullptr;  // [n_heads*v, d_model]
    ggml_tensor * q_norm = nullptr;  // [qk]
    ggml_tensor * k_norm = nullptr;  // [qk]
    // Decoder self-attention only: causal taps over the raw projections.
    ggml_tensor * q_taps = nullptr;  // [n_heads*qk, taps]
    ggml_tensor * k_taps = nullptr;
    ggml_tensor * v_taps = nullptr;
    // Load-time row concatenation of the projections that read the same input
    // (q|k|v|gate for self-attention, q|gate for cross-attention); the
    // separate tensors above are then left unallocated. Null when the parts
    // do not share one type (the graphs then use the separate tensors).
    ggml_tensor * fused  = nullptr;
};

struct WhistleEncBlock {
    ggml_tensor * norm_hmlp_0    = nullptr;
    ggml_tensor * norm_in        = nullptr;
    ggml_tensor * norm_post_attn = nullptr;
    ggml_tensor * norm_conv      = nullptr;
    ggml_tensor * norm_conv_out  = nullptr;
    ggml_tensor * norm_hmlp      = nullptr;
    WhistleAttn   attn;
    ggml_tensor * attn_gate = nullptr;  // [1]
    ggml_tensor * conv_pw1  = nullptr;  // [d_model, 2*d_model]
    ggml_tensor * conv_pw2  = nullptr;  // [d_model, d_model]
    ggml_tensor * conv_dw   = nullptr;  // ne [d_model, k]
    WhistleHmlp   hmlp_0;
    WhistleHmlp   hmlp;
};

struct WhistleDecBlock {
    ggml_tensor * norm_in = nullptr;
    WhistleAttn   attn;
    ggml_tensor * norm_post_attn = nullptr;
    ggml_tensor * attn_gate      = nullptr;
    ggml_tensor * norm_hmlp      = nullptr;
    WhistleHmlp   hmlp;
    ggml_tensor * norm_cross = nullptr;
    WhistleAttn   cross;
    ggml_tensor * norm_post_cross = nullptr;
    ggml_tensor * cross_gate      = nullptr;
};

// mHC (manifold-constrained hyper-connection) parameters for one stack.
struct WhistleMhc {
    ggml_tensor *              a_pre    = nullptr;  // [L]
    ggml_tensor *              a_post   = nullptr;
    ggml_tensor *              a_res    = nullptr;
    ggml_tensor *              b_pre    = nullptr;  // ne [lanes, L]
    ggml_tensor *              b_post   = nullptr;
    ggml_tensor *              b_res    = nullptr;  // ne [lanes, lanes, L]
    ggml_tensor *              phi_pre  = nullptr;  // ne [lanes*d_model, L*lanes]
    ggml_tensor *              phi_post = nullptr;
    ggml_tensor *              phi_res  = nullptr;  // ne [lanes*d_model, L*lanes*lanes]
    // Per layer: rows pre | post | res of that layer in one tensor (see
    // WhistleAttn::fused); empty when not fused.
    std::vector<ggml_tensor *> phi_fused;
};

// Input rotation for Hadamard-domain weights (WhistleHParams::hadamard_group).
struct WhistleHada {
    // H_g = WalshHadamard(g) / sqrt(g), ne [g, g] F32 (aux buffer); null for
    // plain weights.
    ggml_tensor * h    = nullptr;
    // CPU graphs: apply H_g with the fused fast Walsh-Hadamard op instead of
    // a mul_mat by h.
    bool          fwht = false;
};

struct WhistleEngram {
    ggml_tensor * tables     = nullptr;  // ne [sub_dim, n_tables*slots]
    // Tables stored two rows per stored row, ne [2*sub_dim, n_tables*slots/2]
    // (same bytes as F32; lets 256-wide k-quant blocks hold 128-wide rows).
    // All sites share one layout.
    bool          paired     = false;
    ggml_tensor * key_proj   = nullptr;  // [n_tables*sub_dim, d_model]
    ggml_tensor * value_proj = nullptr;
    ggml_tensor * conv_taps  = nullptr;  // ne [d_model, conv_taps]
};

struct WhistleStem {
    ggml_tensor * conv_w = nullptr;  // ne [C, 9]  (row = kt*3 + kf)
    ggml_tensor * dw_1   = nullptr;  // ne [C, 9]
    ggml_tensor * pw_1   = nullptr;  // [C, C]
    ggml_tensor * dw_2   = nullptr;
    ggml_tensor * pw_2   = nullptr;
    ggml_tensor * out    = nullptr;  // [C*F', d_model], input index c*F' + f
};

struct WhistleWeights {
    WhistleStem                  stem;
    std::vector<WhistleEncBlock> enc_blocks;
    WhistleMhc                   enc_mhc;
    ggml_tensor *                enc_final_norm = nullptr;

    ggml_tensor *                token_embd = nullptr;  // [d_model, vocab]
    // CPU repack only: the tied head padded to a multiple of 8 rows (zero rows)
    // in the repacked layout; null otherwise (the head then uses token_embd).
    ggml_tensor *                head_rp    = nullptr;
    std::vector<WhistleDecBlock> dec_blocks;
    WhistleMhc                   dec_mhc;
    std::vector<WhistleEngram>   engrams;
    ggml_tensor *                dec_final_norm = nullptr;
    ggml_tensor *                pe_gate        = nullptr;

    WhistleHada hada;

    ggml_tensor * hada_perm1 = nullptr;  // [hada_n] F32 (integer-valued)
    ggml_tensor * hada_perm2 = nullptr;
};

transcribe_status build_whistle_weights(ggml_context * ctx_meta, const WhistleHParams & hp, WhistleWeights & w);

}  // namespace transcribe::whistle
