// arch/whistle/encoder.h - Whistle frontend, encoder graph, and the block
// helpers (ZCRMSNorm, HadamardMLP, mHC) shared with the decoder graphs.

#pragma once

#include "ggml.h"
#include "weights.h"

#include <functional>
#include <vector>

namespace transcribe::whistle {

struct WhistleAux;

// ---------------------------------------------------------------------------
// Frontend (host). Identified by probing the engine (forward-map Frontend):
//   frames = n // 160, frame i centered on sample 160*i (zero pad 200),
//   symmetric Hann(400) in a 512-point rfft, |X|^2 @ filterbank,
//   ln(mel + 1e-8 * max(mel)), per-utterance per-bin mean/std.
// ---------------------------------------------------------------------------
struct MelResult {
    std::vector<float> mel;  // [n_frames][n_mels] normalized
    int                n_frames         = 0;
    // No-speech statistic: p99 - p10 of per-frame mel energy in dB.
    double             energy_spread_db = 0.0;
};

// fb is ne [n_mels, n_fft/2 + 1] (row = FFT bin), as stored in the GGUF.
bool compute_mel(const float *              pcm,
                 int                        n_samples,
                 const std::vector<float> & fb,
                 int                        n_mels,
                 MelResult &                out,
                 int                        n_threads = 1);

// No-speech gate threshold on MelResult::energy_spread_db. Fitted
// approximation of an undocumented engine gate (user-signed, see
// docs/porting/families/whistle.md): 10*log10(3).
constexpr double kNoSpeechSpreadDb = 4.771212547196624;

// ---------------------------------------------------------------------------
// Graph helpers.
// ---------------------------------------------------------------------------

// ZCRMSNorm: x / rms(x) * (1 + scale), eps 1e-6, over ne0. `scale` must be a
// norm weight already folded to 1 + scale by build_aux (model.cpp).
ggml_tensor * zcrms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale);

// Rows [r0, r0 + n) (ne0 slice) of a projection output y [R, N]; used to
// split the fused projection matmuls (WhistleAttn::fused, phi_fused).
ggml_tensor * out_rows(ggml_context * ctx, ggml_tensor * y, int64_t r0, int64_t n);

// Hadamard-domain weights (WhistleHParams::hadamard_group): returns H_g applied
// to every g-block of x's ne0, so mul_mat(C, hada_in(x)) == mul_mat(W, x).
// Identity when h is null.
ggml_tensor * hada_in(ggml_context * ctx, ggml_tensor * x, const WhistleHada & h);

// HadamardMLP over x [d, N].
ggml_tensor * hmlp(ggml_context * ctx, ggml_tensor * x, const WhistleHmlp & h, const WhistleAux & aux);

// One mHC layer. stream [d, lanes, N]; block maps u [d, N] -> block(u) [d, N].
// Host-memory view of one mHC layer for the fused CPU custom ops.
struct MhcHost {
    int           d = 0, lanes = 0;
    float         a_pre = 0.0f, a_post = 0.0f, a_res = 0.0f;
    const float * pre_bias  = nullptr;  // [lanes], lane offsets folded in
    const float * post_bias = nullptr;  // [lanes]
    const float * b_res     = nullptr;  // [lanes * lanes]
};

struct MhcLayer {
    const WhistleMhc *  mhc       = nullptr;
    ggml_tensor *       pre_bias  = nullptr;  // ne [lanes, L] (aux)
    ggml_tensor *       post_bias = nullptr;
    float               a_pre     = 0.0f;
    float               a_post    = 0.0f;
    float               a_res     = 0.0f;
    int                 layer     = 0;
    bool                cpu_fused = false;    // WhistleAux::cpu_fused_ops
    const MhcHost *     host      = nullptr;  // fused CPU ops when set
    const WhistleHada * hada      = nullptr;  // &WhistleWeights::hada
};

ggml_tensor * mhc_step(ggml_context *                                      ctx,
                       ggml_tensor *                                       stream,
                       const MhcLayer &                                    l,
                       const std::function<ggml_tensor *(ggml_tensor *)> & block);

// Lane mean of a [d, lanes, N] stream -> [d, N].
ggml_tensor * lane_mean(ggml_context * ctx, ggml_tensor * stream);

// ---------------------------------------------------------------------------
// Encoder graph.
// ---------------------------------------------------------------------------
struct EncoderDumps {
    ggml_tensor *              stem_out = nullptr;  // [d, T]
    std::vector<ggml_tensor *> block_out;           // lane mean after each layer, [d, T]
};

struct EncoderBuild {
    ggml_cgraph * graph  = nullptr;
    ggml_tensor * mel_in = nullptr;  // [n_mels, T_mel]
    ggml_tensor * pos_in = nullptr;  // [T_enc] I32 (RoPE positions)
    ggml_tensor * out    = nullptr;  // [d, T_enc] final-normed
    int           T_enc  = 0;
    EncoderDumps  dumps;
};

int predict_t_enc(int n_mel_frames);

EncoderBuild build_encoder_graph(ggml_context *         ctx,
                                 const WhistleWeights & w,
                                 const WhistleHParams & hp,
                                 const WhistleAux &     aux,
                                 int                    n_mel_frames,
                                 bool                   want_dumps);

}  // namespace transcribe::whistle
