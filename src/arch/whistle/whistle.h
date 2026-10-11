// arch/whistle/whistle.h - Whistle model and session types.
//
// INTERNAL to src/arch/whistle/. Defines the concrete transcribe_model /
// transcribe_session subclasses for the Whistle family (Cactus Compute
// Needle-3 speech encoder-decoder). See weights.h for the architecture
// summary and reports/porting/whistle/forward-map.md for provenance.

#pragma once

#include "encoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::whistle {

void apply_family_invariants(transcribe_model & model);

// Host copies of the small per-layer scalars the graphs bake in as
// constants, plus load-time derived tensors (integer permutations, mHC
// bias vectors with the lane offsets folded in).
struct WhistleAux {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * perm1 = nullptr;  // [hada_n] I32
    ggml_tensor * perm2 = nullptr;

    // mHC: b_pre + pre_off and b_post + post_off, ne [lanes, L].
    ggml_tensor * enc_pre_bias  = nullptr;
    ggml_tensor * enc_post_bias = nullptr;
    ggml_tensor * dec_pre_bias  = nullptr;
    ggml_tensor * dec_post_bias = nullptr;

    std::vector<float> enc_a_pre, enc_a_post, enc_a_res;
    std::vector<float> dec_a_pre, dec_a_post, dec_a_res;
    std::vector<float> enc_attn_gate;  // sigmoid applied
    std::vector<float> dec_attn_gate;
    std::vector<float> dec_cross_gate;
    float              pe_gate = 0.0f;  // sigmoid applied

    // Graphs run on the CPU backend: use fused CPU custom ops (bit-identical
    // to the generic ggml graphs they replace) where available.
    bool                  cpu_fused_ops = false;
    // Host views + integer permutations backing the fused CPU HadamardMLP.
    std::vector<HmlpHost> hmlp_host;
    std::vector<int32_t>  perm1_host, perm2_host;
    // Host views backing the fused CPU mHC ops (one per layer).
    std::vector<MhcHost>  enc_mhc_host, dec_mhc_host;
    std::vector<float>    mhc_bias_host;  // pre/post biases + b_res, both stacks

    // frontend.mel_filterbank, ne [num_mels, n_fft/2+1] (row = FFT bin).
    std::vector<float> mel_fb;

    void free();
};

// Decoder state for n_seq parallel hypotheses (beams x utterances).
// Every per-sequence tensor has the sequence index as its slowest
// dimension so beam reordering is one get_rows per tensor.
struct WhistleDecCache {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx   = 0;  // self-attention positions
    int n_seq   = 0;  // hypotheses
    int n_utt   = 0;  // utterances (cross caches); n_seq % n_utt == 0
    int T_enc   = 0;  // cross positions (max over utterances)
    int tap_len = 0;  // ring length for the q/k/v tap history
    int eg_len  = 0;  // ring length for the Engram value history

    // Per decoder layer.
    std::vector<ggml_tensor *> k_self;   // [qk, n_ctx, n_kv, n_seq]
    std::vector<ggml_tensor *> v_self;   // [n_ctx, vd, n_kv, n_seq]  (transposed)
    std::vector<ggml_tensor *> tap_q;    // [n_heads*qk, tap_len, n_seq]  raw projections
    std::vector<ggml_tensor *> tap_k;    // [n_kv*qk, tap_len, n_seq]
    std::vector<ggml_tensor *> tap_v;    // [n_kv*vd, tap_len, n_seq]
    std::vector<ggml_tensor *> k_cross;  // [qk, T_enc, n_heads, n_utt]
    std::vector<ggml_tensor *> v_cross;  // [T_enc, vd, n_heads, n_utt]  (transposed)
    // Per Engram site.
    std::vector<ggml_tensor *> eg_val;  // [d_model, eg_len, n_seq]  value_proj outputs

    void free();
};

bool dec_cache_init(WhistleDecCache &      cache,
                    ggml_backend_t         backend,
                    const WhistleHParams & hp,
                    int                    n_ctx,
                    int                    n_seq,
                    int                    n_utt,
                    int                    T_enc);

struct WhistleModel final : public transcribe_model {
    Tokenizer      tok;
    WhistleHParams hparams;
    WhistleWeights weights;
    WhistleAux     aux;
    ggml_context * ctx_meta    = nullptr;
    ggml_context * ctx_fused   = nullptr;  // fused projection weights (see WhistleAttn::fused)
    // CPU, Hadamard-domain file: the rotation was folded into the linear
    // weights at load, so the graphs feed them plain activations.
    bool           hada_folded = false;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer     = nullptr;
    // CPU only: matmul-only weights in ggml's repacked (interleaved) layout.
    ggml_backend_buffer_t   repack_buffer      = nullptr;
    ggml_context *          ctx_head           = nullptr;  // WhistleWeights::head_rp
    ggml_backend_buffer_t   head_buffer        = nullptr;
    // CPU only: persistent worker pool for every graph compute (a pool per
    // compute would spawn and join threads each decode step). Owned by the
    // model: compute is serialized across a model's sessions (transcribe.h).
    ggml_threadpool_t       threadpool         = nullptr;
    int                     threadpool_threads = 0;

    WhistleModel() = default;
    ~WhistleModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct WhistleSession final : public transcribe_session {
    WhistleDecCache cache;

    std::vector<float>   mel_buf;   // [T_mel][num_mels] normalized log-mel
    std::vector<float>   enc_host;  // [T_enc][d_model] encoder output
    std::vector<float>   mem_host;  // [T_enc][d_model] cross-attention memory
    // Reused ggml metadata arena for the per-graph compute contexts (a fresh
    // multi-MB allocation per decode step costs page faults every step).
    std::vector<uint8_t> graph_meta;
    std::vector<char>    reorder_tmp;

    WhistleSession() = default;
    ~WhistleSession() override;
};

}  // namespace transcribe::whistle
