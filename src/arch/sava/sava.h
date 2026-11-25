// arch/sava/sava.h - Inworld SAVA family (TURN and VAD roles): hparams,
// weights, the per-session stream state and the session types.
//
// INTERNAL to src/arch/sava/.
//
// One step reads one 640-sample frame and emits an on-time speech logit, a
// lookahead speech logit (judging the frame lookahead_frames back) and the
// turn logits of every horizon:
//
//   fe     causal log-mel: 4 Hann windows of 400 ending on each 160 hop
//          (host, fp64: the DFT of a quiet bin is cancellation-bound in fp32)
//   mixer  rms(W [row_{t-1} | row_t]) with row = [mel (512) | ln rms]
//   trunk  Mamba-2 blocks and sliding-window attention blocks (pre-norm,
//          SwiGLU FFN), final RMSNorm
//   vad    down-proj, attention block, RMSNorm, two readouts
//   turn   [rms(trunk) | embed(sigmoid(speech logit of step t - 1))] ->
//          down-proj, attention block, RMSNorm, readout
//
// Steps run as one ggml graph over a block of frames (graph.cpp). Every
// per-frame value is computed by the same kernels in the same order
// whatever the block size, so a stream fed in any chunking is bit-identical
// to a whole-clip run on the CPU. See docs/porting/families/sava.md.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-load-common.h"
#include "transcribe-model.h"
#include "transcribe-turn.h"
#include "transcribe-vad.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace transcribe {
struct Arch;
struct SessionCore;
}  // namespace transcribe

namespace transcribe::sava {

struct AttnHParams {
    int32_t n_heads    = 0;
    int32_t n_kv_heads = 0;
    int32_t head_dim   = 0;
    int32_t window     = 0;
    float   rope_theta = 10000.0f;
    bool    xsa        = false;
    float   xsa_alpha  = 1.0f;
};

struct HParams {
    // Front end.
    int32_t sample_rate      = 16000;
    int32_t frame_samples    = 640;
    int32_t n_fft            = 400;
    int32_t hop              = 160;
    int32_t n_mels           = 128;
    int32_t context_samples  = 240;  // n_fft - hop: the samples before a frame its windows read
    float   mel_floor        = 1e-10f;
    float   log_energy_floor = 1e-5f;

    // Trunk.
    int32_t           hidden  = 512;
    int32_t           ffn     = 2048;
    float             rms_eps = 1e-6f;
    std::vector<bool> layer_is_ssm;  // per trunk block: Mamba-2 (true) or attention
    int32_t           mixer_context_frames = 2;

    int32_t ssm_expand      = 2;
    int32_t ssm_head_dim    = 64;
    int32_t ssm_state       = 128;
    int32_t ssm_groups      = 1;
    int32_t ssm_conv_kernel = 4;

    AttnHParams attn;

    // Heads.
    int32_t     vad_hidden = 512;
    int32_t     vad_ffn    = 1536;
    AttnHParams vad_attn;
    int32_t     turn_hidden = 512;
    int32_t     turn_ffn    = 1536;
    AttnHParams turn_attn;

    int32_t n_horizons       = 4;
    int32_t vad_embed        = 64;
    int32_t lookahead_frames = 3;
    float   no_speech_logit  = -30.0f;

    // Inworld's recommended turn cut-off. The GGUF also records the rest of
    // Inworld's reference turn rule (stt.sava.policy.*); the library does
    // not use it.
    double threshold = 0.9;

    // Derived by read_hparams.
    int32_t hops_per_frame = 0;  // frame_samples / hop
    int32_t n_bins         = 0;  // n_fft / 2 + 1
    int32_t mel_width      = 0;  // hops_per_frame * n_mels
    int32_t row_width      = 0;  // mel_width + 1
    int32_t ssm_inner      = 0;  // expand * hidden
    int32_t ssm_heads      = 0;  // inner / head_dim
    int32_t ssm_conv_dim   = 0;  // inner + 2 * groups * state
};

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp);

struct MlpWeights {
    ggml_tensor * norm = nullptr;  // ne [d]
    ggml_tensor * gate = nullptr;  // ne [d, ffn]
    ggml_tensor * up   = nullptr;  // ne [d, ffn]
    ggml_tensor * down = nullptr;  // ne [ffn, d]
};

struct AttnWeights {
    ggml_tensor * norm   = nullptr;  // ne [d]
    ggml_tensor * qkv    = nullptr;  // ne [d, (H + 2 KV) hd], rows q | k | v
    ggml_tensor * q_norm = nullptr;  // ne [hd]
    ggml_tensor * k_norm = nullptr;  // ne [hd]
    ggml_tensor * out    = nullptr;  // ne [H hd, d]
};

struct SsmWeights {
    ggml_tensor * norm       = nullptr;  // ne [d]
    ggml_tensor * in         = nullptr;  // ne [d, inner + conv_dim + heads], rows z | xBC | dt
    ggml_tensor * conv_w     = nullptr;  // ne [K, conv_dim]
    ggml_tensor * conv_b     = nullptr;  // ne [conv_dim]
    ggml_tensor * dt_b       = nullptr;  // ne [heads]
    ggml_tensor * a          = nullptr;  // ne [1, heads], -exp(A_log)
    ggml_tensor * d          = nullptr;  // ne [heads]
    ggml_tensor * norm_gated = nullptr;  // ne [inner]
    ggml_tensor * out        = nullptr;  // ne [inner, d]
};

struct Block {
    bool        is_ssm = false;
    SsmWeights  ssm;
    AttnWeights attn;
    MlpWeights  mlp;
};

struct Weights {
    ggml_tensor * mixer_proj = nullptr;  // ne [ctx * row_width, d]
    ggml_tensor * mixer_norm = nullptr;  // ne [d]

    std::vector<Block> trunk;
    ggml_tensor *      trunk_norm = nullptr;

    ggml_tensor * vad_norm_in = nullptr;
    ggml_tensor * vad_down    = nullptr;  // ne [d, vad_hidden]
    AttnWeights   vad_attn;
    MlpWeights    vad_mlp;
    ggml_tensor * vad_norm_out = nullptr;
    ggml_tensor * vad_ro_w     = nullptr;  // ne [vad_hidden, 1]
    ggml_tensor * vad_ro_b     = nullptr;  // ne [1]
    ggml_tensor * vad_la_w     = nullptr;
    ggml_tensor * vad_la_b     = nullptr;

    ggml_tensor * turn_norm_in = nullptr;
    ggml_tensor * turn_emb_w   = nullptr;  // ne [vad_embed]
    ggml_tensor * turn_emb_b   = nullptr;  // ne [vad_embed]
    ggml_tensor * turn_down    = nullptr;  // ne [d + vad_embed, turn_hidden]
    AttnWeights   turn_attn;
    MlpWeights    turn_mlp;
    ggml_tensor * turn_norm_out = nullptr;
    ggml_tensor * turn_ro_w     = nullptr;  // ne [turn_hidden, n_horizons]
    ggml_tensor * turn_ro_b     = nullptr;  // ne [n_horizons]
};

transcribe_status bind_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w);

struct Model final : public transcribe_model {
    HParams       hparams;
    Weights       weights;
    TurnModelInfo turn_info;

    // RoPE frequencies (rope.freqs, the reference's fp32 table); angles are
    // (double) position * (double) freq.
    std::vector<float> rope_freqs;

    // Host front end (fp64): mel filterbank [n_mels][n_bins]
    // (frontend.mel_filterbank), periodic Hann window [n_fft] and the FFT's
    // twiddles exp(-2 pi i k / n_fft), k < n_fft.
    std::vector<double> mel_bank;
    std::vector<double> window;
    std::vector<double> twiddle_re;
    std::vector<double> twiddle_im;

    ggml_context *             ctx_meta = nullptr;
    load_common::WeightStorage weight_storage;
    BackendPlan                plan;

    Model() = default;
    ~Model() override;
};

// Per-session stream state that lives in a backend buffer and is read and
// written by the step graph, plus the host-side context.
struct AttnState {
    ggml_tensor * k_ring = nullptr;  // ne [KV hd, ring]: normed, rotated keys, slot = position % ring
    ggml_tensor * v_ring = nullptr;  // ne [ring, KV hd]: values, transposed (the value product reads it in place)
    int32_t       ring   = 0;        // window + k_block_frames
    int32_t       window = 0;
};

struct SsmState {
    ggml_tensor * conv = nullptr;  // ne [K - 1, conv_dim]: the last raw xBC inputs, time first
    ggml_tensor * ssm  = nullptr;  // ne [state, head_dim, heads, 1]
};

// The step graph of the current (n_graph, n_real) shape; its tensors live in
// the session's compute_ctx.
struct GraphTensors {
    ggml_cgraph * graph = nullptr;

    // Inputs.
    ggml_tensor *              mel_in     = nullptr;  // ne [mel_width, n]
    ggml_tensor *              log_energy = nullptr;  // ne [1, n]
    ggml_tensor *              ssm_ids    = nullptr;  // I32 [1]
    ggml_tensor *              rope_cos   = nullptr;  // ne [hd / 2, 1, n]
    ggml_tensor *              rope_sin   = nullptr;
    std::vector<ggml_tensor *> slots;                 // per attention layer: I64 [n], ring slot of each column
    std::vector<ggml_tensor *> masks;                 // per attention layer: ne [ring, n]

    // Outputs.
    ggml_tensor * logit    = nullptr;  // ne [1, n]
    ggml_tensor * logit_la = nullptr;  // ne [1, n]
    ggml_tensor * turn     = nullptr;  // ne [n_horizons, n]

    // Debug dump points (TRANSCRIBE_DUMP_DIR only), ne [C, n].
    ggml_tensor *              mel   = nullptr;
    ggml_tensor *              mixer = nullptr;
    std::vector<ggml_tensor *> trunk_mix;  // per block
    std::vector<ggml_tensor *> trunk_out;
    std::vector<ggml_tensor *> ssm_y;      // per block, null for attention
    ggml_tensor *              trunk_norm  = nullptr;
    ggml_tensor *              vad_hidden  = nullptr;
    ggml_tensor *              speech_in   = nullptr;
    ggml_tensor *              turn_in     = nullptr;
    ggml_tensor *              turn_emb    = nullptr;
    ggml_tensor *              turn_hidden = nullptr;
};

// Debug dump accumulators: every step of the current stream.
struct DumpTrace {
    std::vector<float> mel, log_energy, mixer, trunk_norm, vad_hidden, speech_in, turn_in, turn_emb, turn_hidden, logit,
        logit_la, turn;
    std::vector<std::vector<float>> trunk_mix, trunk_out, ssm_y;
    int64_t                         n_steps = 0;

    void clear();
};

struct Stream {
    // Persistent state (state_buf).
    ggml_context *         state_ctx   = nullptr;
    ggml_backend_buffer_t  state_buf   = nullptr;
    ggml_tensor *          mixer_tail  = nullptr;  // ne [row_width]: the previous frame's mixer row
    ggml_tensor *          prev_speech = nullptr;  // ne [1]: sigmoid of the previous on-time logit
    std::vector<SsmState>  ssm;                    // per trunk block (empty for attention)
    std::vector<AttnState> attn;                   // trunk attention blocks, then vad, then turn

    // Host state.
    std::vector<float> context;  // the last context_samples samples
    int64_t            pos = 0;  // steps taken (the RoPE position of the next frame)

    // Host scratch.
    std::vector<float>   mel, log_energy, mask, rope;
    std::vector<double>  fft_re, fft_im, fft_scratch, fe_win, fe_power;
    std::vector<float>   fe_buf;
    std::vector<int64_t> slots;
    std::vector<float>   logit, logit_la, turn;

    GraphTensors g;
    int          graph_n    = 0;  // columns of the built graph; 0 = none
    int          graph_real = 0;  // real (non-padding) columns

    DumpTrace trace;

    // Unique for the process lifetime: keys the batch graphs that read this
    // stream's state (BatchGraph::key), so a freed stream's graph is never
    // reused for another stream at the same address.
    const uint64_t uid;

    Stream();
    ~Stream();
    Stream(const Stream &)             = delete;
    Stream & operator=(const Stream &) = delete;
};

// Allocate (first call) and zero the stream state.
transcribe_status stream_reset(Stream & st, const Model & m);

// OK when the stream has its state; otherwise retry the failed reset's
// allocation (stream_step and the batched step call it first).
transcribe_status ensure_state(Stream & st, const Model & m);

// Run n_frames whole frames. Writes, per step, the emitted speech
// probability (lookahead; sigmoid(no_speech_logit) for the stream's first
// lookahead_frames steps) to speech[i] when non-null, and the horizon-0
// turn probability to turn[i] when non-null.
transcribe_status stream_step(SessionCore & core,
                              Stream &      st,
                              const Model & m,
                              const float * pcm,
                              int64_t       n_frames,
                              float *       turn,
                              float *       speech);

// Write the stream's trace (TRANSCRIBE_DUMP_DIR only), then drop it.
void stream_dump(Stream & st, const Model & m);

// The graph of one batched step (stream_step_batch): one frame for each of a
// list of at most 64 streams (turn_step_batch splits larger batches into
// chunks). Owned by the session listed first in its chunk, with its own
// compute context and scheduler (the session's single-stream graph keeps its
// own), and reused while the chunk's stream list is the same.
struct BatchGraph {
    ggml_context *        ctx   = nullptr;
    ggml_backend_sched_t  sched = nullptr;
    ggml_cgraph *         graph = nullptr;
    std::vector<uint64_t> key;    // Stream::uid of each column's stream
    int                   n = 0;  // columns (>= the stream count, padded)

    // Inputs.
    ggml_tensor *                           mel_in     = nullptr;  // ne [mel_width, n]
    ggml_tensor *                           log_energy = nullptr;  // ne [1, n]
    ggml_tensor *                           ssm_ids    = nullptr;  // I32 [1]
    ggml_tensor *                           rope_cos   = nullptr;  // ne [hd / 2, 1, n]
    ggml_tensor *                           rope_sin   = nullptr;
    std::vector<std::vector<ggml_tensor *>> slots;                 // [attention layer][stream]: I64 [1]
    std::vector<std::vector<ggml_tensor *>> masks;                 // [attention layer][stream]: ne [ring, 4]

    // Outputs.
    ggml_tensor * logit    = nullptr;  // ne [1, n]
    ggml_tensor * logit_la = nullptr;  // ne [1, n]
    ggml_tensor * turn     = nullptr;  // ne [n_horizons, n]

    // Host scratch.
    std::vector<float> mel, log_energy_h, rope, mask, logit_h, logit_la_h, turn_h;

    void release() noexcept;
    BatchGraph() = default;

    ~BatchGraph() { release(); }

    BatchGraph(const BatchGraph &)             = delete;
    BatchGraph & operator=(const BatchGraph &) = delete;
};

// One frame, pcm[i][0, frame_samples), for each of count distinct streams of
// m, in one compute with bg's graph and n_threads threads. Writes turn[i] /
// speech[i] as stream_step does for streams[i], with exactly its values: per
// column the same kernels in the same reduction order (docs/porting/
// families/sava.md). Adds the compute time to each core's t_encode_us.
transcribe_status stream_step_batch(BatchGraph &          bg,
                                    const Model &         m,
                                    Stream * const *      streams,
                                    SessionCore * const * cores,
                                    int                   count,
                                    int                   n_threads,
                                    const float * const * pcm,
                                    float *               turn,
                                    float *               speech);

struct TurnSession final : public transcribe_turn_session {
    Stream     st;
    BatchGraph batch;  // used when this session is listed first in a batched step's chunk

  protected:
    void on_scratch_released() noexcept override {
        st.g          = GraphTensors{};
        st.graph_n    = 0;
        st.graph_real = 0;
        batch.release();
    }
};

struct VadSession final : public transcribe_vad_session {
    Stream st;

  protected:
    void on_scratch_released() noexcept override {
        st.g          = GraphTensors{};
        st.graph_n    = 0;
        st.graph_real = 0;
    }
};

extern const Arch arch;

}  // namespace transcribe::sava
