// arch/sortformer/sortformer.h - Sortformer-family internal model and
// context types. Sortformer is an `encoder-diarizer`: a NEST/FastConformer
// encoder (the same NeMo ConformerEncoder Parakeet ports, reused verbatim
// via parakeet::build_encoder_graph) followed by a Linear projection, an
// 18-layer post-LN Transformer encoder, and a 4-way sigmoid speaker head.
// No tokenizer, no decoder, no text: the product is a T x 4 per-frame
// speaker-activity probability matrix.
//
// The central dispatcher talks to the family only through the Arch trait
// and the base classes.

#pragma once

#include "../parakeet/encoder.h"  // build_encoder_graph
#include "../parakeet/weights.h"  // ParakeetHParams / ParakeetWeights (conformer)
#include "transcribe/sortformer.h"  // public preset enum + run ext
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::sortformer {

// Family defaults, applied before transcribe::read_capability_kv (KV
// present overrides, KV absent keeps the default). Defined in
// capabilities.cpp.
void apply_family_invariants(transcribe_model & model);

// Concrete model. Owns ctx_meta (every weight tensor's data buffer);
// the destructor frees it, invalidating every borrowed ggml_tensor* in
// `conformer` / `weights`.
struct SortformerModel final : public transcribe_model {
    SortformerHParams hparams;

    // Conformer encoder reuse. `conformer_hp` carries only the encoder +
    // frontend fields parakeet::build_encoder_graph reads; `conformer`
    // holds only pre_encode + blocks (predictor/joint/etc. stay empty).
    transcribe::parakeet::ParakeetHParams conformer_hp;
    transcribe::parakeet::ParakeetWeights conformer;

    // Sortformer-specific weights (encoder_proj + transformer + diar head).
    SortformerWeights weights;

    ggml_context *          ctx_meta = nullptr;
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Fused conformer BN params (computed at load from the raw BN tensors).
    ggml_context *        bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t bn_fused_buffer = nullptr;

    // Mel front-end, constructed once at load() from hparams.
    std::optional<transcribe::MelFrontend> mel;

    SortformerModel() = default;
    ~SortformerModel() override;

    // No tokenizer for a diarizer.
    const transcribe::Tokenizer * tokenizer() const override { return nullptr; }
};

// Streaming operating point (resolved from GGUF stream defaults; a later
// change will expose the presets via a stream extension). All lengths are
// in 80 ms diar frames.
struct SortformerStreamParams {
    int chunk_len              = 188;
    int chunk_left_context     = 1;
    int chunk_right_context    = 1;
    int fifo_len               = 0;
    int spkcache_len           = 188;
    int spkcache_update_period = 188;
    int spkcache_sil_frames_per_spk = 3;
    float sil_threshold        = 0.2f;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest  = 0.05f;
    float strong_boost_rate    = 0.75f;
    float weak_boost_rate      = 1.5f;
    float min_pos_scores_rate  = 0.5f;
    int   max_index            = 99999;
};

// Host-side streaming state (AOSC speaker cache + FIFO), mirroring NeMo's
// StreamingSortformerState. Embeddings are 512-dim pre_encode outputs stored
// row-major [n_frames, enc_d_model]; preds are [n_frames, n_spk].
struct SortformerStreamState {
    std::vector<float> spkcache;        // [spkcache_n * D]
    std::vector<float> spkcache_preds;  // [spkcache_n * S]
    int                spkcache_n = 0;
    bool               spkcache_preds_init = false;  // NeMo: spkcache_preds is None until first compress
    std::vector<float> fifo;            // [fifo_n * D]
    std::vector<float> fifo_preds;      // [fifo_n * S]
    int                fifo_n = 0;
    std::vector<float> mean_sil_emb;    // [D]
    int64_t            n_sil_frames = 0;
    std::vector<float> total_preds;     // [T_total * S], accumulated chunk preds
    int                total_n = 0;
    int                compress_count = 0;  // # _compress_spkcache calls (parity-dump index)

    void reset(int emb_dim) {
        spkcache.clear();
        spkcache_preds.clear();
        spkcache_n          = 0;
        spkcache_preds_init = false;
        fifo.clear();
        fifo_preds.clear();
        fifo_n = 0;
        mean_sil_emb.assign(static_cast<size_t>(emb_dim), 0.0f);
        n_sil_frames = 0;
        total_preds.clear();
        total_n = 0;
        compress_count = 0;
    }
};

// Resolve the streaming operating point, lowest to highest precedence:
// GGUF-shipped stream cfg < public run-ext preset (transcribe_sortformer_
// stream_ext; DEFAULT keeps the GGUF cfg) < TRANSCRIBE_SORTFORMER_STREAM_
// PRESET env (very_high_latency / high_latency / low_latency / small;
// validation hook) < per-field env overrides. Defined in stream.cpp.
SortformerStreamParams resolve_stream_params(const SortformerHParams & hp, transcribe_sortformer_preset preset);

// Exact host ports of the NeMo sync-streaming primitives (batch=1). All
// embeddings row-major [n_frames, emb_dim]; preds row-major [n_frames, n_spk].
// Defined in stream.cpp.
void streaming_update_sync(SortformerStreamState & st, const SortformerStreamParams & p, int n_spk, int emb_dim,
                           const std::vector<float> & chunk_embs, int T_diar,
                           const std::vector<float> & preds, int T_concat, int lc, int rc);

// Compress st.spkcache (spkcache_n > spkcache_len frames) in place down to
// spkcache_len frames, matching sortformer_modules._compress_spkcache.
void compress_spkcache(SortformerStreamState & st, const SortformerStreamParams & p, int n_spk, int emb_dim);

// Concrete context. Owns a per-call compute context and a persistent
// multi-backend scheduler.
struct SortformerSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    // Per-context scratch reused across runs.
    std::vector<float> mel_buf;
    std::vector<float> pos_buf;
    std::vector<float> pos_div_term;
    std::vector<float> probs_host;  // [n_spk * T], read back from diar.preds

    // Streaming scratch (AOSC/FIFO path).
    SortformerStreamState stream;
    std::vector<float>    chunk_mel_buf;    // [n_mels * M] mel window for Graph A
    std::vector<float>    chunk_embs_host;  // [T_diar * enc_d_model] pre_encode readback
    std::vector<float>    concat_host;      // [T_concat * enc_d_model] Graph B input
    std::vector<float>    stream_preds_host;  // [T_concat * n_spk] Graph B readback

    SortformerSession() = default;
    ~SortformerSession() override;
};

}  // namespace transcribe::sortformer
