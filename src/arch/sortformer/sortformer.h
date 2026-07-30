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
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe/sortformer.h"  // public preset enum + run ext
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
struct gguf_context;
typedef struct ggml_backend *        ggml_backend_t;
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
    int   chunk_len                   = 188;
    int   chunk_left_context          = 1;
    int   chunk_right_context         = 1;
    int   fifo_len                    = 0;
    int   spkcache_len                = 188;
    int   spkcache_update_period      = 188;
    int   spkcache_sil_frames_per_spk = 3;
    float sil_threshold               = 0.2f;
    float pred_score_threshold        = 0.25f;
    float scores_boost_latest         = 0.05f;
    float strong_boost_rate           = 0.75f;
    float weak_boost_rate             = 1.5f;
    float min_pos_scores_rate         = 0.5f;
    int   max_index                   = 99999;

    // External-cadence chunking (multitalker bundle only; 0 = off, the
    // standalone uniform-window path). When feat_first_chunk > 0 the core
    // mirrors NeMo's CacheAwareStreamingAudioBuffer windows instead of the
    // uniform streaming_feat_loader ones: the first chunk consumes
    // feat_first_chunk mel frames, each later chunk consumes
    // chunk_len*sub new frames with feat_cache frames of real left
    // context prepended, and drop_pre_encode pre-encode output frames are
    // discarded per later chunk (the duplicated / partially-contexted
    // outputs). chunk_left/right_context must be 0 in this mode.
    int feat_first_chunk = 0;
    int feat_cache       = 0;
    int drop_pre_encode  = 0;
};

// Host-side streaming state (AOSC speaker cache + FIFO), mirroring NeMo's
// StreamingSortformerState. Embeddings are 512-dim pre_encode outputs stored
// row-major [n_frames, enc_d_model]; preds are [n_frames, n_spk].
struct SortformerStreamState {
    std::vector<float> spkcache;                     // [spkcache_n * D]
    std::vector<float> spkcache_preds;               // [spkcache_n * S]
    int                spkcache_n          = 0;
    bool               spkcache_preds_init = false;  // NeMo: spkcache_preds is None until first compress
    std::vector<float> fifo;                         // [fifo_n * D]
    std::vector<float> fifo_preds;                   // [fifo_n * S]
    int                fifo_n = 0;
    std::vector<float> mean_sil_emb;                 // [D]
    int64_t            n_sil_frames = 0;
    std::vector<float> total_preds;                  // [T_total * S], accumulated chunk preds
    int                total_n        = 0;
    int                compress_count = 0;           // # _compress_spkcache calls (parity-dump index)

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
        total_n        = 0;
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
void streaming_update_sync(SortformerStreamState &        st,
                           const SortformerStreamParams & p,
                           int                            n_spk,
                           int                            emb_dim,
                           const std::vector<float> &     chunk_embs,
                           int                            T_diar,
                           const std::vector<float> &     preds,
                           int                            T_concat,
                           int                            lc,
                           int                            rc);

// Compress st.spkcache (spkcache_n > spkcache_len frames) in place down to
// spkcache_len frames, matching sortformer_modules._compress_spkcache.
void compress_spkcache(SortformerStreamState & st, const SortformerStreamParams & p, int n_spk, int emb_dim);

// Scratch for one streaming diarization forward, owned by whichever session
// drives it: the sortformer session (standalone family) or the parakeet
// session (multitalker bundle). compute_ctx is (re)created per chunk by
// run_diar_streaming_core and freed by the destructor.
struct DiarStreamScratch {
    ggml_context *     compute_ctx = nullptr;
    std::vector<float> pos_buf;
    std::vector<float> pos_div_term;

    // Streaming scratch (AOSC/FIFO path).
    SortformerStreamState stream;
    std::vector<float>    chunk_mel_buf;      // [n_mels * M] mel window for Graph A
    std::vector<float>    chunk_embs_host;    // [T_diar * enc_d_model] pre_encode readback
    std::vector<float>    concat_host;        // [T_concat * enc_d_model] Graph B input
    std::vector<float>    stream_preds_host;  // [T_concat * n_spk] Graph B readback

    DiarStreamScratch() = default;
    ~DiarStreamScratch();
};

// Concrete context. Owns a per-call compute context and a persistent
// multi-backend scheduler.
struct SortformerSession final : public transcribe_session {
    ggml_context *       compute_ctx = nullptr;  // offline-forward graph ctx
    ggml_backend_sched_t sched       = nullptr;

    // Per-context scratch reused across runs.
    std::vector<float> mel_buf;
    std::vector<float> probs_host;  // [n_spk * T], read back from diar.preds

    // Streaming scratch (AOSC/FIFO path + rel-pos tables, shared with the
    // offline forward's pos_emb fill).
    DiarStreamScratch scratch;

    SortformerSession() = default;
    ~SortformerSession() override;
};

// ---- Embedded-diarizer surface (multitalker bundle) -------------------- //
//
// A multitalker Parakeet bundle GGUF carries this diarizer's tensors under
// a name prefix (stt.parakeet.diarizer.tensor_prefix, "sortformer.") next
// to the ASR's own tensors, and its hparams under the usual
// stt.sortformer.* KVs. The host family owns the storage (its ctx_meta /
// backend buffers); this surface only resolves borrowed tensor pointers
// and runs the forward on the host's scheduler.

// Weight + hparam core of a diarizer embedded in another family's GGUF.
struct SortformerEmbedded {
    SortformerHParams                     hp;
    transcribe::parakeet::ParakeetHParams conformer_hp;
    transcribe::parakeet::ParakeetWeights conformer;  // pre_encode + blocks only
    SortformerWeights                     weights;
};

// Read stt.sortformer.* hparams from `gguf` and resolve every diarizer
// tensor (conformer + sortformer-specific) in `ctx_meta` under
// `tensor_prefix`. Defined in model.cpp.
transcribe_status init_embedded_diarizer(const gguf_context * gguf,
                                         ggml_context *       ctx_meta,
                                         const char *         tensor_prefix,
                                         SortformerEmbedded & out);

// Fuse the embedded conformer's BatchNorm into scale/bias tensors allocated
// on `backend` (must run AFTER tensor data is uploaded). The caller owns and
// frees *out_ctx / *out_buffer. Defined in model.cpp.
transcribe_status fuse_embedded_diar_bn(SortformerEmbedded &    e,
                                        ggml_backend_t          backend,
                                        ggml_context **         out_ctx,
                                        ggml_backend_buffer_t * out_buffer);

// Full streaming AOSC/FIFO diarization forward over a precomputed mel
// buffer (row-major [n_mels, n_frames]), on the caller's scheduler. On
// success sc.stream.total_preds holds the accumulated row-major
// [total_n, n_spk] probs (trim to ceil(n_frames / subsampling) frames;
// see run_streaming in model.cpp). abort_session is polled per chunk when
// non-null. Defined in model.cpp.
transcribe_status run_diar_streaming_core(DiarStreamScratch &                           sc,
                                          const SortformerHParams &                     hp,
                                          const transcribe::parakeet::ParakeetHParams & chp,
                                          const transcribe::parakeet::ParakeetWeights & conformer,
                                          const SortformerWeights &                     w,
                                          const char *                                  backend,
                                          ggml_backend_sched_t                          sched,
                                          int                                           n_threads,
                                          const SortformerStreamParams &                P,
                                          const float *                                 mel_buf,
                                          int                                           mel_n_mels,
                                          int                                           mel_n_frames,
                                          transcribe_session *                          abort_session);

// Threshold-based probs -> speaker_segment rows (probs row-major [T, n_spk],
// speaker_id 1-based, p = NaN). Shared with the multitalker bundle path.
void probs_to_speaker_segments(transcribe_session *       session,
                               const std::vector<float> & probs,
                               int                        T,
                               int                        n_spk,
                               double                     ms_per_frame,
                               float                      threshold);

}  // namespace transcribe::sortformer
