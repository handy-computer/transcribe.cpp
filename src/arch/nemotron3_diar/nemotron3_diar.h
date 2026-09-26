// arch/nemotron3_diar/nemotron3_diar.h - Nemotron-3-Diarization internal
// model / session types. An `encoder-diarizer` in the Streaming Sortformer
// lineage: feature stacking x8 -> 31-layer pre-LN RoPE Transformer over the
// [speaker cache | FIFO | chunk] concat -> encoder_proj -> x8 subpixel
// upsampler -> sigmoid head, producing T x 8 speaker activity at 10 ms. No
// tokenizer, no decoder, no text.
//
// The product path is always NeMo's synchronous streaming loop (AOSC speaker
// cache + FIFO at 80 ms, driven by the x8 average-pooled probabilities);
// transcribe_run, push-audio streaming and run_batch all drive the same
// per-chunk step (model.cpp) and host state machine (stream.cpp).

#pragma once

#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe/nemotron3_diar.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace transcribe::nemotron3_diar {

// Family defaults, applied before transcribe::read_capability_kv.
void apply_family_invariants(transcribe_model & model);

struct Nemotron3DiarModel final : public transcribe_model {
    Nemotron3DiarHParams hparams;
    Nemotron3DiarWeights weights;

    ggml_context *          ctx_meta = nullptr;
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // F32 copies of the BF16 / F16 matmul weights (CPU default; see load()),
    // so every matmul runs with F32 activations like the fp32-compute
    // reference. The original tensors stay resident in backend_buffer.
    ggml_context *        f32_ctx    = nullptr;
    ggml_backend_buffer_t f32_buffer = nullptr;

    // Host copy of the learned AOSC silence embedding (compress runs on host).
    std::vector<float> sil_emb_host;

    std::optional<transcribe::MelFrontend> mel;

    Nemotron3DiarModel() = default;
    ~Nemotron3DiarModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return nullptr; }
};

// Streaming operating point. Lengths in 80 ms encoder frames.
struct StreamParams {
    int   chunk_len                   = 340;
    int   chunk_left_context          = 0;
    int   chunk_right_context         = 40;
    int   fifo_len                    = 40;
    int   spkcache_len                = 264;
    int   spkcache_update_period      = 300;
    int   spkcache_sil_frames_per_spk = 1;
    float pred_score_threshold        = 0.25f;
    float scores_boost_latest         = 0.05f;
    float strong_boost_rate           = 0.75f;
    float weak_boost_rate             = 1.5f;
    float min_pos_scores_rate         = 0.5f;
    int   max_index                   = 99999;
};

// Resolve the operating point, lowest to highest precedence: GGUF-shipped
// default < public preset (DEFAULT keeps the GGUF cfg) <
// TRANSCRIBE_NEMOTRON3_DIAR_PRESET env (card names + "small"; validation
// hook). Defined in stream.cpp.
StreamParams resolve_stream_params(const Nemotron3DiarHParams & hp, transcribe_nemotron3_diar_preset preset);

// True when `preset` is a valid enum value.
bool preset_is_valid(transcribe_nemotron3_diar_preset preset);

// Host-side AOSC speaker cache + FIFO (NeMo StreamingSortformerState, sync,
// batch 1). Embeddings row-major [n, emb_dim] (512-dim pre_encode outputs);
// preds row-major [n, n_spk] at 80 ms.
struct StreamState {
    std::vector<float> spkcache;
    std::vector<float> spkcache_preds;
    int                spkcache_n = 0;
    bool               compressed = false;
    std::vector<float> fifo;
    std::vector<float> fifo_preds;
    int                fifo_n         = 0;
    int                compress_count = 0;

    // Accumulated chunk outputs at 10 ms, row-major [total_n, n_spk].
    std::vector<float> total_preds;
    int                total_n = 0;

    void reset() {
        spkcache.clear();
        spkcache_preds.clear();
        spkcache_n = 0;
        compressed = false;
        fifo.clear();
        fifo_preds.clear();
        fifo_n         = 0;
        compress_count = 0;
        total_preds.clear();
        total_n = 0;
    }
};

// NeMo streaming_update (sync branch, use_learnable_sil_emb=True). preds80 is
// the x8 average-pooled prediction over the whole [spkcache|fifo|chunk]
// concat ([T_concat, n_spk]); chunk_embs is the chunk's pre_encode output
// ([T_diar, emb_dim]). Does NOT touch total_preds (the caller appends the
// 10 ms chunk slice). Defined in stream.cpp.
void streaming_update(StreamState &              st,
                      const StreamParams &       p,
                      int                        n_spk,
                      int                        emb_dim,
                      const std::vector<float> & chunk_embs,
                      int                        T_diar,
                      const std::vector<float> & preds80,
                      int                        lc,
                      int                        rc,
                      const std::vector<float> & sil_emb);

// Incremental speaker-segment builder over 10 ms probs (threshold 0.5).
// Closed runs are final; open runs are reported up to the last frame seen.
struct SegmentTracker {
    std::vector<int>                                     run_start;  // per speaker, -1 = inactive
    std::vector<transcribe_session::SpeakerSegmentEntry> closed;
    int                                                  n_seen = 0;

    void reset(int n_spk) {
        run_start.assign(static_cast<size_t>(n_spk), -1);
        closed.clear();
        n_seen = 0;
    }
};

// Consume frames [tr.n_seen, T) of probs (row-major [*, n_spk]).
void segments_advance(SegmentTracker & tr, const float * probs, int T, int n_spk, double ms_per_frame);
// Close every open run at frame tr.n_seen.
void segments_close_all(SegmentTracker & tr, double ms_per_frame);
// Rebuild session->speaker_segments = closed + open-to-now, ordered by
// (speaker, t0) like the offline path.
void segments_publish(const SegmentTracker & tr, transcribe_session * session, double ms_per_frame);

// Per-utterance streaming run: operating point, cache state, pending mel,
// output tracking. Used by transcribe_run (whole mel at once), push-audio
// (mel appended incrementally) and run_batch.
struct Utterance {
    StreamParams   P;
    StreamState    st;
    SegmentTracker seg;

    // Mel frames not yet consumed, time-major [n, n_mels], starting at
    // global mel frame `mel_base`.
    std::vector<float> mel_tm;
    int64_t            mel_base   = 0;
    int64_t            mel_frames = 0;  // total final mel frames known so far
    int64_t            stt        = 0;  // next chunk start (global mel frame)
    int                chunk_idx  = 0;
    bool               done       = false;
};

struct Nemotron3DiarSession final : public transcribe_session {
    std::vector<float>   mel_buf;      // row-major [n_mels, n_frames] (run path)
    std::vector<float>   step_prev;    // [S+F, emb_dim] graph input
    std::vector<float>   step_mel;     // [T_diar, sub*n_mels] graph input
    std::vector<float>   step_embs;    // [T_diar, emb_dim] readback
    std::vector<float>   step_hi;      // [8*T_concat, n_spk] readback
    std::vector<float>   step_pooled;  // [T_concat, n_spk]
    std::vector<int32_t> positions;

    // Push-audio state.
    Utterance          utt;
    std::vector<float> pcm_tail;  // unconsumed PCM (from pcm_tail_base)
    int64_t            pcm_tail_base = 0;
    int64_t            pcm_received  = 0;
    std::vector<float> seg_mel;  // scratch
    bool               stream_dump = false;

    Nemotron3DiarSession() = default;
    ~Nemotron3DiarSession() override;
};

}  // namespace transcribe::nemotron3_diar
