// arch/moonshine_streaming/moonshine_streaming.h - Moonshine-Streaming
// model and context types.
//
// This header is INTERNAL to src/arch/moonshine_streaming/. It mirrors
// src/arch/moonshine/moonshine.h since the encoder-decoder + cross-attn
// KV cache shape is shared across both. Differences:
//
//   - `enc_host_for_adapter` holds the post-adapter encoder hidden
//     (instead of the raw encoder output) so the adapter add+proj is
//     applied exactly once per session, mirroring the reference dumper's
//     `apply_adapter()` helper. Cross-KV precompute reads from this
//     buffer, not the encoder output directly.
//   - Phase 4a exposes the streaming ABI as stream-of-whole:
//     feed buffers PCM and finalize runs the one-shot inference path.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-session.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::moonshine_streaming {

void apply_family_invariants(transcribe_model & model);

// ---------------------------------------------------------------------------
// KV cache for the autoregressive decoder.  Same shape as moonshine /
// cohere / whisper — dual cache: self over [d_model, n_ctx], cross over
// [d_model, T_enc].
// ---------------------------------------------------------------------------

struct MoonshineStreamingKvCache {
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int n_ctx = 0;
    int n     = 0;
    int head  = 0;
    int T_enc = 0;

    bool cross_populated = false;

    void free() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        self_k = nullptr;
        self_v = nullptr;
        cross_k = nullptr;
        cross_v = nullptr;
        n = 0;
        head = 0;
        T_enc = 0;
        cross_populated = false;
    }
};

bool kv_cache_init(MoonshineStreamingKvCache & cache,
                   ggml_backend_t              backend,
                   int                         n_ctx,
                   int                         T_enc,
                   int                         d_model,
                   int                         n_layer,
                   ggml_type                   kv_type);

// ---------------------------------------------------------------------------
// Model / context
// ---------------------------------------------------------------------------

struct MoonshineStreamingModel final : public transcribe_model {
    Tokenizer                   tok;
    MoonshineStreamingHParams   hparams;
    MoonshineStreamingWeights   weights;
    ggml_context *              ctx_meta = nullptr;

    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    MoonshineStreamingModel() = default;
    ~MoonshineStreamingModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct MoonshineStreamingSession final : public transcribe_session {
    ggml_context *        compute_ctx = nullptr;
    ggml_backend_sched_t  sched       = nullptr;

    // Host-side mirror of the post-adapter encoder hidden. The adapter
    // pos_emb add (and proj when present) is applied once per session;
    // this host buffer feeds the cross_kv precompute graph.
    std::vector<float> adapter_host;
    int                enc_T = 0;   // T_enc

    MoonshineStreamingKvCache kv_cache;

    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    // ---- incremental streaming state (Phase 4b-encoder + adapter/xkv) ----
    //
    // Each feed extends three host-side committed buffers in lockstep:
    //
    //   stream_adapter_committed  - post-adapter encoder hidden, sized
    //                               [dec_d_model, T_emitted]. Built per
    //                               feed by re-running the encoder over
    //                               a sliding window and immediately
    //                               applying the adapter (pos_emb +
    //                               optional proj) at absolute frame
    //                               positions.
    //
    //   stream_cross_k_committed  - one host vector per decoder layer,
    //   stream_cross_v_committed    each sized [dec_d_model, T_emitted].
    //                               Populated by the cross-KV projection
    //                               graph (k_proj / v_proj) on the
    //                               adapter slice. NOT written to the
    //                               persistent kv_cache yet — that's a
    //                               single bulk upload at finalize.
    //
    // Trimming: stream_pcm_buffer is trimmed each feed to drop PCM
    // older than (T_emitted - L_total - frontend_pad) encoder frames.
    // The 1.8 s of left-context history that the encoder needs to
    // produce numerically-identical frames is preserved; everything
    // before is no longer reachable. stream_pcm_start_sample tracks
    // the absolute sample index of stream_pcm_buffer[0] so the
    // streaming driver can translate absolute frame indices back to
    // buffer-relative offsets.
    //
    // The design relies on the encoder being ergodic (no positional
    // encoding on encoder self-attn) plus per-layer sliding-window
    // attention plus causal-conv frontend, so output frame t is a
    // function only of conv-stack output frames in
    // [t - L_total, t + R_total]; the adapter is a positional
    // embedding add indexed by absolute frame so it's free to apply
    // per-feed; and the cross-KV projection is per-frame linear so
    // accumulating its output across feeds and uploading once at
    // finalize is numerically equivalent to a single batched call.
    //
    // Lifecycle:
    //   stream_begin:    derive geometry, clear scratch + counters.
    //   stream_feed:     append PCM, slide-window encode, apply adapter,
    //                    project cross K/V, append to host buffers,
    //                    trim PCM.
    //   stream_finalize: encode tail (no R_total margin needed),
    //                    apply adapter + project K/V on the tail,
    //                    allocate kv_cache, upload host K/V buffers,
    //                    run AR decoder only.
    //   stream_reset:    clear scratch (keep capacity).
    //
    // These fields are NOT touched by clear_result; the dispatcher
    // wipes lifecycle-agnostic snapshot state there, and the family
    // owns its per-utterance audio + encoder scratch.
    std::vector<float>                stream_pcm_buffer;
    int64_t                           stream_pcm_start_sample = 0;
    std::vector<float>                stream_adapter_committed;
    std::vector<std::vector<float>>   stream_cross_k_committed;
    std::vector<std::vector<float>>   stream_cross_v_committed;
    int32_t                           stream_T_emitted             = 0;
    // T_emitted captured at the last successful partial decode; lets
    // stream_feed / stream_finalize skip a redundant decode when no
    // new encoder frames have been committed since the previous one.
    int32_t                           stream_last_decoded_T        = 0;
    // Recent partial-decode token id sequences, used to compute the
    // longest common prefix that's safe to mark committed across feeds.
    // Reset at stream_begin; updated after every successful partial decode.
    std::deque<std::vector<int32_t>>  stream_token_id_history;
    // Geometry, resolved at stream_begin (constant per stream).
    int32_t                           stream_L_total_frames        = 0;
    int32_t                           stream_R_total_frames        = 0;
    int32_t                           stream_frontend_pad_frames   = 0;
    int32_t                           stream_samples_per_enc_frame = 0;
    // Minimum gap, in encoder frames, between successive per-feed AR
    // decoder runs. Resolved at stream_begin from the Moonshine-
    // Streaming family extension.
    // 0 means "decode on every encoder-frame advance". The finalize
    // path always runs one last decode regardless of this throttle.
    int32_t                           stream_min_decode_frames     = 0;
    transcribe_run_params                 stream_run_params {};

    MoonshineStreamingSession() = default;
    ~MoonshineStreamingSession() override;
};

} // namespace transcribe::moonshine_streaming
