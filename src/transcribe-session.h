// transcribe-session.h - internal base class for the public opaque
// transcribe_session handle.
//
// Mirror of transcribe-model.h. Same Option B layout: a small base owned
// by the central dispatch, derived per-family contexts owning everything
// else. The base has a virtual destructor so transcribe_session_free()
// can `delete ctx;` polymorphically.
//
// The result storage is on the BASE because it's family-agnostic: the
// public C ABI surfaces the same hierarchy (segments / words / tokens
// / full text) regardless of which family produced it. Per-family
// run() drivers populate these vectors during decode; the public
// accessors in transcribe.cpp read them directly with no per-family
// dispatch.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct transcribe_model;

struct transcribe_session {
    // The model this session was constructed from. Borrowed pointer:
    // the caller is required (per the public threading contract) to keep
    // the model alive for the lifetime of every derived session.
    transcribe_model * model = nullptr;

    // True only for sessions created via transcribe_open(), which loads
    // and therefore owns its model. Both transcribe_session_free() and
    // transcribe_close() (now an alias) read this flag and free the
    // owned model after destroying the session. Sessions created via
    // transcribe_session_init() leave this false and their model is
    // freed independently by the caller via transcribe_model_free().
    bool owns_model = false;

    // Cached n_threads value the caller passed at init time. 0 means
    // "library picks a sensible default" (matches the factory).
    int n_threads = 0;

    // Per-call timings, populated by the most recent transcribe_run.
    // Surfaced via the public transcribe_get_timings accessor; reset
    // by transcribe_reset_timings.
    int64_t t_mel_us    = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;

    // -----------------------------------------------------------------
    // Result storage (family-agnostic; populated by per-family run()).
    //
    // The shape mirrors PLAN.md "Results": a flat backing array per
    // level (segments / words / tokens) plus forward and backward
    // pointers between levels. v1 produces a single segment per run;
    // word splitting is family-specific but the per-token / per-word
    // / per-segment fields below are the same for every family.
    //
    // result_kind tells the public accessor what timestamp granularity
    // the family populated. v1 Parakeet TDT produces token-level
    // timestamps from the encoder frame indices; word/segment t0/t1
    // are computed from the contained tokens.
    //
    // has_result is the "transcribe_run produced a real result" flag.
    // When false (no run yet, or the most recent run failed before
    // populating), every accessor returns its safe sentinel ("",
    // 0, NAN).

    struct TokenEntry {
        int         id            = 0;
        std::string text;            // decoded fragment (▁ → space)
        float       p             = 0.0f;
        int64_t     t0_ms         = 0;
        int64_t     t1_ms         = 0;
        int         seg_index     = 0;
        int         word_index    = -1;
    };

    struct WordEntry {
        std::string text;
        int64_t     t0_ms       = 0;
        int64_t     t1_ms       = 0;
        int         seg_index   = 0;
        int         first_token = 0;
        int         n_tokens    = 0;
    };

    struct SegmentEntry {
        std::string text;
        int64_t     t0_ms       = 0;
        int64_t     t1_ms       = 0;
        int         first_word  = 0;
        int         n_words     = 0;
        int         first_token = 0;
        int         n_tokens    = 0;
    };

    std::vector<TokenEntry>      tokens;
    std::vector<WordEntry>       words;
    std::vector<SegmentEntry>    segments;
    std::string                  full_text;
    // ISO short code the model itself predicted on this run, populated
    // only when the caller did NOT pass a language hint (auto/null) and
    // the family actually ran a detection step. Empty string means
    // "unknown" — either no detection ran (English-only model, user
    // supplied a hint, family doesn't support LID) or detection produced
    // a non-language sentinel (e.g. SenseVoice's <|nospeech|>).
    std::string                  detected_language;
    transcribe_timestamp_kind    result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    bool                         has_result  = false;

    // -----------------------------------------------------------------
    // Abort / cancellation (set via transcribe_set_abort_callback).
    //
    // Per-family run() drivers call poll_abort() at chunk boundaries
    // and between greedy decode steps. A callback returning true sets
    // was_aborted and the run path returns TRANSCRIBE_ERR_ABORTED,
    // preserving whatever partial segments accumulated before the
    // abort point.
    //
    // was_aborted is cleared at the top of every transcribe_run; it
    // is NOT cleared by clear_result because the result may be
    // explicitly preserved after an abort (partial-retained contract).
    transcribe_abort_callback    abort_cb        = nullptr;
    void *                       abort_userdata  = nullptr;
    bool                         was_aborted     = false;

    bool poll_abort() {
        if (abort_cb != nullptr && abort_cb(abort_userdata)) {
            was_aborted = true;
            return true;
        }
        return false;
    }

    // -----------------------------------------------------------------
    // Streaming state.
    //
    // Lifecycle (stream_state) is separated from result snapshot so
    // that clear_result() — which is called by both transcribe_run and
    // transcribe_stream_begin — can wipe per-call data without churning
    // the IDLE/ACTIVE/FINISHED/FAILED state machine. The dispatcher
    // manages stream_state explicitly at begin/feed/finalize/reset.
    //
    // The snapshot fields below (revision, committed counts,
    // last_status, audio cursors) ARE cleared by clear_result so that
    // a fresh transcribe_run starts with zeroed streaming bookkeeping
    // and a streaming caller does not see stale counters across runs.
    //
    // Audio cursors are us-precision; the public stream_update struct
    // exposes them as ms.
    transcribe_stream_state      stream_state          = TRANSCRIBE_STREAM_IDLE;
    int32_t                      stream_revision      = 0;
    int                          n_committed_segments = 0;
    int                          n_committed_words    = 0;
    int                          n_committed_tokens   = 0;
    transcribe_status            stream_last_status   = TRANSCRIBE_OK;
    int64_t                      stream_audio_input_us     = 0;
    int64_t                      stream_audio_committed_us = 0;

    void clear_result();

    transcribe_session() = default;
    virtual ~transcribe_session();

    transcribe_session(const transcribe_session &)             = delete;
    transcribe_session & operator=(const transcribe_session &) = delete;
    transcribe_session(transcribe_session &&)                  = delete;
    transcribe_session & operator=(transcribe_session &&)      = delete;
};
