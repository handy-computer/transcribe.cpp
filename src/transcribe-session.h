// transcribe-session.h - internal base class for the public opaque
// transcribe_session handle.
//
// Mirrors transcribe-model.h: a small base owned by the central dispatch,
// derived per-family contexts owning everything else, with a virtual
// destructor so transcribe_session_free() can delete polymorphically.
// Result storage lives on the base because it's family-agnostic (segments
// / words / tokens / full text); per-family run() drivers populate the
// vectors and the public accessors read them directly.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// Read transcribe_session_params::n_ctx with a struct_size guard. n_ctx is
// a trailing field appended after kv_type, so an older caller's smaller
// struct may not include it; in that case (or a NULL params) the default 0
// = "use the model's true max" is returned. Per-family init_context() that
// honors n_ctx should cache the result onto the base session's n_ctx field.
// The negative-value rejection happens once, generically, in
// transcribe_session_init().
inline int32_t transcribe_session_params_n_ctx(const struct transcribe_session_params * params) {
    if (params == nullptr) {
        return 0;
    }
    const size_t field_end = offsetof(struct transcribe_session_params, n_ctx) + sizeof(params->n_ctx);
    if (params->struct_size < static_cast<uint64_t>(field_end)) {
        return 0;
    }
    return params->n_ctx;
}

// Common members (model, n_threads, timings, abort callback, compute
// scratch) live on transcribe::SessionCore; see transcribe-session-core.h.
struct transcribe_session : transcribe::SessionCore {
    // True only for sessions created via transcribe_open(), which loads
    // and therefore owns its model. Both transcribe_session_free() and
    // transcribe_close() (now an alias) read this flag and free the
    // owned model after destroying the session. Sessions created via
    // transcribe_session_init() leave this false and their model is
    // freed independently by the caller via transcribe_model_free().
    bool owns_model = false;

    // Cached n_ctx value the caller passed at init time (decoder context
    // cap in tokens). 0 means "use the model's true maximum from GGUF".
    // A positive value lowers the ceiling to bound KV-cache memory and is
    // clamped down to the model maximum by the family that honors it.
    // Only hard-context-cap families read this; chunked / unbounded
    // families ignore it. See include/transcribe.h
    // (transcribe_session_params::n_ctx).
    int32_t n_ctx = 0;

    // Cached kv_type the caller passed at init time. Hoisted to the base so
    // generic code (e.g. transcribe_session_get_limits's KV-byte estimate)
    // can read it without a per-family hook. Every family's init_context sets
    // this from params; the families resolve AUTO to f16 for the KV cache, so
    // for byte accounting only F32 differs (4 bytes/elem vs 2).
    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    // Result storage (family-agnostic; populated by per-family run()).
    // A flat backing array per level (segments / words / tokens) with
    // forward/backward indices between levels. result_kind is the
    // timestamp granularity the family populated; has_result is false
    // until a run populates, in which case accessors return safe
    // sentinels ("", 0, NAN).

    struct TokenEntry {
        int         id = 0;
        std::string text;  // decoded fragment (▁ → space)
        float       p          = 0.0f;
        int64_t     t0_ms      = 0;
        int64_t     t1_ms      = 0;
        int         seg_index  = 0;
        int         word_index = -1;
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
        int32_t     speaker_id  = 0;  // 1-based; 0 = no attribution
    };

    // Hoisted to transcribe::SpeakerSegmentEntry so every session type
    // that reports speaker segments shares one row type.
    using SpeakerSegmentEntry = transcribe::SpeakerSegmentEntry;

    std::vector<TokenEntry>          tokens;
    std::vector<WordEntry>           words;
    std::vector<SegmentEntry>        segments;
    std::vector<SpeakerSegmentEntry> speaker_segments;
    std::string                      full_text;
    // The model's decoded output before family post-processing (marker
    // parsing, tag filtering, envelope stripping, whitespace trims).
    // Equals full_text modulo whitespace for families that do none.
    std::string                      raw_text;

    // Offline batch results (transcribe_run_batch). The scratch fields
    // above are the single "current result" slot every run() writes into
    // and the single-shot accessors read; batch_results is a separate
    // vector so the serial-fallback path can snapshot each utterance
    // without per-family changes.
    //
    // Source-of-truth rule for the public accessors:
    //   - batch_results non-empty  -> the batch accessors index it, and
    //     the scratch slot mirrors batch_results[0] so the single
    //     accessors stay coherent (they show utterance 0).
    //   - batch_results empty      -> single-shot mode; the batch
    //     accessors synthesize index 0 from the scratch slot.
    // Cleared at the top of every transcribe_run / transcribe_run_batch,
    // NOT by clear_result (which only wipes the scratch slot + streaming
    // snapshot, so a run() inside the fallback loop does not erase
    // already-accumulated entries).
    struct ResultSet {
        std::vector<TokenEntry>          tokens;
        std::vector<WordEntry>           words;
        std::vector<SegmentEntry>        segments;
        std::vector<SpeakerSegmentEntry> speaker_segments;
        std::string                      full_text;
        std::string                      raw_text;
        std::string                      detected_language;
        transcribe_timestamp_kind        result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        bool                             has_result  = false;
        transcribe_status                status      = TRANSCRIBE_OK;
        // Per-utterance timings (us). For a batched run the encoder is one
        // shared dispatch, so a family amortizes its total encode time across
        // the batch (sum over utterances == the real batch encode time);
        // decode is genuinely per-utterance. Lets transcribe_batch_get_timings
        // expose where time goes (encoder vs host decode) per utterance.
        int64_t                          t_mel_us    = 0;
        int64_t                          t_encode_us = 0;
        int64_t                          t_decode_us = 0;
    };

    std::vector<ResultSet> batch_results;

    // Snapshot the current scratch result slot (the fields above that a
    // per-family run() populates) into a standalone ResultSet. Used by the
    // batch dispatcher's serial fallback and by family run_batch() hooks to
    // capture each utterance's result. `st` records the per-utterance
    // terminal status.
    ResultSet capture_result(transcribe_status st = TRANSCRIBE_OK) const {
        ResultSet rs;
        rs.tokens            = tokens;
        rs.words             = words;
        rs.segments          = segments;
        rs.speaker_segments  = speaker_segments;
        rs.full_text         = full_text;
        rs.raw_text          = raw_text;
        rs.detected_language = detected_language;
        rs.result_kind       = result_kind;
        rs.has_result        = has_result;
        rs.status            = st;
        rs.t_mel_us          = t_mel_us;
        rs.t_encode_us       = t_encode_us;
        rs.t_decode_us       = t_decode_us;
        return rs;
    }

    // ISO short code the model itself predicted on this run, populated
    // only when the caller did NOT pass a language hint (auto/null) and
    // the family actually ran a detection step. Empty string means
    // "unknown" — either no detection ran (English-only model, user
    // supplied a hint, family doesn't support LID) or detection produced
    // a non-language sentinel (e.g. SenseVoice's <|nospeech|>).
    std::string               detected_language;
    transcribe_timestamp_kind result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    bool                      has_result  = false;

    // Set by a run() driver when decode stops at the model's context /
    // position cap before end-of-stream (output truncated; partial result
    // retained). Surfaced via transcribe_was_truncated(); cleared at the
    // top of every transcribe_run, NOT by clear_result. Distinct from the
    // up-front TRANSCRIBE_ERR_INPUT_TOO_LONG rejection (couldn't finish vs
    // couldn't start). See docs/input-limits.md.
    bool was_truncated = false;

    // Set with was_truncated when the repetition guard stopped the decode
    // (transcribe-repetition-guard.h) rather than the budget, so the run
    // reports TRANSCRIBE_ERR_OUTPUT_REPETITION. Cleared with was_truncated.
    bool stopped_on_repetition = false;

    void mark_repetition_stop() {
        was_truncated         = true;
        stopped_on_repetition = true;
    }

    // Status of a run() whose decode finished: OK, or the stop that cut it short.
    transcribe_status truncation_status() const {
        if (stopped_on_repetition) {
            return TRANSCRIBE_ERR_OUTPUT_REPETITION;
        }
        return was_truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
    }

    // Streaming state. Lifecycle (stream_state) is separated from the
    // result snapshot so clear_result() can wipe per-call data without
    // churning the IDLE/ACTIVE/FINISHED/FAILED machine, which the
    // dispatcher manages explicitly. The snapshot fields below (revision,
    // committed counts, last_status, audio cursors) ARE cleared by
    // clear_result. Audio cursors are us-precision; the public
    // stream_update struct exposes them as ms.
    transcribe_stream_state         stream_state                     = TRANSCRIBE_STREAM_IDLE;
    int32_t                         stream_revision                  = 0;
    int                             n_committed_segments             = 0;
    int                             n_committed_words                = 0;
    int                             n_committed_tokens               = 0;
    transcribe_status               stream_last_status               = TRANSCRIBE_OK;
    int64_t                         stream_audio_input_us            = 0;
    int64_t                         stream_audio_committed_us        = 0;
    transcribe_stream_commit_policy stream_commit_policy             = TRANSCRIBE_STREAM_COMMIT_AUTO;
    uint32_t                        stream_stable_prefix_agreement_n = 0;

    // Session-owned copies of the caller's run-params strings, refreshed
    // on every transcribe_stream_begin. The dispatcher hands the family
    // hooks a params view whose language/target_language point HERE, so a
    // family that captures *run_params holds pointers into library-owned
    // storage (the public contract lets the caller free its params pointers
    // the moment begin returns). Stable for the stream's lifetime; only the
    // next begin mutates them.
    std::string               stream_language_owned;
    std::string               stream_target_language_owned;
    // Generic prompting strings for the stream's run-params view.
    std::vector<std::string>  stream_vocabulary_owned;
    std::vector<const char *> stream_vocabulary_ptrs;
    std::string               stream_prompt_owned;

    // UI-facing streaming text state. `full_text` above remains the raw
    // model hypothesis. `stream_committed_text` is the append-only public
    // display/input prefix; `stream_tentative_text` is the current raw
    // suffix after `stream_raw_tentative_start_bytes`. The raw history is
    // used by the generic STABLE_PREFIX policy.
    std::string             stream_committed_text;
    std::string             stream_tentative_text;
    uint64_t                stream_raw_tentative_start_bytes = 0;
    std::deque<std::string> stream_raw_history;

    void clear_result();

    transcribe_session() = default;
    ~transcribe_session() override;

    transcribe_session(const transcribe_session &)             = delete;
    transcribe_session & operator=(const transcribe_session &) = delete;
    transcribe_session(transcribe_session &&)                  = delete;
    transcribe_session & operator=(transcribe_session &&)      = delete;
};
