// transcribe-context.h - internal base class for the public opaque
// transcribe_context handle.
//
// Mirror of transcribe-model.h. Same Option B layout: a small base owned
// by the central dispatch, derived per-family contexts owning everything
// else. The base has a virtual destructor so transcribe_context_free()
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

struct transcribe_context {
    // The model this context was constructed from. Borrowed pointer:
    // the caller is required (per the public threading contract) to keep
    // the model alive for the lifetime of every derived context.
    transcribe_model * model = nullptr;

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
    transcribe_timestamp_kind    result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    bool                         has_result  = false;

    void clear_result();

    transcribe_context() = default;
    virtual ~transcribe_context();

    transcribe_context(const transcribe_context &)             = delete;
    transcribe_context & operator=(const transcribe_context &) = delete;
    transcribe_context(transcribe_context &&)                  = delete;
    transcribe_context & operator=(transcribe_context &&)      = delete;
};
