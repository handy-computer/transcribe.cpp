// transcribe-model.cpp - out-of-line definitions for the model and
// context base classes.
//
// Why an out-of-line .cpp at all when both bases are short:
//
//   1. Anchors the virtual destructors in one TU. Otherwise every TU
//      that includes transcribe-model.h would emit a vtable + RTTI
//      copy, which clang and gcc both warn about (-Wweak-vtables).
//   2. Holds set_languages(), which is non-trivial enough that inlining
//      it in the header would force the <vector>/<string> includes on
//      every consumer for no benefit.

#include "transcribe-model.h"
#include "transcribe-session.h"

#include <utility>

// Anchor for the model base vtable. The default destructor is correct;
// we just need it defined in exactly one TU.
transcribe_model::~transcribe_model() = default;

// Same anchoring trick for the context base.
transcribe_session::~transcribe_session() = default;

void transcribe_session::clear_result() {
    tokens.clear();
    words.clear();
    segments.clear();
    full_text.clear();
    detected_language.clear();
    result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    has_result  = false;

    // Stream snapshot — lifecycle (stream_state) is NOT touched here;
    // the streaming dispatcher manages IDLE/ACTIVE/FINISHED/FAILED
    // explicitly. Everything else is per-utterance bookkeeping that
    // belongs with the result it describes.
    stream_revision           = 0;
    n_committed_segments      = 0;
    n_committed_words         = 0;
    n_committed_tokens        = 0;
    stream_last_status        = TRANSCRIBE_OK;
    stream_audio_input_us     = 0;
    stream_audio_committed_us = 0;
    stream_commit_policy      = TRANSCRIBE_STREAM_COMMIT_AUTO;
    stream_stable_prefix_agreement_n = 0;
    stream_committed_text.clear();
    stream_tentative_text.clear();
    stream_raw_tentative_start_bytes = 0;
    stream_raw_history.clear();
}

void transcribe_model::set_languages(std::vector<std::string> langs) {
    // Move the strings into the model so their c_str() pointers stay
    // valid for the model's lifetime. Anything that previously lived in
    // language_storage_ is dropped (and the corresponding entries in
    // language_ptrs_ become dangling, which is why we rebuild the
    // pointer vector below before exposing it through caps).
    language_storage_ = std::move(langs);

    language_ptrs_.clear();
    language_ptrs_.reserve(language_storage_.size());
    for (const auto & s : language_storage_) {
        language_ptrs_.push_back(s.c_str());
    }

    // Publish to the capability struct atomically from the caller's
    // perspective: count and pointer match the same backing storage.
    // An empty vector exposes (0, nullptr) instead of (0, &empty[0])
    // because some callers reasonably treat a non-null pointer as a
    // claim that there is at least one element.
    caps.n_languages = static_cast<int>(language_storage_.size());
    caps.languages   = language_ptrs_.empty() ? nullptr : language_ptrs_.data();
}

void transcribe_model::set_translate_target_languages(std::vector<std::string> langs) {
    // Same discipline as set_languages(): move strings into the model,
    // rebuild the pointer vector, then publish count + pointer together.
    translate_target_storage_ = std::move(langs);

    translate_target_ptrs_.clear();
    translate_target_ptrs_.reserve(translate_target_storage_.size());
    for (const auto & s : translate_target_storage_) {
        translate_target_ptrs_.push_back(s.c_str());
    }

    caps.n_translate_target_languages =
        static_cast<int>(translate_target_storage_.size());
    caps.translate_target_languages =
        translate_target_ptrs_.empty() ? nullptr : translate_target_ptrs_.data();
}
