// transcribe-tokenizer.cpp - implementation of the internal tokenizer.
//
// See transcribe-tokenizer.h for the contract. This file knows nothing
// about model families: it just reads tokenizer.ggml.* keys, and exposes
// id <-> piece + a SentencePiece-style decode().
//
// Strict-now contract on KV reads (per the 2B revision after the
// Finding-2 review): a tokenizer.ggml.* key that is present but has
// the wrong GGUF type is treated as a converter bug and surfaced as
// TRANSCRIBE_ERR_GGUF, NOT silently ignored as if it were absent. The
// helpers in transcribe-meta.h provide the KvResult tri-state that
// makes this distinction observable.

#include "transcribe-tokenizer.h"

#include "transcribe-meta.h"

#include "gguf.h"

#include <cstring>

namespace transcribe {

namespace {

// SentencePiece word-boundary marker, U+2581 "LOWER ONE EIGHTH BLOCK".
// UTF-8: 0xE2 0x96 0x81. Used by NeMo Parakeet (and most other recent
// SentencePiece-tokenized models) as the visible "leading space" prefix
// of any token that starts a new word.
constexpr const char k_sp_space[] = "\xE2\x96\x81";
constexpr int        k_sp_space_len = 3;

// Read a typed-element scalar array KV. ExpectedType is the gguf_type
// the array elements should have; T is the C++ type the loader copies
// them into. The contiguous flat array is reachable via gguf_get_arr_data
// and we just memcpy. Returns KvResult so the caller can distinguish
// "absent" from "present but wrong array element type".
//
// This stays here (rather than being lifted into transcribe-meta.h)
// because the only consumers in the codebase right now are scores and
// token_type below. If a third caller materializes we can promote.
template <typename T>
KvResult read_typed_array_kv(const gguf_context * ctx,
                             const char *         key,
                             gguf_type            expected_element_type,
                             std::vector<T> &     out)
{
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return KvResult::Absent;
    }
    if (gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_ARRAY) {
        return KvResult::BadType;
    }
    if (gguf_get_arr_type(ctx, key_id) != expected_element_type) {
        return KvResult::BadType;
    }

    const size_t n = gguf_get_arr_n(ctx, key_id);
    std::vector<T> tmp(n);
    if (n > 0) {
        const void * data = gguf_get_arr_data(ctx, key_id);
        if (data == nullptr) {
            return KvResult::BadType;
        }
        std::memcpy(tmp.data(), data, n * sizeof(T));
    }
    out = std::move(tmp);
    return KvResult::Ok;
}

} // namespace

const std::string & Tokenizer::token(int id) const {
    static const std::string k_empty;
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
        return k_empty;
    }
    return tokens_[static_cast<size_t>(id)];
}

int Tokenizer::find(const std::string & piece) const {
    // Linear scan. Vocabularies are at most a few thousand entries; the
    // only caller in 2B is the test fixture, which calls this a handful
    // of times. If real callers need fast piece->id we will add an
    // unordered_map<string,int> on load().
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (tokens_[i] == piece) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::string Tokenizer::decode(const int * ids, int n) const {
    if (ids == nullptr || n <= 0) {
        return {};
    }
    std::string out;
    // Each piece is at most a few bytes; reserve a guess that avoids
    // most reallocations without overcommitting on huge sequences.
    out.reserve(static_cast<size_t>(n) * 4);
    for (int i = 0; i < n; ++i) {
        const int id = ids[i];
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) {
            // Defensive: the decode driver is responsible for not
            // emitting blank or out-of-range ids. Skip silently here so
            // a buggy upstream pass does not corrupt the rest of the
            // string.
            continue;
        }
        const std::string & p = tokens_[static_cast<size_t>(id)];
        // SentencePiece word-boundary substitution. We walk the piece
        // byte by byte; the marker is exactly three bytes. The bulk of
        // tokens contain at most one marker (at the start), but
        // multilingual variants do produce mid-piece markers, so we
        // scan the whole string.
        size_t j = 0;
        while (j < p.size()) {
            if (j + k_sp_space_len <= p.size() &&
                std::memcmp(p.data() + j, k_sp_space, k_sp_space_len) == 0)
            {
                out.push_back(' ');
                j += k_sp_space_len;
            } else {
                out.push_back(p[j]);
                ++j;
            }
        }
    }
    return out;
}

transcribe_status Tokenizer::load(const gguf_context * gguf) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Required: model type. Both Absent and BadType are fatal — a
    // missing model string and a model string with the wrong GGUF type
    // are both "we cannot trust this file's tokenizer payload."
    switch (read_string_kv(gguf, "tokenizer.ggml.model", model_)) {
        case KvResult::Absent:
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            break;
    }

    // We accept "unigram" and "bpe" (the two SentencePiece flavors NeMo
    // Parakeet ships in practice). Both share the same decode-side
    // semantics — the visible vocabulary already encodes the
    // SentencePiece word-boundary marker — so a single decode path is
    // correct for both. Recognized-but-unsupported tokenizer model
    // strings (e.g. "wordpiece") surface as NOT_IMPLEMENTED so the
    // caller can tell "the file is fine, the library is just not
    // ready for this tokenizer" from "the file is broken."
    if (model_ != "unigram" && model_ != "bpe") {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    // Required: tokens array. Same Absent/BadType semantics.
    switch (read_string_array_kv(gguf, "tokenizer.ggml.tokens", tokens_)) {
        case KvResult::Absent:
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            break;
    }
    if (tokens_.empty()) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Optional: scores. Absent is fine (we clear and move on); BadType
    // is a converter bug and propagates as ERR_GGUF. The
    // length-must-match contract on Ok is a separate, narrower check.
    switch (read_typed_array_kv<float>(gguf, "tokenizer.ggml.scores",
                                       GGUF_TYPE_FLOAT32, scores_))
    {
        case KvResult::Absent:
            scores_.clear();
            break;
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            if (scores_.size() != tokens_.size()) {
                return TRANSCRIBE_ERR_GGUF;
            }
            break;
    }

    // Optional: token_type. Same shape as scores above.
    switch (read_typed_array_kv<int32_t>(gguf, "tokenizer.ggml.token_type",
                                         GGUF_TYPE_INT32, token_type_))
    {
        case KvResult::Absent:
            token_type_.clear();
            break;
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            if (token_type_.size() != tokens_.size()) {
                return TRANSCRIBE_ERR_GGUF;
            }
            break;
    }

    // Optional special token ids. Each one is independently optional;
    // a present-but-wrong-type id is fatal because no in-range value
    // could meaningfully recover from it.
    auto read_special = [&](const char * key, int & field) -> transcribe_status {
        switch (read_token_id_kv(gguf, key, field)) {
            case KvResult::Absent:  return TRANSCRIBE_OK;
            case KvResult::Ok:      return TRANSCRIBE_OK;
            case KvResult::BadType: return TRANSCRIBE_ERR_GGUF;
        }
        return TRANSCRIBE_ERR_GGUF;
    };

    if (auto st = read_special("tokenizer.ggml.unknown_token_id", unk_id_);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (auto st = read_special("tokenizer.ggml.bos_token_id", bos_id_);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (auto st = read_special("tokenizer.ggml.eos_token_id", eos_id_);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (auto st = read_special("tokenizer.ggml.blank_token_id", blank_id_);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    return TRANSCRIBE_OK;
}

} // namespace transcribe
