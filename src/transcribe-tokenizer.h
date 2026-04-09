// transcribe-tokenizer.h - internal SentencePiece-style tokenizer.
//
// This header is INTERNAL. The public C ABI does not (yet) expose
// tokenizer details directly; per-family models hold a Tokenizer instance
// and the central decode driver will read from it.
//
// What 2B implements:
//
//   - load(): read a fixed set of tokenizer.ggml.* keys from a
//     gguf_context. The set is intentionally the llama.cpp / whisper.cpp
//     intersection so converters and bindings can reuse existing
//     tooling.
//   - id <-> piece lookup.
//   - decode(): join a token-id sequence into a string, replacing the
//     SentencePiece word-boundary marker (U+2581 "▁") with ASCII space.
//
// What 2B does NOT implement:
//
//   - Encode (text -> ids). The decoder loop only needs the inverse
//     direction; bringing in a real SentencePiece encoder is its own
//     dependency story and we will tackle it when (if) we need it.
//   - BPE merge tables. Decoding does not need them.
//
// The tokenizer is decoder-only on purpose for 2B: parakeet inference
// only ever produces token ids and turns them into text, so we never
// take user text and turn it into ids in v1.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace transcribe {

class Tokenizer {
public:
    Tokenizer() = default;

    // Read tokenizer.ggml.* keys from a gguf_context.
    //
    // Required keys:
    //   tokenizer.ggml.model   (string) - "unigram" or "bpe". Both share
    //                           the same SentencePiece-style decode
    //                           contract (▁ -> space) so we accept both.
    //                           Other model strings are rejected with
    //                           TRANSCRIBE_ERR_NOT_IMPLEMENTED until we
    //                           need them.
    //   tokenizer.ggml.tokens  (array of string) - the vocabulary. Must
    //                           be non-empty.
    //
    // Optional keys (parsed if present, ignored otherwise):
    //   tokenizer.ggml.scores         (array of float32)
    //   tokenizer.ggml.token_type     (array of int32)
    //   tokenizer.ggml.unknown_token_id (uint32 / int32)
    //   tokenizer.ggml.bos_token_id     (uint32 / int32)
    //   tokenizer.ggml.eos_token_id     (uint32 / int32)
    //   tokenizer.ggml.blank_token_id   (uint32 / int32) - the CTC /
    //       transducer blank id. Optional in 2B because no decode
    //       driver consumes it yet; will become required when 4 lands.
    //
    // Returns:
    //   TRANSCRIBE_OK                  on success.
    //   TRANSCRIBE_ERR_INVALID_ARG     if gguf is null.
    //   TRANSCRIBE_ERR_GGUF            if a required key is missing or
    //                                  has the wrong type, or if scores /
    //                                  token_type lengths disagree with
    //                                  the token count.
    //   TRANSCRIBE_ERR_NOT_IMPLEMENTED if the tokenizer model string is
    //                                  recognized but not yet supported
    //                                  (e.g. "wordpiece").
    transcribe_status load(const gguf_context * gguf);

    // Vocabulary access. token(id) returns an empty string for an
    // out-of-range id (matching the safe-sentinel pattern used by the
    // public result accessors). find(piece) returns -1 if the piece is
    // not in the vocabulary.
    int                 n_tokens() const { return static_cast<int>(tokens_.size()); }
    const std::string & token   (int id) const;
    int                 find    (const std::string & piece) const;

    // Join a token-id sequence into a single string. SentencePiece
    // convention: U+2581 ("▁") is the visible word-boundary marker and
    // is replaced with ASCII space at decode time. Out-of-range ids are
    // skipped silently — the decoder loop is responsible for not
    // emitting them in the first place; this is just a defensive last
    // line.
    std::string decode(const int * ids, int n) const;

    // Identification + special token ids. -1 if the corresponding key
    // was absent from the GGUF.
    const std::string & model_type() const { return model_; }
    int                 unk_id   () const { return unk_id_; }
    int                 bos_id   () const { return bos_id_; }
    int                 eos_id   () const { return eos_id_; }
    int                 blank_id () const { return blank_id_; }

private:
    std::string              model_;
    std::vector<std::string> tokens_;
    std::vector<float>       scores_;     // optional, may be empty
    std::vector<int32_t>     token_type_; // optional, may be empty

    int unk_id_   = -1;
    int bos_id_   = -1;
    int eos_id_   = -1;
    int blank_id_ = -1;
};

} // namespace transcribe
