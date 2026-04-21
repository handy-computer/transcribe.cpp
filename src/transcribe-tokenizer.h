// transcribe-tokenizer.h - internal tokenizer (decode + encode).
//
// This header is INTERNAL. The public C ABI does not expose tokenizer
// details directly; per-family models hold a Tokenizer instance and
// the decode driver plus the chat-template prompt builders read /
// write through it.
//
// Supported GGUF tokenizer models (tokenizer.ggml.model):
//   "unigram"  - SentencePiece unigram (NeMo Parakeet, Cohere ASR)
//   "bpe"      - SentencePiece BPE (same ▁ decode convention)
//   "gpt2"     - llama.cpp's tag for Hugging Face byte-level BPE
//                (Qwen2/Qwen3 family). Loads merges at init and
//                supports encode() for arbitrary UTF-8 text via the
//                Qwen2 pretokenizer + byte-level BPE merge loop.
//
// What this class does:
//
//   - load(): read a fixed set of tokenizer.ggml.* keys from a
//     gguf_context. The set is intentionally the llama.cpp /
//     whisper.cpp intersection so converters and bindings can reuse
//     existing tooling.
//   - id <-> piece lookup. find() is O(1) via a hash built at load.
//   - decode(): join a token-id sequence into a string. SentencePiece
//     flavors substitute U+2581 ("▁") with ASCII space; "gpt2" inverts
//     the byte-level mapping to recover raw UTF-8 bytes.
//   - encode(): UTF-8 text -> token ids. Implemented for "gpt2" only
//     in 2B (other flavors return NOT_IMPLEMENTED). The encoder runs
//     the Qwen2 pretokenizer, byte-level encodes each pretoken, and
//     greedily applies BPE merges in rank order.
//
// The encoder only exists to render the Qwen3-ASR chat-template
// assistant prefix ("language {Name}<asr_text>") at decode prep time.
// If that's the *only* live consumer, a converter-side pre-
// tokenization would also work; we chose runtime encode so that
// forthcoming features (non-empty system prompt, streaming prefix
// rollback, forced-aligner text input) can reuse the same path.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

namespace transcribe {

class Tokenizer {
public:
    Tokenizer() = default;

    // Read tokenizer.ggml.* keys from a gguf_context.
    //
    // Required keys:
    //   tokenizer.ggml.model   (string) - "unigram", "bpe", or "gpt2".
    //                           Other model strings are rejected with
    //                           TRANSCRIBE_ERR_NOT_IMPLEMENTED.
    //   tokenizer.ggml.tokens  (array of string) - the vocabulary.
    //                           Must be non-empty.
    //
    // Optional keys (parsed if present, ignored otherwise):
    //   tokenizer.ggml.scores         (array of float32)
    //   tokenizer.ggml.token_type     (array of int32)
    //   tokenizer.ggml.merges         (array of string) - "left right"
    //                                 merge pairs in rank order.
    //                                 Required for encode() when
    //                                 model == "gpt2".
    //   tokenizer.ggml.unknown_token_id (uint32 / int32)
    //   tokenizer.ggml.bos_token_id     (uint32 / int32)
    //   tokenizer.ggml.eos_token_id     (uint32 / int32)
    //   tokenizer.ggml.blank_token_id   (uint32 / int32) - the CTC /
    //       transducer blank id.
    //
    // Returns:
    //   TRANSCRIBE_OK                  on success.
    //   TRANSCRIBE_ERR_INVALID_ARG     if gguf is null.
    //   TRANSCRIBE_ERR_GGUF            if a required key is missing,
    //                                  has the wrong type, or arrays
    //                                  are length-inconsistent.
    //   TRANSCRIBE_ERR_NOT_IMPLEMENTED if the tokenizer model string
    //                                  is recognized but not yet
    //                                  supported (e.g. "wordpiece").
    transcribe_status load(const gguf_context * gguf);

    // Vocabulary access. token(id) returns an empty string for an
    // out-of-range id (matching the safe-sentinel pattern used by the
    // public result accessors). find(piece) returns -1 if the piece
    // is not in the vocabulary.
    int                 n_tokens() const { return static_cast<int>(tokens_.size()); }
    const std::string & token   (int id) const;
    int                 find    (const std::string & piece) const;

    // Join a token-id sequence into a single string. SentencePiece
    // tokenizers ("unigram", "bpe") replace the word-boundary marker
    // U+2581 with ASCII space. "gpt2" inverts the byte-level map to
    // recover raw UTF-8 bytes. Out-of-range ids are skipped silently
    // — the decode driver is responsible for not emitting them in
    // the first place; this is just a defensive last line.
    std::string decode(const int * ids, int n) const;

    // UTF-8 text -> token ids.
    //
    // model == "gpt2":
    //   Pretokenize (Qwen2 regex), byte-level encode each pretoken,
    //   then apply BPE merges in rank order. The output is the same
    //   token-id sequence the Hugging Face tokenizer would produce
    //   with add_special_tokens=False (no BOS/EOS added). Special
    //   tokens in the input (e.g. "<|im_start|>") are NOT recognized
    //   by this encoder; if they appear in the text they get
    //   BPE-encoded piece-by-piece, which is usually wrong. Callers
    //   render special tokens via direct id lookups and encode only
    //   the plain-text fragments between them.
    //
    // model == "unigram" / "bpe":
    //   Currently returns TRANSCRIBE_ERR_NOT_IMPLEMENTED; no live
    //   consumer needs it.
    //
    // Returns:
    //   TRANSCRIBE_OK                  on success, out_ids populated.
    //   TRANSCRIBE_ERR_NOT_IMPLEMENTED if the tokenizer model doesn't
    //                                  support encoding yet.
    //   TRANSCRIBE_ERR_GGUF            if "gpt2" is loaded without
    //                                  merges (the encoder needs them).
    transcribe_status encode(const std::string &    text,
                             std::vector<int32_t> & out_ids) const;

    // Identification + special token ids. -1 if the corresponding key
    // was absent from the GGUF.
    const std::string & model_type() const { return model_; }
    int                 unk_id   () const { return unk_id_; }
    int                 bos_id   () const { return bos_id_; }
    int                 eos_id   () const { return eos_id_; }
    int                 blank_id () const { return blank_id_; }

    // True if encode() is available for this tokenizer. Callers that
    // want to gate a feature on encoder presence can branch on this
    // without inspecting model_type().
    bool has_encoder() const;

private:
    std::string              model_;
    std::vector<std::string> tokens_;
    std::vector<float>       scores_;     // optional, may be empty
    std::vector<int32_t>     token_type_; // optional, may be empty

    // O(1) piece -> id lookup built at load time. Duplicates are rare
    // (only from the byte-fallback vocab augmentation); we keep the
    // first-occurrence id which matches the forward tokens_ ordering.
    std::unordered_map<std::string, int32_t> piece_to_id_;

    // BPE merge table (populated only when model_ == "gpt2").
    //
    // merge_rank_ maps a "left<\x1F>right" concatenation to the merge
    // rank (line index in tokenizer.ggml.merges). Lower rank = higher
    // priority = merged earlier. The \x1F (unit separator) byte does
    // not appear in any Qwen byte-level BPE token, so collisions are
    // impossible.
    std::unordered_map<std::string, int32_t> merge_rank_;

    int unk_id_   = -1;
    int bos_id_   = -1;
    int eos_id_   = -1;
    int blank_id_ = -1;
};

} // namespace transcribe
