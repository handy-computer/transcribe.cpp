// tokenizer_decode_only_unit.cpp - unit smoke for the in-memory
// decode-only Tokenizer constructors.
//
// Two paths feed the Tokenizer:
//   - load(gguf_context*) — production path for canonical GGUFs.
//   - load_decode_only_*   — sibling path for non-GGUF sources. The
//     legacy whisper.cpp .bin adapter (PR3) needs a decoder built
//     directly from an in-memory vocab; tests here cover the two
//     decode flavors that loader exposes:
//
//        load_decode_only_gpt2       - tokens stored in the GPT-2
//                                      byte-to-unicode-remapped form
//                                      (matches GGUF "gpt2" decode).
//        load_decode_only_raw_bytes  - tokens stored as raw UTF-8 bytes
//                                      (matches whisper.cpp .bin
//                                      tiktoken-style vocab).
//
// What we verify:
//   - vocab + piece_to_id round-trip via token(), find(), n_tokens().
//   - decode() picks the right strategy: byte-unicode inversion for
//     gpt2-decode-only, verbatim concatenation for raw-bytes.
//   - gpt2-decode-only reports has_encoder() false and rejects encode()
//     because no merges are present.
//   - raw-bytes mode reports has_encoder() true and can encode via
//     tiktoken-style piece_to_id_ ranks.
//   - DecodeOnlySpecials values are carried into bos/eos/unk/blank ids.

#include "gguf.h"
#include "transcribe-tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                                                        \
    do {                                                                                                          \
        const std::string _av = (a);                                                                              \
        const std::string _bv = (b);                                                                              \
        if (_av != _bv) {                                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: \"%s\" != \"%s\"\n", __FILE__, __LINE__, _av.c_str(), _bv.c_str()); \
            ++g_failures;                                                                                         \
        }                                                                                                         \
    } while (0)

// UTF-8 of "é" is the two raw bytes 0xC3 0xA9. Under GPT-2's
// byte-to-unicode remap, byte 0xC3 → U+00C3 ("Ã"), byte 0xA9 → U+00A9
// ("©"). The UTF-8 encoding of those two codepoints is the four bytes
// 0xC3 0x83 0xC2 0xA9 (the bytes of "Ã©"). decode_gpt2_bytes inverts
// the remap per codepoint to recover 0xC3 0xA9.
const char k_eacute_raw[]   = "\xC3\xA9";          // "é" raw UTF-8
const char k_eacute_remap[] = "\xC3\x83\xC2\xA9";  // "Ã©" UTF-8

void test_gpt2_decode_only() {
    transcribe::Tokenizer tok;

    std::vector<std::string> vocab;
    vocab.emplace_back("hello");          // id 0: ASCII passes through
    vocab.emplace_back(k_eacute_remap);   // id 1: byte-remapped "é"
    vocab.emplace_back("<|endoftext|>");  // id 2: special, falls back

    transcribe::Tokenizer::DecodeOnlySpecials specials;
    specials.bos_id = 11;
    specials.eos_id = 22;
    specials.unk_id = 33;

    const auto rc = tok.load_decode_only_gpt2(vocab, specials);
    CHECK(rc == TRANSCRIBE_OK);

    CHECK(tok.n_tokens() == 3);
    CHECK_STR_EQ(tok.token(0), "hello");
    CHECK_STR_EQ(tok.token(1), k_eacute_remap);
    CHECK(tok.find("hello") == 0);
    CHECK(tok.find(k_eacute_remap) == 1);
    CHECK(tok.find("missing") == -1);

    CHECK(tok.bos_id() == 11);
    CHECK(tok.eos_id() == 22);
    CHECK(tok.unk_id() == 33);

    // Encode is unavailable (no merges).
    CHECK(!tok.has_encoder());
    std::vector<int32_t> out;
    CHECK(tok.encode("hello", out) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);

    // Decode: id 0 is ASCII, id 1 inverts the remap to raw "é".
    {
        const int ids[] = { 0 };
        CHECK_STR_EQ(tok.decode(ids, 1), "hello");
    }
    {
        const int ids[] = { 1 };
        CHECK_STR_EQ(tok.decode(ids, 1), k_eacute_raw);
    }
    {
        const int ids[] = { 0, 1 };
        CHECK_STR_EQ(tok.decode(ids, 2), std::string("hello") + k_eacute_raw);
    }
}

void test_raw_bytes_decode_only() {
    transcribe::Tokenizer tok;

    // Build a vocab that includes the 256 single-byte base tokens
    // (ids 0..255) plus a couple of multi-byte merges. This mirrors
    // the structure of a real whisper.cpp .bin: bytes are the bottom
    // of the rank ladder, longer tokens at higher ranks.
    std::vector<std::string> vocab;
    vocab.reserve(258);
    for (int i = 0; i < 256; ++i) {
        vocab.emplace_back(1, static_cast<char>(static_cast<unsigned char>(i)));
    }
    vocab.emplace_back("hello");       // id 256
    vocab.emplace_back(k_eacute_raw);  // id 257: raw "é" bytes

    transcribe::Tokenizer::DecodeOnlySpecials specials;
    specials.eos_id = 50257;

    const auto rc = tok.load_decode_only_raw_bytes(vocab, specials);
    CHECK(rc == TRANSCRIBE_OK);

    CHECK(tok.n_tokens() == 258);
    CHECK_STR_EQ(tok.token(256), "hello");
    CHECK_STR_EQ(tok.token(257), k_eacute_raw);
    CHECK(tok.eos_id() == 50257);

    // Tiktoken-style encoder is now available for raw-bytes mode:
    // piece_to_id_ doubles as the rank table.
    CHECK(tok.has_encoder());

    {
        // 2-byte "é" matches multi-byte token id 257 in one merge step
        // (the only adjacent pair concatenation lands directly on the
        // vocab entry).
        std::vector<int32_t> out;
        CHECK(tok.encode(k_eacute_raw, out) == TRANSCRIBE_OK);
        CHECK(out.size() == 1);
        if (out.size() == 1) {
            CHECK(out[0] == 257);
        }
    }
    {
        // No multi-byte token covers "x" — encode falls back to the
        // single-byte token (id 'x' == 0x78 == 120).
        std::vector<int32_t> out;
        CHECK(tok.encode("x", out) == TRANSCRIBE_OK);
        CHECK(out.size() == 1);
        if (out.size() == 1) {
            CHECK(out[0] == 0x78);
        }
    }
    {
        // "hello" doesn't merge into a single token because real BPE
        // walks adjacent-pair concatenations bottom-up: there's no
        // path "h"+"e"→"he" with our toy vocab. Falls back to 5
        // single-byte tokens. Round-trip invariant still holds:
        // decode(encode(text)) == text.
        std::vector<int32_t> out;
        CHECK(tok.encode("hello", out) == TRANSCRIBE_OK);
        CHECK(out.size() == 5);
        std::vector<int> ids_int(out.begin(), out.end());
        CHECK_STR_EQ(tok.decode(ids_int.data(), static_cast<int>(ids_int.size())), "hello");
    }

    // Decode: raw-bytes mode concatenates verbatim, no remap inversion.
    {
        const int ids[] = { 256, 257 };
        CHECK_STR_EQ(tok.decode(ids, 2), std::string("hello") + k_eacute_raw);
    }
    // Out-of-range ids are silently skipped (defensive).
    {
        const int ids[] = { 256, 9999 };
        CHECK_STR_EQ(tok.decode(ids, 2), "hello");
    }
}

void test_empty_vocab_rejected() {
    transcribe::Tokenizer    tok;
    std::vector<std::string> empty;
    CHECK(tok.load_decode_only_gpt2(empty) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(tok.load_decode_only_raw_bytes(empty) == TRANSCRIBE_ERR_INVALID_ARG);
}

void test_sentencepiece_bpe_canonicalization() {
    const char * tokens[] = {
        "<unk>",
        "\xE2\x96\x81",
        "h",
        "e",
        "l",
        "o",
        "\xE2\x96\x81h",
        "\xE2\x96\x81he",
        "ll",
        "llo",
        "\xE2\x96\x81hello",
        "w",
        "r",
        "d",
        "\xE2\x96\x81w",
        "\xE2\x96\x81wo",
        "\xE2\x96\x81wor",
        "\xE2\x96\x81worl",
        "\xE2\x96\x81world",
    };
    const float scores[] = {
        0.0f,  -20.0f, -20.0f, -20.0f, -20.0f, -20.0f, -9.0f, -8.0f, -7.0f, -6.0f,
        -5.0f, -20.0f, -20.0f, -20.0f, -9.0f,  -8.0f,  -7.0f, -6.0f, -5.0f,
    };
    int32_t types[sizeof(tokens) / sizeof(tokens[0])] = {};
    types[0]                                          = 2;
    for (size_t i = 1; i < sizeof(types) / sizeof(types[0]); ++i) {
        types[i] = 1;
    }

    gguf_context * gguf = gguf_init_empty();
    CHECK(gguf != nullptr);
    if (gguf == nullptr) {
        return;
    }
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "bpe");
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, sizeof(tokens) / sizeof(tokens[0]));
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, sizeof(scores) / sizeof(scores[0]));
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, sizeof(types) / sizeof(types[0]));
    gguf_set_val_u32(gguf, "tokenizer.ggml.unknown_token_id", 0);

    transcribe::Tokenizer tok;
    CHECK(tok.load(gguf) == TRANSCRIBE_OK);
    gguf_free(gguf);
    CHECK(tok.has_bpe_canonicalizer());

    // This valid decoded sequence is deliberately not the canonical BPE
    // segmentation and contains duplicate decoded whitespace.
    const int            noncanonical[] = { 1, 1, 2, 3, 4, 4, 5, 1, 11, 5, 12, 4, 13, 1 };
    std::vector<int32_t> canonical;
    CHECK(tok.canonicalize_bpe(noncanonical, static_cast<int>(sizeof(noncanonical) / sizeof(noncanonical[0])),
                               canonical) == TRANSCRIBE_OK);
    CHECK(canonical == std::vector<int32_t>({ 10, 18 }));

    // The general plain-text encoder remains unavailable for SentencePiece;
    // canonicalization does not claim unigram or arbitrary normalizer support.
    std::vector<int32_t> encoded;
    CHECK(tok.encode("hello world", encoded) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);

    gguf = gguf_init_empty();
    CHECK(gguf != nullptr);
    if (gguf != nullptr) {
        gguf_set_val_str(gguf, "tokenizer.ggml.model", "bpe");
        gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, sizeof(tokens) / sizeof(tokens[0]));
        transcribe::Tokenizer decode_only;
        CHECK(decode_only.load(gguf) == TRANSCRIBE_OK);
        CHECK(!decode_only.has_bpe_canonicalizer());
        gguf_free(gguf);
    }
}

}  // namespace

int main() {
    test_gpt2_decode_only();
    test_raw_bytes_decode_only();
    test_empty_vocab_rejected();
    test_sentencepiece_bpe_canonicalization();

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}
