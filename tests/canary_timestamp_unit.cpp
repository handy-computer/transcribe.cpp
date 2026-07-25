// canary_timestamp_unit.cpp - synthetic CTC forced-alignment tests.

#include "arch/canary/canary.h"
#include "gguf.h"

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

std::vector<float> emissions_for_path(const std::vector<int> & path, int vocab_size) {
    std::vector<float> emissions(path.size() * static_cast<size_t>(vocab_size), -20.0f);
    for (size_t frame = 0; frame < path.size(); ++frame) {
        emissions[frame * static_cast<size_t>(vocab_size) + path[frame]] = 0.0f;
    }
    return emissions;
}

void test_distinct_tokens() {
    constexpr int          vocab = 3;
    constexpr int          blank = 2;
    const std::vector<int> path  = { blank, 0, blank, 1, blank };
    std::vector<int>       alignment;

    CHECK(transcribe::canary::viterbi_ctc_alignment(emissions_for_path(path, vocab), static_cast<int>(path.size()),
                                                    vocab, blank, { 0, 1 }, alignment));
    CHECK(alignment == std::vector<int>({ 0, 1, 2, 3, 4 }));
}

void test_repeated_token_requires_blank() {
    constexpr int          vocab = 4;
    constexpr int          blank = 3;
    const std::vector<int> path  = { blank, 1, blank, 1, blank, 2, blank };
    std::vector<int>       alignment;

    CHECK(transcribe::canary::viterbi_ctc_alignment(emissions_for_path(path, vocab), static_cast<int>(path.size()),
                                                    vocab, blank, { 1, 1, 2 }, alignment));
    CHECK(alignment == std::vector<int>({ 0, 1, 2, 3, 4, 5, 6 }));
}

void test_invalid_inputs() {
    std::vector<int> alignment = { 42 };
    CHECK(!transcribe::canary::viterbi_ctc_alignment({}, 0, 3, 2, { 0 }, alignment));
    CHECK(alignment.empty());

    const std::vector<float> emissions(6, 0.0f);
    CHECK(!transcribe::canary::viterbi_ctc_alignment(emissions, 2, 3, 2, { 2 }, alignment));
    CHECK(!transcribe::canary::viterbi_ctc_alignment(emissions, 2, 3, 2, { 0, 0 }, alignment));
}

void test_aligner_limit_only_applies_to_timestamps() {
    transcribe::canary::CanaryHParams hp;
    hp.aligner_present            = true;
    hp.aligner_subsampling_factor = 8;
    hp.aligner_pos_emb_max_len    = 10;

    CHECK(!transcribe::canary::canary_aligner_input_too_long(80, hp, TRANSCRIBE_TIMESTAMPS_WORD));
    CHECK(transcribe::canary::canary_aligner_input_too_long(81, hp, TRANSCRIBE_TIMESTAMPS_WORD));
    CHECK(transcribe::canary::canary_aligner_input_too_long(81, hp, TRANSCRIBE_TIMESTAMPS_AUTO));
    CHECK(!transcribe::canary::canary_aligner_input_too_long(81, hp, TRANSCRIBE_TIMESTAMPS_NONE));

    hp.aligner_present = false;
    CHECK(!transcribe::canary::canary_aligner_input_too_long(81, hp, TRANSCRIBE_TIMESTAMPS_WORD));
}

void test_keep_special_tags_in_aligned_words() {
    const char *  tokens[] = { "<unk>", "\xE2\x96\x81hello", "\xE2\x96\x81world", "<|start|>", "<|mid|>", "<|end|>" };
    const float   scores[] = { 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f };
    const int32_t types[]  = { 2, 1, 1, 3, 3, 3 };

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

    transcribe::Tokenizer tokenizer;
    CHECK(tokenizer.load(gguf) == TRANSCRIBE_OK);
    gguf_free(gguf);

    const std::vector<std::string> words =
        transcribe::canary::canary_preserve_special_tags(tokenizer, { 3, 1, 4, 2, 5 }, { "hello", "world" });
    CHECK(words == std::vector<std::string>({ "<|start|> hello<|mid|>", "world<|end|>" }));
}

}  // namespace

int main() {
    test_distinct_tokens();
    test_repeated_token_requires_blank();
    test_invalid_inputs();
    test_aligner_limit_only_applies_to_timestamps();
    test_keep_special_tags_in_aligned_words();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::puts("canary timestamp unit: ok");
    return 0;
}
