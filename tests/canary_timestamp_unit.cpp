// canary_timestamp_unit.cpp - synthetic CTC forced-alignment tests.

#include "arch/canary/canary.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <limits>
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

void test_long_form_chunk_schedule() {
    constexpr int sample_rate = 16000;

    const std::vector<transcribe::canary::CanaryChunkSpan> short_chunks =
        transcribe::canary::canary_long_form_chunks(40 * sample_rate, sample_rate);
    CHECK(short_chunks.size() == 1);
    if (short_chunks.size() == 1) {
        CHECK(short_chunks[0].start_sample == 0);
        CHECK(short_chunks[0].n_samples == 40 * sample_rate);
    }

    const int                                              barely_long_samples = 40 * sample_rate + 1;
    const std::vector<transcribe::canary::CanaryChunkSpan> barely_long_chunks =
        transcribe::canary::canary_long_form_chunks(barely_long_samples, sample_rate);
    CHECK(barely_long_chunks.size() == 2);
    if (!barely_long_chunks.empty()) {
        CHECK(static_cast<int64_t>(barely_long_chunks.back().start_sample) + barely_long_chunks.back().n_samples ==
              barely_long_samples);
    }

    const std::vector<transcribe::canary::CanaryChunkSpan> chunks =
        transcribe::canary::canary_long_form_chunks(80 * sample_rate, sample_rate);
    CHECK(chunks.size() == 3);
    if (chunks.size() == 3) {
        CHECK(chunks[0].start_sample == 0);
        CHECK(chunks[0].n_samples == 30 * sample_rate);
        CHECK(chunks[1].start_sample == 29 * sample_rate);
        CHECK(chunks[1].n_samples == 30 * sample_rate);
        CHECK(chunks[2].start_sample == 58 * sample_rate);
        CHECK(chunks[2].n_samples == 22 * sample_rate);
    }

    const int                                              max_samples = std::numeric_limits<int>::max();
    const std::vector<transcribe::canary::CanaryChunkSpan> max_chunks =
        transcribe::canary::canary_long_form_chunks(max_samples, sample_rate);
    CHECK(!max_chunks.empty());
    if (!max_chunks.empty()) {
        CHECK(max_chunks.front().start_sample == 0);
        CHECK(static_cast<int64_t>(max_chunks.back().start_sample) + max_chunks.back().n_samples == max_samples);
        for (size_t i = 0; i < max_chunks.size(); ++i) {
            CHECK(max_chunks[i].start_sample >= 0);
            CHECK(max_chunks[i].n_samples > 0);
            CHECK(max_chunks[i].n_samples <= 40 * sample_rate);
            CHECK(static_cast<int64_t>(max_chunks[i].start_sample) + max_chunks[i].n_samples <= max_samples);
            if (i > 0) {
                CHECK(max_chunks[i].start_sample > max_chunks[i - 1].start_sample);
                CHECK(max_chunks[i].start_sample <
                      static_cast<int64_t>(max_chunks[i - 1].start_sample) + max_chunks[i - 1].n_samples);
            }
        }
    }
}

void test_long_form_token_seam() {
    const transcribe::canary::CanaryTokenSeam exact =
        transcribe::canary::canary_token_seam({ 1, 2, 3, 4, 5 }, { 4, 5, 6, 7 }, 5, 4);
    CHECK(exact.matched);
    CHECK(exact.previous_keep == 5);
    CHECK(exact.current_skip == 2);

    const transcribe::canary::CanaryTokenSeam replacement =
        transcribe::canary::canary_token_seam({ 1, 2, 3, 9 }, { 2, 3, 4 }, 4, 3);
    CHECK(!replacement.matched);
    CHECK(replacement.previous_keep == 4);
    CHECK(replacement.current_skip == 0);

    const transcribe::canary::CanaryTokenSeam interior =
        transcribe::canary::canary_token_seam({ 1, 2, 3, 4 }, { 9, 3, 8 }, 4, 3);
    CHECK(!interior.matched);
    CHECK(interior.previous_keep == 4);
    CHECK(interior.current_skip == 0);

    const transcribe::canary::CanaryTokenSeam strong_interior =
        transcribe::canary::canary_token_seam({ 10, 1, 2, 3, 4, 11 }, { 20, 1, 2, 3, 4, 21 }, 6, 6);
    CHECK(!strong_interior.matched);
    CHECK(strong_interior.previous_keep == 6);
    CHECK(strong_interior.current_skip == 0);

    const transcribe::canary::CanaryTokenSeam exact_one =
        transcribe::canary::canary_token_seam({ 1, 2, 3 }, { 3, 4 }, 3, 2);
    CHECK(exact_one.matched);
    CHECK(exact_one.previous_keep == 3);
    CHECK(exact_one.current_skip == 1);

    const transcribe::canary::CanaryTokenSeam none = transcribe::canary::canary_token_seam({ 1, 2 }, { 3, 4 }, 2, 2);
    CHECK(!none.matched);
    CHECK(none.previous_keep == 2);
    CHECK(none.current_skip == 0);
}

void test_long_form_prefers_quiet_boundaries() {
    constexpr int      sample_rate = 16000;
    std::vector<float> pcm(70 * sample_rate, 0.5f);
    std::fill(pcm.begin() + 32 * sample_rate, pcm.begin() + 33 * sample_rate, 0.0f);

    const std::vector<transcribe::canary::CanaryChunkSpan> chunks =
        transcribe::canary::canary_long_form_chunks(pcm.data(), static_cast<int>(pcm.size()), sample_rate);
    CHECK(chunks.size() == 2);
    if (chunks.size() == 2) {
        CHECK(chunks[0].start_sample == 0);
        CHECK(chunks[0].n_samples >= 32 * sample_rate);
        CHECK(chunks[0].n_samples <= 33 * sample_rate);
        CHECK(chunks[1].start_sample == chunks[0].n_samples - sample_rate);
        CHECK(chunks[1].start_sample + chunks[1].n_samples == static_cast<int>(pcm.size()));
    }

    constexpr int                                          odd_sample_rate = 8016;
    std::vector<float>                                     odd_pcm(40 * odd_sample_rate + 1, 0.5f);
    const std::vector<transcribe::canary::CanaryChunkSpan> odd_chunks =
        transcribe::canary::canary_long_form_chunks(odd_pcm.data(), static_cast<int>(odd_pcm.size()), odd_sample_rate);
    CHECK(!odd_chunks.empty());
    if (!odd_chunks.empty()) {
        CHECK(static_cast<int64_t>(odd_chunks.back().start_sample) + odd_chunks.back().n_samples ==
              static_cast<int64_t>(odd_pcm.size()));
    }
}

}  // namespace

int main() {
    test_distinct_tokens();
    test_repeated_token_requires_blank();
    test_invalid_inputs();
    test_aligner_limit_only_applies_to_timestamps();
    test_keep_special_tags_in_aligned_words();
    test_long_form_chunk_schedule();
    test_long_form_token_seam();
    test_long_form_prefers_quiet_boundaries();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::puts("canary timestamp unit: ok");
    return 0;
}
