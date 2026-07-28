// moss_prompt_builder_unit.cpp - unit tests for MOSS hotword prompt assembly
// (arch/moss/moss.h::build_prompt_tokens) and the shared guarded hotwords
// accessor (transcribe-arch.h::run_params_hotwords). Pure host-side: no model,
// no GGUF. The invariants under test:
//   * empty hotword_ids => prompt is byte-identical to the unbiased assembly
//     (prefix + audio_span + baked suffix);
//   * non-empty hotword_ids are inserted exactly at the baked suffix's
//     body->close split, leaving prefix / audio span / suffix tail untouched;
//   * audio_pad positions never shift when hotwords are inserted (they precede
//     the suffix);
//   * run_params_hotwords honors struct_size gating and NULL/empty as "no hint".

#include "arch/moss/moss.h"
#include "transcribe-arch.h"
#include "transcribe.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

// Minimal hparams for deterministic, model-free prompt assembly. Time markers
// disabled so the audio span is exactly `audio_seq_len` copies of audio_token_id
// (build_audio_span short-circuit), and the suffix carries an explicit
// body->close split (index 3 = the baked <|im_end|> that closes the turn).
transcribe::moss::MossHParams make_hparams() {
    transcribe::moss::MossHParams hp;
    hp.audio_token_id       = 7;
    hp.enable_time_marker   = false;
    hp.prompt_prefix_tokens = { 100, 101 };
    // suffix = tokens("…。") [200,201,202] + close [999(<|im_end|>),300].
    hp.prompt_suffix_tokens = { 200, 201, 202, 999, 300 };
    hp.prompt_suffix_split  = 3;  // insert hotwords after 202, before 999
    return hp;
}

void test_no_hotwords_byte_identical() {
    const transcribe::moss::MossHParams hp = make_hparams();
    std::vector<int32_t>                ids, pos;
    transcribe::moss::build_prompt_tokens(hp, /*audio_seq_len=*/3, /*hotword_ids=*/{}, ids, pos);

    const std::vector<int32_t> expect = { 100, 101, 7, 7, 7, 200, 201, 202, 999, 300 };
    CHECK(ids == expect);
    // Audio pads sit right after the 2-token prefix.
    const std::vector<int32_t> expect_pos = { 2, 3, 4 };
    CHECK(pos == expect_pos);
}

void test_hotwords_inserted_at_split() {
    const transcribe::moss::MossHParams hp = make_hparams();
    std::vector<int32_t>                ids, pos;
    const std::vector<int32_t>          hotwords = { 5000, 5001 };
    transcribe::moss::build_prompt_tokens(hp, /*audio_seq_len=*/3, hotwords, ids, pos);

    // prefix | span | suffix[0,3) | hotwords | suffix[3,end)
    const std::vector<int32_t> expect = { 100, 101, 7, 7, 7, 200, 201, 202, 5000, 5001, 999, 300 };
    CHECK(ids == expect);
    // Inserting hotwords must NOT move the audio-pad positions (they precede
    // the suffix), so decode still scatters features to the same slots.
    const std::vector<int32_t> expect_pos = { 2, 3, 4 };
    CHECK(pos == expect_pos);
}

void test_split_at_suffix_size_disables_hotwords() {
    // Unexpected: no <|im_end|> found at load => split == suffix.size() (no
    // body->close boundary). Hotwords are disabled rather than appended after
    // the turn-closing tokens; the baked prompt is emitted verbatim.
    transcribe::moss::MossHParams hp = make_hparams();
    hp.prompt_suffix_split           = 999;  // >> suffix size; must clamp to size
    std::vector<int32_t>       ids, pos;
    const std::vector<int32_t> hotwords = { 5000 };
    transcribe::moss::build_prompt_tokens(hp, /*audio_seq_len=*/1, hotwords, ids, pos);

    // Baked prompt only, hotwords dropped: prefix | span | full suffix.
    const std::vector<int32_t> expect = { 100, 101, 7, 200, 201, 202, 999, 300 };
    CHECK(ids == expect);
}

void test_run_params_hotwords_guard() {
    // NULL params.
    CHECK(transcribe::run_params_hotwords(nullptr) == nullptr);

    transcribe_run_params params;
    transcribe_run_params_init(&params);

    // Unset (init zeroed the pointer) => no hint.
    CHECK(transcribe::run_params_hotwords(&params) == nullptr);

    // Empty string => no hint.
    params.hotwords = "";
    CHECK(transcribe::run_params_hotwords(&params) == nullptr);

    // Valid string => returned verbatim.
    params.hotwords = "kubernetes, gRPC";
    const char * hw = transcribe::run_params_hotwords(&params);
    CHECK(hw != nullptr && std::string(hw) == "kubernetes, gRPC");

    // struct_size too small to include the tail field => treated as absent even
    // though the pointer is set (old caller that predates the field).
    params.struct_size = offsetof(transcribe_run_params, hotwords);
    CHECK(transcribe::run_params_hotwords(&params) == nullptr);
}

}  // namespace

int main() {
    test_no_hotwords_byte_identical();
    test_hotwords_inserted_at_split();
    test_split_at_suffix_size_disables_hotwords();
    test_run_params_hotwords_guard();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
