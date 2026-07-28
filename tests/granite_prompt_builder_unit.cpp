// granite_prompt_builder_unit.cpp - table-driven tests for the granite
// user-turn instruction selector (arch/granite/granite.h::
// select_granite_instruction). Pure host-side: no model, no GGUF, no
// tokenizer. Granite's keyword-biasing surface form is load-bearing (a
// paraphrase is silently ignored by the model and biasing does nothing), and
// the stem is DISTINCT per variant × task, so these tests pin the exact
// instruction string for every combination — the shape of bug that already
// slipped through once (base-2b appending Keywords to the punctuated-ASR stem
// instead of the trained "transcribe the speech to text." stem).

#include "arch/granite/diarize.h"
#include "arch/granite/granite.h"
#include "transcribe.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Compare the selected instruction against an expected string. `diarize_on` is
// the pre-resolved diarize_requested() result the decode path passes in.
void expect_instruction(const char *                  variant,
                        const transcribe_run_params * params,
                        bool                          diarize_on,
                        const std::string &           want) {
    std::string             got;
    const transcribe_status st = transcribe::granite::select_granite_instruction(variant, params, diarize_on, got);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL %s: select returned status %d (expected OK)\n", variant, (int) st);
        ++g_failures;
        return;
    }
    if (got != want) {
        std::fprintf(stderr, "FAIL %s:\n  got:  '%s'\n  want: '%s'\n", variant, got.c_str(), want.c_str());
        ++g_failures;
    }
}

void expect_error(const char * variant, const transcribe_run_params * params, bool diarize_on, transcribe_status want) {
    std::string             got;
    const transcribe_status st = transcribe::granite::select_granite_instruction(variant, params, diarize_on, got);
    CHECK(st == want);
}

// Per-variant plain-ASR instructions (the leading space on -plus is a real BPE
// nuance: " can" tokenizes differently from "can").
constexpr const char * k_base_2b_asr = "transcribe the speech with proper punctuation and capitalization.";
constexpr const char * k_plus_asr    = " can you transcribe the speech into a written format?";
constexpr const char * k_1b_asr      = "can you transcribe the speech into a written format?";

void test_null_params_defaults_per_variant() {
    // params == nullptr => plain-ASR default, one per variant. No task, no KWB.
    expect_instruction("granite-speech-4.1-2b", nullptr, false, k_base_2b_asr);
    expect_instruction("granite-speech-4.1-2b-plus", nullptr, false, k_plus_asr);
    expect_instruction("granite-4.0-1b-speech", nullptr, false, k_1b_asr);
    // An unrecognized variant falls through to the 1b/plain-ASR default.
    expect_instruction("granite-speech-99", nullptr, false, k_1b_asr);
}

void test_plain_asr_no_hotwords() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);  // task = TRANSCRIBE, timestamps = AUTO, no hotwords
    expect_instruction("granite-speech-4.1-2b", &p, false, k_base_2b_asr);
    expect_instruction("granite-speech-4.1-2b-plus", &p, false, k_plus_asr);
    expect_instruction("granite-4.0-1b-speech", &p, false, k_1b_asr);
}

void test_asr_hotwords_distinct_stem_per_variant() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.hotwords = "kubernetes, gRPC";

    // base-2b swaps to the trained ASR-KWB stem (NOT the punctuated-ASR stem).
    expect_instruction("granite-speech-4.1-2b", &p, false, "transcribe the speech to text. Keywords: kubernetes, gRPC");
    // -plus/1b append to their plain-ASR prompt verbatim (leading space kept).
    expect_instruction("granite-speech-4.1-2b-plus", &p, false,
                       " can you transcribe the speech into a written format? Keywords: kubernetes, gRPC");
    expect_instruction("granite-4.0-1b-speech", &p, false,
                       "can you transcribe the speech into a written format? Keywords: kubernetes, gRPC");
}

void test_translate_no_hotwords() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.task            = TRANSCRIBE_TASK_TRANSLATE;
    p.target_language = "fr";
    // Same AST prompt across variants (capabilities gate which variants may run).
    expect_instruction("granite-speech-4.1-2b", &p, false, "can you translate the speech into French?");
    expect_instruction("granite-4.0-1b-speech", &p, false, "can you translate the speech into French?");
    // Name / alias inputs resolve identically.
    p.target_language = "German";
    expect_instruction("granite-speech-4.1-2b", &p, false, "can you translate the speech into German?");
}

void test_translate_hotwords_base_2b_stem() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.task            = TRANSCRIBE_TASK_TRANSLATE;
    p.target_language = "es";
    p.hotwords        = "Mori, OTEL";
    // base-2b AST+KWB uses the trained "translate the speech to <lang>." stem.
    expect_instruction("granite-speech-4.1-2b", &p, false, "translate the speech to Spanish. Keywords: Mori, OTEL");
    // 1b has no attested AST-KWB stem: append to its plain translate prompt.
    expect_instruction("granite-4.0-1b-speech", &p, false,
                       "can you translate the speech into Spanish? Keywords: Mori, OTEL");
}

void test_translate_requires_valid_target() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.task = TRANSCRIBE_TASK_TRANSLATE;

    // Missing target language.
    p.target_language = nullptr;
    expect_error("granite-speech-4.1-2b", &p, false, TRANSCRIBE_ERR_INVALID_ARG);
    p.target_language = "";
    expect_error("granite-speech-4.1-2b", &p, false, TRANSCRIBE_ERR_INVALID_ARG);

    // Unadvertised target language.
    p.target_language = "klingon";
    expect_error("granite-speech-4.1-2b", &p, false, TRANSCRIBE_ERR_INVALID_ARG);
}

void test_word_timestamps() {
    static const char * k_ts_instruction =
        " Timestamps: Transcribe the speech. After each word, add a timestamp tag "
        "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]";

    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
    expect_instruction("granite-speech-4.1-2b-plus", &p, false, k_ts_instruction);

    // Timestamps + diarize are mutually exclusive tasks upstream.
    expect_error("granite-speech-4.1-2b-plus", &p, /*diarize_on=*/true, TRANSCRIBE_ERR_INVALID_ARG);

    // Timestamps + hotwords: unattested combination, warns but appends best-effort.
    p.hotwords = "gRPC";
    expect_instruction("granite-speech-4.1-2b-plus", &p, false, std::string(k_ts_instruction) + " Keywords: gRPC");
}

void test_diarize_speaker_attribution() {
    const std::string saa = transcribe::granite::k_saa_instruction;

    transcribe_run_params p;
    transcribe_run_params_init(&p);
    // diarize_on resolved true by the caller (feature advertised + diarize=ON).
    expect_instruction("granite-speech-4.1-2b-plus", &p, true, saa);

    // Diarize + hotwords: unattested, warns but appends the clause best-effort.
    p.hotwords = "kubernetes";
    expect_instruction("granite-speech-4.1-2b-plus", &p, true, saa + " Keywords: kubernetes");
}

void test_hotwords_gated_by_struct_size() {
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.hotwords    = "kubernetes";
    // A caller that predates the hotwords field (struct_size stops before it) is
    // treated as "no hint" — base-2b keeps its plain-ASR stem, no Keywords clause.
    p.struct_size = offsetof(transcribe_run_params, hotwords);
    expect_instruction("granite-speech-4.1-2b", &p, false, k_base_2b_asr);
}

}  // namespace

int main() {
    test_null_params_defaults_per_variant();
    test_plain_asr_no_hotwords();
    test_asr_hotwords_distinct_stem_per_variant();
    test_translate_no_hotwords();
    test_translate_hotwords_base_2b_stem();
    test_translate_requires_valid_target();
    test_word_timestamps();
    test_diarize_speaker_attribution();
    test_hotwords_gated_by_struct_size();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
