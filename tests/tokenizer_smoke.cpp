// tokenizer_smoke.cpp - end-to-end smoke for tokenizer ingest through
// the public C ABI, plus the v2/v3 capability KV split.
//
// This test goes one level deeper than loader_smoke: it not only loads
// a GGUF, it asserts that the resulting model has a populated
// transcribe::Tokenizer with the expected vocabulary, special-token
// ids, and decode behavior. It also checks the public model
// introspection accessors (variant, arch, capabilities) come back with
// the values the Parakeet handler should write in 2B.
//
// Two fixtures are exercised:
//
//   tokenizer_minimal.gguf     v2 layout: stt.variant = "tdt-0.6b-v2",
//                              no capability KV, no general.languages.
//                              Pins the family-default code path:
//                              supports_language_detect == false,
//                              n_languages == 0, languages == nullptr
//                              ("information gap, not a claim").
//
//   tokenizer_minimal_v3.gguf  v3 layout: stt.variant = "tdt-0.6b-v3",
//                              stt.capability.lang_detect = true,
//                              general.languages = [en,de,fr,es].
//                              Same toy 16-token vocabulary, same
//                              special token ids. Pins the
//                              "v2 vs v3 are the same code path"
//                              contract: only the descriptive metadata
//                              and the capability/language KV differ;
//                              the loader does not branch on variant.
//
// The test reaches into internal headers (transcribe-model.h,
// transcribe-tokenizer.h, arch/parakeet/parakeet.h) to pull a
// const Tokenizer * out of the model. The public C ABI does not (and
// should not) expose tokenizer details directly in 2B; the test target
// has src/ on its PRIVATE include path so this stays internal.

#include "transcribe.h"

#include "arch/parakeet/parakeet.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#  error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace {

const std::string g_fixtures_dir = TRANSCRIBE_TEST_FIXTURES_DIR;

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",                        \
                         __FILE__, __LINE__, #cond);                        \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                  \
    do {                                                                    \
        const std::string _av = (a);                                        \
        const std::string _bv = (b);                                        \
        if (_av != _bv) {                                                   \
            std::fprintf(stderr,                                            \
                         "FAIL %s:%d: \"%s\" != \"%s\"\n",                  \
                         __FILE__, __LINE__,                                \
                         _av.c_str(), _bv.c_str());                         \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

// Load a fixture and assert: the model is non-null, arch == parakeet,
// variant matches, backend is empty (no runtime backend bound in 2B),
// caps->native_sample_rate == 16000, caps->supports_translate == false.
// Returns the loaded model on success or nullptr on failure (with
// g_failures incremented).
struct transcribe_model * load_or_fail(const char * fixture_name,
                                       const char * expected_variant)
{
    const std::string p = g_fixtures_dir + "/" + fixture_name;
    transcribe_model_load_params mp = transcribe_model_load_default_params();
    struct transcribe_model * model = nullptr;
    const transcribe_status st = transcribe_model_load_file(p.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "FAIL load %s: expected OK, got %s\n",
                     fixture_name, transcribe_status_string(st));
        ++g_failures;
        return nullptr;
    }
    if (model == nullptr) {
        std::fprintf(stderr,
                     "FAIL load %s: model pointer not set\n", fixture_name);
        ++g_failures;
        return nullptr;
    }
    CHECK_STR_EQ(transcribe_model_arch_string(model),    "parakeet");
    CHECK_STR_EQ(transcribe_model_variant_string(model), expected_variant);
    // After phase 4 step 1 the loader binds a runtime backend; the
    // exact label is platform-dependent (Metal on Apple Silicon, CPU
    // elsewhere or on Metal init failure). Just assert non-empty.
    {
        const std::string backend = transcribe_model_backend(model);
        if (backend.empty()) {
            std::fprintf(stderr,
                         "FAIL %s: backend is empty after step 1\n",
                         fixture_name);
            ++g_failures;
        }
    }

    transcribe_capabilities caps_buf = TRANSCRIBE_CAPABILITIES_INIT;
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(caps->native_sample_rate == 16000);
        CHECK(caps->supports_translate == false);
    }
    return model;
}

// Assert the toy 16-token vocabulary round-tripped intact and the
// SentencePiece-style decode behaves as documented. Both fixtures
// share this vocabulary; calling this once per fixture also pins the
// "loader does not corrupt the tokenizer based on capability KV"
// invariant.
void check_toy_vocab(const transcribe::Tokenizer * tok) {
    CHECK(tok->n_tokens() == 16);

    // Spot-check id -> piece. The bytes literal for ▁ is U+2581 in
    // UTF-8 (0xE2 0x96 0x81); the test source uses the actual code
    // point so the file stays UTF-8 throughout.
    CHECK_STR_EQ(tok->token(0),  "<unk>");
    CHECK_STR_EQ(tok->token(1),  "<s>");
    CHECK_STR_EQ(tok->token(2),  "</s>");
    CHECK_STR_EQ(tok->token(3),  "\xE2\x96\x81hello");
    CHECK_STR_EQ(tok->token(4),  "\xE2\x96\x81world");
    CHECK_STR_EQ(tok->token(11), "\xE2\x96\x81the");
    CHECK_STR_EQ(tok->token(15), "<blank>");

    // Out-of-range id returns the empty-string sentinel rather than
    // reading off the end of the vector.
    CHECK_STR_EQ(tok->token(-1),    "");
    CHECK_STR_EQ(tok->token(10000), "");

    // Reverse lookup.
    CHECK(tok->find("<unk>")               == 0);
    CHECK(tok->find("\xE2\x96\x81hello")   == 3);
    CHECK(tok->find("\xE2\x96\x81world")   == 4);
    CHECK(tok->find("definitely not here") == -1);

    // Special token ids match the fixture KV.
    CHECK(tok->unk_id()   == 0);
    CHECK(tok->bos_id()   == 1);
    CHECK(tok->eos_id()   == 2);
    CHECK(tok->blank_id() == 15);

    CHECK_STR_EQ(tok->model_type(), "unigram");

    // Decode: SentencePiece word-boundary marker becomes ASCII space.
    // Sequence: ▁hello ▁world  ->  " hello world".
    {
        const int ids[] = {3, 4};
        CHECK_STR_EQ(tok->decode(ids, 2), " hello world");
    }
    // Continuation piece in the middle: ▁hello s ▁world -> " hellos world".
    {
        const int ids[] = {3, 8, 4};
        CHECK_STR_EQ(tok->decode(ids, 3), " hellos world");
    }
    // First piece has no marker: ed ▁the -> "ed the".
    {
        const int ids[] = {9, 11};
        CHECK_STR_EQ(tok->decode(ids, 2), "ed the");
    }
    // Out-of-range ids are skipped silently.
    {
        const int ids[] = {3, -1, 4};
        CHECK_STR_EQ(tok->decode(ids, 3), " hello world");
    }
    // Empty / null inputs return an empty string.
    CHECK_STR_EQ(tok->decode(nullptr, 0), "");
    {
        const int ids[] = {3};
        CHECK_STR_EQ(tok->decode(ids, 0), "");
    }
}

// ---------------------------------------------------------------------------
// v2: family-default code path
// ---------------------------------------------------------------------------
//
// No capability KV, no general.languages. Asserts the family defaults
// stand: language detect off, languages list is the documented
// information-gap state.
void test_v2_fixture() {
    struct transcribe_model * model =
        load_or_fail("tokenizer_minimal.gguf", "tdt-0.6b-v2");
    if (model == nullptr) return;

    transcribe_capabilities caps_buf = TRANSCRIBE_CAPABILITIES_INIT;
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK(caps->supports_language_detect == false);
        CHECK(caps->supports_streaming       == false);
        // Information gap, not a claim. Documented in
        // arch/parakeet/model.cpp.
        CHECK(caps->n_languages == 0);
        CHECK(caps->languages   == nullptr);
    }

    const transcribe::Tokenizer * tok = model->tokenizer();
    CHECK(tok != nullptr);
    if (tok != nullptr) {
        check_toy_vocab(tok);
    }

    transcribe_model_free(model);
}

// ---------------------------------------------------------------------------
// v3: capability KV + languages KV overrides
// ---------------------------------------------------------------------------
//
// Same code path as v2 — only the descriptive variant string and the
// KV the loader reads differ. The tokenizer is identical (proves the
// loader does not corrupt or branch the tokenizer based on capability
// KV).
void test_v3_fixture() {
    struct transcribe_model * model =
        load_or_fail("tokenizer_minimal_v3.gguf", "tdt-0.6b-v3");
    if (model == nullptr) return;

    transcribe_capabilities caps_buf = TRANSCRIBE_CAPABILITIES_INIT;
    const bool caps_ok =
        transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        // Capability KV overrode the family default.
        CHECK(caps->supports_language_detect == true);
        // Streaming KV not present in the fixture, so the family
        // default (false) stands.
        CHECK(caps->supports_streaming       == false);
        // general.languages was a four-element string array.
        CHECK(caps->n_languages == 4);
        CHECK(caps->languages   != nullptr);
        if (caps->languages != nullptr && caps->n_languages == 4) {
            CHECK_STR_EQ(caps->languages[0], "en");
            CHECK_STR_EQ(caps->languages[1], "de");
            CHECK_STR_EQ(caps->languages[2], "fr");
            CHECK_STR_EQ(caps->languages[3], "es");
        }
    }

    const transcribe::Tokenizer * tok = model->tokenizer();
    CHECK(tok != nullptr);
    if (tok != nullptr) {
        // Identical vocabulary to v2 — confirms the same code path.
        check_toy_vocab(tok);
    }

    transcribe_model_free(model);
}

// ---------------------------------------------------------------------------
// n_threads validation in transcribe_session_init (Finding 3)
// ---------------------------------------------------------------------------
//
// Negative n_threads is undefined input per the public header (0 means
// "library picks", positive means "use this many"). The central
// dispatcher rejects it before the family handler runs.
void test_n_threads_validation() {
    struct transcribe_model * model =
        load_or_fail("tokenizer_minimal.gguf", "tdt-0.6b-v2");
    if (model == nullptr) return;

    transcribe_session_params cp = transcribe_session_default_params();
    cp.n_threads = -1;
    struct transcribe_session * ctx = (struct transcribe_session *)0xdeadbeef;
    const transcribe_status st = transcribe_session_init(model, &cp, &ctx);
    if (st != TRANSCRIBE_ERR_INVALID_ARG) {
        std::fprintf(stderr,
                     "FAIL n_threads<0: expected INVALID_ARG, got %s\n",
                     transcribe_status_string(st));
        ++g_failures;
    }
    // Contract: out_ctx is cleared on every non-OK return.
    if (ctx != nullptr) {
        std::fprintf(stderr, "FAIL n_threads<0: out_ctx not cleared\n");
        ++g_failures;
    }

    // Sanity: n_threads == 0 (library default) succeeds.
    cp.n_threads = 0;
    ctx = nullptr;
    const transcribe_status st_ok = transcribe_session_init(model, &cp, &ctx);
    if (st_ok != TRANSCRIBE_OK) {
        std::fprintf(stderr,
                     "FAIL n_threads=0: expected OK, got %s\n",
                     transcribe_status_string(st_ok));
        ++g_failures;
    }
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL n_threads=0: ctx pointer not set\n");
        ++g_failures;
    }
    transcribe_session_free(ctx);

    transcribe_model_free(model);
}

} // namespace

int main() {
    // Build-system backstop. Same rationale as loader_smoke: the
    // `fixtures` target should make this branch unreachable in normal
    // use, but we exit 77 (CTest "skipped") if someone deletes the
    // files between build and test.
    static const char * const required_fixtures[] = {
        "tokenizer_minimal.gguf",
        "tokenizer_minimal_v3.gguf",
    };
    bool all_present = true;
    for (const char * name : required_fixtures) {
        const std::string p = g_fixtures_dir + "/" + name;
        if (!file_exists(p)) {
            std::fprintf(stderr,
                         "tokenizer_smoke: fixture not found: %s\n", p.c_str());
            all_present = false;
        }
    }
    if (!all_present) {
        std::fprintf(stderr,
                     "tokenizer_smoke: regenerate with:\n"
                     "  cmake --build build --target fixtures\n");
        return 77;
    }

    test_v2_fixture();
    test_v3_fixture();
    test_n_threads_validation();

    if (g_failures > 0) {
        std::fprintf(stderr, "tokenizer_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "tokenizer_smoke: ok\n");
    return EXIT_SUCCESS;
}
