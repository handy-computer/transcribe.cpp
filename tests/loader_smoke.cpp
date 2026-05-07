// loader_smoke.cpp - fixture-driven smoke test for the GGUF loader and
// per-family arch dispatch.
//
// This test pins five contracts that the public C ABI must honor:
//
//   1. Missing path                  -> TRANSCRIBE_ERR_FILE_NOT_FOUND
//   2. Directory at the requested path
//      (i.e. stat() succeeds but the
//      file is not openable as a GGUF) -> TRANSCRIBE_ERR_GGUF
//                                         (NOT FILE_NOT_FOUND — the file
//                                         IS at that path, it's just
//                                         not a usable GGUF)
//   3. Corrupt GGUF magic            -> TRANSCRIBE_ERR_GGUF
//   4. Unrecognized architecture     -> TRANSCRIBE_ERR_UNSUPPORTED_ARCH
//   5. Recognized architecture, GGUF
//      missing the tokenizer payload
//      that load() now requires       -> TRANSCRIBE_ERR_GGUF
//
// The success-path "load actually produces a model" contract is covered
// by the separate transcribe_tokenizer_smoke test, which uses
// tokenizer_minimal.gguf.
//
// The fixtures live next to this file in tests/fixtures/ and are emitted
// by make_gguf_fixtures.py via uv. CMake wires the generator into the
// build graph as a hard build-time dependency of this test target, so
// the .gguf files are guaranteed to be present and current relative to
// the generator script whenever this binary runs. When uv is not on
// PATH, the test target is not registered at all (see
// tests/CMakeLists.txt) — there is no "uv missing, run anyway" path.
//
// The runtime existence check below is therefore a BACKSTOP, not the
// primary mechanism: it catches the pathological case where someone
// deletes the fixtures between build and test, and produces a clear
// regeneration message via SKIP_RETURN_CODE 77. Under normal use it
// should never fire.
//
// We deliberately use the public C API (transcribe.h) only — the test
// must not poke at internal headers, so it locks the contract from the
// caller's perspective.

#include "transcribe.h"

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

bool file_exists(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

// Backstop: under normal use the build graph guarantees these files are
// present and current. We re-check at runtime only so the pathological
// "fixtures deleted between build and test" case produces a clear
// regeneration hint instead of a confusing test failure.
bool fixtures_present() {
    static const char * const required[] = {
        "arch_parakeet.gguf",
        "arch_sensevoice.gguf",
        "arch_funasr_nano.gguf",
        "arch_unknown.gguf",
        "corrupt_magic.gguf",
    };
    bool ok = true;
    for (const char * name : required) {
        const std::string p = g_fixtures_dir + "/" + name;
        if (!file_exists(p)) {
            std::fprintf(stderr,
                         "loader_smoke: fixture not found: %s\n",
                         p.c_str());
            ok = false;
        }
    }
    if (!ok) {
        std::fprintf(stderr,
                     "loader_smoke: the build should have generated these "
                     "via the `fixtures` target. Regenerate with:\n"
                     "  cmake --build build --target fixtures\n"
                     "  (or, by hand: uv run tests/fixtures/make_gguf_fixtures.py)\n");
    }
    return ok;
}

void check_load(const char *      fixture_name,
                transcribe_status expected) {
    const std::string p = g_fixtures_dir + "/" + fixture_name;

    transcribe_model_params mp = transcribe_model_default_params();
    // Sentinel so we can verify the loader clears out_model on every
    // failure path, not just on success.
    struct transcribe_model * m = (struct transcribe_model *)0xdeadbeef;

    const transcribe_status st = transcribe_model_load_file(p.c_str(), &mp, &m);

    if (st != expected) {
        std::fprintf(stderr,
                     "FAIL %s: expected %s, got %s\n",
                     fixture_name,
                     transcribe_status_string(expected),
                     transcribe_status_string(st));
        ++g_failures;
    }

    // The contract from include/transcribe.h: on any non-OK status,
    // *out_model must be set to NULL. The fixtures used here only
    // exercise non-OK paths; the success path lives in
    // transcribe_tokenizer_smoke.
    if (m != nullptr) {
        std::fprintf(stderr,
                     "FAIL %s: out_model not cleared (got %p)\n",
                     fixture_name, (void *)m);
        ++g_failures;
    }
}

// Smoke that does not depend on the fixtures being on disk: a path that
// definitely does not exist must surface as FILE_NOT_FOUND, distinct
// from ERR_GGUF. The leading slash + sentinel-y name avoids any chance
// of colliding with a real file the developer happens to have.
void test_missing_path() {
    transcribe_model_params mp = transcribe_model_default_params();
    struct transcribe_model * m = (struct transcribe_model *)0xdeadbeef;

    const transcribe_status st = transcribe_model_load_file(
        "/__transcribe_loader_smoke_missing__.gguf", &mp, &m);

    if (st != TRANSCRIBE_ERR_FILE_NOT_FOUND) {
        std::fprintf(stderr,
                     "FAIL missing-path: expected FILE_NOT_FOUND, got %s\n",
                     transcribe_status_string(st));
        ++g_failures;
    }
    if (m != nullptr) {
        std::fprintf(stderr, "FAIL missing-path: out_model not cleared\n");
        ++g_failures;
    }
}

// Pin the new (post-Finding-2) precision contract: a directory at the
// requested path must NOT be reported as FILE_NOT_FOUND. The file IS at
// that path — it's just not a usable GGUF, which is exactly what
// TRANSCRIBE_ERR_GGUF means. We use the fixtures directory itself,
// which is guaranteed to exist by the build dependency on `fixtures`.
void test_directory_path() {
    transcribe_model_params mp = transcribe_model_default_params();
    struct transcribe_model * m = (struct transcribe_model *)0xdeadbeef;

    const transcribe_status st = transcribe_model_load_file(
        g_fixtures_dir.c_str(), &mp, &m);

    if (st != TRANSCRIBE_ERR_GGUF) {
        std::fprintf(stderr,
                     "FAIL directory-path: expected ERR_GGUF, got %s\n",
                     transcribe_status_string(st));
        ++g_failures;
    }
    if (m != nullptr) {
        std::fprintf(stderr, "FAIL directory-path: out_model not cleared\n");
        ++g_failures;
    }
}

} // namespace

int main() {
    test_missing_path();

    if (!fixtures_present()) {
        std::fprintf(stderr,
                     "loader_smoke: skipping fixture-dependent cases\n");
        // 77 -> CTest "skipped" via SKIP_RETURN_CODE. This branch is a
        // backstop; the build dependency on `fixtures` should make sure
        // we never reach it under normal use.
        return 77;
    }

    test_directory_path();

    // arch_parakeet.gguf has the architecture KV but no tokenizer
    // payload. The Parakeet handler now requires tokenizer.ggml.* and
    // surfaces the missing payload as TRANSCRIBE_ERR_GGUF — the loader
    // dispatched correctly, the family handler then rejected the file.
    check_load("arch_parakeet.gguf",   TRANSCRIBE_ERR_GGUF);
    // Same shape for sensevoice / funasr_nano: the arch is in the
    // dispatch table, the handler runs, then rejects on the missing
    // hparam / tensor payload. Distinguishes "registered family but
    // bad GGUF" from "unknown architecture".
    check_load("arch_sensevoice.gguf",  TRANSCRIBE_ERR_GGUF);
    check_load("arch_funasr_nano.gguf", TRANSCRIBE_ERR_GGUF);
    check_load("arch_unknown.gguf",    TRANSCRIBE_ERR_UNSUPPORTED_ARCH);
    check_load("corrupt_magic.gguf",   TRANSCRIBE_ERR_GGUF);

    if (g_failures > 0) {
        std::fprintf(stderr, "loader_smoke: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "loader_smoke: ok\n");
    return EXIT_SUCCESS;
}
