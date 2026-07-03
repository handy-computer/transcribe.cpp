// utf8_path_unit.cpp - non-ASCII (UTF-8) path handling through the
// public C ABI.
//
// Regression test for Handy issue #1585: on Windows, model loading and
// backend artifact-dir validation failed for any path containing a
// non-ASCII character (e.g. C:\Users\Jerôme\...), because the library's
// ::stat() / ifstream(const std::string &) calls interpreted the UTF-8
// path bytes in the process ANSI code page. The contract
// (include/transcribe.h) is that every path crossing the C ABI is
// UTF-8; src/transcribe-path.h now routes all file access through an
// explicit UTF-8 -> wide conversion on Windows.
//
// The test copies two build-time fixtures into a directory whose name
// contains non-ASCII characters and asserts, via the public API only:
//
//   1. A complete minimal model at the non-ASCII path loads with
//      TRANSCRIBE_OK — covering the existence pre-check, the
//      magic-bytes sniff, gguf_init_from_file, and the tensor-data
//      streaming reopen.
//   2. arch_unknown.gguf at the non-ASCII path returns
//      ERR_UNSUPPORTED_ARCH, not FILE_NOT_FOUND: the file was found
//      and parsed; only the architecture is unknown.
//   3. A missing file inside the non-ASCII directory still returns
//      FILE_NOT_FOUND — the not-found distinction survives the
//      std::filesystem port.
//   4. transcribe_init_backends accepts the existing non-ASCII
//      directory (anything but FILE_NOT_FOUND) and still rejects a
//      missing non-ASCII directory with FILE_NOT_FOUND.
//
// On POSIX, narrow paths are UTF-8 natively, so cases 1-4 pin the
// contract; on Windows they exercise the actual conversion.
//
// The directory name is spelled in escaped UTF-8 bytes, not literal
// characters, so the test is immune to source-charset differences
// across compilers ("Jerôme-日本語").

#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

namespace fs = std::filesystem;

namespace {

// "transcribe-utf8-Jerôme-日本語" in escaped UTF-8 bytes.
constexpr const char * kDirName =
    "transcribe-utf8-Jer\xC3\xB4me-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E";

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void check_load(const std::string & utf8_path, transcribe_status expected, const char * label) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    struct transcribe_model * m = nullptr;

    const transcribe_status st = transcribe_model_load_file(utf8_path.c_str(), &mp, &m);

    if (st != expected) {
        std::fprintf(stderr, "FAIL %s: expected %s, got %s\n", label, transcribe_status_string(expected),
                     transcribe_status_string(st));
        ++g_failures;
    }
    transcribe_model_free(m);
}

}  // namespace

int main() {
    const fs::path fixtures = fs::u8path(TRANSCRIBE_TEST_FIXTURES_DIR);
    const fs::path src_ok   = fixtures / "tokenizer_minimal.gguf";
    const fs::path src_bad  = fixtures / "arch_unknown.gguf";

    // Backstop, same rationale as loader_smoke: the build dependency on
    // the `fixtures` target guarantees these exist; skip cleanly (77)
    // if someone deleted them between build and test.
    std::error_code ec;
    if (!fs::exists(src_ok, ec) || !fs::exists(src_bad, ec)) {
        std::fprintf(stderr,
                     "utf8_path_unit: fixtures missing; regenerate with:\n"
                     "  cmake --build build --target fixtures\n");
        return 77;
    }

    const fs::path dir = fs::temp_directory_path() / fs::u8path(kDirName);
    fs::remove_all(dir, ec);  // leftovers from a previous crashed run
    fs::create_directories(dir);
    fs::copy_file(src_ok, dir / "tokenizer_minimal.gguf", fs::copy_options::overwrite_existing);
    fs::copy_file(src_bad, dir / "arch_unknown.gguf", fs::copy_options::overwrite_existing);

    // C-API strings must be UTF-8 regardless of platform; u8string()
    // guarantees that on Windows where string() would be ANSI.
    const std::string dir_u8 = dir.u8string();

    // 1. Full successful load through the non-ASCII path. This is the
    //    exact failure mode from Handy #1585 (FILE_NOT_FOUND despite the
    //    file existing).
    check_load(dir_u8 + "/tokenizer_minimal.gguf", TRANSCRIBE_OK, "minimal-model");

    // 2. The file must be FOUND and parsed; only its arch is unknown.
    //    A path-encoding regression turns this into FILE_NOT_FOUND.
    check_load(dir_u8 + "/arch_unknown.gguf", TRANSCRIBE_ERR_UNSUPPORTED_ARCH, "unknown-arch");

    // 3. Genuinely missing file inside the non-ASCII dir: the not-found
    //    classification must survive.
    check_load(dir_u8 + "/__missing__.gguf", TRANSCRIBE_ERR_FILE_NOT_FOUND, "missing-file");

    // 4. Backend artifact-dir validation. The directory exists, so
    //    FILE_NOT_FOUND specifically is a path-encoding bug ("...is not
    //    an existing directory" from Handy #1585). We do not assert OK:
    //    a dynamic-backend build pointed at a module-less directory
    //    legitimately returns ERR_BACKEND.
    const transcribe_status init_st = transcribe_init_backends(dir_u8.c_str());
    CHECK(init_st != TRANSCRIBE_ERR_FILE_NOT_FOUND);

    CHECK(transcribe_init_backends((dir_u8 + "/__missing_dir__").c_str()) == TRANSCRIBE_ERR_FILE_NOT_FOUND);

    fs::remove_all(dir, ec);

    if (g_failures > 0) {
        std::fprintf(stderr, "utf8_path_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "utf8_path_unit: ok\n");
    return EXIT_SUCCESS;
}
