// transcribe-path.h - UTF-8 path handling shared by every file-access site.
//
// Contract: every path crossing the public C ABI is UTF-8 (see
// include/transcribe.h; that is what the FFI bindings send, and what
// path_for_c_api() in transcribe.cpp produces). On POSIX the narrow
// char* file APIs consume UTF-8 bytes directly, so plain std::string
// paths work. On Windows the narrow APIs (::stat, fopen,
// ifstream(const std::string &)) interpret narrow strings in the
// process ANSI code page, NOT UTF-8 — a path containing e.g. "ô"
// fails with ENOENT even though the file exists (Handy issue #1585).
// Every stat/open on the release model-load path must therefore go
// through the helpers here, which route file access through
// std::filesystem::path with an explicit UTF-8 -> native-wide
// conversion on Windows.
//
// Deliberately NOT converted: dev-only I/O — the dump writers in
// transcribe-debug.cpp and the TRANSCRIBE_ENABLE_VALIDATION_HOOKS
// sidecar readers in src/arch/*/model.cpp. Those read/write paths that
// never cross the public ABI. Don't copy file-access idioms from those
// files into release-path code; anything a user-supplied path reaches
// goes through these helpers.

#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace transcribe {

// Build a std::filesystem::path from UTF-8 bytes. On Windows this
// converts to the native wide encoding; elsewhere the narrow encoding
// IS UTF-8 and this is a passthrough. (fs::u8path is the C++17
// spelling; it is deprecated-but-present under C++20.)
inline std::filesystem::path path_from_utf8(const char * utf8) {
#if defined(_WIN32)
    return std::filesystem::u8path(utf8);
#else
    return std::filesystem::path(utf8);
#endif
}

inline std::filesystem::path path_from_utf8(const std::string & utf8) {
    return path_from_utf8(utf8.c_str());
}

// Existence pre-check whose ONLY job is to distinguish "the file is
// not at this path" from every other reason a subsequent open might
// fail.
//
// We deliberately do not check for a regular file. If the path points
// at a directory (or a fifo, or a socket), the file IS at that path —
// it's just not usable, and the caller's open will surface that as a
// more accurate error than FILE_NOT_FOUND.
//
// We also deliberately only report absence on the not-found outcome
// (POSIX ENOENT / ENOTDIR and their Windows equivalents, which
// fs::status maps to file_type::not_found). Other failures (EACCES,
// ENAMETOOLONG, ELOOP, EIO, ...) do NOT conclusively prove the file
// is missing — they prove we can't reach it — so they return true and
// the caller's open reports the real error.
inline bool path_is_present(const char * utf8_path) {
    std::error_code ec;
    return std::filesystem::status(path_from_utf8(utf8_path), ec).type() != std::filesystem::file_type::not_found;
}

}  // namespace transcribe
