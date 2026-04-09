// transcribe-loader.cpp - implementation of the GGUF loader.
//
// See transcribe-loader.h for the contract. This file is intentionally
// architecture-agnostic: it knows nothing about Parakeet, Whisper, or any
// other family. It only reads enough KV to identify which family handler
// the per-arch dispatch should hand the file to.

#include "transcribe-loader.h"

#include "transcribe-meta.h"

#include "gguf.h"

#include <sys/stat.h>

#include <cerrno>

namespace transcribe {

Loader::~Loader() {
    if (gguf_ != nullptr) {
        gguf_free(gguf_);
        gguf_ = nullptr;
    }
}

gguf_context * Loader::release_gguf() {
    gguf_context * out = gguf_;
    gguf_ = nullptr;
    return out;
}

namespace {

// Narrow stat-based pre-check whose ONLY job is to distinguish
// "the file is not at this path" from every other reason
// gguf_init_from_file might return nullptr.
//
// We deliberately do not check S_ISREG here. If stat() succeeds and the
// path points at a directory (or a fifo, or a socket), the file IS at
// that path — it's just not a usable GGUF. fopen() inside ggml will
// return EISDIR / etc. and we will surface that as TRANSCRIBE_ERR_GGUF,
// which is strictly more accurate than collapsing it into FILE_NOT_FOUND.
//
// We also deliberately only short-circuit on ENOENT / ENOTDIR. Other
// stat() failures (EACCES, ENAMETOOLONG, ELOOP, EIO, ...) do NOT
// conclusively prove the file is missing — they prove we can't reach
// it. Returning FILE_NOT_FOUND for those would be wrong. We let
// gguf_init_from_file try and surface whatever it returns as ERR_GGUF.
//
// Returns true if the path either exists OR if we cannot prove it
// doesn't (in which case the caller should hand off to
// gguf_init_from_file unchanged).
bool path_is_present(const char * path) {
    struct stat st {};
    if (::stat(path, &st) == 0) {
        return true;
    }
    // ENOENT: the named file does not exist.
    // ENOTDIR: a non-final component of the path is not a directory,
    //          which is the same observable failure mode as "the file
    //          is not at this path."
    if (errno == ENOENT || errno == ENOTDIR) {
        return false;
    }
    return true;
}

} // namespace

transcribe_status Loader::open(const char * path) {
    if (path == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    path_ = path;

    if (!path_is_present(path)) {
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }

    // Header-only inspection: do not allocate ggml tensors. With ctx=null
    // gguf_init_from_file skips the entire tensor-allocation block; with
    // no_alloc=true any future code path that does pass a ctx will not
    // read the data blob.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;

    gguf_ = gguf_init_from_file(path, init_params);
    if (gguf_ == nullptr) {
        // ggml has already logged a structured error via GGML_LOG_ERROR
        // (corrupt magic, version mismatch, truncated header, IO error
        // after the existence check, etc.). All of those collapse to a
        // single public status.
        return TRANSCRIBE_ERR_GGUF;
    }

    // general.architecture is required. Without it the dispatch layer
    // has nothing to look up in the registry. Both Absent and BadType
    // are fatal: a missing arch and an arch with the wrong GGUF type
    // both leave us with no family to dispatch to.
    switch (read_string_kv(gguf_, "general.architecture", arch_)) {
        case KvResult::Absent:
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            break;
    }

    // stt.variant is optional in 0.x. Per-family handlers default it
    // when absent (e.g. parakeet -> tdt-0.6b-v2). A present-but-wrong-
    // type variant is still a converter bug, though, so BadType is
    // fatal here while Absent silently leaves variant_ empty.
    switch (read_string_kv(gguf_, "stt.variant", variant_)) {
        case KvResult::Absent:
            // variant_ remains empty; family handler will default it.
            break;
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            break;
    }

    return TRANSCRIBE_OK;
}

} // namespace transcribe
