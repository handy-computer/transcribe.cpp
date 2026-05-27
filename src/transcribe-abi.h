// transcribe-abi.h - internal helpers for the size-aware public ABI.
//
// Shared by the central dispatcher (transcribe.cpp) and per-family public
// accessors (e.g. arch/whisper/public.cpp) so the struct_size validation
// and copy-out truncation logic lives in exactly one place. Not part of
// the public API.

#pragma once

#include "transcribe.h"

#include <cstddef>
#include <cstring>

namespace transcribe {

// Strict size check for caller-owned OUTPUT structs (and any input that
// has no "0 means defaults" relaxation): struct_size smaller than the
// prefix the library relies on is rejected.
inline transcribe_status check_struct_size(size_t got, size_t want) {
    if (got < want) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    return TRANSCRIBE_OK;
}

// Input params accept struct_size == 0 as "all defaults" (every input
// field's default is its zero value). A non-zero size that is too small
// to cover the relied-on prefix is still a caller error.
inline transcribe_status check_input_struct_size(size_t got, size_t want) {
    if (got == 0) {
        return TRANSCRIBE_OK;
    }
    return check_struct_size(got, want);
}

// Copy min(caller_size, library_size) bytes from a library-staged struct
// into the caller's output buffer. The min() is the overflow guard: the
// library never writes past what the caller declared, and never reads
// past what it staged.
inline void copy_out_prefix(void * dst, const void * src,
                            size_t caller_size, size_t library_size) {
    const size_t n = caller_size < library_size ? caller_size : library_size;
    std::memcpy(dst, src, n);
}

} // namespace transcribe
