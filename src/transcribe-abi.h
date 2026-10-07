// transcribe-abi.h - internal helpers for the size-aware public ABI.
//
// Shared by the public dispatchers (transcribe.cpp, transcribe-asr.cpp) and
// per-family public accessors (e.g. arch/whisper/public.cpp) so the raw enum
// reads, struct_size validation and copy-out truncation logic live in
// exactly one place. Not part of the public API.

#pragma once

#include "transcribe.h"

#include <cmath>
#include <cstddef>
#include <cstring>

// End offset of `field`: the minimum struct_size accepted for a caller-owned
// struct is the prefix through the last field the library reads or writes.
#define TRANSCRIBE_FIELD_END(type, field) (offsetof(type, field) + sizeof(((type *) 0)->field))

namespace transcribe {

// Raw enum reads at the public ABI boundary
//
// C callers can store ANY int in an enum-typed ABI field; in C++ loading an
// out-of-range value through an enum-typed lvalue is UB (UBSan traps it). So
// every public entry point reads caller-supplied enum values as raw bytes
// first, validates the raw int, and only then converts to the enum type. All
// public enums are int-sized; the static_assert below pins one.

inline int enum_field_raw(const void * field) {
    int raw;
    std::memcpy(&raw, field, sizeof(raw));
    return raw;
}

static_assert(sizeof(transcribe_backend_request) == sizeof(int),
              "public enums must be int-sized for raw boundary reads");

// Rejects NaN / +-Inf; every PCM entry point checks this before touching
// result or stream state.
inline bool pcm_is_finite(const float * pcm, int n_samples) {
    for (int i = 0; i < n_samples; ++i) {
        if (!std::isfinite(pcm[i])) {
            return false;
        }
    }
    return true;
}

// Strict size check for every caller-owned struct (inputs AND outputs):
// struct_size smaller than the prefix the library relies on is rejected.
// `0` is invalid — defaults are reached by passing NULL where the entry
// point accepts a nullable params pointer, never by a zero-sized struct.
inline transcribe_status check_struct_size(uint64_t got, uint64_t want) {
    if (got < want) {
        return TRANSCRIBE_ERR_BAD_STRUCT_SIZE;
    }
    return TRANSCRIBE_OK;
}

// Alias for call-site readability ("this is an input params struct").
// Behavior is identical to check_struct_size: struct_size == 0 is rejected
// (so coincidentally-zero stack memory is not taken as a defaults
// shortcut); defaults come from NULL.
inline transcribe_status check_input_struct_size(uint64_t got, uint64_t want) {
    return check_struct_size(got, want);
}

// Copy min(caller_size, library_size) bytes from a library-staged struct
// into the caller's output buffer. The min() is the overflow guard: the
// library never writes past what the caller declared, and never reads
// past what it staged. caller_size is uint64_t (matches the public
// struct_size field type); library_size is size_t (always sizeof(T)).
inline void copy_out_prefix(void * dst, const void * src, uint64_t caller_size, size_t library_size) {
    const uint64_t lib = static_cast<uint64_t>(library_size);
    const uint64_t n   = caller_size < lib ? caller_size : lib;
    std::memcpy(dst, src, static_cast<size_t>(n));
}

// The body of every public *_init(): zero-fill, then stamp struct_size.
template <typename T> inline void init_sized(T * p) {
    if (p != nullptr) {
        std::memset(p, 0, sizeof(*p));
        p->struct_size = sizeof(*p);
    }
}

// Copy stage times (microseconds) out as transcribe_timings (milliseconds);
// out->struct_size is already checked.
inline void copy_out_timings(int64_t                     load_us,
                             int64_t                     mel_us,
                             int64_t                     encode_us,
                             int64_t                     decode_us,
                             struct transcribe_timings * out) {
    transcribe_timings staged{};
    staged.struct_size = out->struct_size;
    staged.load_ms     = static_cast<float>(load_us) / 1000.0f;
    staged.mel_ms      = static_cast<float>(mel_us) / 1000.0f;
    staged.encode_ms   = static_cast<float>(encode_us) / 1000.0f;
    staged.decode_ms   = static_cast<float>(decode_us) / 1000.0f;
    copy_out_prefix(out, &staged, out->struct_size, sizeof(staged));
}

}  // namespace transcribe
