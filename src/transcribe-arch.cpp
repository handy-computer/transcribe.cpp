// transcribe-arch.cpp - explicit registry of supported architectures.
//
// The registry is intentionally a function-local static array. Pros:
//   - No static-initializer-order dependency between this TU and the
//     per-family TUs that define each Arch instance (the array holds
//     pointers, not copies, and is initialized lazily on first call).
//   - The full set of supported architectures is grep-able from one file.
//   - Adding a family is a two-line edit (forward declaration + array
//     entry) plus the per-family TU.
//
// Cons (acceptable for now):
//   - Linear search. Single-digit n; lookups happen once per model load.

#include "transcribe-arch.h"

#include <cstring>

namespace transcribe {

// Per-family Arch instances. Defined in src/arch/<family>/model.cpp.
namespace parakeet           { extern const Arch arch; }
namespace cohere             { extern const Arch arch; }
namespace canary             { extern const Arch arch; }
namespace qwen3_asr          { extern const Arch arch; }
namespace canary_qwen        { extern const Arch arch; }
namespace whisper            { extern const Arch arch; }
namespace moonshine          { extern const Arch arch; }
namespace moonshine_streaming { extern const Arch arch; }
namespace sensevoice         { extern const Arch arch; }
namespace funasr_nano        { extern const Arch arch; }
namespace gigaam             { extern const Arch arch; }
namespace granite            { extern const Arch arch; }
namespace granite_nar        { extern const Arch arch; }
namespace medasr             { extern const Arch arch; }

const Arch * find_arch(const char * name) {
    if (name == nullptr) {
        return nullptr;
    }

    static const Arch * const k_archs[] = {
        &parakeet::arch,
        &cohere::arch,
        &canary::arch,
        &qwen3_asr::arch,
        &canary_qwen::arch,
        &whisper::arch,
        &moonshine::arch,
        &moonshine_streaming::arch,
        &sensevoice::arch,
        &funasr_nano::arch,
        &gigaam::arch,
        &granite::arch,
        &granite_nar::arch,
        &medasr::arch,
    };
    constexpr size_t k_n = sizeof(k_archs) / sizeof(k_archs[0]);

    for (size_t i = 0; i < k_n; ++i) {
        const Arch * a = k_archs[i];
        if (a == nullptr || a->name == nullptr) {
            continue;
        }
        if (std::strcmp(a->name, name) == 0) {
            return a;
        }
    }
    return nullptr;
}

} // namespace transcribe
