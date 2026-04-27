// transcribe-weights-util.cpp - shared GGUF tensor validation helpers.
//
// Implementation of transcribe::weights::find_tensor. See the header
// for rationale. This is a verbatim hoist of the function that used
// to live in both src/arch/parakeet/weights.cpp and
// src/arch/cohere/weights.cpp (they were character-for-character
// identical apart from the "parakeet:"/"cohere:" log prefix).

#include "transcribe-weights-util.h"

#include "ggml.h"

#include <cstdio>

namespace transcribe::weights {

ggml_tensor * find_tensor(ggml_context *                    ctx_meta,
                          const char *                      name,
                          std::initializer_list<ggml_type>  allowed_types,
                          std::initializer_list<int64_t>    expected_ne,
                          const char *                      error_tag)
{
    ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
    if (t == nullptr) {
        std::fprintf(stderr, "%s: missing tensor \"%s\"\n", error_tag, name);
        return nullptr;
    }

    bool type_ok = false;
    for (ggml_type allowed : allowed_types) {
        if (t->type == allowed) {
            type_ok = true;
            break;
        }
    }
    if (!type_ok) {
        // Build a short human-readable list of allowed types for the
        // diagnostic. Stack buffer is plenty for 1-12 allowed types.
        char allowed_buf[192] = {0};
        size_t off = 0;
        bool first = true;
        for (ggml_type allowed : allowed_types) {
            const char * tn = ggml_type_name(allowed);
            const int n = std::snprintf(allowed_buf + off,
                                        sizeof(allowed_buf) - off,
                                        "%s%s", first ? "" : ",", tn);
            if (n < 0 || static_cast<size_t>(n) >= sizeof(allowed_buf) - off) {
                break;
            }
            off += static_cast<size_t>(n);
            first = false;
        }
        std::fprintf(stderr,
                     "%s: tensor \"%s\" type mismatch: "
                     "expected one of {%s}, got %s\n",
                     error_tag, name, allowed_buf, ggml_type_name(t->type));
        return nullptr;
    }

    const size_t n_expected = expected_ne.size();
    if (n_expected == 0 || n_expected > GGML_MAX_DIMS) {
        std::fprintf(stderr,
                     "%s: bad expected_ne size %zu for \"%s\"\n",
                     error_tag, n_expected, name);
        return nullptr;
    }

    auto it = expected_ne.begin();
    for (size_t i = 0; i < n_expected; ++i, ++it) {
        if (t->ne[i] != *it) {
            std::fprintf(stderr,
                         "%s: tensor \"%s\" shape mismatch: "
                         "expected ne[%zu]=%lld, got %lld\n",
                         error_tag, name, i,
                         static_cast<long long>(*it),
                         static_cast<long long>(t->ne[i]));
            return nullptr;
        }
    }
    // Any dims beyond the expected list must be 1 — otherwise a
    // caller expecting a rank-k tensor could silently accept a
    // higher-rank tensor with matching leading dims.
    for (size_t i = n_expected; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] != 1) {
            std::fprintf(stderr,
                         "%s: tensor \"%s\" has unexpected non-1 "
                         "ne[%zu]=%lld (rank too high)\n",
                         error_tag, name, i,
                         static_cast<long long>(t->ne[i]));
            return nullptr;
        }
    }

    return t;
}

} // namespace transcribe::weights
