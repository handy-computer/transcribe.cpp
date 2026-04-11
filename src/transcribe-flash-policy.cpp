// transcribe-flash-policy.cpp - shared flash-attention policy helpers.
//
// See transcribe-flash-policy.h for rationale.

#include "transcribe-flash-policy.h"

#include <cstdlib>
#include <cstring>

namespace transcribe::flash {

bool is_metal_backend(const char * backend_name) {
    if (backend_name == nullptr) return false;
    return std::strstr(backend_name, "MTL")   != nullptr ||
           std::strstr(backend_name, "Metal") != nullptr;
}

void apply_env_overrides(bool & encoder_use_flash,
                         bool & decoder_use_flash) {
    const char * no_flash = std::getenv("TRANSCRIBE_NO_FLASH");
    if (no_flash != nullptr && no_flash[0] == '1') {
        encoder_use_flash = false;
        decoder_use_flash = false;
    }
    const char * force_flash = std::getenv("TRANSCRIBE_FORCE_FLASH");
    if (force_flash != nullptr && force_flash[0] == '1') {
        encoder_use_flash = true;
        decoder_use_flash = true;
    }
}

} // namespace transcribe::flash
