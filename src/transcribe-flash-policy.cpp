// transcribe-flash-policy.cpp - shared flash-attention policy helpers.
//
// See transcribe-flash-policy.h for rationale.

#include "transcribe-flash-policy.h"

#include "transcribe-backend.h"
#include "transcribe-env.h"

namespace transcribe::flash {

bool per_head_mask_supported(BackendKind kind) {
    return kind != BackendKind::Cuda && kind != BackendKind::Rocm;
}

void apply_env_overrides(bool & encoder_use_flash, bool & decoder_use_flash) {
    if (transcribe::env::flag("TRANSCRIBE_NO_FLASH")) {
        encoder_use_flash = false;
        decoder_use_flash = false;
    }
    if (transcribe::env::flag("TRANSCRIBE_FORCE_FLASH")) {
        encoder_use_flash = true;
        decoder_use_flash = true;
    }
}

}  // namespace transcribe::flash
