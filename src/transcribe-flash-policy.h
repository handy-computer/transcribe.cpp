// transcribe-flash-policy.h - shared flash-attention policy helpers.
//
// INTERNAL. C++17. Consumed only from other .cpp files inside src/.
// Not part of the public include/transcribe.h ABI.
//
// Flash-attention support in ggml is per-backend and per-head-dim.
// Each family picks its own defaults based on encoder/decoder head
// dimensions and which backend it was loaded on; this header hoists
// the two pieces that are always the same:
//
//   1. Detecting whether the primary backend is Metal (so the
//      family can auto-disable flash on head dims the Metal backend
//      lacks a template for).
//
//   2. Honoring the TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH
//      environment overrides -- these are a global "no flash
//      anywhere" / "flash everywhere" debug lever, not a per-family
//      policy knob.
//
// The default on/off decision itself is still per-family: it
// depends on the family's head dim (cohere encoder=160 unsupported
// on Metal, cohere decoder=128 supported; parakeet encoder=128
// supported). Extracting that decision here would paper over a real
// architectural difference.

#pragma once

namespace transcribe::flash {

// Return true if the named backend is ggml's Metal backend.
// Matches either "MTL*" (Metal's short name, e.g. "MTL0") or
// "Metal*" (Metal's long name). All other backends return false.
//
//   backend_name: the string returned by ggml_backend_name() on
//                 the model's primary backend (first in the
//                 backend list). May be nullptr -- returns false.
bool is_metal_backend(const char * backend_name);

// Apply the TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH
// environment overrides to a pair of encoder/decoder flash flags.
// Either override forces both flags in the same direction -- the
// user's intent is "no flash kernels anywhere" or "flash kernels
// everywhere", not "flash on encoder but not decoder".
//
// If both env vars are set simultaneously, FORCE wins (last apply).
// That's a deliberately undocumented edge case; the interesting
// combination is neither-or-one, not both.
//
//   encoder_use_flash: in/out. Updated in-place if an override is
//                      set; left untouched otherwise.
//   decoder_use_flash: same.
void apply_env_overrides(bool & encoder_use_flash,
                         bool & decoder_use_flash);

} // namespace transcribe::flash
