// transcribe-flash-policy.h - shared flash-attention policy helpers.
//
// INTERNAL. C++17. Consumed only from other .cpp files inside src/.
// Not part of the public include/transcribe.h ABI.
//
// Flash-attention support in ggml is per-backend and per-head-dim.
// Each family picks its own defaults based on encoder/decoder head
// dimensions and which backend it was loaded on.
//
// The "which backend" half of that decision now lives in
// transcribe-backend.h as BackendKind — families check
// `plan.primary_kind == BackendKind::Metal` directly instead of
// calling a helper here.
//
// What stays in this header is the one cross-family concern that is
// not backend-classification: honoring the TRANSCRIBE_NO_FLASH /
// TRANSCRIBE_FORCE_FLASH environment overrides. These are a global
// "no flash anywhere" / "flash everywhere" debug lever, not a
// per-family policy knob.
//
// The default on/off decision itself is still per-family: it
// depends on the family's head dim (cohere encoder=160 unsupported
// on Metal, cohere decoder=128 supported; parakeet encoder=128
// supported). Extracting that decision here would paper over a real
// architectural difference.

#pragma once

namespace transcribe::flash {

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
