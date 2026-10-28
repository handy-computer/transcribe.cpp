// transcribe-flash-policy.h - shared flash-attention policy helpers.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// Flash-attention support in ggml is per-backend and per-head-dim, so the
// default on/off decision is per-family (it depends on head dim and the
// backend it was loaded on). What lives here are the cross-family concerns:
// backend limits that every family with the same attention shape hits, and
// honoring the TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH global debug
// overrides.

#pragma once

namespace transcribe {
enum class BackendKind;  // transcribe-backend.h
}

namespace transcribe::flash {

// Whether `kind`'s flash kernels take a mask with one slice per head, which
// is how the conformer's relative-position attention passes its position
// bias. ggml-cuda's kernels (ROCm builds the same ones) only take a mask
// shared by all heads, so on CUDA/ROCm the scheduler runs FLASH_ATTN_EXT on
// the CPU, with a device round trip in every block; the manual path stays on
// the GPU.
bool per_head_mask_supported(BackendKind kind);

// Apply the TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH environment
// overrides to a pair of encoder/decoder flash flags. Either override forces
// both flags in the same direction; if both are set, FORCE wins. The flags
// are in/out: updated in-place if an override is set, left untouched
// otherwise.
void apply_env_overrides(bool & encoder_use_flash, bool & decoder_use_flash);

}  // namespace transcribe::flash
