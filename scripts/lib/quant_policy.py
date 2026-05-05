"""Canonical preset name registry.

Names match llama.cpp's llama-quantize exactly (uppercase). These are
the presets tools/transcribe-quantize accepts; Python code uses this
list to validate preset arguments in downstream drivers
(scripts/bench/run.py, scripts/quant_accuracy.py, ...) without
duplicating the canonical list.

Python does NOT implement any quantization algorithm. This module is
metadata only — see docs/tools/quantization.md.
"""

from __future__ import annotations


# Reference fp tiers produced by scripts/convert-<family>.py. Each
# family selects exactly one as its source/reference dtype.
REFERENCE_TIERS: tuple[str, ...] = ("F32", "F16", "BF16")

# Derived presets produced by tools/transcribe-quantize. Canonical
# order is accuracy -> size (F16 closest to baseline, Q4_K_M smallest
# shippable).
DERIVED_PRESETS: tuple[str, ...] = ("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")

# Per-architecture overrides for DERIVED_PRESETS. Keys are the
# `general.architecture` KV string written by the family's converter
# (NOT the family directory name; e.g. cohere converts to "cohere_asr").
# Used when k-quant tiers would degenerate into Q8_0 because the model's
# row sizes do not divide the k-quant super-block size (256). Shipping
# four near-duplicate Q-tier GGUFs is wasted disk; restrict the matrix
# to the tiers that actually differ for that architecture.
FAMILY_PRESETS: dict[str, tuple[str, ...]] = {
    # moonshine-tiny: hidden=288 / intermediate=1152 / vocab=32768 — none
    # divide 256, so Q6_K/Q5_K_M/Q4_K_M all fall back to Q8_0 storage.
    "moonshine": ("F16", "Q8_0"),
}


def derived_presets_for_arch(arch: str | None) -> tuple[str, ...]:
    """Return the preset matrix for a given general.architecture string,
    or DERIVED_PRESETS if no override exists / arch is unknown."""
    if arch and arch in FAMILY_PRESETS:
        return FAMILY_PRESETS[arch]
    return DERIVED_PRESETS


# All presets that may appear in a GGUF filename.
ALL_PRESETS: tuple[str, ...] = tuple(
    dict.fromkeys(REFERENCE_TIERS + DERIVED_PRESETS)
)


def validate_preset(name: str) -> str:
    """Normalize + validate a preset name. Returns the canonical
    uppercase form. Raises ValueError listing valid presets on mismatch."""
    up = name.upper()
    if up not in ALL_PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; valid: {', '.join(ALL_PRESETS)}"
        )
    return up
