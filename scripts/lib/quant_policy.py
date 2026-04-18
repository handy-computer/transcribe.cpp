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
