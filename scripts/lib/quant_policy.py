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
# row sizes do not divide the k-quant super-block size (256), or when a
# family's output is too sensitive to weight perturbation to certify the
# k tiers. Shipping four near-duplicate Q-tier GGUFs is wasted disk;
# restrict the matrix to the tiers that actually earn their place.
FAMILY_PRESETS: dict[str, tuple[str, ...]] = {
    # moonshine-tiny: hidden=288 / intermediate=1152 / vocab=32768 — none
    # divide 256, so Q6_K/Q5_K_M/Q4_K_M all fall back to Q8_0 storage.
    "moonshine": ("F16", "Q8_0"),
    # moonshine-streaming-tiny: enc hidden=320 / dec hidden=320 /
    # ffn=1280 / vocab=32768 — none divide 256, so Q6_K/Q5_K_M/Q4_K_M
    # land within ~5% of Q8_0 file size (measured Stage 5). Drop the K
    # tiers to avoid shipping near-duplicate GGUFs.
    "moonshine_streaming": ("F16", "Q8_0"),
    # sortformer (streaming diarizer): output depends on discrete AOSC
    # speaker-cache compression decisions; k-quant weight error can
    # deterministically flip a near-tie pick and permute speaker labels
    # mid-stream (Stage 7: Q5_K_M swapped labels on AMI TS3003b at the
    # default CPU operating point, 9.5%->32.1% DER; flip risk is chaotic
    # in quant error, so clean-on-sample Q6_K/Q4_K_M cannot be certified
    # either). K tiers also save little (Q8_0 139MB -> Q4_K_M 92MB) and
    # are slower than Q8_0 on CPU. Ship only the near-reference tiers.
    "sortformer": ("F16", "Q8_0"),
    # ecapa_tdnn (language ID): a 21M-parameter model whose decision is an
    # argmax over 107 logits. Q8_0 ships for its download size (the loader
    # widens it to F16, see src/arch/ecapa_tdnn/model.cpp); the K tiers would
    # save a few MB at most. Ship only the near-reference tiers.
    "ecapa_tdnn": ("F16", "Q8_0"),
    # sava (TURN + VAD): F32 is the reference. F16 holds the bf16 source
    # weights exactly except 1,065 tiny ones (|err| <= 3e-8), and ggml keeps
    # F16 x F32 matmuls in fp32 activations, so it matches F32 to 4e-6 with
    # no decision changes on the Pipecat set; it halves the size. Q8_0 is the
    # small, fast tier. BF16 (convert-sava.py --outtype bf16) is lossless
    # storage, but ggml rounds activations to bf16 and its CPU path is 15-22x
    # slower than F32 without native BF16 (Apple M1); not shipped. K tiers
    # not evaluated.
    "sava": ("F16", "Q8_0"),
    # smart_turn: d_model 384 is not a multiple of 256, so the K tiers fall
    # back to near-Q8_0 storage (near-duplicates); ship only F16 and Q8_0.
    "smart_turn": ("F16", "Q8_0"),
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
