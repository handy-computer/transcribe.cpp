"""GGUF helpers shared across per-family converters.

Scope is deliberately small: just the pieces that every converter
needs regardless of model family. Family-specific logic (tensor
catalogs, hparam maps, special-token handling) stays in
scripts/convert-<family>.py.

Python is conversion-only. The only dtype encoding this module
performs is F32 / F16 / BF16 — block quantization lives entirely in
tools/transcribe-quantize/ (see docs/tools/quantization.md).
"""

from __future__ import annotations

import numpy as np
import gguf
from gguf import GGMLQuantizationType


# GGUF filename convention: `<slug>-<QUANT>.gguf`, matching llama.cpp
# (e.g. `parakeet-tdt-0.6b-v2-Q5_K_M.gguf`). The slug is derived from
# the HF repo id (`nvidia/parakeet-tdt-0.6b-v2` → `parakeet-tdt-0.6b-v2`).
# Quant is the uppercase preset name from transcribe-quantize: F32, F16,
# BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, etc.
def slug_from_repo_id(repo_id: str) -> str:
    """`org/name` → `name`. `name` alone passes through."""
    s = repo_id.strip().strip("/")
    if not s:
        raise ValueError("repo_id is empty")
    return s.rsplit("/", 1)[-1]


def gguf_name(slug: str, quant: str) -> str:
    """Canonical GGUF filename: `<slug>-<QUANT>.gguf` (quant uppercase)."""
    if not slug:
        raise ValueError("slug is empty")
    if not quant:
        raise ValueError("quant is empty")
    return f"{slug}-{quant.upper()}.gguf"


# llama.cpp / whisper.cpp tokenizer.ggml.token_type values. We follow
# the same conventions so an inspector built for either project can
# read our GGUFs without surprises.
TOKEN_TYPE_NORMAL  = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER    = 4
TOKEN_TYPE_UNUSED  = 5
TOKEN_TYPE_BYTE    = 6


def safe_id(method) -> int | None:
    """Wrap a SentencePieceProcessor id accessor: -1 -> None.

    SentencePiece returns -1 when a special token isn't defined. GGUF
    wants the KV absent in that case, which the writer expresses as None.
    """
    v = method()
    return v if v >= 0 else None


def encode_for_gguf(
    arr: np.ndarray,
    ggml_type: GGMLQuantizationType,
) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Pack an fp32 array into GGUF bytes at F32, F16, or BF16.

    This is the only dtype encoding Python is permitted to do. Block
    quantization (Q8_0, Q4_K_M, ...) goes through
    tools/transcribe-quantize/ — see docs/tools/quantization.md.
    """
    if arr.dtype != np.float32:
        raise TypeError(
            f"encode_for_gguf expects fp32 input, got {arr.dtype}"
        )
    if ggml_type == GGMLQuantizationType.F32:
        return arr, GGMLQuantizationType.F32
    if ggml_type == GGMLQuantizationType.F16:
        return arr.astype(np.float16), GGMLQuantizationType.F16
    if ggml_type == GGMLQuantizationType.BF16:
        packed = gguf.quants.quantize(arr, GGMLQuantizationType.BF16)
        return packed, GGMLQuantizationType.BF16
    raise ValueError(
        f"encode_for_gguf only accepts F32/F16/BF16; got {ggml_type!r}. "
        "Block-quantized types go through tools/transcribe-quantize."
    )
