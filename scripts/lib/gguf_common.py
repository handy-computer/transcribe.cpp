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


def reference_dtype_for(
    name: str,
    reference_type: GGMLQuantizationType,
) -> GGMLQuantizationType:
    """Pick the per-tensor GGUF dtype for a converter emitting a
    reference-tier GGUF. Mirrors the bucketing in
    tools/transcribe-quantize/policy.cpp::classify_tensor so the
    produced file matches the loader's allowlist in
    src/transcribe-weights-util.h (TRANSCRIBE_QUANT_{LINEAR,CONV}_TYPES
    and the F32-only slots used by GET_F32).

    Policy:
      - Norm / bias / BN stats / pos biases / frontend buffers → F32
      - Conv kernels (2D, depthwise, 1×1 pointwise)            → F16 if
        reference is BF16 (loader rejects BF16 conv); otherwise the
        reference dtype
      - Linear / Embed (everything else)                       → reference

    Keep this in sync with tools/transcribe-quantize/policy.cpp.
    """
    # Norm bucket.
    if name.endswith(".bias"):
        return GGMLQuantizationType.F32
    if ".bn." in name:
        return GGMLQuantizationType.F32
    if "norm_" in name and name.endswith(".weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".final_norm.weight") or \
       name.endswith(".embed.norm.weight"):
        return GGMLQuantizationType.F32
    # Qwen3-style norm weights that don't match the "norm_" prefix rule
    # (per-head q_norm/k_norm on attention, output RMSNorm before the
    # tied head, pre/post encoder layer norms).
    if name.endswith(".q_norm.weight") or name.endswith(".k_norm.weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".output_norm.weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".ln_post.weight") or name.endswith(".ln_pre.weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".pos_bias_u") or name.endswith(".pos_bias_v"):
        return GGMLQuantizationType.F32
    if name.endswith(".pos_enc"):
        return GGMLQuantizationType.F32
    if name.endswith(".pos_emb.weight"):
        return GGMLQuantizationType.F32
    if name in ("frontend.mel_filterbank", "frontend.window"):
        return GGMLQuantizationType.F32

    # Conv bucket (pointwise + depthwise + 2D).
    is_convpw = (
        (name.endswith(".conv.pointwise1.weight") or
         name.endswith(".conv.pointwise2.weight"))
        and "enc.blocks." in name
    )
    is_conv = ".conv." in name and name.endswith(".weight")
    if is_convpw or is_conv:
        if reference_type == GGMLQuantizationType.BF16:
            return GGMLQuantizationType.F16  # loader has no BF16 conv kernel
        return reference_type

    # Linear / Embed — use the reference dtype as-is.
    return reference_type


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
