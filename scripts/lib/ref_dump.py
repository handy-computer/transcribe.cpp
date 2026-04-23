"""Shared write-side for per-family reference dumpers.

Per the porting plan (`docs/porting/agent-automation-plan.md`), every
`scripts/dump_reference_<family>_<framework>.py` writes `.f32` + `.json`
sidecar pairs through this single helper so the on-disk contract stays
identical across frameworks (NeMo, Transformers, MLX, author repos).
Framework internals and CLI surface stay per-family; only the write
step is shared.

Contract:
    <out_dir>/<name>.f32   raw little-endian float32, row-major
    <out_dir>/<name>.json  sidecar metadata

The sidecar records the tensor identity (name, stage, shape, dtype,
layout), descriptive stats (min/max/mean, useful for eyeballing drift
profiles without reading the binary), and provenance (`source` — whose
forward pass produced this tensor, e.g. `{"framework": "nemo", "model":
"nvidia/parakeet-tdt-0.6b-v2", "hook": "encoder.block0.out"}`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_tensor(
    name: str,
    array: np.ndarray,
    stage: str,
    source: dict[str, Any],
    *,
    out_dir: Path,
) -> None:
    """Write a (<name>.f32, <name>.json) pair into out_dir.

    Arguments:
      name    Tensor identifier as it will appear in the tolerances file
              and compare_tensors reports. Keep stable across runs of
              the same family (per-framework dumpers own the naming
              scheme for that family).
      array   Float32 numpy array. Non-f32 inputs raise — cast before
              calling so the cast site is visible at the call site,
              not hidden in the helper.
      stage   Coarse grouping used by the dump_coverage catalog and
              validate.py's layout discovery, e.g. "encoder", "decoder",
              "mel".
      source  Free-form provenance dict. Typical keys: framework,
              model, hook, dtype_native, notes.
      out_dir Destination directory. Created if absent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if array.dtype != np.float32:
        raise ValueError(
            f"ref_dump.write_tensor: {name!r} is {array.dtype}, expected float32. "
            "Cast at the call site (e.g. .astype(np.float32)) so the cast is explicit."
        )
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    f32_path = out_dir / f"{name}.f32"
    json_path = out_dir / f"{name}.json"

    array.tofile(f32_path)
    meta: dict[str, Any] = {
        "name": name,
        "stage": stage,
        "shape": list(array.shape),
        "dtype": "f32",
        "layout": "row-major",
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "mean": float(array.mean(dtype=np.float64)) if array.size else 0.0,
        "source": source,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")


def write_transcript(
    out_dir: Path,
    text: str,
    *,
    source: dict[str, Any],
    tokens: list[int] | None = None,
) -> None:
    """Write transcript.json alongside a decode dump.

    transcript.json is the behavioral artifact: the text the reference
    framework produced for this audio, plus optional token IDs. Used by
    validate.py to compare against the C++ transcript at the public
    boundary (not just per-tensor).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"text": text, "source": source}
    if tokens is not None:
        payload["tokens"] = list(tokens)
    (out_dir / "transcript.json").write_text(json.dumps(payload, indent=2) + "\n")
