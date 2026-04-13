#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
compare_tensors.py - symmetric directory diff for the per-stage
numerical accuracy harness used by phase 4+ of transcribe.cpp.

Each input directory is expected to contain pairs of files written by
either:

  - the C++ debug dumper in src/transcribe-debug.{h,cpp} (gated on
    TRANSCRIBE_DUMP_DIR)
  - the Python reference dumper in scripts/dump_reference.py

Both sides agree on the same on-disk format:

    <name>.f32   raw little-endian fp32, row-major (C order)
    <name>.json  shape, dtype, layout, optional stage / source

This script walks both directories, finds every tensor name that appears
in *either* dir, loads each side as a numpy array, and reports
element-wise stats per tensor:

    max abs diff
    mean abs diff
    first divergent flat index (or "-" if exact)
    shape on each side

A tensor that exists on only one side is reported as MISSING. It counts
as a failure only when that tensor has an explicit tolerance entry,
which marks it as part of the validation contract. Missing debug-only
tensors are reported for visibility but do not fail the comparison.

Tolerances:
    --max-abs   maximum allowed max absolute element diff (default 1e-3)
    --mean-abs  maximum allowed mean absolute element diff (default 1e-4)
Per-tensor tolerance overrides can be supplied via --tolerances pointing
at a tiny JSON file:

    { "enc.pre_encode.out": {"max_abs": 5e-4, "mean_abs": 5e-5},
      "enc.final":          {"max_abs": 5e-3, "mean_abs": 5e-4} }

Exit code:
    0  every contract tensor was within tolerance
    1  one or more contract tensors exceeded tolerance or were missing

Usage:
    uv run scripts/compare_tensors.py /tmp/cpp /tmp/ref
    uv run scripts/compare_tensors.py /tmp/cpp /tmp/ref --max-abs 5e-4

The "left" side is conventionally the C++ dump and the "right" side
is the reference dump — but the comparison is
symmetric so the order only affects the diff sign in any future
verbose output, not pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TensorPair:
    name: str
    left_shape: tuple[int, ...] | None
    right_shape: tuple[int, ...] | None
    left_data: np.ndarray | None
    right_data: np.ndarray | None


def load_dump(d: Path, name: str) -> tuple[tuple[int, ...] | None, np.ndarray | None]:
    """Load (<name>.f32, <name>.json) from a dump dir.

    Returns (shape, data) where shape is the slow-to-fast tuple from
    the JSON sidecar and data is a 1-D numpy float32 array
    (un-reshaped — comparison is element-wise on the flat layout, so
    a shape mismatch is its own failure mode).
    """
    f32_path = d / f"{name}.f32"
    json_path = d / f"{name}.json"
    if not f32_path.exists() or not json_path.exists():
        return None, None
    meta = json.loads(json_path.read_text())
    if meta.get("dtype") != "f32":
        raise ValueError(f"{name}: unsupported dtype {meta.get('dtype')!r}")
    if meta.get("layout") != "row-major":
        raise ValueError(f"{name}: unsupported layout {meta.get('layout')!r}")
    shape = tuple(int(s) for s in meta["shape"])
    raw = np.fromfile(f32_path, dtype=np.float32)
    expected = int(np.prod(shape)) if shape else 0
    if expected and raw.size != expected:
        raise ValueError(
            f"{name}: f32 element count {raw.size} doesn't match "
            f"shape product {expected} (shape={shape})"
        )
    return shape, raw


def discover_names(d: Path) -> set[str]:
    """Find tensor dump names in a dir.

    A tensor dump has a <name>.f32 payload and usually a <name>.json
    sidecar. If the f32 file is missing but the json sidecar looks like
    tensor metadata, include it so the comparison reports the half-pair.
    Standalone behavioral artifacts such as transcript.json are ignored.
    """
    if not d.exists():
        return set()
    names: set[str] = set()
    for p in d.iterdir():
        if p.suffix == ".f32":
            names.add(p.stem)
            continue
        if p.suffix == ".json" and not p.with_suffix(".f32").exists():
            try:
                meta = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if {"shape", "dtype", "layout"}.issubset(meta):
                names.add(p.stem)
    return names


@dataclass
class CompareResult:
    name: str
    left_shape: tuple[int, ...] | None
    right_shape: tuple[int, ...] | None
    n_elements: int
    max_abs: float
    mean_abs: float
    first_diff_idx: int  # -1 if exact match
    status: str          # "ok", "FAIL", "MISSING-left", "MISSING-right", "SHAPE"


def compare_pair(
    name: str,
    left_dir: Path,
    right_dir: Path,
) -> CompareResult:
    try:
        ls, ld = load_dump(left_dir, name)
    except Exception as e:
        return CompareResult(name, None, None, 0, 0.0, 0.0, -1, f"FAIL ({e})")
    try:
        rs, rd = load_dump(right_dir, name)
    except Exception as e:
        return CompareResult(name, None, None, 0, 0.0, 0.0, -1, f"FAIL ({e})")

    if ld is None and rd is None:
        # Should not happen — discover_names found one of them.
        return CompareResult(name, None, None, 0, 0.0, 0.0, -1, "MISSING-both")
    if ld is None:
        return CompareResult(name, None, rs, rd.size if rd is not None else 0,
                             0.0, 0.0, -1, "MISSING-left")
    if rd is None:
        return CompareResult(name, ls, None, ld.size, 0.0, 0.0, -1, "MISSING-right")
    if ls != rs:
        return CompareResult(name, ls, rs, max(ld.size, rd.size),
                             0.0, 0.0, -1, "SHAPE")

    if ld.size != rd.size:
        # Defensive: load_dump should have caught this via the shape
        # product check, but be paranoid because a mismatch here would
        # crash np.subtract.
        return CompareResult(name, ls, rs, max(ld.size, rd.size),
                             0.0, 0.0, -1, "SHAPE")

    diff = np.abs(ld.astype(np.float64) - rd.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    first_diff = int(np.argmax(diff > 0)) if max_abs > 0 else -1
    return CompareResult(name, ls, rs, int(ld.size), max_abs, mean_abs,
                         first_diff, "ok")


def load_tolerances(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: top-level must be an object")
    return obj


def tolerance_for(
    name: str,
    overrides: dict[str, dict[str, float]],
    default_max_abs: float,
    default_mean_abs: float,
) -> tuple[float, float]:
    o = overrides.get(name, {})
    return (
        float(o.get("max_abs", default_max_abs)),
        float(o.get("mean_abs", default_mean_abs)),
    )


def fmt_shape(s: tuple[int, ...] | None) -> str:
    if s is None:
        return "—"
    return "[" + ",".join(str(x) for x in s) + "]"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Symmetric per-tensor diff between two dump directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("left", type=Path, help="First dump directory (e.g. C++ dump)")
    p.add_argument("right", type=Path, help="Second dump directory (usually reference)")
    p.add_argument("--max-abs", type=float, default=1e-3,
                   help="Default tolerance for max abs element diff (default 1e-3)")
    p.add_argument("--mean-abs", type=float, default=1e-4,
                   help="Default tolerance for mean abs element diff (default 1e-4)")
    p.add_argument("--tolerances", type=Path, default=None,
                   help="Optional JSON file with per-tensor tolerance overrides")
    p.add_argument("--quiet", action="store_true",
                   help="Only print failures, not the per-tensor table")
    args = p.parse_args()

    if not args.left.is_dir():
        print(f"error: {args.left} is not a directory", file=sys.stderr)
        return 2
    if not args.right.is_dir():
        print(f"error: {args.right} is not a directory", file=sys.stderr)
        return 2

    overrides = load_tolerances(args.tolerances)

    names = sorted(discover_names(args.left) | discover_names(args.right))
    if not names:
        print("no dumps found in either directory", file=sys.stderr)
        return 2

    results: list[CompareResult] = []
    for name in names:
        results.append(compare_pair(name, args.left, args.right))

    # Per-tensor table.
    fail = 0
    if not args.quiet:
        print(f"{'name':<40} {'status':<8} {'shape':<20} "
              f"{'max_abs':>12} {'mean_abs':>12} {'first_diff':>10}")
        print("-" * 110)
    for r in results:
        max_tol, mean_tol = tolerance_for(
            r.name, overrides, args.max_abs, args.mean_abs)
        # A MISSING tensor that isn't in the tolerance file is a debug
        # dump present on only one side — not a gate failure. Only
        # count it as a failure if it has an explicit tolerance entry
        # (meaning it's expected to be present on both sides).
        is_missing = r.status.startswith("MISSING")
        in_tolerance_file = r.name in overrides
        if is_missing and not in_tolerance_file:
            ok = True
            flag = r.status  # still show it, but don't fail
        else:
            ok = (r.status == "ok"
                  and r.max_abs  <= max_tol
                  and r.mean_abs <= mean_tol)
            flag = "ok" if ok else r.status if r.status != "ok" else "FAIL"
        if not ok:
            fail += 1
        if args.quiet and ok:
            continue
        shape_str = fmt_shape(r.left_shape) if r.left_shape == r.right_shape \
            else f"{fmt_shape(r.left_shape)}|{fmt_shape(r.right_shape)}"
        first_diff = "-" if r.first_diff_idx < 0 else str(r.first_diff_idx)
        print(f"{r.name:<40} {flag:<8} {shape_str:<20} "
              f"{r.max_abs:>12.3e} {r.mean_abs:>12.3e} {first_diff:>10}")
        if not ok and r.status == "ok":
            # Within "ok" status but exceeded tolerance — show why.
            print(f"  -> tolerance exceeded "
                  f"(max_abs > {max_tol:.1e} or mean_abs > {mean_tol:.1e})")

    print(f"\n{len(results) - fail}/{len(results)} tensors within tolerance")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
