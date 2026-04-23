#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
quantize-all.py — drive transcribe-quantize across the full shipped preset matrix.

Given a reference-dtype GGUF (F32, F16, or BF16 — the converter's output),
produces one quantized GGUF per preset in DERIVED_PRESETS (see
scripts/lib/quant_policy.py), skipping any preset that equals the source
dtype. Output filenames follow the llama.cpp convention:
<variant>-<PRESET>.gguf in the same directory as the input.

Usage:
    uv run scripts/quantize-all.py \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf

    # Pick a subset:
    uv run scripts/quantize-all.py --presets Q8_0,Q4_K_M \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf

Assumes build/bin/transcribe-quantize has been built (cmake --build build
--target transcribe-quantize). See docs/tools/quantization.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Import via path so the script stays `uv run`-compatible (no project context).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.quant_policy import DERIVED_PRESETS, validate_preset  # noqa: E402


def detect_source_preset(path: Path) -> str | None:
    """Return the F32/F16/BF16 suffix embedded in the filename, or None."""
    stem = path.stem
    for tier in ("F32", "F16", "BF16"):
        if stem.endswith(f"-{tier}"):
            return tier
    return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="Reference-dtype GGUF produced by convert-<family>.py")
    p.add_argument("--presets", type=str, default=",".join(DERIVED_PRESETS),
                   help=f"Comma-separated subset of {DERIVED_PRESETS} (default: all)")
    p.add_argument("--quantize-bin", type=Path, default=Path("build/bin/transcribe-quantize"),
                   help="Path to transcribe-quantize (default: build/bin/transcribe-quantize)")
    args = p.parse_args(argv)

    src = args.input.resolve()
    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 2
    if not args.quantize_bin.is_file():
        print(f"error: transcribe-quantize not built at {args.quantize_bin}. "
              f"Run: cmake --build build --target transcribe-quantize", file=sys.stderr)
        return 2

    presets = [validate_preset(x) for x in args.presets.split(",") if x.strip()]
    src_tier = detect_source_preset(src)
    stem_base = src.stem.removesuffix(f"-{src_tier}") if src_tier else src.stem

    results: list[tuple[str, bool, str]] = []
    for preset in presets:
        if preset == src_tier:
            results.append((preset, True, "skipped (equals source dtype)"))
            continue
        out = src.parent / f"{stem_base}-{preset}.gguf"
        if out.exists():
            results.append((preset, True, f"already present at {out}"))
            continue
        cmd = [str(args.quantize_bin), str(src), str(out), "--quant", preset]
        print(f"[quantize-all] {preset}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        ok = rc == 0
        results.append((preset, ok, str(out) if ok else f"transcribe-quantize exited {rc}"))

    print()
    print(f"{'preset':<10} {'status':<5}  detail")
    print(f"{'-'*10} {'-'*5}  {'-'*40}")
    for preset, ok, detail in results:
        print(f"{preset:<10} {('OK' if ok else 'FAIL'):<5}  {detail}")

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
