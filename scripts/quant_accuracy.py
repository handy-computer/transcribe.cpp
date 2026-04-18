#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
quant_accuracy.py - per-quant numerical accuracy gate.

Runs transcribe-cli once per GGUF variant with TRANSCRIBE_DUMP_DIR set,
collects per-stage encoder + decoder activation dumps, and compares each
variant against an F32 baseline using scripts/compare_tensors.py. Prints
a per-variant pass/fail summary keyed off per-quant tolerance bands.

Usage:

    uv run scripts/quant_accuracy.py \\
        --baseline models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F16.gguf \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q8_0.gguf \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q5_K_M.gguf \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q4_K_M.gguf

    # Or against a different audio file:
    uv run scripts/quant_accuracy.py --audio samples/dots.wav \\
        models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q5_K_M.gguf

Tolerance bands per quant family (initial guesses, refine with WER):

    family   rel_max  rel_mean
    f16      1e-3     1e-3
    q8_0     2e-2     2e-2
    q5_k_m   1e-1     1e-1
    q4_k_m   2e-1     2e-1

These are RELATIVE errors: max(|diff|) / max(|baseline|), and
mean(|diff|) / mean(|baseline|). Relative-to-baseline is the right
metric because the encoder's per-block dumps are pre-LayerNorm
intermediates with unbounded magnitudes that grow through the
residual chain — an absolute band that fits the post-LN outputs
would false-positive on the pre-LN intermediates. Reporting relative
makes a single set of bands cover both. Per-tensor absolute values
are still printed for human debugging.

The bands are deliberately loose. The point is to catch a layer-level
numerical *regression* (a failed loader change, a wrong dtype routing,
a kernel that silently fell back to a stub), not to substitute for WER
measurement. WER is the user-facing acceptance gate; this is the dev
gate that runs in seconds.

Quant family is detected from the filename: a GGUF named
*-Q4_K_M.gguf gets the q4_k_m bands, *-F16.gguf gets the f16 bands,
etc. Override with --quant FAMILY=PATH if your filenames don't match.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Per-quant tolerance bands.
# ---------------------------------------------------------------------------
#
# Each entry is (default_max_abs, default_mean_abs) — the GLOBAL band
# applied to every dumped tensor. compare_tensors.py also supports
# per-tensor overrides via a JSON file; if a particular dump point
# turns out to be especially noisy at a given quant level, add a
# per-tensor entry to a tolerance JSON and pass --tolerances.
# (rel_max_tol, rel_mean_tol). Relative errors are diff/|baseline|.
QUANT_BANDS: dict[str, tuple[float, float]] = {
    "f32":    (1e-6, 1e-6),
    "f16":    (1e-3, 1e-3),
    "q8_0":   (2e-2, 2e-2),
    "q5_k_m": (1e-1, 1e-1),
    "q4_k_m": (2e-1, 2e-1),
}


# Filename → quant family. The match is on the suffix of the model
# basename, so models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q5_K_M.gguf is
# detected as q5_k_m.
QUANT_PAT = re.compile(r"-(?P<family>F32|F16|Q8_0|Q5_K_M|Q4_K_M)\.gguf$", re.IGNORECASE)


def detect_family(path: Path) -> str:
    m = QUANT_PAT.search(path.name)
    if m is None:
        raise ValueError(
            f"cannot detect quant family from {path.name}; "
            f"expected suffix in {sorted(QUANT_BANDS)}"
        )
    return m.group("family").lower()


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until we find a directory containing
    CMakeLists.txt + scripts/ (the transcribe.cpp repo root). Used so
    the script Just Works regardless of cwd."""
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("could not locate transcribe.cpp repo root")


def run_dump(
    cli: Path,
    model: Path,
    audio: Path,
    dump_dir: Path,
    *,
    repeat: int = 1,
) -> None:
    """Invoke transcribe-cli with TRANSCRIBE_DUMP_DIR set so the
    library writes per-stage activation dumps. The CLI's --repeat flag
    re-runs the encoder + decoder N times in the same process; for
    accuracy work N=1 is enough (the dumps are identical across
    repeats), but the option is exposed for parity with the bench
    script.
    """
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TRANSCRIBE_DUMP_DIR"] = str(dump_dir)

    cmd = [str(cli), "-q", "-r", str(repeat),
           "-m", str(model), str(audio)]
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"transcribe-cli failed for {model.name} (exit {res.returncode})"
        )
    # Useful single line of provenance per run.
    text_line = next(
        (line for line in res.stdout.splitlines() if line.startswith("text:")),
        "(no text output)",
    )
    print(f"    {text_line[:120]}{'...' if len(text_line) > 120 else ''}")


def load_dump(d: Path, name: str) -> tuple[tuple[int, ...] | None, np.ndarray | None]:
    """Load (<name>.f32, <name>.json) from a dump dir. Mirrors
    compare_tensors.py's loader so this script stays standalone."""
    f32_path = d / f"{name}.f32"
    json_path = d / f"{name}.json"
    if not f32_path.exists() or not json_path.exists():
        return None, None
    import json
    meta = json.loads(json_path.read_text())
    shape = tuple(int(s) for s in meta["shape"])
    raw = np.fromfile(f32_path, dtype=np.float32)
    expected = int(np.prod(shape)) if shape else 0
    if expected and raw.size != expected:
        raise ValueError(
            f"{name}: f32 element count {raw.size} != shape product {expected}"
        )
    return shape, raw


def discover_names(d: Path) -> set[str]:
    if not d.exists():
        return set()
    names: set[str] = set()
    for p in d.iterdir():
        if p.suffix in (".f32", ".json"):
            names.add(p.stem)
    return names


class Row:
    __slots__ = ("name", "max_abs", "mean_abs", "rel_max", "rel_mean", "status")

    def __init__(self, name: str, max_abs: float, mean_abs: float,
                 rel_max: float, rel_mean: float, status: str):
        self.name     = name
        self.max_abs  = max_abs
        self.mean_abs = mean_abs
        self.rel_max  = rel_max
        self.rel_mean = rel_mean
        self.status   = status


def compare_dirs(
    baseline: Path,
    variant: Path,
    *,
    rel_max_tol: float,
    rel_mean_tol: float,
) -> tuple[bool, list[Row]]:
    """Compare each tensor that exists in BOTH dirs. Returns (passed,
    rows). For each tensor we compute both absolute (max/mean abs
    diff) and relative (diff scaled by the baseline magnitude) error
    metrics. Status is judged on the relative metrics so a single set
    of bands works across post-LayerNorm activations and the
    pre-norm residual intermediates whose magnitudes drift through
    the encoder.
    """
    names = sorted(discover_names(baseline) & discover_names(variant))
    rows: list[Row] = []
    passed = True
    for name in names:
        bs, bd = load_dump(baseline, name)
        vs, vd = load_dump(variant, name)
        if bd is None or vd is None or bs != vs or bd.size != vd.size:
            rows.append(Row(name, float("nan"), float("nan"),
                            float("nan"), float("nan"), "SHAPE/MISSING"))
            passed = False
            continue

        b64 = bd.astype(np.float64)
        v64 = vd.astype(np.float64)
        diff = np.abs(b64 - v64)
        max_abs  = float(diff.max())  if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0

        # Relative metrics. Use the baseline's max/mean magnitude as
        # the denominator. Tensors that are entirely zero (rare —
        # dec.embed.0 at the start state, dec.lstm.*.0 before any
        # update) get rel = 0 by convention to avoid div-by-zero
        # producing NaN that pollutes the rest of the table.
        b_max  = float(np.abs(b64).max())  if b64.size else 0.0
        b_mean = float(np.abs(b64).mean()) if b64.size else 0.0
        rel_max  = max_abs  / b_max  if b_max  > 0 else 0.0
        rel_mean = mean_abs / b_mean if b_mean > 0 else 0.0

        status = "ok"
        if rel_max > rel_max_tol or rel_mean > rel_mean_tol:
            status = "FAIL"
            passed = False
        rows.append(Row(name, max_abs, mean_abs, rel_max, rel_mean, status))

    # Failures first, then by rel_mean descending so the worst
    # tensors are at the top of the table.
    rows.sort(key=lambda r: (r.status == "ok", -r.rel_mean))
    return passed, rows


def print_table(
    family: str,
    variant_path: Path,
    rows: list[Row],
    rel_max_tol: float,
    rel_mean_tol: float,
    *,
    head: int = 12,
) -> None:
    print(f"\n=== {family}: {variant_path.name} ===")
    print(f"  bands: rel_max<{rel_max_tol:g}  rel_mean<{rel_mean_tol:g}")
    print(f"  {'tensor':36} {'max_abs':>11} {'mean_abs':>11} "
          f"{'rel_max':>10} {'rel_mean':>10}  status")
    print("  " + "-" * 91)
    for r in rows[:head]:
        if np.isnan(r.max_abs):
            print(f"  {r.name:36} {'—':>11} {'—':>11} {'—':>10} {'—':>10}  {r.status}")
        else:
            print(f"  {r.name:36} {r.max_abs:11.3e} {r.mean_abs:11.3e} "
                  f"{r.rel_max:10.3e} {r.rel_mean:10.3e}  {r.status}")
    if len(rows) > head:
        n_fail = sum(1 for r in rows if r.status != "ok")
        print(f"  ... {len(rows) - head} more rows ({n_fail} total fails)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("variants", nargs="+", type=Path,
                   help="GGUF files to compare against the baseline")
    p.add_argument("--baseline", type=Path, default=None,
                   help="Baseline GGUF (default: <repo>/models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf)")
    p.add_argument("--audio", type=Path, default=None,
                   help="Audio sample (default: <repo>/samples/jfk.wav)")
    p.add_argument("--cli", type=Path, default=None,
                   help="transcribe-cli path (default: <repo>/build/bin/transcribe-cli)")
    p.add_argument("--dump-root", type=Path, default=None,
                   help="Where to put dump dirs (default: /tmp/parakeet_quant_dumps)")
    p.add_argument("--head", type=int, default=10,
                   help="Show top N rows per variant (default 10)")
    p.add_argument("--keep-dumps", action="store_true",
                   help="Don't delete dump dirs on exit")
    args = p.parse_args()

    repo = find_repo_root(Path(__file__).parent)
    baseline = args.baseline or repo / "models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf"
    audio    = args.audio    or repo / "samples/jfk.wav"
    cli      = args.cli      or repo / "build/bin/transcribe-cli"
    dump_root = args.dump_root or Path("/tmp/parakeet_quant_dumps")

    for path in (baseline, audio, cli):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    print(f"baseline: {baseline}")
    print(f"audio:    {audio}")
    print(f"cli:      {cli}")
    print(f"dumps:    {dump_root}\n")

    print("--- generating baseline dumps ---")
    baseline_dump = dump_root / "baseline"
    run_dump(cli, baseline, audio, baseline_dump)

    overall_passed = True
    summary: list[tuple[str, Path, bool, int, int]] = []  # family, path, passed, n_total, n_fail

    for variant in args.variants:
        if not variant.exists():
            print(f"error: {variant} does not exist", file=sys.stderr)
            overall_passed = False
            continue
        family = detect_family(variant)
        rel_max_tol, rel_mean_tol = QUANT_BANDS[family]

        print(f"\n--- generating {family} dumps ---")
        variant_dump = dump_root / family
        run_dump(cli, variant, audio, variant_dump)

        passed, rows = compare_dirs(
            baseline_dump, variant_dump,
            rel_max_tol=rel_max_tol, rel_mean_tol=rel_mean_tol,
        )
        print_table(family, variant, rows, rel_max_tol, rel_mean_tol,
                    head=args.head)
        n_fail = sum(1 for r in rows if r.status != "ok")
        summary.append((family, variant, passed, len(rows), n_fail))
        overall_passed = overall_passed and passed

    print("\n=== summary ===")
    print(f"  {'family':10} {'tensors':>9} {'fails':>7}  result")
    print("  " + "-" * 40)
    for family, _, passed, n, n_fail in summary:
        print(f"  {family:10} {n:>9} {n_fail:>7}  {'OK' if passed else 'FAIL'}")

    if not args.keep_dumps:
        shutil.rmtree(dump_root, ignore_errors=True)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
