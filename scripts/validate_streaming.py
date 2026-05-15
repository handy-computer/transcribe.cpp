#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
validate_streaming.py - parity harness for cache-aware streaming Parakeet.

For each (model, audio, att_context_right) tuple, runs:

  1. Python NeMo streaming dumper:
       scripts/dump_reference_parakeet_nemo.py streaming ...
     producing per-chunk reference tensors under <out>/R<R>/ref/.

  2. C++ transcribe-cli with --stream-chunk-ms <chunk> --stream-att-right R
     and TRANSCRIBE_DUMP_DIR=<out>/R<R>/cpp/, producing the parallel
     C++ dumps.

  3. scripts/compare_tensors.py to diff matching tensor names
     between the two dump dirs.

  4. Final-transcript edit distance (Levenshtein on tokens).

Usage:
    uv run --project scripts/envs/parakeet \\
        scripts/validate_streaming.py \\
        --hf-model nvidia/nemotron-speech-streaming-en-0.6b \\
        --gguf models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-F32.gguf \\
        --audio samples/jfk.wav \\
        --right 13 6 1 0 \\
        --out build/validate_streaming/nemotron/jfk \\
        --stream-chunk-ms 500

Per-R outputs:
    <out>/R<R>/ref/    NeMo reference dumps + stream_history.json
    <out>/R<R>/cpp/    C++ dumps
    <out>/R<R>/summary.json   diff stats + transcripts
    <out>/summary.txt          one-line-per-R parity table

Tolerances: configurable via --max-abs / --mean-abs (same defaults as
compare_tensors.py). Per-tensor overrides via --tolerances <file.json>.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_subprocess(cmd: list[str], *, label: str, env: dict[str, str] | None = None,
                   cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"+ {label}: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        env=({**os.environ, **env} if env else None),
        cwd=str(cwd) if cwd else None,
        capture_output=capture,
        text=True,
    )
    if proc.returncode != 0:
        if capture:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"error: {label} returned {proc.returncode}")
    return proc


# ---------------------------------------------------------------------------
# Reference + C++ dump invocations
# ---------------------------------------------------------------------------


def run_ref_dump(*, hf_model: str, audio: Path, out_dir: Path, right: int,
                 force: bool, pad_and_drop: bool) -> Path:
    """Run NeMo streaming dumper into out_dir. Idempotent: skips when
    stream_history.json already exists, unless --force is set."""
    stamp = out_dir / "stream_history.json"
    if stamp.exists() and not force:
        print(f"  [skip ref] {stamp} already exists (pass --force to re-run)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--project", str(REPO / "scripts/envs/parakeet"),
        str(REPO / "scripts/dump_reference_parakeet_nemo.py"),
        "streaming",
        "--model", hf_model,
        "--audio", str(audio),
        "--out", str(out_dir),
        "--att-context-right", str(right),
    ]
    if pad_and_drop:
        cmd.append("--pad-and-drop-preencoded")
    run_subprocess(cmd, label=f"ref dump R={right}", cwd=REPO)
    return out_dir


def run_cpp_dump(*, gguf: Path, audio: Path, out_dir: Path, right: int,
                 stream_chunk_ms: int, backend: str | None = None,
                 threads: int | None = None) -> tuple[Path, str]:
    """Run transcribe-cli with TRANSCRIBE_DUMP_DIR set to out_dir and
    return (out_dir, final_transcript_text). When backend is 'cpu' the
    Metal backend is bypassed so the only remaining drift sources are
    fp32 reduction order + libm vs PyTorch CPU."""
    # Always clear out_dir to avoid mixing stale files into the diff.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cli = REPO / "build/bin/transcribe-cli"
    if not cli.exists():
        raise SystemExit(
            f"error: {cli} not built; run cmake --build build --target transcribe-cli"
        )

    cmd = [
        str(cli),
        "--model", str(gguf),
        "--stream-chunk-ms", str(stream_chunk_ms),
        "--stream-att-right", str(right),
        str(audio),
    ]
    if backend is not None:
        cmd += ["--backend", backend]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    env = {"TRANSCRIBE_DUMP_DIR": str(out_dir)}
    proc = run_subprocess(cmd, label=f"cpp dump R={right}", env=env,
                          cwd=REPO, capture=True)
    text = ""
    for line in proc.stdout.splitlines():
        if line.startswith("text:"):
            text = line[len("text:"):].strip()
            break
    return out_dir, text


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class CompareSummary:
    n_pairs: int = 0
    n_ok: int = 0
    n_mismatch: int = 0
    n_missing_cpp: int = 0
    n_missing_ref: int = 0
    worst: list[dict[str, Any]] = field(default_factory=list)


def write_stage2_tolerances(ref_dir: Path, out_path: Path) -> None:
    """Walk every ref sidecar (.json) and write a per-tensor tolerance
    map that matches /porting-2-oracle Stage 2's magnitude-aware recipe:

        max_abs  = max(1e-4 × p99_abs, 1e-6)
        mean_abs = max(1e-5 × rms,     1e-6)

    Pure-lookup zero tensors (channel_len when its only valid value is
    a scalar count; ref p99_abs=70, rms=70 — not actually zero) keep
    their magnitude-aware budget. Sidecars without p99_abs/rms get the
    1e-6 floor.
    """
    tolerances: dict[str, dict[str, float]] = {}
    for sidecar in sorted(ref_dir.glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            continue
        name = meta.get("name") or sidecar.stem
        p99 = float(meta.get("p99_abs", 0.0))
        rms = float(meta.get("rms", 0.0))
        tolerances[name] = {
            "max_abs":  max(1e-4 * p99, 1e-6),
            "mean_abs": max(1e-5 * rms, 1e-6),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tolerances, indent=2))


def expand_pattern_tolerances(ref_dir: Path, pattern_path: Path,
                              out_path: Path) -> None:
    """Expand a pattern-keyed tolerance file (one entry per tensor 'kind'
    — e.g. cache_lc_out, mel_in, enc_out) into a per-tensor tolerance
    map that compare_tensors.py can consume directly.

    For each stream.chunk.<N>.<tail> tensor seen in ref_dir, the matching
    pattern entry is selected by stripping a trailing _<layer> suffix
    from <tail> (so cache_lc_out_0, cache_lc_out_12, etc. all map to
    pattern["cache_lc_out"]). Comment fields starting with _ are
    dropped from the emitted tolerance map.
    """
    pattern = json.loads(pattern_path.read_text())
    pattern = {k: v for k, v in pattern.items()
               if not k.startswith("_") and isinstance(v, dict)}
    expanded: dict[str, dict[str, float]] = {}
    for f in sorted(ref_dir.glob("*.f32")):
        name = f.stem
        tail = name.split(".")[-1]
        kind = tail
        if "_" in tail:
            head = tail.rsplit("_", 1)
            if head[-1].isdigit():
                kind = head[0]
        if kind in pattern:
            entry = pattern[kind]
            expanded[name] = {k: v for k, v in entry.items()
                              if not k.startswith("_")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(expanded, indent=2))


def run_compare_tensors(cpp_dir: Path, ref_dir: Path, *,
                        max_abs: float, mean_abs: float,
                        per_tensor_tolerances: Path | None = None,
                        ) -> CompareSummary:
    """Invoke compare_tensors.py and parse its output. Returns a
    CompareSummary; does NOT raise on tolerance failures (we want to
    surface the numbers regardless)."""
    script = REPO / "scripts/compare_tensors.py"
    cmd = [
        "uv", "run", "--project", str(REPO / "scripts/envs/parakeet"),
        str(script),
        str(cpp_dir), str(ref_dir),
        "--max-abs", str(max_abs),
        "--mean-abs", str(mean_abs),
    ]
    if per_tensor_tolerances is not None:
        cmd += ["--tolerances", str(per_tensor_tolerances)]
    proc = subprocess.run(
        cmd, cwd=str(REPO),
        capture_output=True, text=True,
    )
    summary = CompareSummary()
    # compare_tensors.py prints a line per tensor (after a header). The
    # second column is the status: "ok" / "FAIL" / "SHAPE" / "L-ONLY" /
    # "R-ONLY". Header rows are skipped (they don't have a status token
    # in column 2).
    valid_status = {"ok", "FAIL", "SHAPE", "L-ONLY", "R-ONLY"}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1] not in valid_status:
            continue
        status = parts[1]
        summary.n_pairs += 1
        if status == "ok":
            summary.n_ok += 1
        elif status == "FAIL" or status == "SHAPE":
            summary.n_mismatch += 1
            summary.worst.append({"line": line.strip()})
        elif status == "L-ONLY":
            summary.n_missing_ref += 1
        elif status == "R-ONLY":
            summary.n_missing_cpp += 1
    if proc.returncode not in (0, 1):
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"error: compare_tensors returned {proc.returncode}")
    return summary


# ---------------------------------------------------------------------------
# Transcript distance
# ---------------------------------------------------------------------------


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def edit_distance(a: str, b: str) -> int:
    """Plain Levenshtein on characters (jfk.wav is short). For bigger
    inputs swap for python-Levenshtein or jiwer.compute_measures."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def read_ref_transcript(ref_dir: Path) -> str:
    """Read the final transcript from stream_history.json."""
    hist = ref_dir / "stream_history.json"
    if not hist.exists():
        return ""
    data = json.loads(hist.read_text())
    pct = data.get("per_chunk_text") or []
    return pct[-1] if pct else ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Parity harness for cache-aware streaming Parakeet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hf-model", required=True,
                   help="HF model id (e.g. nvidia/nemotron-speech-streaming-en-0.6b)")
    p.add_argument("--gguf", required=True, type=Path,
                   help="C++ side GGUF path")
    p.add_argument("--audio", required=True, type=Path,
                   help="16 kHz mono wav file")
    p.add_argument("--out", required=True, type=Path,
                   help="Base output directory for dumps + summary")
    p.add_argument("--right", type=int, nargs="+", default=[13, 6, 1, 0],
                   help="att_context_right values to test (default: 13 6 1 0)")
    p.add_argument("--stream-chunk-ms", type=int, default=500,
                   help="PCM feed cadence on the C++ side (default: 500)")
    p.add_argument("--max-abs", type=float, default=1e-3,
                   help="compare_tensors max-abs tolerance (default: 1e-3)")
    p.add_argument("--mean-abs", type=float, default=1e-4,
                   help="compare_tensors mean-abs tolerance (default: 1e-4)")
    p.add_argument("--force", action="store_true",
                   help="Re-run reference dump even if cached")
    p.add_argument("--pad-and-drop", action="store_true",
                   help="Reference: treat first chunk like subsequent")
    p.add_argument("--skip-compare", action="store_true",
                   help="Run dumps only; don't invoke compare_tensors.py")
    p.add_argument("--stage2-tolerances", action="store_true",
                   help="Generate magnitude-aware per-tensor tolerances from "
                        "ref sidecars (max(1e-4*p99_abs, 1e-6) / "
                        "max(1e-5*rms, 1e-6)) per the /porting-2-oracle "
                        "Stage 2 recipe. Overrides --max-abs / --mean-abs.")
    p.add_argument("--tolerances", type=Path, default=None,
                   help="Path to a pattern-keyed tolerance file (one entry "
                        "per tensor 'kind') — e.g. "
                        "tests/tolerances/<variant>.streaming.json. The "
                        "harness expands kind entries to per-tensor entries "
                        "at run time. Overrides --stage2-tolerances when both "
                        "are passed.")
    p.add_argument("--backend", default=None,
                   help="C++ compute backend (cpu, metal, etc.). Passed to "
                        "transcribe-cli via --backend; omit to use the CLI "
                        "default (auto = Metal on macOS).")
    p.add_argument("--threads", type=int, default=None,
                   help="C++ thread count. Omit for transcribe-cli default.")
    args = p.parse_args()

    if not args.gguf.exists():
        raise SystemExit(f"error: gguf not found: {args.gguf}")
    if not args.audio.exists():
        raise SystemExit(f"error: audio not found: {args.audio}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for R in args.right:
        print(f"\n========== att_context_right={R} ==========")
        r_dir = args.out / f"R{R}"
        ref_dir = r_dir / "ref"
        cpp_dir = r_dir / "cpp"

        try:
            run_ref_dump(
                hf_model=args.hf_model,
                audio=args.audio.resolve(),
                out_dir=ref_dir,
                right=R,
                force=args.force,
                pad_and_drop=args.pad_and_drop,
            )
            ref_text = read_ref_transcript(ref_dir)
        except SystemExit as e:
            print(f"  [ref FAIL] {e}")
            rows.append({"right": R, "status": "ref_failed", "error": str(e)})
            continue

        try:
            _, cpp_text = run_cpp_dump(
                gguf=args.gguf.resolve(),
                audio=args.audio.resolve(),
                out_dir=cpp_dir,
                right=R,
                stream_chunk_ms=args.stream_chunk_ms,
                backend=args.backend,
                threads=args.threads,
            )
        except SystemExit as e:
            print(f"  [cpp FAIL] {e}")
            rows.append({
                "right": R, "status": "cpp_failed", "error": str(e),
                "ref_text": ref_text,
            })
            continue

        comp_summary: CompareSummary | None = None
        if not args.skip_compare:
            per_tensor_tol: Path | None = None
            if args.tolerances is not None:
                per_tensor_tol = r_dir / "expanded_tolerances.json"
                expand_pattern_tolerances(ref_dir, args.tolerances, per_tensor_tol)
            elif args.stage2_tolerances:
                per_tensor_tol = r_dir / "stage2_tolerances.json"
                write_stage2_tolerances(ref_dir, per_tensor_tol)
            comp_summary = run_compare_tensors(
                cpp_dir=cpp_dir, ref_dir=ref_dir,
                max_abs=args.max_abs, mean_abs=args.mean_abs,
                per_tensor_tolerances=per_tensor_tol,
            )

        ref_norm = normalize_text(ref_text)
        cpp_norm = normalize_text(cpp_text)
        char_dist = edit_distance(ref_norm, cpp_norm)
        rel = char_dist / max(len(ref_norm), 1)

        per_r = {
            "right": R,
            "ref_text": ref_text,
            "cpp_text": cpp_text,
            "ref_text_norm": ref_norm,
            "cpp_text_norm": cpp_norm,
            "char_edit_distance": char_dist,
            "relative_distance": rel,
        }
        if comp_summary is not None:
            per_r["tensor_compare"] = {
                "n_pairs": comp_summary.n_pairs,
                "n_ok": comp_summary.n_ok,
                "n_mismatch": comp_summary.n_mismatch,
                "n_missing_cpp": comp_summary.n_missing_cpp,
                "n_missing_ref": comp_summary.n_missing_ref,
            }
        (r_dir / "summary.json").write_text(json.dumps(per_r, indent=2))
        rows.append(per_r)

        print(f"  ref text:  {ref_text}")
        print(f"  cpp text:  {cpp_text}")
        print(f"  edit_dist: {char_dist} ({rel:.1%} of ref length)")
        if comp_summary is not None:
            print(f"  tensors:   {comp_summary.n_ok}/{comp_summary.n_pairs} ok "
                  f"({comp_summary.n_mismatch} fail, "
                  f"{comp_summary.n_missing_cpp} missing cpp, "
                  f"{comp_summary.n_missing_ref} missing ref)")

    # One-line-per-R summary
    print("\n========== summary ==========")
    summary_lines = [
        f"{'R':>3}  {'tensors_ok':>12}  {'tensors_fail':>12}  "
        f"{'edit_dist':>10}  {'rel':>6}  text"
    ]
    for row in rows:
        if row.get("status") in ("ref_failed", "cpp_failed"):
            summary_lines.append(
                f"{row['right']:>3}  {row['status']}  {row.get('error', '')}"
            )
            continue
        tc = row.get("tensor_compare", {})
        summary_lines.append(
            f"{row['right']:>3}  "
            f"{tc.get('n_ok', '-'):>12}  "
            f"{tc.get('n_mismatch', '-'):>12}  "
            f"{row['char_edit_distance']:>10}  "
            f"{row['relative_distance']:>6.1%}  "
            f"{row['cpp_text'][:80]}"
        )
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (args.out / "summary.txt").write_text(summary_text + "\n")

    # Exit nonzero if any row failed (transcripts disagree heavily OR
    # any tensor exceeded tolerance). 5% relative edit distance is a
    # rough heuristic for "obviously broken".
    any_bad = False
    for row in rows:
        if row.get("status"):
            any_bad = True
        elif row.get("relative_distance", 0.0) > 0.05:
            any_bad = True
        elif row.get("tensor_compare", {}).get("n_mismatch", 0) > 0:
            any_bad = True
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
