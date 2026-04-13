#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
compare.py - compare bench reports and show a timing delta table.

Takes two explicit sets of report files: baseline(s) and candidate(s).
Each set is merged by (family, backend, quant, sample) key; if the
same cell appears in multiple files within a set, the last one wins.
The merged baseline is diffed against the merged candidate.

Usage:
    uv run scripts/bench/compare.py \\
        --baseline reports/perf/apple-m4-max/pre-refactor_parakeet_metal.json \\
        --candidate reports/perf/apple-m4-max/post-refactor_parakeet_metal.json

    # Multiple files per side (e.g. shell glob):
    uv run scripts/bench/compare.py \\
        --baseline reports/perf/apple-m4-max/pre-refactor_*.json \\
        --candidate reports/perf/apple-m4-max/post-refactor_*.json

    # Regression gate with threshold:
    uv run scripts/bench/compare.py --threshold 5.0 \\
        --baseline reports/perf/apple-m4-max/baseline_*.json \\
        --candidate reports/perf/apple-m4-max/candidate_*.json

Output:

    baseline: pre-refactor (abc1234)  vs  post-refactor (def5678)

    family     backend  quant     sample   wall_ms(A)  wall_ms(B)   delta%  status
    parakeet   metal    f16       jfk         120.3       115.1      -4.3%  ok
    parakeet   metal    q8_0     jfk         105.2       108.7      +3.3%  ok
    cohere     metal    f16       jfk         340.1       289.4     -14.9%  ok

Options:
    --threshold PCT     fail (exit 1) if any cell regresses by more than PCT%
                        (default: disabled). Only wall_ms regressions count.
    --fail-on-missing   fail (exit 1) if any cell is in baseline but not in
                        candidate, or vice versa.
    --key FIELD         timing field to compare (default: wall_ms)
    --quiet             only print regressions and new/gone cells
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunKey:
    family: str
    backend: str
    quant: str
    sample: str

    def __hash__(self) -> int:
        return hash((self.family, self.backend, self.quant, self.sample))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunKey):
            return NotImplemented
        return (self.family == other.family and self.backend == other.backend
                and self.quant == other.quant and self.sample == other.sample)


@dataclass(slots=True)
class RunValue:
    wall_ms: float
    encode_ms: float
    decode_ms: float
    total_ms: float
    mel_ms: float
    rtf: float


def parse_quant_from_path(model_path: str) -> str:
    """Extract quant from a model path like 'parakeet-tdt-0.6b-v2.q4_k_m.gguf'."""
    name = Path(model_path).name
    if not name.endswith(".gguf"):
        return "unknown"
    stem = name[: -len(".gguf")]
    if "." not in stem:
        return "unknown"
    return stem.rsplit(".", 1)[1]


def extract_mean(summary: dict, field: str) -> float:
    """Get the mean value from a summary stat block."""
    block = summary.get(field)
    if isinstance(block, dict):
        return float(block.get("mean", 0.0))
    return 0.0


def sample_stem(sample_path: str) -> str:
    """Extract sample stem from path."""
    return Path(sample_path).stem


def load_report(path: Path) -> tuple[str, str, dict[RunKey, RunValue]]:
    """Load a bench report (driver-v1 aggregate or raw bench-v1/v2).

    Returns (label, git_sha, {RunKey: RunValue}).
    """
    data = json.loads(path.read_text())
    schema = data.get("schema", "")

    runs: dict[RunKey, RunValue] = {}

    if schema == "transcribe-bench-driver-v1":
        label = data.get("name") or path.stem
        git_sha = data.get("git_sha", "unknown")
        family = data.get("family", "unknown")
        backend = data.get("backend", "unknown")

        for r in data.get("runs", []):
            summary = r.get("summary", {})
            quant = parse_quant_from_path(r.get("model_path", ""))
            sample = sample_stem(r.get("sample_path", ""))
            rtf = r.get("rtf_wall_mean") or r.get("rtf_mean") or 0.0
            key = RunKey(family=family, backend=backend, quant=quant,
                         sample=sample)
            runs[key] = RunValue(
                wall_ms=extract_mean(summary, "wall_ms"),
                encode_ms=extract_mean(summary, "encode_ms"),
                decode_ms=extract_mean(summary, "decode_ms"),
                total_ms=extract_mean(summary, "total_ms"),
                mel_ms=extract_mean(summary, "mel_ms"),
                rtf=float(rtf),
            )

    elif schema in ("transcribe-bench-v1", "transcribe-bench-v2"):
        label = path.stem
        git_sha = "unknown"
        summary = data.get("summary", {})
        backend = data.get("backend", "unknown")
        quant = parse_quant_from_path(data.get("model_path", ""))
        sample = sample_stem(data.get("sample_path", ""))
        family = Path(data.get("model_path", "")).parent.name or "unknown"
        rtf = data.get("rtf_wall_mean") or data.get("rtf_mean") or 0.0
        key = RunKey(family=family, backend=backend, quant=quant,
                     sample=sample)
        runs[key] = RunValue(
            wall_ms=extract_mean(summary, "wall_ms"),
            encode_ms=extract_mean(summary, "encode_ms"),
            decode_ms=extract_mean(summary, "decode_ms"),
            total_ms=extract_mean(summary, "total_ms"),
            mel_ms=extract_mean(summary, "mel_ms"),
            rtf=float(rtf),
        )

    else:
        print(f"error: {path}: unknown schema {schema!r}", file=sys.stderr)
        sys.exit(2)

    return label, git_sha, runs


def merge_reports(
    paths: list[Path],
) -> tuple[str, str, dict[RunKey, RunValue]]:
    """Load and merge multiple reports into one (label, sha, runs) triple.

    Last-writer-wins on duplicate keys. Labels and shas are joined
    with '+' if they differ across files.
    """
    labels: list[str] = []
    shas: list[str] = []
    merged: dict[RunKey, RunValue] = {}

    for p in paths:
        label, sha, runs = load_report(p)
        if label not in labels:
            labels.append(label)
        if sha not in shas:
            shas.append(sha)
        merged.update(runs)

    return "+".join(labels), "+".join(shas), merged


def get_field(rv: RunValue, field: str) -> float:
    return getattr(rv, field, 0.0)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare bench reports and show timing deltas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", nargs="+", type=Path, required=True,
                   help="Baseline bench report JSON file(s)")
    p.add_argument("--candidate", nargs="+", type=Path, required=True,
                   help="Candidate bench report JSON file(s)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fail if any cell regresses by more than this %% "
                        "(e.g. 5.0 for 5%%)")
    p.add_argument("--fail-on-missing", action="store_true",
                   help="Fail if any cell is present on one side but not "
                        "the other (new or gone)")
    p.add_argument("--key", type=str, default="wall_ms",
                   choices=["wall_ms", "total_ms", "encode_ms", "decode_ms",
                            "mel_ms"],
                   help="Timing field to compare (default: wall_ms)")
    p.add_argument("--quiet", action="store_true",
                   help="Only print regressions and new/gone cells")
    args = p.parse_args()

    # Validate paths exist.
    for side_name, paths in [("baseline", args.baseline),
                              ("candidate", args.candidate)]:
        for rp in paths:
            if not rp.exists():
                print(f"error: {side_name} file does not exist: {rp}",
                      file=sys.stderr)
                return 2

    # Merge each side.
    base_label, base_sha, base_runs = merge_reports(args.baseline)
    cand_label, cand_sha, cand_runs = merge_reports(args.candidate)

    print(f"\nbaseline: {base_label} ({base_sha})"
          f"  vs  {cand_label} ({cand_sha})")
    print(f"comparing: {args.key}\n")

    field = args.key

    # Collect all keys from both sides.
    all_keys = sorted(
        base_runs.keys() | cand_runs.keys(),
        key=lambda k: (k.family, k.backend, k.quant, k.sample),
    )

    header = ("family", "backend", "quant", "sample",
              f"{field}(A)", f"{field}(B)", "delta%", "status")
    rows: list[tuple[str, ...]] = []
    regressions: list[tuple[RunKey, float]] = []
    missing_cells: list[tuple[RunKey, str]] = []  # (key, "new" | "gone")

    for key in all_keys:
        bv = base_runs.get(key)
        cv = cand_runs.get(key)

        if bv is None:
            rows.append((key.family, key.backend, key.quant, key.sample,
                         "-", f"{get_field(cv, field):.1f}", "new", "new"))
            missing_cells.append((key, "new"))
            continue
        if cv is None:
            rows.append((key.family, key.backend, key.quant, key.sample,
                         f"{get_field(bv, field):.1f}", "-", "gone", "gone"))
            missing_cells.append((key, "gone"))
            continue

        a = get_field(bv, field)
        b = get_field(cv, field)
        if a > 0:
            delta_pct = ((b - a) / a) * 100.0
        else:
            delta_pct = 0.0

        # Positive delta = regression (slower), negative = improvement.
        status = "ok"
        if args.threshold is not None and delta_pct > args.threshold:
            status = "REGRESSED"
            regressions.append((key, delta_pct))

        delta_str = f"{delta_pct:+.1f}%"
        rows.append((key.family, key.backend, key.quant, key.sample,
                     f"{a:.1f}", f"{b:.1f}", delta_str, status))

    if not rows:
        print("(no matching cells)")
        return 0

    # Print table.
    widths = [max(len(header[j]),
                  max((len(row[j]) for row in rows), default=0))
              for j in range(len(header))]

    def fmt_row(row: tuple[str, ...]) -> str:
        parts = [row[j].ljust(widths[j]) if j < 4
                 else row[j].rjust(widths[j])
                 for j in range(len(row))]
        return "  " + "   ".join(parts)

    if not args.quiet:
        print(fmt_row(header))
        print("  " + "-" * (sum(widths) + 3 * (len(widths) - 1)))

    for row in rows:
        is_notable = row[7] in ("REGRESSED", "new", "gone")
        if args.quiet and not is_notable:
            continue
        print(fmt_row(row))

    # Summary.
    n_cells = sum(1 for k in all_keys
                  if k in base_runs and k in cand_runs)
    exit_code = 0

    if regressions:
        print(f"\n{len(regressions)} regression(s) exceed "
              f"threshold {args.threshold:.1f}%:")
        for key, pct in regressions:
            print(f"  {key.family}/{key.backend}/{key.quant}/{key.sample}"
                  f": {pct:+.1f}%")
        exit_code = 1

    if missing_cells:
        new_count = sum(1 for _, kind in missing_cells if kind == "new")
        gone_count = sum(1 for _, kind in missing_cells if kind == "gone")
        parts = []
        if new_count:
            parts.append(f"{new_count} new")
        if gone_count:
            parts.append(f"{gone_count} gone")
        print(f"\n{', '.join(parts)} cell(s) between baseline and candidate")
        if args.fail_on_missing:
            exit_code = 1

    if exit_code == 0:
        detail = ""
        if args.threshold is not None:
            detail += f" (threshold {args.threshold:.1f}%)"
        print(f"\n{n_cells} cell(s) compared, no regressions{detail}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
