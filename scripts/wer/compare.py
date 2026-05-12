#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
compare.py — print a WER delta table from two or more score files.

Usage:
    uv run scripts/wer/compare.py \\
        reports/wer/parakeet-tdt-0.6b-v2-F32.test-clean.score.json \\
        reports/wer/parakeet-tdt-0.6b-v2-Q8_0.test-clean.score.json \\
        reports/wer/parakeet-tdt-0.6b-v2-Q4_K_M.test-clean.score.json

The first file is treated as the baseline. Output:

    variant     WER%   delta   CI low  CI high  lat_p50
    f32          3.12   —       2.85    3.40     720
    q8_0         3.14  +0.02    2.87    3.42     620
    q4_k_m       3.38  +0.26    3.10    3.66     580
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Extract quant family from score filename.
# parakeet-tdt-0.6b-v2-Q4_K_M.test-clean.score.json → q4_k_m
FAMILY_PAT = re.compile(
    r"\.(?P<family>f32|bf16|f16|q8_0|q6_k|q5_k_m|q5_0|q5_1|q4_k_m|q4_0|q4_1)\."
)


def extract_family(path: Path) -> str:
    m = FAMILY_PAT.search(path.name)
    return m.group("family") if m else path.stem


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scores", nargs="+", type=Path,
                   help="Score JSON files (first = baseline)")
    args = p.parse_args()

    scores: list[tuple[str, dict]] = []
    for sp in args.scores:
        if not sp.exists():
            print(f"error: {sp} does not exist", file=sys.stderr)
            return 2
        with open(sp) as f:
            scores.append((extract_family(sp), json.load(f)))

    if not scores:
        print("error: no score files", file=sys.stderr)
        return 2

    # Score files now write `error_rate_pct` as the canonical metric
    # field (WER or CER, depending on the language). Older files only
    # have `wer_pct`; fall back to those for backward compat. The
    # `metric` field labels the column header (defaults to WER).
    def rate_pct(rec: dict) -> float:
        return rec.get("error_rate_pct", rec.get("wer_pct", 0.0))

    def rate_ci(rec: dict, side: str) -> float:
        key_new = f"error_rate_ci_{side}"
        key_old = f"wer_ci_{side}"
        return rec.get(key_new, rec.get(key_old, 0.0)) * 100

    baseline_rate = rate_pct(scores[0][1])
    metric_label = scores[0][1].get("metric", "wer").upper()

    # Header.
    print(f"\n{'variant':12} {metric_label+'%':>7} {'delta':>7} "
          f"{'CI low':>8} {'CI high':>8} "
          f"{'lat_p50':>8} {'N':>6}")
    print("-" * 65)

    for family, s in scores:
        rate = rate_pct(s)
        ci_lo = rate_ci(s, "lo")
        ci_hi = rate_ci(s, "hi")
        lat_p50 = s.get("latency_p50_ms", 0)
        n       = s.get("n", 0)

        if family == scores[0][0]:
            delta_str = "—"
        else:
            delta = rate - baseline_rate
            delta_str = f"{delta:+.2f}"

        print(f"{family:12} {rate:7.2f} {delta_str:>7} "
              f"{ci_lo:8.2f} {ci_hi:8.2f} "
              f"{lat_p50:8.0f} {n:6}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
