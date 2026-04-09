#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
compare.py — print a WER delta table from two or more score files.

Usage:
    uv run scripts/wer/compare.py \\
        reports/wer/parakeet-tdt-0.6b-v2.f32.test-clean.score.json \\
        reports/wer/parakeet-tdt-0.6b-v2.q8_0.test-clean.score.json \\
        reports/wer/parakeet-tdt-0.6b-v2.q4_k_m.test-clean.score.json

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
# parakeet-tdt-0.6b-v2.q4_k_m.test-clean.score.json → q4_k_m
FAMILY_PAT = re.compile(
    r"\.(?P<family>f32|f16|q8_0|q5_k_m|q4_k_m)\."
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

    baseline_wer = scores[0][1]["wer_pct"]

    # Header.
    print(f"\n{'variant':12} {'WER%':>7} {'delta':>7} "
          f"{'CI low':>8} {'CI high':>8} "
          f"{'lat_p50':>8} {'N':>6}")
    print("-" * 65)

    for family, s in scores:
        wer_pct = s["wer_pct"]
        ci_lo   = s.get("wer_ci_lo", 0) * 100
        ci_hi   = s.get("wer_ci_hi", 0) * 100
        lat_p50 = s.get("latency_p50_ms", 0)
        n       = s.get("n", 0)

        if family == scores[0][0]:
            delta_str = "—"
        else:
            delta = wer_pct - baseline_wer
            delta_str = f"{delta:+.2f}"

        print(f"{family:12} {wer_pct:7.2f} {delta_str:>7} "
              f"{ci_lo:8.2f} {ci_hi:8.2f} "
              f"{lat_p50:8.0f} {n:6}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
