#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""compare_hyps.py - per-utterance hypothesis diff between two WER report JSONLs.

Aggregate WER can match while individual utterances differ, so a WER delta of
0.00 pp is NOT evidence that two runs produced the same text. This walks both
reports by `id` and reports exact-match rate plus the first N divergences.

The intended use is gating a C++ port against the Stage 2 reference run, which
kept every per-utterance hypothesis for exactly this purpose.

Usage:
    uv run scripts/wer/compare_hyps.py REF.jsonl HYP.jsonl [--show N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uid = d.get("id")
            if uid is None:
                continue
            # Report writers use hyp_text; fall back to hyp/text for older files.
            hyp = d.get("hyp_text")
            if hyp is None:
                hyp = d.get("hyp", d.get("text", ""))
            out[str(uid)] = (hyp or "").strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref", type=Path)
    p.add_argument("hyp", type=Path)
    p.add_argument("--show", type=int, default=5,
                   help="how many divergent utterances to print (default 5)")
    args = p.parse_args()

    ref = load(args.ref)
    hyp = load(args.hyp)

    only_ref = sorted(set(ref) - set(hyp))
    only_hyp = sorted(set(hyp) - set(ref))
    shared = sorted(set(ref) & set(hyp))

    diffs = [u for u in shared if ref[u] != hyp[u]]
    n = len(shared)
    same = n - len(diffs)

    print(f"ref : {args.ref}  ({len(ref)} utts)")
    print(f"hyp : {args.hyp}  ({len(hyp)} utts)")
    print(f"shared: {n}   exact: {same}   differing: {len(diffs)}"
          f"   ({(same / n * 100.0) if n else 0.0:.2f}% identical)")
    if only_ref:
        print(f"  MISSING from hyp: {len(only_ref)}  e.g. {only_ref[:3]}")
    if only_hyp:
        print(f"  EXTRA in hyp    : {len(only_hyp)}  e.g. {only_hyp[:3]}")

    for uid in diffs[: args.show]:
        print(f"\n--- {uid}")
        print(f"  ref: {ref[uid]}")
        print(f"  hyp: {hyp[uid]}")

    if len(diffs) > args.show:
        print(f"\n... and {len(diffs) - args.show} more differing utterances")

    return 0 if not diffs and not only_ref and not only_hyp else 1


if __name__ == "__main__":
    sys.exit(main())
