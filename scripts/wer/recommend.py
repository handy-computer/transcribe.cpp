#!/usr/bin/env python3
"""recommend.py — rank models for a given set of languages.

Answers the question the FLEURS matrix exists to answer: someone speaks
these N languages, which single model should they run?

Ranking is the geometric mean of per-language error rates, which is the
right average here because error rates are ratios: a model that halves the
error on one language and doubles it on another should come out neutral,
which an arithmetic mean does not give you.

Coverage is enforced, not interpolated. The matrix is ragged (whisper
covers 40 languages, canary-180m covers 4), so a model is only ranked if
it has a measured cell for EVERY requested language. Averaging a model
over the subset it happens to support would systematically flatter
narrow models.

Cells come from reports/wer/wer.db, the same table the markdown report
renders from, so a recommendation and the published table cannot disagree
about what a model scored. Run build_db.py first.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sqlite3
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DB = REPO / "reports" / "wer" / "wer.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", required=True,
                    help="comma-separated, e.g. en,de,ja")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--source", default="fleurs",
                    help="dataset family to rank within (default: fleurs)")
    ap.add_argument("--quant", default="Q8_0")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    db = pathlib.Path(args.db)
    if not db.exists():
        print(f"error: {db} not found; run scripts/wer/build_db.py first",
              file=sys.stderr)
        return 2
    con = sqlite3.connect(db)
    cells, models, have_langs = {}, set(), set()
    for model, lang, pct_, metric in con.execute(
        "SELECT r.model, d.lang, r.err_pct, r.metric FROM results r "
        "JOIN datasets d ON d.dataset = r.dataset "
        "WHERE d.source = ? AND r.quant = ?", (args.source, args.quant)
    ):
        cells[f"{model}|{lang}"] = {"pct": pct_, "metric": metric}
        models.add(model)
        have_langs.add(lang)

    want = [l.strip() for l in args.languages.split(",") if l.strip()]
    missing = [l for l in want if l not in have_langs]
    if missing:
        print(f"not measured on {args.source}/{args.quant}: {missing}",
              file=sys.stderr)
        print(f"available: {' '.join(sorted(have_langs))}", file=sys.stderr)
        return 2

    ranked, skipped = [], []
    for m in sorted(models):
        got = [(l, cells.get(f"{m}|{l}")) for l in want]
        if any(v is None for _, v in got):
            have = [l for l, v in got if v is not None]
            skipped.append((m, len(have)))
            continue
        rates = [v["pct"] for _, v in got]
        # Geometric mean; a 0.0 would annihilate it, so floor at 0.01%.
        gm = math.exp(sum(math.log(max(r, 0.01)) for r in rates) / len(rates))
        ranked.append((gm, m, dict(got)))
    ranked.sort()

    metrics = {cells[f"{ranked[0][1]}|{l}"]["metric"] for l in want} if ranked else set()
    print(f"languages: {' '.join(want)}")
    if len(metrics) > 1:
        print(f"NOTE: mixes {'/'.join(sorted(metrics))} across languages. Both are "
              f"error rates so the ranking holds, but the geometric mean is "
              f"not in a single unit.")
    print(f"{len(ranked)} models cover all {len(want)}; "
          f"{len(skipped)} lack at least one\n")
    w = max((len(m) for _, m, _ in ranked[:args.top]), default=10)
    print(f"{'model':{w}s} {'geomean':>8}  " + "".join(f"{l:>8}" for l in want))
    for gm, m, got in ranked[:args.top]:
        print(f"{m:{w}s} {gm:>8.2f}  " +
              "".join(f"{got[l]['pct']:>8.2f}" for l in want))
    if skipped:
        print(f"\nexcluded for coverage: " +
              ", ".join(f"{m} ({n}/{len(want)})" for m, n in sorted(skipped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
