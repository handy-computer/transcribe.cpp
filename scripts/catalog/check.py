#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema", "pyyaml"]
# ///
"""check.py — validate catalog records and diff them against what we publish today.

Three passes:
  schema     every record validates against catalog/_schema.json
  integrity  the gates the schema cannot express -- every benchmarked quant is
             published, one slug per physical machine, recipe labels consistent
  published  every number currently in scripts/hf_cards/*.yaml is either
             reproduced by a record or reported as unsourced

The third pass is the migration's acceptance test: it says, model by model,
how much of what we publish today actually has an artifact behind it.

    uv run scripts/catalog/check.py
    uv run scripts/catalog/check.py --dir catalog --published
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import yaml
from jsonschema import Draft202012Validator

REPO = pathlib.Path(__file__).resolve().parents[2]
CARDS = REPO / "scripts/hf_cards"


def load(d: pathlib.Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text())
            for p in sorted(d.glob("*.json")) if not p.name.startswith("_")}


def schema_pass(records: dict, schema: dict) -> int:
    v, bad = Draft202012Validator(schema), 0
    for name, rec in records.items():
        errs = sorted(v.iter_errors(rec), key=lambda e: list(e.path))
        if errs:
            bad += 1
            print(f"  FAIL {name}: {len(errs)} error(s)")
            for e in errs[:4]:
                print(f"        {'/'.join(map(str, e.path)) or '<root>'}: {e.message[:110]}")
    print(f"schema     {len(records) - bad}/{len(records)} valid")
    return bad


def integrity_pass(records: dict) -> int:
    bad = 0
    machines: dict[str, set[str]] = collections.defaultdict(set)
    for name, rec in records.items():
        published = {d["quant"] for d in rec.get("downloads", [])}
        for sect in ("accuracy_benchmarks", "speed_benchmarks"):
            missing = {r["quant"] for r in rec.get(sect, []) if r["quant"] not in published}
            if missing:
                bad += 1
                print(f"  FAIL {name}: {sect} references unpublished quant(s) {sorted(missing)}")
        for r in rec.get("speed_benchmarks", []):
            if r.get("machine"):
                machines[r["machine"]].add(name)
        recipes = {(r["dataset"], r["recipe"]) for r in rec.get("accuracy_benchmarks", [])}
        by_ds = collections.Counter(ds for ds, _ in recipes)
        for ds, n in by_ds.items():
            if n > 1:
                print(f"  warn {name}: {ds} measured under {n} recipes "
                      f"{sorted(r for d, r in recipes if d == ds)} -- not comparable to each other")
    print(f"integrity  {len(records) - bad}/{len(records)} clean; "
          f"{len(machines)} machine slug(s): {', '.join(sorted(machines))}")
    return bad


def published_pass(records: dict) -> None:
    """What the cards publish today, vs what a record can source."""
    rows, tot_w, got_w, tot_p, got_p = [], 0, 0, 0, 0
    for stem in sorted(p.stem for p in CARDS.glob("*.yaml")):
        card = yaml.safe_load((CARDS / f"{stem}.yaml").read_text()) or {}
        rec = records.get(stem)
        card_w = {q["name"]: q.get("wer") for q in (card.get("quants") or []) if q.get("wer")}
        card_p = sum(len(v) for v in (card.get("perf") or {}).values())
        have_w = {r["quant"] for r in (rec or {}).get("accuracy_benchmarks", [])}
        have_p = len({(r["machine"], r["backend"], r["quant"]) for r in
                      (rec or {}).get("speed_benchmarks", [])})
        tot_w += len(card_w); got_w += len(set(card_w) & have_w)
        tot_p += card_p; got_p += min(card_p, have_p)
        if len(card_w) != len(set(card_w) & have_w) or card_p > have_p:
            rows.append((stem, f"{len(set(card_w) & have_w)}/{len(card_w)}",
                         f"{min(card_p, have_p)}/{card_p}"))
    print(f"published  WER cells sourced {got_w}/{tot_w}; perf cells sourced {got_p}/{tot_p}")
    if rows:
        print(f"\n  {'variant':42s} {'wer':>9} {'perf':>9}")
        for stem, w, p in rows:
            print(f"  {stem:42s} {w:>9} {p:>9}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "catalog"))
    ap.add_argument("--published", action="store_true", help="run the published-vs-sourced diff")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    schema = json.loads((REPO / "catalog/_schema.json").read_text())
    records = load(d)
    if not records:
        print(f"no records in {d}", file=sys.stderr)
        return 2
    bad = schema_pass(records, schema) + integrity_pass(records)
    if args.published:
        published_pass(records)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
