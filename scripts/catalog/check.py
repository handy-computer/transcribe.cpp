#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema"]
# ///
"""Validate the durable catalog JSON records.

Checks the JSON schema plus cross-row integrity that the schema cannot express,
such as benchmark rows referencing a quant the variant does not publish.

    uv run scripts/catalog/check.py
    uv run scripts/catalog/check.py --dir catalog
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

from jsonschema import Draft202012Validator

REPO = pathlib.Path(__file__).resolve().parents[2]


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
    print(f"integrity  {len(records) - bad}/{len(records)} clean; "
          f"{len(machines)} machine slug(s): {', '.join(sorted(machines))}")
    return bad



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "catalog"))
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    schema = json.loads((REPO / "catalog/_schema.json").read_text())
    records = load(d)
    if not records:
        print(f"no records in {d}", file=sys.stderr)
        return 2
    bad = schema_pass(records, schema) + integrity_pass(records)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
