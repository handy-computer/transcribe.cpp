#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Mark migrated published measurements whose original run SHA is unavailable.

This does not invent an engine commit or timing breakdown. It records the
narrow fact we do know: the number survived in a model card/publication before
profile-stamped reports existed. New measurements must carry engine_sha and are
never eligible for this marker.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    records = common.load_records()
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    unknown = selected - records.keys()
    if unknown:
        print(f"unknown catalog variant(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    rows_changed = records_changed = 0
    by_section = {"accuracy_benchmarks": 0, "speed_benchmarks": 0}
    for variant, record in records.items():
        if selected and variant not in selected:
            continue
        changed = False
        for section in by_section:
            for row in record.get(section, []):
                if row.get("engine_sha") or row.get("measurement_provenance"):
                    continue
                # Both row types have a user-facing published number. Speed's
                # durable legacy number may be xRT-only (total_ms is null).
                if section == "speed_benchmarks" and row.get("xrt_compute") is None:
                    continue
                row["measurement_provenance"] = "legacy-published"
                rows_changed += 1
                by_section[section] += 1
                changed = True
        if changed:
            records_changed += 1
            if args.write:
                common.write_record(common.CATALOG_DIR / f"{variant}.json", record)

    print(f"legacy provenance: {rows_changed} row(s) across {records_changed} record(s) "
          f"({by_section['accuracy_benchmarks']} accuracy, "
          f"{by_section['speed_benchmarks']} speed)")
    if not args.write:
        print("dry run: pass --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
