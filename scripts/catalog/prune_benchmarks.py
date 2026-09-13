#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Remove benchmark rows outside the selected publication profile.

The catalog is intentionally not a run archive. This migration keeps one row
per required cell, updates a stale headline pointer to the equivalent profile
recipe, and removes everything else. It never invents a missing measurement.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import profiles  # noqa: E402


def choose_headline(record: dict, expected: list[dict]) -> dict | None:
    if not expected:
        return None
    current = record.get("headline_benchmark") or {}
    if current and common.headline_rows(record):
        return current
    # Preserve the editorial dataset/language choice while moving it onto the
    # profile recipe. If it is no longer eligible, prefer LibriSpeech, then the
    # first profile cell (normally the model's first FLEURS language).
    candidates = [cell for cell in expected
                  if all(cell.get(key) == current.get(key)
                         for key in ("dataset", "split", "language", "metric"))]
    if not candidates:
        candidates = [cell for cell in expected
                      if cell["dataset"] == "librispeech"]
    cell = (candidates or expected)[0]
    return {key: cell[key] for key in common.HEADLINE_KEYS}


def dedupe_profile_rows(rows: list[dict], expected_keys: set[tuple], kind: str
                        ) -> tuple[list[dict], int, int]:
    kept: list[dict] = []
    by_key: dict[tuple, int] = {}
    extra = duplicate = 0
    for row in rows:
        key = profiles.cell_key(row, kind)
        if key not in expected_keys:
            extra += 1
            continue
        if key not in by_key:
            by_key[key] = len(kept)
            kept.append(row)
            continue
        duplicate += 1
        previous = kept[by_key[key]]
        # Prefer a reproducible measurement over an unattributed legacy row.
        previous_valid = bool(previous.get("engine_sha")) and (
            kind != "speed" or previous.get("total_ms") is not None)
        row_valid = bool(row.get("engine_sha")) and (
            kind != "speed" or row.get("total_ms") is not None)
        if row_valid and not previous_valid:
            kept[by_key[key]] = row
    return kept, extra, duplicate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--models", default="")
    parser.add_argument("--write", action="store_true",
                        help="apply the migration (default: report only)")
    args = parser.parse_args()

    profile_id, profile = profiles.load_profile(args.profile)
    records = common.load_records()
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    unknown = selected - records.keys()
    if unknown:
        print(f"unknown catalog variant(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    total_extra = total_duplicate = headlines = changed_records = 0
    for variant, record in records.items():
        if selected and variant not in selected:
            continue
        changed = False
        expected_accuracy = profiles.apply_exceptions(
            record, "accuracy", profiles.expected_accuracy(record, profile))
        expected_cores = {profiles.accuracy_core_key(cell)
                          for cell in expected_accuracy}
        # Retain honest pre-profile rows for a required dataset/language/quant
        # even when the surviving publication did not record the standardized
        # batch/timestamp recipe.
        expected_accuracy += [
            row for row in record.get("accuracy_benchmarks", [])
            if row.get("measurement_provenance") == "legacy-published"
            and profiles.accuracy_core_key(row) in expected_cores
        ]
        expected_speed = profiles.apply_exceptions(
            record, "speed", profiles.expected_speed(record, profile))
        for kind, section, expected in (
            ("accuracy", "accuracy_benchmarks", expected_accuracy),
            ("speed", "speed_benchmarks", expected_speed),
        ):
            keys = {profiles.cell_key(cell, kind) for cell in expected}
            kept, extra, duplicate = dedupe_profile_rows(
                record.get(section, []), keys, kind)
            if extra or duplicate:
                record[section] = kept
                total_extra += extra
                total_duplicate += duplicate
                changed = True
        headline = choose_headline(record, expected_accuracy)
        if record.get("headline_benchmark") != headline:
            record["headline_benchmark"] = headline
            headlines += 1
            changed = True
        if changed:
            changed_records += 1
            print(f"  {variant}: profile rows={len(record['accuracy_benchmarks'])} accuracy, "
                  f"{len(record['speed_benchmarks'])} speed")
            if args.write:
                common.write_record(common.CATALOG_DIR / f"{variant}.json", record)

    print(f"profile {profile_id}: {total_extra} extra and {total_duplicate} duplicate "
          f"row(s) removed across {changed_records} record(s); "
          f"{headlines} headline pointer(s) updated")
    if not args.write:
        print("dry run: pass --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
