#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema"]
# ///
"""Validate the durable catalog JSON records.

Checks the JSON schema, cross-row integrity the schema cannot express (a
benchmark row referencing a quant the variant does not publish), that every
record is paired with the card spec and doc the schema says it owns.

    uv run scripts/catalog/check.py
    uv run scripts/catalog/check.py --publication-profile
    uv run scripts/catalog/check.py --publication-profile --models whisper-tiny
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
sys.path.insert(0, str(REPO / "scripts" / "catalog"))
import common  # noqa: E402
import profiles  # noqa: E402


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
        accuracy_counts = collections.Counter(
            profiles.cell_key(row, "accuracy")
            for row in rec.get("accuracy_benchmarks", []))
        duplicate_accuracy = sum(count - 1 for count in accuracy_counts.values()
                                 if count > 1)
        if duplicate_accuracy:
            bad += 1
            print(f"  FAIL {name}: {duplicate_accuracy} duplicate published "
                  "accuracy cell(s); batch size is recipe metadata, not a "
                  "separate result")
        # A shipped model always says how fast it runs somewhere. The
        # publication profile decides which cells are required; this is the
        # weaker floor underneath it, so a record can never render a page or
        # a card with no performance at all.
        if not rec.get("speed_benchmarks"):
            bad += 1
            print(f"  FAIL {name}: no speed_benchmarks; every shipped model "
                  f"carries at least one measured cell")
        for r in rec.get("speed_benchmarks", []):
            if r.get("machine"):
                machines[r["machine"]].add(name)
    print(f"integrity  {len(records) - bad}/{len(records)} clean; "
          f"{len(machines)} machine slug(s): {', '.join(sorted(machines))}")
    return bad


def pairing_pass(records: dict, selected: bool = False) -> int:
    """Catalog records and card specs pair exactly; docs may be shared.

    Every record names its docs page (its own, or the family page whose
    roll-up lists it), and that file must exist. The editorial card specs under scripts/hf_cards/
    pair one to one with records: generate.py reads both, so an orphan spec
    has no catalog to render from and a record with no spec has no card.
    """
    card_names = {path.stem for path in (REPO / "scripts" / "hf_cards").glob("*.yaml")}
    record_names = set(records)
    missing_cards = sorted(record_names - card_names)
    # Orphan specs are a whole-catalog question; a --models run only asks
    # whether the selected records have their card.
    missing_records = [] if selected else sorted(card_names - record_names)
    if selected:
        card_names &= record_names
    undocumented = sorted(name for name in record_names if not records[name].get("docs_page"))
    bad_pages = sorted(
        f"{name}: docs/models/{records[name]['docs_page']} does not exist"
        for name in record_names
        if records[name].get("docs_page")
        and not (REPO / "docs" / "models" / records[name]["docs_page"]).exists())
    for line in bad_pages:
        print(f"  FAIL {line}")
    for name in missing_cards:
        print(f"  FAIL {name}: no scripts/hf_cards/{name}.yaml")
    for name in missing_records:
        print(f"  FAIL scripts/hf_cards/{name}.yaml: no catalog/{name}.json")
    paired = len(record_names & card_names)
    print(f"pairing    {paired}/{len(record_names | card_names)} catalog/card pairs; "
          f"{len(records) - len(undocumented)}/{len(records)} name a docs page")
    if undocumented:
        print(f"           no docs_page: {', '.join(undocumented)}")
    return len(missing_cards) + len(missing_records) + len(bad_pages)


def publication_pass(records: dict, profile_id: str | None, enforce: bool) -> int:
    """Check publication matrices, including explicit legacy accuracy rows."""
    try:
        resolved_id, profile = profiles.load_profile(profile_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"publication FAIL: {exc}")
        return 1

    problems = 0
    totals = collections.Counter()
    for name, record in records.items():
        # An ASR publication profile does not apply to standalone diarizers.
        if not record.get("capabilities", {}).get("transcribe", {}).get("supported"):
            continue
        accuracy_raw = profiles.expected_accuracy(record, profile)
        speed_raw = profiles.expected_speed(record, profile)
        accuracy = profiles.apply_exceptions(record, "accuracy", accuracy_raw)
        speed = profiles.apply_exceptions(record, "speed", speed_raw)
        exceptions = record.get("benchmark_exceptions") or []

        # An exception must exclude a real profile requirement. This catches
        # stale or misspelled waivers instead of retaining them forever.
        stale_exceptions = 0
        for exception in exceptions:
            kind = exception.get("kind")
            candidates = accuracy_raw if kind == "accuracy" else speed_raw
            if kind not in ("accuracy", "speed") or not any(
                    profiles.exception_matches(exception, kind, cell)
                    for cell in candidates):
                stale_exceptions += 1
        per_model = collections.Counter(stale_exception=stale_exceptions)

        # Accuracy is closed by dataset/language/quant/metric/timestamps. Any
        # batch size satisfies a cell (the row records which); explicitly
        # marked legacy results retain the published recipe (or null when it
        # did not survive).
        expected_by_key = {profiles.profile_key(cell): cell for cell in accuracy}
        expected_keys = set(expected_by_key)
        expected_by_core = {
            profiles.accuracy_core_key(cell): profiles.profile_key(cell)
            for cell in accuracy
        }
        accuracy_rows = record.get("accuracy_benchmarks", [])
        accuracy_counts = collections.Counter(
            profiles.cell_key(row, "accuracy") for row in accuracy_rows)
        accuracy_covered, accuracy_extra, accuracy_invalid = set(), set(), set()
        for row in accuracy_rows:
            key = profiles.profile_key(row)
            legacy = row.get("measurement_provenance") == "legacy-published"
            if key in expected_keys:
                target_key = key
            elif legacy:
                # Grandfather a published pre-profile result under its honest
                # batch/timestamp recipe; do not relabel it as the new recipe.
                target_key = expected_by_core.get(profiles.accuracy_core_key(row))
            else:
                target_key = None
            if target_key is None:
                if legacy:
                    # Preserve pre-profile rows that were already published,
                    # even when their quant was not selected by today's
                    # publication matrix. They are archive data, not drift.
                    totals["accuracy_archived"] += 1
                else:
                    accuracy_extra.add(key)
                continue
            accuracy_covered.add(target_key)
            target = expected_by_key[target_key]
            if (not profiles.has_measurement_provenance(row)
                    or (not legacy and (
                        row.get("backend") != target.get("backend")
                        or row.get("language_hint") != target.get("runtime_language")))):
                accuracy_invalid.add(target_key)
        per_model["accuracy_missing"] = len(expected_keys - accuracy_covered)
        per_model["accuracy_invalid"] = len(accuracy_invalid)
        per_model["accuracy_extra"] = len(accuracy_extra)
        per_model["accuracy_duplicate"] = sum(
            count - 1 for count in accuracy_counts.values() if count > 1)
        totals["accuracy_required"] += len(expected_keys)
        for suffix in ("missing", "invalid", "extra", "duplicate"):
            totals[f"accuracy_{suffix}"] += per_model[f"accuracy_{suffix}"]

        # Both published samples are required for each quant selected by the
        # profile on every machine/backend target.
        speed_expected = {profiles.cell_key(cell, "speed") for cell in speed}
        speed_rows = record.get("speed_benchmarks", [])
        speed_counts = collections.Counter(
            profiles.cell_key(row, "speed") for row in speed_rows)
        speed_actual = set(speed_counts)
        invalid_speed = {
            profiles.cell_key(row, "speed") for row in speed_rows
            if profiles.cell_key(row, "speed") in speed_expected
            and (row.get("xrt_compute") is None
                 or not profiles.has_measurement_provenance(row))
        }
        per_model["speed_missing"] = len(speed_expected - speed_actual)
        per_model["speed_invalid"] = len(invalid_speed)
        per_model["speed_extra"] = len(speed_actual - speed_expected)
        per_model["speed_duplicate"] = sum(
            count - 1 for count in speed_counts.values() if count > 1)
        totals["speed_required"] += len(speed_expected)
        for suffix in ("missing", "invalid", "extra", "duplicate"):
            totals[f"speed_{suffix}"] += per_model[f"speed_{suffix}"]

        count = sum(per_model.values())
        if count:
            problems += count
            details = ", ".join(f"{key}={value}" for key, value in per_model.items() if value)
            print(f"  {'FAIL' if enforce else 'TODO'} {name}: {details}")

    print(f"publication {resolved_id}: accuracy {totals['accuracy_required']} required, "
          f"{totals['accuracy_missing']} missing, {totals['accuracy_invalid']} invalid, "
          f"{totals['accuracy_extra']} extra, {totals['accuracy_archived']} archived legacy; speed "
          f"{totals['speed_required']} required, {totals['speed_missing']} missing, "
          f"{totals['speed_invalid']} invalid, {totals['speed_extra']} extra")
    if problems and not enforce:
        print("            audit only; pass --publication-profile to enforce this gate")
    return problems if enforce else 0


def provenance_pass(records: dict) -> int:
    """Every number names its run or explicitly declares its legacy origin."""
    bad, legacy_speed, legacy_acc = 0, 0, 0
    total_speed = total_acc = 0
    for name, record in records.items():
        for section, kind in (("accuracy_benchmarks", "accuracy"),
                              ("speed_benchmarks", "speed")):
            for row in record.get(section, []):
                if kind == "accuracy":
                    total_acc += 1
                else:
                    total_speed += 1
                if row.get("measurement_provenance") == "legacy-published":
                    if kind == "accuracy":
                        legacy_acc += 1
                    else:
                        legacy_speed += 1
                    continue
                if not row.get("engine_sha"):
                    bad += 1
                    print(f"  FAIL {name}: {kind} row has neither engine_sha nor "
                          f"measurement_provenance=legacy-published")
    print(f"provenance {total_speed - legacy_speed}/{total_speed} speed and "
          f"{total_acc - legacy_acc}/{total_acc} accuracy row(s) name a build; "
          f"{legacy_speed + legacy_acc} explicitly marked legacy-published")
    if legacy_speed:
        print(f"           {legacy_speed} legacy speed row(s) retain published xRT only; "
              f"re-benchmark for stage timings, long-form, and memory")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "catalog"))
    ap.add_argument("--publication-profile", nargs="?", const="",
                    help="enforce exact accuracy and speed matrices; optionally "
                         "name a profile (default: "
                         "catalog/_benchmark_profiles.json default)")
    ap.add_argument("--models", default="",
                    help="comma-separated variants (default: all)")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    schema = json.loads((REPO / "catalog/_schema.json").read_text())
    records = common.load_records(d)
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    unknown = selected - records.keys()
    if unknown:
        print(f"unknown catalog variant(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if selected:
        records = {name: record for name, record in records.items() if name in selected}
    if not records:
        print(f"no records in {d}", file=sys.stderr)
        return 2
    enforce_publication = args.publication_profile is not None
    selected_profile = args.publication_profile or None
    bad = (schema_pass(records, schema) + integrity_pass(records)
           + pairing_pass(records, bool(selected)) + provenance_pass(records)
           + publication_pass(records, selected_profile, enforce_publication))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
