#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Fold bench driver reports into the catalog's speed_benchmarks rows.

`scripts/bench/run.py` writes one report per (variant, backend) under
reports/perf/<machine-slug>/, and reports/ is gitignored -- so the latency
breakdown only exists on the machine that measured it. This is the hop that
moves it into the catalog, where it is durable and publishable.

A cell is identified by (machine, backend, quant, sample). Many reports cover
the same cell, because porting-6-bench runs a hypothesis loop over it, and
those iterations are NOT interchangeable with the published figure -- CPU
cells in particular swing tens of percent with thermal state. Only a
profile-stamped run (`scripts/bench/run.py --profile`) is eligible, and among
those the newest wins.

    uv run scripts/catalog/ingest_perf.py --dry-run
    uv run scripts/catalog/ingest_perf.py
    uv run scripts/catalog/ingest_perf.py --reports reports/perf
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import profiles  # noqa: E402

REPORTS = common.REPO / "reports" / "perf"

# models/<dir>/<stem>-<QUANT>.gguf -- the quant is the last dash-separated
# field, and K-quants carry underscores (Q4_K_M) so the split is on "-".
QUANT_RE = re.compile(r"-([A-Za-z0-9_]+)\.gguf$")


def quant_of(model_path: str) -> str | None:
    match = QUANT_RE.search(model_path)
    return match.group(1) if match else None


@functools.lru_cache(maxsize=1)
def variant_by_directory() -> dict[str, str]:
    """Catalog variant by the lowercased models/ directory that holds it.

    A directory mirrors the upstream repo name (`Qwen3-ASR-0.6B`,
    `SenseVoiceSmall`) while the record is kebab-case, so a report's path
    names its variant only up to case, and sometimes not even that. An
    upstream repo that several records share (the four gigaam variants ship
    from one repo) is ambiguous, so it resolves to nothing rather than to an
    arbitrary one of them; those variants have directories of their own.
    """
    records = common.load_records()
    lookup = {variant.lower(): variant for variant in records}
    by_upstream = collections.defaultdict(list)
    for variant, record in records.items():
        by_upstream[record["upstream_repo"].rsplit("/", 1)[-1].lower()].append(variant)
    for slug, variants in by_upstream.items():
        if len(variants) == 1:
            lookup.setdefault(slug, variants[0])
    return lookup


def variant_of(report: dict, model_path: str) -> str | None:
    """Reports carry `variant` or the older `family`; the path is definitive."""
    name = report.get("variant")
    parts = pathlib.PurePosixPath(model_path.replace("\\", "/")).parts
    if "models" in parts:
        index = len(parts) - 1 - parts[::-1].index("models")
        if index + 1 < len(parts):
            name = parts[index + 1]
    if not name:
        return None
    return variant_by_directory().get(name.lower(), name)


def publishable(report: dict) -> bool:
    """Only a profile-stamped run can become a catalog row: the stamp names
    the recipe (iterations, warmup, samples, thermal policy) the number was
    measured under. Hypothesis-loop and hand-flagged runs are experiments."""
    return bool(report.get("publication_profile"))


def cells(report: dict) -> list[dict]:
    """One catalog-shaped row per run in a bench driver report."""
    out = []
    for run in report.get("runs", []):
        model_path, quant = run.get("model_path", ""), quant_of(run.get("model_path", ""))
        variant = variant_of(report, model_path)
        summary, duration = run.get("summary") or {}, run.get("sample_duration_s")
        total = (summary.get("total_ms") or {}).get("mean")
        wall = (summary.get("wall_ms") or {}).get("mean")
        if not (variant and quant and duration and total):
            continue

        def mean(field: str):
            value = (summary.get(field) or {}).get("mean")
            return None if value is None else round(value, 1)

        out.append({
            "_profile": report.get("publication_profile"),
            "variant": variant,
            "machine": profiles.canonical_machine(report["machine"]["slug"]),
            # The run's own `backend` is the runtime device name (MTL0), not
            # the canonical backend; the driver records that at the top level.
            "backend": (report.get("backend") or run.get("backend", "")).lower(),
            "quant": quant,
            "sample": pathlib.PurePosixPath(run.get("sample_path", "")).stem,
            "sample_duration_s": duration,
            "total_ms": round(total, 1),
            # xrt is recomputed from the unrounded mean rather than carried
            # over: a stored value that no longer matches its own latency is
            # the drift this ingest exists to remove.
            "xrt_compute": round(duration / (total / 1000), 2),
            "wall_ms": None if wall is None else round(wall, 1),
            "xrt_wall": None if wall is None else round(duration / (wall / 1000), 2),
            "load_ms": None if run.get("load_ms") is None else round(run["load_ms"], 1),
            "mel_ms": mean("mel_ms"),
            "encode_ms": mean("encode_ms"),
            "decode_ms": mean("decode_ms"),
            "engine_sha": report.get("git_sha"),
            "publication_profile": report.get("publication_profile"),
            "measured_on": (report.get("timestamp") or "")[:10] or None,
            "_when": report.get("timestamp") or "",
            "_file": report["_file"],
        })
    return out


def collect(reports_dir: pathlib.Path) -> tuple[dict, list[str]]:
    """Newest profile-stamped measurement per cell, plus notes on what was
    skipped."""
    best: dict[tuple, dict] = {}
    superseded, unreadable, experiments = collections.Counter(), [], 0
    for path in sorted(reports_dir.glob("*/*.json")):
        try:
            report = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            unreadable.append(str(path.relative_to(common.REPO)))
            continue
        if not isinstance(report, dict) or report.get("schema") != "transcribe-bench-driver-v1":
            kind = report.get("schema") if isinstance(report, dict) else type(report).__name__
            unreadable.append(f"{path.relative_to(common.REPO)} (schema {kind!r})")
            continue
        if not publishable(report):
            experiments += 1
            continue
        # Reports normally live under the repo, but --reports can name a
        # staging directory elsewhere (or a relative one); the label is only
        # for the operator, so fall back to the path as given.
        try:
            report["_file"] = str(path.resolve().relative_to(common.REPO))
        except ValueError:
            report["_file"] = str(path)
        for row in cells(report):
            key = (row["variant"], row["machine"], row["backend"], row["quant"], row["sample"])
            previous = best.get(key)
            if previous is None or row["_when"] > previous["_when"]:
                if previous is not None:
                    superseded[key] += 1
                best[key] = row
            else:
                superseded[key] += 1
    notes = [f"{path}: unreadable or not a bench report" for path in unreadable]
    if experiments:
        notes.append(f"{experiments} report(s) without a profile stamp ignored")
    if superseded:
        notes.append(f"{sum(superseded.values())} older report(s) superseded on "
                     f"{len(superseded)} cell(s)")
    return best, notes


FIELDS = ("sample_duration_s", "total_ms", "xrt_compute", "wall_ms", "xrt_wall",
          "load_ms", "mel_ms", "encode_ms", "decode_ms", "engine_sha",
          "publication_profile", "measured_on")


def catalog_row(source: dict) -> dict:
    """Strip importer bookkeeping from one measured, catalog-shaped cell."""
    return {field: source[field] for field in (
        "machine", "backend", "quant", "sample", *FIELDS
    )} | {"thermal_gated": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=str(REPORTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drift", type=float, default=2.0,
                        help="percent xrt change worth reporting (default 2)")
    parser.add_argument("--max-drift", type=float, default=5.0,
                        help="refuse a row whose xRT would move more than this "
                             "percent, since the stored value is what the doc "
                             "published (default 5). --force takes it anyway.")
    parser.add_argument("--force", action="store_true",
                        help="apply measurements even when they contradict the "
                             "published xRT")
    args = parser.parse_args()

    reports_dir = pathlib.Path(args.reports)
    if not reports_dir.exists():
        print(f"no reports at {reports_dir}", file=sys.stderr)
        return 2
    measured, notes = collect(reports_dir)
    print(f"{len(measured)} measured cell(s) across "
          f"{len({key[1] for key in measured})} machine slug(s)")
    for note in notes:
        print(f"  note: {note}")

    profile_id, profile = profiles.load_profile()
    filled = updated = added = matched = 0
    drift, refused, unmatched = [], [], []
    for variant, record in common.load_records().items():
        path = common.CATALOG_DIR / f"{variant}.json"
        rows = record.get("speed_benchmarks", [])
        changed = False
        for row in rows:
            key = (variant, row["machine"], row["backend"], row["quant"], row["sample"])
            source = measured.pop(key, None)
            if source is None:
                continue
            matched += 1
            was_null = row.get("total_ms") is None
            before = row.get("xrt_compute")
            # A stored xRT may have been published as compute or as wall (the
            # granite tables quote wall, where the two differ by up to 1.46x).
            # Gate on whichever the doc evidently used, so a definition
            # mismatch is not mistaken for a stale build.
            candidates = [c for c in (source["xrt_compute"], source["xrt_wall"])
                          if c is not None]
            after = min(candidates, key=lambda c: abs(c - before)) if before else source["xrt_compute"]
            moved = abs(after - before) / before * 100 if before else 0.0
            if before and moved > args.max_drift and not args.force:
                # The stored xRT is the number the doc published. A report that
                # disagrees this much is a different build, not a better
                # reading of the same one -- refuse it and name the gap.
                refused.append((variant, row["machine"], row["backend"], row["quant"],
                                row["sample"], before, after, row.get("engine_sha"),
                                source["engine_sha"]))
                continue
            for field in FIELDS:
                if row.get(field) != source[field]:
                    row[field] = source[field]
                    changed = True
            if source.get("engine_sha") and row.pop("measurement_provenance", None):
                changed = True
            if was_null:
                filled += 1
            elif changed:
                updated += 1
            if before and moved > args.drift:
                drift.append((variant, row["machine"], row["backend"], row["quant"],
                              row["sample"], before, after))
        # Profile runs can create rows; the old importer could only refresh
        # placeholders, which made a newly required quant impossible to ingest
        # without first hand-authoring empty catalog cells.
        expected = profiles.apply_exceptions(
            record, "speed", profiles.expected_speed(record, profile))
        expected_keys = {
            (variant, cell["machine"], cell["backend"], cell["quant"], cell["sample"])
            for cell in expected
        }
        existing_keys = {
            (variant, row["machine"], row["backend"], row["quant"], row["sample"])
            for row in rows
        }
        for key in sorted(expected_keys - existing_keys):
            source = measured.get(key)
            if source is None or source.get("_profile") != profile_id:
                continue
            rows.append(catalog_row(source))
            measured.pop(key)
            matched += 1
            added += 1
            changed = True
        if changed and not args.dry_run:
            common.write_record(path, record)

    for key in measured:
        unmatched.append(key)
    print(f"\nmatched {matched} catalog row(s): {added} added, "
          f"{filled} had no latency, {updated} already did and were refreshed, "
          f"{len(refused)} refused")
    if refused:
        print(f"\n{len(refused)} row(s) refused: the newest report on file is a "
              f"different build from the one the doc published, so taking it "
              f"would silently restate a published number.")
        print("  re-run the publication sweep, or pass --force:")
        by_variant = collections.Counter()
        shas = collections.defaultdict(set)
        for variant, _, _, _, _, _, _, _, report_sha in refused:
            by_variant[variant] += 1
            shas[variant].add(report_sha or "?")
        for variant, count in by_variant.most_common(16):
            print(f"    {count:3d} cell(s)  {variant:42s} on file: "
                  f"{', '.join(sorted(shas[variant]))}")
        if len(by_variant) > 16:
            print(f"    ... and {len(by_variant) - 16} more variant(s)")
    if drift:
        print(f"\n{len(drift)} cell(s) whose xRT moved more than {args.drift}%:")
        for variant, machine, backend, quant, sample, before, after in sorted(
                drift, key=lambda d: -abs(d[6] - d[5]) / d[5]):
            pct = round(100 * (after - before) / before)
            print(f"  {variant:38s} {machine:12s} {backend:7s} {quant:7s} "
                  f"{sample:5s} {before:>8} -> {after:<8} {pct:+d}%")
    if unmatched:
        print(f"\n{len(unmatched)} measured cell(s) with no catalog row "
              f"(a bench of something the catalog does not publish):")
        counts = collections.Counter(key[0] for key in unmatched)
        for variant, count in counts.most_common(12):
            print(f"  {variant:42s} {count}")
        if len(counts) > 12:
            print(f"  ... and {len(counts) - 12} more variant(s)")
    if args.dry_run:
        print("\ndry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
