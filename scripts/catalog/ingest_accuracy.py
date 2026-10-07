#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Ingest profile-stamped WER scores into accuracy_benchmarks.

Only exact cells selected by the publication profile are eligible. The Modal
publication sweep writes the hypotheses; score them locally first, then run:

    for f in reports/wer/*.jsonl; do uv run scripts/wer/score.py "$f"; done
    uv run scripts/catalog/ingest_accuracy.py --dry-run
    uv run scripts/catalog/ingest_accuracy.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import profiles  # noqa: E402

REPORTS = common.REPO / "reports" / "wer"


def score_path(record: dict, cell: dict, reports: pathlib.Path) -> pathlib.Path:
    filename = next(item["filename"] for item in record["downloads"]
                    if item["quant"] == cell["quant"])
    model = pathlib.Path(filename).stem
    dataset = common.dataset_slug(cell)
    batch = "" if cell["batch_size"] <= 1 else f".b{cell['batch_size']}"
    timestamps = "" if cell["timestamps"] == "none" else f".ts-{cell['timestamps']}"
    return reports / f"{model}.{dataset}{batch}{timestamps}.score.json"


def row_from_score(cell: dict, score: dict, profile_id: str) -> dict:
    per_utterance = score.get("per_utterance") or []
    metric = cell["metric"]
    return {
        "dataset": cell["dataset"],
        "split": cell["split"],
        "language": cell["language"],
        "language_hint": cell["runtime_language"],
        "backend": cell["backend"],
        "quant": cell["quant"],
        "metric": metric,
        "err_pct": score["error_rate_pct"],
        "ci95": [round(score["error_rate_ci_lo"] * 100, 2),
                 round(score["error_rate_ci_hi"] * 100, 2)],
        "n_utts": score["n"],
        "batch_size": score.get("batch_size") if score.get("batch_size") is not None else cell["batch_size"],
        "timestamps": cell["timestamps"],
        "engine_sha": score["engine_sha"],
        "publication_profile": profile_id,
        "measured_on": None,
        "errors": {
            "sub": score["substitutions"],
            "del": score["deletions"],
            "ins": score["insertions"],
        },
        "empty_hyp": sum(1 for row in per_utterance
                         if not str(row.get("hyp") or "").strip()),
        "utts_over_50pct": sum(1 for row in per_utterance
                               if float(row.get(metric, 0.0)) > 0.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=str(REPORTS))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--models", default="",
                        help="comma-separated variants (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile_id, profile = profiles.load_profile(args.profile)
    reports = pathlib.Path(args.reports)
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    records = common.load_records()
    unknown = selected - records.keys()
    if unknown:
        print(f"error: no catalog record for {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    added = replaced = rejected = missing = unstamped = 0
    for variant, record in records.items():
        if selected and variant not in selected:
            continue
        path = common.CATALOG_DIR / f"{variant}.json"
        rows = record.get("accuracy_benchmarks", [])
        changed = False
        expected = profiles.apply_exceptions(
            record, "accuracy", profiles.expected_accuracy(record, profile))
        for cell in expected:
            source_path = score_path(record, cell, reports)
            if not source_path.exists():
                missing += 1
                continue
            score = json.loads(source_path.read_text())
            recipe = score.get("recipe") or {}
            covered = any(profiles.profile_key(row) == profiles.profile_key(cell)
                          or (row.get("measurement_provenance") == "legacy-published"
                              and profiles.accuracy_core_key(row) == profiles.accuracy_core_key(cell))
                          for row in rows)
            if not recipe.get("publication_profile") and covered:
                # A score from before profile stamping, for a cell the catalog
                # already publishes: superseded history, not a problem.
                unstamped += 1
                continue
            reasons = []
            if recipe.get("publication_profile") != profile_id:
                reasons.append(f"profile={recipe.get('publication_profile')!r}")
            if score.get("metric") != cell["metric"]:
                reasons.append(f"metric={score.get('metric')!r}")
            if score.get("timestamps") != cell["timestamps"]:
                reasons.append(f"timestamps={score.get('timestamps')!r}")
            if recipe.get("backend") != cell["backend"]:
                reasons.append(f"backend={recipe.get('backend')!r}")
            if not score.get("engine_sha"):
                reasons.append("engine_sha is empty")
            if reasons:
                rejected += 1
                print(f"  reject {source_path.name}: {', '.join(reasons)}")
                continue

            # Any batch size satisfies the cell; the newest measurement
            # replaces whatever the cell held and records its own batch size.
            key = profiles.profile_key(cell)
            indices = [index for index, row in enumerate(rows)
                       if profiles.profile_key(row) == key]
            if not indices:
                # A fresh exact run supersedes the matching historical table
                # row even if that row used an older/unknown recipe.
                core = profiles.accuracy_core_key(cell)
                indices = [index for index, row in enumerate(rows)
                           if row.get("measurement_provenance") == "legacy-published"
                           and profiles.accuracy_core_key(row) == core]
            new_row = row_from_score(cell, score, profile_id)
            if indices:
                first = indices[0]
                if rows[first] == new_row and len(indices) == 1:
                    continue
                rows[first] = new_row
                for index in reversed(indices[1:]):
                    del rows[index]
                replaced += 1
            else:
                rows.append(new_row)
                added += 1
            changed = True
        if changed and not args.dry_run:
            common.write_record(path, record)

    print(f"profile {profile_id}: {added} added, {replaced} replaced, "
          f"{rejected} rejected, {unstamped} unstamped score(s) for already published "
          f"cells skipped, {missing} score file(s) absent")
    if args.dry_run:
        print("dry run: nothing written")
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
