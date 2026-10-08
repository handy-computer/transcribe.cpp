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

Language ID records follow their own profile (catalog/_benchmark_profiles.json
`roles`). Their scores come from scripts/langid/score.py --ref --json under
reports/langid/, named after the GGUF like a WER score
(`<gguf stem>.fleurs-mul.score.json`, with its `.agreement.json`); the profile
names the crop, and the importer checks the sweep's own recipe against it
instead of a stamp.
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
LANGID_REPORTS = common.REPO / "reports" / "langid"
LANGID_SCORE = "transcribe-langid-score-v1"


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


def langid_row(record: dict, cell: dict, score: dict, agreement: dict | None,
               profile_id: str) -> tuple[dict | None, list[str]]:
    """The catalog row for one language ID cell, or why the score cannot be
    it. The sweep's header travels in the score, so the recipe is checked
    here rather than trusted from a stamp."""
    filename = next(item["filename"] for item in record["downloads"]
                    if item["quant"] == cell["quant"])
    crop = (score.get("crops") or {}).get(cell["crop_s"])
    reasons = []
    if score.get("schema") != LANGID_SCORE:
        reasons.append(f"schema={score.get('schema')!r}")
    if score.get("engine") != "cpp":
        reasons.append(f"engine={score.get('engine')!r}")
    if pathlib.PurePath(str(score.get("model", ""))).name != filename:
        reasons.append(f"model={score.get('model')!r}")
    for field in ("dataset", "split", "language", "backend"):
        if score.get(field) != cell.get(field):
            reasons.append(f"{field}={score.get(field)!r}")
    if crop is None:
        reasons.append(f"no {cell['crop_s']} s crop")
    if not score.get("engine_sha"):
        reasons.append("engine_sha is empty")
    if sorted(score.get("languages") or []) != sorted(cell.get("pooled_languages") or []):
        reasons.append(f"languages={score.get('languages')!r} are not the profile's pooled_languages")
    if agreement is None:
        reasons.append("no agreement (write it with score.py --ref --json)")
    elif agreement.get("run") != score.get("run"):
        reasons.append(f"agreement is for {agreement.get('run')!r}, not {score.get('run')!r}")
    elif not agreement.get("same_rows", False):
        reasons.append("agreement run does not cover the reference's rows")
    if reasons:
        return None, reasons
    row = {
        "dataset": cell["dataset"],
        "split": cell["split"],
        "language": cell["language"],
        "backend": cell["backend"],
        "quant": cell["quant"],
        "metric": cell["metric"],
        "acc_pct": crop["acc_pct"],
        "ci95": crop["ci95"],
        "n_utts": crop["n"],
        "batch_size": cell["batch_size"],
        "timestamps": cell["timestamps"],
        "engine_sha": score["engine_sha"],
        "publication_profile": profile_id,
        "measured_on": (score.get("created") or "")[:10] or None,
    }
    row["agreement"] = {
        "n_agree": agreement["n_agree"],
        "n": agreement["n"],
        "max_abs_logit_delta": float(f"{agreement['max_abs_logit_delta']:.2g}"),
    }
    return row, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=None,
                        help="score directory (default: reports/wer, and "
                             "reports/langid for language ID records)")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--models", default="",
                        help="comma-separated variants (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    records = common.load_records()
    unknown = selected - records.keys()
    if unknown:
        print(f"error: no catalog record for {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    added = replaced = rejected = missing = unstamped = 0
    used: set[str] = set()
    for variant, record in records.items():
        if selected and variant not in selected:
            continue
        profile_id, profile = profiles.profile_for(record, args.profile)
        langid = profile.get("role") == "langid"
        reports = pathlib.Path(args.reports) if args.reports else (
            LANGID_REPORTS if langid else REPORTS)
        path = common.CATALOG_DIR / f"{variant}.json"
        rows = record.get("accuracy_benchmarks", [])
        changed = False
        expected = profiles.apply_exceptions(
            record, "accuracy", profiles.expected_accuracy(record, profile))
        if expected:
            used.add(profile_id)
        for cell in expected:
            source_path = score_path(record, cell, reports)
            if not source_path.exists():
                missing += 1
                continue
            score = json.loads(source_path.read_text())
            if langid:
                agreement_path = source_path.with_name(
                    source_path.name.replace(".score.json", ".agreement.json"))
                agreement = (json.loads(agreement_path.read_text())
                             if agreement_path.exists() else None)
                new_row, reasons = langid_row(record, cell, score, agreement, profile_id)
            else:
                new_row, reasons = None, []
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
            new_row = new_row or row_from_score(cell, score, profile_id)
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

    print(f"profile {', '.join(sorted(used)) or '-'}: {added} added, {replaced} replaced, "
          f"{rejected} rejected, {unstamped} unstamped score(s) for already published "
          f"cells skipped, {missing} score file(s) absent")
    if args.dry_run:
        print("dry run: nothing written")
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
