#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Ingest native VAD reference-agreement reports, never ASR accuracy rows.

    uv run scripts/catalog/ingest_parity.py reports/vad/*.json \
        --models silero-vad-v6.2 --dry-run
    uv run scripts/catalog/ingest_parity.py --reports reports/vad \
        --models silero-vad-v6.2

Require the pinned reference artifact, actual native build provenance, default
segmentation and an exact publication-profile cell. Any differing segment
list, malformed report or failed probability gate rejects the entire import
before writes. The newest timestamp wins when several reports cover a cell.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import sys
from datetime import date, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import profiles  # noqa: E402

SCHEMA = "transcribe-vad-parity-v1"
# Corpus expansion is stdlib-only and shared with the report writer.
_spec = importlib.util.spec_from_file_location("vad_parity_corpus", common.REPO / "scripts/vad/parity.py")
vad_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vad_parity)
ROW_FIELDS = ("dataset", "split", "language", "quant", "backend", "reference",
              "n_files", "n_identical", "n_segments", "audio_duration_s",
              "segmentation_params", "engine_sha", "measured_on",
              "publication_profile", "max_abs_prob_delta")


def finite_number(value, minimum: float = 0, *, positive: bool = False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > minimum if positive else value >= minimum))


def row_from_report(record: dict, report: dict) -> dict:
    """Validate recipe, provenance and all aggregates before making a row."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(report.get("schema") == SCHEMA, "not a native VAD parity report")
    require(report.get("tool") == "scripts/vad/parity.py", "unexpected measurement tool")
    require(record.get("role") == "vad", "record is not VAD")
    require(report.get("variant") == record["variant"], "variant mismatch")
    source = record.get("source_artifact") or {}
    require(source.get("package") == "silero-vad", "unsupported reference package")
    require(report.get("source_artifact") == source, "reference source artifact mismatch")
    require(isinstance(report.get("vad_info"), dict) and report["vad_info"] == record.get("vad_info"),
            "native VAD frontend does not match the record")
    profile_id, profile = profiles.profile_for(record)
    require(profile.get("role") == "vad", "not governed by a VAD publication profile")
    require(report.get("publication_profile") == profile_id, "publication profile mismatch")
    require(report.get("reference") == f"{source['package']}=={source['version']}",
            "reference must match the canonical source artifact")
    require(report.get("backend") == "cpu", "backend must be cpu")
    require(report.get("segmentation_params") == "defaults", "segmentation parameters must be defaults")
    require(report.get("reference_threads") == 1, "reference must use one Torch thread")
    require(type(report.get("threads")) is int and report["threads"] > 0, "invalid native threads")
    filename = next((item["filename"] for item in record["downloads"]
                     if item["quant"] == report.get("quant")), None)
    require(filename is not None, "quant is not a shipped download")
    model_path = pathlib.PurePosixPath(str(report.get("model_path", "")).replace("\\", "/"))
    require(model_path.name == filename, "model filename does not match the quant's download")
    require(bool(report.get("library")), "no native library provenance")
    require(isinstance(report.get("engine_sha"), str)
            and re.fullmatch(r"[0-9a-f]{7,40}", report["engine_sha"]) is not None,
            "invalid native build SHA")
    try:
        measured_on = date.fromisoformat(report["measured_on"])
        timestamp = datetime.fromisoformat(report["timestamp"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid measurement date/timestamp") from exc
    require(timestamp.tzinfo is not None and timestamp.date() == measured_on
            and measured_on.isoformat() == report["measured_on"], "inconsistent measurement date")
    for field in ("n_files", "n_identical", "n_segments"):
        require(type(report.get(field)) is int and report[field] >= 0, f"invalid {field}")
    require(report["n_files"] > 0, "empty report")
    require(report["n_identical"] == report["n_files"], "segment lists differ; refusing nonzero diff")
    require(finite_number(report.get("audio_duration_s"), positive=True), "invalid audio duration")
    require(finite_number(report.get("max_abs_prob_delta")), "missing/invalid maximum probability delta")
    require(report["max_abs_prob_delta"] <= 1, "probability delta exceeds 1")
    files = report.get("files")
    require(isinstance(files, list) and len(files) == report["n_files"], "file count mismatch")
    labels = []
    for entry in files:
        require(isinstance(entry, dict), "invalid per-file entry")
        require(isinstance(entry.get("audio"), str) and entry["audio"], "missing file identity")
        labels.append(entry["audio"])
        require(entry.get("identical") is True, "per-file segment lists differ")
        require(type(entry.get("n_segments")) is int and entry["n_segments"] >= 0,
                "invalid per-file segment count")
        require(entry.get("n_native_segments") == entry["n_segments"], "native segment count mismatch")
        require(finite_number(entry.get("audio_duration_s"), positive=True), "invalid per-file duration")
        require(type(entry.get("n_probs")) is int and entry["n_probs"] > 0, "invalid per-file probability count")
        audio_samples = round(entry["audio_duration_s"] * report["vad_info"]["sample_rate"])
        expected_probs = math.ceil(audio_samples / report["vad_info"]["frame_samples"])
        require(entry["n_probs"] == expected_probs, "per-file probability count does not cover the full audio")
        require(finite_number(entry.get("max_abs_prob_delta")) and entry["max_abs_prob_delta"] <= 1,
                "invalid per-file probability delta")
    require(len(set(labels)) == len(labels), "duplicate audio file identities")
    audio_paths = [pathlib.Path(label) if pathlib.Path(label).is_absolute() else common.REPO / label
                   for label in labels]
    vad_parity.validate_corpus_files(audio_paths, report["dataset"], report["split"],
                                    report["language"], record["variant"])
    require(sum(entry["n_segments"] for entry in files) == report["n_segments"], "segment aggregate mismatch")
    require(math.isclose(sum(entry["audio_duration_s"] for entry in files), report["audio_duration_s"],
                         rel_tol=1e-10, abs_tol=1e-8), "duration aggregate mismatch")
    require(max(entry["max_abs_prob_delta"] for entry in files) == report["max_abs_prob_delta"],
            "probability delta is not the maximum across all files")
    key = profiles.cell_key(report, "reference_parity")
    expected = profiles.apply_exceptions(
        record, "reference_parity", profiles.expected_reference_parity(record, profile))
    target = next((cell for cell in expected if profiles.cell_key(cell, "reference_parity") == key), None)
    require(target is not None, "dataset/split/language/quant/reference is not a publication cell")
    row = {field: report[field] for field in ROW_FIELDS}
    require(profiles.valid_reference_parity(row, target, profile_id), "publication probability/provenance gate failed")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=pathlib.Path, help="parity report paths")
    parser.add_argument("--reports", type=pathlib.Path, default=common.REPO / "reports/vad",
                        help="report directory, used when no paths are given")
    parser.add_argument("--models", default="", help="comma-separated variants (default: VAD records)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    records = common.load_records()
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    unknown = selected - records.keys()
    if unknown:
        parser.error(f"no catalog record for {', '.join(sorted(unknown))}")
    selected = selected or {variant for variant, record in records.items() if record.get("role") == "vad"}
    paths = args.paths or sorted(args.reports.rglob("*.json"))
    if not paths:
        print("error: no parity reports", file=sys.stderr)
        return 2
    best = {}
    rejected = 0
    for path in paths:
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict):
                raise ValueError("report is not an object")
            variant = report.get("variant")
            if variant not in records:
                raise ValueError(f"no catalog record for {variant!r}")
            if variant not in selected:
                continue
            row = row_from_report(records[variant], report)
            key = (variant, *profiles.cell_key(row, "reference_parity"))
            timestamp = datetime.fromisoformat(report["timestamp"].replace("Z", "+00:00"))
            previous = best.get(key)
            if previous is None or timestamp > previous[0]:
                best[key] = (timestamp, row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected += 1
            print(f"reject {path}: {exc}", file=sys.stderr)
    if rejected:
        print(f"{rejected} report(s) rejected; nothing written", file=sys.stderr)
        return 1
    added = replaced = 0
    changes = {}
    for (variant, *key_parts), (_, row) in sorted(best.items()):
        record = records[variant]
        rows = record.setdefault("reference_parity", [])
        key = tuple(key_parts)
        indices = [index for index, existing in enumerate(rows)
                   if profiles.cell_key(existing, "reference_parity") == key]
        if indices:
            if len(indices) == 1 and rows[indices[0]] == row:
                continue
            rows[indices[0]] = row
            for index in reversed(indices[1:]):
                del rows[index]
            replaced += 1
        else:
            rows.append(row)
            added += 1
        changes[variant] = record
    if not args.dry_run:
        for variant, record in changes.items():
            common.write_record(common.CATALOG_DIR / f"{variant}.json", record)
    print(f"{len(best)} parity cell(s): {added} added, {replaced} replaced")
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
