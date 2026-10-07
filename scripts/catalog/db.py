#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Build the portable catalog database from catalog JSON records.

The SQLite file is a disposable query artifact; catalog/*.json is the source
of truth. Tables mirror the record sections one to one (downloads, accuracy,
speed) so a query reads like the JSON it came from.

    uv run scripts/catalog/db.py                       # build/catalog.db
    uv run scripts/catalog/db.py --out path/to/catalog.db
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import profiles  # noqa: E402

DEFAULT_DB = common.REPO / "build" / "catalog.db"

SCHEMA = """
PRAGMA user_version = 1;
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE models(
    variant TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    display_name TEXT NOT NULL,
    params INTEGER NOT NULL,
    license_spdx TEXT NOT NULL,
    license_display TEXT NOT NULL,
    upstream_repo TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    published_repo TEXT,
    language_tag_form TEXT,
    encoder_window_s REAL,
    long_form_strategy TEXT NOT NULL,
    max_audio_s REAL,
    max_output_tokens INTEGER,
    -- Which accuracy row-set this model publishes as its headline number.
    -- Recipe fields may be null to span quant rows measured with different
    -- batch sizes while each quant still has one published result.
    headline_dataset TEXT,
    headline_metric TEXT,
    headline_batch_size INTEGER,
    headline_timestamps TEXT
);

CREATE TABLE languages(
    lang TEXT PRIMARY KEY
);
CREATE TABLE model_languages(
    variant TEXT NOT NULL REFERENCES models(variant),
    lang TEXT NOT NULL REFERENCES languages(lang),
    PRIMARY KEY(variant, lang)
);
CREATE TABLE language_aliases(
    variant TEXT NOT NULL REFERENCES models(variant),
    alias TEXT NOT NULL REFERENCES languages(lang),
    canonical TEXT NOT NULL REFERENCES languages(lang),
    PRIMARY KEY(variant, alias)
);

CREATE TABLE capabilities(
    variant TEXT NOT NULL REFERENCES models(variant),
    capability TEXT NOT NULL,
    supported INTEGER NOT NULL,
    verified INTEGER,
    note TEXT,
    details_json TEXT NOT NULL,
    PRIMARY KEY(variant, capability)
);

CREATE TABLE downloads(
    variant TEXT NOT NULL REFERENCES models(variant),
    quant TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    PRIMARY KEY(variant, quant)
);

CREATE TABLE datasets(
    dataset_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    language TEXT NOT NULL REFERENCES languages(lang)
);
CREATE TABLE accuracy(
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    variant TEXT NOT NULL REFERENCES models(variant),
    quant TEXT NOT NULL,
    metric TEXT NOT NULL,
    language_hint TEXT,
    backend TEXT,
    err_pct REAL NOT NULL CHECK(err_pct >= 0),
    ci_lo REAL,
    ci_hi REAL,
    n_utts INTEGER NOT NULL CHECK(n_utts > 0),
    batch_size INTEGER,
    timestamps TEXT,
    engine_sha TEXT,
    measurement_provenance TEXT,
    measured_on TEXT,
    substitutions INTEGER,
    deletions INTEGER,
    insertions INTEGER,
    empty_hyp INTEGER,
    utts_over_50pct INTEGER,
    publication_profile TEXT,
    scoring TEXT,
    mode TEXT
);
CREATE UNIQUE INDEX accuracy_identity ON accuracy(
    dataset_id, variant, quant, metric,
    IFNULL(timestamps, ''), IFNULL(scoring, ''), IFNULL(mode, '')
);

CREATE TABLE machines(
    machine TEXT PRIMARY KEY
);
CREATE TABLE speed(
    variant TEXT NOT NULL REFERENCES models(variant),
    machine TEXT NOT NULL REFERENCES machines(machine),
    backend TEXT NOT NULL,
    quant TEXT NOT NULL,
    sample TEXT NOT NULL,
    sample_duration_s REAL NOT NULL,
    total_ms REAL,
    xrt_compute REAL NOT NULL,
    wall_ms REAL,
    xrt_wall REAL,
    load_ms REAL,
    mel_ms REAL,
    encode_ms REAL,
    decode_ms REAL,
    engine_sha TEXT,
    measurement_provenance TEXT,
    measured_on TEXT,
    thermal_gated INTEGER,
    publication_profile TEXT,
    PRIMARY KEY(variant, machine, backend, quant, sample)
);

-- The per-quant column a model card and its doc print.
CREATE VIEW headline AS
SELECT a.variant, d.dataset, d.split, d.language,
       a.quant, a.metric, a.err_pct, a.ci_lo, a.ci_hi, a.n_utts
FROM accuracy a
JOIN models m ON m.variant = a.variant
JOIN datasets d ON d.dataset_id = a.dataset_id
WHERE a.dataset_id = m.headline_dataset
  AND a.metric = m.headline_metric
  AND (m.headline_batch_size IS NULL OR a.batch_size = m.headline_batch_size)
  AND (m.headline_timestamps IS NULL OR a.timestamps = m.headline_timestamps);
"""


def dataset_id(row: dict) -> str:
    """Primary key of the `datasets` table.

    The report slug drops whichever dimension its dataset holds constant, so
    it is not always a full identity: `fleurs-es` names its language and
    `librispeech-test-clean` has only one, but a dataset that varies by both
    keeps the language segment so two languages cannot collide on one key.
    """
    dataset, split, language = row["dataset"], row["split"], row["language"]
    if (dataset in common.LANGUAGE_KEYED_DATASETS
            and split == common.PUBLISHED_SPLIT.get(dataset)):
        return common.dataset_slug(row)                  # fleurs-es
    if dataset in common.SINGLE_LANGUAGE_DATASETS:
        return common.dataset_slug(row)                  # librispeech-test-clean
    return f"{dataset}-{split}-{language}"               # ami-ihm-test-en


def build(records: dict[str, dict], out: pathlib.Path) -> dict[str, int]:
    if not records:
        raise RuntimeError("no catalog records")

    langs = {str(lang) for record in records.values() for lang in record.get("languages", [])}
    langs.update(row["language"] for record in records.values()
                 for row in record.get("accuracy_benchmarks", []))
    for record in records.values():
        for alias, canonical in (record.get("language_aliases") or {}).items():
            langs.update((alias, canonical))
    machines = {row["machine"] for record in records.values()
                for row in record.get("speed_benchmarks", [])}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO languages VALUES (?)", [(lang,) for lang in sorted(langs)])
        con.executemany("INSERT INTO machines VALUES (?)", [(m,) for m in sorted(machines)])

        datasets: dict[str, tuple[str, str, str]] = {}
        for record in records.values():
            for row in record.get("accuracy_benchmarks", []):
                datasets[dataset_id(row)] = (row["dataset"], row["split"], row["language"])
        con.executemany("INSERT INTO datasets VALUES (?,?,?,?)", [
            (key, *value) for key, value in sorted(datasets.items())])

        for variant, record in records.items():
            license_info = record["license"]
            headline = record.get("headline_benchmark") or {}
            con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                variant, record["family"], record["display_name"], record["params"],
                license_info["spdx"], license_info["display"], record["upstream_repo"],
                record["upstream_commit"], record.get("published_repo"),
                record.get("language_tag_form"), record.get("encoder_window_s"),
                record["long_form_strategy"], record.get("max_audio_s"),
                record.get("max_output_tokens"),
                dataset_id(headline) if headline else None, headline.get("metric"),
                headline.get("batch_size"), headline.get("timestamps")))
            con.executemany("INSERT INTO model_languages VALUES (?,?)", [
                (variant, str(lang)) for lang in record.get("languages", [])])
            con.executemany("INSERT INTO language_aliases VALUES (?,?,?)", [
                (variant, alias, canonical)
                for alias, canonical in (record.get("language_aliases") or {}).items()])
            con.executemany("INSERT INTO capabilities VALUES (?,?,?,?,?,?)", [
                (variant, name, int(bool(cap.get("supported"))),
                 None if cap.get("verified") is None else int(cap["verified"]),
                 cap.get("note"), json.dumps(cap, separators=(",", ":"), sort_keys=True))
                for name, cap in record.get("capabilities", {}).items()])
            con.executemany("INSERT INTO downloads VALUES (?,?,?,?)", [
                (variant, item["quant"], item["filename"], item["size_bytes"])
                for item in record.get("downloads", [])])
            con.executemany(
                "INSERT INTO accuracy VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                    (dataset_id(row), variant, row["quant"], row["metric"],
                     row.get("language_hint"), row.get("backend"), row["err_pct"],
                     (row.get("ci95") or [None, None])[0],
                     (row.get("ci95") or [None, None])[1], row["n_utts"],
                     row.get("batch_size"), row.get("timestamps"), row.get("engine_sha"),
                     row.get("measurement_provenance"), row.get("measured_on"),
                     (row.get("errors") or {}).get("sub"),
                     (row.get("errors") or {}).get("del"),
                     (row.get("errors") or {}).get("ins"), row.get("empty_hyp"),
                     row.get("utts_over_50pct"), row.get("publication_profile"),
                     row.get("scoring"), row.get("mode"))
                    for row in record.get("accuracy_benchmarks", [])])
            con.executemany(
                "INSERT INTO speed VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                    (variant, row["machine"], row["backend"], row["quant"], row["sample"],
                     row["sample_duration_s"], row.get("total_ms"), row["xrt_compute"],
                     row.get("wall_ms"), row.get("xrt_wall"), row.get("load_ms"), row.get("mel_ms"), row.get("encode_ms"),
                     row.get("decode_ms"), row.get("engine_sha"),
                     row.get("measurement_provenance"), row.get("measured_on"),
                     None if row.get("thermal_gated") is None else int(row["thermal_gated"]),
                     row.get("publication_profile"))
                    for row in record.get("speed_benchmarks", [])])

        profile_id, _ = profiles.load_profile()
        con.executemany("INSERT INTO meta VALUES (?,?)", [
            ("generated", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("source", "catalog/*.json"),
            ("benchmark_profile", profile_id),
            ("rebuild", "uv run scripts/catalog/db.py (drops and recreates; never hand-edit)"),
        ])
        con.commit()
        counts = {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("models", "languages", "model_languages",
                                "language_aliases", "capabilities", "downloads", "datasets",
                                "accuracy", "machines", "speed")}
    finally:
        con.close()
    os.replace(tmp, out)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(common.CATALOG_DIR))
    parser.add_argument("--out", default=str(DEFAULT_DB))
    args = parser.parse_args()
    try:
        counts = build(common.load_records(pathlib.Path(args.dir)), pathlib.Path(args.out))
    except (OSError, ValueError, KeyError, sqlite3.Error, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for table, count in counts.items():
        print(f"  {table:20s} {count:>6}")
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
