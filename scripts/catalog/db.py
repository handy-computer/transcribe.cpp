#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Build the portable catalog database from catalog JSON records.

The SQLite file is a disposable query artifact; catalog/*.json is the source of
truth. The database contains the complete model catalog, every accuracy row and
every speed row. It replaces the former specialized WER database.

    uv run scripts/catalog/db.py
    uv run scripts/catalog/db.py --out path/to/catalog.db
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "reports" / "wer" / "wer.db"
ORIGINAL_DB = REPO / "reports" / "wer" / "wer.db.original"

SCHEMA = """
PRAGMA user_version = 8;
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE models(
    model TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    display_name TEXT NOT NULL,
    params INTEGER NOT NULL,
    params_m REAL NOT NULL,
    architecture_pattern TEXT,
    license TEXT NOT NULL,
    license_display TEXT NOT NULL,
    upstream_repo TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    published_repo TEXT,
    language_tag_form TEXT,
    encoder_window_s REAL,
    long_form_strategy TEXT NOT NULL,
    max_audio_s REAL,
    max_output_tokens INTEGER
);

CREATE TABLE languages(
    lang TEXT PRIMARY KEY
);
CREATE TABLE model_languages(
    model TEXT NOT NULL REFERENCES models(model),
    lang TEXT NOT NULL REFERENCES languages(lang),
    PRIMARY KEY(model, lang)
);
CREATE TABLE language_aliases(
    model TEXT NOT NULL REFERENCES models(model),
    alias TEXT NOT NULL REFERENCES languages(lang),
    canonical TEXT NOT NULL REFERENCES languages(lang),
    PRIMARY KEY(model, alias)
);
CREATE VIEW model_languages_canonical AS
SELECT ml.model, COALESCE(a.canonical, ml.lang) AS lang
FROM model_languages ml
LEFT JOIN language_aliases a ON a.model = ml.model AND a.alias = ml.lang;

CREATE TABLE capabilities(
    model TEXT NOT NULL REFERENCES models(model),
    capability TEXT NOT NULL,
    supported INTEGER NOT NULL,
    verified INTEGER,
    note TEXT,
    details_json TEXT NOT NULL,
    PRIMARY KEY(model, capability)
);

CREATE TABLE quants(
    model TEXT NOT NULL REFERENCES models(model),
    quant TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    size_gb REAL NOT NULL,
    PRIMARY KEY(model, quant)
);

CREATE TABLE datasets(
    dataset TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    split TEXT NOT NULL,
    lang TEXT NOT NULL REFERENCES languages(lang)
);
CREATE TABLE results(
    result_id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL REFERENCES datasets(dataset),
    model TEXT NOT NULL REFERENCES models(model),
    quant TEXT NOT NULL,
    metric TEXT NOT NULL,
    err_pct REAL NOT NULL CHECK(err_pct >= 0),
    ci_lo REAL,
    ci_hi REAL,
    n_utts INTEGER NOT NULL CHECK(n_utts > 0),
    batch_size INTEGER,
    timestamps TEXT,
    engine_sha TEXT,
    measured_on TEXT,
    substitutions INTEGER,
    deletions INTEGER,
    insertions INTEGER,
    empty_hyp INTEGER,
    utts_over_50pct INTEGER
);
CREATE UNIQUE INDEX results_identity ON results(
    dataset, model, quant, metric,
    IFNULL(batch_size, 0), IFNULL(timestamps, '')
);

CREATE TABLE rigs(
    rig TEXT PRIMARY KEY,
    display TEXT NOT NULL
);
CREATE TABLE perf(
    model TEXT NOT NULL REFERENCES models(model),
    rig TEXT NOT NULL REFERENCES rigs(rig),
    backend TEXT NOT NULL,
    quant TEXT NOT NULL,
    sample TEXT NOT NULL,
    sample_s REAL NOT NULL,
    total_ms REAL,
    xrt REAL NOT NULL,
    load_ms REAL,
    mel_ms REAL,
    encode_ms REAL,
    decode_ms REAL,
    engine_sha TEXT,
    measured_on TEXT,
    thermal_gated INTEGER,
    PRIMARY KEY(model, rig, backend, quant, sample)
);

-- Friendly full-catalog views. The base table names retain compatibility with
-- the original WER database; these expose the terminology used by catalog JSON.
CREATE VIEW downloads AS
SELECT model AS variant, quant, filename, size_bytes FROM quants;
CREATE VIEW accuracy AS
SELECT r.model AS variant, d.source AS dataset, d.split, d.lang AS language,
       r.quant, r.metric, r.err_pct, r.ci_lo, r.ci_hi, r.n_utts,
       r.batch_size, r.timestamps, r.engine_sha, r.measured_on,
       r.substitutions, r.deletions, r.insertions, r.empty_hyp,
       r.utts_over_50pct
FROM results r JOIN datasets d ON d.dataset = r.dataset;
CREATE VIEW speed AS
SELECT model AS variant, rig AS machine, backend, quant, sample,
       sample_s AS sample_duration_s, total_ms, xrt AS xrt_compute,
       load_ms, mel_ms, encode_ms, decode_ms, engine_sha, measured_on,
       thermal_gated
FROM perf;
"""


def dataset_id(row: dict) -> str:
    source, split, lang = row["dataset"], row["split"], row["language"]
    if source == "fleurs" and split == "test":
        return f"fleurs-{lang}"
    if source == "librispeech":
        return f"librispeech-{split}"
    return f"{source}-{split}-{lang}"


def load_records(directory: pathlib.Path) -> list[dict]:
    return [json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))
            if not path.name.startswith("_")]


def build(directory: pathlib.Path, out: pathlib.Path) -> dict[str, int]:
    records = load_records(directory)
    if not records:
        raise RuntimeError(f"no catalog records in {directory}")

    langs = {str(lang) for record in records for lang in record.get("languages", [])}
    langs.update(row["language"] for record in records
                 for row in record.get("accuracy_benchmarks", []))
    langs.update(alias for record in records
                 for alias in (record.get("language_aliases") or {}))
    langs.update(canonical for record in records
                 for canonical in (record.get("language_aliases") or {}).values())
    rigs = {row["machine"] for record in records
            for row in record.get("speed_benchmarks", [])}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO languages VALUES (?)", [
            (lang,) for lang in sorted(langs)])
        con.executemany("INSERT INTO rigs VALUES (?,?)", [
            (rig, rig.replace("-", " ").title()) for rig in sorted(rigs)])

        datasets: dict[str, tuple[str, str, str]] = {}
        for record in records:
            for row in record.get("accuracy_benchmarks", []):
                datasets[dataset_id(row)] = (row["dataset"], row["split"], row["language"])
        con.executemany("INSERT INTO datasets VALUES (?,?,?,?)", [
            (key, *value) for key, value in sorted(datasets.items())])

        for record in records:
            model = record["variant"]
            license_info = record["license"]
            con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                model, record["family"], record["display_name"], record["params"],
                record["params"] / 1e6, record.get("architecture_pattern"),
                license_info["spdx"], license_info["display"], record["upstream_repo"],
                record["upstream_commit"], record.get("published_repo"), record.get("language_tag_form"),
                record.get("encoder_window_s"), record["long_form_strategy"],
                record.get("max_audio_s"), record.get("max_output_tokens")))
            con.executemany("INSERT INTO model_languages VALUES (?,?)", [
                (model, str(lang)) for lang in record.get("languages", [])])
            con.executemany("INSERT INTO language_aliases VALUES (?,?,?)", [
                (model, alias, canonical)
                for alias, canonical in (record.get("language_aliases") or {}).items()])
            con.executemany("INSERT INTO capabilities VALUES (?,?,?,?,?,?)", [
                (model, name, int(bool(cap.get("supported"))),
                 None if cap.get("verified") is None else int(cap["verified"]),
                 cap.get("note"), json.dumps(cap, separators=(",", ":"), sort_keys=True))
                for name, cap in record.get("capabilities", {}).items()])
            con.executemany("INSERT INTO quants VALUES (?,?,?,?,?)", [
                (model, item["quant"], item["filename"], item["size_bytes"],
                 item["size_bytes"] / 1e9) for item in record.get("downloads", [])])
            con.executemany(
                "INSERT INTO results(dataset,model,quant,metric,err_pct,ci_lo,ci_hi,n_utts,"
                "batch_size,timestamps,engine_sha,measured_on,substitutions,deletions,insertions,"
                "empty_hyp,utts_over_50pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                    (dataset_id(row), model, row["quant"], row["metric"], row["err_pct"],
                     (row.get("ci95") or [None, None])[0],
                     (row.get("ci95") or [None, None])[1], row["n_utts"],
                     row.get("batch_size"), row.get("timestamps"), row.get("engine_sha"),
                     row.get("measured_on"), (row.get("errors") or {}).get("sub"),
                     (row.get("errors") or {}).get("del"),
                     (row.get("errors") or {}).get("ins"), row.get("empty_hyp"),
                     row.get("utts_over_50pct"))
                    for row in record.get("accuracy_benchmarks", [])])
            con.executemany(
                "INSERT INTO perf VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                    (model, row["machine"], row["backend"], row["quant"], row["sample"],
                     row["sample_duration_s"], row.get("total_ms"), row["xrt_compute"],
                     row.get("load_ms"), row.get("mel_ms"), row.get("encode_ms"),
                     row.get("decode_ms"), row.get("engine_sha"), row.get("measured_on"),
                     None if row.get("thermal_gated") is None else int(row["thermal_gated"]))
                    for row in record.get("speed_benchmarks", [])])

        con.executemany("INSERT INTO meta VALUES (?,?)", [
            ("generated", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("source", "catalog/*.json"),
            ("rebuild", "uv run scripts/catalog/db.py (drops and recreates; never hand-edit)"),
            ("dataset_scope", "all catalog accuracy rows"),
        ])
        con.commit()
        counts = {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("models", "languages", "model_languages",
                                "language_aliases", "capabilities", "quants", "datasets",
                                "results", "rigs", "perf")}
    finally:
        con.close()
    os.replace(tmp, out)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(REPO / "catalog"))
    parser.add_argument("--out", default=str(DEFAULT_DB))
    args = parser.parse_args()
    out = pathlib.Path(args.out)

    if out == DEFAULT_DB and out.exists() and not ORIGINAL_DB.exists():
        shutil.copy2(out, ORIGINAL_DB)
        print(f"preserved original database: {ORIGINAL_DB}")

    try:
        counts = build(pathlib.Path(args.dir), out)
    except (OSError, ValueError, KeyError, sqlite3.Error, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for table, count in counts.items():
        print(f"  {table:20s} {count:>6}")
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
