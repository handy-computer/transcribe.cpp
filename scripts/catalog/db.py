#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""db.py — fold catalog records into build/catalog.db.

A derived index, never a store: dropped and rebuilt on every run, nothing
hand-edited in it, and it never writes anywhere the WER tooling reads. Because
the input is already structured, this is a fold with no parsers -- which is the
whole point of the records existing.

    uv run scripts/catalog/db.py
    uv run scripts/catalog/db.py --out build/catalog.db
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[2]

SCHEMA = """
PRAGMA user_version = 1;
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE models(
    variant TEXT PRIMARY KEY, family TEXT NOT NULL, display_name TEXT,
    params INTEGER, license TEXT, upstream_repo TEXT, published_repo TEXT,
    long_form_strategy TEXT, max_audio_s REAL);
CREATE TABLE languages(
    variant TEXT REFERENCES models(variant), lang TEXT,
    PRIMARY KEY(variant, lang));
CREATE TABLE capabilities(
    variant TEXT REFERENCES models(variant), capability TEXT,
    supported INTEGER NOT NULL, verified INTEGER, note TEXT,
    PRIMARY KEY(variant, capability));
CREATE TABLE downloads(
    variant TEXT REFERENCES models(variant), quant TEXT,
    filename TEXT NOT NULL, size_bytes INTEGER,
    PRIMARY KEY(variant, quant));
CREATE TABLE accuracy(
    variant TEXT REFERENCES models(variant), dataset TEXT, split TEXT,
    language TEXT, quant TEXT, metric TEXT NOT NULL,
    err_pct REAL NOT NULL, ci_lo REAL, ci_hi REAL, n_utts INTEGER NOT NULL,
    recipe TEXT NOT NULL, engine_sha TEXT, measured_on TEXT,
    sub INTEGER, del_ INTEGER, ins INTEGER, empty_hyp INTEGER,
    PRIMARY KEY(variant, dataset, split, language, quant, recipe));
CREATE TABLE speed(
    variant TEXT REFERENCES models(variant), machine TEXT, backend TEXT,
    quant TEXT, sample TEXT, sample_duration_s REAL,
    total_ms REAL NOT NULL, xrt REAL NOT NULL,
    load_ms REAL, mel_ms REAL, encode_ms REAL, decode_ms REAL,
    engine_sha TEXT, measured_on TEXT, thermal_gated INTEGER,
    PRIMARY KEY(variant, machine, backend, quant, sample));
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "catalog"))
    ap.add_argument("--out", default=str(REPO / "build" / "catalog.db"))
    args = ap.parse_args()

    records = [json.loads(p.read_text())
               for p in sorted(pathlib.Path(args.dir).glob("*.json"))
               if not p.name.startswith("_")]
    if not records:
        print("no records", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)

    for r in records:
        v = r["variant"]
        con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)",
                    (v, r.get("family"), r.get("display_name"), r.get("params"),
                     (r.get("license") or {}).get("spdx"), r.get("upstream_repo"),
                     r.get("published_repo"), r.get("long_form_strategy"),
                     r.get("max_audio_s")))
        con.executemany("INSERT OR IGNORE INTO languages VALUES (?,?)",
                        [(v, l) for l in r.get("languages", [])])
        con.executemany("INSERT INTO capabilities VALUES (?,?,?,?,?)",
                        [(v, k, int(bool(c.get("supported"))),
                          None if c.get("verified") is None else int(c["verified"]),
                          c.get("note")) for k, c in (r.get("capabilities") or {}).items()])
        con.executemany("INSERT INTO downloads VALUES (?,?,?,?)",
                        [(v, d["quant"], d["filename"], d.get("size_bytes"))
                         for d in r.get("downloads", [])])
        con.executemany("INSERT OR REPLACE INTO accuracy VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(v, a["dataset"], a["split"], a["language"], a["quant"], a["metric"],
                          a["err_pct"], (a.get("ci95") or [None, None])[0],
                          (a.get("ci95") or [None, None])[1], a["n_utts"], a["recipe"],
                          a.get("engine_sha"), a.get("measured_on"),
                          (a.get("errors") or {}).get("sub"), (a.get("errors") or {}).get("del"),
                          (a.get("errors") or {}).get("ins"), a.get("empty_hyp"))
                         for a in r.get("accuracy_benchmarks", [])])
        con.executemany("INSERT OR REPLACE INTO speed VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(v, s["machine"], s["backend"], s["quant"], s["sample"],
                          s["sample_duration_s"], s["total_ms"], s["xrt_compute"],
                          s.get("load_ms"), s.get("mel_ms"), s.get("encode_ms"),
                          s.get("decode_ms"), s.get("engine_sha"), s.get("measured_on"),
                          None if s.get("thermal_gated") is None else int(s["thermal_gated"]))
                         for s in r.get("speed_benchmarks", [])])

    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("generated", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("source", "catalog/*.json"),
        ("rebuild", "uv run scripts/catalog/db.py (drops and recreates; never hand-edit)")])
    con.commit()
    for t in ("models", "languages", "capabilities", "downloads", "accuracy", "speed"):
        print(f"  {t:14s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>6}")
    print(f"\n{out}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
