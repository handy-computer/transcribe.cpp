"""Every catalog row property reaches the database, or is excluded on purpose.

    uv run --with pytest pytest scripts/catalog/test_db_mapping.py
"""
import json
import pathlib
import re
import sqlite3
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import db  # noqa: E402

SCHEMA = json.loads((HERE.parents[1] / "catalog" / "_schema.json").read_text())

# Row properties that are flattened or renamed rather than stored one to one.
ACCURACY_MAPPED = {"dataset": "dataset_id", "split": "dataset_id", "language": "dataset_id",
                   "ci95": "ci_lo/ci_hi", "errors": "substitutions/deletions/insertions",
                   "agreement": "agreement_n_agree/agreement_n/agreement_max_abs_logit_delta"}
SPEED_MAPPED = {}
PARITY_MAPPED = {key: "dataset_id" for key in ("dataset", "split", "language")}


def columns(table: str) -> set[str]:
    con = sqlite3.connect(":memory:")
    con.executescript(db.SCHEMA)
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def check(section: str, table: str, mapped: dict[str, str]) -> None:
    props = set(SCHEMA["properties"][section]["items"]["properties"])
    cols = columns(table)
    missing = sorted(p for p in props if p not in cols and p not in mapped)
    assert not missing, f"{section} properties with no {table} column: {missing}"
    for prop, target in mapped.items():
        for col in re.split(r"/", target):
            assert col in cols, f"{section}.{prop} maps to missing column {col}"


def test_accuracy_rows_reach_the_database():
    check("accuracy_benchmarks", "accuracy", ACCURACY_MAPPED)


def test_speed_rows_reach_the_database():
    check("speed_benchmarks", "speed", SPEED_MAPPED)


def test_download_rows_reach_the_database():
    check("downloads", "downloads", {})


def test_reference_parity_rows_reach_the_database():
    check("reference_parity", "reference_parity", PARITY_MAPPED)


def test_model_metadata_reaches_the_database():
    cols = columns("models")
    for prop, col in {"role": "role", "docs_page": "docs_page", "upstream_url": "upstream_url",
                      "source_artifact": "source_artifact_json", "vad_info": "vad_info_json"}.items():
        assert prop in SCHEMA["properties"]
        assert col in cols


def test_existing_asr_and_langid_rows_are_unchanged(tmp_path):
    records = db.common.load_records()
    selected = {name: record for name, record in records.items()
                if name == "whisper-tiny" or record.get("role") == "langid"}
    assert selected
    path = tmp_path / "catalog.db"
    db.build(selected, path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM reference_parity").fetchone()[0] == 0
        for variant, record in selected.items():
            model = dict(con.execute("SELECT * FROM models WHERE variant=?", (variant,)).fetchone())
            assert model["role"] == record.get("role", "asr")
            assert model["vad_info_json"] is None
            accuracy = con.execute(
                "SELECT a.*, d.dataset, d.split, d.language FROM accuracy a "
                "JOIN datasets d USING(dataset_id) WHERE variant=?", (variant,)).fetchall()
            actual = {db.profiles.cell_key(dict(row), "accuracy"): dict(row) for row in accuracy}
            for row in record["accuracy_benchmarks"]:
                stored = actual[db.profiles.cell_key(row, "accuracy")]
                for key, value in row.items():
                    if key not in ACCURACY_MAPPED:
                        assert stored[key] == value
                assert [stored["ci_lo"], stored["ci_hi"]] == row["ci95"]
                agreement = row.get("agreement") or {}
                for key in ("n_agree", "n", "max_abs_logit_delta"):
                    assert stored[f"agreement_{key}"] == agreement.get(key)
            speed = con.execute("SELECT * FROM speed WHERE variant=?", (variant,)).fetchall()
            actual = {db.profiles.cell_key(dict(row), "speed"): dict(row) for row in speed}
            for row in record["speed_benchmarks"]:
                stored = actual[db.profiles.cell_key(row, "speed")]
                assert all(stored[key] == value for key, value in row.items())
                assert stored["feed_samples"] is None
                assert stored["median_ms"] is None
            expected_headline = db.common.headline_rows(record)
            headline = con.execute("SELECT * FROM headline WHERE variant=?", (variant,)).fetchall()
            assert {row["quant"] for row in headline} == set(expected_headline)
