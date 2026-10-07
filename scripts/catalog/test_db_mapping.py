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
                   "ci95": "ci_lo/ci_hi", "errors": "substitutions/deletions/insertions"}
SPEED_MAPPED = {}


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
