"""VAD catalog contracts. All numbers here are synthetic test fixtures.

    uv run --with pytest --with jsonschema pytest scripts/catalog/test_vad_contract.py
"""
import copy
import json
import pathlib
import sqlite3
import sys
import wave

import pytest
from jsonschema import Draft202012Validator

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
import common  # noqa: E402
import db  # noqa: E402
import profiles  # noqa: E402

SCHEMA = json.loads((common.REPO / "catalog/_schema.json").read_text())
PROFILE_ID, PROFILE = profiles.load_profile("vad-publication-v1")


@pytest.fixture
def record():
    rec = {
        "schema": "transcribe-catalog-v1", "variant": "test-vad", "family": "silero_vad",
        "role": "vad", "display_name": "Test VAD", "params": 1,
        "license": {"spdx": "mit", "display": "MIT"},
        "upstream_repo": "snakers4/silero-vad", "upstream_commit": "abcdef0",
        "upstream_url": "https://github.com/snakers4/silero-vad",
        "source_artifact": {"package": "silero-vad", "version": "6.2.3",
                            "filename": "silero_vad.jit", "sha256": "a" * 64,
                            "url": "https://pypi.org/project/silero-vad/6.2.3/"},
        "published_repo": None, "docs_page": "test-vad.md", "languages": [],
        "vad_info": {"sample_rate": 16000, "frame_samples": 512},
        "long_form_strategy": "chunked-unbounded",
        "capabilities": {name: {"supported": False} for name in (
            "transcribe", "translate", "lang_detect", "timestamps", "streaming",
            "diarize", "batching")},
        "downloads": [{"quant": "F32", "filename": "test-F32.gguf", "size_bytes": 1}],
        "accuracy_benchmarks": [], "headline_benchmark": None,
        "speed_benchmarks": [], "reference_parity": [],
    }
    provenance = {"engine_sha": "abcdef01", "measured_on": "2026-04-01",
                  "publication_profile": PROFILE_ID}
    for cell in profiles.expected_reference_parity(rec, PROFILE):
        row = {key: cell[key] for key in profiles.PARITY_KEY}
        row.update(provenance, n_files=cell["n_files"], n_identical=cell["n_files"], n_segments=3,
                   audio_duration_s=10.0, segmentation_params="defaults")
        if "max_abs_prob_delta" in cell:
            row["max_abs_prob_delta"] = 1e-6
        rec["reference_parity"].append(row)
    for cell in profiles.expected_speed(rec, PROFILE):
        rec["speed_benchmarks"].append({**cell, **provenance, "total_ms": 2.0,
                                      "median_ms": 1.5, "p95_ms": 3.0,
                                      "xrt_compute": cell["sample_duration_s"] * 1000 / 2.0})
    return rec


def test_schema_and_complete_publication(record):
    Draft202012Validator.check_schema(SCHEMA)
    assert not list(Draft202012Validator(SCHEMA).iter_errors(record))
    assert check.integrity_pass({"test": record}) == 0
    assert check.provenance_pass({"test": record}) == 0
    assert check.publication_pass({"test": record}, None, True) == 0
    # Explicit ASR selection cannot silently skip a VAD's role profile.
    assert check.publication_pass({"test": record}, "asr-publication-v2", True) == 0
    assert common.headline_rows(record) == {}


def test_empty_languages_only_for_vad(record):
    for role in (None, "asr", "langid", "diarize"):
        rec = copy.deepcopy(record)
        if role is None:
            rec.pop("role")
        else:
            rec["role"] = role
        assert list(Draft202012Validator(SCHEMA).iter_errors(rec))


def test_vad_info_required_from_metadata(record):
    record.pop("vad_info")
    assert list(Draft202012Validator(SCHEMA).iter_errors(record))
    assert check.publication_pass({"test": record}, None, True) > 0


@pytest.mark.parametrize("key,value", [
    ("upstream_url", "https://huggingface.co/snakers4/silero-vad"),
    ("upstream_commit", "v6.2.3"),
    ("upstream_commit", "a" * 40),
    ("source_artifact", {"package": "silero-vad"}),
])
def test_source_schema(record, key, value):
    record[key] = value
    assert list(Draft202012Validator(SCHEMA).iter_errors(record))


def test_source_optional_and_short_github_sha_unchanged(record):
    record.pop("source_artifact")
    record.pop("upstream_url")
    assert not list(Draft202012Validator(SCHEMA).iter_errors(record))


def test_exact_suites_every_download_independent_of_languages(record):
    record["downloads"].append({"quant": "F16", "filename": "test-F16.gguf", "size_bytes": 1})
    cells = profiles.expected_reference_parity(record, PROFILE)
    suites = {tuple(cell[k] for k in ("dataset", "split", "language")) for cell in cells}
    assert suites == {("golden", "validation", "mul"), ("ami", "ihm-test", "en"),
                      ("librispeech", "test-clean", "en"),
                      *(("fleurs", "test", lang) for lang in ("zh", "ja", "ar", "de"))}
    assert len(cells) == 14
    expected_counts = {("golden", "mul"): 9, ("ami", "en"): 16,
                       ("librispeech", "en"): 2620, ("fleurs", "zh"): 945,
                       ("fleurs", "ja"): 650, ("fleurs", "ar"): 428,
                       ("fleurs", "de"): 862}
    assert all(cell["n_files"] == expected_counts[cell["dataset"], cell["language"]]
               for cell in cells)
    assert all(cell["backend"] == "cpu" and cell["reference"] == "silero-vad==6.2.3"
               for cell in cells)
    assert profiles.expected_accuracy(record, PROFILE) == []
    assert len(profiles.expected_speed(record, PROFILE)) == 6


def test_recipe_matches_shipped_audio():
    spec = PROFILE["speed"]
    with wave.open(str(common.REPO / "samples" / f"{spec['source_sample']}.wav")) as audio:
        assert audio.getnframes() == spec["source_samples"]
        assert audio.getframerate() == spec["sample_rate"]
    assert spec["iterations"] == 1
    assert spec["full_calls_only"] is True


@pytest.mark.parametrize("key,value", [
    ("n_identical", 1), ("n_identical", 3), ("n_identical", -1),
    ("n_files", 0), ("n_segments", -1), ("audio_duration_s", 0),
    ("audio_duration_s", float("nan")), ("segmentation_params", "custom"),
    ("max_abs_prob_delta", 1.01e-5), ("max_abs_prob_delta", None),
    ("max_abs_prob_delta", float("inf")), ("max_abs_prob_delta", -1),
    ("engine_sha", None), ("engine_sha", "unknown"), ("measured_on", "2026-02-30"),
    ("publication_profile", "asr-publication-v2"),
])
def test_invalid_parity_publication(record, key, value):
    record["reference_parity"][0][key] = value
    assert check.publication_pass({"test": record}, None, True) > 0


@pytest.mark.parametrize("key,value", [
    ("n_files", 0), ("n_identical", 10), ("n_segments", -1),
    ("audio_duration_s", float("nan")), ("dataset", ""),
    ("max_abs_prob_delta", float("nan")), ("max_abs_prob_delta", 1.01),
])
def test_parity_integrity_floor(record, key, value):
    record["reference_parity"][0][key] = value
    assert check.integrity_pass({"test": record}) > 0


def test_parity_duplicate_integrity_floor(record):
    record["reference_parity"].append(copy.deepcopy(record["reference_parity"][0]))
    assert check.integrity_pass({"test": record}) > 0


@pytest.mark.parametrize("kind", ["reference_parity", "speed_benchmarks"])
@pytest.mark.parametrize("key,value", [("engine_sha", None), ("engine_sha", "not-a-build"),
                                      ("measured_on", "2026-02-30"),
                                      ("publication_profile", None)])
def test_vad_provenance_floor(record, kind, key, value):
    record[kind][0][key] = value
    assert check.provenance_pass({"test": record}) > 0


@pytest.mark.parametrize("index", range(7))
def test_publication_requires_the_full_suite(record, index):
    row = record["reference_parity"][index]
    # Exact agreement on a one-file subset is not a full-split measurement.
    row["n_files"] = row["n_identical"] = 1
    assert check.integrity_pass({"test": record}) == 0
    assert check.publication_pass({"test": record}, None, True) > 0


def test_probability_bound_and_required_golden_delta(record):
    row = record["reference_parity"][0]
    row["max_abs_prob_delta"] = 1e-5
    assert check.publication_pass({"test": record}, None, True) == 0
    row.pop("max_abs_prob_delta")
    assert check.publication_pass({"test": record}, None, True) > 0


@pytest.mark.parametrize("kind", ["reference_parity", "speed_benchmarks"])
def test_missing_duplicate_extra_and_unpublished_quant(record, kind):
    for change in ("missing", "duplicate", "extra", "quant"):
        rec = copy.deepcopy(record)
        if change == "missing":
            rec[kind].pop()
        elif change == "duplicate":
            rec[kind].append(copy.deepcopy(rec[kind][0]))
        elif change == "extra":
            rec[kind][0]["backend"] = "metal"
        else:
            rec[kind][0]["quant"] = "Q8_0"
            assert check.integrity_pass({"test": rec}) > 0
        assert check.publication_pass({"test": rec}, None, True) > 0
    # Audit reports missing data but remains advisory.
    record[kind] = []
    assert check.publication_pass({"test": record}, None, False) == 0


@pytest.mark.parametrize("key,value", [
    ("feed_samples", 1024), ("frame_samples", 256), ("threads", 2),
    ("warmup_calls", 31), ("n_calls", 6162), ("n_calls", 6160),
    ("sample_duration_s", 197.1549375), ("total_ms", None),
    ("total_ms", 0), ("total_ms", float("nan")), ("median_ms", 4.0),
    ("median_ms", 0), ("p95_ms", -1), ("p95_ms", float("inf")),
    ("xrt_compute", 999), ("engine_sha", None), ("measured_on", None),
    ("publication_profile", None), ("measurement_provenance", "legacy-published"),
])
def test_speed_recipe_invariants(record, key, value):
    record["speed_benchmarks"][0][key] = value
    assert check.publication_pass({"test": record}, None, True) > 0


def test_missing_speed_statistics(record):
    for key in ("feed_samples", "frame_samples", "threads", "warmup_calls", "n_calls",
                "median_ms", "p95_ms"):
        rec = copy.deepcopy(record)
        rec["speed_benchmarks"][0].pop(key)
        assert check.publication_pass({"test": rec}, None, True) > 0


def test_metadata_frame_geometry_matches_profile_and_speed(record):
    record["vad_info"]["frame_samples"] = 256
    assert check.publication_pass({"test": record}, None, True) > 0
    record["vad_info"] = {"sample_rate": 8000, "frame_samples": 512}
    assert check.publication_pass({"test": record}, None, True) > 0


def test_threads_and_feed_recipe_do_not_change_speed_identity(record):
    row = record["speed_benchmarks"][0]
    assert profiles.cell_key(row, "speed") == profiles.cell_key(
        {**row, "threads": 2, "feed_samples": 1024}, "speed")


def test_no_waivers_or_asr_headline(record):
    record["benchmark_exceptions"] = [{"kind": "speed", "match": {"sample": "*"},
                                       "reason": "synthetic test waiver"}]
    assert check.publication_pass({"test": record}, None, True) > 0
    record.pop("benchmark_exceptions")
    record["headline_benchmark"] = common.load_record("whisper-tiny")["headline_benchmark"]
    assert list(Draft202012Validator(SCHEMA).iter_errors(record))
    assert check.publication_pass({"test": record}, None, True) > 0


def test_no_silent_skip_when_transcribe_is_false(record):
    record["reference_parity"] = []
    record["speed_benchmarks"] = []
    assert check.publication_pass({"test": record}, None, True) == 10


def test_no_speech_parity_is_valid(record, tmp_path):
    record["reference_parity"][0]["n_segments"] = 0
    assert not list(Draft202012Validator(SCHEMA).iter_errors(record))
    assert check.integrity_pass({"test": record}) == 0
    assert check.publication_pass({"test": record}, None, True) == 0
    path = tmp_path / "catalog.db"
    db.build({record["variant"]: record}, path)
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT n_segments FROM reference_parity WHERE dataset_id=?",
                           (db.dataset_id(record["reference_parity"][0]),)).fetchone()[0] == 0


def test_vad_database_roundtrip(record, tmp_path):
    path = tmp_path / "catalog.db"
    counts = db.build({record["variant"]: record}, path)
    assert counts["reference_parity"] == 7
    assert counts["accuracy"] == 0
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
        model = dict(con.execute("SELECT * FROM models").fetchone())
        for key in ("role", "docs_page", "upstream_url", "upstream_commit", "published_repo"):
            assert model[key] == record[key]
        assert json.loads(model["source_artifact_json"]) == record["source_artifact"]
        assert json.loads(model["vad_info_json"]) == record["vad_info"]
        assert con.execute("SELECT count(*) FROM model_languages").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM headline").fetchone()[0] == 0
        rows = con.execute("SELECT p.*, d.dataset, d.split, d.language FROM reference_parity p "
                           "JOIN datasets d USING(dataset_id)").fetchall()
        actual = {profiles.cell_key(dict(row), "reference_parity"): dict(row) for row in rows}
        for row in record["reference_parity"]:
            stored = actual[profiles.cell_key(row, "reference_parity")]
            assert all(stored[key] == value for key, value in row.items())
            assert stored["max_abs_prob_delta"] == row.get("max_abs_prob_delta")
        rows = con.execute("SELECT * FROM speed").fetchall()
        actual = {profiles.cell_key(dict(row), "speed"): dict(row) for row in rows}
        for row in record["speed_benchmarks"]:
            assert all(actual[profiles.cell_key(row, "speed")][key] == value
                       for key, value in row.items())
