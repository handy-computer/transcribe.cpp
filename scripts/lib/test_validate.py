#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Focused unit tests for validate.py transcript and timestamp plumbing.

Run standalone:  uv run scripts/lib/test_validate.py
Run with pytest: uv run --with pytest --with numpy python -m pytest scripts/lib/test_validate.py
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


_VALIDATE_PATH = Path(__file__).resolve().parents[1] / "validate.py"
_SPEC = importlib.util.spec_from_file_location("transcribe_validate", _VALIDATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate)

_PARAKEET_DUMPER_PATH = (
    Path(__file__).resolve().parents[1] / "dump_reference_parakeet_nemo.py"
)
_PARAKEET_DUMPER_SPEC = importlib.util.spec_from_file_location(
    "transcribe_parakeet_dumper", _PARAKEET_DUMPER_PATH
)
assert _PARAKEET_DUMPER_SPEC is not None and _PARAKEET_DUMPER_SPEC.loader is not None
parakeet_dumper = importlib.util.module_from_spec(_PARAKEET_DUMPER_SPEC)
_PARAKEET_DUMPER_SPEC.loader.exec_module(parakeet_dumper)


def test_parse_cli_jsonl_result() -> None:
    output = "\n".join([
        "backend log",
        '{"type":"batch_header","load_ms":12.5}',
        (
            '{"file":"jfk.wav","text":"hello","words":'
            '[{"t0_ms":80,"t1_ms":240,"text":"hello"}],"tokens":'
            '[{"t0_ms":80,"t1_ms":240,"text":" hello","id":7,"p":0.9}]}'
        ),
    ])
    result = validate.parse_cli_result(output)
    assert result is not None
    assert result["text"] == "hello"
    assert result["words"][0]["t0_ms"] == 80
    assert result["tokens"][0]["id"] == 7


def test_cpp_transcript_v2_carries_timestamp_rows() -> None:
    words = [{"t0_ms": 80, "t1_ms": 240, "text": "hello"}]
    tokens = [
        {
            "t0_ms": 80,
            "t1_ms": 240,
            "text": " hello",
            "id": 7,
            "p": 0.9,
        }
    ]
    with tempfile.TemporaryDirectory() as raw_dir:
        out_dir = Path(raw_dir)
        validate.write_cpp_transcript(
            out_dir,
            family="parakeet",
            variant="unit",
            case="jfk",
            gguf=Path("unit.gguf"),
            backend="cpu",
            text="hello",
            words=words,
            tokens=tokens,
        )
        payload = json.loads((out_dir / "transcript.json").read_text())
    assert payload["schema"] == "transcribe-cpp-transcript-v2"
    assert payload["words"] == words
    assert payload["tokens"] == tokens


def test_nemo_timestamp_rows_extracts_char_token_text() -> None:
    hypothesis = SimpleNamespace(
        timestamp={
            "word": [
                {"start": 0.0, "end": 0.8, "word": "And so,"},
            ],
            "char": [
                {"start": 0.0, "end": 0.32, "char": ["And"]},
                {"start": 0.72, "end": 0.8, "char": ["so"]},
                {"start": 0.8, "end": 0.8, "char": [","]},
            ],
        }
    )

    words, tokens = parakeet_dumper.nemo_timestamp_rows(hypothesis)

    assert words == [{"start_s": 0.0, "end_s": 0.8, "text": "And so,"}]
    assert tokens == [
        {"start_s": 0.0, "end_s": 0.32, "text": "And"},
        {"start_s": 0.72, "end_s": 0.8, "text": "so"},
        {"start_s": 0.8, "end_s": 0.8, "text": ","},
    ]


def test_timestamp_compare_normalizes_and_reports_stats() -> None:
    reference = {
        "words": [
            {"start_s": 0.08, "end_s": 0.48, "text": "Hello,"},
            {"start_s": 0.56, "end_s": 1.0, "text": "WORLD"},
        ]
    }
    cpp = {
        "words": [
            {"t0_ms": 120, "t1_ms": 520, "text": "hello"},
            {"t0_ms": 600, "t1_ms": 1040, "text": "world!"},
        ]
    }
    result = validate.case_timestamp_compare(
        reference, cpp, text_mode="normalized", tolerance_ms=80.0
    )
    assert result["match"] is True
    assert result["max_deviation_ms"] == 40.0
    assert result["mean_deviation_ms"] == 40.0


def test_timestamp_compare_reports_count_mismatch() -> None:
    reference = {
        "words": [
            {"start_s": 0.0, "end_s": 0.08, "text": "one"},
            {"start_s": 0.08, "end_s": 0.16, "text": "two"},
        ]
    }
    cpp = {"words": [{"t0_ms": 0, "t1_ms": 80, "text": "one"}]}
    result = validate.case_timestamp_compare(
        reference, cpp, text_mode="exact", tolerance_ms=80.0
    )
    assert result["match"] is False
    assert "word count mismatch: reference=2, C++=1" in result["reason"]


def test_timestamp_compare_enforces_endpoint_tolerance() -> None:
    reference = {
        "words": [{"start_s": 0.0, "end_s": 0.08, "text": "one"}]
    }
    cpp = {"words": [{"t0_ms": 0, "t1_ms": 161, "text": "one"}]}
    result = validate.case_timestamp_compare(
        reference, cpp, text_mode="exact", tolerance_ms=80.0
    )
    assert result["match"] is False
    assert result["max_deviation_ms"] == 81.0
    assert "exceeds 80.000 ms" in result["reason"]


_TESTS = [
    test_parse_cli_jsonl_result,
    test_cpp_transcript_v2_carries_timestamp_rows,
    test_nemo_timestamp_rows_extracts_char_token_text,
    test_timestamp_compare_normalizes_and_reports_stats,
    test_timestamp_compare_reports_count_mismatch,
    test_timestamp_compare_enforces_endpoint_tolerance,
]


def main() -> int:
    failures = 0
    for test in _TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
