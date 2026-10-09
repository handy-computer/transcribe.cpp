#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""VAD segment-gate regressions: uv run scripts/test_validate_vad.py."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import validate
from vad.segment_output import parse_segments


class VadCliOutputTests(unittest.TestCase):
    def test_blank_no_speech(self):
        for text in ("", "\n", " \n\n"):
            self.assertEqual(parse_segments(text), [])

    def test_normal_segments(self):
        self.assertEqual(parse_segments(
            "\nsegment: 0 start=100 end=1000 t0=0.006 t1=0.062\n"
            "segment: 1 start=2000 end=3000 t0=0.125 t1=0.188\n\n"),
            [{"start": 100, "end": 1000}, {"start": 2000, "end": 3000}])

    def test_unexpected_nonblank_is_error(self):
        for text in ("noise", "segment: 0", "segment: 1 start=0 end=1 t0=0 t1=1",
                     "segment: 0 start=3 end=2 t0=0 t1=1"):
            with self.assertRaises(ValueError):
                parse_segments(text)


class VadSegmentGateTests(unittest.TestCase):
    def compare(self, family, role, reference, cpp):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            manifest = {"variant": "test", "cases": ["clip"]}
            if role is not None:
                manifest["role"] = role
            base = repo / "build" / "validate" / family / "test" / "clip"
            for name, segments in (("ref", reference), ("cpp", cpp)):
                directory = base / name
                directory.mkdir(parents=True)
                if segments is not None:
                    (directory / "segments.json").write_text(
                        json.dumps({"segments": segments})
                    )
            args = argparse.Namespace(family=family, variant=None, report=True)
            with patch.object(validate, "find_repo_root", return_value=repo), \
                 patch.object(validate, "load_manifest", return_value=manifest), \
                 patch.object(validate.subprocess, "run") as run, \
                 patch.object(validate, "write_report_bundle") as report, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.return_value.returncode = 0
                run.return_value.stdout = ""
                run.return_value.stderr = ""
                status = validate.cmd_compare(args)
                return status, report.call_args.kwargs["transcript_results"]

    def test_vad_requires_both_artifacts(self):
        for family, role in (("silero_vad", None), ("future_vad", "vad")):
            for reference, cpp in ((None, None), (None, []), ([], None)):
                with self.subTest(family=family, reference=reference, cpp=cpp):
                    status, results = self.compare(family, role, reference, cpp)
                    self.assertEqual(status, 1)
                    self.assertEqual(results[0]["mode"], "segments")
                    self.assertFalse(results[0]["match"])

    def test_empty_segment_lists_are_valid(self):
        status, results = self.compare("silero_vad", "vad", [], [])
        self.assertEqual(status, 0)
        self.assertTrue(results[0]["match"])

    def test_exact_sample_boundaries_required(self):
        reference = [{"start": 100, "end": 1000}]
        for cpp, expected in ((reference, 0), ([{"start": 101, "end": 1000}], 1)):
            with self.subTest(cpp=cpp):
                status, _ = self.compare("silero_vad", "vad", reference, cpp)
                self.assertEqual(status, expected)

    def test_non_vad_without_segments_is_unchanged(self):
        status, results = self.compare("asr_family", "asr", None, None)
        self.assertEqual(status, 0)
        self.assertEqual(results, [])

    def test_reference_segments_still_opt_in_other_families(self):
        status, results = self.compare("other", None, [], None)
        self.assertEqual(status, 1)
        self.assertFalse(results[0]["match"])


if __name__ == "__main__":
    unittest.main()
