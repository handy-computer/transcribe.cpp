#!/usr/bin/env python3
"""Cheap VAD measurement/report regression tests; no corpus inference.

    uv run --project scripts/envs/silero_vad --no-sync scripts/vad/test_measurements.py
"""
from __future__ import annotations

import contextlib
import copy
import ctypes
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load("vad_measure_bench", "scripts/vad/bench.py")
parity = load("vad_measure_parity", "scripts/vad/parity.py")
perf = load("vad_measure_ingest_perf", "scripts/catalog/ingest_perf.py")
ingest = load("vad_measure_ingest_parity", "scripts/catalog/ingest_parity.py")


def record():
    data = ingest.common.load_record("silero-vad-v6.2")
    data["reference_parity"] = []
    data["speed_benchmarks"] = []
    return data


def report():
    source = record()["source_artifact"]
    golden = REPO / "tests/golden/silero_vad/silero-vad-v6.2.manifest.json"
    files = [{"audio": parity.path_label(path), "identical": True,
              "n_segments": 2, "n_native_segments": 2, "audio_duration_s": 10.0,
              "n_probs": 313, "max_abs_prob_delta": 0.000001}
             for path in parity.expand([str(golden)])]
    return {
        "schema": ingest.SCHEMA, "tool": "scripts/vad/parity.py",
        "variant": "silero-vad-v6.2", "source_artifact": source,
        "vad_info": {"sample_rate": 16000, "frame_samples": 512},
        "publication_profile": "vad-publication-v1", "reference": f"{source['package']}=={source['version']}",
        "backend": "cpu", "segmentation_params": "defaults", "reference_threads": 1,
        "threads": 1, "quant": "F32", "model_path": "models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf",
        "library": "build-vad-review-shared/src/libtranscribe.dylib", "engine_sha": "24fda783",
        "measured_on": "2026-08-19", "timestamp": "2026-08-19T12:00:00+00:00",
        "dataset": "golden", "split": "validation", "language": "mul",
        "n_files": len(files), "n_identical": len(files), "n_segments": 2 * len(files),
        "audio_duration_s": 10.0 * len(files), "max_abs_prob_delta": 0.000001, "files": files,
    }


class FeedTests(unittest.TestCase):
    def test_summary_definition_and_precision(self):
        result = bench.summarize([index / 1000 for index in range(1, 21)])
        self.assertEqual(result, {"mean": 0.0105, "median": 0.0105, "p95": 0.019})
        self.assertEqual(bench.summarize([0.01728349])["mean"], 0.017283)
        with self.assertRaises(ValueError):
            bench.summarize([])

    def test_exact_feeds_and_prepared_pointers(self):
        import numpy as np

        pcm = np.arange(10003, dtype=np.float32)
        for size in (512, 2048, 8192):
            owner, pointers = bench.prepare_feeds(pcm, size)
            self.assertEqual(len(pointers), len(pcm) // size)
            self.assertTrue(owner.flags.c_contiguous)
            self.assertEqual(owner.dtype, np.float32)
            for index, pointer in enumerate(pointers):
                self.assertEqual(pointer[0], pcm[index * size])
                self.assertEqual(pointer[size - 1], pcm[(index + 1) * size - 1])
            self.assertNotEqual(owner.ctypes.data, pcm.ctypes.data)

    def test_timing_excludes_reset_and_result_copy_and_preserves_state(self):
        events = []
        pointers = [object(), object(), object()]
        handle = object()

        def feed(actual_handle, pointer, size):
            self.assertIs(actual_handle, handle)
            self.assertEqual(size, 2048)
            events.append(("feed", pointer))
            return 0

        def reset(actual_handle):
            self.assertIs(actual_handle, handle)
            events.append(("reset", None))

        def clock():
            events.append(("clock", None))
            return sum(event[0] == "clock" for event in events) * 1000

        with mock.patch.object(bench.time, "perf_counter_ns", side_effect=clock):
            timings = bench.measure_feeds(feed, handle, pointers, 2048, 32, reset)
        self.assertEqual(len(timings), 3)
        self.assertEqual(timings, [0.001] * 3)
        self.assertEqual([event[0] for event in events[:33]], ["feed"] * 32 + ["reset"])
        self.assertEqual([event[0] for event in events[33:]], ["clock", "feed", "clock"] * 3)
        self.assertEqual([event[1] for event in events[33:] if event[0] == "feed"], pointers)

    def test_native_error_aborts_measurement(self):
        with self.assertRaisesRegex(RuntimeError, "status 3"):
            bench.measure_feeds(lambda *args: 3, None, [None], 512, 0, lambda handle: None)


class ParityTests(unittest.TestCase):
    def test_probability_delta_all_frames_and_invalid_values(self):
        self.assertAlmostEqual(parity.probability_delta([0.1, 0.2], [0.1, 0.203]), 0.003)
        self.assertEqual(parity.probability_delta([], []), 0)
        for reference, native in (([0.1], []), ([float("nan")], [0]), ([0], [2])):
            with self.assertRaises(ValueError):
                parity.probability_delta(reference, native)

    def test_jsonl_and_golden_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "input.jsonl"
            manifest.write_text(json.dumps({"audio": "samples/jfk.wav"}) + "\n")
            self.assertEqual(parity.expand([str(manifest)]), [REPO / "samples/jfk.wav"])
            golden = root / "test.manifest.json"
            golden.write_text(json.dumps({"schema": "transcribe-golden-manifest-v1",
                                          "cases": [{"audio": "jfk"}, {"audio": "samples/noise.wav"}]}))
            self.assertEqual(parity.expand([str(golden)]), [REPO / "samples/jfk.wav", REPO / "samples/noise.wav"])

    def test_source_metadata_catalog_and_legacy_fallback(self):
        source, info = parity.measurement_metadata("silero-vad-v6.2", require_record=True)
        self.assertEqual(source, record()["source_artifact"])
        self.assertEqual(info, record()["vad_info"])
        with mock.patch.object(Path, "exists", return_value=False):
            legacy_source, legacy_info = parity.measurement_metadata("silero-vad-v6.2")
            self.assertEqual(legacy_source["sha256"], source["sha256"])
            self.assertEqual(legacy_source["version"], source["version"])
            self.assertEqual(legacy_info, info)
            with self.assertRaisesRegex(ValueError, "require a catalog record"):
                parity.measurement_metadata("silero-vad-v6.2", require_record=True)

    def test_publication_requires_whole_corpus(self):
        golden = REPO / "tests/golden/silero_vad/silero-vad-v6.2.manifest.json"
        files = parity.expand([str(golden)])
        parity.validate_corpus_files(files, "golden", "validation", "mul", "silero-vad-v6.2")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parity.validate_corpus_files(files[:1], "golden", "validation", "mul", "silero-vad-v6.2")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parity.validate_corpus_files(files + files[:1], "golden", "validation", "mul", "silero-vad-v6.2")

    def test_legacy_cli_output_parser(self):
        native_output = "noise\nsegment: 0 start=12 end=34\nsegment: 1 start=99 end=123\n"
        with mock.patch.object(parity.subprocess, "run", return_value=mock.Mock(stdout=native_output)) as run:
            result = parity.cpp_segments(Path("cli"), Path("model.gguf"), Path("audio.wav"), 4)
        self.assertEqual(result, [{"start": 12, "end": 34}, {"start": 99, "end": 123}])
        self.assertIn("--threads", run.call_args.args[0])

    def test_report_requires_native_library(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            parity.main(["--gguf", "test-F32.gguf", "--out", "unused.json", "unused.wav"])
        self.assertEqual(exc.exception.code, 2)


class IngestTests(unittest.TestCase):
    def test_streaming_precision_and_optional_fields(self):
        run = {"model_path": "models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf",
               "sample": "love-loss-32ms", "sample_duration_s": 0.032,
               "feed_samples": 512, "frame_samples": 512, "threads": 1,
               "n_calls": 6161, "warmup_calls": 32,
               "summary": {"total_ms": {"mean": 0.01728349, "median": 0.016721, "p95": 0.022987}}}
        source = {"variant": "silero-vad-v6.2", "publication_profile": "vad-publication-v1",
                  "machine": {"slug": "apple-m4"}, "backend": "cpu", "git_sha": "24fda783",
                  "timestamp": "2026-08-19T12:00:00+00:00", "_file": "report.json", "runs": [run]}
        row = perf.catalog_row(perf.cells(source)[0])
        self.assertEqual(row["total_ms"], 0.017283)
        self.assertEqual(row["median_ms"], 0.016721)
        self.assertEqual(row["p95_ms"], 0.022987)
        self.assertEqual(row["feed_samples"], 512)
        self.assertEqual(row["n_calls"], 6161)
        self.assertEqual(row["machine"], "m4")
        self.assertAlmostEqual(row["xrt_compute"], 32 / row["total_ms"], delta=0.005)
        for field in perf.STREAM_FIELDS:
            run.pop(field)
        run["summary"] = {"total_ms": {"mean": 123.456}}
        row = perf.catalog_row(perf.cells(source)[0])
        self.assertEqual(row["total_ms"], 123.5)
        self.assertNotIn("n_calls", row)
        self.assertNotIn("median_ms", row)

    def test_perf_unreadable_relative_and_external_report_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            machine = root / "machine"
            machine.mkdir()
            (machine / "bad.json").write_text("{broken json")
            (machine / "legacy.json").write_text(json.dumps({"schema": "legacy"}))
            best, notes = perf.collect(root)
            self.assertEqual(best, {})
            self.assertEqual(len(notes), 2)
            import os

            relative = Path(os.path.relpath(root, Path.cwd()))
            best, notes = perf.collect(relative)
            self.assertEqual(best, {})
            self.assertEqual(len(notes), 2)

    def test_perf_models_filter_does_not_write_other_records(self):
        selected, other = record(), record()
        other["variant"] = "other"
        with tempfile.TemporaryDirectory() as directory:
            cell = {"variant": "silero-vad-v6.2", "machine": "m4", "backend": "cpu", "quant": "F32",
                    "sample": "love-loss-32ms", "_profile": "vad-publication-v1",
                    **{field: None for field in perf.FIELDS}}
            cell.update({"sample_duration_s": 0.032, "total_ms": 0.2, "xrt_compute": 160,
                         "engine_sha": "24fda783", "publication_profile": "vad-publication-v1"})
            measured = {("silero-vad-v6.2", "m4", "cpu", "F32", "love-loss-32ms"): cell}
            with mock.patch.object(sys, "argv", ["ingest_perf.py", "--reports", directory,
                                                  "--models", "silero-vad-v6.2"]), \
                    mock.patch.object(perf.common, "load_records", return_value={"silero-vad-v6.2": selected,
                                                                                "other": other}), \
                    mock.patch.object(perf, "collect", return_value=(measured, [])), \
                    mock.patch.object(perf.common, "write_record") as write, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(perf.main(), 0)
            self.assertEqual(write.call_count, 1)
            self.assertEqual(write.call_args.args[0].stem, "silero-vad-v6.2")
            self.assertEqual(other["speed_benchmarks"], [])

    def test_parity_typed_row_separate_from_accuracy(self):
        original = record()
        before = copy.deepcopy(original)
        row = ingest.row_from_report(original, report())
        self.assertEqual(set(row), set(ingest.ROW_FIELDS))
        self.assertEqual(row["n_files"], len(report()["files"]))
        self.assertEqual(original, before)
        self.assertNotIn("err_pct", row)
        self.assertNotIn("metric", row)

    def test_reject_reference_profile_source_and_diff(self):
        invalid = {"reference": "silero-vad==6.2.2", "publication_profile": "asr-publication-v2",
                   "engine_sha": "unknown", "n_identical": 0, "backend": "metal",
                   "segmentation_params": "custom", "quant": "Q8_0", "language": "en",
                   "source_artifact": {}, "max_abs_prob_delta": 0.1, "n_files": True,
                   "measured_on": "yesterday", "audio_duration_s": float("nan"),
                   "model_path": "wrong-F32.gguf", "vad_info": {"sample_rate": 8000}}
        for field, value in invalid.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                source = report()
                source[field] = value
                ingest.row_from_report(record(), source)

    def test_reject_inconsistent_probability_aggregates(self):
        for field, value in (("n_probs", 312), ("identical", False),
                             ("max_abs_prob_delta", 0.000002), ("n_segments", 3)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                source = report()
                source["files"][0][field] = value
                ingest.row_from_report(record(), source)

    def test_partial_corpus_cannot_be_imported_under_a_profile_stamp(self):
        source = report()
        source["files"] = source["files"][:1]
        source.update({"n_files": 1, "n_identical": 1, "n_segments": 2, "audio_duration_s": 10.0})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ingest.row_from_report(record(), source)

    def test_rejection_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            good, bad = Path(directory) / "good.json", Path(directory) / "bad.json"
            good.write_text(json.dumps(report()))
            source = report()
            source["n_identical"] = 0
            bad.write_text(json.dumps(source))
            with mock.patch.object(ingest.common, "load_records", return_value={"silero-vad-v6.2": record()}), \
                    mock.patch.object(ingest.common, "write_record") as write, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ingest.main([str(good), str(bad), "--models", "silero-vad-v6.2"]), 1)
            write.assert_not_called()

    def test_newest_report_wins_and_accuracy_untouched(self):
        data = record()
        data["accuracy_benchmarks"] = [{"sentinel": "ground truth, not parity"}]
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old.json", Path(directory) / "new.json"
            old.write_text(json.dumps(report()))
            source = report()
            source["timestamp"] = "2026-08-19T13:00:00+00:00"
            source["engine_sha"] = "574bb8e4"
            new.write_text(json.dumps(source))
            with mock.patch.object(ingest.common, "load_records", return_value={"silero-vad-v6.2": data}), \
                    mock.patch.object(ingest.common, "write_record") as write, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ingest.main([str(new), str(old), "--models", "silero-vad-v6.2"]), 0)
            self.assertEqual(write.call_count, 1)
            self.assertEqual(data["reference_parity"][0]["engine_sha"], "574bb8e4")
            self.assertEqual(data["accuracy_benchmarks"], [{"sentinel": "ground truth, not parity"}])


if __name__ == "__main__":
    unittest.main()
