from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare.py")
SPEC = importlib.util.spec_from_file_location("bench_compare", SCRIPT)
assert SPEC and SPEC.loader
bench_compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_compare
SPEC.loader.exec_module(bench_compare)


def report(text: str, token_ids: str) -> dict:
    return {
        "schema": "transcribe-bench-driver-v1",
        "name": "test",
        "git_sha": "deadbee",
        "variant": "gigaam-v3-e2e-rnnt",
        "backend": "metal",
        "runs": [{
            "model_path": "models/gigaam-v3-e2e-rnnt-Q4_K_M.gguf",
            "sample_path": "samples/ru.wav",
            "hyp_text": text,
            "token_ids_csv": token_ids,
            "summary": {"wall_ms": {"mean": 100.0}},
        }],
    }


class CompareOutputIdentityTest(unittest.TestCase):
    def run_compare(self, baseline: dict, candidate: dict, *extra: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_path.write_text(json.dumps(baseline))
            candidate_path.write_text(json.dumps(candidate))
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--baseline", str(baseline_path),
                 "--candidate", str(candidate_path), *extra],
                text=True, capture_output=True, check=False,
            )

    def test_identical_utf8_outputs_pass(self) -> None:
        result = self.run_compare(report("Привет, мир?", "1,2,3"), report("Привет, мир?", "1,2,3"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("outputs identical", result.stdout)

    def test_transcript_byte_change_fails(self) -> None:
        result = self.run_compare(report("Привет, мир?", "1,2,3"), report("привет мир", "1,2,3"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("OUTPUT_CHANGED", result.stdout)
        self.assertIn("transcript", result.stdout)

    def test_token_change_fails_even_when_text_matches(self) -> None:
        result = self.run_compare(report("same", "1,2,3"), report("same", "1,9,3"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("token_ids", result.stdout)

    def test_output_change_can_be_explicitly_allowed(self) -> None:
        result = self.run_compare(report("before", "1"), report("after", "2"), "--allow-output-change")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
