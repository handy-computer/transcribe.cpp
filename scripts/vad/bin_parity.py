#!/usr/bin/env python3
"""Nine-clip native .bin parity against upstream JIT with SAME STORED WEIGHTS.

uv run --project scripts/envs/silero_vad scripts/vad/bin_parity.py \
  --bin /path/ggml-silero-v5.1.2.bin --v5-jit /path/silero-v5.1.2.jit
uv run --project scripts/envs/silero_vad scripts/vad/bin_parity.py \
  --bin /path/ggml-silero-v6.2.0.bin

V5 JIT is fetched from tag v5.1.2 if omitted (SHA256 verified). Neither binaries
nor reports are committed. Existing GGUF validate.py remains unchanged. Uses
the family tensor tolerances (documented v5 cell-state scale exception); exact offline segments and live events,
plus bit-identical native offline/streaming probabilities at 1 and 4 threads.
Offline/live probability policies use the pinned 6.2.3 upstream APIs, regardless
of weight version. They are not the legacy v5 segmentation policy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from silero_vad.utils_vad import VADIterator

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from compare_tensors import compare_pair, load_tolerances, tolerance_for
from dump_reference_silero_vad_author import cmd_encoder
from lib.silero_bin_reference import load_stored_model
from gen_iterator_vectors import ProbabilityModel
from segment_output import parse_segments


def events(probs):
    iterator = VADIterator(ProbabilityModel(probs))
    result = []
    for _ in probs:
        event = iterator(torch.zeros(512))
        if event:
            kind, sample = next(iter(event.items()))
            result.append(f"{kind} {sample}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--v5-jit", type=Path)
    ap.add_argument("--cli", type=Path, default=REPO / "build/bin/transcribe-cli")
    ap.add_argument("--stream-runner", type=Path, default=REPO / "build/bin/transcribe_silero_vad_real_smoke")
    ap.add_argument("--iterator-runner", type=Path, default=REPO / "build/bin/transcribe_vad_iterator_unit")
    ap.add_argument("--out", type=Path, help="Default: build/validate/silero_vad/bin-VERSION")
    args = ap.parse_args()
    args.bin = args.bin.resolve()
    torch.set_num_threads(1)
    _, provenance = load_stored_model(args.bin, args.v5_jit)
    out = args.out or REPO / "build/validate/silero_vad" / ("bin-" + provenance["binary_version"])
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manifest = json.loads((REPO / "tests/golden/silero_vad/silero-vad-v6.2.manifest.json").read_text())
    tolerance_path = REPO / manifest["tolerance_file"]
    if provenance["binary_version"] == "5.1.2":
        tolerance_path = REPO / "tests/tolerances/silero_vad-bin-v5.1.2.json"
    tolerances = load_tolerances(tolerance_path)
    worst = mean_worst = 0.0
    failures = []
    for case in manifest["cases"]:
        name = case["audio"]
        audio = REPO / "samples" / (name + ".wav")
        ref = out / name / "ref"
        cmd_encoder(argparse.Namespace(bin=args.bin, v5_jit=args.v5_jit, torch_threads=1,
                                       audio=audio, out=ref))
        expected_segments = json.loads((ref / "segments.json").read_text())["segments"]
        ref_probs = np.fromfile(ref / "vad.probs.f32", dtype="<f4")
        expected_events = events(ref_probs)
        for threads in (1, 4):
            cpp = out / name / f"cpp-{threads}"
            cpp.mkdir(parents=True, exist_ok=True)
            output = cpp / "segments.txt"
            env = {**os.environ, "TRANSCRIBE_DUMP_DIR": str(cpp)}
            subprocess.run([str(args.cli), "--backend", "cpu", "--threads", str(threads),
                            "-m", str(args.bin), "-o", str(output), str(audio)],
                           env=env, check=True, capture_output=True)
            for tensor in tolerances:
                if tensor.startswith("_"):
                    continue
                result = compare_pair(tensor, cpp, ref)
                max_tol, mean_tol = tolerance_for(tensor, tolerances, 1e-3, 1e-4)
                if result.status != "ok" or result.max_abs > max_tol or result.mean_abs > mean_tol:
                    failures.append(f"{name}/{threads} {tensor}: {result.status} "
                                    f"max={result.max_abs:.3e} mean={result.mean_abs:.3e}")
                if tensor == "vad.probs":
                    worst = max(worst, result.max_abs)
                    mean_worst = max(mean_worst, result.mean_abs)
            if parse_segments(output.read_text()) != expected_segments:
                failures.append(f"{name}/{threads}: offline segments differ")
            native_probs = np.fromfile(cpp / "vad.probs.f32", dtype="<f4")
            stream_path = cpp / "stream.probs.f32"
            # No dump env: only the utility's public-API streaming probabilities.
            stream_env = {k: v for k, v in os.environ.items() if k != "TRANSCRIBE_DUMP_DIR"}
            subprocess.run([str(args.stream_runner), "--stream-probs", str(args.bin), str(audio),
                            str(stream_path), str(threads)], check=True, capture_output=True, env=stream_env)
            streamed = np.fromfile(stream_path, dtype="<f4")
            if not np.array_equal(native_probs, streamed):
                failures.append(f"{name}/{threads}: stream != offline probabilities")
            for probability_path in (cpp / "vad.probs.f32", stream_path):
                actual_events = subprocess.check_output(
                    [str(args.iterator_runner), "--probs", str(probability_path)], text=True).splitlines()
                if actual_events != expected_events:
                    failures.append(f"{name}/{threads}/{probability_path.name}: live events differ")
            print(f"{name}/{threads}: {len(native_probs)} frames, {len(expected_segments)} segments, "
                  f"{len(expected_events)} events")
    print(f"stored-weight {provenance['binary_version']}: 9 clips x 2 thread counts, "
          f"vad.probs max_abs={worst:.3e}, worst mean_abs={mean_worst:.3e}, "
          f"{len(failures)} failures")
    for failure in failures:
        print("FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
