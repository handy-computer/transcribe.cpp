#!/usr/bin/env python3
"""
bench.py - language ID latency through the Python binding: backend x GGUF x
scored duration, median / p90 of the native mel + encode time over repeated
runs on one warm session (tools/transcribe-bench is ASR-only).

Usage:
  uv run --project scripts/envs/ecapa_tdnn scripts/langid/bench.py \\
      --library build-shared/src/libtranscribe.dylib \\
      --gguf models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-{F32,F16,Q8_0}.gguf \\
      --backends cpu,metal --durations 3,5,10,30 --out reports/langid/bench-m4-max.json

Audio: the first N seconds of samples/long-45s.wav (52 s of FLEURS English),
so every duration scores real speech. The session's max_audio_ms is 60000,
so nothing is cropped away.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", nargs="+", type=Path, required=True)
    p.add_argument("--backends", default="cpu,metal")
    p.add_argument("--durations", default="3,5,10,30")
    p.add_argument("--threads", type=int, default=0, help="0 = library default")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--library", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.library is not None:
        os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python" / "src"))
    import soundfile as sf
    import transcribe_cpp as t

    pcm, sr = sf.read(str(REPO_ROOT / "samples" / "long-45s.wav"), dtype="float32")
    assert sr == 16000
    durations = [float(d) for d in args.durations.split(",")]
    rows = []
    for gguf in args.gguf:
        for backend in args.backends.split(","):
            t0 = time.perf_counter()
            model = t.Model(str(gguf), backend=backend)
            load_ms = (time.perf_counter() - t0) * 1000
            lid = model.langid_session(n_threads=args.threads, max_audio_ms=60000)
            for d in durations:
                clip = np.ascontiguousarray(pcm[: int(d * 16000)])
                for _ in range(args.warmup):
                    lid.run(clip)
                compute, wall = [], []
                for _ in range(args.repeat):
                    w0 = time.perf_counter()
                    lid.run(clip)
                    wall.append((time.perf_counter() - w0) * 1000)
                    tm = lid.timings
                    compute.append(tm.mel_ms + tm.encode_ms)
                compute.sort()
                row = {
                    "gguf": gguf.name, "backend": backend, "bound": model.backend,
                    "duration_s": d, "threads": args.threads,
                    "median_ms": round(statistics.median(compute), 2),
                    "p90_ms": round(compute[int(0.9 * (len(compute) - 1))], 2),
                    "wall_median_ms": round(statistics.median(wall), 2),
                    "load_ms": round(load_ms, 1),
                }
                rows.append(row)
                print(f"{gguf.name:42} {backend:6} {d:5.0f}s  median {row['median_ms']:8.2f} ms  "
                      f"p90 {row['p90_ms']:8.2f}  wall {row['wall_median_ms']:8.2f}", flush=True)
            lid.close()
            model.close()

    if args.out:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "machine": platform.processor() or platform.machine(),
            "engine_sha": commit, "warmup": args.warmup, "repeat": args.repeat,
            "audio": "samples/long-45s.wav (first N s)", "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
