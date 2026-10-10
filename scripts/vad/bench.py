#!/usr/bin/env python3
"""
bench.py - this machine's VAD publication speed cells
(catalog/_benchmark_profiles.json) through the Python binding:
tools/transcribe-bench is ASR-only.

  uv run --project scripts/envs/silero_vad scripts/vad/bench.py \\
      --library build-shared/src/libtranscribe.dylib
  uv run scripts/catalog/ingest_perf.py

Two kinds of cell, both native encode + decode time (total_ms):
  <clip>       transcribe_vad_run over the whole of samples/<clip>.wav
  <clip>-<N>ms one pass of transcribe_vad_stream_feed over the clip in N ms
               chunks; the time per feed call, averaged over the pass
Writes one bench-driver report (scripts/bench/run.py's shape) per backend to
reports/perf/<machine-slug>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAM_RE = re.compile(r"^(?P<clip>.+)-(?P<ms>\d+)ms$")


def bench_driver():
    """scripts/bench/run.py, for its profiles and machine detection."""
    spec = importlib.util.spec_from_file_location(
        "bench_driver", REPO_ROOT / "scripts" / "bench" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="silero-vad-v6.2")
    p.add_argument("--library", type=Path, help="a shared libtranscribe (TRANSCRIBE_LIBRARY)")
    args = p.parse_args(argv)

    driver = bench_driver()
    profiles = driver.benchmark_profiles
    machine = driver.detect_machine()
    record = driver.catalog_common.load_record(args.variant)
    profile_id, profile = profiles.profile_for(record)
    target = profiles.target_for_machine(profile, machine["slug"])
    if target is None:
        p.error(f"machine {machine['slug']!r} is not a target in profile {profile_id}")
    cells = [cell for cell in profiles.expected_speed(record, profile)
             if cell["machine"] == target["machine"]]
    quants = {cell["quant"] for cell in cells}
    ggufs = [REPO_ROOT / "models" / args.variant / item["filename"]
             for item in record["downloads"] if item["quant"] in quants]
    backends = list(dict.fromkeys(cell["backend"] for cell in cells))
    samples = list(dict.fromkeys(cell["sample"] for cell in cells))
    warmup, repeat = int(profile["speed"]["warmup"]), int(profile["speed"]["iterations"])

    if args.library is not None:
        os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python" / "src"))
    import numpy as np
    import soundfile as sf
    import transcribe_cpp as t

    timestamp = driver.now_utc_iso()
    commit = t.native_commit()
    name = f"{args.variant}-publication"
    for backend in backends:
        runs = []
        for gguf in ggufs:
            t0 = time.perf_counter()
            model = t.Model(str(gguf), backend=backend)
            load_ms = (time.perf_counter() - t0) * 1000
            vad = model.vad_session(n_threads=1)
            for sample in samples:
                stream = STREAM_RE.match(sample)
                clip = stream["clip"] if stream else sample
                sample_path = REPO_ROOT / "samples" / f"{clip}.wav"
                pcm, sr = sf.read(str(sample_path), dtype="float32")
                assert sr == 16000 and pcm.ndim == 1
                audio = np.ascontiguousarray(pcm)
                if stream:
                    feed = int(stream["ms"]) * 16
                    chunks = [audio[i:i + feed] for i in range(0, len(audio) - feed + 1, feed)]
                    duration_s = feed / 16000

                    def once(chunks=chunks):
                        # Mean per-feed time (ms) over one pass of the clip.
                        enc = dec = wall = 0.0
                        vad.reset()
                        for chunk in chunks:
                            w0 = time.perf_counter()
                            vad.feed(chunk)
                            wall += time.perf_counter() - w0
                            tm = vad.timings
                            enc += tm.encode_ms
                            dec += tm.decode_ms
                        n = len(chunks)
                        return enc / n, dec / n, wall * 1000 / n
                else:
                    duration_s = len(audio) / 16000

                    def once(audio=audio):
                        w0 = time.perf_counter()
                        vad.run(audio)
                        wall = (time.perf_counter() - w0) * 1000
                        tm = vad.timings
                        return tm.encode_ms, tm.decode_ms, wall
                for _ in range(warmup):
                    once()
                per_iter = []
                for _ in range(repeat):
                    enc, dec, wall = once()
                    per_iter.append({"encode_ms": enc, "decode_ms": dec,
                                     "total_ms": enc + dec, "wall_ms": wall})
                runs.append({
                    "model_path": str(gguf.relative_to(REPO_ROOT)),
                    "sample": sample,
                    "sample_path": str(sample_path.relative_to(REPO_ROOT)),
                    "sample_duration_s": duration_s,
                    "backend": model.backend,
                    "threads": 1,
                    "load_ms": round(load_ms, 1),
                    "per_iter": per_iter,
                    "summary": {field: {"mean": statistics.fmean(it[field] for it in per_iter)}
                                for field in ("encode_ms", "decode_ms", "total_ms", "wall_ms")},
                })
                print(f"{gguf.name:32} {backend:4} {sample:10} mean "
                      f"{runs[-1]['summary']['total_ms']['mean']:9.4f} ms", flush=True)
            vad.close()
            model.close()

        out = REPO_ROOT / "reports" / "perf" / machine["slug"] / \
            f"{driver.slugify(name)}_{args.variant}_{backend}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "transcribe-bench-driver-v1",
            "timestamp": timestamp,
            "name": name,
            "publication_profile": profile_id,
            "machine": machine,
            "git_sha": commit if commit != "unknown" else driver.get_git_sha(REPO_ROOT),
            "variant": args.variant,
            "backend": backend,
            "iters": repeat,
            "warmup": warmup,
            "tool": "scripts/vad/bench.py",
            "runs": runs,
        }, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
