#!/usr/bin/env python3
"""
bench.py - this machine's language ID publication speed cells
(catalog/_benchmark_profiles.json) through the Python binding:
tools/transcribe-bench is ASR-only.

  uv run --project scripts/envs/ecapa_tdnn scripts/langid/bench.py --profile \\
      --library build-shared/src/libtranscribe.dylib
  uv run scripts/catalog/ingest_perf.py

The profile names the GGUFs, backends, samples, iterations, warmup and
thermal gate. A sample `<clip>-<N>s` is the first N seconds of
samples/<clip>.wav; the cell's time is the native mel + encode. Writes one
bench-driver report (scripts/bench/run.py's shape) per backend to
reports/perf/<machine-slug>/<variant>-publication_<variant>_<backend>.json.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RE = re.compile(r"^(?P<clip>.+)-(?P<seconds>\d+(?:\.\d+)?)s$")


def bench_driver():
    """scripts/bench/run.py, for its profiles, machine detection and thermal
    gate. Loaded by path: this directory has a run.py of its own."""
    spec = importlib.util.spec_from_file_location(
        "bench_driver", REPO_ROOT / "scripts" / "bench" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", nargs="?", const="", default="",
                   help="profile id (default: the variant's role profile)")
    p.add_argument("--variant", default="lang-id-voxlingua107-ecapa")
    p.add_argument("--library", type=Path, help="a shared libtranscribe (TRANSCRIBE_LIBRARY)")
    args = p.parse_args(argv)

    driver = bench_driver()
    profiles = driver.benchmark_profiles
    machine = driver.detect_machine()
    record = driver.catalog_common.load_record(args.variant)
    profile_id, profile = profiles.profile_for(record, args.profile or None)
    target = profiles.target_for_machine(profile, machine["slug"])
    if target is None:
        p.error(f"machine {machine['slug']!r} is not a target in profile {profile_id}")
    cells = [cell for cell in profiles.apply_exceptions(
                 record, "speed", profiles.expected_speed(record, profile))
             if cell["machine"] == target["machine"]]
    quants = {cell["quant"] for cell in cells}
    ggufs = [REPO_ROOT / "models" / args.variant / item["filename"]
             for item in record["downloads"] if item["quant"] in quants]
    backends = list(dict.fromkeys(cell["backend"] for cell in cells))
    samples = [SAMPLE_RE.match(name) for name in dict.fromkeys(cell["sample"] for cell in cells)]
    if not all(samples) or len({m["clip"] for m in samples}) != 1:
        p.error(f"profile samples must be <clip>-<N>s on one clip: "
                f"{sorted({cell['sample'] for cell in cells})}")
    clip = samples[0]["clip"]
    durations = [m["seconds"] for m in samples]
    warmup, repeat = int(profile["speed"]["warmup"]), int(profile["speed"]["iterations"])
    cooldown_c = float(target.get("cooldown_tctl_c", 0.0))

    if args.library is not None:
        os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python" / "src"))
    import soundfile as sf
    import transcribe_cpp as t

    sample_path = REPO_ROOT / "samples" / f"{clip}.wav"
    pcm, sr = sf.read(str(sample_path), dtype="float32")
    assert sr == 16000 and pcm.size >= float(max(durations, key=float)) * 16000

    timestamp = driver.now_utc_iso()
    commit = t.native_commit()
    name = f"{args.variant}-publication"
    for backend in backends:
        runs = []
        for gguf in ggufs:
            t0 = time.perf_counter()
            model = t.Model(str(gguf), backend=backend)
            load_ms = (time.perf_counter() - t0) * 1000
            lid = model.langid_session(n_threads=0)
            for seconds in durations:
                d = float(seconds)
                driver.cooldown_wait(cooldown_c, 300.0, 10.0)
                audio = np.ascontiguousarray(pcm[: int(d * 16000)])
                for _ in range(warmup):
                    lid.run(audio)
                per_iter = []
                for _ in range(repeat):
                    w0 = time.perf_counter()
                    lid.run(audio)
                    wall = (time.perf_counter() - w0) * 1000
                    tm = lid.timings
                    per_iter.append({"mel_ms": tm.mel_ms, "encode_ms": tm.encode_ms,
                                     "total_ms": tm.mel_ms + tm.encode_ms, "wall_ms": wall})
                runs.append({
                    "model_path": str(gguf.relative_to(REPO_ROOT)),
                    "sample": f"{clip}-{seconds}s",
                    "sample_path": str(sample_path.relative_to(REPO_ROOT)),
                    "sample_duration_s": d,
                    "backend": model.backend,
                    "threads": 0,
                    "load_ms": round(load_ms, 1),
                    "per_iter": per_iter,
                    "summary": {field: {"mean": statistics.fmean(it[field] for it in per_iter)}
                                for field in ("mel_ms", "encode_ms", "total_ms", "wall_ms")},
                })
                print(f"{gguf.name:42} {backend:6} {d:5.0f}s  mean "
                      f"{runs[-1]['summary']['total_ms']['mean']:8.2f} ms", flush=True)
            lid.close()
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
            # The build the native library reports, which is what ran.
            "git_sha": commit if commit != "unknown" else driver.get_git_sha(REPO_ROOT),
            "variant": args.variant,
            "backend": backend,
            "iters": repeat,
            "warmup": warmup,
            "tool": "scripts/langid/bench.py",
            "runs": runs,
        }, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
