#!/usr/bin/env python3
"""
bench.py - language ID latency through the Python binding: backend x GGUF x
scored duration, the native mel + encode time over repeated runs on one warm
session (tools/transcribe-bench is ASR-only).

Usage:
  # the publication cells for this machine (catalog/_benchmark_profiles.json)
  uv run --project scripts/envs/ecapa_tdnn scripts/langid/bench.py --profile \\
      --library build-shared/src/libtranscribe.dylib
  uv run scripts/catalog/ingest_perf.py

  # an experiment
  uv run --project scripts/envs/ecapa_tdnn scripts/langid/bench.py \\
      --library build-shared/src/libtranscribe.dylib \\
      --gguf models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-{F32,F16,Q8_0}.gguf \\
      --backends cpu,metal --durations 3,5,10,30

Audio: the first N seconds of samples/<clip>.wav (default ru-long, 33.8 s of
Russian), so every duration scores real speech; the cell's sample is named
`<clip>-<N>s`. The session's max_audio_ms is 60000, so nothing is cropped
away.

Writes one report per backend to reports/perf/<machine-slug>/
<name>_<variant>_<backend>.json in the bench driver's shape
(scripts/bench/run.py), which scripts/catalog/ingest_perf.py folds into the
catalog. As there, only a --profile run is eligible: it takes its GGUFs,
samples, backends, iterations, warmup and thermal gate from the profile.
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
    """scripts/bench/run.py, for its machine detection and thermal gate.
    Loaded by path: this directory has a run.py of its own."""
    spec = importlib.util.spec_from_file_location(
        "bench_driver", REPO_ROOT / "scripts" / "bench" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def seconds_label(seconds: float) -> str:
    return f"{seconds:g}s"


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p90": ordered[int(0.9 * (len(ordered) - 1))]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", nargs="?", const="", default=None,
                   help="run this machine's publication cells (optionally name "
                        "the profile; default: the variant's role profile)")
    p.add_argument("--variant", default="lang-id-voxlingua107-ecapa",
                   help="catalog variant (default %(default)s)")
    p.add_argument("--gguf", nargs="+", type=Path, default=None,
                   help="default with --profile: the variant's downloads under models/")
    p.add_argument("--backends", default=None, help="default cpu,metal")
    p.add_argument("--sample", default=None, help="clip under samples/ (default ru-long)")
    p.add_argument("--durations", default=None, help="default 3,5,10,30")
    p.add_argument("--threads", type=int, default=0, help="0 = library default")
    p.add_argument("--warmup", type=int, default=None, help="default 3")
    p.add_argument("--repeat", type=int, default=None, help="default 20")
    p.add_argument("--library", type=Path, default=None)
    p.add_argument("--name", default=None,
                   help="report filename prefix (default: a timestamp, or "
                        "<variant>-publication with --profile)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "perf")
    args = p.parse_args(argv)

    driver = bench_driver()
    profiles = driver.benchmark_profiles
    machine = driver.detect_machine()
    record = driver.catalog_common.load_record(args.variant)
    profile_id, cooldown_c = None, 0.0
    if args.profile is not None:
        conflicting = [flag for flag, value in (
            ("--gguf", args.gguf), ("--backends", args.backends), ("--sample", args.sample),
            ("--durations", args.durations), ("--warmup", args.warmup),
            ("--repeat", args.repeat)) if value is not None]
        if conflicting:
            p.error(f"--profile supplies {', '.join(conflicting)}; do not override it")
        profile_id, profile = profiles.profile_for(record, args.profile or None)
        target = profiles.target_for_machine(profile, machine["slug"])
        if target is None:
            p.error(f"machine {machine['slug']!r} is not a target in profile {profile_id}")
        cells = [cell for cell in profiles.apply_exceptions(
                     record, "speed", profiles.expected_speed(record, profile))
                 if cell["machine"] == target["machine"]]
        quants = {cell["quant"] for cell in cells}
        args.gguf = [REPO_ROOT / "models" / args.variant / item["filename"]
                     for item in record["downloads"] if item["quant"] in quants]
        args.backends = ",".join(dict.fromkeys(cell["backend"] for cell in cells))
        samples = [SAMPLE_RE.match(name) for name in
                   dict.fromkeys(cell["sample"] for cell in cells)]
        if not all(samples) or len({m["clip"] for m in samples}) != 1:
            p.error(f"profile samples must be <clip>-<N>s on one clip: "
                    f"{sorted({cell['sample'] for cell in cells})}")
        args.sample = samples[0]["clip"]
        args.durations = ",".join(m["seconds"] for m in samples)
        args.warmup = int(profile["speed"]["warmup"])
        args.repeat = int(profile["speed"]["iterations"])
        cooldown_c = float(target.get("cooldown_tctl_c", 0.0))
    elif not args.gguf:
        p.error("--gguf is required without --profile")
    backends = (args.backends or "cpu,metal").split(",")
    clip = args.sample or "ru-long"
    durations = [float(d) for d in (args.durations or "3,5,10,30").split(",")]
    warmup = 3 if args.warmup is None else args.warmup
    repeat = 20 if args.repeat is None else args.repeat
    missing = [str(path) for path in args.gguf if not path.exists()]
    if missing:
        p.error(f"no such GGUF: {', '.join(missing)}")

    if args.library is not None:
        os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python" / "src"))
    import soundfile as sf
    import transcribe_cpp as t

    sample_path = REPO_ROOT / "samples" / f"{clip}.wav"
    pcm, sr = sf.read(str(sample_path), dtype="float32")
    assert sr == 16000
    too_long = [d for d in durations if d * 16000 > pcm.size]
    if too_long:
        p.error(f"{sample_path.name} is {pcm.size / 16000:.1f} s; cannot score {too_long} s")

    timestamp = driver.now_utc_iso()
    commit = t.native_commit()
    name = args.name or (f"{args.variant}-publication" if profile_id
                         else driver.timestamp_for_filename(timestamp))
    for backend in backends:
        runs = []
        for gguf in args.gguf:
            t0 = time.perf_counter()
            model = t.Model(str(gguf), backend=backend)
            load_ms = (time.perf_counter() - t0) * 1000
            lid = model.langid_session(n_threads=args.threads, max_audio_ms=60000)
            for d in durations:
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
                try:
                    model_path = str(gguf.resolve().relative_to(REPO_ROOT))
                except ValueError:
                    model_path = str(gguf)
                run = {
                    "model_path": model_path,
                    "sample": f"{clip}-{seconds_label(d)}",
                    "sample_path": str(sample_path.relative_to(REPO_ROOT)),
                    "sample_duration_s": d,
                    "backend": model.backend,
                    "threads": args.threads,
                    "load_ms": round(load_ms, 1),
                    "per_iter": per_iter,
                    "summary": {field: summary([it[field] for it in per_iter])
                                for field in ("mel_ms", "encode_ms", "total_ms", "wall_ms")},
                }
                runs.append(run)
                total = run["summary"]["total_ms"]
                print(f"{gguf.name:42} {backend:6} {d:5.0f}s  mean {total['mean']:8.2f} ms  "
                      f"median {total['median']:8.2f}  p90 {total['p90']:8.2f}", flush=True)
            lid.close()
            model.close()

        out = args.out_dir / machine["slug"] / f"{driver.slugify(name)}_{args.variant}_{backend}.json"
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
