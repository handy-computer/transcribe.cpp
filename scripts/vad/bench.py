#!/usr/bin/env python3
"""Measure the VAD publication profile with bare native stream_feed calls.

    uv run --project scripts/envs/silero_vad scripts/vad/bench.py --profile \
        --library build-vad-review-shared/src/libtranscribe.dylib

Each cell is one full love-loss clip pass, in exact 512/2048/8192-sample
feeds (32/128/512 ms). Only full feeds are timed; the partial tail is
omitted as specified by the profile. Warmup uses 32 feeds, then the stream
is reset outside timing. Recurrent state is
preserved throughout the measured pass. PCM, pointers and handles are
prepared beforehand; no loading, reset, result capture or tuple copying is
timed. total_ms is mean wall latency per native feed, not full-clip time.
p50 is the ordinary median; p95 uses nearest rank (ceil(0.95 * n)).
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAMPLE_RE = re.compile(r"^(?P<clip>.+)-(?P<ms>32|128|512)ms$")


def bench_driver():
    spec = importlib.util.spec_from_file_location("vad_bench_driver", REPO / "scripts/bench/run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        raise ValueError("a feed measurement must contain at least one call")
    ordered = sorted(latencies_ms)
    return {"mean": round(statistics.fmean(ordered), 6),
            "median": round(statistics.median(ordered), 6),
            "p95": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 6)}


def prepare_feeds(pcm, feed_samples: int):
    """Return owning contiguous PCM and preloaded pointers for full feeds."""
    import numpy as np

    n_calls = len(pcm) // feed_samples
    if not n_calls:
        raise ValueError("benchmark clip has no complete feed")
    audio = np.array(pcm[:n_calls * feed_samples], dtype=np.float32, order="C", copy=True)
    pointers = [ctypes.cast(audio.ctypes.data + offset * audio.itemsize,
                            ctypes.POINTER(ctypes.c_float))
                for offset in range(0, len(audio), feed_samples)]
    return audio, pointers


def measure_feeds(feed, handle, pointers, feed_samples: int, warmup: int, reset) -> list[float]:
    for index in range(warmup):
        status = feed(handle, pointers[index % len(pointers)], feed_samples)
        if status:
            raise RuntimeError(f"transcribe_vad_stream_feed warmup: status {status}")
    reset(handle)
    timings = []
    for pointer in pointers:
        start = time.perf_counter_ns()
        status = feed(handle, pointer, feed_samples)
        elapsed = time.perf_counter_ns() - start
        if status:
            raise RuntimeError(f"transcribe_vad_stream_feed: status {status}")
        timings.append(elapsed / 1_000_000)
    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", nargs="?", const="vad-publication-v1", default="vad-publication-v1")
    parser.add_argument("--variant", default="silero-vad-v6.2")
    parser.add_argument("--library", type=Path)
    args = parser.parse_args(argv)
    driver = bench_driver()
    profiles = driver.benchmark_profiles
    record = driver.catalog_common.load_record(args.variant)
    profile_id, profile = profiles.profile_for(record, args.profile)
    if profile_id != args.profile or profile.get("role") != "vad":
        parser.error("the variant must be governed by the requested VAD profile")
    machine = driver.detect_machine()
    target = profiles.target_for_machine(profile, machine["slug"])
    if target is None:
        parser.error(f"machine {machine['slug']!r} is not a target in {profile_id}")
    warmup = int(profile["speed"]["warmup_calls"])
    repeat = int(profile["speed"]["iterations"])
    threads = int(profile["speed"].get("threads", target.get("threads", 1)))
    if (warmup, repeat, threads) != (32, 1, 1):
        parser.error("VAD publication requires warmup=32, iterations=1, threads=1")
    cells = [cell for cell in profiles.apply_exceptions(
        record, "speed", profiles.expected_speed(record, profile))
        if cell["machine"] == target["machine"]]
    if not cells or any(cell["backend"] != "cpu" for cell in cells):
        parser.error("this driver requires CPU speed cells")
    matches = {cell["sample"]: SAMPLE_RE.fullmatch(cell["sample"]) for cell in cells}
    if any(match is None or match["clip"] != "love-loss" for match in matches.values()):
        parser.error("VAD publication samples must be love-loss-{32,128,512}ms")
    sample_rate = int(profile["speed"]["sample_rate"])
    feeds = profile["speed"]["feed_samples"]
    if any(feeds[name] != int(match["ms"]) * sample_rate // 1000
           for name, match in matches.items()):
        parser.error("profile feed_samples does not match the sample IDs")

    if args.library:
        os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(REPO / "bindings/python/src"))
    import soundfile as sf
    import transcribe_cpp as t
    from transcribe_cpp import _generated

    lib = ctypes.CDLL(t.library_path())
    _generated.configure(lib)
    commit = lib.transcribe_version_commit().decode("ascii")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        parser.error(f"native library does not report a build SHA: {commit!r}")
    sample_path = REPO / "samples/love-loss.wav"
    pcm, sr = sf.read(str(sample_path), dtype="float32")
    if sr != sample_rate or pcm.ndim != 1:
        parser.error("love-loss.wav must be 16 kHz mono")
    if len(pcm) != profile["speed"]["source_samples"]:
        parser.error("love-loss.wav does not have the profile's source_samples")
    if not profile["speed"].get("full_calls_only"):
        parser.error("VAD publication requires full_calls_only")
    # Keep all owning buffers alive until every native call finishes.
    prepared = {name: prepare_feeds(pcm, feeds[name]) for name in matches}
    runs = []
    for item in record["downloads"]:
        model_cells = [cell for cell in cells if cell["quant"] == item["quant"]]
        if not model_cells:
            continue
        gguf = REPO / "models" / args.variant / item["filename"]
        start = time.perf_counter()
        with t.Model(str(gguf), backend="cpu") as model:
            load_ms = (time.perf_counter() - start) * 1000
            info = model.vad_info
            native_info = {"sample_rate": info.sample_rate, "frame_samples": info.frame_samples}
            if native_info != record["vad_info"] or native_info != {
                    "sample_rate": sample_rate, "frame_samples": profile["speed"]["frame_samples"]}:
                parser.error("native VAD info does not match the record and publication profile")
            with model.vad_session(n_threads=threads) as session:
                handle = session._h
                for cell in model_cells:
                    driver.cooldown_wait(float(target.get("cooldown_tctl_c", 0)), 300.0, 10.0)
                    sample = cell["sample"]
                    audio, pointers = prepared[sample]
                    feed_samples = feeds[sample]
                    lib.transcribe_vad_stream_reset(handle)
                    latencies = measure_feeds(lib.transcribe_vad_stream_feed, handle, pointers,
                                             feed_samples, warmup, lib.transcribe_vad_stream_reset)
                    runs.append({
                        "model_path": str(gguf.relative_to(REPO)), "sample": sample,
                        "sample_path": str(sample_path.relative_to(REPO)),
                        "sample_duration_s": feed_samples / info.sample_rate,
                        "clip_duration_s": len(pcm) / info.sample_rate,
                        "omitted_tail_samples": len(pcm) - len(audio),
                        "backend": "cpu", "threads": threads,
                        "feed_samples": feed_samples, "frame_samples": info.frame_samples,
                        "n_calls": len(pointers), "warmup_calls": warmup,
                        "load_ms": round(load_ms, 6),
                        "per_iter": [{"total_ms": round(value, 6)} for value in latencies],
                        "summary": {"total_ms": summarize(latencies)},
                    })
                    print(f"{gguf.name} {sample}: {runs[-1]['summary']['total_ms']}", flush=True)
    name = f"{args.variant}-publication"
    out = REPO / "reports/perf" / machine["slug"] / f"{driver.slugify(name)}_{args.variant}_cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "transcribe-bench-driver-v1", "timestamp": driver.now_utc_iso(),
        "name": name, "publication_profile": profile_id, "machine": machine,
        "git_sha": commit, "variant": args.variant, "backend": "cpu",
        "iters": repeat, "warmup": warmup, "tool": "scripts/vad/bench.py",
        "library": t.library_path(),
        "timing": "wall latency per bare native stream_feed; excludes result capture",
        "percentiles": "median: ordinary median; p95: nearest rank ceil(0.95*n)",
        "runs": runs,
    }, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
