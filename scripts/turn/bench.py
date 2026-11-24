#!/usr/bin/env python3
"""
bench.py - this machine's TURN publication speed cells
(catalog/_benchmark_profiles.json, turn-publication-v1) through the Python
binding's raw C ABI: tools/transcribe-bench is ASR-only and the binding has
no turn session wrapper yet.

  uv run --project scripts/envs/smart_turn scripts/turn/bench.py --variant smart-turn-v3.2 \\
      --library build-shared/src/libtranscribe.dylib
  uv run --project scripts/envs/smart_turn scripts/turn/bench.py --variant sava \\
      --library build-shared/src/libtranscribe.dylib
  uv run scripts/catalog/ingest_perf.py

Cells (the time is the native compute, mel + encode + decode, from
transcribe_turn_get_timings; wall time is recorded beside it):
  <clip>-<N>s        transcribe_turn_run on the first N s of samples/<clip>.wav
  <clip>-<N>ms       one pass of transcribe_turn_stream_feed over the clip in
                     N ms feeds; the mean time per feed (streaming models)
  <clip>-<N>ms-x<U>  one pass of transcribe_turn_stream_feed_batch over U
                     sessions each fed the clip in N ms feeds; the mean time
                     per tick, which carries U * N ms of audio
  -1t suffix         one thread; without it the library default (n_threads 0)

Before every cell it waits for a quiet machine (1-minute load average below
--max-load and none of the --busy processes running) and records the load
averages before and after it. Writes one bench-driver report (scripts/bench/
run.py's shape) to reports/perf/<machine-slug>/<variant>-publication_<variant>_cpu.json.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_RE = re.compile(r"^(?P<clip>.+?)-(?:(?P<seconds>\d+(?:\.\d+)?)s|(?P<ms>\d+)ms(?:-x(?P<users>\d+))?)"
                     r"(?P<one>-1t)?$")
BUSY = ("transcribe-turn-eval", "transcribe-quantize", "turn-perf-bench", "transcribe-cli",
        "transcribe-bench", "transcribe-batch-bench", "swift-frontend", "cc1plus", "clang")


def bench_driver():
    """scripts/bench/run.py, for its profiles and machine detection."""
    spec = importlib.util.spec_from_file_location("bench_driver", REPO_ROOT / "scripts" / "bench" / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def busy_processes(names: tuple[str, ...]) -> list[str]:
    out = subprocess.run(["ps", "-Ao", "comm="], capture_output=True, text=True).stdout
    running = {Path(line.strip()).name for line in out.splitlines() if line.strip()}
    return sorted(name for name in names if name in running)


def wait_quiet(max_load: float, names: tuple[str, ...], timeout_s: float) -> bool:
    """Wait until the machine is quiet; False when it never got there."""
    t0 = time.time()
    while True:
        load = os.getloadavg()[0]
        busy = busy_processes(names)
        if load < max_load and not busy:
            return True
        if time.time() - t0 > timeout_s:
            return False
        print(f"  waiting for quiet: load {load:.2f}, busy {busy}", flush=True)
        time.sleep(30)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--library", type=Path, help="a shared libtranscribe (TRANSCRIBE_LIBRARY)")
    p.add_argument("--max-load", type=float, default=3.0, help="1-minute load average to wait for")
    p.add_argument("--quiet-timeout", type=float, default=3600.0, help="seconds to wait for quiet per cell")
    p.add_argument("--busy", default=",".join(BUSY), help="process names that must not be running")
    p.add_argument("--smoke", action="store_true",
                   help="1 iteration, no warmup, no quiet gate, report to /tmp without a profile stamp")
    args = p.parse_args(argv)
    if args.smoke:
        args.max_load, args.busy = float("inf"), ""
    busy_names = tuple(name for name in args.busy.split(",") if name)

    driver = bench_driver()
    profiles = driver.benchmark_profiles
    machine = driver.detect_machine()
    record = driver.catalog_common.load_record(args.variant)
    profile_id, profile = profiles.profile_for(record)
    if profile.get("role") != "turn":
        p.error(f"{args.variant} is not a TURN record (profile {profile_id})")
    target = profiles.target_for_machine(profile, machine["slug"])
    if target is None:
        p.error(f"machine {machine['slug']!r} is not a target in profile {profile_id}")
    cells = [cell for cell in profiles.apply_exceptions(record, "speed", profiles.expected_speed(record, profile))
             if cell["machine"] == target["machine"]]
    if not cells:
        p.error(f"{args.variant} has no publication cells on {target['machine']}")
    quants = {cell["quant"] for cell in cells}
    ggufs = [REPO_ROOT / "models" / args.variant / item["filename"]
             for item in record["downloads"] if item["quant"] in quants]
    backends = list(dict.fromkeys(cell["backend"] for cell in cells))
    samples = list(dict.fromkeys(cell["sample"] for cell in cells))
    if any(not CELL_RE.match(sample) for sample in samples):
        p.error(f"unparsed samples: {[s for s in samples if not CELL_RE.match(s)]}")
    warmup, repeat = int(profile["speed"]["warmup"]), int(profile["speed"]["iterations"])
    if args.smoke:
        warmup, repeat = 0, 1

    if args.library is not None:
        os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "python" / "src"))
    import soundfile as sf
    import transcribe_cpp as t
    lib, gen = t._lib, t._generated

    def check(status: int, what: str) -> None:
        if status != 0:
            raise SystemExit(f"error: {what} failed with status {status}")

    def session(model, n_threads: int) -> ctypes.c_void_p:
        params = gen.transcribe_turn_session_params()
        lib.transcribe_turn_session_params_init(ctypes.byref(params))
        params.n_threads = n_threads
        handle = ctypes.c_void_p()
        check(lib.transcribe_turn_session_init(model._h, ctypes.byref(params), ctypes.byref(handle)),
              "turn_session_init")
        return handle

    def stage_ms(handle) -> tuple[float, float, float]:
        tm = gen.transcribe_timings()
        lib.transcribe_timings_init(ctypes.byref(tm))
        check(lib.transcribe_turn_get_timings(handle, ctypes.byref(tm)), "turn_get_timings")
        return tm.mel_ms, tm.encode_ms, tm.decode_ms

    def floats(audio: np.ndarray):
        return audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    timestamp = driver.now_utc_iso()
    commit = t.native_commit()
    name = f"{args.variant}-publication"
    for backend in backends:
        runs = []
        for gguf in ggufs:
            t0 = time.perf_counter()
            model = t.Model(str(gguf), backend=backend)
            load_ms = (time.perf_counter() - t0) * 1000
            info = gen.transcribe_turn_info()
            lib.transcribe_turn_info_init(ctypes.byref(info))
            check(lib.transcribe_turn_get_info(model._h, ctypes.byref(info)), "turn_get_info")
            for sample in samples:
                m = CELL_RE.match(sample)
                n_threads = 1 if m["one"] else 0
                sample_path = REPO_ROOT / "samples" / f"{m['clip']}.wav"
                pcm, sr = sf.read(str(sample_path), dtype="float32")
                assert sr == 16000 and pcm.ndim == 1
                users = int(m["users"] or 1)
                handles = [session(model, n_threads) for _ in range(users)]
                if m["seconds"]:
                    seconds = float(m["seconds"])
                    if pcm.size < seconds * 16000:
                        raise SystemExit(f"error: {sample_path} is shorter than {seconds} s")
                    audio = np.ascontiguousarray(pcm[: int(seconds * 16000)])
                    duration_s = audio.size / 16000

                    def once(audio=audio, h=handles[0]):
                        w0 = time.perf_counter()
                        check(lib.transcribe_turn_run(h, floats(audio), audio.size, None), "turn_run")
                        wall = (time.perf_counter() - w0) * 1000
                        return (*stage_ms(h), wall)
                else:
                    if info.frame_samples <= 0:
                        raise SystemExit(f"error: {args.variant} cannot stream, but the profile asks for {sample}")
                    feed = int(m["ms"]) * 16
                    audio = np.ascontiguousarray(pcm)
                    chunks = [np.ascontiguousarray(audio[i:i + feed])
                              for i in range(0, audio.size - feed + 1, feed)]
                    duration_s = users * feed / 16000
                    session_array = (ctypes.c_void_p * users)(*[h.value for h in handles])
                    lengths = (ctypes.c_int * users)(*([feed] * users))
                    pcm_arrays = [(ctypes.POINTER(ctypes.c_float) * users)(*([floats(c)] * users))
                                  for c in chunks]

                    def once(chunks=chunks, handles=handles, session_array=session_array,
                             lengths=lengths, pcm_arrays=pcm_arrays, users=users):
                        # Mean per-feed (per-tick) times over one pass of the clip.
                        mel = enc = dec = wall = 0.0
                        for h in handles:
                            lib.transcribe_turn_stream_reset(h)
                        for i, chunk in enumerate(chunks):
                            w0 = time.perf_counter()
                            if users == 1:
                                status = lib.transcribe_turn_stream_feed(handles[0], floats(chunk), chunk.size)
                            else:
                                status = lib.transcribe_turn_stream_feed_batch(
                                    session_array, pcm_arrays[i], lengths, users)
                            wall += time.perf_counter() - w0
                            check(status, "turn_stream_feed")
                            a, b, c = stage_ms(handles[0])
                            mel, enc, dec = mel + a, enc + b, dec + c
                        n = len(chunks)
                        return mel / n, enc / n, dec / n, wall * 1000 / n

                if not wait_quiet(args.max_load, busy_names, args.quiet_timeout):
                    raise SystemExit(f"error: the machine did not get quiet for {sample}")
                load_before = [round(x, 2) for x in os.getloadavg()]
                for _ in range(warmup):
                    once()
                per_iter = []
                for _ in range(repeat):
                    mel, enc, dec, wall = once()
                    per_iter.append({"mel_ms": mel, "encode_ms": enc, "decode_ms": dec,
                                     "total_ms": mel + enc + dec, "wall_ms": wall})
                load_after = [round(x, 2) for x in os.getloadavg()]
                for h in handles:
                    lib.transcribe_turn_session_free(h)
                runs.append({
                    "model_path": str(gguf.relative_to(REPO_ROOT)),
                    "sample": sample,
                    "sample_path": str(sample_path.relative_to(REPO_ROOT)),
                    "sample_duration_s": duration_s,
                    "backend": model.backend,
                    "threads": n_threads,
                    "users": users,
                    "load_ms": round(load_ms, 1),
                    "load_avg_before": load_before,
                    "load_avg_after": load_after,
                    "per_iter": per_iter,
                    "summary": {field: {"mean": statistics.fmean(it[field] for it in per_iter),
                                        "median": statistics.median(it[field] for it in per_iter),
                                        "min": min(it[field] for it in per_iter)}
                                for field in ("mel_ms", "encode_ms", "decode_ms", "total_ms", "wall_ms")},
                })
                s = runs[-1]["summary"]["total_ms"]
                print(f"{gguf.name:28} {backend:4} {sample:16} mean {s['mean']:9.3f} ms  median "
                      f"{s['median']:9.3f}  min {s['min']:9.3f}  load {load_before[0]:.2f}->{load_after[0]:.2f}",
                      flush=True)
            model.close()

        out = REPO_ROOT / "reports" / "perf" / machine["slug"] / \
            f"{driver.slugify(name)}_{args.variant}_{backend}.json"
        if args.smoke:
            out = Path("/tmp") / f"turn-bench-smoke_{args.variant}_{backend}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "transcribe-bench-driver-v1",
            "timestamp": timestamp,
            "name": name,
            "publication_profile": None if args.smoke else profile_id,
            "machine": machine,
            "machine_target": target["machine"],
            "machine_display": target.get("display"),
            # The build the native library reports, which is what ran.
            "git_sha": commit if commit != "unknown" else driver.get_git_sha(REPO_ROOT),
            "variant": args.variant,
            "backend": backend,
            "iters": repeat,
            "warmup": warmup,
            "max_load": args.max_load,
            "tool": "scripts/turn/bench.py",
            "runs": runs,
        }, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
