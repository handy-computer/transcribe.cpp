#!/usr/bin/env python3
"""
eval_cpp.py - score pipecat-ai/smart-turn-data-v3.2-test with a transcribe.cpp
TURN model (through tools/transcribe-turn-eval, the public ABI) and compare it
clip by clip with the reference baseline written by
run_reference_smart_turn_onnx.py.

  cmake -B build -DTRANSCRIBE_BUILD_TOOLS=ON && cmake --build build --target transcribe-turn-eval
  uv run --project scripts/envs/smart_turn scripts/turn/eval_cpp.py \\
      --model models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf \\
      --ref reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl \\
      --out reports/turn/smart-turn-v3.2-F32.smart-turn-data-v3.2-test.jsonl

Rows: the reference row plus "p_cpp" and "ended_cpp". The summary has the
accuracy with upstream's rule (p > 0.5), the decision agreement with the
reference fp32 and int8 ONNX, and |p_cpp - p_fp32| statistics.
"""
from __future__ import annotations

import argparse
import io
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "turn"))
import run_reference_smart_turn_onnx as ref  # noqa: E402

THRESHOLD = 0.5


def decode(flac: bytes) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=False)
    if pcm.ndim != 1 or sr != 16000:
        raise SystemExit("error: clip is not 16 kHz mono")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def worker(tool: Path, model: Path, jobs: queue.Queue, results: dict, threads: int) -> None:
    proc = subprocess.Popen([str(tool), str(model), "--threads", str(threads)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin and proc.stdout
    while True:
        item = jobs.get()
        if item is None:
            break
        idx, pcm = item
        proc.stdin.write(np.int32(pcm.size).tobytes() + pcm.tobytes())
        proc.stdin.flush()
        line = proc.stdout.readline().decode().split()
        if len(line) != 2:
            raise SystemExit(f"error: transcribe-turn-eval failed on clip {idx}")
        results[idx] = (float(line[0]), line[1] == "1")
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("error: transcribe-turn-eval exited with an error")


def acc(rows: list[dict], key: str) -> dict:
    tp = sum(1 for r in rows if r["label"] and r[key] > THRESHOLD)
    tn = sum(1 for r in rows if not r["label"] and r[key] <= THRESHOLD)
    n = len(rows)
    fp = sum(1 for r in rows if not r["label"] and r[key] > THRESHOLD)
    fn = sum(1 for r in rows if r["label"] and r[key] <= THRESHOLD)
    return {"n": n, "acc_pct": round(100.0 * (tp + tn) / n, 2), "fpr_pct": round(100.0 * fp / n, 2),
            "fnr_pct": round(100.0 * fn / n, 2)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tool", type=Path, default=REPO_ROOT / "build" / "bin" / "transcribe-turn-eval")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--threads", type=int, default=1, help="threads per tool process")
    p.add_argument("--limit", type=int, default=0, help="score only the first N clips (0 = all)")
    args = p.parse_args(argv)

    lines = args.ref.read_text().splitlines()
    header = json.loads(lines[0])
    ref_rows = {r["id"]: r for r in map(json.loads, lines[1:])}

    jobs: queue.Queue = queue.Queue(maxsize=64)
    results: dict = {}
    pool = [threading.Thread(target=worker, args=(args.tool, args.model, jobs, results, args.threads))
            for _ in range(args.jobs)]
    for t in pool:
        t.start()
    t0 = time.time()
    order = []
    for i, (meta, flac) in enumerate(ref.items(header["shards"])):
        if args.limit and i >= args.limit:
            break
        order.append(meta["id"])
        jobs.put((i, decode(flac)))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1} clips, {time.time() - t0:.0f} s", file=sys.stderr)
    for _ in pool:
        jobs.put(None)
    for t in pool:
        t.join()

    rows = []
    for i, cid in enumerate(order):
        r = dict(ref_rows[cid])
        r["p_cpp"], r["ended_cpp"] = results[i]
        rows.append(r)
    dp = np.abs(np.array([r["p_cpp"] - r["p_fp32"] for r in rows]))
    agree_fp32 = sum(1 for r in rows if (r["p_cpp"] > THRESHOLD) == (r["p_fp32"] > THRESHOLD))
    agree_int8 = sum(1 for r in rows if (r["p_cpp"] > THRESHOLD) == (r["p_int8"] > THRESHOLD))
    flips = [{"id": r["id"], "p_cpp": r["p_cpp"], "p_fp32": r["p_fp32"]} for r in rows
             if (r["p_cpp"] > THRESHOLD) != (r["p_fp32"] > THRESHOLD)]
    summary = {
        "cpp": acc(rows, "p_cpp"),
        "ref_fp32": acc(rows, "p_fp32"),
        "ref_int8": acc(rows, "p_int8"),
        "decision_agreement_with_fp32_pct": round(100.0 * agree_fp32 / len(rows), 4),
        "decision_agreement_with_int8_pct": round(100.0 * agree_int8 / len(rows), 2),
        "decision_flips_vs_fp32": flips,
        "p_abs_diff_vs_fp32": {"max": float(dp.max()), "mean": float(dp.mean()),
                               "p99": float(np.percentile(dp, 99))},
    }
    out_header = {"schema": "transcribe-turn-cpp-v1", "model": str(args.model), "ref": str(args.ref),
                  "dataset": header["dataset"], "dataset_revision": header["dataset_revision"],
                  "threshold": THRESHOLD, "rule": "p > 0.5 (as the reference)", "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(json.dumps(out_header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} ({len(rows)} clips, {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
