#!/usr/bin/env python3
"""
run_cpp_multitalker.py - drive the C++ multitalker bundle over an AMI
manifest and emit SegLST hypotheses for cpWER scoring.

One transcribe-cli invocation per meeting (batch size 1: multitalker is a
transcribe_run path), diarize=ON, --batch-jsonl output parsed into
<out-dir>/<meeting>.seglst.json. Supervision mode is selected via
TRANSCRIBE_MULTITALKER_MODE and forwarded verbatim from --mode.

Usage:
    uv run scripts/diar/run_cpp_multitalker.py \
      --model models/multitalker-parakeet-streaming-0.6b-v1/bundle/multitalker-parakeet-streaming-0.6b-v1-F32.gguf \
      --manifest samples/diar/ami-ihm-test.manifest.jsonl \
      --mode kernel \
      --out-dir reports/cpwer/cpp-kernel \
      [--meetings IS1009a,ES2004a] [--backend cpu|metal|cuda] [--cli PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["masked", "kernel"], default="masked")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--cli", default=str(REPO / "build" / "bin" / "transcribe-cli"))
    ap.add_argument("--meetings", default="", help="comma-separated meeting-id filter")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = REPO / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    want = {m.strip() for m in args.meetings.split(",") if m.strip()}

    env = dict(os.environ)
    env["TRANSCRIBE_MULTITALKER_MODE"] = args.mode

    entries = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    n_ok = 0
    for e in entries:
        mid = e["id"].split(".")[0]
        if want and mid not in want:
            continue
        audio = e["audio"]
        if not Path(audio).is_absolute():
            audio = str(REPO / audio)

        with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False, dir=tmp_dir) as bf:
            bf.write(audio + "\n")
            batch_list = bf.name
        cmd = [
            args.cli, "-q", "--diarize", "--backend", args.backend,
            "--batch", batch_list, "--batch-jsonl",
            "-m", args.model,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        os.unlink(batch_list)
        if proc.returncode != 0:
            print(f"FAIL {mid}: exit {proc.returncode}\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue

        row = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    cand = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cand.get("segments"):
                    row = cand
        if row is None:
            print(f"FAIL {mid}: no JSONL row with segments in CLI output", file=sys.stderr)
            continue

        seglst = []
        for seg in row["segments"]:
            text = (seg.get("text") or "").strip()
            if not text or seg.get("speaker_id", 0) <= 0:
                continue
            seglst.append(
                {
                    "session_id": mid,
                    "speaker": f"S{seg['speaker_id']}",
                    "start_time": seg["t0_ms"] / 1000.0,
                    "end_time": seg["t1_ms"] / 1000.0,
                    "words": text,
                }
            )
        (out_dir / f"{mid}.seglst.json").write_text(json.dumps(seglst, indent=1) + "\n")
        n_ok += 1
        print(f"ok {mid}: {len(seglst)} segments in {time.time() - t0:.0f}s")

    print(f"{n_ok} meetings -> {out_dir}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
