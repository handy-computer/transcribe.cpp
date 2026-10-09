#!/usr/bin/env python3
"""parity.py - speech-segment parity of transcribe.cpp's VAD against the
reference (silero-vad 6.2.3 get_speech_timestamps, default parameters) over a
corpus of 16 kHz mono WAVs.

    uv run --project scripts/envs/silero_vad scripts/vad/parity.py \
        --gguf models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf \
        samples/diar/ami-ihm-test/*.wav

Inputs are WAV paths, or .jsonl manifests whose rows carry an "audio" path
(scripts/wer manifests). For each file the reference segments are computed in
process and the C++ segments read from `transcribe-cli -o`. Reports, per file
and in total: files whose segment lists are identical, segment count deltas,
and the boundary error (|start| / |end| differences in samples) over segments
that pair up one to one. Exit status 1 when any file differs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from segment_output import parse_segments

REPO = Path(__file__).resolve().parents[2]


def expand(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                if line.strip():
                    a = Path(json.loads(line)["audio"])
                    out.append(a if a.is_absolute() else (REPO / a))
        else:
            out.append(p)
    return out


def cpp_segments(cli: Path, gguf: Path, wav: Path, threads: int) -> list[dict[str, int]]:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "segments.txt"
        r = subprocess.run([str(cli), "--backend", "cpu", "--threads", str(threads), "-m", str(gguf),
                            "-o", str(output), str(wav)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"error: transcribe-cli failed on {wav}:\n{r.stdout}\n{r.stderr}")
        return parse_segments(output.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--cli", type=Path, default=REPO / "build" / "bin" / "transcribe-cli")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N files only")
    args = ap.parse_args()

    files = expand(args.inputs)
    if args.limit < 0:
        ap.error("--limit must be nonnegative")
    if args.limit:
        files = files[:args.limit]
    if not files:
        ap.error("inputs contain no audio files")

    import silero_vad
    from silero_vad import get_speech_timestamps, load_silero_vad

    torch.set_num_threads(1)
    model = load_silero_vad()

    n_same = n_ref = n_cpp = 0
    seconds = 0.0
    starts: list[int] = []
    ends: list[int] = []
    differing = []
    for i, wav in enumerate(files):
        pcm, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if sr != 16000 or pcm.ndim != 1:
            raise SystemExit(f"error: {wav} is not 16 kHz mono")
        if len(pcm) == 0:
            raise SystemExit(f"error: {wav} contains no audio samples")
        seconds += len(pcm) / 16000.0
        ref = get_speech_timestamps(torch.from_numpy(pcm), model)
        cpp = cpp_segments(args.cli, args.gguf, wav, args.threads)
        n_ref += len(ref)
        n_cpp += len(cpp)
        if ref == cpp:
            n_same += 1
        else:
            differing.append((wav, ref, cpp))
        if len(ref) == len(cpp):
            starts += [abs(a["start"] - b["start"]) for a, b in zip(ref, cpp)]
            ends += [abs(a["end"] - b["end"]) for a, b in zip(ref, cpp)]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} files, {n_same} identical", file=sys.stderr)

    print(f"silero-vad {silero_vad.__version__} vs {args.gguf.name}: {len(files)} files, {seconds / 3600:.2f} h")
    print(f"  identical segment lists: {n_same}/{len(files)}")
    print(f"  segments: reference {n_ref}, c++ {n_cpp}")
    if starts:
        s, e = np.array(starts), np.array(ends)
        print(f"  paired boundaries: {len(s)} segments, start diff != 0 on {int((s != 0).sum())}, "
              f"end diff != 0 on {int((e != 0).sum())}, max {int(max(s.max(), e.max()))} samples")
    for wav, ref, cpp in differing[:10]:
        print(f"  DIFF {wav.name}: reference {len(ref)} segments, c++ {len(cpp)}")
        for k, (a, b) in enumerate(zip(ref, cpp)):
            if a != b:
                print(f"    first difference at segment {k}: reference {a}, c++ {b}")
                break
    return 0 if not differing else 1


if __name__ == "__main__":
    raise SystemExit(main())
