#!/usr/bin/env python3
"""parity.py - speech-segment parity of transcribe.cpp's VAD against the
reference (silero-vad 6.2.3 get_speech_timestamps, default parameters) over a
corpus of 16 kHz mono WAVs.

    uv run --project scripts/envs/silero_vad scripts/vad/parity.py \
        --gguf models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf \
        samples/diar/ami-ihm-test/*.wav

Inputs are WAV paths, or .jsonl manifests whose rows carry an "audio" path
(scripts/wer manifests). Reports identical segment lists and boundary
differences over segments that pair one to one. Exit 1 when any file differs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
SEGMENT = re.compile(r"segment: \d+ start=(\d+) end=(\d+)")


def expand(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.suffix != ".jsonl":
            out.append(p)
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                a = Path(json.loads(line)["audio"])
                out.append(a if a.is_absolute() else REPO / a)
    return out


def cpp_segments(cli: Path, gguf: Path, wav: Path, threads: int) -> list[dict[str, int]]:
    r = subprocess.run([str(cli), "--backend", "cpu", "--threads", str(threads), "-m", str(gguf), str(wav)],
                       capture_output=True, text=True, check=True)
    return [{"start": int(m[1]), "end": int(m[2])}
            for m in (SEGMENT.match(line.strip()) for line in r.stdout.splitlines()) if m]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--cli", type=Path, default=REPO / "build" / "bin" / "transcribe-cli")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import silero_vad

    torch.set_num_threads(1)
    model = silero_vad.load_silero_vad()
    files = expand(args.inputs)
    n_same = n_paired = n_moved = 0
    seconds = 0.0
    for wav in files:
        pcm, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if sr != 16000 or pcm.ndim != 1:
            raise SystemExit(f"error: {wav} is not 16 kHz mono")
        seconds += len(pcm) / 16000.0
        ref = silero_vad.get_speech_timestamps(torch.from_numpy(pcm), model)
        cpp = cpp_segments(args.cli, args.gguf, wav, args.threads)
        if ref == cpp:
            n_same += 1
        else:
            print(f"DIFF {wav.name}: reference {len(ref)} segments, c++ {len(cpp)}; "
                  f"first difference {next((a, b) for a, b in zip(ref + [None], cpp + [None]) if a != b)}")
        if len(ref) == len(cpp):
            n_paired += len(ref)
            n_moved += sum(a != b for a, b in zip(ref, cpp))

    print(f"silero-vad {silero_vad.__version__} vs {args.gguf.name}: {len(files)} files, {seconds / 3600:.2f} h")
    print(f"  identical segment lists: {n_same}/{len(files)}")
    print(f"  paired segments: {n_paired}, with a boundary difference: {n_moved}")
    return 0 if n_same == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
