#!/usr/bin/env python3
"""bench_reference.py - time upstream Silero VAD (silero-vad 6.2.3) scoring a
clip, for the speed comparison in docs/models/silero-vad-v6.2.md.

    uv run --project scripts/envs/silero_vad scripts/vad/bench_reference.py samples/love-loss.wav

Engines, all CPU, one thread (upstream's own setting: load_silero_vad and
OnnxWrapper pin torch / onnxruntime to 1 thread):
  jit       TorchScript, the get_speech_timestamps loop (one call per frame)
  onnx      OnnxWrapper.audio_forward (one ONNX call per frame)
  sequence  SileroVADSequence (one ONNX call per 512-frame block)
Each engine scores the clip --repeat times after one warm-up; the minimum is
reported as milliseconds and as x real time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf
import torch


def best_of(fn, repeat: int) -> float:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()
    if args.repeat <= 0:
        p.error("--repeat must be positive")

    from silero_vad import load_silero_vad

    pcm, sr = sf.read(args.audio, dtype="float32")
    assert sr == 16000 and pcm.ndim == 1
    audio = torch.from_numpy(pcm)
    dur = len(pcm) / 16000.0

    jit = load_silero_vad()

    def run_jit():
        jit.reset_states()
        with torch.inference_mode():
            for s in range(0, len(audio), 512):
                c = audio[s:s + 512]
                if len(c) < 512:
                    c = torch.nn.functional.pad(c, (0, 512 - len(c)))
                jit(c, 16000).item()

    onnx = load_silero_vad(onnx=True)
    seq = load_silero_vad(sequence=True)
    engines = {
        "jit": run_jit,
        "onnx": lambda: onnx.audio_forward(audio, 16000),
        "sequence": lambda: seq.audio_forward(pcm, 16000),
    }
    print(f"{args.audio}: {dur:.1f} s, {int(np.ceil(len(pcm) / 512))} frames")
    for name, fn in engines.items():
        t = best_of(fn, args.repeat)
        print(f"  {name:<9} {t * 1000:9.2f} ms  {dur / t:8.0f}x realtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
