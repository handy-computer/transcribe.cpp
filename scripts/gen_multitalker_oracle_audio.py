#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "soundfile>=0.12",
# ]
# ///
"""Deterministically synthesize a 2-speaker ENGLISH multitalker oracle clip.

The multitalker Parakeet + Sortformer pipeline needs an English
multi-speaker fixture (the sortformer 2spk mix is cross-language, which is
right for a diarizer oracle but wrong for scoring English ASR hyps). This
builds a reproducible 2-speaker English mixture from two committed clips
with distinct voices, plus a ground-truth RTTM for the authored timeline.

Speaker A: samples/jfk.wav (JFK, 1961)
Speaker B: samples/whole-earth.wav (different male voice)

Timeline: mostly turn-taking with one authored 0.5 s overlap at
[5.0, 5.5] s so the overlap path is exercised without dominating the clip.

Outputs (16 kHz mono, deterministic):
    samples/multitalker-2spk-mix.wav
    tests/golden/parakeet/multitalker-2spk-mix.rttm

Run:
    uv run scripts/gen_multitalker_oracle_audio.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
REPO = Path(__file__).resolve().parent.parent
SPK_A_SRC = REPO / "samples" / "jfk.wav"
SPK_B_SRC = REPO / "samples" / "whole-earth.wav"
OUT_WAV = REPO / "samples" / "multitalker-2spk-mix.wav"
OUT_RTTM = REPO / "tests" / "golden" / "parakeet" / "multitalker-2spk-mix.rttm"
CLIP_ID = "multitalker-2spk-mix"

# Authored timeline in seconds. Each speaker's source audio is consumed
# sequentially into that speaker's windows, so window boundaries cut the
# source mid-sentence — fine for parity fixtures (we compare hyps against
# the NeMo reference, not against a ground-truth transcript).
SPK_A_WINDOWS = [(0.0, 5.5), (9.5, 14.0)]
SPK_B_WINDOWS = [(5.0, 9.5)]
TOTAL_SEC = 14.0


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.9).astype(np.float32)


def _lay_windows_sequential(src: np.ndarray, windows: list[tuple[float, float]], total_n: int) -> np.ndarray:
    """Consume src sequentially into each window (no tiling restart per
    window), so speech flows naturally across a speaker's turns."""
    track = np.zeros(total_n, dtype=np.float32)
    cursor = 0
    for start, end in windows:
        i0 = int(round(start * SR))
        i1 = int(round(end * SR))
        n = i1 - i0
        seg = src[cursor : cursor + n]
        if len(seg) < n:  # source exhausted: wrap deterministically
            reps = int(np.ceil(n / max(1, len(src))))
            seg = np.concatenate([seg, np.tile(src, reps)])[:n]
        track[i0:i1] = seg
        cursor += n
    return track


def main() -> int:
    for p in (SPK_A_SRC, SPK_B_SRC):
        if not p.exists():
            print(f"error: missing source clip {p}", file=sys.stderr)
            return 1
    total_n = int(round(TOTAL_SEC * SR))
    a = _lay_windows_sequential(_load_mono_16k(SPK_A_SRC), SPK_A_WINDOWS, total_n)
    b = _lay_windows_sequential(_load_mono_16k(SPK_B_SRC), SPK_B_WINDOWS, total_n)
    mix = a + b
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = (mix / peak * 0.9).astype(np.float32)

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    OUT_RTTM.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_WAV), mix, SR, subtype="PCM_16")

    lines = []
    for spk, windows in (("spk_A", SPK_A_WINDOWS), ("spk_B", SPK_B_WINDOWS)):
        for start, end in windows:
            lines.append(f"SPEAKER {CLIP_ID} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>")
    OUT_RTTM.write_text("\n".join(lines) + "\n")

    dur = len(mix) / SR
    print(f"wrote {OUT_WAV.relative_to(REPO)} ({dur:.2f}s, 16kHz mono)")
    print(f"wrote {OUT_RTTM.relative_to(REPO)} ({len(lines)} turns, 2 speakers, overlap [5.0,5.5]s)")
    sha256 = hashlib.sha256(OUT_WAV.read_bytes()).hexdigest()
    print("sha256:", sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
