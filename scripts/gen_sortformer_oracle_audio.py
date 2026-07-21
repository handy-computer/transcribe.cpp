#!/usr/bin/env python3
"""Deterministically synthesize a 2-speaker diarization oracle clip.

Sortformer is a diarizer, so the oracle case must be multi-speaker;
samples/jfk.wav and the other committed clips are all single-speaker.
This builds a reproducible 2-speaker mixture from two distinct committed
mono clips, with a controlled overlap region, plus a ground-truth RTTM.

Because we author the timeline, the RTTM is exact: it doubles as a
known-answer for a mini-DER sanity check without any external corpus.
Speaker identity does not depend on language (Sortformer keys on speaker
acoustics), so a cross-language pair maximizes speaker distinctness.

Outputs (16 kHz mono, deterministic):
    samples/sortformer-2spk-mix.wav
    tests/golden/sortformer/sortformer-2spk-mix.rttm

Run:
    uv run scripts/gen_sortformer_oracle_audio.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
REPO = Path(__file__).resolve().parent.parent
SPK_A_SRC = REPO / "samples" / "jfk.wav"      # English male, very distinctive
SPK_B_SRC = REPO / "samples" / "zh.wav"       # different speaker/timbre
OUT_WAV = REPO / "samples" / "sortformer-2spk-mix.wav"
OUT_RTTM = REPO / "tests" / "golden" / "sortformer" / "sortformer-2spk-mix.rttm"
CLIP_ID = "sortformer-2spk-mix"

# Authored timeline in seconds. Overlap is the [9.0, 10.5] region where
# both speakers are active. Total clip length 12.0 s.
SPK_A_WINDOWS = [(0.0, 3.0), (7.0, 10.5)]
SPK_B_WINDOWS = [(3.5, 6.5), (9.0, 12.0)]
TOTAL_SEC = 12.0


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        # Deterministic linear resample; good enough for a diarization
        # oracle clip (no ASR scoring on this audio).
        n_out = int(round(len(audio) * SR / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    # Peak-normalize each source so both speakers sit at a similar level.
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.9).astype(np.float32)


def _tile_to(src: np.ndarray, n: int) -> np.ndarray:
    if len(src) >= n:
        return src[:n]
    reps = int(np.ceil(n / max(1, len(src))))
    return np.tile(src, reps)[:n]


def _lay_windows(src: np.ndarray, windows: list[tuple[float, float]], total_n: int) -> np.ndarray:
    track = np.zeros(total_n, dtype=np.float32)
    for start, end in windows:
        i0 = int(round(start * SR))
        i1 = int(round(end * SR))
        track[i0:i1] = _tile_to(src, i1 - i0)
    return track


def main() -> int:
    for p in (SPK_A_SRC, SPK_B_SRC):
        if not p.exists():
            print(f"error: missing source clip {p}", file=sys.stderr)
            return 1
    total_n = int(round(TOTAL_SEC * SR))
    a = _lay_windows(_load_mono_16k(SPK_A_SRC), SPK_A_WINDOWS, total_n)
    b = _lay_windows(_load_mono_16k(SPK_B_SRC), SPK_B_WINDOWS, total_n)
    mix = a + b
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = (mix / peak * 0.9).astype(np.float32)

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    OUT_RTTM.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_WAV), mix, SR, subtype="PCM_16")

    lines = []
    for spk, windows in (("spk_A", SPK_A_WINDOWS), ("spk_B", SPK_B_WINDOWS)):
        for start, end in windows:
            lines.append(
                f"SPEAKER {CLIP_ID} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>"
            )
    OUT_RTTM.write_text("\n".join(lines) + "\n")

    dur = len(mix) / SR
    print(f"wrote {OUT_WAV.relative_to(REPO)} ({dur:.2f}s, 16kHz mono)")
    print(f"wrote {OUT_RTTM.relative_to(REPO)} ({len(lines)} turns, 2 speakers, overlap [9.0,10.5]s)")
    # Emit a sha so the artifact is verifiably reproducible.
    sha = subprocess.run(["shasum", "-a", "256", str(OUT_WAV)], capture_output=True, text=True)
    print("sha256:", sha.stdout.split()[0] if sha.returncode == 0 else "n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
