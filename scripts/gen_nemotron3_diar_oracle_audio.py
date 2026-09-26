#!/usr/bin/env python3
"""Deterministically synthesize an 8-speaker diarization oracle clip.

Nemotron-3-Diarization tracks up to 8 speakers, and the existing
diarization fixtures (sortformer-2spk-mix, AMI) never exercise speaker
channels 5-8. This builds a reproducible 8-speaker mixture (the model's
cap, so every output channel is exercised) from eight committed mono
clips by different speakers (jfk, six FLEURS `-long` bench clips,
product-names), with
speaker returns (arrival-order identity must survive
the speaker cache) and three short overlap regions, plus a ground-truth
RTTM. Speaker identity does not depend on language, so a cross-language
pool maximizes speaker distinctness.

The clip is 92 s so that AOSC speaker-cache compression triggers at every
preset (very_high_latency compresses at step 0; the low-latency presets
after ~40 s), i.e. the streaming oracle covers FIFO -> cache -> compress.

Each source is trimmed of leading/trailing silence (the FLEURS clips open
with up to ~5 s of near-silence), then each turn reads the next unused
stretch of its source (wrapping when exhausted), so a returning speaker
says new words. The RTTM is authored at turn level: it includes any short
pauses inside a source window, so it is a sanity reference, not a
frame-exact label.

A second, non-aligned case is the same mix truncated to 91.337 s
(1461392 samples: not a multiple of the 160-sample hop, 9133 mel frames
not a multiple of 8). It exercises NeMo's floor(n/160) framing, the
zero-padded final feature-stacking group and the partial final chunk,
none of which the 92 s clip hits.

Outputs (16 kHz mono, deterministic):
    samples/nemotron3-diar-8spk-mix.wav
    samples/nemotron3-diar-8spk-mix-trunc.wav
    tests/golden/nemotron3_diar/nemotron3-diar-8spk-mix.rttm

Run:
    uv run scripts/gen_nemotron3_diar_oracle_audio.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
REPO = Path(__file__).resolve().parent.parent
CLIP_ID = "nemotron3-diar-8spk-mix"
OUT_WAV = REPO / "samples" / f"{CLIP_ID}.wav"
OUT_RTTM = REPO / "tests" / "golden" / "nemotron3_diar" / f"{CLIP_ID}.rttm"
OUT_TRUNC_WAV = REPO / "samples" / f"{CLIP_ID}-trunc.wav"
TOTAL_SEC = 92.0
TRUNC_SAMPLES = 1461392  # 91.337 s

# Speaker label -> committed source clip (all 16 kHz mono).
SOURCES = {
    "spk_A": REPO / "samples" / "jfk.wav",
    "spk_B": REPO / "samples" / "zh-long.wav",
    "spk_C": REPO / "samples" / "ru-long.wav",
    "spk_D": REPO / "samples" / "ja-long.wav",
    "spk_E": REPO / "samples" / "ko-long.wav",
    "spk_F": REPO / "samples" / "vi-long.wav",
    "spk_G": REPO / "samples" / "ar-long.wav",
    # Not uk-long.wav: the reference merges that reader into spk_E's channel,
    # leaving output channel 8 unexercised. product-names.wav is separated
    # cleanly (its provenance is unrecorded, like jfk.wav; see samples/README.md).
    "spk_H": REPO / "samples" / "product-names.wav",
}

# Authored timeline (speaker, start s, end s), in arrival order A..H; every
# speaker returns once. Overlaps: B/C [10.0, 10.5], E/F [26.5, 27.0],
# H/D [81.0, 81.5].
TURNS = [
    ("spk_A", 0.0, 5.0),
    ("spk_B", 5.5, 10.5),
    ("spk_C", 10.0, 16.0),
    ("spk_D", 16.5, 21.5),
    ("spk_E", 22.0, 27.0),
    ("spk_F", 26.5, 32.0),
    ("spk_G", 32.5, 37.5),
    ("spk_H", 38.0, 43.0),
    ("spk_A", 43.5, 48.5),
    ("spk_C", 49.0, 54.0),
    ("spk_B", 54.5, 59.5),
    ("spk_G", 60.0, 65.0),
    ("spk_F", 65.5, 70.5),
    ("spk_E", 71.0, 76.0),
    ("spk_H", 76.5, 81.5),
    ("spk_D", 81.0, 87.0),
    ("spk_A", 87.5, 92.0),
]


def _trim_silence(audio: np.ndarray, floor_db: float = -30.0, frame: int = 320, run: int = 5) -> np.ndarray:
    """Drop leading/trailing silence: speech starts/ends at the first/last run of
    `run` consecutive 20 ms frames within |floor_db| of the clip peak (a lone
    click in the lead-in does not count)."""
    n = len(audio) // frame * frame
    rms = np.sqrt((audio[:n].reshape(-1, frame) ** 2).mean(axis=1))
    loud = 20 * np.log10(rms / (np.abs(audio).max() + 1e-12) + 1e-12) > floor_db
    sustained = np.convolve(loud.astype(int), np.ones(run, dtype=int), mode="valid") == run
    starts = np.flatnonzero(sustained)
    if starts.size == 0:
        return audio
    return audio[starts[0] * frame:(starts[-1] + run) * frame]


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"error: {path} is {sr} Hz, expected {SR}")
    audio = _trim_silence(audio)
    # Peak-normalize each source so all speakers sit at a similar level.
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.9).astype(np.float32)


def _take(src: np.ndarray, offset: int, n: int) -> tuple[np.ndarray, int]:
    """Next n samples of src starting at offset, wrapping; returns (chunk, new offset)."""
    idx = (offset + np.arange(n)) % len(src)
    return src[idx], (offset + n) % len(src)


def main() -> int:
    for p in SOURCES.values():
        if not p.exists():
            print(f"error: missing source clip {p}", file=sys.stderr)
            return 1
    total_n = int(round(TOTAL_SEC * SR))
    srcs = {spk: _load_mono_16k(p) for spk, p in SOURCES.items()}
    offsets = {spk: 0 for spk in SOURCES}
    mix = np.zeros(total_n, dtype=np.float32)
    for spk, start, end in TURNS:
        i0, i1 = int(round(start * SR)), int(round(end * SR))
        chunk, offsets[spk] = _take(srcs[spk], offsets[spk], i1 - i0)
        mix[i0:i1] += chunk
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = (mix / peak * 0.9).astype(np.float32)

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    OUT_RTTM.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_WAV), mix, SR, subtype="PCM_16")
    lines = [
        f"SPEAKER {CLIP_ID} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>"
        for spk, start, end in sorted(TURNS, key=lambda t: t[1])
    ]
    OUT_RTTM.write_text("\n".join(lines) + "\n")

    sf.write(str(OUT_TRUNC_WAV), mix[:TRUNC_SAMPLES], SR, subtype="PCM_16")

    print(f"wrote {OUT_WAV.relative_to(REPO)} ({len(mix) / SR:.2f}s, 16kHz mono)")
    print(f"wrote {OUT_TRUNC_WAV.relative_to(REPO)} ({TRUNC_SAMPLES / SR:.3f}s)")
    print(f"wrote {OUT_RTTM.relative_to(REPO)} ({len(lines)} turns, {len(SOURCES)} speakers)")
    sha = subprocess.run(["shasum", "-a", "256", str(OUT_WAV)], capture_output=True, text=True)
    print("sha256:", sha.stdout.split()[0] if sha.returncode == 0 else "n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
