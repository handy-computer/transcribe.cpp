#!/usr/bin/env python3
"""gen_segment_vectors.py - test vectors for the VAD segmentation port
(transcribe::vad_probs_to_segments), straight from the reference
silero_vad.get_speech_timestamps_from_probs (silero-vad 6.2.3).

    uv run --project scripts/envs/silero_vad scripts/vad/gen_segment_vectors.py

Writes tests/fixtures/vad_segment_vectors.inc, consumed by
tests/vad_dispatch_unit.cpp. A third of the vectors are a sticky random
walk over a 1/16 probability grid, a third clean speech bursts, a third long
speech with short repeated gaps under max-speech splitting (equal-length
candidate silences). Parameters cover the defaults,
explicit neg_threshold, zero padding, both max-speech split modes, and ms
values aligned to the frame so every comparison is hit at equality.
"""

from __future__ import annotations

import random
from pathlib import Path

import silero_vad
from silero_vad import get_speech_timestamps_from_probs

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "fixtures" / "vad_segment_vectors.inc"
FRAME = 512
N_VECTORS = 360

# Probabilities are on a 1/16 grid (exact in float32, and equal to the grid
# thresholds below, so `>=` vs `>` is exercised) and are stored one char per
# frame, 'a' + 16 * p.
LEVELS = 16


def walk(rng: random.Random, n: int) -> list[int]:
    """Sticky random walk: runs of every length at every level."""
    level = rng.choice([1, 15])
    out = []
    for _ in range(n):
        if rng.random() < 0.08:
            level = rng.choice([0, 2, 4, 5, 6, 8, 10, 12, 14, 16])
        out.append(max(0, min(LEVELS, level + rng.randint(-1, 1))))
    return out


def bursts(rng: random.Random, n: int) -> list[int]:
    """Clean speech bursts and gaps with lengths from a small set, so silences
    of equal length (ties for the max-speech split) are common."""
    out: list[int] = []
    speech = rng.random() < 0.5
    while len(out) < n:
        k = rng.choice([1, 2, 3, 4, 6, 10]) if not speech else rng.choice([2, 5, 8, 20, 40])
        out += [rng.choice([12, 16]) if speech else rng.choice([0, 1, 4]) for _ in range(k)]
        speech = not speech
    return out[:n]


def long_speech(rng: random.Random, n: int) -> list[int]:
    """Long speech with short, repeated gaps: the max-speech split path, with
    several candidate silences of equal length per segment."""
    out: list[int] = []
    gaps = rng.sample([1, 2, 3, 4], 2)
    while len(out) < n:
        out += [16] * rng.randint(3, 25)
        out += [0] * rng.choice(gaps)
    return out[:n]


def split_params(rng: random.Random) -> dict:
    return {
        "threshold": 0.5,
        "neg_threshold": None,
        "min_speech_ms": rng.choice([0, 250]),
        "min_silence_ms": rng.choice([500, 1000]),
        "speech_pad_ms": rng.choice([0, 30]),
        "max_speech_ms": rng.choice([1000, 1500, 2048, 3000]),
        "min_silence_at_max_speech_ms": rng.choice([0, 32, 64, 98]),
        "use_max_possible_silence": rng.choice([True, True, False]),
    }


# ms values whose sample counts (16 * ms) are multiples of the 512-sample
# frame, so position differences can land exactly on each bound.
ALIGNED = [32, 64, 96, 128]


def params(rng: random.Random) -> dict:
    p = {
        # 0.6 / 0.45 are not exact in float: they pin double thresholds.
        "threshold": rng.choice([0.5, 0.5, 0.25, 0.375, 0.75, 0.6]),
        "neg_threshold": rng.choice([None, None, 0.125, 0.25, 0.3125, 0.45]),
        "min_speech_ms": rng.choice([250, 0, 100, 1000] + ALIGNED),
        "min_silence_ms": rng.choice([100, 0, 50, 500] + ALIGNED),
        "speech_pad_ms": rng.choice([30, 0, 200, 16, 32, 48]),
        "max_speech_ms": rng.choice([0, 0, 1000, 2500, 5000, 1024, 2048, 1500]),
        "min_silence_at_max_speech_ms": rng.choice([98, 0, 300] + ALIGNED),
        "use_max_possible_silence": rng.choice([True, False]),
    }
    if p["neg_threshold"] is not None and p["neg_threshold"] > p["threshold"]:
        p["neg_threshold"] = None
    return p


def main() -> int:
    rng = random.Random(6_2_3)
    rows = []
    for v in range(N_VECTORS):
        n = rng.randint(20, 400)
        kind = v % 3
        q = (walk, bursts, long_speech)[kind](rng, n)
        probs = [k / LEVELS for k in q]
        n_samples = n * FRAME - rng.choice([0, 0, 1, rng.randint(0, FRAME - 1)])
        p = split_params(rng) if kind == 2 else params(rng)
        segs = get_speech_timestamps_from_probs(
            probs,
            sampling_rate=16000,
            threshold=p["threshold"],
            neg_threshold=p["neg_threshold"],
            min_speech_duration_ms=p["min_speech_ms"],
            min_silence_duration_ms=p["min_silence_ms"],
            speech_pad_ms=p["speech_pad_ms"],
            max_speech_duration_s=p["max_speech_ms"] / 1000 if p["max_speech_ms"] else float("inf"),
            min_silence_at_max_speech=p["min_silence_at_max_speech_ms"],
            use_max_poss_sil_at_max_speech=p["use_max_possible_silence"],
            audio_length_samples=n_samples,
        )
        rows.append((p, n_samples, q, segs))

    lines = [
        f"// Generated by scripts/vad/gen_segment_vectors.py from silero-vad {silero_vad.__version__}",
        "// get_speech_timestamps_from_probs. Do not edit.",
        "// {threshold, neg_threshold (-1 = default), min_speech_ms, min_silence_ms, speech_pad_ms,",
        "//  max_speech_ms, min_silence_at_max_speech_ms, use_max_possible_silence, n_samples,",
        "//  probs ('a' + 16 * p per frame), segments (start, end pairs)}",
    ]
    for p, n_samples, q, segs in rows:
        neg = -1.0 if p["neg_threshold"] is None else p["neg_threshold"]
        flat = [x for s in segs for x in (s["start"], s["end"])]
        lines.append(
            f"{{ {p['threshold']!r}, {neg!r}, {p['min_speech_ms']}, {p['min_silence_ms']}, {p['speech_pad_ms']}, "
            f"{p['max_speech_ms']}, {p['min_silence_at_max_speech_ms']}, "
            f"{'true' if p['use_max_possible_silence'] else 'false'}, {n_samples},\n"
            f"  \"{''.join(chr(ord('a') + k) for k in q)}\",\n"
            f"  {{ {', '.join(map(str, flat))} }} }},")
    OUT.write_text("\n".join(lines) + "\n")
    n_split = sum(1 for p, *_ in rows if p["max_speech_ms"])
    n_segs = sum(len(r[3]) for r in rows)
    print(f"wrote {OUT}: {len(rows)} vectors, {n_segs} segments, {n_split} with max_speech splitting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
