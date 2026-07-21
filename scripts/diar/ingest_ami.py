#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.6",
#     "numpy>=1.26",
#     "soundfile>=0.12",
# ]
# ///
"""
ingest_ami.py - build an AMI diarization manifest from diarizers-community/ami.

Diarization analog of scripts/wer/ingest.py. Sortformer has no transcript,
so the acceptance corpus needs multi-speaker audio + ground-truth speaker
RTTMs, not reference text. This pulls the AMI corpus from the HF dataset
diarizers-community/ami (ungated), which ships one full meeting per row with
`audio` + parallel `timestamps_start` / `timestamps_end` / `speakers` lists.

Config `ihm` = mixed-headset audio (NVIDIA's "AMI Test IHM" condition);
`sdm` = single distant mic.

Output (mirrors the wer/ layout):
  samples/diar/ami-<config>-<split>/<uri>.wav     16-bit PCM mono 16 kHz
  samples/diar/ami-<config>-<split>/<uri>.rttm    ground-truth speaker turns
  samples/diar/ami-<config>-<split>.manifest.jsonl
      {"id","audio","rttm","duration","num_speakers"}

Usage:
  uv run scripts/diar/ingest_ami.py --config ihm --split test
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
REPO = Path(__file__).resolve().parent.parent.parent


def _uri(path: str) -> str:
    # "EN2002b.Mix-Headset.wav" -> "EN2002b.Mix-Headset"
    return Path(path).name.rsplit(".wav", 1)[0]


def _decode_16k_mono(raw: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio.astype(np.float32)


def _write_rttm(path: Path, uri: str, starts, ends, speakers) -> int:
    lines = []
    for s, e, spk in zip(starts, ends, speakers):
        dur = float(e) - float(s)
        if dur <= 0:
            continue
        lines.append(f"SPEAKER {uri} 1 {float(s):.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="ihm", choices=["ihm", "sdm"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset, Audio

    ds_id = f"ami-{args.config}-{args.split}"
    out_dir = REPO / "samples" / "diar" / ds_id
    manifest = REPO / "samples" / "diar" / f"{ds_id}.manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and not args.force:
        n = sum(1 for _ in open(manifest))
        print(f"OK already exists: {manifest} ({n} meetings). Use --force to rebuild.")
        return 0

    print(f"loading diarizers-community/ami [{args.config}/{args.split}] ...", flush=True)
    ds = load_dataset("diarizers-community/ami", args.config, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))

    entries = []
    total_dur = 0.0
    for i, ex in enumerate(ds):
        uri = _uri(ex["audio"]["path"] or f"{ds_id}-{i:03d}")
        audio = _decode_16k_mono(ex["audio"]["bytes"])
        wav = out_dir / f"{uri}.wav"
        rttm = out_dir / f"{uri}.rttm"
        sf.write(str(wav), audio, SR, subtype="PCM_16")
        n_turns = _write_rttm(rttm, uri, ex["timestamps_start"], ex["timestamps_end"], ex["speakers"])
        dur = len(audio) / SR
        total_dur += dur
        n_spk = len(set(ex["speakers"]))
        entries.append({
            "id": uri,
            "audio": str(wav.relative_to(REPO)),
            "rttm": str(rttm.relative_to(REPO)),
            "duration": round(dur, 2),
            "num_speakers": n_spk,
        })
        print(f"  [{i}] {uri}: {dur/60:.1f}min, {n_spk} spk, {n_turns} turns", flush=True)

    with open(manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"\nmanifest: {manifest.relative_to(REPO)}")
    print(f"{len(entries)} meetings, {total_dur/60:.1f} min total audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
