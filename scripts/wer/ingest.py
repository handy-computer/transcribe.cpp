#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "soundfile>=0.12",
# ]
# ///
"""
ingest.py - Build a manifest + wav files from a LibriSpeech split.

Walks the extracted LibriSpeech directory tree, decodes each .flac to
16-bit PCM mono 16 kHz wav (the format transcribe-cli expects), and
writes a one-line-per-utterance manifest.jsonl linking each wav to its
reference text.

Usage:
    uv run scripts/wer/ingest.py

  Defaults:
    --raw     samples/wer/raw/LibriSpeech/test-clean
    --out-dir samples/wer/test-clean
    --manifest samples/wer/test-clean.manifest.jsonl

The script is idempotent: existing wav files are skipped (by checking
file existence, not content hash — fast enough for a one-shot ingest).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def write_wav_16k_mono(data: np.ndarray, sr: int, out_path: Path) -> None:
    """Write 16-bit PCM mono wav at 16 kHz. Resamples if sr != 16000
    via simple linear interpolation (good enough for ASR eval — all
    LibriSpeech splits are already 16 kHz so this is a no-op path
    in practice)."""
    if sr != 16000:
        # Crude resample via linear interp. LibriSpeech is 16 kHz
        # so this should never trigger.
        n_out = int(len(data) * 16000 / sr)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, n_out)
        data = np.interp(x_new, x_old, data)
        sr = 16000

    # Convert float64/float32 → int16 range.
    if data.dtype in (np.float32, np.float64):
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)

    sf.write(str(out_path), data, sr, subtype="PCM_16", format="WAV")


def main() -> int:
    repo = find_repo_root(Path(__file__).parent)

    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path,
                   default=repo / "samples/wer/raw/LibriSpeech/test-clean",
                   help="Extracted LibriSpeech split directory")
    p.add_argument("--out-dir", type=Path,
                   default=repo / "samples/wer/test-clean",
                   help="Where to write converted wav files")
    p.add_argument("--manifest", type=Path,
                   default=repo / "samples/wer/test-clean.manifest.jsonl",
                   help="Output manifest path")
    args = p.parse_args()

    if not args.raw.is_dir():
        print(f"error: {args.raw} does not exist. Run scripts/wer/setup.sh first.",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # LibriSpeech layout:
    #   <split>/<spk>/<chap>/<spk>-<chap>-<utt>.flac
    #   <split>/<spk>/<chap>/<spk>-<chap>.trans.txt
    #
    # trans.txt has one line per utterance:
    #   <spk>-<chap>-<utt> UPPERCASE TEXT ...
    #
    # We walk trans.txt files (one per chapter) for the authoritative
    # utterance list, then decode the corresponding flac.

    trans_files = sorted(args.raw.rglob("*.trans.txt"))
    if not trans_files:
        print(f"error: no .trans.txt files found in {args.raw}", file=sys.stderr)
        return 2

    entries: list[dict] = []
    n_converted = 0
    n_skipped = 0

    for tf in trans_files:
        chap_dir = tf.parent
        for line in tf.read_text().strip().splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, ref_text = parts
            flac_path = chap_dir / f"{utt_id}.flac"
            if not flac_path.exists():
                print(f"warning: {flac_path} not found, skipping",
                      file=sys.stderr)
                continue

            wav_path = args.out_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                data, sr = sf.read(str(flac_path), dtype="float32")
                # LibriSpeech is mono but guard against multi-channel.
                if data.ndim > 1:
                    data = data[:, 0]
                write_wav_16k_mono(data, sr, wav_path)
                n_converted += 1
            else:
                n_skipped += 1

            entries.append({
                "id": utt_id,
                "audio": str(wav_path),
                "ref_text": ref_text,
            })

    # Sort by id for deterministic ordering.
    entries.sort(key=lambda e: e["id"])

    with open(args.manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"manifest: {args.manifest}")
    print(f"  {len(entries)} utterances")
    print(f"  {n_converted} converted, {n_skipped} skipped (already existed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
