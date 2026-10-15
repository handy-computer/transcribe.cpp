#!/usr/bin/env python3
"""Scoreable whole-clip Phonon-2 publisher reference hypotheses.

Run with uv run --project scripts/envs/parakeet-phonon. Uses the independently
hash-checked publisher loader and stock Transformers greedy generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dump_reference_parakeet_phonon import load
from lib.phonon import BASE_REVISION, REPO, REVISION


def main() -> int:
    import soundfile as sf
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        parser.error("manifest must contain unique utterance IDs")
    if any(row.get("language", "en") != "en" for row in rows):
        parser.error("Phonon-2 reference requires English inputs")
    model, processor, info = load(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with args.out.open("x") as out, torch.inference_mode():
        out.write(json.dumps({
            "type": "batch_header", "framework": "transformers",
            "framework_version": version("transformers"), "torch_version": version("torch"),
            "model": args.model, "model_revision": args.revision,
            "base_revision": BASE_REVISION, "device": "cpu", "dtype": "float32",
            "batch_size": 1, "language": "en", "torch_threads": args.torch_threads,
            "reference_load": info,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "recipe": {"decode": "greedy", "timestamps": "none", "segmentation": "whole-clip",
                       "batch_size": 1, "language": "en", "backend": "cpu"},
        }) + "\n")
        out.flush()
        for index, row in enumerate(rows, 1):
            pcm, sr = sf.read(row["audio"], dtype="float32")
            if sr != 16000 or pcm.ndim != 1:
                raise ValueError(f"expected mono 16 kHz audio: {row['audio']}")
            tick = time.monotonic()
            inputs = processor([pcm], sampling_rate=sr, return_tensors="pt", padding=True)
            generated = model.generate(**inputs)
            sequences = getattr(generated, "sequences", generated)
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            out.write(json.dumps({
                "id": row["id"], "ref_text": row["ref_text"], "hyp_text": text,
                "latency_ms": (time.monotonic() - tick) * 1000, "error": "",
            }) + "\n")
            out.flush()
            if index % 25 == 0 or index == len(rows):
                elapsed = time.monotonic() - start
                print(f"[{index}/{len(rows)}] {elapsed:.1f}s; "
                      f"ETA {(len(rows)-index)*elapsed/index/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
