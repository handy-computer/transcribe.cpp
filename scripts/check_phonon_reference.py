#!/usr/bin/env python3
"""Paired whole-clip reference/native checks on a WAV list (not a WER benchmark).

Run in scripts/envs/parakeet-phonon with --model <source bundle>,
--gguf <converted file>, --list <WAV list>, --out <report.json>.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dump_reference_parakeet_phonon import load
from lib.phonon import REPO, REVISION


def main() -> int:
    import soundfile as sf
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cli", default="build/bin/transcribe-cli")
    args = parser.parse_args()
    files = [line.strip() for line in Path(args.list).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not files:
        parser.error("WAV list is empty")
    command = [args.cli, "-m", args.gguf, "--backend", "cpu", "--threads", "1",
               "--batch", args.list, "--batch-jsonl", "-q"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    native_rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    native_rows = [row for row in native_rows if row.get("type") != "batch_header"]
    native = {row["file"]: row for row in native_rows}
    if set(native) != set(files):
        raise ValueError("native runner did not return every input")
    model, processor, info = load(args)
    rows = []
    with torch.inference_mode():
        for filename in files:
            pcm, sr = sf.read(filename, dtype="float32")
            if sr != 16000 or pcm.ndim != 1:
                raise ValueError(f"expected 16 kHz mono: {filename}")
            inputs = processor([pcm], sampling_rate=sr, return_tensors="pt", padding=True)
            generated = model.generate(**inputs)
            sequences = getattr(generated, "sequences", generated)
            reference = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            cpp = native[filename]["text"]
            row = {"file": filename, "duration_s": len(pcm) / sr, "reference": reference,
                   "native": cpp, "match": reference == cpp}
            rows.append(row)
            print(f"{'PASS' if row['match'] else 'FAIL'} {filename}: {reference!r}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reference_load": info, "revision": REVISION, "rows": rows,
                               "passed": all(row["match"] for row in rows)}, indent=2) + "\n")
    return 0 if all(row["match"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
