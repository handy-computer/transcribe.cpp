#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
run.py — transcribe-cli driver for WER evaluation.

Reads a manifest.jsonl, invokes transcribe-cli in --batch mode (model
loads once, processes all utterances in a single process), and writes
hypotheses + per-utterance timings to a report JSONL file.

Usage:
    uv run scripts/wer/run.py \\
        --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \\
        --manifest samples/wer/test-clean.manifest.jsonl

  Options:
    --out PATH      output report file (default: auto-derived from model name)
    --cli PATH      transcribe-cli binary (default: build/bin/transcribe-cli)

Output JSONL:
    - First line (batch header):
        {"type": "batch_header", "load_ms": ...}
      Captured once from transcribe-cli's --batch-jsonl header line. score.py
      and compare.py ignore this line.
    - Remaining lines (one per utterance):
        {"id": "...", "ref_text": "...", "hyp_text": "...", "mel_ms": ...,
         "encode_ms": ..., "decode_ms": ...}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def derive_out_path(repo: Path, model: Path, manifest: Path) -> Path:
    model_stem = model.stem
    dataset = manifest.stem.replace(".manifest", "")
    return repo / "reports" / "wer" / f"{model_stem}.{dataset}.jsonl"


def main() -> int:
    repo = find_repo_root(Path(__file__).parent)

    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="GGUF model file")
    p.add_argument("--manifest", type=Path,
                   default=repo / "samples/wer/test-clean.manifest.jsonl",
                   help="Input manifest")
    p.add_argument("--cli", type=Path,
                   default=repo / "build/bin/transcribe-cli",
                   help="transcribe-cli binary")
    p.add_argument("--out", type=Path, default=None,
                   help="Output report file (default: auto)")
    p.add_argument("--backend",
                   choices=("auto", "cpu", "metal", "vulkan"),
                   default=None,
                   help="Compute backend (default: transcribe-cli default)")
    args = p.parse_args()

    for path in (args.model, args.manifest, args.cli):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    out_path = args.out or derive_out_path(repo, args.model, args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest and build id→ref_text lookup.
    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    total = len(manifest)

    # Build audio→entry lookup keyed on audio path.
    audio_to_entry: dict[str, dict] = {}
    for e in manifest:
        audio_to_entry[e["audio"]] = e

    print(f"model:    {args.model}")
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {out_path}")
    print(f"mode:     batch (single process, model loads once)")

    # Write the batch file list (one wav path per line).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".list", delete=False
    ) as tf:
        batch_path = tf.name
        for e in manifest:
            tf.write(e["audio"] + "\n")

    # Invoke transcribe-cli in batch mode.
    cmd = [
        str(args.cli), "-q",
        "-m", str(args.model),
        "--batch", batch_path,
        "--batch-jsonl",
    ]
    if args.backend:
        cmd += ["--backend", args.backend]
    print(f"  $ {' '.join(cmd[:6])} ...")

    t_start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception as e:
        print(f"error: failed to start transcribe-cli: {e}", file=sys.stderr)
        return 1

    n_done = 0
    n_errors = 0

    with open(out_path, "w") as fout:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            # CLI emits a one-shot batch_header before any per-file line.
            # Persist it verbatim as the first record in the output JSONL.
            if result.get("type") == "batch_header":
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                continue

            audio_path = result.get("file", "")
            entry = audio_to_entry.get(audio_path, {})

            out_entry = {
                "id": entry.get("id", Path(audio_path).stem),
                "ref_text": entry.get("ref_text", ""),
                "hyp_text": result.get("text", ""),
                "mel_ms": result.get("mel_ms", 0),
                "encode_ms": result.get("encode_ms", 0),
                "decode_ms": result.get("decode_ms", 0),
                "error": result.get("error", ""),
            }
            # Compute total latency for backwards compat with score.py.
            out_entry["latency_ms"] = round(
                out_entry["mel_ms"] + out_entry["encode_ms"] +
                out_entry["decode_ms"], 1
            )
            if out_entry.get("error"):
                n_errors += 1

            fout.write(json.dumps(out_entry) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 200 == 0 or n_done == total:
                elapsed = time.monotonic() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] "
                      f"{rate:.1f} utt/s, ETA {eta:.0f}s, "
                      f"errors={n_errors}")

    proc.wait()
    wall = time.monotonic() - t_start

    # Clean up temp file.
    Path(batch_path).unlink(missing_ok=True)

    print(f"\ndone. {n_done} utterances in {wall:.1f}s "
          f"({n_done/wall:.1f} utt/s), {n_errors} errors")
    print(f"report: {out_path}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
