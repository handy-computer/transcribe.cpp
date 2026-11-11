#!/usr/bin/env python3
"""
run_reference_gigaam_author.py — GigaAM author-repo batch transcribe.

Loads a GigaAM variant once through scripts/lib/gigaam_hf.py, the loader
the reference dumper uses, and runs greedy decode over a WER manifest.
Writes run.py-compatible JSONL so scripts/wer/score.py can score the
output the same way it scores the C++ port's report.

The model runs in F32 on CPU, as in the reference dumps.

Usage (from repo root):

    uv run --project scripts/envs/gigaam \\
      scripts/wer/run_reference_gigaam_author.py \\
        --variant v3_e2e_rnnt \\
        --manifest samples/wer/fleurs-ru.512.manifest.jsonl \\
        --out      reports/wer/gigaam-v3-e2e-rnnt-REF.fleurs-ru-512.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo scripts/
from lib.gigaam_hf import VARIANTS, load_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--variant", required=True,
                   choices=sorted(VARIANTS),
                   help="GigaAM variant key of scripts/lib/gigaam_hf.py VARIANTS.")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads for deterministic runs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    t0 = time.monotonic()
    model = load_model(args.variant)
    load_ms = (time.monotonic() - t0) * 1000

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "gigaam-author",
            "model": args.variant,
            "model_revision": VARIANTS[args.variant].revision,
        }) + "\n")
        fout.flush()

        for entry in manifest:
            uid = entry["id"]
            audio_path = entry["audio"]
            ref_text = entry.get("ref_text", "")

            t_start = time.monotonic()
            err = ""
            hyp_text = ""
            try:
                result = model.transcribe(audio_path)
                hyp_text = result.text if hasattr(result, "text") else str(result)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text.strip(),
                "raw_text": hyp_text,
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "error": err,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 25 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                    f"ETA {eta/60:.1f} min, errors={n_errors}",
                    flush=True,
                )

    wall = time.monotonic() - t_loop
    print(
        f"\ndone. {n_done} utterances in {wall:.1f}s "
        f"({n_done / wall:.2f} utt/s), {n_errors} errors"
    )
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
