#!/usr/bin/env python3
"""
run_reference_gigaam_author.py — GigaAM author-repo batch transcribe.

Loads a GigaAM-v3 variant via `gigaam.load_model(variant, fp16_encoder=False)`
once and runs greedy decode over a WER manifest. Writes run.py-compatible
JSONL so scripts/wer/score.py can score the output the same way it scores
the C++ port's report.

Mirrors the dump-time setup (scripts/dump_reference_gigaam_author.py):
forces fp16_encoder=False so the encoder runs in fp32 on CPU.

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

# Same per-variant revision pins as the dumper. GigaAM-v3 keys run under
# scripts/envs/gigaam; multilingual_* keys run under
# scripts/envs/gigaam-multilingual (gigaam @559d88d registers them). The
# multilingual revision is the HF mirror pin, matching the golden manifest.
_VARIANT_REVISIONS = {
    "v3_e2e_rnnt": "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e",
    "v3_e2e_ctc":  "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e",
    "v3_rnnt":     "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e",
    "v3_ctc":      "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e",
    "multilingual_ctc": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
}


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
                   choices=sorted(_VARIANT_REVISIONS.keys()),
                   help="GigaAM variant key passed to gigaam.load_model.")
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

    import gigaam

    print(
        f"loading: gigaam.load_model({args.variant!r}, "
        f"fp16_encoder=False, device='cpu')",
        flush=True,
    )
    t0 = time.monotonic()
    model = gigaam.load_model(
        args.variant,
        fp16_encoder=False,
        device="cpu",
    )
    model.eval()
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
            "model_revision": _VARIANT_REVISIONS[args.variant],
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
