#!/usr/bin/env python3
"""
run_reference_sensevoice.py — FunASR SenseVoiceSmall batch transcribe.

Loads SenseVoiceSmall once via funasr.AutoModel and runs it over a
WER manifest, writing run.py-compatible JSONL so scripts/wer/score.py
can score the output. Mirrors the dump-time setup
(scripts/dump_reference_sensevoice_funasr.py): pinned revision,
dither=0.0, CPU device by default.

Usage (from repo root):

    uv run --project scripts/envs/sensevoice \\
      scripts/wer/run_reference_sensevoice.py \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --out      reports/wer/sensevoice-small-REF.test-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


CONTROL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def strip_control_tokens(text: str) -> str:
    return CONTROL_TOKEN_RE.sub("", text).strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", default="FunAudioLLM/SenseVoiceSmall",
                   help="HF repo id (default: FunAudioLLM/SenseVoiceSmall)")
    p.add_argument("--revision",
                   default="3eb3b4eeffc2f2dde6051b853983753db33e35c3",
                   help="HF revision pin (matches Stage 2 oracle).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu — matches oracle).")
    p.add_argument("--language", default="en",
                   help="Language hint (default: en for LibriSpeech).")
    p.add_argument("--use-itn", action="store_true",
                   help="Enable inverse text normalization (default: off).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = leave default).")
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

    from funasr import AutoModel

    print(
        f"loading: model={args.model} revision={args.revision} "
        f"device={args.device}"
    )
    auto = AutoModel(
        model=args.model,
        model_revision=args.revision,
        hub="hf",
        device=args.device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        dither=0.0,
    )
    # See dump_reference_sensevoice_funasr.py: dither= kwarg is silently
    # consumed by AutoModel, must override the frontend attribute.
    auto.kwargs["frontend"].dither = 0.0

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")
    print(f"language: {args.language}  use_itn: {args.use_itn}")

    t0 = time.monotonic()
    n_done = 0
    n_errors = 0

    with open(args.out, "w") as fout:
        # Header (run.py-compatible).
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round((time.monotonic() - t0) * 1000, 1),
            "framework": "funasr",
            "model": args.model,
            "revision": args.revision,
            "device": args.device,
            "language": args.language,
            "use_itn": bool(args.use_itn),
        }) + "\n")
        fout.flush()

        t_loop = time.monotonic()
        for entry in manifest:
            audio_path = entry["audio"]
            uid = entry["id"]
            ref_text = entry.get("ref_text", "")

            t_start = time.monotonic()
            err = ""
            raw_text = ""
            try:
                res = auto.generate(
                    input=audio_path,
                    language=args.language,
                    use_itn=args.use_itn,
                    cache={},
                )
                if isinstance(res, list) and res:
                    raw_text = res[0].get("text", "")
                else:
                    err = "empty result"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            hyp_text = strip_control_tokens(raw_text)
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text,
                "raw_text": raw_text,
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "error": err,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 50 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{total}] "
                    f"{rate:.2f} utt/s, ETA {eta/60:.1f} min, "
                    f"errors={n_errors}",
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
