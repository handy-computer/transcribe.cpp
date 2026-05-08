#!/usr/bin/env python3
"""
run_reference_canary_nemo.py — NeMo Canary multitask AED batch transcribe.

Loads an EncDecMultiTaskModel once and runs it over a WER manifest,
writing run.py-compatible JSONL so scripts/wer/score.py can score the
output. Mirrors the dump-time setup (scripts/dump_reference_canary_nemo.py):
pinned variant, beam=1 greedy, ASR task with explicit source_lang ==
target_lang.

Usage (from repo root):

    uv run --project scripts/envs/canary \\
      scripts/wer/run_reference_canary_nemo.py \\
        --model nvidia/canary-180m-flash \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --out      reports/wer/canary-180m-flash-REF.test-clean-512.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", default="nvidia/canary-180m-flash",
                   help="HF repo id for the EncDecMultiTaskModel checkpoint.")
    p.add_argument("--language", default="en",
                   help="Source/target language code (ASR mode; default: en).")
    p.add_argument("--task", default="asr",
                   choices=["asr", "ast", "s2t_translation"])
    p.add_argument("--pnc", default="yes", choices=["yes", "no"])
    p.add_argument("--toggle-timestamps", default="no",
                   choices=["yes", "no", "none"])
    p.add_argument("--beam", type=int, default=1,
                   help="Beam size (1 = greedy; the model card numbers use 1 for canary-1b-flash and 5 for canary-1b).")
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

    print(f"loading: {args.model}")
    t0 = time.monotonic()
    from nemo.collections.asr.models import EncDecMultiTaskModel
    model = EncDecMultiTaskModel.from_pretrained(args.model)
    model.eval()

    if args.beam > 0:
        try:
            cfg = model.cfg.decoding
            cfg.beam.beam_size = args.beam
            model.change_decoding_strategy(cfg)
        except Exception as exc:
            print(f"warning: failed to set beam_size={args.beam}: {exc}")

    load_ms = (time.monotonic() - t0) * 1000

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")
    print(f"language: {args.language}  task: {args.task}  pnc: {args.pnc}  beam: {args.beam}")

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "nemo",
            "model": args.model,
            "task": args.task,
            "language": args.language,
            "pnc": args.pnc,
            "beam": args.beam,
        }) + "\n")
        fout.flush()

        # NeMo's canary transcribe() expects a manifest file with
        # source_lang / target_lang / task / pnc / answer fields per row.
        # We iterate one utterance at a time (deterministic, easy to log
        # progress) and feed it via a single-row temp manifest, mirroring
        # the dumper's approach.
        for entry in manifest:
            uid = entry["id"]
            audio_path = entry["audio"]
            ref_text = entry.get("ref_text", "")

            t_start = time.monotonic()
            err = ""
            hyp_text = ""
            try:
                row = {
                    "audio_filepath": audio_path,
                    "duration": entry.get("duration", 0.0) or 0.0,
                    "source_lang": args.language,
                    "target_lang": args.language,
                    "task": args.task,
                    "taskname": args.task,  # canary-1 reads "taskname"
                    "pnc": args.pnc,
                    "answer": "na",
                }
                if args.toggle_timestamps in ("yes", "no"):
                    row["timestamp"] = args.toggle_timestamps

                with tempfile.NamedTemporaryFile(
                    "w", suffix=".jsonl", delete=False, encoding="utf-8"
                ) as fh:
                    fh.write(json.dumps(row) + "\n")
                    tmp_path = Path(fh.name)
                try:
                    out = model.transcribe(str(tmp_path), batch_size=1)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

                if isinstance(out, (list, tuple)) and out:
                    first = out[0]
                    if isinstance(first, (list, tuple)):
                        first = first[0]
                    if hasattr(first, "text"):
                        hyp_text = first.text
                    elif isinstance(first, str):
                        hyp_text = first
                    else:
                        hyp_text = str(first)
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

            if n_done % 50 == 0 or n_done == total:
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
