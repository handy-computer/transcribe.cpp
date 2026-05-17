#!/usr/bin/env python3
"""
run_reference_canary_qwen_nemo.py — NeMo SALM (canary-qwen) batch transcribe.

Loads `nemo.collections.speechlm2.models.SALM` once and runs it over a WER
manifest, writing run.py-compatible JSONL so scripts/wer/score.py can
score the output. Mirrors the dump-time setup
(scripts/dump_reference_canary_qwen_nemo.py): SALM chat-style prompt with
the audio_locator placeholder, greedy generate(), tokenizer.ids_to_text
for the decode.

Usage (from repo root):

    uv run --project scripts/envs/canary_qwen \\
      scripts/wer/run_reference_canary_qwen_nemo.py \\
        --model    nvidia/canary-qwen-2.5b \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --out      reports/wer/canary-qwen-2.5b-REF.test-clean-512.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument("--model", default="nvidia/canary-qwen-2.5b",
                   help="HF repo id for the SALM checkpoint.")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Per-utterance generation cap (matches the model card example).")
    p.add_argument("--prompt-template",
                   default="Transcribe the following: {locator}",
                   help="Chat-content template; {locator} is replaced with the model's audio_locator_tag.")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads for deterministic runs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--device", default="cpu",
                   help="torch device (cpu / cuda / mps). Default: cpu for determinism.")
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
    import nemo.collections.speechlm2 as slm
    model = slm.SALM.from_pretrained(
        args.model, map_location=args.device, strict=True,
    )
    model.eval()

    # Match the dump-time determinism overrides: zero out preprocessor
    # dither so repeat runs produce identical mel inputs (the SALM cfg
    # declares 1e-5 by default which adds tiny per-run noise).
    pre = getattr(model, "perception", None)
    pre = getattr(pre, "preprocessor", None) if pre is not None else None
    if pre is not None and hasattr(pre, "featurizer") and hasattr(pre.featurizer, "dither"):
        prev = float(pre.featurizer.dither)
        if prev != 0.0:
            print(f"  overriding preprocessor dither {prev} -> 0.0 for deterministic runs")
            pre.featurizer.dither = 0.0

    locator = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
    content_template = args.prompt_template
    if "{locator}" not in content_template:
        # Append a locator if the user gave a template without one;
        # SALM's audio_locator scatter requires at least one placeholder.
        content_template = content_template.rstrip() + " " + locator

    load_ms = (time.monotonic() - t0) * 1000

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")
    print(f"prompt:   '{content_template.format(locator=locator)}'")
    print(f"max_new_tokens: {args.max_new_tokens}  device: {args.device}")

    tokenizer = getattr(model, "tokenizer", None)

    def decode_ids(ids) -> str:
        if hasattr(ids, "cpu"):
            ids = ids.detach().cpu().tolist()
        elif not isinstance(ids, list):
            ids = list(ids)
        if tokenizer is not None and hasattr(tokenizer, "ids_to_text"):
            try:
                return tokenizer.ids_to_text([int(x) for x in ids]).strip()
            except Exception as exc:
                print(f"  warning: ids_to_text failed: {exc}")
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode([int(x) for x in ids],
                                        skip_special_tokens=True).strip()
            except Exception as exc:
                print(f"  warning: tokenizer.decode failed: {exc}")
        return ""

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "nemo-salm",
            "model": args.model,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "prompt_template": content_template,
            "audio_locator": locator,
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
                content = content_template.format(locator=locator)
                prompts = [[{
                    "role": "user",
                    "content": content,
                    "audio": [audio_path],
                }]]
                with torch.inference_mode():
                    answer_ids = model.generate(
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                    )

                # answer_ids: tensor [B, T] or list-of-lists.
                if hasattr(answer_ids, "cpu"):
                    row_ids = answer_ids[0]
                elif isinstance(answer_ids, (list, tuple)) and answer_ids:
                    row_ids = answer_ids[0]
                else:
                    row_ids = []
                hyp_text = decode_ids(row_ids)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text,
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
