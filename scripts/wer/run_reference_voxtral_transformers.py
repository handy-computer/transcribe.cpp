#!/usr/bin/env python3
"""
run_reference_voxtral_transformers.py — Voxtral (2507) WER baseline.

Loads a Voxtral variant via the mainline HuggingFace Transformers
implementation (`VoxtralForConditionalGeneration`) and runs greedy decode
over a WER manifest using the dedicated transcription path
(`processor.apply_transcription_request`). Writes run.py-compatible JSONL
so scripts/wer/score.py scores it the same way it scores the C++ port.

The transcription request is the same path the dumper uses, so prompt
parity with the Oracle tensor dumps is automatic. Voxtral's recommended
transcription setting is temperature 0.0 (greedy), which we use here.

Uniform contract (--manifest/--model/--out/--device/--batch-size) so the
Modal reference_sweep drives every family the same way.

Usage (from repo root):

    uv run --project scripts/envs/voxtral \\
      scripts/wer/run_reference_voxtral_transformers.py \\
        --model mistralai/Voxtral-Mini-3B-2507 \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --out reports/wer/voxtral-mini-3b-2507-REF.test-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text/language).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", required=True,
                   help="HF repo id (mistralai/Voxtral-Mini-3B-2507) or local dir.")
    p.add_argument("--revision", default=None,
                   help="HF revision to pin (ignored for local paths).")
    p.add_argument("--language", default="en",
                   help="ISO 639-1 language hint for the transcription request "
                        "('auto' or empty lets Voxtral auto-detect). A manifest "
                        "entry's own 'language' field overrides this per-utterance.")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'cuda'/'mps').")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--max-new-tokens", type=int, default=440,
                   help="Cap per utterance. LibriSpeech utterances are short; "
                        "440 tokens (~33s of text) is generous.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Group N utterances per batched apply_transcription_request "
                        "+ generate() call. On any batch error it falls back to "
                        "per-utterance so a bad sample can't drop the group.")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    import transformers
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None
    model_id = str(Path(args.model).resolve()) if local_only else args.model

    print(f"loading: {args.model}  (transformers {transformers.__version__}, device={args.device})")
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision, local_files_only=local_only,
    )
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model, revision=revision, local_files_only=local_only,
        dtype=dtype, attn_implementation="eager",
    ).eval().to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    default_lang = None if args.language in (None, "", "auto") else args.language

    def lang_for(entry: dict):
        el = entry.get("language")
        if el in (None, "", "auto"):
            return default_lang
        return el

    def decode_new(ids: list[int]) -> str:
        eos_id = 2
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        tok = processor.tokenizer
        for attempt in (
            lambda: tok.decode(ids, skip_special_tokens=True),
            lambda: tok.decode(ids),
        ):
            try:
                return attempt().strip()
            except Exception:  # noqa: BLE001
                continue
        return ""

    def infer_one(entry: dict) -> str:
        inputs = processor.apply_transcription_request(
            language=lang_for(entry), audio=entry["audio"],
            model_id=model_id, return_tensors="pt",
        )
        inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1)
        seq = gen.sequences[0] if hasattr(gen, "sequences") else gen[0]
        return decode_new(seq.detach().cpu().tolist()[prompt_len:])

    def infer_batch(entries: list) -> list:
        # apply_transcription_request accepts lists of audio + language and
        # right-pads input_ids; for batched generate we need left padding,
        # so build per-row requests and left-pad here.
        rows = [processor.apply_transcription_request(
                    language=lang_for(e), audio=e["audio"],
                    model_id=model_id, return_tensors="pt")
                for e in entries]
        maxlen = max(int(r["input_ids"].shape[1]) for r in rows)
        pad_id = 11
        input_ids, attn, feats = [], [], []
        for r in rows:
            ids = r["input_ids"][0]
            n = maxlen - ids.shape[0]
            input_ids.append(torch.cat([torch.full((n,), pad_id, dtype=ids.dtype), ids]))
            attn.append(torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(ids.shape[0], dtype=torch.long)]))
            feats.append(r["input_features"])
        batch = {
            "input_ids": torch.stack(input_ids).to(args.device),
            "attention_mask": torch.stack(attn).to(args.device),
            "input_features": torch.cat(feats).to(args.device),
        }
        with torch.inference_mode():
            gen = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1)
        seqs = gen.sequences if hasattr(gen, "sequences") else gen
        return [decode_new(row.detach().cpu().tolist()[maxlen:]) for row in seqs]

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    n_done = n_errors = 0
    t_loop = time.monotonic()
    bs = max(1, args.batch_size)

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header", "load_ms": round(load_ms, 1),
            "framework": "transformers", "model": args.model,
            "language": args.language, "dtype": args.dtype,
        }) + "\n")
        fout.flush()

        for start in range(0, total, bs):
            group = manifest[start:start + bs]
            k = len(group)
            t_start = time.monotonic()
            hyps = [""] * k
            errs = [""] * k
            if bs == 1:
                try:
                    hyps[0] = infer_one(group[0])
                except Exception as e:  # noqa: BLE001
                    errs[0] = f"{type(e).__name__}: {e}"
                    n_errors += 1
            else:
                try:
                    hyps = infer_batch(group)
                except Exception:
                    for i, e_ in enumerate(group):
                        try:
                            hyps[i] = infer_one(e_)
                        except Exception as e2:  # noqa: BLE001
                            errs[i] = f"{type(e2).__name__}: {e2}"
                            n_errors += 1
            per_ms = round((time.monotonic() - t_start) * 1000 / k, 1)

            for i, entry in enumerate(group):
                fout.write(json.dumps({
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": (hyps[i] or "").strip(),
                    "mel_ms": 0, "encode_ms": 0,
                    "decode_ms": per_ms, "latency_ms": per_ms,
                    "error": errs[i],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1

            if start // bs % 10 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                      f"ETA {eta/60:.1f} min, errors={n_errors}", flush=True)

    wall = time.monotonic() - t_loop
    print(f"\ndone. {n_done} utterances in {wall:.1f}s "
          f"({n_done / wall:.2f} utt/s), {n_errors} errors")
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
