#!/usr/bin/env python3
"""
run_reference_cohere_transformers.py — Cohere ASR WER baseline.

Loads a Cohere Transcribe variant via native Hugging Face Transformers
(`CohereAsrForConditionalGeneration`, `trust_remote_code=False` — the
remote-code path is known-broken, see the family doc and HF discussion
#28 on cohere-transcribe-03-2026) and runs greedy decode over a WER
manifest. Writes run.py-compatible JSONL so scripts/wer/score.py can
score the output the same way it scores the C++ port's report.

Language comes from each manifest entry's `language` field (ingest.py
writes it), overridable globally with --language. The processor builds
the 10-token decoder prompt from the language, so no prompt text is
constructed here.

Usage (from repo root):

    uv run --project scripts/envs/cohere \\
      scripts/wer/run_reference_cohere_transformers.py \\
        --model CohereLabs/cohere-transcribe-arabic-07-2026 \\
        --manifest samples/wer/fleurs-ar.manifest.jsonl \\
        --out reports/wer/cohere-transcribe-arabic-07-2026-REF.fleurs-ar.jsonl
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
                   help="Input manifest JSONL (id/audio/ref_text/language).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", required=True,
                   help="HF repo id (CohereLabs/cohere-transcribe-03-2026) "
                        "or local directory.")
    p.add_argument("--revision", default=None,
                   help="HF revision to pin (ignored for local paths).")
    p.add_argument("--language", default=None,
                   help="Force one language for every utterance. Default: "
                        "each entry's manifest `language` field, else 'en'.")
    p.add_argument("--no-punctuation", action="store_true",
                   help="Use the <|nopnc|> prompt slot instead of <|pnc|>.")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; 'cuda' on Modal).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dtype", default="bf16",
                   choices=["bf16", "f16", "f32"],
                   help="Model dtype (default bf16 = checkpoint dtype).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Group N utterances per batched generate() call. "
                        ">1 batches audio through the processor; on any batch "
                        "error it falls back to per-utterance so a bad sample "
                        "can't drop the group. Uniform across reference "
                        "runners so the Modal reference_sweep drives every "
                        "family the same way.")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    import soundfile as sf
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None

    print(
        f"loading: {args.model}  (transformers {transformers.__version__}, "
        f"device={args.device}, dtype={args.dtype})"
    )
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
    )
    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
        dtype=dtype,
    ).eval().to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    conv_dtype = model.model.encoder.subsampling.layers[0].weight.dtype

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    def entry_language(entry: dict) -> str:
        return args.language or entry.get("language") or "en"

    def load_pcm(audio_path: str):
        pcm, sr = sf.read(audio_path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        if sr != 16000:
            raise RuntimeError(f"cohere_asr expects 16kHz; got {sr}Hz")
        return pcm

    # Shared generate path: the processor emits input_features plus the
    # language-conditioned decoder_input_ids prompt (same 10-token length
    # for every row, so one slice drops the prompt for the whole batch).
    def infer_group(entries: list) -> list:
        language = entry_language(entries[0])
        if any(entry_language(e_) != language for e_ in entries):
            raise RuntimeError("mixed languages in one batch group")
        pcms = [load_pcm(e_["audio"]) for e_ in entries]
        inputs = processor(
            audio=pcms,
            language=language,
            punctuation=not args.no_punctuation,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: (v.to(args.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        if inputs["input_features"].dtype != conv_dtype:
            inputs["input_features"] = inputs["input_features"].to(conv_dtype)
        prompt_len = int(inputs["decoder_input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1)
        seqs = gen.sequences if hasattr(gen, "sequences") else gen
        eos_id = processor.tokenizer.eos_token_id
        out = []
        for row in seqs:
            ids = row.detach().cpu().tolist()[prompt_len:]
            if eos_id is not None and eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            out.append(processor.tokenizer.decode(
                ids, skip_special_tokens=True).strip())
        return out

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "transformers",
            "model": args.model,
            "language": args.language,
            "dtype": args.dtype,
        }) + "\n")
        fout.flush()

        bs = max(1, args.batch_size)
        for start in range(0, total, bs):
            group = manifest[start:start + bs]
            k = len(group)
            t_start = time.monotonic()
            hyps = [""] * k
            errs = [""] * k
            try:
                hyps = infer_group(group)
            except Exception:
                if k == 1:
                    e = sys.exc_info()[1]
                    errs[0] = f"{type(e).__name__}: {e}"
                    n_errors += 1
                else:
                    # Fall back to per-utterance for this group.
                    for i, e_ in enumerate(group):
                        try:
                            hyps[i] = infer_group([e_])[0]
                        except Exception as e2:
                            errs[i] = f"{type(e2).__name__}: {e2}"
                            n_errors += 1
            per_ms = round((time.monotonic() - t_start) * 1000 / k, 1)

            for i, entry in enumerate(group):
                rec = {
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": (hyps[i] or "").strip(),
                    "mel_ms": 0,
                    "encode_ms": 0,
                    "decode_ms": per_ms,
                    "latency_ms": per_ms,
                    "error": errs[i],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1

            if start // bs % 10 == 0 or n_done == total:
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
