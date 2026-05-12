#!/usr/bin/env python3
"""
run_reference_qwen3_asr_author.py — Qwen3-ASR author-repo batch transcribe.

Loads Qwen3-ASR once via the first-party qwen_asr package (registered into
the transformers Auto* registries) and runs greedy decode over a WER
manifest. Writes run.py-compatible JSONL so scripts/wer/score.py can score
the output the same way it scores the C++ port's report.

This script mirrors the prompt + post-processing of the C++ runtime:
the prompt template is `apply_chat_template` plus `language X<asr_text>`
when --language is set, and the emitted text is the suffix after
`<asr_text>` (matching src/arch/qwen3_asr/model.cpp).

Usage (from repo root):

    uv run --project scripts/envs/qwen3_asr \\
      scripts/wer/run_reference_qwen3_asr_author.py \\
        --model Qwen/Qwen3-ASR-0.6B \\
        --manifest samples/wer/fleurs-zh.manifest.jsonl \\
        --language zh \\
        --out      reports/wer/Qwen3-ASR-0.6B-REF.fleurs-zh.jsonl

Pass --device mps on Apple Silicon for ~5x speedup over CPU.
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
                   help="HF repo id (Qwen/Qwen3-ASR-0.6B) or local directory.")
    p.add_argument("--revision", default=None,
                   help="HF revision to pin (ignored for local paths).")
    p.add_argument("--language", default=None,
                   help="BCP-47 / language-name hint suffixed onto the "
                        "prompt as 'language X<asr_text>'. If unset, the "
                        "model auto-detects.")
    p.add_argument("--context", default="",
                   help="System-prompt context string (default: empty).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'mps' on Apple).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--max-new-tokens", type=int, default=256)
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

    import soundfile as sf
    import transformers
    from transformers import AutoConfig, AutoModel, AutoProcessor
    # Side-effect registration: wires Qwen3ASR classes into Auto* registries.
    import qwen_asr.inference.qwen3_asr  # noqa: F401

    print(
        f"loading: {args.model}  (transformers {transformers.__version__}, "
        f"device={args.device})"
    )
    t0 = time.monotonic()
    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None
    config = AutoConfig.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
    )
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
        fix_mistral_regex=True,
    )
    model = AutoModel.from_pretrained(
        args.model, config=config, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    # Build the prompt once. Same prompt for every utterance in the batch
    # since --language is run-level.
    messages = [
        {"role": "system", "content": args.context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    if args.language:
        prompt_text = prompt_text + f"language {args.language}<asr_text>"

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"language: {args.language or '(auto)'}")
    print(f"output:   {args.out}")

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()
    sep = "<asr_text>"

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "qwen_asr",
            "model": args.model,
            "language": args.language,
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
                pcm, sr = sf.read(audio_path, dtype="float32")
                if pcm.ndim > 1:
                    pcm = pcm[:, 0]
                inputs = processor(
                    text=[prompt_text], audio=[pcm],
                    return_tensors="pt", padding=True,
                )
                inputs = inputs.to(args.device).to(model.dtype)
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False, num_beams=1,
                    )
                token_ids = gen.sequences[0].detach().cpu().tolist()
                prompt_ids = inputs.get("input_ids")
                if prompt_ids is not None:
                    prompt_len = int(prompt_ids.shape[1])
                    if (len(token_ids) > prompt_len
                            and token_ids[:prompt_len]
                                == prompt_ids[0].detach().cpu().tolist()):
                        token_ids = token_ids[prompt_len:]
                eos_id = processor.tokenizer.eos_token_id
                if eos_id in token_ids:
                    token_ids = token_ids[:token_ids.index(eos_id)]
                raw_text = processor.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).strip()
                # Strip the `language X<asr_text>` prefix if the model
                # echoed it. The user-facing transcript is the suffix
                # after the separator.
                if sep in raw_text:
                    hyp_text = raw_text.split(sep, 1)[1]
                else:
                    hyp_text = raw_text
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text.strip(),
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "error": err,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
