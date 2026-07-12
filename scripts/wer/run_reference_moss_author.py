#!/usr/bin/env python3
"""
run_reference_moss_author.py — MOSS-Transcribe-Diarize author-repo batch transcribe.

Loads MOSS-Transcribe-Diarize once via transformers trust_remote_code and runs
greedy decode over a WER manifest. Writes run.py-compatible JSONL so
scripts/wer/score.py can score it the same way it scores the C++ port.

The model always emits the canonical diarized+timestamped format
`[start][Sxx]text[end]`. For plain WER/CER scoring this runner de-diarizes the
hypothesis: every `[...]` bracket span (timestamps and speaker labels) is
metadata and is stripped, leaving the transcription text. This mirrors the
de-diarization the C++ WER path must apply.

Usage (from repo root):

    uv run --project scripts/envs/moss \\
      scripts/wer/run_reference_moss_author.py \\
        --model OpenMOSS-Team/MOSS-Transcribe-Diarize \\
        --revision d7231bbae2587a4af278735eb765b318c4f64edd \\
        --manifest samples/wer/librispeech-test-clean.manifest.jsonl \\
        --out reports/wer/moss-transcribe-diarize-REF.librispeech-test-clean.jsonl

Pass --device mps on Apple Silicon for a large speedup over CPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def dediarize(raw: str) -> str:
    return " ".join(re.sub(r"\[[^\]]*\]", " ", raw).split())


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text/language).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", required=True,
                   help="HF repo id (OpenMOSS-Team/MOSS-Transcribe-Diarize) or local dir.")
    p.add_argument("--revision", default=None,
                   help="HF revision to pin (ignored for local paths).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'mps' on Apple).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"],
                   help="Compute dtype (default: bf16, matches config).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--max-new-tokens", type=int, default=1024)
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
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
    print(f"loading: {args.model}  (transformers {transformers.__version__}, "
          f"device={args.device}, dtype={args.dtype})")
    t0 = time.monotonic()
    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
        dtype=dtype, attn_implementation="eager",
    ).eval().to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    # Build the prompt text once (audio value is irrelevant to template render;
    # the pcm is passed per-utterance to processor(audio=...)).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    n_done = n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "moss_author",
            "model": args.model,
            "prompt": "DEFAULT_PROMPT (timestamp+diarize, zh)",
            "dediarized": True,
        }) + "\n")
        fout.flush()

        for entry in manifest:
            uid = entry["id"]
            audio_path = entry["audio"]
            ref_text = entry.get("ref_text", "")

            t_start = time.monotonic()
            err = hyp_text = ""
            try:
                pcm, sr = sf.read(audio_path, dtype="float32")
                if pcm.ndim > 1:
                    pcm = pcm[:, 0]
                inputs = processor(text=prompt_text, audio=[pcm], return_tensors="pt")
                inputs = inputs.to(args.device)
                with torch.inference_mode():
                    gen = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        input_features=inputs["input_features"],
                        audio_feature_lengths=inputs["audio_feature_lengths"],
                        audio_chunk_mapping=inputs["audio_chunk_mapping"],
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False, num_beams=1,
                    )
                prompt_len = int(inputs["attention_mask"][0].sum().item())
                gen_ids = gen[0].detach().cpu().tolist()[prompt_len:]
                eos_id = processor.tokenizer.eos_token_id
                if eos_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos_id)]
                raw_text = processor.tokenizer.decode(
                    gen_ids, skip_special_tokens=True).strip()
                hyp_text = dediarize(raw_text)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            fout.write(json.dumps({
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text.strip(),
                "mel_ms": 0, "encode_ms": 0,
                "decode_ms": elapsed_ms, "latency_ms": elapsed_ms,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 25 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                      f"ETA {eta/60:.1f} min, errors={n_errors}", flush=True)

    wall = time.monotonic() - t_loop
    print(f"\ndone. {n_done} utterances in {wall:.1f}s "
          f"({n_done/wall:.2f} utt/s), {n_errors} errors")
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
