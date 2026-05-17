#!/usr/bin/env python3
"""
run_reference_granite_transformers.py — IBM Granite Speech WER baseline.

Loads a granite-speech variant via the mainline HuggingFace Transformers
implementation (`GraniteSpeechForConditionalGeneration` /
`GraniteSpeechPlusForConditionalGeneration`) and runs greedy decode over
a WER manifest. Writes run.py-compatible JSONL so scripts/wer/score.py
can score the output the same way it scores the C++ port's report.

Prompt mirrors the C++ runtime + the dump-reference script:

    USER: <|audio|>{instruction}\\n ASSISTANT:

(no chat-template variant). The plus variant inherits its own chat
template; this script defers to processor.apply_chat_template when one
is present so prompt parity with the dumper is automatic.

Usage (from repo root):

    uv run --project scripts/envs/granite \\
      scripts/wer/run_reference_granite_transformers.py \\
        --model ibm-granite/granite-4.0-1b-speech \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --out reports/wer/granite-4.0-1b-speech-REF.test-clean.jsonl
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
                   help="HF repo id (ibm-granite/granite-4.0-1b-speech) "
                        "or local directory.")
    p.add_argument("--revision", default=None,
                   help="HF revision to pin (ignored for local paths).")
    p.add_argument(
        "--instruction",
        default=None,
        help="User instruction after the <|audio|> placeholder. When "
             "omitted, the default is picked per-variant from --model "
             "to match the published model card: "
             "1b/plus -> 'can you transcribe the speech into a written "
             "format?' (with a leading space for plus); "
             "4.1-2b base -> 'transcribe the speech with proper "
             "punctuation and capitalization.' "
             "Override only if you know what you are doing.",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="System message content. When omitted, picked per-variant "
             "from --model to match the model card. Pass empty string "
             "to force no system role. Only applied when the variant's "
             "chat template includes a system role (plus today).",
    )
    p.add_argument("--language", default=None,
                   help="Ignored (granite chat prompt is fixed).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'mps' on Apple).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dtype", default="bf16",
                   choices=["bf16", "f16", "f32"])
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
        torch.set_num_interop_threads(1)

    import soundfile as sf
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.granite_speech import (
        GraniteSpeechForConditionalGeneration,
    )
    try:
        from transformers.models.granite_speech_plus import (
            GraniteSpeechPlusForConditionalGeneration,
        )
    except Exception:
        GraniteSpeechPlusForConditionalGeneration = None  # type: ignore

    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None

    print(
        f"loading: {args.model}  (transformers {transformers.__version__}, "
        f"device={args.device})"
    )
    t0 = time.monotonic()
    cfg = AutoConfig.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
    )
    cls_map = {"granite_speech": GraniteSpeechForConditionalGeneration}
    if GraniteSpeechPlusForConditionalGeneration is not None:
        cls_map["granite_speech_plus"] = GraniteSpeechPlusForConditionalGeneration
    cls = cls_map.get(cfg.model_type)
    if cls is None:
        raise SystemExit(
            f"unsupported model_type={cfg.model_type!r}; "
            f"expected granite_speech or granite_speech_plus"
        )

    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
    )
    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]
    model = cls.from_pretrained(
        args.model, config=cfg, revision=revision,
        trust_remote_code=False, local_files_only=local_only,
        dtype=dtype,
        attn_implementation="eager",
    ).eval().to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    # Pick per-variant prompt to match the model card. The variant is
    # inferred from --model: the substring after the last '/' is the
    # repo basename (also the local dir name when passing a path),
    # which is exactly stt.variant in the GGUF.
    variant = args.model.rstrip("/").rsplit("/", 1)[-1]
    if args.instruction is not None:
        instruction = args.instruction
    elif variant == "granite-speech-4.1-2b":
        instruction = "transcribe the speech with proper punctuation and capitalization."
    elif variant == "granite-speech-4.1-2b-plus":
        # Leading space matters: BPE tokenizes " can" vs "can" differently.
        instruction = " can you transcribe the speech into a written format?"
    else:
        # granite-4.0-1b-speech and any other variant.
        instruction = "can you transcribe the speech into a written format?"

    if args.system_prompt is not None:
        system_content = args.system_prompt
    elif variant == "granite-speech-4.1-2b-plus":
        system_content = (
            "Knowledge Cutoff Date: April 2024.\n"
            "Today's Date: December 19, 2024.\n"
            "You are Granite, developed by IBM. You are a helpful AI assistant"
        )
    else:
        system_content = ""

    # Build the prompt once. Same prompt every utterance.
    user_message = f"<|audio|>{instruction}"
    chat = []
    if system_content:
        chat.append({"role": "system", "content": system_content})
    chat.append({"role": "user", "content": user_message})
    has_template = (
        getattr(processor.tokenizer, "chat_template", None) is not None
        or getattr(processor, "chat_template", None) is not None
    )
    if has_template:
        prompt = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"USER: {user_message}\n ASSISTANT:"
    print(f"variant: {variant}")
    print(f"prompt: {prompt!r}")

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
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
            "framework": "transformers",
            "model": args.model,
            "language": args.language,
            "dtype": args.dtype,
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
                if sr != 16000:
                    raise RuntimeError(
                        f"granite-speech expects 16kHz; got {sr}Hz"
                    )
                inputs = processor(
                    text=prompt, audio=[torch.from_numpy(pcm)],
                    sampling_rate=sr, return_tensors="pt",
                    device=args.device,
                )
                inputs = {
                    k: (v.to(args.device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
                prompt_len = int(inputs["input_ids"].shape[1])

                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                if hasattr(gen, "sequences"):
                    token_ids = gen.sequences[0].detach().cpu().tolist()
                else:
                    token_ids = gen[0].detach().cpu().tolist()
                if len(token_ids) > prompt_len:
                    token_ids = token_ids[prompt_len:]
                eos_id = processor.tokenizer.eos_token_id
                if eos_id is not None and eos_id in token_ids:
                    token_ids = token_ids[:token_ids.index(eos_id)]
                hyp_text = processor.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                ).strip()
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
