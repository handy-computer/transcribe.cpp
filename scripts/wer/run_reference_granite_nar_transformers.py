#!/usr/bin/env python3
"""
run_reference_granite_nar_transformers.py — IBM Granite Speech NAR
WER baseline against `ibm-granite/granite-speech-4.1-2b-nar`.

Mirrors the README inference path (AutoProcessor + AutoModel +
model.transcribe + processor.batch_decode) over a WER manifest. Writes
scripts/wer/score.py-compatible JSONL records.

Usage (from repo root):

    uv run --project scripts/envs/granite_nar \\
      scripts/wer/run_reference_granite_nar_transformers.py \\
        --model ibm-granite/granite-speech-4.1-2b-nar \\
        --revision 99a4df9007ac5682f9daa093fb7008ff606e9a5d \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --device mps \\
        --out reports/wer/granite-speech-4.1-2b-nar-REF.test-clean.jsonl
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
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True,
                   help="HF repo id or local directory.")
    p.add_argument("--revision", default=None,
                   help="HF revision (commit hash) to pin against drift")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--dtype", default="bf16",
                   choices=["bf16", "f16", "f32"])
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--language", default=None,
                   help="Ignored (NAR is single-task multilingual ASR).")
    p.add_argument("--limit", type=int, default=0)
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
    from transformers import AutoModel, AutoProcessor

    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None
    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]

    rev_text = f", revision={revision}" if revision else ""
    print(
        f"loading {args.model} (transformers {transformers.__version__}, "
        f"device={args.device}{rev_text}, dtype={args.dtype})..."
    )
    t0 = time.monotonic()

    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
    )
    model = AutoModel.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
        dtype=dtype, attn_implementation=args.attn_impl,
        device_map=args.device,
    ).eval()

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
            "framework": "transformers",
            "model": args.model,
            "revision": args.revision,
            "language": args.language,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "device": args.device,
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
                        f"granite_nar expects 16kHz; got {sr}Hz"
                    )
                waveform = torch.from_numpy(pcm)
                inputs = processor([waveform], device=args.device)

                with torch.inference_mode():
                    output = model.transcribe(**inputs)

                transcriptions = processor.batch_decode(output.preds)
                hyp_text = (transcriptions[0] if transcriptions else "").strip()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text,
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
