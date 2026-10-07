#!/usr/bin/env python3
"""
run_reference_granite5_ctc_transformers.py — Granite Speech 5.0 TurboCTC
WER baseline.

Mirrors the model card path (AutoProcessor + AutoModelForCTC +
model.generate + processor.batch_decode(skip_special_tokens=True)) over a
WER manifest. Writes scripts/wer/score.py-compatible JSONL records.

DTYPE: defaults to f32, not the bf16 the weights ship as. The BF16->F32
upcast is lossless (same shipped weights) and F32 activations over BF16
weights is exactly the transcribe.cpp compute regime, so this baseline is
the closest reference to what the C++ will actually compute. See the
oracle_dtype note in tests/golden/granite5_ctc/*.manifest.json.

Usage:

    uv run --project scripts/envs/granite5_ctc \\
      scripts/wer/run_reference_granite5_ctc_transformers.py \\
        --model ibm-granite/granite-speech-5.0-470m-turboctc \\
        --revision 18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0 \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --device mps \\
        --out reports/wer/granite-speech-5.0-470m-turboctc-REF.test-clean.jsonl
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
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--dtype", default="f32", choices=["bf16", "f16", "f32"],
                   help="Reference dtype; f32 is the porting baseline (see module docstring).")
    p.add_argument("--attn-impl", default="eager",
                   choices=["eager", "sdpa", "flex_attention"],
                   help="GraniteSpeech5 sets _supports_flash_attn=False "
                        "(the Shaw relative-position bias is a float mask).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Utterances per forward. >1 exercises the padding-mask "
                        "path; keep at 1 for the reference baseline so a mask "
                        "bug cannot perturb the gate.")
    p.add_argument("--language", default=None,
                   help="Ignored (granite5_ctc is monolingual English).")
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
    from transformers import AutoModelForCTC, AutoProcessor

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
        args.model, revision=revision, local_files_only=local_only,
    )
    model = AutoModelForCTC.from_pretrained(
        args.model, revision=revision, local_files_only=local_only,
        dtype=dtype, attn_implementation=args.attn_impl,
    ).to(args.device).eval()

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

    batch_size = max(1, args.batch_size)

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
            "batch_size": batch_size,
        }) + "\n")
        fout.flush()

        for start in range(0, total, batch_size):
            chunk = manifest[start:start + batch_size]
            t_start = time.monotonic()
            hyps = [""] * len(chunk)
            errs = [""] * len(chunk)
            try:
                pcms = []
                for entry in chunk:
                    pcm, sr = sf.read(entry["audio"], dtype="float32")
                    if pcm.ndim > 1:
                        pcm = pcm[:, 0]
                    if sr != 16000:
                        raise RuntimeError(
                            f"granite5_ctc expects 16kHz; got {sr}Hz "
                            f"for {entry['id']}")
                    pcms.append(pcm)

                inputs = processor(
                    pcms if len(pcms) > 1 else pcms[0],
                    sampling_rate=16000,
                )
                inputs = inputs.to(args.device)

                with torch.inference_mode():
                    token_ids = model.generate(**inputs)

                texts = processor.batch_decode(
                    token_ids, skip_special_tokens=True)
                for i in range(len(chunk)):
                    hyps[i] = (texts[i] if i < len(texts) else "").strip()
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                errs = [msg] * len(chunk)
                n_errors += len(chunk)

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            per_utt_ms = round(elapsed_ms / len(chunk), 1)
            for i, entry in enumerate(chunk):
                fout.write(json.dumps({
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": hyps[i],
                    "mel_ms": 0,
                    "encode_ms": 0,
                    "decode_ms": per_utt_ms,
                    "latency_ms": per_utt_ms,
                    "error": errs[i],
                }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += len(chunk)

            if n_done % 100 < batch_size or n_done == total:
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
