#!/usr/bin/env python3
"""
run_reference_granite_nar_transformers.py — IBM Granite NLE (NAR) WER baseline.

Loads `ibm-granite/granite-speech-4.1-2b-nar` via the upstream
NLENARDecoder modeling code (trust_remote_code=True) and runs the
single-pass NAR forward over a WER manifest. Writes run.py-compatible
JSONL so scripts/wer/score.py can score it like the C++ port report.

The NAR decode is one forward (encoder + projector + bidirectional LLM
+ CTC head) — no autoregressive token loop and no generation config.

Usage (from repo root):

    uv run --project scripts/envs/granite_nar \\
      scripts/wer/run_reference_granite_nar_transformers.py \\
        --model ibm-granite/granite-speech-4.1-2b-nar \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --out reports/wer/granite-speech-4.1-2b-nar-REF.test-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
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
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'mps' on Apple).")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--dtype", default="bf16",
                   choices=["bf16", "f16", "f32"])
    p.add_argument("--language", default=None,
                   help="Ignored (NAR is single-task ASR).")
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
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None
    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]

    print(
        f"loading: {args.model}  (transformers {transformers.__version__}, "
        f"device={args.device}, dtype={args.dtype})"
    )
    t0 = time.monotonic()

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
    )
    cfg = AutoConfig.from_pretrained(
        args.model, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "sdpa"

    # Force bidirectional attention without flash-attn-2.
    #
    # transformers' GraniteModel.forward unconditionally builds a causal
    # mask via create_causal_mask() and passes it to every attention
    # layer as attention_mask. Both eager and sdpa then apply it, so
    # is_causal=False on the layers is ineffective — the mask wins.
    # flash_attention_2 is the only backend that bypasses this path
    # entirely, which is why modeling_nle.py asserts it.
    #
    # Patching create_causal_mask to return None lets sdpa see
    # attention_mask=None + module.is_causal=False (NLE sets this) and
    # compute a true bidirectional attention. Same effect for eager.
    from transformers.models.granite import modeling_granite as _granite_mod
    _granite_mod.create_causal_mask = lambda **kw: None
    print("patched transformers.models.granite.modeling_granite."
          "create_causal_mask -> None  (forces bidirectional)")

    model = AutoModel.from_pretrained(
        args.model, config=cfg, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
        dtype=dtype, attn_implementation="eager",
    ).eval().to(args.device)

    # The flash-attn assert in modeling_nle.py only checks
    # `self.config.attn_implementation == "flash_attention_2"`. Bypass
    # the assert (we've made sdpa bidirectional via the mask patch).
    original_forward = model.forward.__func__

    def patched_forward(self, *fargs, **fkwargs):
        c = self.config
        if c.attn_implementation != "flash_attention_2":
            saved = c.attn_implementation
            c.attn_implementation = "flash_attention_2"
            try:
                return original_forward(self, *fargs, **fkwargs)
            finally:
                c.attn_implementation = saved
        return original_forward(self, *fargs, **fkwargs)

    model.forward = types.MethodType(patched_forward, model)
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
                        f"granite_nar expects 16kHz; got {sr}Hz"
                    )
                feats = feature_extractor(
                    [torch.from_numpy(pcm)], device=args.device
                )
                input_features = feats["input_features"].to(args.device)
                attention_mask = feats["attention_mask"].to(args.device)

                with torch.inference_mode():
                    gen_out = model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                    )
                # NAR returns text_preds directly (single-pass CTC decode).
                text_preds = getattr(gen_out, "text_preds", None) or []
                hyp_text = (text_preds[0] if text_preds else "").strip()
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
