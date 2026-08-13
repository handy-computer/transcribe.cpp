#!/usr/bin/env python3
"""
run_reference_crisperwhisper_author.py — CrisperWhisper 2.0 WER baseline.

Drives the publisher's own package (`crisperwhisper`, MIT,
https://github.com/nyrahealth/CrisperWhisper) with the pure-torch
`transformers` backend over a WER manifest, writing run.py-compatible
JSONL so scripts/wer/score.py scores it exactly as it scores the C++ port.

Uniform contract (--manifest/--model/--out/--device/--batch-size) so the
Modal reference_sweep drives every family the same way. `--batch-size` is
accepted for contract compatibility but ignored: the transformers backend
has no batched transcribe path (`transcribe_dual` is CT2-only), so
batching here would mean re-implementing the publisher's decode loop.

Two knobs that matter, both decided at intake sign-off
------------------------------------------------------
--mode {verbatim,intended}
    `verbatim` (the model's default) emits [UM]/[UH]/[laughter] and
    preserves stutters and false starts by design. Scored against
    LibriSpeech's clean references that inflates WER for *correct*
    behaviour. The Stage 7 gate binds on `intended`; `verbatim` is
    reported alongside for the honest picture. Run both, gate one.

--rewind-features (default: off)
    `transcribe()` defaults hallucination_mitigation, early_eot_recovery,
    and temperature_fallback all to True, and all three can rewind and
    re-decode. All three are OUT OF SCOPE for this port (family doc
    Capability Validation), so the baseline is measured with them OFF —
    gating C++ against a reference carrying recovery machinery the port
    does not implement would be measuring the wrong thing. Pass
    --rewind-features to measure the publisher's out-of-the-box numbers
    instead; that run is not the gate.

Usage (from repo root):

    uv run --project scripts/envs/crisperwhisper \\
      scripts/wer/run_reference_crisperwhisper_author.py \\
        --model nyralabs/CrisperWhisper2.0_small \\
        --manifest samples/wer/librispeech-test-clean.manifest.jsonl \\
        --mode intended \\
        --out reports/wer/crisperwhisper-2.0-small-REF.librispeech-test-clean.jsonl
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
        allow_abbrev=False,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text/language).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", required=True,
                   help="HF repo id (nyralabs/CrisperWhisper2.0_small) or local dir.")
    p.add_argument("--mode", default="intended", choices=["verbatim", "intended"],
                   help="Transcription mode (default: intended — the Stage 7 gate). "
                        "Run again with --mode verbatim for the reported "
                        "second number.")
    p.add_argument("--language", default="en",
                   help="ISO 639-1 language hint. A manifest entry's own "
                        "'language' field overrides this per-utterance. Note the "
                        "author harness has no auto-detect path; a language is "
                        "always forced.")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'cuda'/'mps').")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Cap per utterance (default: 256, the reference "
                        "transcribe() default).")
    p.add_argument("--dtype", default="f32", choices=["bf16", "f16", "f32"],
                   help="Compute dtype. Default bf16: the weights ship BF16 and the "
                        "shipped GGUF executes in BF16 (ggml rounds activations "
                        "to BF16 in its BF16 matmul kernel), so bf16 is the "
                        "regime the C++ port is gated against. f32 upcasts "
                        "weights and keeps activations F32; use it only to score "
                        "an F32 GGUF."
                        "upcast to F32 compute, mirroring ggml's "
                        "BF16-storage/F32-compute regime.")
    p.add_argument("--rewind-features", action="store_true",
                   help="Enable hallucination_mitigation + early_eot_recovery + "
                        "temperature_fallback. OFF by default; see module docstring.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Accepted for contract compatibility; ignored (the "
                        "transformers backend has no batched transcribe path).")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    import crisperwhisper
    import transformers
    from crisperwhisper import CrisperWhisperModel

    compute_type = {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}[args.dtype]

    print(f"loading: {args.model}  (crisperwhisper "
          f"{getattr(crisperwhisper, '__version__', '?')}, transformers "
          f"{transformers.__version__}, device={args.device}, "
          f"compute_type={compute_type})")
    t0 = time.monotonic()
    model = CrisperWhisperModel(
        args.model,
        backend="transformers",
        compute_type=compute_type,
        device=args.device,
    )
    load_ms = (time.monotonic() - t0) * 1000

    rewind = bool(args.rewind_features)
    default_lang = args.language or "en"

    def lang_for(entry: dict) -> str:
        el = entry.get("language")
        if el in (None, "", "auto"):
            return default_lang
        return str(el)

    def infer_one(entry: dict) -> str:
        result = model.transcribe(
            entry["audio"],
            language=lang_for(entry),
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            hallucination_mitigation=rewind,
            early_eot_recovery=rewind,
            temperature_fallback=rewind,
            word_timestamps=False,
        )
        return (result.text or "").strip()

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"mode:     {args.mode}  rewind_features={rewind}")
    print(f"output:   {args.out}")

    n_done = n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "author_repo_crisperwhisper",
            "crisperwhisper_version": getattr(crisperwhisper, "__version__", "unknown"),
            "transformers_version": transformers.__version__,
            "model": args.model,
            "language": args.language,
            "dtype": args.dtype,
            "mode": args.mode,
            "rewind_features": rewind,
            "max_new_tokens": args.max_new_tokens,
        }) + "\n")
        fout.flush()

        for entry in manifest:
            t_start = time.monotonic()
            hyp, err = "", ""
            try:
                hyp = infer_one(entry)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                n_errors += 1
            per_ms = round((time.monotonic() - t_start) * 1000, 1)

            fout.write(json.dumps({
                "id": entry["id"],
                "ref_text": entry.get("ref_text", ""),
                "hyp_text": hyp,
                "mel_ms": 0, "encode_ms": 0,
                "decode_ms": per_ms, "latency_ms": per_ms,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 25 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                      f"ETA {eta / 60:.1f} min, errors={n_errors}", flush=True)

    wall = time.monotonic() - t_loop
    print(f"\ndone. {n_done} utterances in {wall:.1f}s "
          f"({n_done / wall:.2f} utt/s), {n_errors} errors")
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
