#!/usr/bin/env python3
"""
run_reference_parakeet_kestrel.py — parakeet-ultra reference WER via kestrel.

Transcribes a WER manifest through kestrel's ParakeetTdtRuntime (clips over
30 s are VAD-segmented and stitched). Writes run.py-compatible JSONL for
scripts/wer/score.py.

Mirrors scripts/dump_reference_parakeet_kestrel.py: float32 weights, no
language hint (kestrel rejects language forcing).

Usage (from repo root):

    uv run --project scripts/envs/parakeet-kestrel \\
      scripts/wer/run_reference_parakeet_kestrel.py \\
        --manifest samples/wer/librispeech-test-clean.manifest.jsonl \\
        --model    moondream/parakeet-ultra \\
        --revision 73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b \\
        --out      reports/wer/parakeet-ultra-REF.librispeech-test-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path

ULTRA_REVISION = "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", default="moondream/parakeet-ultra",
                   help="HF repo id or local snapshot dir.")
    p.add_argument("--revision", default=ULTRA_REVISION,
                   help="HF revision (kestrel itself tracks main; pin it).")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"),
                   help="Torch device (default cpu).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Requests per runtime.forward call. 1 keeps every "
                        "utterance on the single-row decode path (default).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="CPU threads (0 = kestrel's default).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    import torch
    from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime
    from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt

    t0 = time.monotonic()
    loaded = load_parakeet_tdt(args.model, revision=args.revision,
                               device=args.device, dtype=torch.float32)
    cfg = types.SimpleNamespace(
        model=args.model, device=args.device, dtype=torch.float32,
        cpu_threads=args.torch_threads or None, enable_cuda_graphs=False,
    )
    runtime = ParakeetTdtRuntime(cfg, model=loaded.model, tokenizer=loaded.tokenizer)
    load_ms = (time.monotonic() - t0) * 1000
    print(f"loaded {args.model}@{args.revision[:12]} on {args.device} "
          f"(vad_head={loaded.model.vad_head is not None}) in {load_ms:.0f} ms", flush=True)

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")
    # Kestrel takes no language hint; the manifest's language is recorded
    # only so score.py picks the right normalizer (absent means English).
    languages = {entry.get("language") for entry in manifest}
    language = languages.pop() if len(languages) == 1 else None

    n_done = n_errors = 0
    audio_s = 0.0
    t_loop = time.monotonic()
    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "kestrel",
            "framework_version": f"kestrel=={version('kestrel')}, kestrel-kernels=={version('kestrel-kernels')}",
            "model": args.model,
            "model_revision": args.revision,
            "device": args.device,
            "batch_size": args.batch_size,
            "language": language,
        }) + "\n")
        fout.flush()

        for start in range(0, total, args.batch_size):
            group = manifest[start:start + args.batch_size]
            requests = []
            for entry in group:
                pcm, sr = sf.read(entry["audio"], dtype="float32", always_2d=False)
                audio_s += pcm.size / sr
                requests.append({"audio": pcm, "sample_rate": sr, "timestamps": "none"})
            t_start = time.monotonic()
            try:
                results = runtime.forward("transcribe", requests)
            except Exception as e:  # whole-cohort failure
                results = [e] * len(group)
            elapsed_ms = round((time.monotonic() - t_start) * 1000 / len(group), 1)
            for entry, result in zip(group, results):
                err, hyp = "", ""
                if isinstance(result, Exception):
                    err = f"{type(result).__name__}: {result}"
                    n_errors += 1
                else:
                    hyp = str(result["text"])
                fout.write(json.dumps({
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": hyp.strip(),
                    "raw_text": hyp,
                    "mel_ms": 0,
                    "encode_ms": 0,
                    "decode_ms": elapsed_ms,
                    "latency_ms": elapsed_ms,
                    "error": err,
                }) + "\n")
                n_done += 1
            fout.flush()
            if n_done % 100 < len(group) or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.2f} utt/s, xRT {audio_s / wall:.1f}, "
                      f"ETA {eta / 60:.1f} min, errors={n_errors}", flush=True)

    wall = time.monotonic() - t_loop
    print(f"\ndone. {n_done} utterances, {audio_s / 3600:.2f} h audio in {wall:.1f}s, {n_errors} errors")
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
