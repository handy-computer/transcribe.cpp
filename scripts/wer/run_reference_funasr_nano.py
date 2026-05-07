#!/usr/bin/env python3
"""
run_reference_funasr_nano.py — FunASR Fun-ASR-Nano-2512 batch transcribe.

Loads Fun-ASR-Nano via funasr.AutoModel and runs it over a WER manifest,
writing run.py-compatible JSONL so scripts/wer/score.py can score the output.
Mirrors the dump-time setup (scripts/dump_reference_funasr_nano_funasr.py):
pinned revision, dither=0.0 with baseline re-snapshot so auto.generate's
runtime reset doesn't restore the default dither=1.0, FunASR's broken
absolute-import shim for `funasr.models.fun_asr_nano.{ctc,tools}`.

Usage (from repo root):

    uv run --project scripts/envs/funasr_nano \\
      scripts/wer/run_reference_funasr_nano.py \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --out      reports/wer/fun-asr-nano-2512-REF.test-clean.jsonl \\
        --torch-threads 8
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _patch_fun_asr_nano_imports() -> None:
    # Same shim as scripts/dump_reference_funasr_nano_funasr.py — FunASR 1.3.1
    # ships fun_asr_nano/model.py with broken absolute imports
    # (`from ctc import CTC`, `from tools.utils import forced_align`). Without
    # this shim AutoModel raises "FunASRNano is not registered" because the
    # funasr package's submodule walk silently swallows the ImportError.
    os.environ.setdefault("FUNASR_DISABLE_UPDATE", "1")
    import funasr  # noqa: F401  (triggers the silent submodule walk)
    sub_ctc = importlib.import_module("funasr.models.fun_asr_nano.ctc")
    sys.modules.setdefault("ctc", sub_ctc)
    sub_tools_pkg = importlib.import_module("funasr.models.fun_asr_nano.tools")
    sub_tools_utils = importlib.import_module(
        "funasr.models.fun_asr_nano.tools.utils"
    )
    sys.modules.setdefault("tools", sub_tools_pkg)
    sys.modules["tools.utils"] = sub_tools_utils
    importlib.import_module("funasr.models.fun_asr_nano.model")


_patch_fun_asr_nano_imports()


CONTROL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def strip_control_tokens(text: str) -> str:
    return CONTROL_TOKEN_RE.sub("", text).strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512",
                   help="HF repo id (default: FunAudioLLM/Fun-ASR-Nano-2512)")
    p.add_argument("--revision",
                   default="a7088d620f755dcdca575b63db184c3ad55b2865",
                   help="HF revision pin (matches Stage 2 oracle).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu — matches oracle).")
    p.add_argument("--language", default="en",
                   help="Language hint (default: en for LibriSpeech).")
    p.add_argument("--use-itn", action="store_true",
                   help="Enable inverse text normalization (default: off).")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max new tokens per utterance (default: 256).")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = leave default).")
    p.add_argument("--report-every", type=int, default=25,
                   help="Print progress every N utterances (default: 25).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="If --out already exists, skip ids already present.")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        # Outer ops sequential; the per-utterance kaldi.fbank + matmul stack
        # parallelizes within the BLAS / mkl_atlas backend.
        torch.set_num_interop_threads(1)
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")

    from funasr import AutoModel

    print(
        f"loading: model={args.model} revision={args.revision} "
        f"device={args.device}"
    )
    t_load_start = time.monotonic()
    auto = AutoModel(
        model=args.model,
        model_revision=args.revision,
        hub="hf",
        device=args.device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        trust_remote_code=True,
        dither=0.0,
    )
    auto.kwargs["frontend"].dither = 0.0
    # FunASR's AutoModel snapshots `self.kwargs` at __init__ and resets it via
    # copy.deepcopy on every auto.generate() — which would wipe our dither=0
    # and let auto.generate run with the default dither=1.0 (non-deterministic
    # noise added to kaldi fbank). Re-snapshot so the deterministic dither
    # survives the reset.
    auto._store_base_configs()
    load_ms = round((time.monotonic() - t_load_start) * 1000, 1)
    print(f"loaded in {load_ms / 1000:.1f}s")

    # Manifest.
    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")
    print(f"language: {args.language}  use_itn: {args.use_itn}  max_length: {args.max_length}")

    # Resume support: if the output file exists and --resume is set, skip
    # already-processed ids.
    done_ids: set[str] = set()
    if args.resume and args.out.exists():
        with open(args.out) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") == "batch_header":
                    continue
                if isinstance(rec, dict) and "id" in rec:
                    done_ids.add(rec["id"])
        print(f"resume: skipping {len(done_ids)} already-processed ids")

    open_mode = "a" if args.resume and args.out.exists() else "w"

    t_loop = time.monotonic()
    n_done = 0
    n_errors = 0

    with open(args.out, open_mode) as fout:
        if open_mode == "w":
            fout.write(json.dumps({
                "type": "batch_header",
                "load_ms": load_ms,
                "framework": "funasr",
                "model": args.model,
                "revision": args.revision,
                "device": args.device,
                "language": args.language,
                "use_itn": bool(args.use_itn),
                "torch_threads": torch.get_num_threads(),
            }) + "\n")
            fout.flush()

        for entry in manifest:
            uid = entry["id"]
            audio_path = entry["audio"]
            ref_text = entry.get("ref_text", "")

            if uid in done_ids:
                continue

            t_start = time.monotonic()
            err = ""
            raw_text = ""
            try:
                res = auto.generate(
                    input=audio_path,
                    language=args.language,
                    itn=bool(args.use_itn),
                    max_length=args.max_length,
                    device=args.device,
                )
                if isinstance(res, list) and res:
                    raw_text = res[0].get("text", "")
                else:
                    err = "empty result"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            hyp_text = strip_control_tokens(raw_text)
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text,
                "raw_text": raw_text,
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "error": err,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1

            if n_done % args.report_every == 0 or n_done == total - len(done_ids):
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                remaining = (total - len(done_ids)) - n_done
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  [{n_done + len(done_ids)}/{total}] "
                    f"{rate:.2f} utt/s, ETA {eta/60:.1f} min, "
                    f"errors={n_errors}, last_ms={elapsed_ms:.0f}",
                    flush=True,
                )

    wall = time.monotonic() - t_loop
    rate = n_done / wall if wall > 0 else 0
    print(
        f"\ndone. {n_done} new utterances in {wall:.1f}s "
        f"({rate:.2f} utt/s), {n_errors} errors"
    )
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
