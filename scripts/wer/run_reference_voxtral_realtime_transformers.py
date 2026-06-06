#!/usr/bin/env python3
"""
run_reference_voxtral_realtime_transformers.py — Voxtral Realtime (2602) WER baseline.

Loads a Voxtral Realtime variant via the mainline HuggingFace Transformers
implementation (`VoxtralRealtimeForConditionalGeneration`) and runs greedy
decode over a WER manifest. Writes run.py-compatible JSONL so
scripts/wer/score.py scores it the same way it scores the C++ port.

Two modes (default `offline`):

  offline    one whole-clip generate() per utterance. This is the path Stage 4
             implements first, so it is the acceptance-gate baseline.
  streaming  drives the model with the canonical chunked audio generator (from
             the HF model docs); matches the publisher's real-time setting.
             batch-size is forced to 1.

The realtime processor is auto-detect only (TranscriptionRequest language=None),
so there is no per-utterance language hint; --language is accepted and ignored.

Uniform contract (--manifest/--model/--out/--device/--batch-size) so the Modal
reference_sweep drives every family the same way.

Usage (from repo root):

    uv run --project scripts/envs/voxtral_realtime \\
      scripts/wer/run_reference_voxtral_realtime_transformers.py \\
        --model models/Voxtral-Mini-4B-Realtime-2602 \\
        --manifest samples/wer/test-clean.manifest.jsonl \\
        --out reports/wer/voxtral-mini-4b-realtime-2602-REF.test-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(path, dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16kHz audio, got {sr} Hz in {path}")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text/language).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", required=True,
                   help="HF repo id (mistralai/Voxtral-Mini-4B-Realtime-2602) or local dir.")
    p.add_argument("--revision", default=None,
                   help="HF revision to pin (ignored for local paths).")
    p.add_argument("--language", default="auto",
                   help="Accepted for a uniform CLI but ignored: the realtime "
                        "processor is auto-detect only.")
    p.add_argument("--mode", default="offline", choices=["offline", "streaming"],
                   help="offline (whole-clip, the acceptance gate) or streaming "
                        "(chunked generator, publisher real-time setting).")
    p.add_argument("--num-delay-tokens", type=int, default=None,
                   help="override delay tokens (default: processor value = 480ms).")
    p.add_argument("--device", default="cpu",
                   help="torch device (default: cpu; pass 'cuda'/'mps').")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0 = unchanged).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Group N utterances per offline generate() call. Falls "
                        "back to per-utterance on batch error. Forced to 1 for "
                        "streaming mode.")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    import transformers
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    local_only = Path(args.model).is_dir()
    revision = args.revision if not local_only else None
    model_id = str(Path(args.model).resolve()) if local_only else args.model

    print(f"loading: {args.model}  (transformers {transformers.__version__}, "
          f"device={args.device}, mode={args.mode})")
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        args.model, revision=revision, local_files_only=local_only,
    )
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        args.model, revision=revision, local_files_only=local_only,
        dtype=dtype, attn_implementation="sdpa",
    ).eval().to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    def cast(v):
        if not hasattr(v, "to"):
            return v
        if torch.is_floating_point(v):
            return v.to(args.device, dtype=dtype)
        return v.to(args.device)

    def decode_new(ids: list[int]) -> str:
        eos_id = 2
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        tok = processor.tokenizer
        for attempt in (
            lambda: processor.batch_decode([ids], skip_special_tokens=True)[0],
            lambda: tok.decode(ids, skip_special_tokens=True),
            lambda: tok.decode(ids),
        ):
            try:
                return attempt().strip()
            except Exception:  # noqa: BLE001
                continue
        return ""

    def infer_offline(entry: dict) -> str:
        arr = load_audio(entry["audio"])
        inputs = processor(audio=arr, is_streaming=False, return_tensors="pt")
        inputs = {k: cast(v) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(**inputs, do_sample=False, num_beams=1)
        seq = gen.sequences[0] if hasattr(gen, "sequences") else gen[0]
        return decode_new(seq.detach().cpu().tolist()[prompt_len:])

    def infer_offline_batch(entries: list) -> list:
        arrs = [load_audio(e["audio"]) for e in entries]
        inputs = processor(audio=arrs, is_streaming=False, return_tensors="pt")
        inputs = {k: cast(v) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(**inputs, do_sample=False, num_beams=1)
        seqs = gen.sequences if hasattr(gen, "sequences") else gen
        return [decode_new(row.detach().cpu().tolist()[prompt_len:]) for row in seqs]

    def infer_streaming(entry: dict) -> str:
        arr = load_audio(entry["audio"])
        hop = processor.feature_extractor.hop_length
        win = processor.feature_extractor.win_length
        n_rpt = processor.num_right_pad_tokens  # method in transformers 5.10.2
        n_rpt = n_rpt() if callable(n_rpt) else n_rpt
        audio = np.pad(arr, (0, n_rpt * processor.raw_audio_length_per_tok))
        first = processor(
            audio=audio[: processor.num_samples_first_audio_chunk],
            is_streaming=True, is_first_audio_chunk=True, return_tensors="pt",
        )
        first = {k: cast(v) for k, v in first.items()}

        def feature_gen():
            yield first["input_features"]
            mel_frame_idx = processor.num_mel_frames_first_audio_chunk
            start_idx = mel_frame_idx * hop - win // 2
            while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
                chunk = processor(
                    audio=audio[start_idx:end_idx],
                    is_streaming=True, is_first_audio_chunk=False, return_tensors="pt",
                )
                yield cast(chunk["input_features"])
                mel_frame_idx += processor.audio_length_per_tok
                start_idx = mel_frame_idx * hop - win // 2

        prompt_len = int(first["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(
                input_ids=first["input_ids"],
                input_features=feature_gen(),
                num_delay_tokens=first["num_delay_tokens"],
                do_sample=False, num_beams=1,
                return_dict_in_generate=True,
            )
        return decode_new(gen.sequences[0].detach().cpu().tolist()[prompt_len:])

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[:args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    bs = 1 if args.mode == "streaming" else max(1, args.batch_size)
    n_done = n_errors = 0
    t_loop = time.monotonic()

    # Header language drives score.py's normalizer choice. The model is
    # auto-detect (it ignores --language for inference), but SCORING must be
    # routed by the DATA language, not the model's detection mode: derive it
    # from the manifest (authoritative), falling back to an explicit --language,
    # then "auto". This mirrors run.py and every other reference runner so the
    # EnglishTextNormalizer is selected for English data instead of the
    # BasicTextNormalizer (which leaves numbers/$/Mr. unexpanded and inflates
    # WER on LibriSpeech-style references).
    manifest_langs = {e.get("language") for e in manifest if e.get("language")}
    if len(manifest_langs) == 1:
        header_language = next(iter(manifest_langs))
    elif args.language and args.language != "auto":
        header_language = args.language
    else:
        header_language = "auto"

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header", "load_ms": round(load_ms, 1),
            "framework": "transformers", "model": args.model,
            "language": header_language, "dtype": args.dtype, "mode": args.mode,
        }) + "\n")
        fout.flush()

        for start in range(0, total, bs):
            group = manifest[start:start + bs]
            k = len(group)
            t_start = time.monotonic()
            hyps = [""] * k
            errs = [""] * k
            if bs == 1:
                try:
                    hyps[0] = (infer_streaming if args.mode == "streaming" else infer_offline)(group[0])
                except Exception as e:  # noqa: BLE001
                    errs[0] = f"{type(e).__name__}: {e}"
                    n_errors += 1
            else:
                try:
                    hyps = infer_offline_batch(group)
                except Exception:
                    for i, e_ in enumerate(group):
                        try:
                            hyps[i] = infer_offline(e_)
                        except Exception as e2:  # noqa: BLE001
                            errs[i] = f"{type(e2).__name__}: {e2}"
                            n_errors += 1
            per_ms = round((time.monotonic() - t_start) * 1000 / k, 1)

            for i, entry in enumerate(group):
                fout.write(json.dumps({
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": (hyps[i] or "").strip(),
                    "mel_ms": 0, "encode_ms": 0,
                    "decode_ms": per_ms, "latency_ms": per_ms,
                    "error": errs[i],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1

            if start // bs % 10 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                      f"ETA {eta/60:.1f} min, errors={n_errors}", flush=True)

    wall = time.monotonic() - t_loop
    print(f"\ndone. {n_done} utterances in {wall:.1f}s "
          f"({n_done / wall:.2f} utt/s), {n_errors} errors")
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
