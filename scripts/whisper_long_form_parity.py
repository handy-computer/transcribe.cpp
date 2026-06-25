#!/usr/bin/env python3
"""whisper_long_form_parity.py — long-form regression check.

Run HuggingFace's WhisperForConditionalGeneration end-to-end on a long-form
clip with condition_on_prev_tokens=True and return_timestamps=True, then run
transcribe.cpp's transcribe-cli with matching params. Compare normalized text
(token-level WER) and segment counts.

This script targets the specific regression the parity plan calls out:
chunks that end with a closed timestamp pair plus discarded tail, where HF
applies skip_ending_double_timestamps when stitching prev-context for the
next chunk (PRs #34537 / #35750). The end-to-end text match is the
observable proxy; if our prev-context assembly diverges from HF, the
transcript drifts visibly across chunk boundaries.

Run via the repo-local Whisper reference environment:

    cmake --build build --target transcribe-cli
    uv run --project scripts/envs/whisper \\
      scripts/whisper_long_form_parity.py \\
      --model openai/whisper-tiny \\
      --gguf models/whisper-tiny/whisper-tiny-F32.gguf \\
      --audio samples/whole-earth.wav

Exits 0 when WER <= --wer-tol (default 0.05); non-zero on divergence.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def run_hf_reference(
    *,
    model_id: str,
    audio: Path,
    language: str,
    condition: bool,
    initial_prompt: str | None,
    prompt_condition: str,
) -> tuple[str, list[dict]]:
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"[hf] loading {model_id} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()

    pcm, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"audio must be 16k, got {sr}")
    print(f"[hf] audio {audio.name} samples={pcm.size} dur={pcm.size/sr:.2f}s",
          flush=True)

    inputs = processor(
        pcm,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=False,
    )

    gen_kwargs = dict(
        language=language,
        task="transcribe",
        return_timestamps=True,
        return_segments=True,
        condition_on_prev_tokens=condition,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        do_sample=False,
        num_beams=1,
    )
    if initial_prompt:
        gen_kwargs["prompt_ids"] = processor.get_prompt_ids(
            initial_prompt, return_tensors="pt"
        )
        gen_kwargs["prompt_condition_type"] = {
            "first": "first-segment",
            "all": "all-segments",
        }[prompt_condition]

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    sequences = out["sequences"] if isinstance(out, dict) else out.sequences
    text = processor.batch_decode(
        sequences, skip_special_tokens=True
    )[0].strip()
    segments = out["segments"][0] if isinstance(out, dict) else out.segments[0]
    seg_summary = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "n_tokens": int(s["tokens"].numel()),
        }
        for s in segments
    ]
    return text, seg_summary


def run_cpp(
    *,
    cli: Path,
    gguf: Path,
    audio: Path,
    language: str,
    condition: bool,
    initial_prompt: str | None,
    prompt_condition: str,
) -> tuple[str, list[dict]]:
    # Drive the CLI via --batch / --batch-jsonl so output lands as
    # parseable JSON. A single-line batch file pointing at the target
    # WAV gives us one record back.
    cmd = [
        str(cli),
        "-q",
        "--backend", "cpu",
        "--threads", "1",
        "-m", str(gguf),
        "--language", language,
        "--timestamps", "segment",
        "--batch", "/dev/stdin",
        "--batch-jsonl",
    ]
    if condition:
        cmd.append("--condition-on-prev-tokens")
    if initial_prompt:
        cmd.extend(["--initial-prompt", initial_prompt])
        cmd.extend(["--prompt-condition", prompt_condition])
    print(f"[cpp] running: {' '.join(cmd)} <<< {audio}", flush=True)
    res = subprocess.run(
        cmd,
        input=str(audio) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"transcribe-cli failed (rc={res.returncode})"
        )

    text = ""
    segments: list[dict] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") == "batch_header":
            continue
        text = (rec.get("text") or "").strip()
        segments = [
            {
                "start": float(s["t0_ms"]) / 1000.0,
                "end":   float(s["t1_ms"]) / 1000.0,
                "n_tokens": -1,
            }
            for s in rec.get("segments", [])
        ]
        break
    return text, segments


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id (e.g. openai/whisper-tiny)")
    p.add_argument("--gguf", required=True, type=Path, help="path to converted GGUF")
    p.add_argument("--audio", required=True, type=Path,
                   help="long-form WAV (>30s) to compare on")
    p.add_argument("--cli", type=Path, default=Path("build/bin/transcribe-cli"),
                   help="transcribe-cli binary path")
    p.add_argument("--language", default="en")
    p.add_argument("--initial-prompt",
                   help="initial prompt text / glossary to pass to both HF and C++. "
                        "NOTE: with --prompt-condition first (the default), the C++ "
                        "engine DELIBERATELY diverges from HF — it primes the prompt "
                        "on the first window only (whisper.cpp/OpenAI behavior) "
                        "instead of every window. HF re-primes every window, which "
                        "can collapse interior windows on out-of-distribution "
                        "prompts; the C++ behavior is the robust drop-in. So a "
                        "prompted run with prompt-condition=first is EXPECTED to "
                        "show non-zero WER vs HF and is not a regression. Use "
                        "--prompt-condition all for a like-for-like HF comparison.")
    p.add_argument("--prompt-condition", choices=["first", "all"], default="first",
                   help="prompt placement when --initial-prompt is set")
    p.add_argument("--no-condition", action="store_true",
                   help="run with condition_on_prev_tokens=False")
    p.add_argument("--wer-tol", type=float, default=0.05,
                   help="max acceptable WER between HF ref and C++ output")
    args = p.parse_args()

    if not args.cli.exists():
        print(f"error: cli not found: {args.cli}", file=sys.stderr)
        return 1
    if not args.gguf.exists():
        print(f"error: gguf not found: {args.gguf}", file=sys.stderr)
        return 1
    if not args.audio.exists():
        print(f"error: audio not found: {args.audio}", file=sys.stderr)
        return 1

    condition = not args.no_condition

    hf_text, hf_segs = run_hf_reference(
        model_id=args.model,
        audio=args.audio,
        language=args.language,
        condition=condition,
        initial_prompt=args.initial_prompt,
        prompt_condition=args.prompt_condition,
    )
    cpp_text, cpp_segs = run_cpp(
        cli=args.cli,
        gguf=args.gguf,
        audio=args.audio,
        language=args.language,
        condition=condition,
        initial_prompt=args.initial_prompt,
        prompt_condition=args.prompt_condition,
    )

    ref_words = normalize_text(hf_text)
    hyp_words = normalize_text(cpp_text)
    err = wer(ref_words, hyp_words)
    seg_delta = abs(len(hf_segs) - len(cpp_segs))

    print()
    print(f"condition_on_prev_tokens = {condition}")
    print(f"initial_prompt           = {bool(args.initial_prompt)}")
    if args.initial_prompt:
        print(f"prompt_condition         = {args.prompt_condition}")
    print(f"hf  segs={len(hf_segs)} words={len(ref_words)}")
    print(f"cpp segs={len(cpp_segs)} words={len(hyp_words)}")
    print(f"wer (cpp vs hf): {err:.4f}")
    print(f"|seg_delta|     : {seg_delta}")
    print()
    print("HF text  :", hf_text)
    print("CPP text :", cpp_text)

    ok = err <= args.wer_tol
    if not ok:
        print(f"\nFAIL: WER {err:.4f} > tolerance {args.wer_tol:.4f}",
              file=sys.stderr)
        return 1
    print(f"\nOK: WER {err:.4f} <= tolerance {args.wer_tol:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
