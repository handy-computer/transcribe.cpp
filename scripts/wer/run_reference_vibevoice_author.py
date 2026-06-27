#!/usr/bin/env python3
"""
run_reference_vibevoice_author.py - VibeVoice-ASR reference WER runner.

Uniform contract: --manifest --model --out --device --batch-size. Emits the
per-utterance reference hyp JSONL that porting-2-oracle Step 7 scores with
scripts/wer/score.py and that Stage 4 / Stage 7 diff C++ against.

VibeVoice-ASR emits structured JSON ([{Speaker,Start,End,Content}, ...]); the
reference hyp_text is the concatenation of the Content fields (speaker labels +
timestamps stripped), i.e. the plain-WER normalizer for this family. The
acoustic VAE path is forced to its deterministic mean (mode) - content is
identical to the stochastic path, see reports/porting/vibevoice.

The model can't run on a 16 GB laptop; this is meant to run on a >=40 GB GPU,
typically via:
    modal run scripts/wer/remote/modal_sweep.py::reference_sweep \
        --variants vibevoice:vibevoice-asr --gpu L40S

The author `vibevoice` package is not on PyPI and the HF repo ships no modeling
code, so the source is bootstrapped at runtime: set $VIBEVOICE_SRC to a checkout,
or it is git-cloned (pinned) to /tmp/VibeVoice.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VIBEVOICE_SHA = "303b2833e01cff4578ec278bbfe536da54bd19fe"
LM_TOKENIZER = "Qwen/Qwen2.5-7B"


def _ensure_vibevoice_on_path() -> str:
    src = os.environ.get("VIBEVOICE_SRC")
    if not src:
        src = "/tmp/VibeVoice"
        if not Path(src, "vibevoice").is_dir():
            print(f"cloning vibevoice {VIBEVOICE_SHA[:12]} -> {src}", flush=True)
            subprocess.run(["git", "clone", "https://github.com/microsoft/VibeVoice", src],
                           check=True)
            subprocess.run(["git", "-C", src, "checkout", VIBEVOICE_SHA], check=True)
    if not Path(src, "vibevoice").is_dir():
        raise SystemExit(f"vibevoice source not found at {src}")
    sys.path.insert(0, src)
    return src


def _load_model(model_id: str, device: str):
    import torch
    import transformers
    from transformers import PretrainedConfig

    transformers.logging.set_verbosity_error()
    def _safe_repr(self):
        try:
            return f"{self.__class__.__name__} {self.to_json_string()}"
        except Exception:
            return f"{self.__class__.__name__}(<unserializable>)"
    PretrainedConfig.__repr__ = _safe_repr

    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    import vibevoice.modular.modular_vibevoice_tokenizer as tok

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_id, language_model_pretrained_name=LM_TOKENIZER)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=device).eval()

    # Deterministic acoustic path (mean). Content is identical to the sampled
    # path; this just removes the VAE noise so the baseline is reproducible.
    def mean_sample(self, dist_type="fix"):
        return self.mean, self.std
    tok.VibeVoiceTokenizerEncoderOutput.sample = mean_sample
    return processor, model


def _parse_hyp(text: str) -> str:
    """Pull the structured JSON the model emits and join its Content fields.
    Falls back to the post-'assistant' text if JSON parsing fails."""
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            parts = [str(seg.get("Content", "")).strip()
                     for seg in arr if isinstance(seg, dict)]
            joined = " ".join(p for p in parts if p)
            if joined:
                return joined
        except Exception:
            pass
    if "assistant" in text:
        text = text.rsplit("assistant", 1)[1]
    return text.strip()


def main() -> int:
    import librosa
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="accepted for the uniform contract; runner is serial")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _ensure_vibevoice_on_path()

    manifest = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    if args.limit:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utts) -> {args.out}", flush=True)

    t0 = time.monotonic()
    processor, model = _load_model(args.model, args.device)
    eos = processor.tokenizer.eos_token_id
    load_ms = round((time.monotonic() - t0) * 1000, 1)
    print(f"loaded in {load_ms/1000:.0f}s", flush=True)

    n_done = n_err = 0
    t_loop = time.monotonic()
    with open(args.out, "w") as fout:
        fout.write(json.dumps({"type": "batch_header", "load_ms": load_ms,
                               "framework": "author_repo_vibevoice",
                               "model": args.model}) + "\n")
        fout.flush()
        for entry in manifest:
            uid = entry["id"]
            ref_text = entry.get("ref_text", "")
            err, hyp_text = "", ""
            t_start = time.monotonic()
            try:
                pcm, _ = librosa.load(entry["audio"], sr=24000, mono=True)
                inputs = processor(audio=[pcm.astype("float32")], sampling_rate=24000,
                                   return_tensors="pt", add_generation_prompt=True)
                inputs = {k: (v.to(args.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, num_beams=1,
                                         eos_token_id=eos, pad_token_id=eos,
                                         stop_strings=["}]"], tokenizer=processor.tokenizer)
                raw = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
                hyp_text = _parse_hyp(raw)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            fout.write(json.dumps({
                "id": uid, "ref_text": ref_text, "hyp_text": hyp_text.strip(),
                "mel_ms": 0, "encode_ms": 0, "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms, "error": err,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 100 == 0:
                rate = n_done / (time.monotonic() - t_loop)
                print(f"[{n_done}/{total}] {rate:.2f} utt/s "
                      f"ETA {(total-n_done)/rate:.0f}s err={n_err}", flush=True)
    print(f"done: {n_done} utts, {n_err} errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
