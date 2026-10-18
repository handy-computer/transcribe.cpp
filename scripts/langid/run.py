#!/usr/bin/env python3
"""
run.py - score language ID manifests with the SpeechBrain reference or
transcribe.cpp (through the Python binding) into one JSONL sweep: a header
line, then one row per (utterance, crop) with all 107 pre-softmax logits.

  uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py \\
      --engine speechbrain --model speechbrain/lang-id-voxlingua107-ecapa \\
      --manifest samples/langid/fleurs-*.manifest.jsonl \\
      --out reports/langid/ref-speechbrain-untrimmed.jsonl

  uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py \\
      --engine cpp --model models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf \\
      --library build-shared/src/libtranscribe.dylib \\
      --manifest samples/langid/fleurs-*.manifest.jsonl \\
      --out reports/langid/cpp-f32-untrimmed.jsonl

A crop of N is the first N seconds of the clip; `full` is the whole clip, or
its first MAX_AUDIO_S when longer (transcribe.cpp always scores only the first
transcribe_langid_info::max_audio_ms, so the script cuts it for both
engines). A row's `audio_s` is what was scored. Both engines get the same
float samples (the clips are 16-bit, so a crop is exact). SpeechBrain runs a
batch of one: padding a batch changes its normalisation span.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SAMPLE_RATE = 16000
# transcribe_langid_info::max_audio_ms of the VoxLingua107 model: longer input
# is scored on its first MAX_AUDIO_S.
MAX_AUDIO_S = 30


class SpeechBrainEngine:
    """The pinned checkpoint; logits from a forward hook on the pre-softmax
    `mods.classifier.out` (classify_batch only returns log-softmax)."""

    def __init__(self, model: str, threads: int) -> None:
        import speechbrain
        import torch

        import dump_reference_ecapa_tdnn_speechbrain as dumper

        self.torch = torch
        args = argparse.Namespace(model=model, revision=dumper.DEFAULT_REVISION,
                                  device="cpu", torch_threads=threads)
        dumper.configure_torch(args)
        self.clf, checkpoint_dir = dumper.load_reference(args)
        ind2lab = self.clf.hparams.label_encoder.ind2lab
        self.labels = [dumper.split_label(str(ind2lab[i]))[0] for i in range(len(ind2lab))]
        self._captured: list = []
        self.clf.mods.classifier.out.register_forward_hook(
            lambda _m, _i, out: self._captured.append(out))
        self.recipe = {
            "revision": dumper.DEFAULT_REVISION,
            "checkpoint_dir": str(checkpoint_dir),
            "device": "cpu",
            "model_dtype": "f32",
            "batch_size": 1,
            "torch_threads": threads,
            "speechbrain": speechbrain.__version__,
            "torch": torch.__version__,
        }

    def logits(self, pcm: np.ndarray) -> np.ndarray:
        torch = self.torch
        self._captured.clear()
        # wav_lens is relative to the padded batch: 1.0 for one unpadded clip.
        with torch.inference_mode():
            self.clf.classify_batch(torch.from_numpy(pcm).unsqueeze(0), torch.ones(1))
        (out,) = self._captured
        return out.detach().to(torch.float32).reshape(len(self.labels)).numpy()


class CppEngine:
    """The GGUF loaded once; each clip is one unrestricted LangIdSession.run,
    so every label's logit comes back."""

    def __init__(self, model: Path, backend: str, threads: int, library: Path | None) -> None:
        if library is not None:
            os.environ["TRANSCRIBE_LIBRARY"] = str(library.resolve())
        sys.path.insert(0, str(REPO_ROOT / "bindings" / "python" / "src"))
        import transcribe_cpp
        from gguf import GGUFReader

        self.model = transcribe_cpp.Model(str(model), backend=backend)
        self.session = self.model.langid_session(n_threads=threads)
        self.labels = [code for code, _name in self.model.langid_labels]
        fields = GGUFReader(str(model)).fields
        kv = lambda key: str(fields[key].contents()) if key in fields else None  # noqa: E731
        with open(model, "rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        self.recipe = {
            "backend": backend,
            "threads": threads,
            "gguf_sha256": sha,
            "gguf_source_commit": kv("general.source.commit"),
            "gguf_source_repo": kv("general.name"),
            "native_version": transcribe_cpp.native_version(),
            "native_commit": transcribe_cpp.native_commit(),
            "backend_bound": self.model.backend,
        }

    def logits(self, pcm: np.ndarray) -> np.ndarray:
        logits = np.zeros(len(self.labels), dtype=np.float32)
        for c in self.session.run(pcm).candidates:
            logits[c.index] = c.logit
        return logits


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE or pcm.ndim != 1:
        raise SystemExit(f"error: {path} is not 16 kHz mono")
    return pcm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, choices=("speechbrain", "cpp"))
    p.add_argument("--model", required=True,
                   help="HF repo id / checkpoint dir (speechbrain) or GGUF path (cpp)")
    p.add_argument("--manifest", required=True, nargs="+", type=Path)
    p.add_argument("--crops", default="3,5,10,full")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--library", type=Path,
                   help="cpp only: a shared libtranscribe (TRANSCRIBE_LIBRARY)")
    p.add_argument("--backend", default="cpu", help="cpp only")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-utts", type=int, default=0,
                   help="cap utterances per manifest (0 = all); for smoke tests")
    args = p.parse_args(argv)

    crops = [c if c == "full" else int(c) for c in args.crops.split(",")]
    rows = []
    for m in args.manifest:
        lines = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
        rows += lines[: args.max_utts or None]
    longest_s = max(r["duration_s"] for r in rows)

    if args.engine == "speechbrain":
        engine = SpeechBrainEngine(args.model, args.threads)
    else:
        engine = CppEngine(Path(args.model), args.backend, args.threads, args.library)

    header = {
        "type": "header",
        "engine": args.engine,
        "model": args.model,
        "recipe": {
            "crops": [str(c) for c in crops],
            "max_audio_s": MAX_AUDIO_S,
            "manifests": [str(m) for m in args.manifest],
            "manifest_sha256": {str(m): hashlib.sha256(m.read_bytes()).hexdigest()
                                for m in args.manifest},
            "n_utterances": len(rows),
            "sample_rate": SAMPLE_RATE,
            "logit_kind": "pre_softmax",
            "longest_clip_s": longest_s,
            **engine.recipe,
        },
        "labels": engine.labels,
        "created": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "w") as f:
        f.write(json.dumps(header) + "\n")
        for crop in crops:
            for i, row in enumerate(rows):
                pcm = read_audio(REPO_ROOT / row["audio"])
                if crop != "full":
                    pcm = pcm[: crop * SAMPLE_RATE]
                else:
                    pcm = pcm[:MAX_AUDIO_S * SAMPLE_RATE]
                logits = engine.logits(pcm)
                z = np.exp(logits.astype(np.float64) - logits.max())
                top = int(np.argmax(z))
                f.write(json.dumps({
                    "id": row["id"],
                    "language": row["language"],
                    "crop_s": crop,
                    "audio_s": round(pcm.size / SAMPLE_RATE, 4),
                    "logits": [round(float(x), 6) for x in logits],
                    "top1": engine.labels[top],
                    "top1_prob": round(float(z[top] / z.sum()), 6),
                }) + "\n")
                if (i + 1) % 200 == 0 or i + 1 == len(rows):
                    print(f"  crop {crop}: {i + 1}/{len(rows)}  {time.time() - t0:.0f}s",
                          file=sys.stderr)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
