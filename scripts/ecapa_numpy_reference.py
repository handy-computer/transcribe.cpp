#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "gguf>=0.10",
#     "soundfile>=0.12",
# ]
# ///
"""
ecapa_numpy_reference.py - dump the NumPy forward pass (scripts/lib/
ecapa_numpy.py) in the same on-disk format as the SpeechBrain reference
dumper and the C++ debug dumper.

This is the middle term of a three-way comparison:

    SpeechBrain  <--compare_tensors-->  NumPy  <--compare_tensors-->  C++
      (the truth)                     (the GGUF)                 (the ship)

The NumPy side reads ONLY the GGUF, so a NumPy-vs-SpeechBrain pass proves
the converter's algebra (BN as affine, the three forward BN folds, the
ASP/MFA weight splits, tap-major kernels), and a C++-vs-NumPy diff after
that isolates graph bugs from conversion bugs.

It deliberately runs on plain `uv run` rather than the pinned SpeechBrain
env: nothing here needs torch, and being able to re-run the reference
forward pass without a 2 GB dependency tree is worth a PEP-723 header.

Usage:

    uv run scripts/ecapa_numpy_reference.py \
      --gguf models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf \
      --audio samples/fleurs-en.wav \
      --out build/validate/ecapa_tdnn/lang-id-voxlingua107-ecapa/fleurs-en/numpy

    uv run scripts/compare_tensors.py \
      build/validate/ecapa_tdnn/lang-id-voxlingua107-ecapa/fleurs-en/numpy \
      build/validate/ecapa_tdnn/lang-id-voxlingua107-ecapa/fleurs-en/ref \
      --tolerances tests/tolerances/ecapa_tdnn.json

Writes `<name>.f32` + `<name>.json` for every stage tensor in the dump contract, plus `prediction.json` in the same shape the SpeechBrain dumper
writes so validate.py can compare `label_index` on either pair.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.ecapa_numpy import EcapaNumpy  # noqa: E402
from lib.ref_dump import write_tensor  # noqa: E402

# Which coarse stage each contract tensor belongs to, for the sidecar's
# `stage` field. Same grouping the SpeechBrain dumper uses so the two sets
# of sidecars are directly comparable.
STAGES = {
    "fe.mel": "frontend",
    "enc.blk.0.out": "encoder",
    "enc.blk.1.tdnn1.out": "encoder",
    "enc.blk.1.res2.out": "encoder",
    "enc.blk.1.se.out": "encoder",
    "enc.blk.1.out": "encoder",
    "enc.blk.2.out": "encoder",
    "enc.blk.3.out": "encoder",
    "enc.mfa.out": "encoder",
    "enc.asp.attn_logits": "encoder",
    "enc.asp.out": "encoder",
    "enc.emb": "encoder",
    "cls.hidden": "classifier",
    "cls.logits_raw": "classifier",
    "cls.log_probs": "classifier",
}


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if int(sr) != sample_rate:
        raise SystemExit(
            f"error: {path} is {sr} Hz; this model's front end is "
            f"{sample_rate} Hz only. Resample before dumping.")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Dump the NumPy ECAPA-TDNN forward pass from a GGUF.")
    p.add_argument("--gguf", required=True, help="ecapa_tdnn GGUF path")
    p.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    p.add_argument("--out", required=True, help="output directory for dumps")
    args = p.parse_args(argv)

    gguf_path = Path(args.gguf).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = EcapaNumpy(gguf_path)
    pcm = load_audio(audio_path, model.sample_rate)
    n_samples = int(pcm.size)
    expected_T = n_samples // model.hop + 1
    print(f"audio: {audio_path.name} samples={n_samples} "
          f"({n_samples / model.sample_rate:.3f} s) frames={expected_T}")

    result = model.forward(pcm)

    T = int(result["fe.mel"].shape[0])
    if T != expected_T:
        raise SystemExit(
            f"error: fe.mel has T={T} but floor({n_samples}/{model.hop}) + 1 "
            f"= {expected_T}")

    # The dtype mix actually used for this run. Recorded per dump so an
    # F16/Q8_0 comparison is never mistaken for an F32 parity number.
    quant_kinds = sorted(set(model.tensor_types.values()))
    source = {
        "kind": "ecapa-tdnn-numpy",
        "framework": "numpy",
        "framework_version": np.__version__,
        "gguf": str(gguf_path),
        "gguf_file_type": int(model.meta.get("general.file_type", -1)),
        "gguf_tensor_types": quant_kinds,
        "variant": model.meta.get("stt.variant"),
        "source_hf_repo": model.meta.get("general.name"),
        "source_hf_revision": model.meta.get("general.source.commit"),
        "audio": audio_path.name,
        "n_samples": n_samples,
        "sample_rate": model.sample_rate,
        "n_frames": T,
        "n_mels": model.n_mels,
        "compute_dtype": "f32",
    }

    missing = [n for n in STAGES if n not in result]
    if missing:
        raise SystemExit(f"error: forward() did not return {missing}")

    for name, stage in STAGES.items():
        arr = np.ascontiguousarray(result[name], dtype=np.float32)
        print(f"  {name}: shape={arr.shape} min={arr.min():.4e} "
              f"max={arr.max():.4e} mean={arr.mean():.6e}")
        write_tensor(name, arr, stage=stage, source=source, out_dir=out_dir)

    pred = dict(result["prediction"])
    pred["source"] = source
    (out_dir / "prediction.json").write_text(json.dumps(pred, indent=2) + "\n")
    print(f"prediction: {pred['code']} ({pred['name']}) "
          f"log_prob={pred['log_prob']:.6f} "
          f"prob={math.exp(pred['log_prob']):.4f}")
    print("top5: " + ", ".join(f"{e['code']}={e['log_prob']:.3f}"
                               for e in pred["top5"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
