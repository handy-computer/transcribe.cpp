#!/usr/bin/env python3
"""
dump_reference_parakeet_nemo.py - generate Parakeet TDT reference tensors
from the NVIDIA NeMo implementation (canonical reference).

Run through the repo-local Parakeet reference environment:

    uv run --project scripts/envs/parakeet \
      scripts/dump_reference_parakeet_nemo.py encoder \
      --model nvidia/parakeet-tdt-0.6b-v2 \
      --audio samples/jfk.wav \
      --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/encoder/ref

    uv run --project scripts/envs/parakeet \
      scripts/dump_reference_parakeet_nemo.py decode \
      --model nvidia/parakeet-tdt-0.6b-v2 \
      --audio samples/jfk.wav \
      --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/decode/ref

The --model flag accepts either an HF model name (e.g.
"nvidia/parakeet-tdt-0.6b-v2") or a local path to a .nemo checkpoint.
NeMo downloads and caches the model on first use.

Tensor output uses the shared reference dump contract:

    <name>.f32    raw little-endian float32, row-major
    <name>.json   sidecar metadata (shape, dtype, source provenance)

The decode command also writes transcript.json as a behavioral artifact.

Bridge validation: compare outputs against the existing MLX-based dumps
from scripts/dump_reference.py using scripts/compare_tensors.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (shared contract with dump_reference_cohere_transformers.py)
# ---------------------------------------------------------------------------

def resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


def torch_dtype(name: str):
    import torch

    if name == "f32":
        return torch.float32
    if name == "f16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def configure_torch(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(0)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def write_dump(
    out_dir: Path,
    name: str,
    data: np.ndarray,
    *,
    source: dict[str, Any],
    stage: str,
) -> None:
    """Write a (<name>.f32, <name>.json) pair into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if data.dtype != np.float32:
        raise ValueError(f"only float32 tensors are supported, got {data.dtype}")
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)

    f32_path = out_dir / f"{name}.f32"
    json_path = out_dir / f"{name}.json"
    data.tofile(f32_path)
    meta = {
        "name": name,
        "stage": stage,
        "shape": list(data.shape),
        "dtype": "f32",
        "layout": "row-major",
        "source": source,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {f32_path} ({data.size * 4} bytes)")
    print(f"  wrote {json_path}")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def write_json_artifact(out_dir: Path, name: str, data: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {path}")


def to_np(t) -> np.ndarray:
    """Convert a torch Tensor to contiguous fp32 numpy, squeezing
    leading size-1 dims to match C++ dump conventions."""
    import torch

    if isinstance(t, torch.Tensor):
        a = t.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        a = np.asarray(t, dtype=np.float32)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a, dtype=np.float32)


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# NeMo model loading
# ---------------------------------------------------------------------------

def load_model(args: argparse.Namespace):
    """Load a Parakeet TDT model via NeMo.

    Accepts either an HF model name (e.g. "nvidia/parakeet-tdt-0.6b-v2")
    or a local path to a .nemo file or extracted directory.
    """
    from nemo.collections.asr.models import ASRModel

    model_id = args.model
    local = Path(model_id).expanduser()

    if local.exists():
        print(f"Loading Parakeet TDT from local path: {local}")
        model = ASRModel.restore_from(str(local), map_location="cpu")
    else:
        print(f"Loading Parakeet TDT from HuggingFace: {model_id}")
        model = ASRModel.from_pretrained(model_id, map_location="cpu")

    model.eval()
    return model


def make_source(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
) -> dict[str, Any]:
    import torch

    source: dict[str, Any] = {
        "kind": "parakeet-nemo",
        "model": args.model,
        "model_dtype": args.model_dtype,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
    }
    try:
        import nemo

        source["nemo_version"] = nemo.__version__
    except (ImportError, AttributeError):
        pass
    return source


# ---------------------------------------------------------------------------
# Hook-based intermediate capture
# ---------------------------------------------------------------------------

def capture_intermediates(model, block_set: set[int]):
    """Register forward hooks on key encoder sub-modules.

    Uses hooks rather than manual layer walking so that NeMo's own forward
    pass handles masks, layer-drop, and any internal bookkeeping correctly.

    Returns (intermediates_dict, hook_handle_list).
    """
    intermediates: dict[str, Any] = {}
    hooks = []

    def _hook(name, extract_idx=0):
        def fn(_module, _input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[extract_idx].detach().clone()
            else:
                intermediates[name] = output.detach().clone()

        return fn

    # Preprocessor -> mel spectrogram  (output is (processed_signal, length))
    hooks.append(model.preprocessor.register_forward_hook(_hook("mel", 0)))

    # pre_encode / conv-subsampling  (output is (x, lengths))
    hooks.append(model.encoder.pre_encode.register_forward_hook(_hook("enc.pre_encode.out", 0)))

    # Positional encoding  (output is (x, pos_emb))
    hooks.append(model.encoder.pos_enc.register_forward_hook(_hook("enc.pos_emb", 1)))

    # Per-block conformer layer outputs  (output is x tensor in non-streaming)
    for i in sorted(block_set):
        if i < len(model.encoder.layers):
            hooks.append(
                model.encoder.layers[i].register_forward_hook(_hook(f"enc.block.{i}.out"))
            )

    return intermediates, hooks


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_mel(args: argparse.Namespace) -> int:
    """Dump mel spectrogram from NeMo's AudioToMelSpectrogramPreprocessor."""
    configure_torch(args)
    import torch

    model = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    source = make_source(args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    with torch.inference_mode():
        processed, _proc_len = model.preprocessor(
            input_signal=audio_tensor, length=length_tensor
        )

    # NeMo preprocessor output: (B, n_mels, T). Squeeze batch -> (n_mels, T).
    mel = to_np(processed)
    print(
        f"mel: shape={mel.shape} min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.6f} std={mel.std():.6f}"
    )
    write_dump(out_dir, "enc.mel.in", mel, source=source, stage="frontend.mel.norm")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    """Dump encoder intermediates: mel, pre_encode, pos_emb, per-block, final."""
    configure_torch(args)
    import torch

    model = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    source = make_source(args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr)

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_dump(out_dir, name, a, source=source, stage=stage)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    block_set = set(args.blocks)
    block_set.add(0)  # always include block 0
    intermediates, hooks = capture_intermediates(model, block_set)

    with torch.inference_mode():
        # model.forward() runs preprocessor + spec-aug (no-op in eval) + encoder.
        # NeMo encoder returns (B, D, T) — channels-first.
        encoded, encoded_len = model.forward(
            input_signal=audio_tensor,
            input_signal_length=length_tensor,
        )

    # --- Dump captured intermediates ---

    # Mel: (B, n_mels, T) -> (n_mels, T)
    dump("enc.mel.in", intermediates["mel"], "frontend.mel.norm")

    # Pre-encode output: (B, T_sub, D_model) -> (T_sub, D_model)
    dump("enc.pre_encode.out", intermediates["enc.pre_encode.out"], "encoder.pre_encode")

    # Positional encoding: shape depends on encoding type, dump as-is
    dump("enc.pos_emb", intermediates["enc.pos_emb"], "encoder.pos_emb")

    # Per-block outputs: (B, T_sub, D_model) -> (T_sub, D_model)
    for i in sorted(block_set):
        key = f"enc.block.{i}.out"
        if key in intermediates:
            dump(key, intermediates[key], f"encoder.block{i}.out")

    # Encoder final: encoded is (B, D, T), transpose to (B, T, D) for dump
    enc_final = encoded.transpose(1, 2)
    dump("enc.final", enc_final, "encoder.final")

    for h in hooks:
        h.remove()

    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Dump end-to-end: encoder + decoder first step + joint + greedy transcript.

    First-step intermediates dumped (matching C++ decoder.cpp dump points):

      dec.enc_out          encoder output (T_enc, d_model)
      dec.embed.0          predictor input at step 0 (zeros = start state)
      dec.lstm.<l>.h.0     per-layer LSTM hidden state after one step
      dec.lstm.<l>.c.0     per-layer LSTM cell state after one step
      dec.joint.0          joint network output for (encoder frame 0,
                           predictor start state)
    """
    configure_torch(args)
    import torch

    model = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    source = make_source(args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr)

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_dump(out_dir, name, a, source=source, stage=stage)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    with torch.inference_mode():
        # ----- 1. Encoder -----------------------------------------------
        encoded, encoded_len = model.forward(
            input_signal=audio_tensor,
            input_signal_length=length_tensor,
        )
        # encoded: (B, D, T).  Transpose to (B, T, D) for dumps.
        enc_t = encoded.transpose(1, 2)
        dump("dec.enc_out", enc_t, "decoder.enc_out")

        # ----- 2. Decoder start state -----------------------------------
        # C++ decoder starts with zero input + zero LSTM state, one step.
        # NeMo predict(add_sos=True) prepends a SOS zero vector, making
        # a 2-step sequence — use add_sos=False to get the single-step
        # output that matches C++ and MLX.
        pred_hidden = model.decoder.pred_hidden
        embed_zeros = np.zeros((pred_hidden,), dtype=np.float32)
        dump("dec.embed.0", embed_zeros, "decoder.embed")

        # predict(add_sos=False) returns (g, (h_n, c_n))
        #   g:   (B, 1, pred_hidden)  — single-step output
        #   h_n: (n_layers, B, pred_hidden)
        #   c_n: (n_layers, B, pred_hidden)
        g, hid = model.decoder.predict(
            y=None, state=None, add_sos=False, batch_size=1,
        )

        h_all, c_all = hid
        n_layers = h_all.shape[0]
        for layer in range(n_layers):
            dump(f"dec.lstm.{layer}.h.0", h_all[layer], "decoder.lstm")
            dump(f"dec.lstm.{layer}.c.0", c_all[layer], "decoder.lstm")

        # ----- 3. Joint network -----------------------------------------
        # First encoder frame + start predictor state.
        # joint_after_projection expects ALREADY-PROJECTED inputs
        # (both at joint_hidden dim), so project first via the joint's
        # enc/pred linear layers.
        enc_frame0 = enc_t[:, 0:1, :]  # (B, 1, D_enc)
        enc_proj = model.joint.enc(enc_frame0)  # (B, 1, joint_hidden)
        dec_proj = model.joint.pred(g)  # (B, 1, joint_hidden)
        joint_out = model.joint.joint_after_projection(enc_proj, dec_proj)
        # joint_out: (B, 1, 1, V+1) -> squeeze to (V+1,)
        dump("dec.joint.0", joint_out, "decoder.joint")

    # ----- 4. Full greedy transcription ---------------------------------
    if not args.skip_transcript:
        print("\n  Running full greedy transcription...")
        transcriptions = model.transcribe(
            audio=[str(audio_path)],
            batch_size=1,
        )
        # NeMo transcribe() may return list[str] or (list[Hypothesis], ...)
        if isinstance(transcriptions, (list, tuple)) and len(transcriptions) > 0:
            first = transcriptions[0]
            if isinstance(first, (list, tuple)):
                first = first[0]
            text = first if isinstance(first, str) else first.text
        else:
            text = str(transcriptions)

        print(f"  Transcription: {text}")

        transcript = {
            "schema": "transcribe-reference-transcript-v1",
            "text": text,
            "normalized_text": normalize_text(text),
            "source": source,
        }
        write_json_artifact(out_dir, "transcript.json", transcript)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="HF model name (e.g. nvidia/parakeet-tdt-0.6b-v2) or local .nemo path",
    )
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )
    p.add_argument(
        "--model-dtype",
        choices=["f32", "f16", "bf16"],
        default="f32",
        help="Torch dtype for model weights and compute (default: f32)",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Parakeet TDT reference tensors from NeMo (canonical reference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("mel", help="Dump mel spectrogram only")
    add_common_args(mp)
    mp.set_defaults(func=cmd_mel)

    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    add_common_args(ep)
    ep.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=[0, 12, 23],
        help="Block indices to dump (default: 0 12 23). Block 0 is always included.",
    )
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser("decode", help="Dump encoder + decoder first step + joint + transcript")
    add_common_args(dp)
    dp.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Only dump tensors; do not run full greedy transcription",
    )
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
