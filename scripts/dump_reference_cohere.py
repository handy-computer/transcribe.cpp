#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx>=0.20; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "mlx-audio>=0.2; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "sentencepiece>=0.2",
#     "soundfile>=0.12",
#     "numpy>=1.26",
#     "safetensors>=0.4",
# ]
# ///
"""
dump_reference_cohere.py - generate per-stage reference tensors for
Cohere ASR numerical accuracy validation in transcribe.cpp.

Uses the MLX reference implementation (mlx-audio) as the source of
truth. The HF model on CPU produces gibberish; the MLX implementation
is the known-working reference. Apple Silicon only.

Output format matches the C++ TensorDumper (see src/transcribe-debug.h):

    <name>.f32    raw little-endian float32, row-major
    <name>.json   sidecar metadata (shape, dtype, source provenance)

Usage:
    uv run scripts/dump_reference_cohere.py encoder \\
        --model ~/sandboxes/transcribe/models/cohere-transcribe-03-2026 \\
        --audio samples/jfk.wav \\
        --out ~/sandboxes/transcribe/dumps/ref

    uv run scripts/dump_reference_cohere.py decode \\
        --model ~/sandboxes/transcribe/models/cohere-transcribe-03-2026 \\
        --audio samples/jfk.wav \\
        --out ~/sandboxes/transcribe/dumps/ref \\
        --language en
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def write_dump(
    out_dir: Path,
    name: str,
    data: np.ndarray,
    *,
    source: dict,
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


def to_np(t) -> np.ndarray:
    """Convert an MLX array to contiguous fp32 numpy, squeezing
    leading size-1 dims to match C++ dump conventions."""
    import mlx.core as mx
    # Cast to fp32 in MLX first (avoids numpy bf16 issues)
    if t.dtype != mx.float32:
        t = t.astype(mx.float32)
    mx.eval(t)
    a = np.array(t, copy=False)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a)


def load_model(model_dir: Path, *, match_f16: bool = False):
    """Load the Cohere ASR model via mlx-audio.

    When match_f16=True, weights are round-tripped through float16 to
    match the precision of the GGUF converter (bf16 -> f16 -> f32).
    This gives a fair comparison against C++ tensor dumps.
    """
    import mlx.core as mx
    from mlx_audio.stt import load_model as mlx_load_model

    label = "f16-matched" if match_f16 else "fp32"
    print(f"Loading Cohere ASR model from {model_dir} (MLX, {label})...")
    model = mlx_load_model(str(model_dir))

    def cast_param(v):
        if not hasattr(v, 'astype'):
            return v
        if match_f16:
            return v.astype(mx.float16).astype(mx.float32)
        return v.astype(mx.float32)

    fp32_params = {}
    for k, v in model.parameters().items():
        if isinstance(v, dict):
            fp32_params[k] = {kk: cast_param(vv) for kk, vv in v.items()}
        else:
            fp32_params[k] = cast_param(v)
    model.update(fp32_params)

    return model


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a wav file as mono float32 at native sample rate."""
    import soundfile as sf
    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return pcm, int(sr)


def cmd_encoder(args: argparse.Namespace) -> int:
    """Dump encoder intermediates: mel, pre_encode, pos_emb, per-block, encoder final, enc-dec proj."""
    import mlx.core as mx

    model_dir = Path(args.model).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser()

    model = load_model(model_dir)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    source = {
        "kind": "cohere-mlx",
        "model": model_dir.name,
        "audio": audio_path.name,
        "n_samples": int(pcm.size),
        "sample_rate": sr,
    }

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.6f}")
        write_dump(out_dir, name, a, source=source, stage=stage)

    # 1. Mel frontend
    input_features, lengths = model.audio_frontend([pcm])
    dump("enc.mel.in", input_features, "frontend.mel.norm")

    # Cast to model dtype if needed
    conv_weight = model.encoder.pre_encode.out.weight
    if input_features.dtype != conv_weight.dtype:
        input_features = input_features.astype(conv_weight.dtype)

    encoder = model.encoder

    # 2. Pre-encode (conv subsampling)
    x, length_out = encoder.pre_encode(input_features, lengths)
    dump("enc.pre_encode.out", x, "encoder.pre_encode")

    # 3. Positional encoding
    x, pos_emb = encoder.pos_enc(x)
    dump("enc.pos_emb", pos_emb, "encoder.pos_emb")

    # 4. Create masks (must match encoder._create_masks exactly)
    pad_mask, attention_mask = encoder._create_masks(
        length_out, x.shape[1], x.dtype
    )

    # 5. Per-block intermediates
    block_set = set(args.blocks)
    full_blocks = {0}

    for i, layer in enumerate(encoder.layers):
        x = layer(x, pos_emb, attention_mask=attention_mask, pad_mask=pad_mask)
        if i in block_set or i in full_blocks:
            dump(f"enc.block.{i}.out", x, f"encoder.block{i}.out")

    # 6. Final encoder output
    dump("enc.final", x, "encoder.final")

    # 7. Encoder-decoder projection
    if model.encoder_decoder_proj is not None:
        enc_proj = model.encoder_decoder_proj(x)
        dump("enc_dec_proj.out", enc_proj, "encoder.enc_dec_proj")

    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Dump end-to-end: encoder + decoder first steps + logits + greedy output."""
    import mlx.core as mx

    model_dir = Path(args.model).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser()
    language = args.language or "en"

    model = load_model(model_dir)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    source = {
        "kind": "cohere-mlx",
        "model": model_dir.name,
        "audio": audio_path.name,
        "n_samples": int(pcm.size),
        "sample_rate": sr,
        "language": language,
    }

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_dump(out_dir, name, a, source=source, stage=stage)

    # 1. Encode
    encoder_hidden, encoder_lengths, encoder_mask = model._encode_waveforms([pcm])
    dump("enc.final", encoder_hidden, "encoder.final")

    # 2. Build prompt tokens
    tokenizer = model._tokenizer
    prompt_tokens = tokenizer.build_prompt_tokens(language, punctuation=True)
    prompt_str = tokenizer.decode(prompt_tokens)
    print(f"  prompt: {prompt_str}")
    print(f"  prompt_ids: {prompt_tokens}")

    # 3. Run decoder on prompt
    prompt_ids = mx.array([prompt_tokens], dtype=mx.int32)
    logits, cache = model.transf_decoder(
        prompt_ids,
        encoder_hidden,
        encoder_mask=encoder_mask,
        cache=None,
        start_pos=0,
    )
    dump("dec.out_before_head", logits, "decoder.output_before_head")

    logits = model.log_softmax(logits)
    dump("dec.logits", logits, "decoder.logits")

    # 4. First predicted token
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)
    pred_token_id = int(mx.argmax(last_logits).item())
    pred_token = tokenizer.decode([pred_token_id])
    print(f"  first predicted token: id={pred_token_id} text='{pred_token}'")

    # 5. Full greedy decode for verification
    print("\n  Running full greedy decode...")
    output = model.transcribe(
        language=language,
        audio_files=[str(audio_path)],
    )
    print(f"  Transcription: {output[0]}")

    return 0


def cmd_mel(args: argparse.Namespace) -> int:
    """Dump just the mel spectrogram."""
    model_dir = Path(args.model).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser()

    model = load_model(model_dir)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    input_features, lengths = model.audio_frontend([pcm])
    mel_np = to_np(input_features)

    print(f"mel: shape={mel_np.shape} min={mel_np.min():.4f} max={mel_np.max():.4f} "
          f"mean={mel_np.mean():.6f} std={mel_np.std():.6f}")

    source = {
        "kind": "cohere-mlx",
        "model": model_dir.name,
        "audio": audio_path.name,
        "n_samples": int(pcm.size),
        "sample_rate": sr,
    }
    write_dump(out_dir, "enc.mel.in", mel_np, source=source, stage="frontend.mel.norm")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump per-stage reference tensors for Cohere ASR (MLX, Apple Silicon only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # mel subcommand
    mp = sub.add_parser("mel", help="Dump mel spectrogram only")
    mp.add_argument("--model", required=True, help="Path to Cohere ASR HF model directory")
    mp.add_argument("--audio", required=True, help="16 kHz mono wav file")
    mp.add_argument("--out", required=True, help="Output directory")
    mp.set_defaults(func=cmd_mel)

    # encoder subcommand
    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    ep.add_argument("--model", required=True, help="Path to Cohere ASR HF model directory")
    ep.add_argument("--audio", required=True, help="16 kHz mono wav file")
    ep.add_argument("--out", required=True, help="Output directory")
    ep.add_argument("--blocks", type=int, nargs="*", default=[0, 23, 47],
                    help="Block indices to dump (default: 0 23 47)")
    ep.set_defaults(func=cmd_encoder)

    # decode subcommand
    dp = sub.add_parser("decode", help="Dump end-to-end: encoder + decoder + logits")
    dp.add_argument("--model", required=True, help="Path to Cohere ASR HF model directory")
    dp.add_argument("--audio", required=True, help="16 kHz mono wav file")
    dp.add_argument("--out", required=True, help="Output directory")
    dp.add_argument("--language", default="en", help="Language code (default: en)")
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
