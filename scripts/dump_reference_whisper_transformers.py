#!/usr/bin/env python3
"""
dump_reference_whisper_transformers.py - generate Whisper reference
tensors from the Hugging Face Transformers implementation
(`WhisperForConditionalGeneration`).

Run through the repo-local Whisper reference environment:

    uv run --project scripts/envs/whisper \
      scripts/dump_reference_whisper_transformers.py encoder \
      --model openai/whisper-tiny \
      --audio samples/jfk.wav \
      --out build/validate/whisper/whisper-tiny/jfk/encoder/ref

Tensor output uses the shared reference dump contract. Tensors are
written in the reference implementation's natural row-major shape with a
leading batch dim squeezed; C++ dumpers are expected to match this
contract, not the other way around.

    <name>.f32    raw little-endian float32, row-major
    <name>.json   sidecar metadata

The decode command also writes transcript.json as a behavioral artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.float32:
        return "f32"
    return str(dtype).removeprefix("torch.")


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
        "min": float(data.min()) if data.size else 0.0,
        "max": float(data.max()) if data.size else 0.0,
        "mean": float(data.mean(dtype=np.float64)) if data.size else 0.0,
        "source": source,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {f32_path} ({data.size * 4} bytes)")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def write_json_artifact(out_dir: Path, name: str, data: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {path}")


def to_np(t) -> np.ndarray:
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


def resolve_model(raw: str) -> tuple[str, bool]:
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


def load_reference(args: argparse.Namespace):
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Whisper model from {model_id} ({source}, "
        f"Transformers {transformers.__version__}, device={args.device})..."
    )
    processor = WhisperProcessor.from_pretrained(
        model_id,
        local_files_only=local_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=local_only,
    ).eval()
    model.to(args.device)
    return processor, model


def make_source(
    *,
    args: argparse.Namespace,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    language: str | None = None,
    task: str | None = None,
    model_dtype: str | None = None,
) -> dict[str, Any]:
    import torch
    import transformers

    source: dict[str, Any] = {
        "kind": "whisper-transformers-native",
        "transformers_version": transformers.__version__,
        "transformers_file": transformers.__file__,
        "model": model_id,
        "model_dtype": model_dtype,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
    }
    if language is not None:
        source["language"] = language
    if task is not None:
        source["task"] = task
    return source


def frontend_inputs(processor, pcm: np.ndarray, sr: int, device: str):
    """Run WhisperFeatureExtractor and move tensors to device.

    WhisperProcessor expects 16 kHz mono audio. The feature extractor pads or
    trims to a fixed 30-second window (3000 mel frames). Output layout is
    [batch, n_mels=80, n_frames=3000].
    """
    result = processor(
        audio=pcm,
        sampling_rate=sr,
        return_tensors="pt",
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in result.items()}


def build_prompt_ids(
    processor,
    language: str,
    task: str,
    *,
    no_timestamps: bool = True,
) -> list[int]:
    """Build the Whisper decoder prompt: [SOT, <|lang|>, <|task|>, <|notimestamps|>].

    For whisper-tiny (multilingual) this is 4 tokens. The English-only `.en`
    variants drop the language token.
    """
    config = processor.tokenizer
    decoder_start = config.convert_tokens_to_ids("<|startoftranscript|>")
    if decoder_start is None or decoder_start < 0:
        decoder_start = 50258
    ids: list[int] = [decoder_start]

    lang_token = f"<|{language}|>"
    lang_id = config.convert_tokens_to_ids(lang_token)
    if lang_id is not None and lang_id >= 0:
        ids.append(int(lang_id))

    task_token = f"<|{task}|>"
    task_id = config.convert_tokens_to_ids(task_token)
    if task_id is not None and task_id >= 0:
        ids.append(int(task_id))

    if no_timestamps:
        nots_id = config.convert_tokens_to_ids("<|notimestamps|>")
        if nots_id is not None and nots_id >= 0:
            ids.append(int(nots_id))

    return ids


def dump_encoder(
    *,
    model,
    inputs: dict[str, Any],
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int],
):
    import torch
    import torch.nn.functional as F

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )
        write_dump(out_dir, name, a, source=source, stage=stage)

    encoder = model.model.encoder
    input_features = inputs["input_features"]

    # Mel frontend input: transpose to [n_mels, n_frames] matching C++ dump layout.
    dump("enc.mel.in", input_features.transpose(-2, -1), "frontend.mel.norm")

    with torch.inference_mode():
        # conv1 (kernel=3, stride=1) + GELU  -> [B, d_model, n_frames]
        x = F.gelu(encoder.conv1(input_features))
        dump("enc.conv1.out", x.transpose(-2, -1), "encoder.conv1")

        # conv2 (kernel=3, stride=2) + GELU  -> [B, d_model, n_frames/2]
        x = F.gelu(encoder.conv2(x))
        dump("enc.conv2.out", x.transpose(-2, -1), "encoder.conv2")

        # permute to [B, T, d_model]
        x = x.permute(0, 2, 1)

        # Sinusoidal positional embedding: precomputed weight [max_source_positions, d_model].
        embed_pos = encoder.embed_positions.weight
        dump("enc.pos_emb", embed_pos, "encoder.pos_emb")

        x = x + embed_pos
        dump("enc.embed.out", x, "encoder.embed")

        # Transformer blocks.
        block_set = set(blocks)
        block_set.add(0)
        block_set.add(len(encoder.layers) - 1)
        for i, layer in enumerate(encoder.layers):
            # transformers 5.x: WhisperEncoderLayer.forward returns a bare tensor.
            x = layer(x, attention_mask=None)
            if i in block_set:
                dump(f"enc.block.{i}.out", x, f"encoder.block{i}.out")

        # Final layer norm.
        x = encoder.layer_norm(x)
        dump("enc.final", x, "encoder.final")

    return x


def dump_decoder(
    *,
    model,
    encoder_hidden,
    prompt_ids: list[int],
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int],
    device: str,
):
    import torch

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )
        write_dump(out_dir, name, a, source=source, stage=stage)

    decoder = model.model.decoder
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    print(f"  prompt_ids: {prompt_ids}")

    with torch.inference_mode():
        # Token embedding (shared with lm_head via tied weights).
        tok_emb = decoder.embed_tokens(input_ids)
        dump("dec.token_emb", tok_emb, "decoder.embedding")

        # Learned positional embedding. WhisperPositionalEmbedding returns
        # weight[past_len:past_len+seq_len] given (input_ids, past_key_values_length).
        pos_emb = decoder.embed_positions(input_ids, past_key_values_length=0)
        dump("dec.pos_emb", pos_emb, "decoder.position_embedding")

        x = tok_emb + pos_emb
        dump("dec.embed_sum", x, "decoder.embed_sum")

        # In transformers 5.x, WhisperAttention self-attn sets is_causal=True when
        # is_decoder=True, so the SDPA backend generates the causal mask internally
        # for prefill (no past KV). Passing attention_mask=None is correct here.
        block_set = set(blocks)
        block_set.add(0)
        block_set.add(len(decoder.layers) - 1)
        for i, layer in enumerate(decoder.layers):
            # transformers 5.x: WhisperDecoderLayer.forward returns a bare tensor.
            x = layer(
                x,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )
            if i in block_set:
                dump(f"dec.block.{i}.out", x, f"decoder.block{i}.out")

        # Final layer norm.
        x = decoder.layer_norm(x)
        dump("dec.out_before_head", x, "decoder.output_before_head")

        # Tied lm_head projection to vocab logits.
        logits_raw = model.proj_out(x)
        dump("dec.logits_raw", logits_raw, "decoder.logits_raw")

        log_probs = torch.log_softmax(logits_raw, dim=-1)
        dump("dec.logits", log_probs, "decoder.logits")

    last_logits = logits_raw[0, -1, :]
    return int(torch.argmax(last_logits).item())


def dump_mid_generation(
    *,
    model,
    encoder_hidden,
    prompt_ids: list[int],
    out_dir: Path,
    source: dict[str, Any],
    device: str,
    gen_step_n: int = 20,
):
    """Greedy-decode `gen_step_n` tokens and dump the logits that would
    predict token `gen_step_n` (i.e. the step graph's output at the
    `gen_step_n`-th autoregressive step, matching the C++ runner's
    step-loop iteration at `step == gen_step_n - 1`).

    Exercises the n_past > 0 path — the prompt-pass dumps only cover
    n_past == 0, so without this the KV cache update code has zero
    tensor-level coverage.

    Greedy logic mirrors the C++ driver exactly: apply suppress_tokens
    every step, apply begin_suppress_tokens at the first generated
    token, argmax.
    """
    import torch

    gc = model.generation_config
    suppress = list(gc.suppress_tokens or [])
    begin_suppress = list(gc.begin_suppress_tokens or [])
    vocab = int(model.config.vocab_size)

    decoder = model.model.decoder
    current = list(prompt_ids)

    def forward_last_logits(ids: list[int]):
        inp = torch.tensor([ids], device=device, dtype=torch.long)
        out = decoder(
            input_ids=inp,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        )
        return model.proj_out(out.last_hidden_state[0, -1, :])

    with torch.inference_mode():
        # Step 0: the prompt-pass already ran; redo its last-row logits
        # here so we can apply the same suppress rules the C++ path
        # applies before the argmax.
        step_logits = forward_last_logits(current).clone()
        for idx in suppress:
            if 0 <= idx < vocab:
                step_logits[idx] = float("-inf")
        for idx in begin_suppress:
            if 0 <= idx < vocab:
                step_logits[idx] = float("-inf")
        next_id = int(torch.argmax(step_logits).item())

        # Steps 1..gen_step_n-1: append prev next_id, forward, suppress
        # (no begin-suppress after the first generated token), argmax.
        for _ in range(1, gen_step_n):
            current.append(next_id)
            step_logits = forward_last_logits(current).clone()
            for idx in suppress:
                if 0 <= idx < vocab:
                    step_logits[idx] = float("-inf")
            next_id = int(torch.argmax(step_logits).item())

        # One more forward with [prompt + generated[0..gen_step_n-1]]
        # so we capture the pre-argmax logits that would pick
        # generated[gen_step_n]. This matches the C++ runner's
        # step_db.out at step == gen_step_n - 1 after compute.
        current.append(next_id)
        inp = torch.tensor([current], device=device, dtype=torch.long)
        out = decoder(
            input_ids=inp,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        )
        logits_final = model.proj_out(out.last_hidden_state[0, -1:, :])

    name = f"dec.logits_raw.gen{gen_step_n}"
    stage = f"decoder.logits_raw.gen{gen_step_n}"
    a = to_np(logits_final)
    print(
        f"  {name}: shape={a.shape} "
        f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
    )
    write_dump(out_dir, name, a, source=source, stage=stage)


def generate_transcript(
    *,
    model,
    processor,
    inputs: dict[str, Any],
    language: str,
    task: str,
    source: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
        no_timestamps=True,
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            input_features=inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    tokenizer = processor.tokenizer
    generated_list = generated_ids[0].detach().cpu().tolist()
    token_ids = [t for t in generated_list if t != tokenizer.eos_token_id]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    return {
        "schema": "transcribe-reference-transcript-v1",
        "text": text,
        "normalized_text": normalize_text(text),
        "token_ids": token_ids,
        "generated_ids": generated_list,
        "language": language,
        "task": task,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "source": source,
    }


def cmd_mel(args: argparse.Namespace) -> int:
    configure_torch(args)
    import transformers
    from transformers import WhisperProcessor

    model_id, local_only = resolve_model(args.model)
    source_label = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Whisper processor from {model_id} ({source_label}, "
        f"Transformers {transformers.__version__}, device={args.device})..."
    )
    processor = WhisperProcessor.from_pretrained(
        model_id,
        local_files_only=local_only,
    )

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    inputs = frontend_inputs(processor, pcm, sr, args.device)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
    )
    mel = to_np(inputs["input_features"])
    if mel.ndim == 2:
        mel = mel.T  # [n_mels, n_frames]
    print(
        f"mel: shape={mel.shape} min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.6f} std={mel.std():.6f}"
    )
    write_dump(out_dir, "enc.mel.in", mel, source=source, stage="frontend.mel.norm")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    configure_torch(args)
    processor, model = load_reference(args)

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    model_id, _ = resolve_model(args.model)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    inputs = frontend_inputs(processor, pcm, sr, args.device)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype_name(model),
    )
    dump_encoder(
        model=model,
        inputs=inputs,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks),
    )
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    configure_torch(args)
    processor, model = load_reference(args)

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    model_id, _ = resolve_model(args.model)
    language = args.language or "en"
    task = args.task or "transcribe"
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    inputs = frontend_inputs(processor, pcm, sr, args.device)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        language=language,
        task=task,
        model_dtype=model_dtype_name(model),
    )

    encoder_hidden = dump_encoder(
        model=model,
        inputs=inputs,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks),
    )

    prompt_ids = build_prompt_ids(processor, language, task, no_timestamps=True)
    pred_id = dump_decoder(
        model=model,
        encoder_hidden=encoder_hidden,
        prompt_ids=prompt_ids,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks),
        device=args.device,
    )
    pred_token = processor.tokenizer.decode([pred_id])
    print(f"  first predicted token: id={pred_id} text={pred_token!r}")

    dump_mid_generation(
        model=model,
        encoder_hidden=encoder_hidden,
        prompt_ids=prompt_ids,
        out_dir=out_dir,
        source=source,
        device=args.device,
        gen_step_n=20,
    )

    if not args.skip_transcript:
        transcript = generate_transcript(
            model=model,
            processor=processor,
            inputs=inputs,
            language=language,
            task=task,
            source=source,
            args=args,
        )
        print(f"  transcript: {transcript['text']!r}")
        write_json_artifact(out_dir, "transcript.json", transcript)

    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id or local path to the Whisper model directory")
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument("--language", default="en", help="Language code (default: en)")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Whisper reference tensors from Transformers.",
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
        default=[0, 1, 2, 3],
        help="Block indices to dump (default: all 4 blocks for whisper-tiny)",
    )
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser("decode", help="Dump encoder + decoder prompt logits + transcript")
    add_common_args(dp)
    dp.add_argument("--task", default="transcribe", help="transcribe or translate (default: transcribe)")
    dp.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated transcript tokens (default: 256)",
    )
    dp.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Only dump tensors; do not generate transcript.json",
    )
    dp.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Block indices to dump (default: all 4 blocks for whisper-tiny)",
    )
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
