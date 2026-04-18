#!/usr/bin/env python3
"""
dump_reference_cohere_transformers.py - generate Cohere ASR reference
tensors from the native Hugging Face Transformers implementation.

Run through the repo-local Cohere reference environment:

    uv run --project scripts/envs/cohere \
      scripts/dump_reference_cohere_transformers.py encoder \
      --model ../models/cohere-transcribe-03-2026 \
      --transformers-ref ../refs/huggingface/transformers \
      --audio samples/jfk.wav \
      --out build/validate/cohere/cohere-transcribe-03-2026/jfk/encoder/ref

This intentionally uses trust_remote_code=False. The remote-code path is
kept out of the golden path because it has produced garbage generations
with some Transformers/version combinations.

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


def use_transformers_ref(path: Path | None) -> None:
    if path is None:
        env = os.environ.get("TRANSCRIBE_TRANSFORMERS_REF")
        path = resolve_path(env) if env else None
    if path is None:
        return
    src = path / "src"
    if not src.is_dir():
        raise SystemExit(f"error: transformers ref has no src directory: {src}")
    sys.path.insert(0, str(src))


def model_dtype_name(model) -> str:
    """Return a canonical short dtype string for the loaded model (bf16, f32, f16, ...)."""
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
    print(f"  wrote {json_path}")


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
    """Resolve a model argument to (model_id_or_path, local_files_only).

    If the raw string points to an existing local directory, resolve it
    and use local_files_only=True. Otherwise treat it as a HuggingFace
    model ID and let the library download it.
    """
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


def load_reference(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Cohere ASR model from {model_id} ({source}, "
        f"Transformers {transformers.__version__}, device={args.device})..."
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=local_only,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        trust_remote_code=False,
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
    model_dtype: str | None = None,
) -> dict[str, Any]:
    import torch
    import transformers

    source: dict[str, Any] = {
        "kind": "cohere-transformers-native",
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
    return source


def frontend_inputs(processor, pcm: np.ndarray, sr: int, device: str, language: str):
    result = processor(
        audio=[pcm],
        language=language,
        sampling_rate=sr,
        return_tensors="pt",
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in result.items()}


def dump_encoder(
    *,
    model,
    inputs: dict[str, Any],
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int],
) -> tuple[Any, Any, Any | None]:
    import torch

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )
        write_dump(out_dir, name, a, source=source, stage=stage)

    encoder = model.model.encoder
    input_features = inputs["input_features"]
    attention_mask = inputs.get("attention_mask")

    # Transpose to [num_mels, n_frames] to match C++ dump layout.
    dump("enc.mel.in", input_features.transpose(-2, -1), "frontend.mel.norm")

    conv_dtype = encoder.subsampling.layers[0].weight.dtype
    if input_features.dtype != conv_dtype:
        input_features = input_features.to(dtype=conv_dtype)

    with torch.inference_mode():
        x = encoder.subsampling(input_features, attention_mask)
        dump("enc.pre_encode.out", x, "encoder.pre_encode")

        pos_emb = encoder.encode_positions(x)
        dump("enc.pos_emb", pos_emb, "encoder.pos_emb")

        output_mask = None
        att_mask = None
        if attention_mask is not None:
            output_mask = encoder._get_output_attention_mask(
                attention_mask,
                target_length=x.shape[1],
            )
            att_mask = output_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
            att_mask = att_mask & att_mask.transpose(1, 2)
            att_mask = att_mask.unsqueeze(1)

        block_set = set(blocks)
        block_set.add(0)
        for i, layer in enumerate(encoder.layers):
            x = layer(x, attention_mask=att_mask, position_embeddings=pos_emb)
            if i in block_set:
                dump(f"enc.block.{i}.out", x, f"encoder.block{i}.out")

        dump("enc.final", x, "encoder.final")

        enc_proj = model.model.decoder.proj(x)
        dump("enc_dec_proj.out", enc_proj, "encoder.enc_dec_proj")

    return x, output_mask, enc_proj


def decoder_masks(input_ids, encoder_hidden_states, encoder_lengths, attention_mask=None):
    import torch

    dtype = encoder_hidden_states.dtype
    batch_size, tgt_len = input_ids.shape

    query_positions = torch.arange(tgt_len, device=input_ids.device)[:, None]
    key_positions = torch.arange(tgt_len, device=input_ids.device)[None, :]
    causal_bool = key_positions > query_positions
    self_attention_mask = torch.zeros(
        (batch_size, 1, tgt_len, tgt_len),
        device=input_ids.device,
        dtype=dtype,
    )
    self_attention_mask.masked_fill_(causal_bool[None, None, :, :], float("-inf"))

    if attention_mask is not None:
        key_padding = (1.0 - attention_mask[:, None, None, :].to(dtype=dtype)) * -1e9
        self_attention_mask = self_attention_mask + key_padding

    src_len = encoder_hidden_states.shape[1]
    enc_positions = torch.arange(src_len, device=encoder_hidden_states.device)[None, :]
    valid = enc_positions < encoder_lengths.to(device=encoder_hidden_states.device)[:, None]
    cross_attention_mask = (1.0 - valid[:, None, None, :].to(dtype=dtype)) * -1e9
    return self_attention_mask, cross_attention_mask


def generate_transcript(
    *,
    model,
    processor,
    pcm: np.ndarray,
    sr: int,
    prompt: str,
    language: str,
    source: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    inputs = processor(
        audio=[pcm],
        language=language,
        punctuation=not args.no_punctuation,
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    prompt_len = int(inputs["decoder_input_ids"].shape[1])
    generated_list = generated_ids[0].detach().cpu().tolist()
    prompt_ids = inputs["decoder_input_ids"][0].detach().cpu().tolist()
    token_ids = generated_list[prompt_len:] if generated_list[:prompt_len] == prompt_ids else generated_list
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id in token_ids:
        token_ids = token_ids[:token_ids.index(eos_token_id)]
    text = processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    return {
        "schema": "transcribe-reference-transcript-v1",
        "text": text,
        "normalized_text": normalize_text(text),
        "token_ids": token_ids,
        "generated_ids": generated_list,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "prompt_length": prompt_len,
        "language": language,
        "punctuation": not args.no_punctuation,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "source": source,
    }


def cmd_mel(args: argparse.Namespace) -> int:
    use_transformers_ref(resolve_path(args.transformers_ref) if args.transformers_ref else None)
    configure_torch(args)
    import transformers
    from transformers import AutoProcessor

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Cohere ASR processor from {model_id} ({source}, "
        f"Transformers {transformers.__version__}, device={args.device})..."
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=local_only,
    )

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    inputs = frontend_inputs(processor, pcm, sr, args.device, args.language)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        language=args.language,
    )
    mel = to_np(inputs["input_features"])
    # Transpose to [num_mels, n_frames] to match C++ dump layout.
    if mel.ndim == 2:
        mel = mel.T
    print(
        f"mel: shape={mel.shape} min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.6f} std={mel.std():.6f}"
    )
    write_dump(out_dir, "enc.mel.in", mel, source=source, stage="frontend.mel.norm")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    use_transformers_ref(resolve_path(args.transformers_ref) if args.transformers_ref else None)
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

    inputs = frontend_inputs(processor, pcm, sr, args.device, args.language)
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
    import torch

    use_transformers_ref(resolve_path(args.transformers_ref) if args.transformers_ref else None)
    configure_torch(args)
    processor, model = load_reference(args)

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    model_id, _ = resolve_model(args.model)
    language = args.language or "en"
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    inputs = frontend_inputs(processor, pcm, sr, args.device, language)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        language=language,
        model_dtype=model_dtype_name(model),
    )

    encoder_hidden, encoder_attention_mask, _ = dump_encoder(
        model=model,
        inputs=inputs,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks),
    )

    tokenizer = processor.tokenizer
    input_ids = inputs["decoder_input_ids"].to(args.device)
    prompt = "".join(tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist()))

    print(f"  prompt: {prompt}")
    print(f"  prompt_ids: {input_ids[0].detach().cpu().tolist()}")

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )
        write_dump(out_dir, name, a, source=source, stage=stage)

    with torch.inference_mode():
        decoder = model.model.decoder

        # --- enc-dec projection (done inside decoder.forward, replicate here) ---
        enc_proj = decoder.proj(encoder_hidden)

        # --- token + positional embedding + layernorm ---
        tok_emb = decoder.embed_tokens(input_ids)
        dump("dec.token_emb", tok_emb, "decoder.embedding")

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = decoder.pos_emb(position_ids.squeeze(0))
        dump("dec.pos_emb", pos_emb, "decoder.position_embedding")

        x = decoder.embedding_layernorm(tok_emb + pos_emb)
        dump("dec.embed_norm", x, "decoder.embed_norm")

        # --- causal + cross-attention masks ---
        from transformers.masking_utils import create_causal_mask, create_bidirectional_mask

        causal_mask = create_causal_mask(
            config=decoder.config,
            inputs_embeds=x,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        cross_mask = create_bidirectional_mask(
            config=decoder.config,
            inputs_embeds=x,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=enc_proj,
        )

        # --- decoder layers ---
        for i, layer in enumerate(decoder.layers):
            x = layer(
                x,
                causal_mask,
                enc_proj,
                encoder_attention_mask=cross_mask,
                position_ids=position_ids,
            )
            dump(f"dec.block.{i}.out", x, f"decoder.block{i}.out")

        # --- final norm ---
        x = decoder.norm(x)
        dump("dec.out_before_head", x, "decoder.output_before_head")

        logits_raw = model.proj_out(x)
        dump("dec.logits_raw", logits_raw, "decoder.logits_raw")

        logits = torch.log_softmax(logits_raw, dim=-1)
        dump("dec.logits", logits, "decoder.logits")

    last_logits = logits[0, -1, :]
    pred_token_id = int(torch.argmax(last_logits).item())
    pred_token = tokenizer.decode([pred_token_id])
    print(f"  first predicted token: id={pred_token_id} text={pred_token!r}")

    if not args.skip_transcript:
        transcript = generate_transcript(
            model=model,
            processor=processor,
            pcm=pcm,
            sr=sr,
            prompt=prompt,
            language=language,
            source=source,
            args=args,
        )
        print(f"  transcript: {transcript['text']!r}")
        write_json_artifact(out_dir, "transcript.json", transcript)

    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="Path to Cohere ASR HF model directory")
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--transformers-ref",
        default=os.environ.get("TRANSCRIBE_TRANSFORMERS_REF"),
        help="Optional path to a local Hugging Face transformers checkout",
    )
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
        description="Dump Cohere ASR reference tensors from Transformers.",
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
        default=[0, 23, 47],
        help="Block indices to dump (default: 0 23 47)",
    )
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser("decode", help="Dump encoder + decoder prompt logits")
    add_common_args(dp)
    dp.add_argument("--no-punctuation", action="store_true", help="Use <|nopnc|> in the prompt")
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
        default=[0, 23, 47],
        help="Block indices to dump (default: 0 23 47)",
    )
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
