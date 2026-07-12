#!/usr/bin/env python3
"""
dump_reference_moss_author.py - generate MOSS-Transcribe-Diarize reference
tensors from the OpenMOSS custom trust_remote_code implementation (loaded
via transformers AutoModelForCausalLM).

Architecture (see modeling_moss_transcribe_diarize.py):
    log-mel input_features -> WhisperEncoder (24 layers)
      -> 4x time merge (B, T, 1024) -> (B, T/4, 4096)
      -> VQAdaptor (Linear 4096->1024, SiLU, Linear 1024->1024, LayerNorm)
      -> masked_scatter into Qwen3-0.6B text embeddings at <|audio_pad|> (151671)
      -> Qwen3 decoder (28 layers) -> lm_head -> logits

Usage:

    uv run --project scripts/envs/moss \
      scripts/dump_reference_moss_author.py decode \
      --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
      --audio samples/jfk.wav \
      --out build/validate/moss/moss-transcribe-diarize/jfk/decode/ref

Writes:
    <name>.f32       raw little-endian fp32, row-major (leading batch squeezed)
    <name>.json      per-tensor sidecar (rms + p99_abs for tolerances)
    transcript.json  behavioral artifact (de-diarized text + raw + token ids)

Dump points (match tests/tolerances/moss.json):

    enc.mel.in           input log-mel after WhisperFeatureExtractor
    enc.pos_add.out      whisper hidden after conv stack + positional add
    enc.block.<i>.out    selected whisper encoder layer outputs
    enc.ln_post.out      after whisper encoder final layer_norm
    enc.merge.out        after 4x temporal merge (adaptor input, dim 4096)
    enc.adaptor.out      after VQAdaptor (injected audio features, dim 1024)
    dec.token_emb        LM input embeddings pre audio injection
    dec.audio_injected   LM embeddings after masked_scatter of audio features
    dec.block.<i>.out    selected Qwen3 decoder layer outputs
    dec.out_before_head  after final Qwen3 RMSNorm
    dec.logits_raw       first-step pre-softmax next-token logits
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


# The canonical timestamped+diarize instruction shipped in the OpenMOSS
# GitHub inference harness (moss_transcribe_diarize/inference_utils.py
# DEFAULT_PROMPT). Used for English audio as well; the model has no
# language/task token.
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


# ---------------------------------------------------------------------------
# Shared reference-dump helpers
# ---------------------------------------------------------------------------


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


def to_np(t) -> np.ndarray:
    import torch

    if isinstance(t, torch.Tensor):
        a = t.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        a = np.asarray(t, dtype=np.float32)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a, dtype=np.float32)


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
        raise ValueError(f"only float32 tensors supported, got {data.dtype}")
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    (out_dir / f"{name}.f32").write_bytes(data.tobytes())
    if data.size:
        flat64 = data.astype(np.float64, copy=False).reshape(-1)
        rms = float(np.sqrt(np.mean(flat64 * flat64)))
        p99_abs = float(np.quantile(np.abs(flat64), 0.99))
        a_min, a_max, a_mean = float(data.min()), float(data.max()), float(flat64.mean())
    else:
        rms = p99_abs = a_min = a_max = a_mean = 0.0
    meta = {
        "name": name,
        "stage": stage,
        "shape": list(data.shape),
        "dtype": "f32",
        "layout": "row-major",
        "min": a_min,
        "max": a_max,
        "mean": a_mean,
        "rms": rms,
        "p99_abs": p99_abs,
        "source": source,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {out_dir / f'{name}.f32'} "
          f"({data.size * 4} bytes, shape={list(data.shape)})")


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


def dediarize(raw: str) -> str:
    """Strip [start], [Sxx], [end] bracket spans from the canonical
    [start][Sxx]text[end] format, leaving plain transcription text. All
    metadata (timestamps + speaker labels) lives inside brackets; text
    lives outside them."""
    stripped = re.sub(r"\[[^\]]*\]", " ", raw)
    return " ".join(stripped.split())


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


# ---------------------------------------------------------------------------
# Model / processor loading
# ---------------------------------------------------------------------------


def resolve_model(raw: str) -> tuple[str, bool]:
    p = Path(raw).expanduser().resolve()
    if p.is_dir():
        return str(p), True
    return raw, False


def load_reference(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    revision = getattr(args, "revision", None) if not local_only else None
    pin_note = f"@{revision}" if revision else ""
    print(
        f"Loading MOSS-Transcribe-Diarize from {model_id}{pin_note} ({source}, "
        f"transformers {transformers.__version__}, device={args.device})"
    )

    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision,
        trust_remote_code=True, local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="eager",  # hookable + deterministic
    ).eval()
    model.to(args.device)
    return processor, model


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    return {
        torch.bfloat16: "bf16",
        torch.float16: "f16",
        torch.float32: "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


def make_source(
    *,
    args: argparse.Namespace,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    model_dtype: str,
) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "kind": "moss-author-repo",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model": model_id,
        "model_dtype": model_dtype,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
        "prompt": "DEFAULT_PROMPT (timestamp+diarize, zh)",
    }


# ---------------------------------------------------------------------------
# Forward hooks
# ---------------------------------------------------------------------------


class CaptureHook:
    """Record the first output tensor a module emits during forward."""

    def __init__(self) -> None:
        self.value = None

    def __call__(self, module, inputs, output) -> None:
        if self.value is not None:
            return
        if isinstance(output, tuple):
            output = output[0]
        self.value = output


def register_encoder_hooks(enc, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the WhisperEncoder at standard dump points.

    WhisperEncoder.forward: conv1 -> gelu -> conv2 -> gelu -> permute
      -> + embed_positions -> layers[i]... -> layer_norm
    """
    hooks: dict[str, CaptureHook] = {}

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    for i in blocks:
        if 0 <= i < len(enc.layers):
            hook_forward(f"enc.block.{i}.out", enc.layers[i])
    hook_forward("enc.ln_post.out", enc.layer_norm)

    # "after conv stack + positional add" = input to encoder layer[0].
    pos_add = CaptureHook()

    def pre0(module, args):
        if pos_add.value is None and len(args) >= 1:
            pos_add.value = args[0]

    enc.layers[0].register_forward_pre_hook(pre0)
    hooks["enc.pos_add.out"] = pos_add
    return hooks


def register_adaptor_hooks(vq_adaptor) -> dict[str, CaptureHook]:
    """Capture the 4x-merged adaptor input (dim 4096) and adaptor output
    (dim 1024, the injected audio features)."""
    hooks: dict[str, CaptureHook] = {}

    out_hook = CaptureHook()
    vq_adaptor.register_forward_hook(out_hook)
    hooks["enc.adaptor.out"] = out_hook

    merge_hook = CaptureHook()

    def pre(module, args):
        if merge_hook.value is None and len(args) >= 1:
            merge_hook.value = args[0]

    vq_adaptor.register_forward_pre_hook(pre)
    hooks["enc.merge.out"] = merge_hook
    return hooks


def register_decoder_hooks(text_model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the Qwen3 text model (language_model) at standard dump points."""
    hooks: dict[str, CaptureHook] = {}

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    hook_forward("dec.token_emb", text_model.embed_tokens)
    for i in blocks:
        if 0 <= i < len(text_model.layers):
            hook_forward(f"dec.block.{i}.out", text_model.layers[i])
    hook_forward("dec.out_before_head", text_model.norm)
    return hooks


def register_injection_hook(backbone) -> dict[str, CaptureHook]:
    """Capture the LM input embeddings after masked_scatter of audio
    features (inputs_embeds kwarg passed to language_model.forward)."""
    hooks: dict[str, CaptureHook] = {}
    inj = CaptureHook()

    def pre(module, args, kwargs):
        if inj.value is not None:
            return
        emb = kwargs.get("inputs_embeds")
        if emb is not None:
            inj.value = emb

    backbone.language_model.register_forward_pre_hook(pre, with_kwargs=True)
    hooks["dec.audio_injected"] = inj
    return hooks


# ---------------------------------------------------------------------------
# decode subcommand
# ---------------------------------------------------------------------------


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    processor, model = load_reference(args)
    backbone = model.model  # MossTranscribeDiarizeModel
    enc = backbone.whisper_encoder
    text_model = backbone.language_model

    pcm, sr = load_audio(Path(args.audio))
    if sr != 16000:
        print(f"error: expected 16 kHz audio, got {sr}", file=sys.stderr)
        return 2
    print(f"Loaded {Path(args.audio).name}: {pcm.size} samples @ {sr} Hz")

    enc_blocks = sorted({0, len(enc.layers) - 1, *args.enc_blocks})
    dec_blocks = sorted({0, text_model.config.num_hidden_layers - 1, *args.dec_blocks})

    enc_hooks = register_encoder_hooks(enc, enc_blocks)
    adaptor_hooks = register_adaptor_hooks(backbone.vq_adaptor)
    dec_hooks = register_decoder_hooks(text_model, dec_blocks)
    inj_hooks = register_injection_hook(backbone)

    # Mirror the reference harness (inference_utils.prepare_inputs):
    # apply_chat_template(add_generation_prompt=True) then processor(text, audio).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(args.audio)},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt_text, audio=[pcm], return_tensors="pt")
    inputs = inputs.to(args.device)

    input_features = inputs.get("input_features")
    if input_features is None:
        print("error: processor did not return input_features", file=sys.stderr)
        return 3

    source = make_source(
        args=args,
        model_id=str(Path(args.model).expanduser().resolve())
        if Path(args.model).is_dir() else args.model,
        audio_path=Path(args.audio),
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype_name(model),
    )

    out_dir = Path(args.out).expanduser().resolve()

    # enc.mel.in: log-mel [n_chunks, n_mels, T]; squeeze single chunk to [n_mels, T].
    write_dump(out_dir, "enc.mel.in", to_np(input_features),
               source=source, stage="frontend.mel.norm")

    with torch.inference_mode():
        gen = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    prompt_len = int(inputs["attention_mask"][0].sum().item())
    seq = gen.sequences[0].detach().cpu().tolist()
    gen_ids = seq[prompt_len:]
    eos_id = processor.tokenizer.eos_token_id
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    raw_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    plain_text = dediarize(raw_text)

    def flush(name: str, value, stage: str) -> None:
        if value is None:
            print(f"  WARN: no capture for {name}")
            return
        write_dump(out_dir, name, to_np(value), source=source, stage=stage)

    for name, hook in enc_hooks.items():
        flush(name, hook.value, stage=name)
    for name, hook in adaptor_hooks.items():
        flush(name, hook.value, stage=name)
    for name, hook in inj_hooks.items():
        flush(name, hook.value, stage=name)
    for name, hook in dec_hooks.items():
        flush(name, hook.value, stage=name)
    if gen.scores:
        flush("dec.logits_raw", gen.scores[0], stage="dec.logits_raw")

    transcript = {
        "schema": "transcribe-reference-transcript-v1",
        "text": plain_text,
        "raw_text": raw_text,
        "normalized_text": normalize_text(plain_text),
        "token_ids": gen_ids,
        "prompt_len": prompt_len,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "source": source,
    }
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Raw transcript:   {raw_text!r}")
    print(f"Plain transcript: {plain_text!r}")
    print(f"Wrote transcript.json ({len(gen_ids)} tokens)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", required=True,
                    help="HF repo id (OpenMOSS-Team/MOSS-Transcribe-Diarize) or local dir")
    sp.add_argument("--revision", default=None,
                    help="HF revision to pin (ignored for local paths).")
    sp.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    sp.add_argument("--out", required=True, help="Output directory for dumps")
    sp.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    sp.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"],
                    help="Reference compute dtype (default: bf16, matches config)")
    sp.add_argument("--torch-threads", type=int, default=1,
                    help="torch.set_num_threads (0 = unchanged)")
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                    help="Extra encoder block indices to capture")
    sp.add_argument("--dec-blocks", type=int, nargs="*", default=[],
                    help="Extra decoder block indices to capture")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="MOSS-Transcribe-Diarize reference dumper.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("decode", help="Run greedy decode + capture all dump points")
    _add_common(sp)
    sp.set_defaults(func=cmd_decode)

    # Single-pass decode captures enc.* + adaptor + dec.* in one forward, so
    # `encoder` is an alias letting validate.py drive both stages uniformly.
    ep = sub.add_parser("encoder", help="Alias for decode (full dump pass)")
    _add_common(ep)
    ep.set_defaults(func=cmd_decode)

    args = p.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
