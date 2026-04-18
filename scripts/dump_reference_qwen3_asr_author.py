#!/usr/bin/env python3
"""
dump_reference_qwen3_asr_author.py - generate Qwen3-ASR reference tensors
from the Alibaba first-party qwen_asr package (the canonical
implementation registered via transformers AutoModel).

Usage:

    uv run --project scripts/envs/qwen3_asr \
      scripts/dump_reference_qwen3_asr_author.py decode \
      --model ../models/Qwen3-ASR-0.6B \
      --audio samples/jfk.wav \
      --out build/validate/qwen3_asr/qwen3-asr-0.6b/jfk/ref

Writes:
    <name>.f32    raw little-endian fp32, row-major (leading batch
                  squeezed to match the validation contract)
    <name>.json   per-tensor sidecar
    transcript.json  behavioral artifact (prompt + token ids + text)

Dump points (match tests/tolerances/qwen3_asr.json):

    enc.mel.in            input log-mel after frontend normalization
    enc.subsample.out     after 3x Conv2d + flatten + conv_out linear
    enc.pos_add.out       after sinusoidal PE add
    enc.block.<i>.out     selected encoder block outputs
    enc.ln_post.out       after post-LayerNorm
    enc.proj.out          after proj1 + GELU + proj2 (output_dim)
    dec.token_emb         LM input embeddings pre-injection
    dec.audio_injected    embeddings after audio feature scatter
    dec.block.<i>.out     selected LM block outputs
    dec.out_before_head   after final RMSNorm
    dec.logits_raw        pre-softmax next-token logits
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
# Shared reference-dump helpers (trimmed copy of the Cohere dumper's style)
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
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {out_dir / f'{name}.f32'} "
          f"({data.size * 4} bytes, shape={list(data.shape)})")


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


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
    """Load Qwen3-ASR via the qwen_asr author package.

    qwen_asr.inference.qwen3_asr has side-effect registrations that wire
    Qwen3ASRConfig + Qwen3ASRForConditionalGeneration + Qwen3ASRProcessor
    into the transformers Auto* registries. After the import, we can use
    plain AutoConfig/AutoModel/AutoProcessor.
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoProcessor

    # Side-effect imports: register the author repo classes with Auto*.
    # Keep this inside the function so a static import failure (missing
    # qwen_asr in the env) surfaces with a clear error, not at module load.
    import qwen_asr.inference.qwen3_asr  # noqa: F401

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Qwen3-ASR from {model_id} ({source}, "
        f"transformers {transformers.__version__}, device={args.device})"
    )

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=False, local_files_only=local_only,
    )
    # fix_mistral_regex=True matches the author-repo call in Qwen3ASRModel.from_pretrained
    # and suppresses the upstream tokenizer regex warning.
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=False, local_files_only=local_only,
        fix_mistral_regex=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=False,
        local_files_only=local_only,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # for hookability + deterministic refs
    ).eval()
    model.to(args.device)
    return processor, model


def build_text_prompt(processor, context: str, force_language: str | None) -> str:
    """Mirror Qwen3ASRModel._build_text_prompt: render the chat template
    for a single (context, audio) request. If force_language is set, append
    "language X<asr_text>" to the assistant prompt to force text-only output.
    """
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}<asr_text>"
    return base


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    return {
        torch.bfloat16: "bf16",
        torch.float16:  "f16",
        torch.float32:  "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


def make_source(
    *,
    args: argparse.Namespace,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    language: str | None,
    model_dtype: str,
) -> dict[str, Any]:
    import torch
    import transformers
    import qwen_asr

    src: dict[str, Any] = {
        "kind": "qwen3_asr-author-repo",
        "qwen_asr_version": getattr(qwen_asr, "__version__", "unknown"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model": model_id,
        "model_dtype": model_dtype,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
    }
    if language is not None:
        src["language"] = language
    return src


# ---------------------------------------------------------------------------
# Forward hooks
# ---------------------------------------------------------------------------


class CaptureHook:
    """Record the first output tensor a module emits during forward."""

    def __init__(self) -> None:
        self.value = None

    def __call__(self, module, inputs, output) -> None:  # noqa: D401
        if self.value is not None:
            return
        if isinstance(output, tuple):
            output = output[0]
        self.value = output


def register_encoder_hooks(enc, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the audio encoder at the standard dump points.

    The reference encoder's forward (see modeling_qwen3_asr.py) is:
      conv2d1 -> GELU -> conv2d2 -> GELU -> conv2d3 -> GELU
        -> permute+view(B, T, C*F) -> conv_out (Linear)  => subsample.out
        -> + positional_embedding                         => pos_add.out
        -> [layer_i for i in 0..N-1]                      => block.i.out
        -> ln_post                                        => ln_post.out
        -> proj1 -> GELU -> proj2                         => proj.out

    We hook the canonical module at each boundary:
      conv_out       -> enc.subsample.out
      positional_embedding is a buffer-add in the parent forward and has
      no dedicated module; the next meaningful boundary is the first
      encoder layer's input, which we capture by hooking layer[0] with a
      pre-hook.
      layers[i]      -> enc.block.i.out
      ln_post        -> enc.ln_post.out
      proj2          -> enc.proj.out
    """
    hooks: dict[str, CaptureHook] = {}

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    # Post-subsample (after conv_out Linear produces [B, T, d_model]).
    hook_forward("enc.subsample.out", enc.conv_out)

    # Block outputs.
    for i in blocks:
        if i < 0 or i >= len(enc.layers):
            continue
        hook_forward(f"enc.block.{i}.out", enc.layers[i])

    # Post-LayerNorm and final projection.
    hook_forward("enc.ln_post.out", enc.ln_post)
    hook_forward("enc.proj.out",    enc.proj2)

    # Capture "after positional add" via a pre-hook on layer[0]. The
    # pre-hook sees (hidden_states, cu_seqlens) as positional args.
    pos_add = CaptureHook()

    def pre0(module, args):
        if pos_add.value is None and len(args) >= 1:
            pos_add.value = args[0]

    enc.layers[0].register_forward_pre_hook(pre0)
    hooks["enc.pos_add.out"] = pos_add

    return hooks


def register_decoder_hooks(text_model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the text LM (thinker.model) at standard dump points.

    text_model is Qwen3ASRThinkerTextModel with:
      embed_tokens -> (audio injected in parent forward) -> layers[i]... -> norm

    We hook:
      embed_tokens   -> dec.token_emb          (pre-injection)
      layers[i]      -> dec.block.i.out
      norm           -> dec.out_before_head
    """
    hooks: dict[str, CaptureHook] = {}

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    hook_forward("dec.token_emb", text_model.embed_tokens)
    for i in blocks:
        if i < 0 or i >= len(text_model.layers):
            continue
        hook_forward(f"dec.block.{i}.out", text_model.layers[i])
    hook_forward("dec.out_before_head", text_model.norm)
    return hooks


def register_injection_hook(thinker) -> dict[str, CaptureHook]:
    """Capture the LM input embeddings after audio-feature scatter.

    In qwen_asr the injection happens inside Qwen3ASRThinkerForConditional
    Generation.forward before calling self.model(...) on the fused
    inputs_embeds. The cleanest capture point is a pre-hook on
    thinker.model (Qwen3ASRThinkerTextModel.forward) that reads the
    `inputs_embeds` kwarg.
    """
    hooks: dict[str, CaptureHook] = {}
    inj = CaptureHook()

    def pre(module, args, kwargs):
        if inj.value is not None:
            return
        emb = kwargs.get("inputs_embeds")
        if emb is None and args:
            # positional calling convention: first arg is input_ids
            # (not embeds), so we ignore it.
            return
        inj.value = emb

    thinker.model.register_forward_pre_hook(pre, with_kwargs=True)
    hooks["dec.audio_injected"] = inj
    return hooks


# ---------------------------------------------------------------------------
# decode subcommand
# ---------------------------------------------------------------------------


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    processor, model = load_reference(args)
    thinker = model.thinker if hasattr(model, "thinker") else model
    enc = thinker.audio_tower
    text_model = thinker.model

    pcm, sr = load_audio(Path(args.audio))
    if sr != 16000:
        print(f"error: expected 16 kHz audio, got {sr}", file=sys.stderr)
        return 2
    print(f"Loaded {Path(args.audio).name}: {pcm.size} samples @ {sr} Hz")

    enc_blocks = sorted({0, enc.config.encoder_layers - 1, *args.enc_blocks})
    dec_blocks = sorted({0, text_model.config.num_hidden_layers - 1, *args.dec_blocks})

    enc_hooks = register_encoder_hooks(enc, enc_blocks)
    dec_hooks = register_decoder_hooks(text_model, dec_blocks)
    inj_hooks = register_injection_hook(thinker)

    # Run the processor. Qwen3ASRProcessor requires both `text` and `audio`;
    # the text is the rendered chat template prompt (see Qwen3ASRModel
    # ._build_text_prompt in the author repo).
    prompt_text = build_text_prompt(processor, context=args.context,
                                    force_language=args.language)
    inputs = processor(
        text=[prompt_text],
        audio=[pcm],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(args.device).to(model.dtype)

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
        language=args.language,
        model_dtype=model_dtype_name(model),
    )

    out_dir = Path(args.out).expanduser().resolve()

    # enc.mel.in is the processor output (log-mel, possibly [B, n_mels, T]).
    # We squeeze the batch dim via to_np and write in the C++ contract
    # shape: [num_mels, n_frames].
    write_dump(
        out_dir, "enc.mel.in", to_np(input_features),
        source=source, stage="frontend.mel.norm",
    )

    # Greedy decode (single beam) to get a transcript + logits hook.
    # Qwen3ASRForConditionalGeneration.generate() unconditionally passes
    # return_dict_in_generate=True; don't pass it again here.
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            output_scores=True,
        )

    token_ids = gen.sequences[0].detach().cpu().tolist()
    # Drop the prompt prefix when it is echoed.
    prompt_ids = inputs.get("input_ids")
    if prompt_ids is not None:
        prompt_len = int(prompt_ids.shape[1])
        if len(token_ids) > prompt_len and token_ids[:prompt_len] == prompt_ids[0].detach().cpu().tolist():
            token_ids = token_ids[prompt_len:]
    eos_id = processor.tokenizer.eos_token_id
    if eos_id in token_ids:
        token_ids = token_ids[:token_ids.index(eos_id)]
    text = processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    # Dump captured intermediates.
    def flush(name: str, value, stage: str) -> None:
        if value is None:
            print(f"  WARN: no capture for {name}")
            return
        write_dump(out_dir, name, to_np(value), source=source, stage=stage)

    for name, hook in enc_hooks.items():
        flush(name, hook.value, stage=name)
    for name, hook in inj_hooks.items():
        flush(name, hook.value, stage=name)
    for name, hook in dec_hooks.items():
        flush(name, hook.value, stage=name)

    # First-step logits (pre-softmax next-token distribution).
    if gen.scores:
        flush("dec.logits_raw", gen.scores[0], stage="dec.logits_raw")

    transcript = {
        "schema": "transcribe-reference-transcript-v1",
        "text": text,
        "normalized_text": normalize_text(text),
        "token_ids": token_ids,
        "prompt_ids": prompt_ids[0].detach().cpu().tolist() if prompt_ids is not None else [],
        "language": args.language,
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
    print(f"Transcript: {text!r}")
    print(f"Wrote transcript.json ({len(token_ids)} tokens)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", required=True,
                    help="HF repo id (Qwen/Qwen3-ASR-0.6B) or local directory")
    sp.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    sp.add_argument("--out", required=True, help="Output directory for dumps")
    sp.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    sp.add_argument("--torch-threads", type=int, default=1,
                    help="torch.set_num_threads (0 = unchanged)")
    sp.add_argument("--language", default=None,
                    help="Optional language hint (e.g. English); when set, "
                         "the prompt is suffixed with 'language X<asr_text>' "
                         "to force text-only output")
    sp.add_argument("--context", default="",
                    help="System-prompt context string (mirrors the "
                         "`context` arg to Qwen3ASRModel.transcribe)")
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                    help="Extra encoder block indices to capture")
    sp.add_argument("--dec-blocks", type=int, nargs="*", default=[],
                    help="Extra decoder block indices to capture")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Qwen3-ASR reference dumper.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("decode", help="Run greedy decode + capture all dump points")
    _add_common(sp)
    sp.set_defaults(func=cmd_decode)

    # Qwen3-ASR decode captures all enc.* + dec.* dump points in a single
    # pass (same shape as the Cohere dumper), so `encoder` is an alias
    # that lets scripts/validate.py drive both stages uniformly.
    ep = sub.add_parser("encoder", help="Alias for decode (full dump pass)")
    _add_common(ep)
    ep.set_defaults(func=cmd_decode)

    args = p.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
