#!/usr/bin/env python3
"""
dump_reference_granite_transformers.py - generate IBM Granite Speech
reference tensors and transcript from the mainline Hugging Face
Transformers implementation (`GraniteSpeechForConditionalGeneration`).

Usage:

    uv run --project scripts/envs/granite \
      scripts/dump_reference_granite_transformers.py decode \
      --model ibm-granite/granite-4.0-1b-speech \
      --audio samples/jfk.wav \
      --out build/validate/granite/granite-4.0-1b-speech/jfk/decode/ref

Writes:
    <name>.f32    raw little-endian fp32, row-major (leading batch
                  squeezed to match the validation contract)
    <name>.json   per-tensor sidecar (with rms / p99_abs)
    transcript.json  behavioral artifact (prompt + token ids + text)

Dump points (match tests/tolerances/granite.json):

    enc.mel.in              input log-mel feature tensor as fed to the encoder
                            (the audio_processor's output for input_features)
    enc.input_linear.out    after encoder.input_linear (input_dim -> hidden_dim)
    enc.block.<i>.out       selected Conformer block outputs (in hidden_dim space)
    enc.out                 final encoder output (last_hidden_state from encoder)
    proj.qformer.out        Q-Former cross-attention output (last_hidden_state)
    proj.out                projector final output (audio embeddings in text-hidden space)
    dec.token_emb           LM token embeddings pre audio-token scatter
    dec.audio_injected      LM input embeddings after audio scatter
    dec.block.<i>.out       selected LM decoder block outputs
    dec.out_before_head     LM output after final RMSNorm, pre lm_head
    dec.logits_raw          pre-softmax logits at the prefill position

The full decode pass also writes transcript.json with the greedy-decoded
text and token IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Use the shared write helpers so the on-disk contract (and rms/p99_abs
# sidecar fields) stay identical across families.
sys.path.insert(0, str(Path(__file__).parent))
from lib.ref_dump import write_tensor, write_transcript


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


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def resolve_model(raw: str) -> tuple[str, bool]:
    """Resolve a model argument to (model_id_or_path, local_files_only)."""
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


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


def load_reference(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
    from transformers.models.granite_speech_plus import GraniteSpeechPlusForConditionalGeneration

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Granite Speech model from {model_id} ({source}, "
        f"transformers {transformers.__version__}, device={args.device})..."
    )
    cfg = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=local_only,
    )
    cls = {
        "granite_speech": GraniteSpeechForConditionalGeneration,
        "granite_speech_plus": GraniteSpeechPlusForConditionalGeneration,
    }.get(cfg.model_type)
    if cls is None:
        raise SystemExit(
            f"dumper does not handle model_type={cfg.model_type!r}; expected granite_speech or granite_speech_plus"
        )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=local_only,
    )
    dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[args.dtype]
    model = cls.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="eager",  # for deterministic hooks
    ).eval()
    model.to(args.device)
    return processor, model


# ---------------------------------------------------------------------------
# Hooks
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
        # transformers returns ModelOutput objects for some submodules
        # (e.g. Blip2QFormer, GraniteSpeechCTCEncoder). Unwrap to the
        # primary tensor.
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        self.value = output


def register_encoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the audio encoder at the standard dump points.

    Granite Speech encoder forward (modeling_granite_speech.py:309):
        hidden = input_linear(mel)           => enc.input_linear.out
        for layer in layers:
            hidden = layer(hidden, ...)      => enc.block.<i>.out
            (mid-layer CTC bypass at idx == num_layers // 2)
        return BaseModelOutputWithPooling(last_hidden_state=hidden)   => enc.out

    The encoder itself is the topmost wrapped module; hooking it captures
    the post-loop output (after the optional mid-layer CTC bypass adds in).
    """
    hooks: dict[str, CaptureHook] = {}
    enc = model.encoder

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    hook_forward("enc.input_linear.out", enc.input_linear)
    for i in blocks:
        if i < 0 or i >= len(enc.layers):
            continue
        hook_forward(f"enc.block.{i}.out", enc.layers[i])
    hook_forward("enc.out", enc)

    return hooks


def register_projector_hooks(model) -> dict[str, CaptureHook]:
    """Hook the projector.

    GraniteSpeechEncoderProjector.forward (modeling_granite_speech.py:87):
        hidden = window_pad_and_reshape(encoder_out)
        qformer_out = self.qformer(query_embeds=self.query,
                                   encoder_hidden_states=hidden)
                                             => proj.qformer.out
        proj = self.linear(qformer_out.reshape)
                                             => proj.out (also = projector forward output)
    """
    hooks: dict[str, CaptureHook] = {}
    proj = model.projector

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    hook_forward("proj.qformer.out", proj.qformer)
    hook_forward("proj.out", proj)

    return hooks


def register_decoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the inner Granite LM.

    GraniteForCausalLM has:
      model       (the inner GraniteModel)
        embed_tokens   => dec.token_emb (pre-injection embedding lookup)
        layers[i]      => dec.block.<i>.out
        norm           => dec.out_before_head
      lm_head      (or tied to embed_tokens; covered by dec.logits_raw)

    Audio injection happens INSIDE
    GraniteSpeechForConditionalGeneration.get_merged_audio_embeddings, which
    is called from .forward BEFORE delegating to self.language_model.
    We capture the post-injection inputs_embeds with a pre-hook on the LLM.
    """
    hooks: dict[str, CaptureHook] = {}
    lm = model.language_model
    inner = lm.model  # GraniteModel
    norm = inner.norm

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    hook_forward("dec.token_emb", inner.embed_tokens)
    for i in blocks:
        if i < 0 or i >= len(inner.layers):
            continue
        hook_forward(f"dec.block.{i}.out", inner.layers[i])
    hook_forward("dec.out_before_head", norm)

    # Post-injection inputs_embeds: caught at the call to the inner GraniteModel.
    inj = CaptureHook()

    def pre(module, args, kwargs):
        if inj.value is not None:
            return
        emb = kwargs.get("inputs_embeds")
        if emb is None:
            return
        inj.value = emb

    inner.register_forward_pre_hook(pre, with_kwargs=True)
    hooks["dec.audio_injected"] = inj
    return hooks


def make_source(
    *,
    args: argparse.Namespace,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    model_dtype: str,
    prompt: str,
) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "kind": "granite-speech-transformers-native",
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
        "prompt": prompt,
    }


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    processor, model = load_reference(args)
    model_id, _ = resolve_model(args.model)
    model_dtype = model_dtype_name(model)

    audio_path = Path(args.audio).expanduser().resolve()
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        raise SystemExit(
            f"Granite Speech expects 16kHz audio; got {sr} Hz in {audio_path}"
        )

    # Build the chat-templated prompt with an audio placeholder.
    user_message = f"<|audio|>{args.instruction}"
    chat = [{"role": "user", "content": user_message}]
    if processor.tokenizer.chat_template is not None or processor.chat_template is not None:
        prompt = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
    else:
        prompt = f"USER: {user_message}\n ASSISTANT:"
    print(f"prompt: {prompt!r}")

    inputs = processor(
        text=prompt,
        audio=[torch.from_numpy(pcm)],
        sampling_rate=sr,
        return_tensors="pt",
        device=args.device,
    )
    inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype,
        prompt=prompt,
    )

    # Write the raw audio frontend input that the encoder will receive.
    # `inputs["input_features"]` is the audio_processor's output and is the
    # canonical "mel.in" reference for downstream C++ frontend validation.
    mel = to_np(inputs["input_features"])
    write_tensor(
        "enc.mel.in",
        mel,
        stage="encoder",
        source={**source, "hook": "audio_processor.output:input_features"},
        out_dir=out_dir,
    )
    print(
        f"  enc.mel.in: shape={mel.shape} "
        f"min={mel.min():.4e} max={mel.max():.4e} mean={mel.mean():.6e}"
    )

    # Determine the encoder / decoder blocks to capture.
    n_enc_blocks = len(model.encoder.layers)
    n_dec_blocks = len(model.language_model.model.layers)
    enc_blocks = sorted({0, n_enc_blocks // 2, n_enc_blocks - 1, *args.enc_blocks})
    dec_blocks = sorted({0, n_dec_blocks // 2, n_dec_blocks - 1, *args.dec_blocks})

    enc_hooks = register_encoder_hooks(model, enc_blocks)
    proj_hooks = register_projector_hooks(model)
    dec_hooks = register_decoder_hooks(model, dec_blocks)

    print(
        f"hooks: encoder={list(enc_hooks)} projector={list(proj_hooks)} "
        f"decoder={list(dec_hooks)}"
    )

    # Greedy decode to capture the prefill forward AND produce a transcript.
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = gen_out.sequences  # [1, prompt_len + new_tokens]
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = sequences[0, prompt_len:].tolist()
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"transcript: {text!r}")

    # First-token logits across the LM vocab; lets Stage 4 compare the
    # next-token distribution at the prefill boundary.
    if gen_out.scores:
        logits0 = to_np(gen_out.scores[0])  # [vocab]
        write_tensor(
            "dec.logits_raw",
            logits0,
            stage="decoder",
            source={**source, "hook": "generate.scores[0]"},
            out_dir=out_dir,
        )
        print(
            f"  dec.logits_raw: shape={logits0.shape} "
            f"min={logits0.min():.4e} max={logits0.max():.4e} mean={logits0.mean():.6e}"
        )

    # Flush all captured tensors to disk via the shared writer.
    def flush(name: str, t, stage: str) -> None:
        if t is None:
            print(f"  WARN: {name}: no value captured")
            return
        a = to_np(t)
        write_tensor(
            name,
            a,
            stage=stage,
            source={**source, "hook": name},
            out_dir=out_dir,
        )
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )

    for name, hook in enc_hooks.items():
        flush(name, hook.value, stage="encoder")
    for name, hook in proj_hooks.items():
        flush(name, hook.value, stage="projector")
    for name, hook in dec_hooks.items():
        flush(name, hook.value, stage="decoder")

    write_transcript(
        out_dir,
        text=text,
        source={**source, "hook": "generate.sequences"},
        tokens=new_tokens,
    )
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    decode = sub.add_parser("decode", help="full encoder+projector+LM dump + transcript")
    decode.add_argument("--model", required=True, help="HF repo id or local path")
    decode.add_argument("--audio", required=True, help="path to audio file (16kHz)")
    decode.add_argument("--out", required=True, help="output directory for dumps")
    decode.add_argument(
        "--instruction",
        default="can you transcribe the speech into a written format?",
        help="user instruction following the <|audio|> placeholder",
    )
    decode.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    decode.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "f16", "f32"],
        help="model dtype",
    )
    decode.add_argument("--torch-threads", type=int, default=4)
    decode.add_argument(
        "--enc-blocks",
        type=int,
        nargs="*",
        default=[],
        help="extra encoder block indices to dump (first/mid/last always included)",
    )
    decode.add_argument(
        "--dec-blocks",
        type=int,
        nargs="*",
        default=[],
        help="extra LM block indices to dump (first/mid/last always included)",
    )
    decode.add_argument("--max-new-tokens", type=int, default=64)
    decode.set_defaults(func=cmd_decode)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
