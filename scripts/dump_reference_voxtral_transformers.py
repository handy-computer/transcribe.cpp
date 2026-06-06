#!/usr/bin/env python3
"""
dump_reference_voxtral_transformers.py - generate Voxtral (2507) reference
tensors and transcript from the mainline Hugging Face Transformers
implementation (`VoxtralForConditionalGeneration`).

Usage:

    uv run --project scripts/envs/voxtral \
      scripts/dump_reference_voxtral_transformers.py decode \
      --model mistralai/Voxtral-Mini-3B-2507 \
      --audio samples/jfk.wav \
      --out build/validate/voxtral/voxtral-mini-3b-2507/jfk/decode/ref

Writes:
    <name>.f32    raw little-endian fp32, row-major (leading batch
                  squeezed to match the validation contract)
    <name>.json   per-tensor sidecar (with rms / p99_abs)
    transcript.json  behavioral artifact (prompt tokens + decoded text)

Architecture (audio-llm / token injection, no cross-attention):
    audio_tower (Whisper-large-v3 encoder)  -> last_hidden_state (T=1500, 1280)
    get_audio_features: reshape (T,1280)->(T/4,5120)  [4x frame grouping]
    multi_modal_projector: Linear(5120->H) -> GELU -> Linear(H->H)
    forward: masked_scatter audio embeds into LM input embeds at
             input_ids == audio_token_id (24)
    language_model (Llama / Ministral) -> logits

Dump points (match tests/tolerances/voxtral.json):

    enc.mel.in            input log-mel features fed to the encoder
    enc.block.<i>.out     selected encoder block outputs (1280-dim)
    enc.out               final encoder output (last_hidden_state)
    proj.out              projector output = audio embeds in text-hidden space
    dec.token_emb         LM token embeddings pre audio-token scatter
    dec.audio_injected    LM input embeddings after audio scatter
    dec.block.<i>.out     selected LM decoder block outputs
    dec.out_before_head   LM output after final RMSNorm, pre lm_head
    dec.logits_raw        pre-softmax logits at the prefill position

The decode pass also writes transcript.json with the greedy-decoded text
and the generated token IDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Shared write helpers keep the on-disk contract (and rms/p99_abs sidecar
# fields) identical across families.
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


def resolve_model(raw: str) -> tuple[str, bool]:
    """Resolve a model argument to (model_id_or_path, local_files_only)."""
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    return {
        torch.bfloat16: "bf16",
        torch.float16: "f16",
        torch.float32: "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


def load_reference(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Voxtral model from {model_id} ({source}, "
        f"transformers {transformers.__version__}, device={args.device})..."
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[args.dtype]
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="eager",  # deterministic hooks
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
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        self.value = output


def _hook_forward(hooks: dict[str, CaptureHook], name: str, module) -> None:
    h = CaptureHook()
    module.register_forward_hook(h)
    hooks[name] = h


def register_encoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the Whisper-style audio encoder (model.audio_tower).

    VoxtralEncoder.forward: conv1/conv2 -> +embed_positions -> layers ->
    final layer_norm -> last_hidden_state. Hooking the encoder module
    captures the post-final-norm output (enc.out); per-layer hooks capture
    block outputs (1280-dim).
    """
    hooks: dict[str, CaptureHook] = {}
    enc = model.audio_tower
    for i in blocks:
        if 0 <= i < len(enc.layers):
            _hook_forward(hooks, f"enc.block.{i}.out", enc.layers[i])
    _hook_forward(hooks, "enc.out", enc)
    return hooks


def register_projector_hooks(model) -> dict[str, CaptureHook]:
    """Hook the multimodal projector (audio embeds in text-hidden space).

    The 4x frame-grouping reshape (1500,1280)->(375,5120) happens in
    model.get_audio_features BEFORE the projector, so the projector output
    is the (375, H) audio-embedding tensor scattered into the LM.
    """
    hooks: dict[str, CaptureHook] = {}
    _hook_forward(hooks, "proj.out", model.multi_modal_projector)
    return hooks


def register_decoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the inner Llama/Ministral LM.

    language_model (LlamaForCausalLM):
      model (LlamaModel)
        embed_tokens  => dec.token_emb (pre-injection embedding lookup)
        layers[i]     => dec.block.<i>.out
        norm          => dec.out_before_head

    Audio injection (masked_scatter) happens in
    VoxtralForConditionalGeneration.forward BEFORE delegating to
    language_model; we catch the post-injection inputs_embeds with a
    pre-hook on the inner LlamaModel.
    """
    hooks: dict[str, CaptureHook] = {}
    inner = model.language_model.model  # LlamaModel
    _hook_forward(hooks, "dec.token_emb", inner.embed_tokens)
    for i in blocks:
        if 0 <= i < len(inner.layers):
            _hook_forward(hooks, f"dec.block.{i}.out", inner.layers[i])
    _hook_forward(hooks, "dec.out_before_head", inner.norm)

    inj = CaptureHook()

    def pre(module, args, kwargs):
        if inj.value is not None:
            return
        emb = kwargs.get("inputs_embeds")
        if emb is not None:
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
    language: str | None,
) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "kind": "voxtral-transformers-native",
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
        "language": language,
        "request": "transcription",
    }


def decode_tokens(processor, token_ids: list[int]) -> str:
    """Decode generated token IDs via the mistral-common tokenizer."""
    tok = processor.tokenizer
    for attempt in (
        lambda: tok.decode(token_ids, skip_special_tokens=True),
        lambda: tok.decode(token_ids),
        lambda: processor.decode(token_ids, skip_special_tokens=True),
    ):
        try:
            return attempt().strip()
        except Exception:  # noqa: BLE001 - fall through to next decode form
            continue
    return ""


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    processor, model = load_reference(args)
    model_id, _ = resolve_model(args.model)
    model_dtype = model_dtype_name(model)

    audio_path = Path(args.audio).expanduser().resolve()
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        raise SystemExit(f"Voxtral expects 16kHz audio; got {sr} Hz in {audio_path}")

    language = None if args.language in (None, "", "auto") else args.language
    inputs = processor.apply_transcription_request(
        language=language,
        audio=str(audio_path),
        model_id=model_id,
        return_tensors="pt",
    )
    inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype,
        language=language,
    )

    # Frontend input the encoder receives (the WhisperFeatureExtractor mel).
    mel = to_np(inputs["input_features"])
    write_tensor(
        "enc.mel.in",
        mel,
        stage="encoder",
        source={**source, "hook": "processor.input_features"},
        out_dir=out_dir,
    )
    print(f"  enc.mel.in: shape={mel.shape} min={mel.min():.4e} max={mel.max():.4e} mean={mel.mean():.6e}")

    n_enc_blocks = len(model.audio_tower.layers)
    n_dec_blocks = len(model.language_model.model.layers)
    enc_blocks = sorted({0, n_enc_blocks // 2, n_enc_blocks - 1, *args.enc_blocks})
    dec_blocks = sorted({0, n_dec_blocks // 2, n_dec_blocks - 1, *args.dec_blocks})

    enc_hooks = register_encoder_hooks(model, enc_blocks)
    proj_hooks = register_projector_hooks(model)
    dec_hooks = register_decoder_hooks(model, dec_blocks)
    print(f"hooks: encoder={list(enc_hooks)} projector={list(proj_hooks)} decoder={list(dec_hooks)}")

    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = gen_out.sequences
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = sequences[0, prompt_len:].tolist()
    text = decode_tokens(processor, new_tokens)
    print(f"transcript: {text!r}")

    if gen_out.scores:
        logits0 = to_np(gen_out.scores[0])  # [vocab]
        write_tensor(
            "dec.logits_raw",
            logits0,
            stage="decoder",
            source={**source, "hook": "generate.scores[0]"},
            out_dir=out_dir,
        )
        print(f"  dec.logits_raw: shape={logits0.shape} min={logits0.min():.4e} max={logits0.max():.4e} mean={logits0.mean():.6e}")

    # Mid-generation step logits (validate.py autoregressive coverage): the
    # logits for the 9th generated token, i.e. after 8 greedy steps. Gated in
    # C++ at the matching step (cur_past == T_prompt + 7).
    if gen_out.scores and len(gen_out.scores) > 8:
        logits8 = to_np(gen_out.scores[8])  # [vocab]
        write_tensor(
            "dec.logits_raw.gen8",
            logits8,
            stage="decoder",
            source={**source, "hook": "generate.scores[8]"},
            out_dir=out_dir,
        )
        print(f"  dec.logits_raw.gen8: shape={logits8.shape} min={logits8.min():.4e} max={logits8.max():.4e} mean={logits8.mean():.6e}")

    def flush(name: str, t, stage: str) -> None:
        if t is None:
            print(f"  WARN: {name}: no value captured")
            return
        a = to_np(t)
        write_tensor(name, a, stage=stage, source={**source, "hook": name}, out_dir=out_dir)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")

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


def cmd_encoder(args: argparse.Namespace) -> int:
    """No-op alias.

    validate.py runs both `encoder` and `decode` subcommands per family
    dumper. Voxtral captures every tensor (enc.*, proj.*, dec.*) in a
    single decode pass, so the encoder pass is intentionally empty.
    """
    Path(args.out).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--audio", required=True, help="path to audio file (16kHz)")
    p.add_argument("--out", required=True, help="output directory for dumps")
    p.add_argument(
        "--language",
        default="en",
        help="ISO 639-1 language hint for the transcription request; "
             "'auto' (or empty) lets the model auto-detect.",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    p.add_argument("--torch-threads", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    encoder = sub.add_parser("encoder", help="no-op (decode pass dumps every tensor)")
    add_common_args(encoder)
    encoder.set_defaults(func=cmd_encoder)

    decode = sub.add_parser("decode", help="full encoder+projector+LM dump + transcript")
    add_common_args(decode)
    decode.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                        help="extra encoder block indices (first/mid/last always included)")
    decode.add_argument("--dec-blocks", type=int, nargs="*", default=[],
                        help="extra LM block indices (first/mid/last always included)")
    decode.add_argument("--max-new-tokens", type=int, default=128)
    decode.set_defaults(func=cmd_decode)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
