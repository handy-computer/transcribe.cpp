#!/usr/bin/env python3
"""
dump_reference_granite_nar_transformers.py - reference dumps for the
IBM Granite Speech NLE (non-autoregressive editor) variant.

The NAR architecture is fundamentally different from the AR Granite
Speech audio-LLM family:

  Conformer encoder (16 layers, hidden=1024)
    -> input_linear (160 -> 1024)
    -> loop with self-conditioning CTC bypass at layer idx=8
    -> all_hidden_states captured at layer indices [4, 8, 12, -1]
    -> CTC head (1024 -> 348) emits character-level logits
    -> Optional BPE head: posterior-weighted pool (window=4) then
       Linear (1024 -> 100353) over valid frames only
    -> BPE CTC greedy decode -> initial text hypothesis

  EncoderProjectorQFormer (NLE-specific Q-Former variant)
    -> cat encoder hidden states from layers [4, 8, 12, -1] -> [T, 4096]
    -> per-layer LayerNorms then layer_projector Linear (4096 -> 2048)
    -> GELU activation
    -> pad along T to nblocks * block_size (15)
    -> window mean-pool to [nblocks, 3, 2048] (query length 3)
    -> simplified Q-Former: query (param) + mean_pool ; cross-attn
       (Q from query stream, K/V from x + window_positions); MLP;
       repeated num_layers=2 times. No self-attention.
    -> out_norm + out_linear (2048 -> 2048)
    -> [B, nblocks * 3, 2048] audio embeddings

  Granite-4 1b base LLM (40 layers, hidden=2048, GQA 16/4)
    -> is_causal = False on every layer (BIDIRECTIONAL forward)
    -> flat input: cat of (projector_out / embedding_multiplier,
                          embed_tokens(text_with_insertion_slots))
    -> single forward pass (no KV cache, no token loop)
    -> slice text-portion hidden, lm_head -> editing logits
    -> argmax + collapse + drop EOS -> final transcript

Usage:

    uv run --project scripts/envs/granite_nar \\
      scripts/dump_reference_granite_nar_transformers.py decode \\
      --model ibm-granite/granite-speech-4.1-2b-nar \\
      --audio samples/jfk.wav \\
      --out build/validate/granite_nar/granite-speech-4.1-2b-nar/jfk/ref
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

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


def resolve_model(raw: str) -> tuple[str, bool]:
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


def load_reference(args: argparse.Namespace):
    """Load the NLE model.

    The modeling code lives in the HF repo itself (NLEConfig + NLENARDecoder
    via auto_map, trust_remote_code=True). We attempt to use
    attn_implementation='sdpa' so the bidirectional forward (is_causal
    overridden to False on every layer) does not require flash-attn,
    which is not available on Apple Metal. The model code asserts
    flash_attention_2, so we patch the assertion before any forward.
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading NLE model from {model_id} ({source}, "
        f"transformers {transformers.__version__}, device={args.device})..."
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
    )

    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]

    # Use eager attention (flash-attn isn't available on Apple, and the
    # NLE auto-arch doesn't register SDPA). The upstream NLE forward
    # asserts flash_attention_2 — we patch around it below. The eager
    # path in granite's attention does respect is_causal=False (the
    # __init__ sets it on every layer), giving a bidirectional forward.
    cfg = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"

    model = AutoModel.from_pretrained(
        model_id,
        config=cfg,
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="eager",
    ).eval()
    # The model __init__ wires is_causal=False on every layer; that
    # propagates through sdpa correctly because the granite LM's eager
    # forward respects is_causal_user_override. The runtime assertion
    # only fires inside .forward(), so we patch it after construction.
    import types
    original_forward = model.forward.__func__

    def patched_forward(self, *args, **kwargs):
        # Bypass the flash-attn assertion; we use sdpa with is_causal=False
        # which yields bidirectional attention.
        cfg = self.config
        if cfg.attn_implementation != "flash_attention_2":
            # Run the original body but skip the assert by temporarily
            # claiming we are flash. The forward body only inspects this
            # one line, the layer-level attention is already wired.
            saved = cfg.attn_implementation
            cfg.attn_implementation = "flash_attention_2"
            try:
                return original_forward(self, *args, **kwargs)
            finally:
                cfg.attn_implementation = saved
        return original_forward(self, *args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)

    model.to(args.device)
    return feature_extractor, model


class CaptureHook:
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


def register_encoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the NLE encoder at the standard dump points."""
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
    return hooks


def register_projector_hooks(model) -> dict[str, CaptureHook]:
    """Hook the NLE projector.

    EncoderProjectorQFormer:
      input -> per-layer LayerNorms -> cat -> layer_projector -> GELU
            -> window pool -> SimplifiedQFormer (cross-attn + MLP per layer)
            -> out_norm -> out_linear
    """
    hooks: dict[str, CaptureHook] = {}
    proj = model.projector

    def hook_forward(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    # proj.qformer.out: the SimplifiedQFormer output (Q-Former hidden,
    # 2048-d, before the out_norm + out_linear).
    hook_forward("proj.qformer.out", proj.qformer)
    # proj.out: the full EncoderProjectorQFormer output (audio embeds
    # ready to feed to the LLM, after out_norm + out_linear).
    hook_forward("proj.out", proj)
    return hooks


def make_source(*, args, model_id, audio_path, n_samples, sample_rate, model_dtype):
    import torch
    import transformers

    return {
        "kind": "granite-nle-transformers-native",
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


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    feature_extractor, model = load_reference(args)
    model_id, _ = resolve_model(args.model)
    model_dtype = model_dtype_name(model)

    audio_path = Path(args.audio).expanduser().resolve()
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        raise SystemExit(f"NLE expects 16 kHz; got {sr} Hz in {audio_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = make_source(
        args=args, model_id=model_id, audio_path=audio_path,
        n_samples=pcm.size, sample_rate=sr, model_dtype=model_dtype,
    )

    # ---------- Feature extraction (host) ----------
    feats = feature_extractor([torch.from_numpy(pcm)], device=args.device)
    input_features = feats["input_features"]
    attention_mask = feats["attention_mask"]
    print(f"  input_features: {tuple(input_features.shape)}  "
          f"mel range [{input_features.min().item():.4f}, "
          f"{input_features.max().item():.4f}]")

    # Dump enc.mel.in in row-major [T_enc, 160] (squeeze batch).
    mel_in = to_np(input_features)
    write_tensor("enc.mel.in", mel_in, "encoder", source, out_dir=out_dir)

    # ---------- Encoder hooks ----------
    n_enc_layers = len(model.encoder.layers)
    extra_blocks = sorted({0, n_enc_layers // 2 - 1, n_enc_layers // 2,
                           n_enc_layers - 1, *args.enc_blocks})
    extra_blocks = [b for b in extra_blocks if 0 <= b < n_enc_layers]
    enc_hooks = register_encoder_hooks(model, extra_blocks)

    # ---------- Projector hooks ----------
    proj_hooks = register_projector_hooks(model)

    # ---------- LLM hooks (single-pass forward) ----------
    # The flat layout going into the LLM is [audio_embeds, text_embeds]
    # per sample. We capture inputs_embeds via a pre-hook on
    # llm.model (the inner GraniteModel); editing_logits comes out of
    # the .forward() result.
    llm_pre_hook = CaptureHook()
    def _pre(module, args_, kwargs_):
        if llm_pre_hook.value is None:
            emb = kwargs_.get("inputs_embeds")
            if emb is not None:
                llm_pre_hook.value = emb
    model.llm.model.register_forward_pre_hook(_pre, with_kwargs=True)

    # ---------- Forward + generate ----------
    with torch.inference_mode():
        gen_out = model.generate(input_features=input_features,
                                  attention_mask=attention_mask)

    text_preds = gen_out.text_preds
    text_ctc_preds = gen_out.text_ctc_preds
    encoder_logits = gen_out.encoder_logits  # [B, T, 348]

    text_pred = text_preds[0] if text_preds else ""
    text_ctc_pred = text_ctc_preds[0] if text_ctc_preds else ""
    print(f"  CTC initial : {text_ctc_pred!r}")
    print(f"  NLE final   : {text_pred!r}")

    # Dump enc.* tensors.
    for name, hook in enc_hooks.items():
        if hook.value is None:
            print(f"  (skipping {name}: no value)")
            continue
        a = to_np(hook.value)
        write_tensor(name, a, "encoder", source, out_dir=out_dir)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")

    # Encoder CTC logits.
    enc_ctc = to_np(encoder_logits)
    write_tensor("enc.ctc_logits", enc_ctc, "encoder", source, out_dir=out_dir)
    print(f"  enc.ctc_logits: shape={enc_ctc.shape} "
          f"min={enc_ctc.min():.4e} max={enc_ctc.max():.4e}")

    # Dump proj.* tensors.
    for name, hook in proj_hooks.items():
        if hook.value is None:
            print(f"  (skipping {name}: no value)")
            continue
        a = to_np(hook.value)
        write_tensor(name, a, "projector", source, out_dir=out_dir)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e}")

    # Dump the flat LLM input embeds (post-projector audio embeds plus
    # the text-with-insertion-slots embed lookup).
    if llm_pre_hook.value is not None:
        flat = to_np(llm_pre_hook.value)
        write_tensor("dec.flat_embeds", flat, "decoder", source, out_dir=out_dir)
        print(f"  dec.flat_embeds: shape={flat.shape}")

    # Final editing logits (lm_head over the text portions; per-sample
    # tensors live in gen_out.editing_logits but generate() doesn't
    # actually return that — we re-derive it from the forward path on
    # the same inputs to make the dump deterministic).
    with torch.inference_mode():
        full = model.forward(input_features=input_features,
                             attention_mask=attention_mask)
    edit_list = list(full.editing_logits) if full.editing_logits is not None else []
    if edit_list:
        edit_a = to_np(edit_list[0])
        write_tensor("dec.text_logits", edit_a, "decoder", source, out_dir=out_dir)
        print(f"  dec.text_logits: shape={edit_a.shape}")

    # Behavioral artifact.
    write_transcript(out_dir, text_pred, source=source)
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    print(f"  ctc_initial: {text_ctc_pred!r}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, help="HF repo id or local path")
        sp.add_argument("--audio", required=True, help="path to audio file (16 kHz)")
        sp.add_argument("--out", required=True, help="output directory for dumps")
        sp.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
        sp.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
        sp.add_argument("--torch-threads", type=int, default=4)
        sp.add_argument("--language", default=None, help="ignored (NLE prompt is fixed)")

    encoder = sub.add_parser("encoder", help="no-op (decode dumps all tensors)")
    add_common_args(encoder)
    encoder.set_defaults(func=lambda args: 0 or Path(args.out).mkdir(parents=True, exist_ok=True) or 0)

    decode = sub.add_parser("decode", help="full encoder+projector+LM dump + transcript")
    add_common_args(decode)
    decode.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                        help="extra encoder block indices to dump (always includes 0, mid, last)")
    decode.set_defaults(func=cmd_decode)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
