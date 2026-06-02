#!/usr/bin/env python3
"""
dump_reference_granite_nar_transformers.py — reference dumps for
IBM Granite Speech NAR (non-autoregressive editor).

Mirrors the canonical inference path from the HuggingFace model card
verbatim:

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                       attn_implementation=..., dtype=...)
    inputs = processor([waveform], device=device)
    output = model.transcribe(**inputs)
    text   = processor.batch_decode(output.preds)

See: https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar

Architecture (per modeling_granite_speech_nar.py, snapshot 99a4df9...):

  GraniteSpeechNarCTCEncoder
    -> input_linear (160 -> 1024)
    -> 16 x GraniteSpeechNarConformerBlock (Shaw-relative block-local attn)
    -> self-conditioned char-CTC bypass at layer 8 (1-indexed; after 0-indexed
       layer 7): mid_logits = encoder.out(dropout(hidden)) over 348 chars,
       softmax, then encoder.out_mid (348 -> 1024) injected as residual
    -> all_hidden_states tuple captured at indices [4, 8, 12, -1]
    -> posterior-weighted pool (window=4), encoder.out_bpe (1024 -> 100353)
       on valid frames -> BPE-CTC flat logits (output.encoder_logits)

  GraniteSpeechNarProjector  (windowed simplified Q-Former)
    -> per-layer LayerNorms over the 4 captured layers -> cat -> layer_projector
    -> GELU -> pad along T to nblocks * block_size -> mean-pool to [nblocks, q, H]
    -> 2 x QFormerLayer (cross-attn from query stream to x+window_positions,
       MLP, no self-attn) -> out_norm -> out_linear -> [B, nblocks*q, llm_dim]

  GraniteSpeechNarLM (Granite-4 base, bidirectional)
    -> GraniteSpeechNarModel.forward applies create_bidirectional_mask()
       natively; no need to patch any mask helpers.
    -> flat input layout per sample: [audio_emb, text_emb_with_insertion_slots]
       (audio is pre-divided by embedding_multiplier when
       scale_projected_embeddings=True; multiplier is then applied to the whole
       flat sequence inside GraniteSpeechNarModel.forward)
    -> single forward pass (no KV cache, no token loop)
    -> per-sample text-portion logits returned in output.logits[i]

Dump points:
  enc.mel.in              input_features [T_enc, 160]  (host log-mel after stack)
  enc.input_linear.out    output of encoder.input_linear  [T_enc, 1024]
  enc.block.{i}.out       output of encoder.layers[i]    [T_enc, 1024]  (pre-bypass)
  enc.ctc_logits          char-CTC mid_logits at bypass  [T_enc, 348]
                           (hook on encoder.out; eval-mode dropout is identity)
  proj.qformer.out        output of projector.qformer    [nblocks, q, 2048]
  proj.out                output of projector            [nblocks*q, 2048]
  dec.flat_embeds         inputs_embeds going into LM    [T_total, 2048]
                           (captured PRE embedding_multiplier scaling)
  dec.text_logits         output.logits[0]               [n_text, 100352]

Usage:

    uv run --project scripts/envs/granite_nar \\
      scripts/dump_reference_granite_nar_transformers.py decode \\
      --model ibm-granite/granite-speech-4.1-2b-nar \\
      --audio samples/jfk.wav \\
      --out build/validate/granite_nar/granite-speech-4.1-2b-nar/jfk/ref
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    """Load processor + model exactly as the HF model card README does,
    with two CPU-platform adaptations: eager attention (no flash-attn on
    Apple) and an explicit dtype kwarg.

    No mask patching is needed for the current modeling snapshot: the
    NAR LM uses GraniteSpeechNarModel which calls create_bidirectional_mask()
    natively, and GraniteSpeechNarAttention pins is_causal=False.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor

    model_id, local_only = resolve_model(args.model)
    revision = None if local_only else args.revision
    source = "local path" if local_only else "HuggingFace"
    rev_text = f", revision={revision}" if revision else ""
    print(
        f"loading {model_id} ({source}, transformers {transformers.__version__}"
        f", device={args.device}{rev_text})..."
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_only,
    )

    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map=args.device,
    ).eval()

    return processor, model


class CaptureHook:
    """Captures the first forward output of a module."""

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


class PreHookEmbeds:
    """Captures inputs_embeds from a forward-pre-hook (kwargs path)."""

    def __init__(self) -> None:
        self.value = None

    def __call__(self, module, args_, kwargs_) -> None:
        if self.value is not None:
            return
        emb = kwargs_.get("inputs_embeds")
        if emb is not None:
            self.value = emb


class PreHookInput:
    """Captures the first positional arg of a forward-pre-hook.

    Used to dump intermediate hidden_states within a conformer block by
    hooking the downstream submodule (attn / conv / ff2 / post_norm) —
    its input is the running residual at that point in block.forward.
    """

    def __init__(self) -> None:
        self.value = None

    def __call__(self, module, inputs) -> None:
        if self.value is not None or not inputs:
            return
        self.value = inputs[0]


def register_hooks(model, enc_block_idx: list[int]) -> dict[str, object]:
    """Attach hooks at every dump point. Returns a name -> hook dict."""
    hooks: dict[str, object] = {}

    def fwd(name: str, module) -> None:
        h = CaptureHook()
        module.register_forward_hook(h)
        hooks[name] = h

    enc = model.encoder
    fwd("enc.input_linear.out", enc.input_linear)
    for i in enc_block_idx:
        if 0 <= i < len(enc.layers):
            fwd(f"enc.block.{i}.out", enc.layers[i])
    # The char-CTC head is `encoder.out`. It's invoked exactly once per
    # forward at self_conditioning_layer to produce mid_logits.
    fwd("enc.ctc_logits", enc.out)

    # Sub-step dumps inside block 0 to localize bf16 cascade drift.
    # block.forward applies the FF1/attn/conv/FF2 residuals in sequence;
    # we capture each running hidden_state by pre-hooking the next
    # submodule.
    block0 = enc.layers[0]
    sub_hook_targets = [
        ("enc.block.0.post_ff1",  block0.attn),
        ("enc.block.0.post_attn", block0.conv),
        ("enc.block.0.post_conv", block0.ff2),
        ("enc.block.0.post_ff2",  block0.post_norm),
    ]
    for name, target_module in sub_hook_targets:
        h = PreHookInput()
        target_module.register_forward_pre_hook(h)
        hooks[name] = h

    proj = model.projector
    fwd("proj.qformer.out", proj.qformer)
    fwd("proj.out", proj)

    pre = PreHookEmbeds()
    model.language_model.model.register_forward_pre_hook(pre, with_kwargs=True)
    hooks["dec.flat_embeds"] = pre

    return hooks


def cmd_decode(args: argparse.Namespace) -> int:
    import torch
    import transformers

    configure_torch(args)
    processor, model = load_reference(args)
    model_id, _ = resolve_model(args.model)
    model_dtype = model_dtype_name(model)

    audio_path = Path(args.audio).expanduser().resolve()
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        raise SystemExit(f"NAR expects 16 kHz; got {sr} Hz in {audio_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = {
        "kind": "granite-speech-nar-transformers",
        "transformers_version": transformers.__version__,
        "transformers_file": transformers.__file__,
        "model": model_id,
        "model_revision": args.revision,
        "model_dtype": model_dtype,
        "attn_implementation": args.attn_impl,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "audio": audio_path.name,
        "n_samples": int(pcm.size),
        "sample_rate": int(sr),
    }

    # ---- Processor: matches the README's `inputs = processor([waveform], device=device)` ----
    waveform = torch.from_numpy(pcm)
    inputs = processor([waveform], device=args.device)
    input_features = inputs["input_features"]
    print(f"  input_features: {tuple(input_features.shape)}  "
          f"mel range [{input_features.min().item():.4f}, "
          f"{input_features.max().item():.4f}]")

    write_tensor("enc.mel.in", to_np(input_features), "encoder", source, out_dir=out_dir)

    # ---- Register hooks ----
    n_enc_layers = len(model.encoder.layers)
    block_idx = sorted({0, n_enc_layers // 2 - 1, n_enc_layers // 2,
                        n_enc_layers - 1, *args.enc_blocks})
    block_idx = [b for b in block_idx if 0 <= b < n_enc_layers]
    hooks = register_hooks(model, block_idx)

    # ---- Inference: README path ----
    with torch.inference_mode():
        output = model.transcribe(**inputs, output_encoder_logits=False)

    transcriptions = processor.batch_decode(output.preds)
    text_pred = transcriptions[0] if transcriptions else ""

    # CTC-collapsed initial hypothesis from the BPE-CTC encoder head.
    initial_pred = ""
    if output.encoder_preds:
        initial_list = processor.batch_decode(output.encoder_preds)
        initial_pred = initial_list[0] if initial_list else ""
    print(f"  CTC initial : {initial_pred!r}")
    print(f"  NAR final   : {text_pred!r}")

    # ---- Dump captured tensors ----
    enc_names = ["enc.input_linear.out", "enc.ctc_logits"] + [
        f"enc.block.{i}.out" for i in block_idx
    ] + [
        "enc.block.0.post_ff1",
        "enc.block.0.post_attn",
        "enc.block.0.post_conv",
        "enc.block.0.post_ff2",
    ]
    for name in enc_names:
        h = hooks.get(name)
        if h is None or h.value is None:
            print(f"  (skipping {name}: no capture)")
            continue
        a = to_np(h.value)
        write_tensor(name, a, "encoder", source, out_dir=out_dir)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")

    for name in ("proj.qformer.out", "proj.out"):
        h = hooks.get(name)
        if h is None or h.value is None:
            print(f"  (skipping {name}: no capture)")
            continue
        a = to_np(h.value)
        write_tensor(name, a, "projector", source, out_dir=out_dir)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e}")

    pre = hooks["dec.flat_embeds"]
    if pre.value is not None:
        a = to_np(pre.value)
        write_tensor("dec.flat_embeds", a, "decoder", source, out_dir=out_dir)
        print(f"  dec.flat_embeds: shape={a.shape}")

    # Per-sample text-portion lm_head logits. transcribe() returns logits
    # alongside preds; output.logits[0] is the first sample's [n_text, vocab]
    # text-portion slice (NLE bypasses the /logits_scaling divisor — see
    # GraniteSpeechNarLM.forward).
    if output.logits:
        a = to_np(output.logits[0])
        write_tensor("dec.text_logits", a, "decoder", source, out_dir=out_dir)
        print(f"  dec.text_logits: shape={a.shape}")

    write_transcript(out_dir, text_pred, source=source)
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    print(f"  ctc_initial: {initial_pred!r}")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    # Encoder-only stage is a no-op: decode already writes every tensor
    # the dump_coverage.json catalog expects. Keeping the stage directory
    # present so validate.py's enumerator finds it.
    Path(args.out).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, help="HF repo id or local path")
        sp.add_argument("--revision", default=None,
                        help="HF revision (commit hash) to pin against drift")
        sp.add_argument("--audio", required=True, help="path to 16 kHz mono audio")
        sp.add_argument("--out", required=True, help="output directory for dumps")
        sp.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
        sp.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
        sp.add_argument("--attn-impl", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="attn backend for the LM (encoder always uses SDPA-math)")
        sp.add_argument("--torch-threads", type=int, default=4)
        sp.add_argument("--language", default=None,
                        help="ignored (NAR is single-task English-only family)")

    encoder = sub.add_parser("encoder", help="no-op (decode dumps all tensors)")
    add_common_args(encoder)
    encoder.set_defaults(func=cmd_encoder)

    decode = sub.add_parser("decode", help="full encoder+projector+LM dump + transcript")
    add_common_args(decode)
    decode.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                        help="extra encoder block indices to dump (default: 0, mid-1, mid, last)")
    decode.set_defaults(func=cmd_decode)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
