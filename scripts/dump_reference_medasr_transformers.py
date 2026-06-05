#!/usr/bin/env python3
"""
dump_reference_medasr_transformers.py — reference dumps for google/medasr
(LASR-CTC).

Mirrors the canonical inference path from the model card README verbatim:

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    text   = processor.batch_decode(outputs)[0]

See: https://huggingface.co/google/medasr

Reference framework pin: `transformers @ 65dc261512cbdb1ee72b88ae5b222f2605aad8e5`
(v5.0.0.dev0). v5.0.0 is unreleased; the LASR class registry only exists
on this exact dev commit.

Architecture (per transformers/models/lasr/modeling_lasr.py at the pinned
commit):

  LasrFeatureExtractor
    -> manual unfold + rfft, Hann symmetric (win_length=400, n_fft=512,
       hop=160), kaldi-mel weight matrix (HTK-style DC bin excluded,
       lower=125 Hz, upper=7500 Hz, 128 bands), log(clamp(min=1e-5)).
       NO preemph, NO dither, NO per-feature CMVN. Input features
       returned in f32; STFT internal precision is f64.

  LasrEncoder
    -> LasrEncoderSubsampling
         Linear(128 -> 512) ReLU -> Conv1d(512, 512, k=5, s=2) ReLU ->
         Conv1d(512, 256, k=5, s=2) ReLU -> Linear(256 -> 512)
         Effective downsampling = 4x. 40 ms output frame stride (25 fps).
    -> LasrEncoderRotaryEmbedding (rope_theta=10000, default)
    -> 17 x LasrEncoderBlock
         (norm_ff1 -> feed_forward1 -> macaron residual [1.5, 0.5])
         (norm_attn -> self_attn -> residual)
         (norm_conv -> conv -> macaron residual [2.0, 1.0])
         (norm_ff2 -> feed_forward2 -> macaron residual [1.5, 0.5])
         (norm_out)
       Non-standard residual scalars are load-bearing — see intake risks.
    -> out_norm

  LasrForCTC
    -> ctc_head = Conv1d(hidden=512, vocab=512, kernel_size=1)
       (Conv-1 not Linear, but functionally equivalent — matches NeMo
       CTC decoding-layer convention.)

CTC blank id = 0 (`<epsilon>`), NOT vocab_size - 1.

Dump points:
  mel.in                    input_features [T_mel, 128] (host log-mel)
  enc.subsampling.out       output of encoder.subsampler   [T_enc, 512]
  enc.block.{i}.out         output of encoder.layers[i]    [T_enc, 512]
  enc.block.0.post_ff1      hidden_state after FF1 residual
  enc.block.0.post_attn     hidden_state after self_attn residual
  enc.block.0.post_conv     hidden_state after conv residual
  enc.block.0.post_ff2      hidden_state after FF2 residual (pre norm_out)
  enc.out_norm.out          output of encoder.out_norm     [T_enc, 512]
  enc.ctc_logits            logits from ctc_head           [T_enc, 512]

Usage:

    uv run --project scripts/envs/medasr \\
      scripts/dump_reference_medasr_transformers.py decode \\
      --model google/medasr \\
      --audio samples/jfk.wav \\
      --out build/validate/medasr/medasr/jfk/decode/ref
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
    """Load processor + model exactly as the model card README does."""
    import torch
    import transformers
    from transformers import AutoModelForCTC, AutoProcessor

    model_id, local_only = resolve_model(args.model)
    revision = None if local_only else args.revision
    source = "local path" if local_only else "HuggingFace"
    rev_text = f", revision={revision}" if revision else ""
    print(
        f"loading {model_id} ({source}, transformers {transformers.__version__}"
        f", device={args.device}{rev_text})..."
    )

    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, local_files_only=local_only,
    )

    dtype = {"bf16": torch.bfloat16,
             "f16":  torch.float16,
             "f32":  torch.float32}[args.dtype]

    model = AutoModelForCTC.from_pretrained(
        model_id, revision=revision, local_files_only=local_only,
        dtype=dtype, attn_implementation=args.attn_impl,
    ).to(args.device).eval()

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


class PreHookInput:
    """Captures the first positional arg of a forward-pre-hook.

    Used to dump intermediate hidden_states within a conformer block by
    pre-hooking the next submodule — its input is the running residual
    at that point in block.forward.
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
    fwd("enc.subsampling.out", enc.subsampler)
    for i in enc_block_idx:
        if 0 <= i < len(enc.layers):
            fwd(f"enc.block.{i}.out", enc.layers[i])
    fwd("enc.out_norm.out", enc.out_norm)
    fwd("enc.ctc_logits", model.ctc_head)

    # Sub-step dumps inside block 0 to localize residual / scalar drift.
    # The block applies FF1 / attn / conv / FF2 / norm_out in sequence;
    # capture each running hidden_state by pre-hooking the next submodule.
    # `feed_forward1` is the FIRST submodule in the block, so its
    # pre-hook input == block input; we don't dump that (covered by
    # subsampling.out + nothing in between). Pre-hook downstream:
    #   pre-hook(norm_self_att.forward) -> hidden after FF1 residual
    #   pre-hook(norm_conv.forward)     -> hidden after attn residual
    #   pre-hook(norm_feed_forward2)    -> hidden after conv residual
    #   pre-hook(norm_out.forward)      -> hidden after FF2 residual
    block0 = enc.layers[0]
    sub_hook_targets = [
        ("enc.block.0.post_ff1",  block0.norm_self_att),
        ("enc.block.0.post_attn", block0.norm_conv),
        ("enc.block.0.post_conv", block0.norm_feed_forward2),
        ("enc.block.0.post_ff2",  block0.norm_out),
    ]
    for name, target_module in sub_hook_targets:
        h = PreHookInput()
        target_module.register_forward_pre_hook(h)
        hooks[name] = h

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
        raise SystemExit(f"medasr expects 16 kHz; got {sr} Hz in {audio_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = {
        "kind": "medasr-transformers",
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

    # README path: processor(pcm, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = processor(pcm, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    input_features = inputs["input_features"]
    print(f"  input_features: {tuple(input_features.shape)}  "
          f"mel range [{input_features.min().item():.4f}, "
          f"{input_features.max().item():.4f}]")

    write_tensor("mel.in", to_np(input_features), "encoder", source, out_dir=out_dir)

    # Register hooks
    n_enc_layers = len(model.encoder.layers)
    block_idx = sorted({0, n_enc_layers // 2 - 1, n_enc_layers // 2,
                        n_enc_layers - 1, *args.enc_blocks})
    block_idx = [b for b in block_idx if 0 <= b < n_enc_layers]
    hooks = register_hooks(model, block_idx)

    # Inference: README path. CTC `generate` does argmax + collapse-repeats +
    # drop-blanks internally (see LasrForCTC.generate at the pinned commit);
    # batch_decode then maps token IDs to a text string.
    with torch.inference_mode():
        token_ids = model.generate(**inputs)

    # skip_special_tokens=True matches scripts/wer/run_reference_medasr_transformers.py
    # so the validate.py transcript and the WER reference transcript agree on
    # one decode convention (the README path uses the non-skipping variant but
    # leaves `</s>` in the transcript, which inflates dataset WER by ~0.2pp
    # on test-clean and breaks Stage 4 transcript_compare against any C++ that
    # also strips specials — see medasr.json tolerances _comment).
    transcriptions = processor.batch_decode(token_ids, skip_special_tokens=True)
    text_pred = (transcriptions[0] if transcriptions else "").strip()
    print(f"  transcript : {text_pred!r}")

    # Dump captured tensors
    fixed_names = ["enc.subsampling.out", "enc.out_norm.out", "enc.ctc_logits"]
    block_names = [f"enc.block.{i}.out" for i in block_idx]
    substep_names = [
        "enc.block.0.post_ff1",
        "enc.block.0.post_attn",
        "enc.block.0.post_conv",
        "enc.block.0.post_ff2",
    ]
    for name in fixed_names + block_names + substep_names:
        h = hooks.get(name)
        if h is None or h.value is None:
            print(f"  (skipping {name}: no capture)")
            continue
        a = to_np(h.value)
        write_tensor(name, a, "encoder", source, out_dir=out_dir)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")

    # token_ids are post-collapse CTC output tokens after generate().
    tokens_list: list[int] = []
    if hasattr(token_ids, "tolist"):
        flat = token_ids[0] if token_ids.dim() > 1 else token_ids
        tokens_list = [int(t) for t in flat.tolist()]
    write_transcript(out_dir, text_pred, source=source, tokens=tokens_list)
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    # Encoder-only stage is a no-op: decode already writes every tensor
    # the dump_coverage.json catalog expects. Keep the stage directory
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
        sp.add_argument("--dtype", default="f32", choices=["bf16", "f16", "f32"],
                        help="Reference dtype. F32 is the storage dtype on disk "
                             "and the porting reference; bf16/f16 only for ad-hoc probes.")
        sp.add_argument("--attn-impl", default="eager",
                        choices=["eager", "sdpa", "flex_attention"],
                        help="LASR does NOT support flash-attn (see _supports_flash_attn=False).")
        sp.add_argument("--torch-threads", type=int, default=4)
        sp.add_argument("--language", default=None,
                        help="ignored (medasr is monolingual English)")

    encoder = sub.add_parser("encoder", help="no-op (decode dumps all tensors)")
    add_common_args(encoder)
    encoder.set_defaults(func=cmd_encoder)

    decode = sub.add_parser("decode", help="full encoder + CTC head dump + transcript")
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
