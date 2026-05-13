#!/usr/bin/env python3
"""
dump_reference_gigaam_author.py - generate GigaAM reference tensors via
the salute-developers/GigaAM author package.

Single subcommand `decode` runs the full forward pass (frontend +
encoder + optional decode) and dumps every stage. The `encoder` alias
exists for the standard dumper CLI shape; it forwards to the same
function. For the SSL variant (encoder-only checkpoint), the decode
section is skipped and no transcript is written.

Usage:

    uv run --project scripts/envs/gigaam \\
      scripts/dump_reference_gigaam_author.py decode \\
      --variant v3_e2e_rnnt \\
      --audio samples/ru.wav \\
      --out build/validate/gigaam/gigaam-v3-e2e-rnnt/ru/decode/ref

Dump points (match tests/tolerances/gigaam.json):

    frontend.mel.out          log-mel features after FeatureExtractor (preprocessor)
    enc.subsample.out         after pre_encode (StridingSubsampling, conv1d factor 4)
    enc.pos_emb               rotary [cos; sin] stacked PE tensor
    enc.block.{0,7,15}.out    sparse ConformerLayer outputs (16 layers total)
    enc.out                   final encoder output (post-transpose)

    For RNN-T heads (rnnt, e2e_rnnt):
        rnnt.encoded            encoder output as fed into the joint network
                                (transposed to [B, T, D])
        rnnt.tokens             greedy-decoded token id sequence (transcript.json)
        rnnt.text               decoded transcript (transcript.json)

    For CTC heads (ctc, e2e_ctc):
        ctc.logits.raw          CTCHead output before log_softmax (Conv1d projection)
        ctc.log_probs           after log_softmax
        ctc.tokens              greedy-decoded token id sequence (transcript.json)
        ctc.text                decoded transcript (transcript.json)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Re-use the shared ref_dump helper for write_tensor + write_transcript so
# the on-disk contract is identical across families.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from lib.ref_dump import write_tensor, write_transcript  # noqa: E402


# Branch SHA pin per variant — locked at Stage 1 intake.
_VARIANT_REVISIONS = {
    "v3_e2e_rnnt": "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e",  # main
    "v3_e2e_ctc":  "cec030b4c4f35d928e4a9044a3bdb29ebd499fac",
    "v3_rnnt":     "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8",
    "v3_ctc":      "15ef3b5a88da78f93134b3cb7f015c70aefa8946",
}
_HF_REPO = "ai-sage/GigaAM-v3"


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


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    return {
        torch.bfloat16: "bf16",
        torch.float16:  "f16",
        torch.float32:  "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


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


def hook_module(module, hook: CaptureHook):
    return module.register_forward_hook(hook)


def load_reference(args: argparse.Namespace):
    """Load a GigaAM-v3 variant via gigaam.load_model.

    Forces fp16_encoder=False (the package default is True; F32 is what
    our intake declares and what every other reference dumper uses).
    Forces device=cpu so the autocast branch in GigaAMASR.forward is
    skipped — the cuda path enters torch.autocast(float16) regardless of
    fp16_encoder.
    """
    import gigaam

    if args.variant not in _VARIANT_REVISIONS:
        raise SystemExit(
            f"unknown variant {args.variant!r}; expected one of "
            f"{sorted(_VARIANT_REVISIONS)}"
        )
    print(
        f"Loading gigaam.load_model({args.variant!r}, fp16_encoder=False, "
        f"device={args.device!r})"
    )
    model = gigaam.load_model(
        args.variant,
        fp16_encoder=False,
        device=args.device,
    )
    model.eval()
    return model


def is_rnnt_variant(variant: str) -> bool:
    return variant.endswith("_rnnt")


def is_ctc_variant(variant: str) -> bool:
    return variant.endswith("_ctc")


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    model = load_reference(args)

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the model's own audio loader so we exercise the same pipeline
    # that real users hit. load_audio returns float32 PCM at 16 kHz.
    import gigaam

    pcm = gigaam.load_audio(str(audio_path))  # [N], float32, 16 kHz
    pcm_np = pcm.detach().cpu().numpy().astype(np.float32)
    print(f"audio: {audio_path.name} samples={pcm_np.size} sr=16000")

    device = torch.device(args.device)
    wav = pcm.to(device).unsqueeze(0)  # [1, N]
    length = torch.full([1], wav.shape[-1], dtype=torch.long, device=device)

    source: dict[str, Any] = {
        "kind": "gigaam-author",
        "framework": "author_repo_gigaam",
        "hf_repo": _HF_REPO,
        "hf_revision": _VARIANT_REVISIONS[args.variant],
        "variant": args.variant,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "model_dtype": model_dtype_name(model),
        "audio": audio_path.name,
        "n_samples": int(pcm_np.size),
        "sample_rate": 16000,
        "fp16_encoder": False,
    }

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )
        write_tensor(name, a, stage=stage, source=source, out_dir=out_dir)

    # ---- frontend (preprocessor) -----------------------------------------
    preprocessor = model.preprocessor
    with torch.inference_mode():
        feats, feat_lengths = preprocessor(wav, length)
    # feats: [B=1, n_mels=64, T_mel]
    print(
        f"frontend: feats.shape={tuple(feats.shape)} "
        f"feat_lengths={feat_lengths.tolist()}"
    )
    dump("frontend.mel.out", feats, "frontend.mel")

    # ---- encoder hooks ---------------------------------------------------
    encoder = model.encoder
    pre_h = CaptureHook(); hook_module(encoder.pre_encode, pre_h)
    pos_h = CaptureHook(); hook_module(encoder.pos_enc, pos_h)

    layers = list(encoder.layers)
    n_layers = len(layers)
    # Sparse layer dumps: first, mid, last. For 16-layer Conformer that's 0/7/15.
    layer_idxs = sorted({0, n_layers // 2 - 1, n_layers - 1})
    layer_hooks: dict[int, CaptureHook] = {}
    for i in layer_idxs:
        h = CaptureHook()
        hook_module(layers[i], h)
        layer_hooks[i] = h

    # ---- forward through encoder -----------------------------------------
    # Replicate GigaAMASR.forward: preprocessor + encoder (no autocast on CPU).
    with torch.inference_mode():
        encoded, encoded_len = encoder(feats, feat_lengths)
    # encoded: [B=1, D=768, T_enc] (post-transpose inside encoder.forward)
    print(
        f"encoder: encoded.shape={tuple(encoded.shape)} "
        f"encoded_len={encoded_len.tolist()}"
    )

    dump("enc.subsample.out", pre_h.value, "encoder.subsample")
    # pos_enc returns (audio_signal, pos_emb); CaptureHook stored the first
    # element of the tuple (audio_signal). The PE tensor itself lives on
    # the module as `.pe` after `extend_pe`. Dump it separately.
    if hasattr(encoder.pos_enc, "pe"):
        dump("enc.pos_emb", encoder.pos_enc.pe, "encoder.pos_emb")
    for i, h in layer_hooks.items():
        dump(f"enc.block.{i}.out", h.value, f"encoder.block.{i}")
    dump("enc.out", encoded, "encoder.out")

    # ---- decode (variant-specific) ---------------------------------------
    head = model.head
    decoding = model.decoding

    if is_rnnt_variant(args.variant):
        # The joint/predictor loop is data-dependent and per-frame; rather
        # than instrumenting the loop, we dump the encoder input that
        # feeds it. Stage 4 will validate the encoder side numerically and
        # the transcript / token ids behaviorally.
        encoded_for_joint = encoded.transpose(1, 2).contiguous()  # [B, T, D]
        dump("rnnt.encoded", encoded_for_joint, "decoder.rnnt.encoded")

        with torch.inference_mode():
            results = decoding.decode(head, encoded, encoded_len)
        text, token_ids, token_frames = results[0]
        print(f"transcript: {text!r}")
        print(
            f"tokens: {len(token_ids)} ids, "
            f"first few = {token_ids[: min(10, len(token_ids))]}"
        )
        write_transcript(
            out_dir,
            text,
            source={
                **source,
                "decoder": "RNNTGreedyDecoding",
                "blank_id": int(decoding.blank_id),
                "token_frames": [int(f) for f in token_frames],
            },
            tokens=[int(t) for t in token_ids],
        )
    elif is_ctc_variant(args.variant):
        with torch.inference_mode():
            log_probs = head(encoder_output=encoded)
        dump("ctc.log_probs", log_probs, "decoder.ctc.log_probs")

        # Pre-log_softmax logits: re-run the underlying Conv1d directly
        # to capture the input to the log_softmax.
        with torch.inference_mode():
            raw_logits = head.decoder_layers(encoded).transpose(1, 2)
        dump("ctc.logits.raw", raw_logits, "decoder.ctc.logits_raw")

        with torch.inference_mode():
            results = decoding.decode(head, encoded, encoded_len)
        text, token_ids, token_frames = results[0]
        print(f"transcript: {text!r}")
        print(
            f"tokens: {len(token_ids)} ids, "
            f"first few = {token_ids[: min(10, len(token_ids))]}"
        )
        write_transcript(
            out_dir,
            text,
            source={
                **source,
                "decoder": "CTCGreedyDecoding",
                "blank_id": int(decoding.blank_id),
                "token_frames": [int(f) for f in token_frames],
            },
            tokens=[int(t) for t in token_ids],
        )
    else:
        raise SystemExit(
            f"variant {args.variant!r} is neither ctc nor rnnt — refusing to "
            "dump decode stages without explicit handling."
        )

    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--variant", required=True,
        choices=sorted(_VARIANT_REVISIONS),
        help="GigaAM-v3 variant key (v3_e2e_rnnt, v3_e2e_ctc, v3_rnnt, "
             "v3_ctc)",
    )
    p.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    p.add_argument("--out", required=True, help="Output directory for dumps")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch.set_num_threads (0 = unchanged)")
    # Standard dumper CLI shape compatibility (validate.py always passes
    # --model and optionally --language). GigaAM loads via
    # gigaam.load_model(variant_key), so --model is accepted but the
    # value is informational only. --language is a no-op (monolingual ru).
    p.add_argument("--model", default=_HF_REPO,
                   help="HF repo id (informational; load path is "
                        "gigaam.load_model(variant_key))")
    p.add_argument("--language", default=None,
                   help="Language hint (no-op; GigaAM is monolingual ru)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="GigaAM-v3 reference dumper (salute-developers/GigaAM)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser(
        "decode",
        help="Full forward pass: frontend + encoder + (variant-specific) decode",
    )
    add_common_args(dp)
    dp.set_defaults(func=cmd_decode)

    # Alias matching the standard dumper CLI shape.
    ep = sub.add_parser(
        "encoder",
        help="Alias for decode (single-pass dumper)",
    )
    add_common_args(ep)
    ep.set_defaults(func=cmd_decode)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
