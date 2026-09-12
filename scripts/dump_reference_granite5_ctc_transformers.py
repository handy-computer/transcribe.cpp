#!/usr/bin/env python3
"""
dump_reference_granite5_ctc_transformers.py — reference dumps for
ibm-granite/granite-speech-5.0-470m-turboctc (Granite Speech 5.0 TurboCTC).

Mirrors the canonical inference path from the model card verbatim:

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")
    inputs = processor(speech, sampling_rate=16000)
    outputs = model.generate(**inputs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

See: https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc

Reference framework pin: mainline `transformers >= 5.16`. NO
trust_remote_code — config.json carries no auto_map, so AutoModelForCTC
resolves to transformers.models.granite_speech5. The repo also ships
`configuration_ctc_conformer.py` / `modeling_ctc_conformer.py` /
`granite_encoder.py`, which are a STALE alternate packaging using the
granite-speech-4.x tensor names (to_q/to_kv, ff1, depth_conv, batch_norm);
they cannot load the shipped safetensors and are not the reference.

Architecture (transformers/models/granite_speech5/modular_granite_speech5.py):

  GraniteSpeech5FeatureExtractor
    -> torchaudio MelSpectrogram(sr=16000, n_fft=512, win_length=400,
       hop=160, n_mels=80) at library defaults (power=2.0, Hann periodic,
       center=True, reflect pad, htk mel scale, norm=None)
       -> clamp_min(1e-10) -> log10
       -> per-utterance floor at (global max over time AND mel) - 8.0
       -> x/4 + 1
       -> concat compute_deltas(win_length=3)  [80 -> 160]
       -> stack 2 consecutive frames           [160 -> 320]
       Framing: mel_frames = n_samples // 160; num_frames = 2*ceil(mel_frames/2);
       the WAVEFORM is right-padded to (num_frames-1)*160+1 samples, then the
       melspec is sliced to [..., :num_frames].

  GraniteSpeech5Encoder
    -> input_linear: Linear(320 -> 1024), bias
    -> 16 x conformer block (macaron: 0.5*ff1, attn, conv, 0.5*ff2, norm_out)
         blocks 0 and 1 are GraniteSpeech5EncoderSubsamplingBlock: the conv
         module's depthwise conv runs at stride 2 and the residual is
         mean-pooled over frame pairs, then the conv output is TRIMMED to the
         pooled length. Total time downsample 8x (2x stacking + 4x here),
         100 Hz mel -> 12.5 Hz output.
         attn: block-local over context_size=128 with Shaw relative positions
         (per-layer rel_pos_emb [1025, 128], index clamp(i-j, +-128)+512).
         q/k/v bias-free, o_proj biased; both FFN linears biased.
         conv norm is BatchNorm1d, kept in F32 by
         _keep_in_fp32_modules_strict=["conv.norm"].
    -> after block index 7 (num_hidden_layers//2 - 1 + 1): self-conditioned CTC
         hidden += out_mid(softmax(out(hidden)))

  GraniteSpeech5ForCTC
    -> ctc_head: Linear(1024 -> 16384), TIED to encoder.out
       (tie_word_embeddings=true; the checkpoint has no ctc_head.* tensors)

CTC blank id = 0 (`<|blank|>`), which is also pad_token_id. The model id
space is 1:1 with the tokenizer — no offset, no special-token block.

Dump points:
  mel.in                      input_features                [T_in, 320]
  enc.input_linear.out        encoder.input_linear output   [T_in, 1024]
  enc.block.{i}.out           encoder.layers[i] output      [T_i, 1024]
  enc.block.{0,1}.post_ff1    hidden after the ff1 residual (pre-subsample rate)
  enc.block.{0,1}.post_attn   hidden after the attn residual (pre-subsample rate)
  enc.block.{0,1}.post_conv   hidden after pooled-residual + trimmed conv
                              (POST-subsample rate — the key subsampling observable)
  enc.block.{0,1}.post_ff2    hidden after the ff2 residual (pre norm_out)
  enc.block.2.post_{...}      same four points on the first NON-subsampling block
  enc.ctc.mid_logits          encoder.out output   (self-conditioning) [T_enc, 16384]
  enc.ctc.mid_injection       encoder.out_mid output                   [T_enc, 1024]
  enc.out                     encoder last_hidden_state                [T_enc, 1024]
  enc.ctc_logits              ctc_head output                          [T_enc, 16384]

Usage:

    uv run --project scripts/envs/granite5_ctc \\
      scripts/dump_reference_granite5_ctc_transformers.py decode \\
      --model ibm-granite/granite-speech-5.0-470m-turboctc \\
      --audio samples/jfk.wav \\
      --out build/validate/granite5_ctc/granite-speech-5.0-470m-turboctc/jfk/decode/ref
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
    """Load processor + model exactly as the model card does."""
    import torch
    import transformers
    from transformers import AutoModelForCTC, AutoProcessor

    model_id, local_only = resolve_model(args.model)
    revision = None if local_only else args.revision
    source = "local path" if local_only else "HuggingFace"
    rev_text = f", revision={revision}" if revision else ""
    print(
        f"loading {model_id} ({source}, transformers {transformers.__version__}"
        f", device={args.device}{rev_text}, dtype={args.dtype})..."
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

    Used to dump the running residual inside a conformer block by
    pre-hooking the NEXT submodule in block.forward.
    """

    def __init__(self) -> None:
        self.value = None

    def __call__(self, module, inputs) -> None:
        if self.value is not None or not inputs:
            return
        self.value = inputs[0]


# Blocks whose four macaron sub-steps get dumped. 0 and 1 are the
# subsampling blocks (the structurally novel part of this family); 2 is
# the first plain block, for a same-shape control at the 12.5 Hz rate.
SUBSTEP_BLOCKS = (0, 1, 2)


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
    # Self-conditioning path. `encoder.out` is invoked exactly once inside
    # GraniteSpeech5Encoder.forward (the mid-layer CTC posterior); the final
    # CTC projection is the separate — but weight-tied — model.ctc_head.
    fwd("enc.ctc.mid_logits", enc.out)
    fwd("enc.ctc.mid_injection", enc.out_mid)
    fwd("enc.ctc_logits", model.ctc_head)

    # Sub-step dumps inside the blocks that matter. The block applies
    # ff1 / attn / conv / ff2 / norm_out in sequence against named norms, so
    # pre-hooking each norm captures the running residual at that point:
    #   pre-hook(norm_self_att)      -> hidden after the ff1 residual
    #   pre-hook(norm_conv)          -> hidden after the attn residual
    #   pre-hook(norm_feed_forward2) -> hidden after the conv residual
    #   pre-hook(norm_out)           -> hidden after the ff2 residual
    # On a subsampling block the same submodule order holds, so post_conv
    # lands AFTER the mean-pooled residual + trimmed conv output — exactly
    # the observable that localizes a subsampling bug.
    for b in SUBSTEP_BLOCKS:
        if b >= len(enc.layers):
            continue
        blk = enc.layers[b]
        for name, target in (
            (f"enc.block.{b}.post_ff1", blk.norm_self_att),
            (f"enc.block.{b}.post_attn", blk.norm_conv),
            (f"enc.block.{b}.post_conv", blk.norm_feed_forward2),
            (f"enc.block.{b}.post_ff2", blk.norm_out),
        ):
            h = PreHookInput()
            target.register_forward_pre_hook(h)
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
        raise SystemExit(f"granite5_ctc expects 16 kHz; got {sr} Hz in {audio_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cfg = model.config.encoder_config
    source = {
        "kind": "granite5_ctc-transformers",
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
        "context_size": int(enc_cfg.context_size),
        "subsample_layers": list(enc_cfg.subsample_layers),
        "notes": (
            "conv.norm (BatchNorm1d) is kept in F32 by "
            "_keep_in_fp32_modules_strict, so the reference forward is not "
            "pure BF16 at the conv norm even when --dtype bf16."
        ),
    }

    # Model-card path. The processor runs the mel front-end on `device`.
    inputs = processor(pcm, sampling_rate=sr, device=args.device)
    inputs = inputs.to(args.device)
    input_features = inputs["input_features"]
    print(f"  input_features: {tuple(input_features.shape)}  "
          f"range [{input_features.min().item():.4f}, "
          f"{input_features.max().item():.4f}]")
    if "attention_mask" in inputs:
        am = inputs["attention_mask"]
        print(f"  attention_mask: {tuple(am.shape)}  valid={int(am.sum().item())}")

    write_tensor("mel.in", to_np(input_features), "encoder", source, out_dir=out_dir)

    n_enc_layers = len(model.encoder.layers)
    block_idx = sorted({0, 1, 2,
                        n_enc_layers // 2 - 1, n_enc_layers // 2,
                        n_enc_layers - 1, *args.enc_blocks})
    block_idx = [b for b in block_idx if 0 <= b < n_enc_layers]
    hooks = register_hooks(model, block_idx)

    # generate() is a single forward + argmax + CTC collapse; there is no
    # autoregressive loop. Padded positions are set back to pad_token_id.
    with torch.inference_mode():
        token_ids = model.generate(**inputs)

    # The encoder's last_hidden_state is the ctc_head's input; capture it from
    # the ctc_head pre-hook rather than re-running the encoder.
    transcriptions = processor.batch_decode(token_ids, skip_special_tokens=True)
    text_pred = (transcriptions[0] if transcriptions else "").strip()
    print(f"  transcript : {text_pred!r}")

    fixed_names = [
        "enc.input_linear.out",
        "enc.ctc.mid_logits",
        "enc.ctc.mid_injection",
        "enc.ctc_logits",
    ]
    block_names = [f"enc.block.{i}.out" for i in block_idx]
    substep_names = [
        f"enc.block.{b}.{s}"
        for b in SUBSTEP_BLOCKS
        for s in ("post_ff1", "post_attn", "post_conv", "post_ff2")
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

    # enc.out == the encoder's last_hidden_state == the final block's output.
    # Alias it under a stable name so Stage 4 can compare the encoder boundary
    # without knowing the layer count.
    last_block = hooks.get(f"enc.block.{n_enc_layers - 1}.out")
    if last_block is not None and last_block.value is not None:
        a = to_np(last_block.value)
        write_tensor("enc.out", a, "encoder", source, out_dir=out_dir)
        print(f"  enc.out: shape={a.shape} "
              f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")

    # token_ids are per-frame argmax with padded frames set to pad_token_id.
    # Collapse them the way processor.batch_decode does so transcript.json
    # carries the post-collapse ids, matching the medasr/parakeet convention.
    flat = token_ids[0] if token_ids.dim() > 1 else token_ids
    raw_ids = [int(t) for t in flat.tolist()]
    collapsed: list[int] = []
    prev = -1
    for tid in raw_ids:
        if tid != prev and tid != model.config.pad_token_id:
            collapsed.append(tid)
        prev = tid
    write_transcript(out_dir, text_pred, source=source, tokens=collapsed)
    print(f"  frames={len(raw_ids)}  collapsed_tokens={len(collapsed)}")
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    # Encoder-only stage is a no-op: decode already writes every tensor the
    # dump_coverage.json catalog expects (this family is encoder-only — the
    # "decoder" is a single tied Linear plus an argmax). Keep the stage
    # directory present so validate.py's enumerator finds it.
    Path(args.out).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, help="HF repo id or local path")
        sp.add_argument("--revision", default=None,
                        help="HF revision (commit hash) to pin against drift")
        sp.add_argument("--audio", required=True, help="path to 16 kHz mono audio")
        sp.add_argument("--out", required=True, help="output directory for dumps")
        sp.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
        sp.add_argument("--dtype", default="f32", choices=["bf16", "f16", "f32"],
                        help="Reference dtype. F32 is the porting reference: the BF16->F32 "
                             "upcast is lossless, so these are the shipped weights, and "
                             "F32 activations on BF16 weights is exactly the transcribe.cpp "
                             "compute regime. Running the oracle at bf16 (what dtype='auto' "
                             "would load) injects 2-3%% of p99_abs of reference-only noise, "
                             "~250x the 1e-4*p99_abs Stage 2 tolerance budget, which would "
                             "force blind tolerance widening at Stage 4. bf16/f16 are for "
                             "ad-hoc precision probes only; the transcript is unchanged.")
        sp.add_argument("--attn-impl", default="eager",
                        choices=["eager", "sdpa", "flex_attention"],
                        help="GraniteSpeech5 does NOT support flash-attn "
                             "(_supports_flash_attn=False: the Shaw bias is a float mask).")
        sp.add_argument("--torch-threads", type=int, default=4)
        sp.add_argument("--language", default=None,
                        help="ignored (granite5_ctc is monolingual English)")

    encoder = sub.add_parser("encoder", help="no-op (decode dumps all tensors)")
    add_common_args(encoder)
    encoder.set_defaults(func=cmd_encoder)

    decode = sub.add_parser("decode", help="full encoder + CTC head dump + transcript")
    add_common_args(decode)
    decode.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                        help="extra encoder block indices to dump "
                             "(default: 0, 1, 2, mid-1, mid, last)")
    decode.set_defaults(func=cmd_decode)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
