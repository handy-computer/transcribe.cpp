#!/usr/bin/env python3
"""
dump_reference_sensevoice_funasr.py - generate SenseVoiceSmall reference
tensors via the FunASR toolkit (the only published path: there is no HF
Transformers shim, no NeMo / ESPnet equivalent, and the model card's
canonical inference example uses `funasr.AutoModel`).

Usage:

    uv run --project scripts/envs/sensevoice \
      scripts/dump_reference_sensevoice_funasr.py decode \
      --model FunAudioLLM/SenseVoiceSmall \
      --audio samples/jfk.wav \
      --out build/validate/sensevoice/sensevoice-small/jfk/decode/ref

Writes:
    <name>.f32       raw little-endian float32, row-major (batch dim squeezed)
    <name>.json      per-tensor sidecar via shared scripts.lib.ref_dump
    transcript.json  behavioral artifact (greedy-CTC text + token ids)

Architecture summary (SenseVoiceSmall = encoder-CTC):
    audio (16 kHz)
      -> WavFrontend: kaldi fbank (80 mels, 25 ms / 10 ms, hamming)
                      + LFR stack (m=7, n=6) + per-feature CMVN
                      => [T_lfr, 560]
      -> 4-token prefix prepended in mel-feature space:
           [language, event, emotion, textnorm] (each 560-d Embedding)
                      => [4 + T_lfr, 560]
      -> SenseVoiceEncoderSmall:
           x *= sqrt(d_model); add SinusoidalPositionEncoder
           encoders0[0]   (560 -> 512 projection SAN-M block)
           encoders[0..48]              (49 SAN-M blocks)
           after_norm
           tp_encoders[0..19]           (20 SAN-M blocks)
           tp_norm
                      => [4 + T_lfr, 512]
      -> CTC head (Linear ctc_lo, vocab=25055) -> log-softmax

Dump points (one per non-trivial boundary; chosen so Stage 4 has
coverage of every place numerics can drift):

    frontend.fbank.lfr.cmvn.out   LFR-stacked + CMVN'd input features
    enc.prefix.lid_emb            language prefix embedding
    enc.prefix.event_emo_emb      event/emotion prefix embeddings (2 tokens)
    enc.prefix.textnorm_emb       textnorm prefix embedding
    enc.input.with_prefix         features after the 4-token prefix prepend
    enc.embed.out                 after SinusoidalPositionEncoder add
    enc.encoders0.0.out           after first SAN-M block (560 -> 512)
    enc.encoders.{0,mid,last}.out main-tier SAN-M block samples
    enc.after_norm.out            after the inter-tier LayerNorm
    enc.tp_encoders.{0,mid,last}.out  tp-tier SAN-M block samples
    enc.tp_norm.out               final encoder output
    ctc.logits.raw                pre-softmax CTC logits
    ctc.log_probs                 log-softmax CTC distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.ref_dump import write_tensor, write_transcript


# ---------------------------------------------------------------------------
# Helpers
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
    """Convert a torch.Tensor to a row-major float32 numpy array with the
    leading batch dim squeezed (matches the C++ validation contract)."""
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


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    return {
        torch.bfloat16: "bf16",
        torch.float16:  "f16",
        torch.float32:  "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


# ---------------------------------------------------------------------------
# Hook utilities
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


def hook_module(module, hook: CaptureHook):
    return module.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------


def load_reference(args: argparse.Namespace):
    """Build a funasr.AutoModel for SenseVoiceSmall.

    AutoModel is the model card's published entry path. It downloads via
    HF Hub (`hub="hf"`) and instantiates the SenseVoiceSmall class from
    funasr's class registry. Pinning to a specific HF revision is done
    by passing `model_revision`. We force `dither=0.0` so the kaldi
    fbank frontend is deterministic (its default is 1.0, which adds
    Gaussian noise to the waveform every call).
    """
    import torch
    from funasr import AutoModel

    print(
        f"Loading SenseVoiceSmall via funasr.AutoModel: model={args.model} "
        f"revision={args.revision} device={args.device}"
    )

    auto = AutoModel(
        model=args.model,
        model_revision=args.revision,
        hub="hf",
        device=args.device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        # Force deterministic frontend.
        dither=0.0,
    )
    return auto


# ---------------------------------------------------------------------------
# Decode subcommand
# ---------------------------------------------------------------------------


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    auto = load_reference(args)

    inner = auto.model              # SenseVoiceSmall
    frontend = auto.kwargs["frontend"]  # WavFrontend
    tokenizer = auto.kwargs["tokenizer"]  # SentencepiecesTokenizer
    encoder = inner.encoder

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm, sr = load_audio(audio_path)
    if sr != frontend.fs:
        print(
            f"error: audio sr={sr} but frontend.fs={frontend.fs}; "
            "supply 16 kHz mono audio.",
            file=sys.stderr,
        )
        return 2
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    device = torch.device(args.device)
    waveform = torch.from_numpy(pcm).to(device).unsqueeze(0)  # [1, N]
    wav_lens = torch.tensor([pcm.size], dtype=torch.int64, device=device)

    # ---- frontend ---------------------------------------------------------
    with torch.inference_mode():
        feats, feats_lens = frontend(waveform, wav_lens)
    # feats: [1, T_lfr, 560]; feats_lens: [1]
    print(f"frontend: feats.shape={tuple(feats.shape)} feats_lens={feats_lens.tolist()}")

    source = {
        "kind": "sensevoice-funasr",
        "framework": "funasr",
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "model_dtype": model_dtype_name(inner),
        "audio": audio_path.name,
        "n_samples": int(pcm.size),
        "sample_rate": int(sr),
        "language": args.language,
        "use_itn": bool(args.use_itn),
        "lfr_m": frontend.lfr_m,
        "lfr_n": frontend.lfr_n,
        "n_mels": frontend.n_mels,
    }

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} "
            f"min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}"
        )
        write_tensor(name, a, stage=stage, source=source, out_dir=out_dir)

    dump("frontend.fbank.lfr.cmvn.out", feats, "frontend.lfr.cmvn")

    # ---- prefix embeddings (mirror SenseVoiceSmall.inference) ------------
    # Indices come from inner.lid_dict / textnorm_dict; event/emo prefix
    # uses the literal indices [1, 2] in the parent embedding table.
    lang = args.language
    lid_idx = inner.lid_dict.get(lang, 0)
    textnorm_key = "withitn" if args.use_itn else "woitn"
    textnorm_idx = inner.textnorm_dict[textnorm_key]

    with torch.inference_mode():
        lid_emb = inner.embed(torch.tensor([[lid_idx]], device=device))            # [1,1,560]
        event_emo_emb = inner.embed(torch.tensor([[1, 2]], device=device))         # [1,2,560]
        textnorm_emb = inner.embed(torch.tensor([[textnorm_idx]], device=device))  # [1,1,560]

    dump("enc.prefix.lid_emb",       lid_emb,       "encoder.prefix.lid")
    dump("enc.prefix.event_emo_emb", event_emo_emb, "encoder.prefix.event_emo")
    dump("enc.prefix.textnorm_emb",  textnorm_emb,  "encoder.prefix.textnorm")

    # Concat order matches inference(): textnorm appended to speech first,
    # then [lid, event, emo] prepended.
    speech = torch.cat((textnorm_emb, feats), dim=1)
    speech = torch.cat((lid_emb, event_emo_emb, speech), dim=1)
    speech_lengths = feats_lens + 4

    dump("enc.input.with_prefix", speech, "encoder.input.with_prefix")

    # ---- encoder hooks ---------------------------------------------------
    # encoder.embed is the SinusoidalPositionEncoder; capture its output to
    # see the post-PE features. Then sample encoders0[0], encoders[0|mid|last],
    # after_norm, tp_encoders[0|mid|last], tp_norm.
    embed_h = CaptureHook(); hook_module(encoder.embed, embed_h)

    e0_h = CaptureHook(); hook_module(encoder.encoders0[0], e0_h)

    enc_main = encoder.encoders
    main_idxs = sorted({0, len(enc_main) // 2, len(enc_main) - 1})
    main_hooks: dict[int, CaptureHook] = {}
    for i in main_idxs:
        h = CaptureHook(); hook_module(enc_main[i], h); main_hooks[i] = h

    after_norm_h = CaptureHook(); hook_module(encoder.after_norm, after_norm_h)

    enc_tp = encoder.tp_encoders
    tp_idxs = sorted({0, len(enc_tp) // 2, len(enc_tp) - 1})
    tp_hooks: dict[int, CaptureHook] = {}
    for i in tp_idxs:
        h = CaptureHook(); hook_module(enc_tp[i], h); tp_hooks[i] = h

    tp_norm_h = CaptureHook(); hook_module(encoder.tp_norm, tp_norm_h)

    # CTC head: ctc_lo Linear projects encoder output to vocab logits.
    ctc_logits_h = CaptureHook(); hook_module(inner.ctc.ctc_lo, ctc_logits_h)

    # ---- run encoder + ctc -----------------------------------------------
    with torch.inference_mode():
        encoder_out, encoder_out_lens = encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        ctc_log_probs = inner.ctc.log_softmax(encoder_out)

    dump("enc.embed.out",       embed_h.value, "encoder.embed.pos_added")
    dump("enc.encoders0.0.out", e0_h.value,    "encoder.encoders0.0")
    for i, h in main_hooks.items():
        dump(f"enc.encoders.{i}.out", h.value, f"encoder.encoders.{i}")
    dump("enc.after_norm.out", after_norm_h.value, "encoder.after_norm")
    for i, h in tp_hooks.items():
        dump(f"enc.tp_encoders.{i}.out", h.value, f"encoder.tp_encoders.{i}")
    dump("enc.tp_norm.out",  tp_norm_h.value,  "encoder.tp_norm")
    dump("ctc.logits.raw",   ctc_logits_h.value, "ctc.logits.raw")
    dump("ctc.log_probs",    ctc_log_probs,    "ctc.log_probs")

    # ---- greedy CTC decode -> transcript --------------------------------
    # Mirrors SenseVoiceSmall.inference(): per-batch argmax over the time
    # axis, collapse repeats, drop blanks, then SP decode.
    with torch.inference_mode():
        x = ctc_log_probs[0, : encoder_out_lens[0].item(), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        mask = yseq != inner.blank_id
        token_int = yseq[mask].tolist()
        text = tokenizer.decode(token_int)

    print(f"transcript: {text!r}")
    print(f"tokens (post-collapse, post-blank-strip): {len(token_int)}")

    transcript_source = {
        **source,
        "normalized_text": normalize_text(text),
        "blank_id": int(inner.blank_id),
    }
    write_transcript(
        out_dir,
        text,
        source=transcript_source,
        tokens=[int(t) for t in token_int],
    )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF repo id (default: FunAudioLLM/SenseVoiceSmall)")
    p.add_argument("--revision", default="3eb3b4eeffc2f2dde6051b853983753db33e35c3",
                   help="HF revision (branch/tag/commit SHA) to pin the download to")
    p.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    p.add_argument("--out",   required=True, help="Output directory for dumps")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch.set_num_threads (0 = unchanged)")
    p.add_argument("--language", default="auto",
                   help="Language hint for the prefix embedding "
                        "(auto/zh/en/yue/ja/ko/nospeech). Default: auto.")
    p.add_argument("--use-itn", action="store_true",
                   help="Set the textnorm prefix to 'withitn' instead of 'woitn'")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SenseVoiceSmall reference dumper (FunASR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("decode", help="Run frontend + encoder + CTC; dump all stages and transcript")
    add_common_args(dp)
    dp.set_defaults(func=cmd_decode)

    # Single-pass dumper: encoder is an alias for decode (mirrors qwen3_asr).
    ep = sub.add_parser("encoder", help="Alias for decode (full dump pass)")
    add_common_args(ep)
    ep.set_defaults(func=cmd_decode)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
