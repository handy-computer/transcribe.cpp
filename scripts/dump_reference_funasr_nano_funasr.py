#!/usr/bin/env python3
"""
dump_reference_funasr_nano_funasr.py - generate Fun-ASR-Nano reference
tensors via the FunASR toolkit.

Usage:

    uv run --project scripts/envs/funasr_nano \
      scripts/dump_reference_funasr_nano_funasr.py decode \
      --model FunAudioLLM/Fun-ASR-Nano-2512 \
      --audio samples/jfk.wav \
      --out build/validate/funasr_nano/fun-asr-nano-2512/jfk/decode/ref

Architecture (audio-LLM):
    audio (16 kHz)
      -> WavFrontend  (kaldi fbank 80 mels, 25/10 ms, hamming + LFR(7,6) + CMVN)
                              => [T_lfr, 560]
      -> SenseVoiceEncoderSmall (50 SAN-M blocks + 20 tp_blocks; identical
         to sensevoice-small, frozen)
                              => [T_lfr, 512]
      -> Audio adaptor (Transformer):
           linear1 (512 -> 2048) -> ReLU -> linear2 (2048 -> 1024)
           blocks[0..1] : 1024-dim self-attn + bottleneck FFN (1024->256->1024)
                              => [T_lfr, 1024]
      -> Qwen3-0.6B LLM:
           audio embeddings spliced into the chat-template prompt at the
           <|startofspeech|>!!<|endofspeech|> position; LLM autoregressively
           generates the response.

Dump points (one per non-trivial boundary; chosen to give Stage 4 coverage
of every place numerics can drift):

    frontend.fbank.lfr.cmvn.out          LFR-stacked + CMVN'd input features
    enc.embed.out                         after SinusoidalPositionEncoder add
    enc.encoders0.0.out                   after first SAN-M block (560 -> 512)
    enc.encoders.{0,mid,last}.out         main-tier SAN-M block samples
    enc.after_norm.out                    after the inter-tier LayerNorm
    enc.tp_encoders.{0,mid,last}.out      tp-tier SAN-M block samples
    enc.tp_norm.out                       final encoder output
    adaptor.linear1.out                   after the first 512 -> 2048 linear
    adaptor.linear2.out                   after the 2048 -> 1024 linear
    adaptor.blocks.0.out                  after the first transformer block
    adaptor.out                           final adaptor output (audio embeddings)
    dec.inputs_embeds.with_audio          LLM input embeddings with audio injected
    dec.logits_raw.prefill                LLM logits from the prompt-pass forward
    dec.logits_raw.gen8                   LLM logits from the 9th forward (gen step 8)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# FunASR 1.3.1 ships funasr/models/fun_asr_nano/model.py with broken absolute
# imports (`from ctc import CTC`, `from tools.utils import forced_align`).
# These should be relative imports. Until upstream ships a fix, we alias the
# package siblings into top-level sys.modules entries before triggering the
# import. The funasr package's `__init__.py` walks submodules and silently
# swallows ImportError, so the registration would otherwise just not happen
# and AutoModel raises "FunASRNano is not registered".
def _patch_fun_asr_nano_imports() -> None:
    os.environ.setdefault("FUNASR_DISABLE_UPDATE", "1")
    import funasr  # noqa: F401  (triggers the silent submodule walk)
    sub_ctc = importlib.import_module("funasr.models.fun_asr_nano.ctc")
    sys.modules.setdefault("ctc", sub_ctc)
    sub_tools_pkg = importlib.import_module("funasr.models.fun_asr_nano.tools")
    sub_tools_utils = importlib.import_module("funasr.models.fun_asr_nano.tools.utils")
    sys.modules.setdefault("tools", sub_tools_pkg)
    sys.modules["tools.utils"] = sub_tools_utils
    importlib.import_module("funasr.models.fun_asr_nano.model")


_patch_fun_asr_nano_imports()

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


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def model_dtype_summary(model) -> dict[str, str]:
    """Return per-top-level-child dtype summary."""
    import torch

    summary: dict[str, str] = {}
    for child_name, child in model.named_children():
        dtypes = sorted({str(p.dtype) for p in child.parameters(recurse=True)})
        summary[child_name] = ",".join(dtypes) if dtypes else "(none)"
    return summary


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


class CaptureAllHook:
    """Record every tensor a module emits during forward — used for the LLM
    lm_head where each generation step triggers a separate forward call."""

    def __init__(self) -> None:
        self.values: list = []

    def __call__(self, module, inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self.values.append(output)


def hook_module(module, hook):
    return module.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------


def load_reference(args: argparse.Namespace):
    """Build a funasr.AutoModel for FunASRNano.

    Same kwargs shape as the sensevoice dumper, plus `trust_remote_code=True`
    (FunASR's AutoModel requires it for FunASRNano even though the HF repo
    carries no remote .py — the class lives in funasr.models.fun_asr_nano).
    """
    from funasr import AutoModel

    print(
        f"Loading FunASRNano via funasr.AutoModel: model={args.model} "
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
        trust_remote_code=True,
        # Same dither workaround as sensevoice: FunASR's AutoModel silently
        # drops `dither=0.0` passed to the constructor; the WavFrontend's
        # default is 1.0 (non-deterministic). Override after construction.
        dither=0.0,
    )
    auto.kwargs["frontend"].dither = 0.0
    return auto


# ---------------------------------------------------------------------------
# Decode subcommand
# ---------------------------------------------------------------------------


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    auto = load_reference(args)

    inner = auto.model            # FunASRNano
    frontend = auto.kwargs["frontend"]
    tokenizer = auto.kwargs["tokenizer"]
    encoder = inner.audio_encoder
    adaptor = inner.audio_adaptor
    llm = inner.llm

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"audio: {audio_path.name}")

    source: dict[str, Any] = {
        "kind": "funasr-nano-funasr",
        "framework": "funasr",
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "model_dtype_per_module": model_dtype_summary(inner),
        "audio": audio_path.name,
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

    # ---- frontend (compute directly; hooks on WavFrontend.forward don't
    # fire reliably through FunASR's extract_fbank wrapper, so just call it
    # ourselves once. Cheap: kaldi fbank on ~10 s of audio is ms-level.) --
    import soundfile as sf
    pcm_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm_np.ndim > 1:
        pcm_np = pcm_np.mean(axis=1)
    if sr != frontend.fs:
        print(
            f"error: audio sr={sr} but frontend.fs={frontend.fs}; supply 16 kHz mono.",
            file=sys.stderr,
        )
        return 2
    pcm = torch.from_numpy(pcm_np)[None, :]  # [1, N]
    pcm_lens = torch.tensor([pcm_np.size], dtype=torch.int32)
    with torch.inference_mode():
        feats, feats_lens = frontend(pcm, pcm_lens)
    print(f"frontend: feats.shape={tuple(feats.shape)} feats_lens={feats_lens.tolist()}")

    # ---- install hooks on every other dump point ------------------------
    embed_h = CaptureHook(); hook_module(encoder.embed, embed_h)
    e0_h = CaptureHook();    hook_module(encoder.encoders0[0], e0_h)

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

    adap_lin1_h   = CaptureHook(); hook_module(adaptor.linear1, adap_lin1_h)
    adap_lin2_h   = CaptureHook(); hook_module(adaptor.linear2, adap_lin2_h)
    adap_block0_h = CaptureHook(); hook_module(adaptor.blocks[0], adap_block0_h)
    adap_out_h    = CaptureHook(); hook_module(adaptor, adap_out_h)

    # LLM lm_head fires once per forward; capture all of them so we can
    # extract prefill (call 0) and gen step 8 (call 9 with KV cache).
    lm_head_h = CaptureAllHook(); hook_module(llm.lm_head, lm_head_h)

    # ---- run the full reference inference --------------------------------
    print("running auto.generate(...) with hooks active ...")
    # Note on dtype: the checkpoint is mixed (encoder/adaptor F32, LLM BF16).
    # Passing bf16=True to FunASR's inference_prepare casts the SPEECH tensor
    # to BF16 *before* the encoder, which crashes because the encoder weights
    # are F32 (RuntimeError: mat1 and mat2 must have the same dtype, but got
    # BFloat16 and Float). Running with default kwargs makes inference_llm
    # also cast the LLM to F32 — this is lossless (BF16 -> F32) and gives a
    # well-defined F32 oracle. Stage 4 / Stage 5 will revisit if a BF16-
    # faithful reference is needed for downstream tolerances.
    result = auto.generate(
        input=str(audio_path),
        language=args.language if args.language else None,
        itn=bool(args.use_itn),
        max_length=args.max_length,
        device=args.device,
    )

    # Reference produced a single result dict in a list.
    transcript_text = result[0]["text"]
    print(f"transcript: {transcript_text!r}")

    # ---- dump captured tensors -------------------------------------------
    # Frontend was computed directly above; the rest came from forward hooks.
    diag = {
        "enc.embed":     embed_h.value is not None,
        "enc.encoders0": e0_h.value is not None,
        "enc.after_norm":after_norm_h.value is not None,
        "enc.tp_norm":   tp_norm_h.value is not None,
        "adaptor.l1":    adap_lin1_h.value is not None,
        "adaptor.l2":    adap_lin2_h.value is not None,
        "adaptor.b0":    adap_block0_h.value is not None,
        "adaptor.out":   adap_out_h.value is not None,
    }
    print(f"hook fire summary: {diag}")

    def maybe_dump(name: str, t, stage: str) -> None:
        if t is None:
            print(f"  SKIP {name} (hook never fired)", file=sys.stderr)
            return
        dump(name, t, stage)

    dump("frontend.fbank.lfr.cmvn.out", feats, "frontend.lfr.cmvn")
    maybe_dump("enc.embed.out",                embed_h.value,"encoder.embed.pos_added")
    maybe_dump("enc.encoders0.0.out",          e0_h.value,   "encoder.encoders0.0")
    for i, h in main_hooks.items():
        maybe_dump(f"enc.encoders.{i}.out", h.value, f"encoder.encoders.{i}")
    maybe_dump("enc.after_norm.out",  after_norm_h.value, "encoder.after_norm")
    for i, h in tp_hooks.items():
        maybe_dump(f"enc.tp_encoders.{i}.out", h.value, f"encoder.tp_encoders.{i}")
    maybe_dump("enc.tp_norm.out",     tp_norm_h.value,  "encoder.tp_norm")

    maybe_dump("adaptor.linear1.out",  adap_lin1_h.value,   "adaptor.linear1")
    maybe_dump("adaptor.linear2.out",  adap_lin2_h.value,   "adaptor.linear2")
    maybe_dump("adaptor.blocks.0.out", adap_block0_h.value, "adaptor.blocks.0")
    maybe_dump("adaptor.out",          adap_out_h.value,    "adaptor.out")

    # LLM logits — collect prefill (call 0) and gen step 8 (call 9).
    n_calls = len(lm_head_h.values)
    print(f"llm.lm_head fired {n_calls} time(s)")
    if n_calls < 1:
        print("ERROR: lm_head hook never fired during auto.generate", file=sys.stderr)
        return 2

    # lm_head call 0 covers the entire prompt+audio prefix; logits shape is
    # [1, T_prefix, vocab]. The "prefill" is conventionally the last position
    # of that pass (the slot that will produce the first generated token).
    prefill_full = lm_head_h.values[0]
    if prefill_full.dim() == 3:
        prefill_last = prefill_full[:, -1:, :]
    else:
        prefill_last = prefill_full
    dump("dec.logits_raw.prefill", prefill_last, "decoder.logits.prefill")

    # gen8 = output of the 9th forward call (index 8 if 0-indexed).
    if n_calls > 8:
        gen8 = lm_head_h.values[8]
        # With KV cache, single-step calls return [1, 1, vocab].
        if gen8.dim() == 3 and gen8.shape[1] == 1:
            gen8 = gen8[:, 0:1, :]
        dump("dec.logits_raw.gen8", gen8, "decoder.logits.gen8")
    else:
        # Generation finished before gen8; warn but don't fail. The Stage 4
        # mid-gen requirement is "at least one gen<N>, N>=8". For very short
        # utterances we can drop down to the last available step and rename.
        last_idx = n_calls - 1
        if last_idx >= 1:
            print(f"WARN: only {n_calls} forward calls — using gen{last_idx} "
                  f"in place of gen8", file=sys.stderr)
            late = lm_head_h.values[last_idx]
            if late.dim() == 3 and late.shape[1] == 1:
                late = late[:, 0:1, :]
            dump(f"dec.logits_raw.gen{last_idx}", late, f"decoder.logits.gen{last_idx}")
        else:
            print("WARN: no single-step LLM forwards captured; skipping mid-gen dump",
                  file=sys.stderr)

    # ---- transcript ------------------------------------------------------
    # The LLM-generated tokens — re-encode the response text via the LLM
    # tokenizer so the transcript sidecar carries token ids the C++ validator
    # can compare. The reference dict only exposes `text`.
    tokens = tokenizer.encode(transcript_text, add_special_tokens=False)
    transcript_source = {
        **source,
        "normalized_text": normalize_text(transcript_text),
        "label": result[0].get("label"),
    }
    write_transcript(
        out_dir,
        transcript_text,
        source=transcript_source,
        tokens=[int(t) for t in tokens],
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF repo id (e.g. FunAudioLLM/Fun-ASR-Nano-2512)")
    p.add_argument("--revision", default="a7088d620f755dcdca575b63db184c3ad55b2865",
                   help="HF revision (branch/tag/commit SHA) to pin the download to")
    p.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    p.add_argument("--out",   required=True, help="Output directory for dumps")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--torch-threads", type=int, default=8,
                   help="torch.set_num_threads. Higher than sensevoice's 1 because the "
                        "Qwen3-0.6B LLM does ~24 F32 forward passes during generation; "
                        "single-threaded CPU is minutes-per-utterance. Reduction-order "
                        "non-determinism at >1 thread is sub-tolerance for our budgets.")
    p.add_argument("--language", default="en",
                   help="Language hint passed through to FunASRNano.get_prompt "
                        "(e.g. 'en', 'zh', 'ja', '中文'). Default: 'en'.")
    p.add_argument("--use-itn", action="store_true",
                   help="Enable inverse text normalization in the prompt")
    p.add_argument("--max-length", type=int, default=128,
                   help="Max new tokens; must be > 8 to dump gen8 logits")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fun-ASR-Nano reference dumper (FunASR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("decode", help="Run frontend + encoder + adaptor + LLM; dump all stages and transcript")
    add_common_args(dp)
    dp.set_defaults(func=cmd_decode)

    # Single-pass dumper: encoder is an alias for decode (mirrors qwen3_asr / sensevoice).
    ep = sub.add_parser("encoder", help="Alias for decode (full dump pass)")
    add_common_args(ep)
    ep.set_defaults(func=cmd_decode)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
