#!/usr/bin/env python3
"""
dump_reference_parakeet_nemo.py - generate Parakeet reference tensors
from the NVIDIA NeMo implementation (canonical reference).

Covers every Parakeet variant in the family: TDT, RNN-T, CTC, and
hybrid TDT+CTC (which dumps the TDT path only — per intake, hybrid
variants ship the TDT head at runtime; pure CTC is covered by the
parakeet-ctc-* variants).

Run through the repo-local Parakeet reference environment:

    uv run --project scripts/envs/parakeet \
      scripts/dump_reference_parakeet_nemo.py encoder \
      --model nvidia/parakeet-tdt-0.6b-v2 \
      --audio samples/jfk.wav \
      --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/encoder/ref

    uv run --project scripts/envs/parakeet \
      scripts/dump_reference_parakeet_nemo.py decode \
      --model nvidia/parakeet-tdt-0.6b-v2 \
      --audio samples/jfk.wav \
      --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/decode/ref

The --model flag accepts either an HF model name (e.g.
"nvidia/parakeet-tdt-0.6b-v2") or a local path to a .nemo checkpoint.
NeMo downloads and caches the model on first use.

Tensor output uses the shared reference dump contract via
scripts.lib.ref_dump (write_tensor records rms / p99_abs needed for
Stage 6 magnitude-aware tolerances; write_transcript records text).

    <name>.f32    raw little-endian float32, row-major
    <name>.json   sidecar metadata (shape, dtype, rms, p99_abs, source)

The decode command also writes transcript.json as a behavioral artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Load scripts/lib/ref_dump.py by absolute path. We can't do
# `from scripts.lib.ref_dump import ...` because nemo_toolkit ships a
# top-level `scripts/` package that shadows the repo's scripts/ on
# sys.path. Loading by file path bypasses the shadow.
import importlib.util as _importlib_util

_REF_DUMP_PATH = Path(__file__).resolve().parent / "lib" / "ref_dump.py"
_spec = _importlib_util.spec_from_file_location("transcribe_ref_dump", _REF_DUMP_PATH)
_ref_dump = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_ref_dump)
write_tensor = _ref_dump.write_tensor
write_transcript = _ref_dump.write_transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


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
    """Convert a torch Tensor to contiguous fp32 numpy, squeezing
    leading size-1 dims to match C++ dump conventions."""
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


# ---------------------------------------------------------------------------
# NeMo model loading + arch detection
# ---------------------------------------------------------------------------

def _patch_conformer_for_offline(force_regular_att_style: bool = False) -> None:
    """Strip streaming-only ConformerEncoder kwargs before NeMo 2.7.2 instantiates.

    The unified-en variant carries Dynamic-Chunked-Convolution and chunked
    attention config keys that newer NeMo recognises but 2.7.2 rejects.
    Dropping those rejected-but-inert kwargs is always safe.

    `att_context_style` is the one kwarg whose override is NOT always safe.
    Cache-aware streaming models (nemotron-speech-streaming-en-0.6b and
    similar) are TRAINED with `att_context_style='chunked_limited'`; forcing
    'regular' substitutes a different attention mask shape at inference,
    which is a numerical-correctness issue. parakeet-unified-en-0.6b is the
    one ported variant where forcing 'regular' is the right choice (its v1
    C++ port explicitly targets offline / full-context mode and was
    validated against regular-style reference tensors). Default is to
    preserve the model's native style; pass `force_regular_att_style=True`
    only when the caller is parakeet-unified-en-0.6b or another variant
    whose port deliberately targets the offline-substitute mask.

    The cfg also sets several boolean/scalar fields to None that
    explicitly-pass-None breaks (`use_bias=None` -> `int * None` fail in
    `d_ff = d_model * ff_expansion_factor`). Filter those out so the
    constructor's named defaults take effect.
    """
    import inspect
    import nemo.collections.asr.modules.conformer_encoder as ce

    if getattr(ce.ConformerEncoder, "_transcribe_offline_patched", False):
        return

    original_init = ce.ConformerEncoder.__init__
    accepted = set(inspect.signature(original_init).parameters.keys())
    streaming_only = ("att_chunk_context_size", "streaming_loss_weight",
                      "att_chunk_size")
    offline_overrides: dict[str, Any] = {
        "conv_context_style": None,
    }
    if force_regular_att_style:
        offline_overrides["att_context_style"] = "regular"

    def patched_init(self, *args, **kwargs):
        dropped = []
        # Drop unknown kwargs (covers any future streaming-only key the
        # constructor doesn't accept yet).
        for k in list(kwargs.keys()):
            if k not in accepted:
                kwargs.pop(k)
                dropped.append(f"unknown:{k}")
        for k in streaming_only:
            if k in kwargs:
                kwargs.pop(k)
                dropped.append(k)
        for k, fallback in offline_overrides.items():
            if k in kwargs and kwargs[k] not in (None, "regular"):
                dropped.append(f"{k}={kwargs[k]!r}->{fallback!r}")
                if fallback is None:
                    kwargs.pop(k)
                else:
                    kwargs[k] = fallback
        # Filter None values for any kwarg whose default is not None and
        # whose annotation is bool. Passing None for a bool kwarg
        # propagates as a falsy value into the constructor body which
        # then fails on math like `int * None`.
        for name, param in inspect.signature(original_init).parameters.items():
            if name == "self":
                continue
            if name in kwargs and kwargs[name] is None and param.default is not None:
                kwargs.pop(name)
                dropped.append(f"None->default:{name}")
        if dropped:
            print(f"[offline-patch] dropped/overrode encoder kwargs: {dropped}")
        return original_init(self, *args, **kwargs)

    ce.ConformerEncoder.__init__ = patched_init
    ce.ConformerEncoder._transcribe_offline_patched = True


def load_model(args: argparse.Namespace):
    """Load a Parakeet model via NeMo. Architecture is inferred at runtime."""
    _patch_conformer_for_offline(force_regular_att_style=bool(getattr(args, "offline_only", False)))

    from nemo.collections.asr.models import ASRModel

    model_id = args.model
    local = Path(model_id).expanduser()

    if local.exists():
        print(f"Loading Parakeet from local path: {local}")
        try:
            model = ASRModel.restore_from(str(local), map_location="cpu")
        except TypeError as e:
            if "abstract class" not in str(e):
                raise
            from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
            from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
            last = e
            for cls in (EncDecRNNTBPEModel, EncDecCTCModelBPE):
                try:
                    model = cls.restore_from(str(local), map_location="cpu")
                    break
                except Exception as e2:
                    last = e2
            else:
                raise last
    else:
        print(f"Loading Parakeet from HuggingFace: {model_id}")
        try:
            model = ASRModel.from_pretrained(model_id, map_location="cpu")
        except TypeError as e:
            if "abstract class" not in str(e):
                raise
            from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
            from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
            last = e
            for cls in (EncDecRNNTBPEModel, EncDecCTCModelBPE):
                try:
                    model = cls.from_pretrained(model_id, map_location="cpu")
                    break
                except Exception as e2:
                    last = e2
            else:
                raise last

    model.eval()
    return model


def detect_arch(model) -> str:
    """Infer decode-path family from the loaded model.

    Returns one of:
      'tdt'    EncDecRNNTBPEModel with non-zero duration head
      'rnnt'   EncDecRNNTBPEModel with no duration head (plain RNN-T)
      'hybrid' EncDecHybridRNNTCTCBPEModel (we dump the TDT path only)
      'ctc'    EncDecCTCModelBPE / EncDecCTCBPEModel (encoder + linear head)
    """
    cls_name = model.__class__.__name__

    if "Hybrid" in cls_name:
        return "hybrid"
    if "CTC" in cls_name and "RNNT" not in cls_name:
        return "ctc"
    if "RNNT" in cls_name:
        joint = getattr(model, "joint", None)
        extra = int(getattr(joint, "num_extra_outputs", 0) or 0) if joint is not None else 0
        return "tdt" if extra > 0 else "rnnt"
    raise SystemExit(f"error: unrecognised parakeet model class: {cls_name}")


def make_source(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    arch: str,
) -> dict[str, Any]:
    import torch

    source: dict[str, Any] = {
        "kind": "parakeet-nemo",
        "model": args.model,
        "model_dtype": "f32",
        "device": "cpu",
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
        "arch": arch,
    }
    try:
        import nemo

        source["nemo_version"] = nemo.__version__
    except (ImportError, AttributeError):
        pass
    return source


def select_blocks(model, requested: list[int] | None) -> list[int]:
    """Resolve which encoder block indices to dump.

    If --blocks is given, intersect with available layers.
    Otherwise default to {0, mid, last} based on actual depth so the
    same script works for S/L/XL FastConformer variants without a
    per-variant flag.
    """
    n = len(model.encoder.layers)
    if n == 0:
        return []
    if requested:
        return sorted({i for i in requested if 0 <= i < n})
    if n == 1:
        return [0]
    return sorted({0, n // 2, n - 1})


# ---------------------------------------------------------------------------
# Hook-based intermediate capture
# ---------------------------------------------------------------------------

# Imported lazily inside capture_intermediates() / its wrapper. NeMo's
# import surface is heavy and already paid for elsewhere; importing here
# avoids a hard dep at module load time when the dumper is just resolving
# CLI args.
try:
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        RelPositionMultiHeadAttention,
        RelPositionMultiHeadAttentionLongformer,
    )
except Exception:
    RelPositionMultiHeadAttention = None  # type: ignore[assignment]
    RelPositionMultiHeadAttentionLongformer = None  # type: ignore[assignment]

def capture_intermediates(model, block_indices: list[int],
                          sub_block_indices: list[int] | None = None):
    """Register forward hooks on key encoder sub-modules.

    For block indices in `sub_block_indices`, also wrap the layer's
    forward to capture residual state after FF1, attn, conv, FF2 — the
    same observer points exposed by the C++ BlockObserver. These are
    saved under `enc.block.<i>.{ff1,attn,conv,ff2}` to match the C++
    dump filenames.

    Returns (intermediates_dict, hook_handle_list, restore_callbacks).
    """
    import torch

    intermediates: dict[str, Any] = {}
    hooks = []
    restore_cbs: list[Any] = []

    def _hook(name, extract_idx=0):
        def fn(_module, _input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[extract_idx].detach().clone()
            else:
                intermediates[name] = output.detach().clone()

        return fn

    hooks.append(model.preprocessor.register_forward_hook(_hook("mel", 0)))
    hooks.append(model.encoder.pre_encode.register_forward_hook(_hook("enc.pre_encode.out", 0)))
    hooks.append(model.encoder.pos_enc.register_forward_hook(_hook("enc.pos_emb", 1)))
    hooks.append(model.encoder.register_forward_hook(_hook("enc.encoder_out", 0)))

    for i in block_indices:
        if i < len(model.encoder.layers):
            hooks.append(
                model.encoder.layers[i].register_forward_hook(_hook(f"enc.block.{i}.out"))
            )

    # Sub-block capture: wrap each requested layer's forward so we can
    # snapshot the running residual at the same points the C++ observer
    # tags (after_ff1, after_attn, after_conv, after_ff2). The wrapper
    # replicates the exact forward body — there is no other way to
    # reach the residual variable mid-pass through a torch hook.
    if sub_block_indices:
        for idx in sub_block_indices:
            if idx >= len(model.encoder.layers):
                continue
            layer = model.encoder.layers[idx]
            original_forward = layer.forward

            def make_wrapped(layer=layer, idx=idx, original_forward=original_forward):
                def wrapped_forward(x, att_mask=None, pos_emb=None, pad_mask=None,
                                    cache_last_channel=None, cache_last_time=None):
                    residual = x
                    x_n = layer.norm_feed_forward1(x)
                    ff1 = layer.feed_forward1(x_n)
                    residual = residual + layer.dropout(ff1) * layer.fc_factor
                    intermediates[f"enc.block.{idx}.ff1"] = residual.detach().clone()

                    x_n = layer.norm_self_att(residual)
                    if isinstance(layer.self_attn, RelPositionMultiHeadAttention):
                        attn_out = layer.self_attn(query=x_n, key=x_n, value=x_n,
                                                   mask=att_mask, pos_emb=pos_emb,
                                                   cache=cache_last_channel)
                    elif isinstance(layer.self_attn, RelPositionMultiHeadAttentionLongformer):
                        attn_out = layer.self_attn(query=x_n, key=x_n, value=x_n,
                                                   pad_mask=pad_mask, pos_emb=pos_emb,
                                                   cache=cache_last_channel)
                    else:
                        attn_out = layer.self_attn(query=x_n, key=x_n, value=x_n,
                                                   mask=att_mask, cache=cache_last_channel)
                    if cache_last_channel is not None:
                        attn_out, cache_last_channel = attn_out
                    residual = residual + layer.dropout(attn_out)
                    intermediates[f"enc.block.{idx}.attn"] = residual.detach().clone()

                    x_n = layer.norm_conv(residual)
                    conv_out = layer.conv(x_n, pad_mask=pad_mask, cache=cache_last_time)
                    if cache_last_time is not None:
                        conv_out, cache_last_time = conv_out
                    residual = residual + layer.dropout(conv_out)
                    intermediates[f"enc.block.{idx}.conv"] = residual.detach().clone()

                    x_n = layer.norm_feed_forward2(residual)
                    ff2 = layer.feed_forward2(x_n)
                    residual = residual + layer.dropout(ff2) * layer.fc_factor
                    intermediates[f"enc.block.{idx}.ff2"] = residual.detach().clone()

                    out = layer.norm_out(residual)
                    if cache_last_channel is None:
                        return out
                    return out, cache_last_channel, cache_last_time
                return wrapped_forward

            layer.forward = make_wrapped()
            restore_cbs.append((layer, original_forward))

    return intermediates, hooks, restore_cbs


def run_encoder_forward(model, audio_tensor, length_tensor):
    """Drive preprocessor + encoder, regardless of arch.

    We avoid `model.forward()` because its return signature varies by
    architecture (CTC returns 3-tuple; RNN-T/TDT return 2-tuple). The
    encoder always returns (encoded, encoded_len) — channels-first
    (B, D, T) — so calling preprocessor + encoder directly is
    arch-independent.
    """
    import torch

    with torch.inference_mode():
        processed, proc_len = model.preprocessor(
            input_signal=audio_tensor, length=length_tensor
        )
        encoded, encoded_len = model.encoder(audio_signal=processed, length=proc_len)
    return encoded, encoded_len


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_mel(args: argparse.Namespace) -> int:
    """Dump mel spectrogram from NeMo's AudioToMelSpectrogramPreprocessor."""
    configure_torch(args)
    import torch

    model = load_model(args)
    arch = detect_arch(model)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr} arch={arch}")

    source = make_source(
        args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr, arch=arch
    )

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    with torch.inference_mode():
        processed, _proc_len = model.preprocessor(
            input_signal=audio_tensor, length=length_tensor
        )

    mel = to_np(processed)
    print(
        f"mel: shape={mel.shape} min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.6f} std={mel.std():.6f}"
    )
    write_tensor("enc.mel.in", mel, "frontend.mel.norm", source, out_dir=out_dir)
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    """Dump encoder intermediates: mel, pre_encode, pos_emb, per-block, final."""
    configure_torch(args)
    import torch

    model = load_model(args)
    arch = detect_arch(model)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr} arch={arch}")
    print(f"encoder: {len(model.encoder.layers)} layers")

    source = make_source(
        args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr, arch=arch
    )

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage, source, out_dir=out_dir)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    block_indices = select_blocks(model, args.blocks)
    sub_blocks = sorted(set(args.sub_blocks or []))
    print(f"dumping blocks: {block_indices}  sub-blocks: {sub_blocks}")
    intermediates, hooks, restore_cbs = capture_intermediates(
        model, block_indices, sub_block_indices=sub_blocks,
    )

    encoded, _encoded_len = run_encoder_forward(model, audio_tensor, length_tensor)

    dump("enc.mel.in", intermediates["mel"], "frontend.mel.norm")
    dump("enc.pre_encode.out", intermediates["enc.pre_encode.out"], "encoder.pre_encode")
    dump("enc.pos_emb", intermediates["enc.pos_emb"], "encoder.pos_emb")

    for i in block_indices:
        key = f"enc.block.{i}.out"
        if key in intermediates:
            dump(key, intermediates[key], f"encoder.block{i}.out")

    # Sub-block intermediates (matching C++ observer points). These get
    # written under enc.block.<i>.{ff1,attn,conv,ff2} so compare_tensors
    # picks them up by name without any extra plumbing.
    for i in sub_blocks:
        for tag in ("ff1", "attn", "conv", "ff2"):
            key = f"enc.block.{i}.{tag}"
            if key in intermediates:
                dump(key, intermediates[key], f"encoder.block{i}.{tag}")

    enc_final = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)
    dump("enc.final", enc_final, "encoder.final")

    for h in hooks:
        h.remove()
    for layer, original_forward in restore_cbs:
        layer.forward = original_forward

    return 0


def _dump_transducer_first_step(model, enc_t, dump):
    """Shared TDT/RNN-T predictor + joint dump (single-step start state).

    enc_t: (B, T, D) encoder output (already transposed).
    """
    pred_hidden = model.decoder.pred_hidden
    embed_zeros = np.zeros((pred_hidden,), dtype=np.float32)
    dump("dec.embed.0", embed_zeros, "decoder.embed")

    g, hid = model.decoder.predict(
        y=None, state=None, add_sos=False, batch_size=1,
    )

    h_all, c_all = hid
    n_layers = h_all.shape[0]
    for layer in range(n_layers):
        dump(f"dec.lstm.{layer}.h.0", h_all[layer], "decoder.lstm")
        dump(f"dec.lstm.{layer}.c.0", c_all[layer], "decoder.lstm")

    enc_frame0 = enc_t[:, 0:1, :]
    enc_proj = model.joint.enc(enc_frame0)
    dec_proj = model.joint.pred(g)
    joint_out = model.joint.joint_after_projection(enc_proj, dec_proj)
    dump("dec.joint.0", joint_out, "decoder.joint")


def _dump_ctc_logprobs(model, encoded, dump):
    """CTC head dump: encoder output (B, D, T) -> log-probs (B, T, V).

    NeMo's ConvASRDecoder takes channels-first encoder output and
    returns log-softmax over the vocab-with-blank dimension. The first
    frame slice mirrors the transducer 'dec.joint.0' single-step dump.
    """
    log_probs = model.decoder(encoder_output=encoded)  # (B, T, V)
    dump("dec.ctc.logprobs", log_probs, "decoder.ctc.logprobs")
    dump("dec.ctc.logprobs.0", log_probs[:, 0:1, :], "decoder.ctc.logprobs.0")


def cmd_decode(args: argparse.Namespace) -> int:
    """Dump end-to-end: encoder + decoder first step + joint/CTC + greedy transcript.

    Per-arch first-step intermediates dumped:
      tdt / rnnt / hybrid (TDT path):
        dec.enc_out          encoder output (T_enc, d_model)
        dec.embed.0          predictor input at step 0 (zeros)
        dec.lstm.<l>.h.0     per-layer LSTM hidden state after one step
        dec.lstm.<l>.c.0     per-layer LSTM cell state after one step
        dec.joint.0          joint output for (encoder frame 0, predictor start)
      ctc:
        dec.enc_out          encoder output (T_enc, d_model)
        dec.ctc.logprobs     log-softmax over (T, vocab+blank)
        dec.ctc.logprobs.0   first-frame slice (mirrors dec.joint.0 shape role)
    """
    configure_torch(args)
    import torch

    model = load_model(args)
    arch = detect_arch(model)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr} arch={arch}")

    source = make_source(
        args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr, arch=arch
    )

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage, source, out_dir=out_dir)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    encoded, _encoded_len = run_encoder_forward(model, audio_tensor, length_tensor)
    enc_t = encoded.transpose(1, 2)
    dump("dec.enc_out", enc_t, "decoder.enc_out")

    with torch.inference_mode():
        if arch == "ctc":
            _dump_ctc_logprobs(model, encoded, dump)
        else:
            # tdt, rnnt, hybrid (TDT path)
            _dump_transducer_first_step(model, enc_t, dump)

    if not args.skip_transcript:
        print("\n  Running full greedy transcription...")
        # Some .nemo archives (notably parakeet-unified-en-0.6b) ship without
        # a validation_ds config, which NeMo's transcribe() pipeline tries to
        # read for use_start_end_token. Stub it so transcribe() works.
        if getattr(model.cfg, "validation_ds", None) is None:
            from omegaconf import OmegaConf
            model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})
        transcriptions = model.transcribe(audio=[str(audio_path)], batch_size=1)
        if isinstance(transcriptions, (list, tuple)) and len(transcriptions) > 0:
            first = transcriptions[0]
            if isinstance(first, (list, tuple)):
                first = first[0]
            text = first if isinstance(first, str) else first.text
        else:
            text = str(transcriptions)

        print(f"  Transcription: {text}")
        write_transcript(out_dir, text, source=source)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="HF model name (e.g. nvidia/parakeet-tdt-0.6b-v2) or local .nemo path",
    )
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )
    p.add_argument(
        "--language",
        default="en",
        help="Accepted for validate.py compatibility; Parakeet is English-only.",
    )
    p.add_argument(
        "--offline-only",
        action="store_true",
        help=(
            "Force att_context_style='regular' (per-token sliding-window mask) "
            "even when the model ships with 'chunked_limited'. Required for "
            "parakeet-unified-en-0.6b, whose v1 C++ port deliberately targets "
            "offline / full-context mode. DO NOT pass for cache-aware streaming "
            "models like nemotron-speech-streaming-en-0.6b — those must run "
            "with their native chunked_limited mask to reproduce published WER. "
            "Default off: preserve the model's native att_context_style."
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Parakeet reference tensors from NeMo (canonical reference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("mel", help="Dump mel spectrogram only")
    add_common_args(mp)
    mp.set_defaults(func=cmd_mel)

    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    add_common_args(ep)
    ep.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=None,
        help="Block indices to dump. Default: {0, n_layers/2, n_layers-1} based on actual depth.",
    )
    ep.add_argument(
        "--sub-blocks",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Block indices for which to ALSO dump sub-block intermediates "
            "(after FF1, attn, conv, FF2). Matches the C++ BlockObserver "
            "tags. Default: empty (no sub-block dumps)."
        ),
    )
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser(
        "decode",
        help="Dump encoder + decoder first step + joint/CTC + transcript",
    )
    add_common_args(dp)
    dp.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Only dump tensors; do not run full greedy transcription",
    )
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
