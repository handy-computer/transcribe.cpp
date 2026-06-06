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
        # Drop unknown kwargs (covers any streaming-only key the
        # constructor doesn't accept — e.g. NeMo 2.7.x rejects
        # `att_chunk_context_size` while upstream main accepts it).
        for k in list(kwargs.keys()):
            if k not in accepted:
                kwargs.pop(k)
                dropped.append(f"unknown:{k}")
        # Belt-and-suspenders: when we're forcing the encoder to
        # regular-style for the unified-en offline port, drop the
        # streaming kwargs even if the constructor would accept them.
        # The regular path doesn't read these and dropping makes the
        # downstream `set_default_att_context_size` no-op explicit.
        # We do NOT drop them when the caller wants the model's native
        # streaming style (chunked_limited / chunked_limited_with_rc).
        if force_regular_att_style:
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


def is_prompt_aware(model) -> bool:
    """True iff the model conditions the encoder output on a target_lang prompt
    (EncDecRNNTBPEModelWithPrompt / EncDecHybridRNNTCTCBPEModelWithPrompt).
    Predecessor English-only variants have neither attribute and short-circuit.
    """
    return bool(getattr(model, "concat", False)) and hasattr(model, "prompt_kernel")


def resolve_prompt_id(model, lang: str) -> int:
    """Map a target language tag (e.g. 'en-US') to its prompt embedding index.
    Raises with the legal key list so misconfigured manifests fail loudly.
    """
    prompt_dict = model.cfg.model_defaults.get("prompt_dictionary", {})
    if lang not in prompt_dict:
        sample = list(prompt_dict.keys())[:12]
        raise SystemExit(
            f"error: unknown target_lang '{lang}'; not in prompt_dictionary. "
            f"Sample keys: {sample}"
        )
    return int(prompt_dict[lang])


def apply_prompt(model, encoded_bdt, prompt_id: int):
    """Apply the prompt_kernel to a (B, D, T) encoder output, returning
    a (B, D, T) prompt-conditioned encoder output. Mirrors the forward()
    body of EncDecRNNTBPEModelWithPrompt so the joint head receives the
    same input it would at transcribe() time.
    """
    import torch

    enc_btd = encoded_bdt.transpose(1, 2)  # (B, D, T) -> (B, T, D)
    batch_size, time_steps, _ = enc_btd.shape
    num_prompts = int(model.num_prompts)
    prompt = torch.zeros(batch_size, time_steps, num_prompts,
                         dtype=enc_btd.dtype, device=enc_btd.device)
    prompt_indices = torch.full((batch_size,), prompt_id, dtype=torch.long,
                                device=enc_btd.device)
    prompt.scatter_(2, prompt_indices.view(batch_size, 1, 1)
                    .expand(-1, time_steps, -1), 1.0)
    concat = torch.cat([enc_btd, prompt], dim=-1)
    out_btd = model.prompt_kernel(concat).to(enc_btd.dtype)
    return out_btd.transpose(1, 2)  # (B, T, D) -> (B, D, T)


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

    # Prompt-aware (multilingual nemotron-3.5): also dump the post-prompt
    # encoder output, since that is what feeds the joint/CTC head.
    if is_prompt_aware(model):
        prompt_id = resolve_prompt_id(model, args.language)
        print(f"prompt: target_lang={args.language!r} prompt_id={prompt_id}")
        with torch.inference_mode():
            enc_prompted = apply_prompt(model, encoded, prompt_id)
        dump("enc.prompted", enc_prompted.transpose(1, 2), "encoder.prompted")

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


def cmd_streaming(args: argparse.Namespace) -> int:
    """Dump per-chunk cache-aware streaming intermediates.

    Iterates NeMo's CacheAwareStreamingAudioBuffer over the audio,
    calling conformer_stream_step per chunk. For each step we dump:

      stream.chunk.<step>.mel_in      input mel frames fed to the
                                      encoder this chunk (after the
                                      buffer prepends pre_encode_cache)
      stream.chunk.<step>.enc_out     encoder output for this chunk
                                      (the streaming-aware slice, post
                                      drop_extra_pre_encoded)
      stream.chunk.<step>.cache_lc_in_<L>
      stream.chunk.<step>.cache_lc_out_<L>
                                      per-layer last_channel cache
                                      (B, D, T_cache) before/after the
                                      chunk for selected layers L
      stream.chunk.<step>.cache_lt_in_<L>
      stream.chunk.<step>.cache_lt_out_<L>
                                      per-layer last_time cache
                                      (B, D, k-1) before/after the
                                      chunk for selected layers L
      stream.chunk.<step>.channel_len cache_last_channel_len (B,) — the
                                      valid-frame counter NeMo masks
                                      against
      stream.chunk.<step>.tokens      cumulative emitted-token-id list
                                      after this step (decode best path)

    Plus stream/transcript.json with the final streaming + offline
    transcripts (matches conformer_stream_step's return_transcription).

    --att-context-right N picks one of the model's training-time
    lookahead settings via set_default_att_context_size. nemotron-
    speech-streaming-en-0.6b accepts N in {0, 1, 6, 13}. When omitted
    the model's first entry (max-accuracy) is used.

    --pad-and-drop-preencoded mirrors the reference script's flag for
    ONNX-export-style first-chunk semantics (use the subsequent chunk
    config for the first chunk too). Off by default; on means
    drop_extra_pre_encoded fires on step 0 as well.
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

    if model.encoder.att_context_style != "chunked_limited":
        print(
            f"error: model att_context_style={model.encoder.att_context_style!r}; "
            f"streaming dump only works for chunked_limited models",
            file=sys.stderr,
        )
        return 1

    # Select the target latency setting. set_default_att_context_size
    # also triggers setup_streaming_params() internally, which updates
    # encoder.streaming_cfg.{chunk_size,shift_size,pre_encode_cache_size,
    # drop_extra_pre_encoded} consistently.
    if args.att_context_right is not None:
        left = model.encoder.att_context_size[0]
        target = [left, int(args.att_context_right)]
        if not hasattr(model.encoder, "set_default_att_context_size"):
            print("error: model does not support multiple lookaheads", file=sys.stderr)
            return 1
        try:
            model.encoder.set_default_att_context_size(att_context_size=target)
        except Exception as e:
            print(f"error: set_default_att_context_size({target}) failed: {e}",
                  file=sys.stderr)
            return 1

    cfg = model.encoder.streaming_cfg
    chunk_size = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
    pre_encode_cache_size = (
        cfg.pre_encode_cache_size[1] if isinstance(cfg.pre_encode_cache_size, list)
        else cfg.pre_encode_cache_size
    )
    print(
        f"streaming: arch={arch} att_context={list(model.encoder.att_context_size)} "
        f"chunk_size={chunk_size} pre_encode_cache_size={pre_encode_cache_size} "
        f"drop_extra_pre_encoded={cfg.drop_extra_pre_encoded}"
    )

    source = make_source(
        args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr, arch=arch
    )
    source["streaming"] = {
        "att_context_size": [int(model.encoder.att_context_size[0]),
                             int(model.encoder.att_context_size[1])],
        "chunk_size": int(chunk_size),
        "pre_encode_cache_size": int(pre_encode_cache_size),
        "drop_extra_pre_encoded": int(cfg.drop_extra_pre_encoded),
        "pad_and_drop_preencoded": bool(args.pad_and_drop_preencoded),
    }

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        write_tensor(name, a, stage, source, out_dir=out_dir)

    # Set up the streaming buffer the same way the reference script
    # does. compare_vs_offline=False so we only run streaming; offline
    # parity is the validate harness's job.
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=bool(args.pad_and_drop_preencoded),
    )
    streaming_buffer.append_audio_file(str(audio_path), stream_id=-1)

    # Decide which encoder layers to snapshot. {0, n/2, n-1} keeps the
    # dump size bounded (24 layers * 4 tensors * N chunks = a lot).
    n_layers = len(model.encoder.layers)
    sel_layers = sorted({0, n_layers // 2, n_layers - 1}) if n_layers else []
    if args.layers:
        sel_layers = sorted({i for i in args.layers if 0 <= i < n_layers})
    print(f"streaming: dumping cache for layers {sel_layers}")

    # Register a forward hook on the encoder to capture the
    # streaming-aware encoded tensor (post drop_extra_pre_encoded) on
    # every call. conformer_stream_step calls model.encoder under the
    # hood; the hook fires once per chunk.
    captured: dict[str, Any] = {}

    def _enc_hook(_module, _inputs, output):
        # output = (encoded, encoded_len, cache_lc_next, cache_lt_next, channel_len_next)
        captured["enc_out"] = output[0].detach().clone()

    enc_handle = model.encoder.register_forward_hook(_enc_hook)

    # Initial cache state (zeros, as NeMo defines it).
    cache_lc, cache_lt, channel_len = model.encoder.get_initial_cache_state(
        batch_size=1
    )

    per_chunk_text: list[str] = []
    cumulative_tokens: list[int] = []
    previous_hypotheses = None
    pred_out_stream = None
    iterator = iter(streaming_buffer)

    with torch.inference_mode():
        for step_num, (chunk_audio, chunk_lengths) in enumerate(iterator):
            # Snapshot the cache *before* the chunk (i.e. the inputs).
            for layer_idx in sel_layers:
                # cache_last_channel: shape (n_layers, B, T_cache, D)
                dump(
                    f"stream.chunk.{step_num}.cache_lc_in_{layer_idx}",
                    cache_lc[layer_idx],
                    "streaming.cache_in",
                )
                # cache_last_time: shape (n_layers, B, D, k-1)
                dump(
                    f"stream.chunk.{step_num}.cache_lt_in_{layer_idx}",
                    cache_lt[layer_idx],
                    "streaming.cache_in",
                )
            dump(
                f"stream.chunk.{step_num}.mel_in",
                chunk_audio,
                "streaming.mel_in",
            )

            drop_extra = (
                cfg.drop_extra_pre_encoded
                if (step_num != 0 or args.pad_and_drop_preencoded)
                else 0
            )

            (
                pred_out_stream,
                transcribed_texts,
                cache_lc,
                cache_lt,
                channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk_audio.to(torch.float32),
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_lc,
                cache_last_time=cache_lt,
                cache_last_channel_len=channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=drop_extra,
                return_transcription=True,
            )

            # Snapshot the cache *after* the chunk (i.e. the outputs).
            for layer_idx in sel_layers:
                dump(
                    f"stream.chunk.{step_num}.cache_lc_out_{layer_idx}",
                    cache_lc[layer_idx],
                    "streaming.cache_out",
                )
                dump(
                    f"stream.chunk.{step_num}.cache_lt_out_{layer_idx}",
                    cache_lt[layer_idx],
                    "streaming.cache_out",
                )
            dump(
                f"stream.chunk.{step_num}.channel_len",
                channel_len.to(torch.float32),
                "streaming.channel_len",
            )
            if "enc_out" in captured:
                # Encoder output for this chunk (post drop_extra_pre_encoded).
                # Layout: (B, D, T_valid). We dump (T_valid, D) so it
                # mirrors the C++ row-major [T_enc, d_enc] layout.
                enc = captured["enc_out"].transpose(1, 2)  # (B, T, D)
                dump(
                    f"stream.chunk.{step_num}.enc_out",
                    enc,
                    "streaming.enc_out",
                )

            # pred_out_stream is per-step decoded best path (list of tensors
            # for transducer; tensor for CTC). For the parity harness we
            # care about the cumulative token id sequence.
            if isinstance(pred_out_stream, list):
                tokens_so_far = [int(x) for tensor in pred_out_stream for x in tensor.flatten().tolist()]
            else:
                tokens_so_far = [int(x) for x in pred_out_stream.flatten().tolist()]
            cumulative_tokens = tokens_so_far

            if isinstance(transcribed_texts, (list, tuple)) and transcribed_texts:
                first = transcribed_texts[0]
                step_text = first if isinstance(first, str) else getattr(first, "text", str(first))
            else:
                step_text = ""
            per_chunk_text.append(step_text)
            print(f"  step[{step_num}]: text={step_text!r}")

    enc_handle.remove()
    final_text = per_chunk_text[-1] if per_chunk_text else ""
    write_transcript(out_dir, final_text, source=source)

    # Also write per-chunk text history + final token ids as a JSON
    # sidecar so the harness doesn't have to re-derive them.
    import json
    history_path = out_dir / "stream_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump({
            "per_chunk_text": per_chunk_text,
            "final_tokens": cumulative_tokens,
            "streaming": source["streaming"],
        }, f, indent=2)
    print(f"streaming: wrote {history_path}")

    return 0


def cmd_buffered_streaming(args: argparse.Namespace) -> int:
    """Dump per-chunk BUFFERED streaming intermediates for unified models.

    Mirrors NeMo's reference inference loop at
    `examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`
    (lines 397-447). The model's encoder is reconfigured to
    `chunked_limited_with_rc` with a runtime-selected (L, C, R) tuple
    via `set_default_att_context_size`; each chunk re-runs the FULL
    encoder over a sliding [left | chunk | right] PCM window; the
    RNN-T greedy-batch label-looping decoder runs over just the chunk
    frames with state carried across chunks.

    --left-secs / --chunk-secs / --right-secs select the (L, C, R)
    tuple in seconds (default: 5.6 / 1.04 / 1.04 — the model card's
    "best accuracy" row for parakeet-unified-en-0.6b).

    For each step we dump:

      stream.chunk.<step>.audio_in    PCM window fed to the encoder
                                      this chunk (left+chunk+right
                                      samples; trailing zeros on the
                                      final chunk).
      stream.chunk.<step>.enc_out     encoder output frames AFTER
                                      slicing off the `left` frames
                                      (covers chunk+right except on
                                      the final chunk where it covers
                                      the remaining tail).
      stream.chunk.<step>.tokens      cumulative emitted-token-id list
                                      after this chunk.

    Plus `transcript.json` with the final text and `stream_history.json`
    with per-chunk text + the resolved geometry (so the harness can
    cross-check).
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

    enc_style = getattr(model.encoder, "att_context_style", "regular")
    if enc_style != "chunked_limited_with_rc":
        print(
            f"error: model att_context_style={enc_style!r}; "
            f"buffered_streaming requires chunked_limited_with_rc",
            file=sys.stderr,
        )
        return 1

    # Mirror the reference script's deterministic preprocessor setup
    # (speech_to_text_streaming_infer_rnnt.py lines 207-208 and 277-278).
    # Dither defaults to 1e-05 which would inject per-run noise; the
    # parity harness needs byte-stable mel features. pad_to=0 is also
    # what the reference uses so the mel time axis is exactly
    # n_samples/hop without trailing zero-padded frames.
    try:
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
    except AttributeError:
        pass

    # Resolve geometry. NeMo's preprocessor frame stride is 10ms; the
    # encoder subsamples ×8, so each encoder frame represents
    # frame_stride_sec * subsampling_factor = 0.08s (80ms) of audio.
    feature_stride_sec = float(model.cfg.preprocessor["window_stride"])
    features_per_sec = 1.0 / feature_stride_sec
    encoder_subsampling_factor = int(model.encoder.subsampling_factor)
    # NeMo's `make_divisible_by` rounding ensures the audio-sample count
    # is a multiple of the subsampling factor; the (hop, subsample) pair
    # always satisfies this for parakeet.
    features_frame2audio_samples = (
        int(sr * feature_stride_sec) // encoder_subsampling_factor
    ) * encoder_subsampling_factor
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

    left_frames  = int(args.left_secs  * features_per_sec / encoder_subsampling_factor)
    chunk_frames = int(args.chunk_secs * features_per_sec / encoder_subsampling_factor)
    right_frames = int(args.right_secs * features_per_sec / encoder_subsampling_factor)
    if chunk_frames < 1:
        print(f"error: chunk_secs={args.chunk_secs} resolves to {chunk_frames} encoder frames; must be >= 1",
              file=sys.stderr)
        return 1

    samples_left  = left_frames  * encoder_frame2audio_samples
    samples_chunk = chunk_frames * encoder_frame2audio_samples
    samples_right = right_frames * encoder_frame2audio_samples

    # Apply the runtime (L, C, R) to the encoder. This both swaps the
    # default att_context_size and triggers setup_streaming_params() —
    # buffered streaming does not actually use streaming_cfg (no cache
    # carry), but the chunked-attention mask shape is controlled by
    # att_context_size at forward time.
    model.encoder.set_default_att_context_size(
        att_context_size=[left_frames, chunk_frames, right_frames]
    )

    # Configure the decoding pipeline. The reference script requires
    # greedy_batch + loop_labels=True; we mirror that.
    from omegaconf import OmegaConf, open_dict
    import copy
    decoding_cfg = copy.deepcopy(model.cfg.decoding) if hasattr(model.cfg, "decoding") else OmegaConf.create({})
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy_batch"
        if "greedy" not in decoding_cfg:
            decoding_cfg.greedy = OmegaConf.create({})
        decoding_cfg.greedy.loop_labels = True
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.greedy.preserve_alignments = False
        decoding_cfg.tdt_include_token_duration = False
        decoding_cfg.fused_batch_size = -1
        if "beam" in decoding_cfg:
            decoding_cfg.beam.return_best_hypothesis = True
    model.change_decoding_strategy(decoding_cfg)

    # The TDT decode driver exposed by NeMo for buffered streaming.
    decoding_computer = model.decoding.decoding.decoding_computer

    source = make_source(
        args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr, arch=arch
    )
    source["buffered_streaming"] = {
        "att_context_size": [left_frames, chunk_frames, right_frames],
        "left_secs":  float(args.left_secs),
        "chunk_secs": float(args.chunk_secs),
        "right_secs": float(args.right_secs),
        "encoder_frame2audio_samples": int(encoder_frame2audio_samples),
        "samples_left":  int(samples_left),
        "samples_chunk": int(samples_chunk),
        "samples_right": int(samples_right),
    }
    print(
        f"buffered streaming: (L,C,R)_frames=[{left_frames}, {chunk_frames}, {right_frames}] "
        f"= [{samples_left}, {samples_chunk}, {samples_right}] samples "
        f"= [{args.left_secs:.2f}, {args.chunk_secs:.2f}, {args.right_secs:.2f}] s"
    )

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        write_tensor(name, a, stage, source, out_dir=out_dir)

    # Per-chunk intermediate capture for layer-by-layer divergence
    # bisect against the cpp port. The captures land in `intermediates`
    # and are dumped + cleared inside the per-step loop. Hooks fire
    # exactly once per encoder forward pass (i.e. once per chunk), so
    # the dict reflects the most recent chunk.
    intermediates: dict[str, Any] = {}

    def _hook(name, extract_idx=0):
        def fn(_module, _input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[extract_idx].detach().clone()
            else:
                intermediates[name] = output.detach().clone()
        return fn

    block_hooks = []
    block_hooks.append(model.preprocessor.register_forward_hook(
        _hook("enc.mel.in", 0)))
    block_hooks.append(model.encoder.pre_encode.register_forward_hook(
        _hook("enc.pre_encode.out", 0)))
    for i in range(len(model.encoder.layers)):
        block_hooks.append(model.encoder.layers[i].register_forward_hook(
            _hook(f"enc.block.{i}.out")))

    # Sub-block wrapping for block 0 — captures the residual after each
    # of FF1/attn/conv/FF2 so cpp-vs-ref drift can be localized to a
    # specific Conformer sub-component. Mirrors `_set_up_intermediates_capture`
    # in the offline path. Wrapping installs a custom forward; restored
    # at the end of the loop.
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        RelPositionMultiHeadAttention,
        RelPositionMultiHeadAttentionLongformer,
    )
    block0_layer = model.encoder.layers[0]
    block0_original_forward = block0_layer.forward
    def block0_wrapped(x, att_mask=None, pos_emb=None, pad_mask=None,
                       cache_last_channel=None, cache_last_time=None):
        layer = block0_layer
        residual = x
        x_n = layer.norm_feed_forward1(x)
        ff1 = layer.feed_forward1(x_n)
        residual = residual + layer.dropout(ff1) * layer.fc_factor
        intermediates["enc.block.0.ff1"] = residual.detach().clone()

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
        intermediates["enc.block.0.attn"] = residual.detach().clone()

        x_n = layer.norm_conv(residual)
        conv_out = layer.conv(x_n, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            conv_out, cache_last_time = conv_out
        residual = residual + layer.dropout(conv_out)
        intermediates["enc.block.0.conv"] = residual.detach().clone()

        x_n = layer.norm_feed_forward2(residual)
        ff2 = layer.feed_forward2(x_n)
        residual = residual + layer.dropout(ff2) * layer.fc_factor
        intermediates["enc.block.0.ff2"] = residual.detach().clone()

        out = layer.norm_out(residual)
        if cache_last_channel is None:
            return out
        return out, cache_last_channel, cache_last_time
    block0_layer.forward = block0_wrapped

    # Spin up the StreamingBatchedAudioBuffer NeMo's reference script uses.
    from nemo.collections.asr.parts.utils.streaming_utils import (
        ContextSize,
        StreamingBatchedAudioBuffer,
    )

    context_samples = ContextSize(
        left=samples_left, chunk=samples_chunk, right=samples_right,
    )

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)  # [1, T]
    audio_lengths = torch.tensor([pcm.size], dtype=torch.long)

    buffer = StreamingBatchedAudioBuffer(
        batch_size=1,
        context_samples=context_samples,
        dtype=audio_tensor.dtype,
        device=audio_tensor.device,
    )
    rest_audio_lengths = audio_lengths.clone()

    current_batched_hyps = None
    state = None
    left_sample = 0
    right_sample = min(context_samples.chunk + context_samples.right, audio_tensor.shape[1])

    per_chunk_text: list[str] = []
    cumulative_tokens: list[int] = []
    step_num = 0

    with torch.inference_mode():
        while left_sample < audio_tensor.shape[1]:
            chunk_length = min(right_sample, audio_tensor.shape[1]) - left_sample
            is_last_chunk_batch = chunk_length >= rest_audio_lengths
            is_last_chunk = right_sample >= audio_tensor.shape[1]
            chunk_lengths_batch = torch.where(
                is_last_chunk_batch,
                rest_audio_lengths,
                torch.full_like(rest_audio_lengths, fill_value=chunk_length),
            )
            buffer.add_audio_batch_(
                audio_tensor[:, left_sample:right_sample],
                audio_lengths=chunk_lengths_batch,
                is_last_chunk=is_last_chunk,
                is_last_chunk_batch=is_last_chunk_batch,
            )

            # Dump the PCM window the encoder sees this chunk.
            dump(
                f"stream.chunk.{step_num}.audio_in",
                buffer.samples[0],
                "buffered_streaming.audio_in",
            )

            encoder_output, encoder_output_len = model(
                input_signal=buffer.samples,
                input_signal_length=buffer.context_size_batch.total(),
            )
            encoder_output = encoder_output.transpose(1, 2)  # (B, T_enc, D)
            encoder_context = buffer.context_size.subsample(
                factor=encoder_frame2audio_samples
            )
            encoder_context_batch = buffer.context_size_batch.subsample(
                factor=encoder_frame2audio_samples
            )
            # Drop left-context encoder frames before decoding.
            encoder_output_chunk = encoder_output[:, encoder_context.left:]

            # Dump the chunk-aligned encoder output the decoder sees.
            dump(
                f"stream.chunk.{step_num}.enc_out",
                encoder_output_chunk[0],
                "buffered_streaming.enc_out",
            )

            # Dump per-chunk per-block intermediates (for cpp-vs-ref
            # encoder fp32 drift bisect).
            for cap_name, cap_tensor in intermediates.items():
                # Convert NeMo's [B, T, D] (or [B, D, T] for pre_encode)
                # to the cpp-port layout [T, D]. pre_encode returns [B, T, D],
                # encoder layers return [B, T, D] too. encoder.pre_encode is
                # special-cased upstream. Trust the squeeze + match.
                t = cap_tensor[0] if cap_tensor.dim() == 3 else cap_tensor
                dump(
                    f"stream.chunk.{step_num}.{cap_name}",
                    t,
                    f"buffered_streaming.{cap_name}",
                )

            chunk_batched_hyps, _, state = decoding_computer(
                x=encoder_output_chunk,
                out_len=torch.where(
                    is_last_chunk_batch,
                    encoder_output_len - encoder_context_batch.left,
                    encoder_context_batch.chunk,
                ),
                prev_batched_state=state,
                multi_biasing_ids=None,
            )
            if current_batched_hyps is None:
                current_batched_hyps = chunk_batched_hyps
            else:
                current_batched_hyps.merge_(chunk_batched_hyps)

            # Snapshot cumulative tokens after this chunk. Use NeMo's
            # `batched_hyps_to_hypotheses` to extract the clean per-batch
            # `y_sequence` (already strips blanks); raw
            # `current_batched_hyps.transcript` includes the blank id
            # 1024 which SPM rejects.
            from nemo.collections.asr.parts.utils.rnnt_utils import (
                batched_hyps_to_hypotheses,
            )
            hyps_view = batched_hyps_to_hypotheses(
                current_batched_hyps, None, batch_size=1
            )
            tokens_so_far: list[int] = []
            if hyps_view and getattr(hyps_view[0], "y_sequence", None) is not None:
                tokens_so_far = [int(x) for x in hyps_view[0].y_sequence.tolist()]
            cumulative_tokens = tokens_so_far
            step_text = model.tokenizer.ids_to_text(tokens_so_far) if tokens_so_far else ""
            per_chunk_text.append(step_text)
            print(f"  step[{step_num}]: text={step_text!r}")

            rest_audio_lengths = rest_audio_lengths - chunk_lengths_batch
            left_sample = right_sample
            right_sample = min(right_sample + context_samples.chunk, audio_tensor.shape[1])
            step_num += 1

    for h in block_hooks:
        h.remove()
    block0_layer.forward = block0_original_forward

    final_text = per_chunk_text[-1] if per_chunk_text else ""
    write_transcript(out_dir, final_text, source=source)

    import json
    history_path = out_dir / "stream_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump({
            "per_chunk_text": per_chunk_text,
            "final_tokens": cumulative_tokens,
            "buffered_streaming": source["buffered_streaming"],
        }, f, indent=2)
    print(f"buffered streaming: wrote {history_path}")
    print(f"final transcript: {final_text!r}")

    return 0


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

    # For prompt-aware models the joint head sees the prompt-conditioned
    # encoder output, not the bare encoder output. Compute it once and
    # feed the joint dump from the conditioned version.
    if is_prompt_aware(model):
        prompt_id = resolve_prompt_id(model, args.language)
        print(f"prompt: target_lang={args.language!r} prompt_id={prompt_id}")
        with torch.inference_mode():
            encoded_for_joint = apply_prompt(model, encoded, prompt_id)
        # Capture both so Stage 4 can validate the bare encoder and the
        # prompt-conditioned output independently.
        dump("dec.enc_out", encoded.transpose(1, 2), "decoder.enc_out")
        dump("dec.enc_out_prompted", encoded_for_joint.transpose(1, 2),
             "decoder.enc_out_prompted")
        enc_t = encoded_for_joint.transpose(1, 2)
    else:
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
        if is_prompt_aware(model):
            # The prompt model's transcribe() routes through a Lhotse
            # dataloader whose dataset requires per-cut supervision language
            # tags. NeMo's built-in temp manifest does not emit any, so the
            # dataset throws on prompt_key='None'. Bypass it: run greedy
            # decoding directly on the prompt-conditioned encoder output we
            # already computed.
            _, encoded_len = run_encoder_forward(model, audio_tensor, length_tensor)
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded_for_joint,
                encoded_lengths=encoded_len,
                return_hypotheses=False,
            )
            first = hyps[0] if isinstance(hyps, (list, tuple)) and len(hyps) > 0 else hyps
            text = first if isinstance(first, str) else getattr(first, "text", str(first))
            print(f"  Transcription: {text}")
            write_transcript(out_dir, text, source=source)
            return 0
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
        help=(
            "Target language tag. Ignored by English-only parakeet variants. "
            "For prompt-conditioned models (e.g. nemotron-3.5-asr-streaming-0.6b) "
            "this MUST be a key in the model's prompt_dictionary "
            "(e.g. 'en-US', 'en', 'es-ES'). Default 'en' (predecessor-compatible)."
        ),
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

    sp = sub.add_parser(
        "streaming",
        help="Dump per-chunk cache-aware streaming intermediates",
    )
    add_common_args(sp)
    sp.add_argument(
        "--att-context-right",
        type=int,
        default=None,
        help=(
            "Pick a right-context (lookahead) value from the model's "
            "training menu. nemotron-speech-streaming-en-0.6b accepts "
            "{0, 1, 6, 13}. Default: model's first entry (max accuracy)."
        ),
    )
    sp.add_argument(
        "--pad-and-drop-preencoded",
        action="store_true",
        help=(
            "Treat the first chunk like a subsequent one (use subsequent "
            "chunk_size + drop_extra_pre_encoded=streaming_cfg value). "
            "Mirrors the reference script's flag; matches ONNX export."
        ),
    )
    sp.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Encoder layer indices to dump cache_lc/lt for. Default: "
            "{0, n_layers/2, n_layers-1} based on actual depth."
        ),
    )
    sp.set_defaults(func=cmd_streaming)

    bp = sub.add_parser(
        "buffered_streaming",
        help="Dump per-chunk buffered RNN-T streaming intermediates (parakeet-unified)",
    )
    add_common_args(bp)
    bp.add_argument(
        "--left-secs",
        type=float,
        default=5.6,
        help=(
            "Left-context window in seconds. Default 5.6s — the trained "
            "default for parakeet-unified-en-0.6b (matches 70 encoder "
            "frames at the 80ms post-subsample rate)."
        ),
    )
    bp.add_argument(
        "--chunk-secs",
        type=float,
        default=1.04,
        help=(
            "Chunk size in seconds. Default 1.04s (13 encoder frames) — "
            "the model card's best-accuracy row for parakeet-unified-en-0.6b."
        ),
    )
    bp.add_argument(
        "--right-secs",
        type=float,
        default=1.04,
        help=(
            "Right-context (lookahead) in seconds. Default 1.04s "
            "(13 encoder frames) — matches the best-accuracy row."
        ),
    )
    bp.set_defaults(func=cmd_buffered_streaming)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
