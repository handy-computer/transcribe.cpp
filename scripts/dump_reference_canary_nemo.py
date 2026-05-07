#!/usr/bin/env python3
"""
dump_reference_canary_nemo.py - generate Canary multitask AED reference
tensors via the NVIDIA NeMo implementation (canonical reference).

Run through the repo-local Canary reference environment:

    uv run --project scripts/envs/canary \
      scripts/dump_reference_canary_nemo.py encoder \
      --model nvidia/canary-180m-flash \
      --audio samples/jfk.wav \
      --out build/validate/canary/canary-180m-flash/jfk/encoder/ref

    uv run --project scripts/envs/canary \
      scripts/dump_reference_canary_nemo.py decode \
      --model nvidia/canary-180m-flash \
      --audio samples/jfk.wav \
      --source-lang en --target-lang en --task asr --pnc yes \
      --out build/validate/canary/canary-180m-flash/jfk/decode/ref

The --model flag accepts an HF model name (e.g. "nvidia/canary-1b-v2")
or a local path to a .nemo checkpoint.

Architecture summary (Canary = encoder-decoder multitask AED):
    audio (16 kHz mono)
      -> NeMo AudioToMelSpectrogramPreprocessor: 128-mel slaney+htk
         filterbank, 25 ms / 10 ms hann_periodic, preemph=0.97,
         per-feature mean/var normalization
                      => [B, n_mels, T]
      -> FastConformer encoder: depth-wise striding subsampling (factor 8)
         + N relative-position-attention conformer blocks (rel_pos)
                      => [B, T_enc, d_model]
      -> Multitask prompt fed to TransformerDecoder *before* BOS:
         canary-1b:                   [source_lang, target_lang, taskname, pnc]
         canary-1b-v2/1b-flash/180m-: [source_lang, target_lang, task, pnc, toggle_timestamps]
      -> TransformerDecoder (M layers, cross-attends to encoder output)
                      => per-step logits over the concatenated SP vocab
      -> Greedy/beam decode -> transcript

Dump points (one per non-trivial boundary; chosen so Stage 4 has
coverage of every place numerics can drift):

    enc.mel.in                    preprocessor mel-spectrogram output
    enc.pre_encode.out            after FastConformer subsampling
    enc.pos_emb                   relative positional encoding tensor
    enc.block.{0,mid,last}.out    encoder conformer block samples
    enc.final                     final encoder output (cross-attn keys/vals)
    dec.prompt_ids                multitask prompt token ids (int)
    dec.embed.0                   decoder embedding at prompt slot 0
    dec.embed.<bos_pos>           decoder embedding at first slot after prompt
    dec.layer.{0,mid,last}.self_attn.out  decoder self-attn output (first step)
    dec.layer.{0,mid,last}.cross_attn.out decoder cross-attn output (first step)
    dec.lm_head.logits.0          first-step LM head logits
    transcript.json               decoded text + multitask context

Tensor output uses the shared reference dump contract via
scripts.lib.ref_dump (write_tensor records rms / p99_abs needed for
Stage 6 magnitude-aware tolerances; write_transcript records text +
optional token ids).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Load scripts/lib/ref_dump.py by absolute path. We can't do
# `from scripts.lib.ref_dump import ...` because nemo_toolkit 2.7.2 ships
# a top-level `scripts/` package that shadows the repo's scripts/ on
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
    """Convert a torch.Tensor to a row-major float32 numpy array with
    leading size-1 dims squeezed (matches the C++ validation contract)."""
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


# ---------------------------------------------------------------------------
# NeMo model loading
# ---------------------------------------------------------------------------

def load_model(args: argparse.Namespace):
    """Load a Canary multitask AED via NeMo's EncDecMultiTaskModel.

    Accepts an HF model name (e.g. "nvidia/canary-1b-v2") or a local path
    to a .nemo file. Falls back to the generic ASRModel.from_pretrained
    if EncDecMultiTaskModel is not available in the installed NeMo.
    """
    try:
        from nemo.collections.asr.models import EncDecMultiTaskModel
        loader_cls = EncDecMultiTaskModel
    except ImportError:
        from nemo.collections.asr.models import ASRModel
        loader_cls = ASRModel

    model_id = args.model
    local = Path(model_id).expanduser()

    if local.exists():
        print(f"Loading Canary from local path: {local}")
        model = loader_cls.restore_from(str(local), map_location="cpu")
    else:
        print(f"Loading Canary from HuggingFace: {model_id}")
        model = loader_cls.from_pretrained(model_id, map_location="cpu")

    model.eval()
    return model


def make_source(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    source: dict[str, Any] = {
        "kind": "canary-nemo",
        "framework": "nemo",
        "model": args.model,
        "model_dtype": "f32",
        "device": "cpu",
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
    }
    try:
        import nemo

        source["nemo_version"] = nemo.__version__
    except (ImportError, AttributeError):
        pass
    if extra:
        source.update(extra)
    return source


# ---------------------------------------------------------------------------
# Hook-based intermediate capture
# ---------------------------------------------------------------------------

def _block_indices(n_layers: int, requested: list[int]) -> list[int]:
    """Resolve (0, mid, last) defaults to concrete layer indices for a
    layer stack of size n_layers. User-specified layers pass through."""
    if requested:
        return sorted({i for i in requested if 0 <= i < n_layers})
    if n_layers <= 0:
        return []
    if n_layers == 1:
        return [0]
    if n_layers == 2:
        return [0, 1]
    return sorted({0, n_layers // 2, n_layers - 1})


def capture_encoder_intermediates(model, blocks: list[int]):
    """Register forward hooks on FastConformer encoder sub-modules.

    Returns (intermediates_dict, hook_handle_list).
    """
    intermediates: dict[str, Any] = {}
    hooks = []

    def _hook(name, extract_idx=0):
        def fn(_module, _input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[extract_idx].detach().clone()
            else:
                intermediates[name] = output.detach().clone()

        return fn

    if hasattr(model, "preprocessor"):
        hooks.append(model.preprocessor.register_forward_hook(_hook("mel", 0)))

    enc = model.encoder
    if hasattr(enc, "pre_encode"):
        hooks.append(enc.pre_encode.register_forward_hook(_hook("enc.pre_encode.out", 0)))
    if hasattr(enc, "pos_enc"):
        # NeMo's RelPositionalEncoding returns (x, pos_emb).
        hooks.append(enc.pos_enc.register_forward_hook(_hook("enc.pos_emb", 1)))

    layers = getattr(enc, "layers", None)
    if layers is not None:
        for i in blocks:
            if i < len(layers):
                hooks.append(
                    layers[i].register_forward_hook(_hook(f"enc.block.{i}.out"))
                )

    return intermediates, hooks


def capture_decoder_intermediates(model, dec_blocks: list[int]):
    """Register forward hooks on the TransformerDecoder.

    NeMo's TransformerDecoder layer exposes:
      first_sub_layer  — self-attention block
      second_sub_layer — encdec (cross) attention block
      feed_forward     — FFN block

    The decoder runs autoregressively; hooks fire once per decode step.
    To capture only the first step's tensors, each named entry is set
    on first call and not overwritten.

    Returns (intermediates_dict, hook_handle_list).
    """
    intermediates: dict[str, Any] = {}
    hooks = []

    def _hook(name, extract_idx=0):
        def fn(_module, _input, output):
            if name in intermediates:
                return  # first-step only
            if isinstance(output, tuple):
                intermediates[name] = output[extract_idx].detach().clone()
            else:
                intermediates[name] = output.detach().clone()

        return fn

    # NeMo EncDecMultiTaskModel wraps the inner TransformerDecoder under
    # `transf_decoder._decoder`. Older builds may use `decoder` or expose
    # layers directly.
    decoder = None
    for attr in ("transf_decoder", "decoder"):
        candidate = getattr(model, attr, None)
        if candidate is None:
            continue
        if hasattr(candidate, "_decoder"):
            decoder = candidate._decoder
            break
        if hasattr(candidate, "layers"):
            decoder = candidate
            break

    if decoder is None:
        return intermediates, hooks

    layers = getattr(decoder, "layers", None) or getattr(decoder, "_layers", None)
    if layers is None:
        return intermediates, hooks

    for i in dec_blocks:
        if i >= len(layers):
            continue
        layer = layers[i]
        for sub_name, hook_name in (
            # canonical NeMo TransformerDecoderLayer naming
            ("first_sub_layer", f"dec.layer.{i}.self_attn.out"),
            ("second_sub_layer", f"dec.layer.{i}.cross_attn.out"),
            ("third_sub_layer", f"dec.layer.{i}.ffn.out"),
            # alternate naming used in older / variant configs
            ("self_attn", f"dec.layer.{i}.self_attn.out"),
            ("encdec_attn", f"dec.layer.{i}.cross_attn.out"),
            ("feed_forward", f"dec.layer.{i}.ffn.out"),
        ):
            sub = getattr(layer, sub_name, None)
            if sub is None:
                continue
            # Skip if we've already hooked this layer's sub via the
            # canonical name (avoid double-registering).
            if hook_name in {h_name for h_name in intermediates}:
                continue
            hooks.append(sub.register_forward_hook(_hook(hook_name)))

    return intermediates, hooks


def run_encoder_pipeline(model, audio_tensor, length_tensor):
    """Run preprocessor + encoder (+ projection + optional transf_encoder)
    matching EncDecMultiTaskModel.forward() up through enc_states.

    Returns:
        processed_signal              (B, D_mel, T_mel)  preprocessor output
        encoded_native                (B, D_enc, T_enc)  encoder output, channels-first
        enc_states_proj               (B, T_enc, D_dec)  decoder cross-attn KV source
        encoded_len                   (B,)               post-subsampling lengths
    """
    import torch

    with torch.inference_mode():
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=audio_tensor, length=length_tensor
        )
        encoded_native, encoded_len = model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        # Channels-first -> seq-first for the projection.
        enc_states = encoded_native.permute(0, 2, 1)
        enc_states_proj = model.encoder_decoder_proj(enc_states)
        if getattr(model, "use_transf_encoder", False):
            from nemo.collections.asr.parts.submodules.classifier import lens_to_mask
            enc_mask = lens_to_mask(encoded_len, enc_states_proj.shape[1]).to(enc_states_proj.dtype)
            enc_states_proj = model.transf_encoder(
                encoder_states=enc_states_proj, encoder_mask=enc_mask
            )
    return processed_signal, encoded_native, enc_states_proj, encoded_len


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_encoder(args: argparse.Namespace) -> int:
    """Dump encoder intermediates: mel, pre_encode, pos_emb, per-block, native
    encoder output, and the cross-attention KV source (post encoder_decoder_proj
    and optional transf_encoder)."""
    configure_torch(args)
    import torch

    model = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    source = make_source(args=args, audio_path=audio_path, n_samples=pcm.size, sample_rate=sr)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)

    n_enc_layers = len(model.encoder.layers) if hasattr(model.encoder, "layers") else 0
    blocks = _block_indices(n_enc_layers, args.blocks)
    print(f"encoder layers: {n_enc_layers}; dumping blocks: {blocks}")

    intermediates, hooks = capture_encoder_intermediates(model, blocks)

    processed_signal, encoded_native, enc_states_proj, encoded_len = run_encoder_pipeline(
        model, audio_tensor, length_tensor
    )

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage=stage, source=source, out_dir=out_dir)

    if "mel" in intermediates:
        dump("enc.mel.in", intermediates["mel"], "frontend.mel.norm")
    if "enc.pre_encode.out" in intermediates:
        dump("enc.pre_encode.out", intermediates["enc.pre_encode.out"], "encoder.pre_encode")
    if "enc.pos_emb" in intermediates:
        dump("enc.pos_emb", intermediates["enc.pos_emb"], "encoder.pos_emb")
    for i in blocks:
        key = f"enc.block.{i}.out"
        if key in intermediates:
            dump(key, intermediates[key], f"encoder.block{i}.out")

    # Native encoder output (B, D_enc, T_enc) -> (T_enc, D_enc) after squeeze.
    dump("enc.native", encoded_native.transpose(1, 2), "encoder.native_out")
    # Decoder cross-attention KV source (post encoder_decoder_proj +
    # optional transf_encoder).
    dump("enc.final", enc_states_proj, "encoder.final")

    for h in hooks:
        h.remove()

    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Dump end-to-end: encoder + decoder first-step intermediates + transcript.

    The full multitask transcribe() runs the encoder, builds the prompt
    sequence (source_lang / target_lang / task / pnc / toggle_timestamps),
    and runs greedy/beam decoding through the TransformerDecoder. We
    capture per-layer self-attn / cross-attn / FFN outputs at the first
    autoregressive step plus the final logits, then verify the produced
    transcript is plausible.
    """
    configure_torch(args)
    import torch

    model = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    multitask_extra = {
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "task": args.task,
        "pnc": args.pnc,
        "toggle_timestamps": args.toggle_timestamps,
    }
    source = make_source(
        args=args,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        extra=multitask_extra,
    )

    n_enc_layers = len(model.encoder.layers) if hasattr(model.encoder, "layers") else 0
    enc_blocks = _block_indices(n_enc_layers, args.blocks)

    # Decoder layer count: probe both transf_decoder._decoder.layers and
    # decoder.layers shapes; default to 0 if probing fails.
    n_dec_layers = 0
    decoder_obj = None
    for attr in ("transf_decoder", "decoder"):
        candidate = getattr(model, attr, None)
        if candidate is None:
            continue
        inner = getattr(candidate, "_decoder", candidate)
        layers = getattr(inner, "layers", None) or getattr(inner, "_layers", None)
        if layers is not None:
            n_dec_layers = len(layers)
            decoder_obj = inner
            break
    dec_blocks = _block_indices(n_dec_layers, args.dec_blocks)
    print(f"encoder layers: {n_enc_layers}; decoder layers: {n_dec_layers}")
    print(f"dumping encoder blocks: {enc_blocks}; decoder blocks: {dec_blocks}")

    enc_intermediates, enc_hooks = capture_encoder_intermediates(model, enc_blocks)
    dec_intermediates, dec_hooks = capture_decoder_intermediates(model, dec_blocks)

    text: str = ""
    tokens: list[int] | None = None

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage=stage, source=source, out_dir=out_dir)

    audio_tensor = torch.tensor(pcm, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([pcm.size], dtype=torch.long)
    processed_signal, encoded_native, enc_states_proj, encoded_len = run_encoder_pipeline(
        model, audio_tensor, length_tensor
    )
    dump("enc.native", encoded_native.transpose(1, 2), "encoder.native_out")
    dump("enc.final", enc_states_proj, "encoder.final")
    if "mel" in enc_intermediates:
        dump("enc.mel.in", enc_intermediates["mel"], "frontend.mel.norm")
    for i in enc_blocks:
        key = f"enc.block.{i}.out"
        if key in enc_intermediates:
            dump(key, enc_intermediates[key], f"encoder.block{i}.out")

    # ----- Run the canonical multitask transcribe() to capture decoder
    # intermediates and produce the reference transcript. ----------------
    if not args.skip_transcript:
        manifest_path: Path | None = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as fh:
                # Carry both `task` (canary-1b-v2 / 1b-flash / 180m-flash) and
                # `taskname` (original canary-1b) keys. NeMo's manifest reader
                # will pick the field its config expects and ignore the other.
                manifest_entry = {
                    "audio_filepath": str(audio_path),
                    "duration": float(pcm.size) / sr,
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                    "task": args.task,
                    "taskname": args.task,
                    "pnc": args.pnc,
                    "answer": "na",
                }
                if args.toggle_timestamps in ("yes", "no"):
                    # `timestamp` is the field the v2 / flash manifest reader
                    # expects; canary-1b does not consume it.
                    manifest_entry["timestamp"] = args.toggle_timestamps
                fh.write(json.dumps(manifest_entry) + "\n")
                manifest_path = Path(fh.name)

            # Reduce beam to 1 for deterministic dumps unless overridden.
            if args.beam > 0:
                try:
                    decode_cfg = model.cfg.decoding
                    decode_cfg.beam.beam_size = args.beam
                    model.change_decoding_strategy(decode_cfg)
                except Exception as exc:
                    print(f"  warning: could not set beam_size={args.beam}: {exc}")

            print("\n  Running canonical transcribe() ...")
            output = model.transcribe(
                str(manifest_path),
                batch_size=1,
            )
            if isinstance(output, (list, tuple)) and output:
                first = output[0]
                if isinstance(first, (list, tuple)):
                    first = first[0]
                if hasattr(first, "text"):
                    text = first.text
                    raw_tokens = getattr(first, "y_sequence", None)
                    if raw_tokens is not None:
                        try:
                            tokens = [int(x) for x in raw_tokens.tolist()] if hasattr(raw_tokens, "tolist") else list(raw_tokens)
                        except (TypeError, ValueError):
                            tokens = None
                elif isinstance(first, str):
                    text = first
                else:
                    text = str(first)
            else:
                text = str(output)
            print(f"  transcribe() text: {text!r}")
        finally:
            if manifest_path is not None and manifest_path.exists():
                manifest_path.unlink()

    # ----- Decoder intermediates (captured from the transcribe() pass) ---
    for i in dec_blocks:
        for sub in ("self_attn", "cross_attn", "ffn"):
            key = f"dec.layer.{i}.{sub}.out"
            if key in dec_intermediates:
                dump(key, dec_intermediates[key], f"decoder.layer{i}.{sub}")

    # ----- Transcript artifact ------------------------------------------
    if text:
        write_transcript(out_dir, text, source=source, tokens=tokens)
        print(f"  wrote {out_dir / 'transcript.json'}")

    for h in enc_hooks + dec_hooks:
        h.remove()

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="HF model name (e.g. nvidia/canary-1b-v2) or local .nemo path",
    )
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )


def add_block_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=[],
        help="Encoder block indices to dump (default: 0, mid, last)",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Canary multitask AED reference tensors from NeMo (canonical reference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    add_common_args(ep)
    add_block_args(ep)
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser(
        "decode",
        help="Dump encoder + decoder first-step intermediates + canonical transcribe() transcript",
    )
    add_common_args(dp)
    add_block_args(dp)
    dp.add_argument(
        "--dec-blocks",
        type=int,
        nargs="*",
        default=[],
        help="Decoder block indices to dump (default: 0, mid, last)",
    )
    dp.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    dp.add_argument("--target-lang", default="en", help="Target language code (default: en — ASR)")
    dp.add_argument(
        "--task",
        default="asr",
        choices=["asr", "ast", "s2t_translation"],
        help="Multitask task token. canary-1b uses s2t_translation; v2/flash use ast.",
    )
    dp.add_argument("--pnc", default="yes", choices=["yes", "no"], help="Punctuation & capitalization toggle")
    dp.add_argument(
        "--toggle-timestamps",
        default="no",
        choices=["yes", "no", "none"],
        help="Timestamp toggle (canary-1b-v2/1b-flash/180m-flash). Use 'none' to omit slot for canary-1b.",
    )
    dp.add_argument(
        "--beam",
        type=int,
        default=1,
        help="Beam size for transcribe() (default: 1 = greedy; canary-1b's published numbers use 5)",
    )
    dp.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Only dump tensors; skip transcribe() entirely.",
    )
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
