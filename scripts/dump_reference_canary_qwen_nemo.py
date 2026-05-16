#!/usr/bin/env python3
"""
dump_reference_canary_qwen_nemo.py - generate Canary-Qwen SALM reference
tensors via the NVIDIA NeMo implementation (canonical reference).

Run through the repo-local Canary-Qwen reference environment:

    uv run --project scripts/envs/canary_qwen \
      scripts/dump_reference_canary_qwen_nemo.py decode \
      --model nvidia/canary-qwen-2.5b \
      --audio samples/jfk.wav \
      --out build/validate/canary_qwen/canary-qwen-2.5b/jfk/decode/ref

The --model flag accepts an HF model name (e.g. "nvidia/canary-qwen-2.5b")
or a local path NeMo can resolve.

Architecture summary (Canary-Qwen = audio-LLM, NeMo SALM):
    audio (16 kHz mono)
      -> NeMo AudioToMelSpectrogramPreprocessor: 128-mel slaney+htk
         filterbank, 25 ms / 10 ms hann_periodic, preemph=0.97,
         per-feature mean/var normalization
                      => [B, n_mels, T]
      -> FastConformer encoder (32 layers, d_model=1024, rel_pos,
         dw_striding factor=8): same module class as canary-1b-flash
                      => [B, T_enc, d_model=1024]
      -> IdentityConnector (no params) -> projection to output_dim=2048
                      => [B, T_enc, 2048]
      -> SCATTER into Qwen3-1.7B input_embeds at positions where
         input_ids == audio_locator_tag id (`<|audioplaceholder|>`)
                      => [B, T_lm, 2048]
      -> Qwen3-1.7B causal LM (28 layers, GQA 16/8, head_dim=128, RoPE,
         tied word embeddings) with LoRA on q_proj/v_proj
                      => per-step logits over the Qwen3 BPE vocab
      -> Greedy decode -> transcript

Dump points (one per non-trivial boundary; mirror the canary + qwen3_asr
families so the C++ Stage 4 has a single contract):

    enc.mel.in                   preprocessor mel-spectrogram output
    enc.pre_encode.out           after FastConformer subsampling (factor 8)
    enc.pos_emb                  relative positional encoding tensor
    enc.block.{0,16,31}.out      encoder conformer block samples
    enc.final                    final encoder output (B, T_enc, 1024)
    perception.proj.out          after AudioPerceptionModule projection (B, T_enc, 2048)
    dec.token_emb                LM input embeddings pre-injection (B, T_lm, 2048)
    dec.audio_injected           LM input embeddings AFTER audio scatter
    dec.block.{0,14,27}.out      Qwen3 LM block outputs (first step)
    dec.out_before_head          after LM final RMSNorm
    dec.logits_raw.gen0          first-step LM head logits over the Qwen3 vocab
    transcript.json              decoded text + token ids + prompt ids

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
# `from scripts.lib.ref_dump import ...` because nemo_toolkit ships a
# top-level `scripts/` package that shadows the repo's scripts/ on sys.path.
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
# NeMo SALM model loading
# ---------------------------------------------------------------------------

def load_model(args: argparse.Namespace):
    """Load a Canary-Qwen SALM model via NeMo's speechlm2.SALM class.

    Accepts an HF model name (e.g. "nvidia/canary-qwen-2.5b") or a local
    path NeMo can resolve. Requires NeMo >= 2.5.0 (SALM is in
    nemo.collections.speechlm2, added in 2.5.0).
    """
    from nemo.collections.speechlm2.models import SALM

    model_id = args.model
    local = Path(model_id).expanduser()

    if local.exists():
        print(f"Loading SALM from local path: {local}")
        model = SALM.from_pretrained(str(local), map_location="cpu")
    else:
        print(f"Loading SALM from HuggingFace: {model_id}")
        model = SALM.from_pretrained(model_id, map_location="cpu")

    model.eval()

    # Override dither for deterministic dumps. Config declares dither=1e-5
    # (NeMo training default); inference must use 0.0 to keep mel and all
    # downstream tensors reproducible. Stage 4 numerical comparisons depend
    # on this.
    pre = getattr(model, "perception", None)
    pre = getattr(pre, "preprocessor", None)
    if pre is not None and hasattr(pre, "featurizer") and hasattr(pre.featurizer, "dither"):
        prev = float(pre.featurizer.dither)
        if prev != 0.0:
            print(f"  overriding preprocessor dither {prev} -> 0.0 for deterministic dumps")
            pre.featurizer.dither = 0.0
    elif pre is not None and hasattr(pre, "dither"):
        prev = float(pre.dither)
        if prev != 0.0:
            print(f"  overriding preprocessor dither {prev} -> 0.0 for deterministic dumps")
            pre.dither = 0.0

    return model


def make_source(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    model_dtype: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    source: dict[str, Any] = {
        "kind": "canary-qwen-salm-nemo",
        "framework": "nemo",
        "model": args.model,
        "model_dtype": model_dtype,
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


def model_dtype_name(model) -> str:
    import torch

    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return "unknown"
    return {
        torch.bfloat16: "bf16",
        torch.float16:  "f16",
        torch.float32:  "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


# ---------------------------------------------------------------------------
# Hook-based intermediate capture
# ---------------------------------------------------------------------------

class CaptureHook:
    """Record the first output tensor a module emits during forward."""

    def __init__(self, extract_idx: int = 0) -> None:
        self.value = None
        self.extract_idx = extract_idx

    def __call__(self, module, inputs, output) -> None:
        if self.value is not None:
            return  # first-step only (LM hooks fire once per AR step)
        if isinstance(output, tuple):
            output = output[self.extract_idx]
        self.value = output


class CaptureNthHook:
    """Record the Nth (0-indexed) output tensor a module emits.

    Used to capture mid-generation logits: with `target=8`, the hook
    fires on the 9th invocation of `lm_head`. The 1st invocation is
    prefill (gen0); the 2nd through 9th are autoregressive steps; the
    9th invocation thus corresponds to the C++ side's `dec.logits_raw.gen8`
    naming (= 9th lm_head call = step iter 7 of the step loop).
    """

    def __init__(self, target: int, extract_idx: int = 0) -> None:
        self.value = None
        self.target = int(target)
        self.extract_idx = extract_idx
        self._n_calls = 0

    def __call__(self, module, inputs, output) -> None:
        idx = self._n_calls
        self._n_calls += 1
        if idx != self.target or self.value is not None:
            return
        if isinstance(output, tuple):
            output = output[self.extract_idx]
        self.value = output


def _block_indices(n_layers: int, requested: list[int]) -> list[int]:
    if requested:
        return sorted({i for i in requested if 0 <= i < n_layers})
    if n_layers <= 0:
        return []
    if n_layers == 1:
        return [0]
    if n_layers == 2:
        return [0, 1]
    return sorted({0, n_layers // 2, n_layers - 1})


def find_perception(model):
    """Locate model.perception and its sub-modules. Returns a dict with
    keys: perception, preprocessor, encoder, modality_adapter."""
    perception = getattr(model, "perception", None)
    if perception is None:
        return {}
    return {
        "perception": perception,
        "preprocessor": getattr(perception, "preprocessor", None),
        "encoder": getattr(perception, "encoder", None),
        "modality_adapter": getattr(perception, "modality_adapter", None),
    }


def find_llm_inner(model):
    """Locate the inner Qwen3 transformer (the .model attr of a HF
    Qwen3ForCausalLM, possibly wrapped by PEFT).

    Returns (llm_for_causal_lm, qwen3_text_model, embed_tokens, layers, norm, lm_head).
    Any field can be None if probing fails.

    Note on embed_tokens: SALM deletes `self.llm.model.embed_tokens` at
    construction time and exposes the embedding at the top level as
    `model.embed_tokens` (so it can do audio-placeholder scatter before
    handing inputs_embeds to the inner Qwen3). We must hook the top-level
    SALM module, not the inner text_model, otherwise the hook never fires
    and dec.token_emb is silently empty.
    """
    llm = getattr(model, "llm", None)
    if llm is None:
        return (None, None, None, None, None, None)

    # PEFT wrapper exposes the base model as .base_model.model
    base = llm
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model

    # base is now Qwen3ForCausalLM (.model = Qwen3Model, .lm_head = Linear)
    text_model = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    # Top-level SALM embedding (the inner one is deleted by SALM ctor).
    embed_tokens = getattr(model, "embed_tokens", None)
    layers = getattr(text_model, "layers", None) if text_model is not None else None
    norm = getattr(text_model, "norm", None) if text_model is not None else None

    return (llm, text_model, embed_tokens, layers, norm, lm_head)


def register_perception_hooks(perception_parts: dict, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the perception module (preprocessor + FastConformer encoder +
    modality_adapter + projection)."""
    hooks: dict[str, CaptureHook] = {}

    def hook_forward(name: str, module, *, extract_idx: int = 0) -> None:
        if module is None:
            return
        h = CaptureHook(extract_idx=extract_idx)
        module.register_forward_hook(h)
        hooks[name] = h

    pre = perception_parts.get("preprocessor")
    enc = perception_parts.get("encoder")
    perception = perception_parts.get("perception")

    # Preprocessor returns (processed_signal, processed_signal_length).
    hook_forward("enc.mel.in", pre, extract_idx=0)

    if enc is not None:
        # FastConformer encoder sub-modules (mirror canary dumper).
        if hasattr(enc, "pre_encode"):
            hook_forward("enc.pre_encode.out", enc.pre_encode, extract_idx=0)
        if hasattr(enc, "pos_enc"):
            # NeMo RelPositionalEncoding returns (x, pos_emb).
            hook_forward("enc.pos_emb", enc.pos_enc, extract_idx=1)

        layers = getattr(enc, "layers", None)
        if layers is not None:
            for i in blocks:
                if 0 <= i < len(layers):
                    hook_forward(f"enc.block.{i}.out", layers[i], extract_idx=0)

        # FastConformer encoder forward returns (encoded, encoded_len).
        hook_forward("enc.final", enc, extract_idx=0)

    # AudioPerceptionModule's outer forward returns the projection output
    # (encoder d_model -> output_dim=2048). Capture it.
    if perception is not None:
        hook_forward("perception.proj.out", perception, extract_idx=0)

    return hooks


def register_lm_hooks(text_model, embed_tokens, layers, norm, lm_head,
                      blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the inner Qwen3 transformer + lm_head + the SALM-top-level
    embed_tokens (see find_llm_inner for why we don't use the inner one)."""
    hooks: dict[str, CaptureHook] = {}

    def hook_forward(name: str, module, *, extract_idx: int = 0) -> None:
        if module is None:
            return
        h = CaptureHook(extract_idx=extract_idx)
        module.register_forward_hook(h)
        hooks[name] = h

    # `embed_tokens` here is SALM's top-level embedding (see
    # find_llm_inner doc) — the only path through which prompt tokens
    # become embeddings before audio-placeholder scatter.
    hook_forward("dec.token_emb", embed_tokens, extract_idx=0)

    if layers is not None:
        for i in blocks:
            if 0 <= i < len(layers):
                hook_forward(f"dec.block.{i}.out", layers[i], extract_idx=0)

    hook_forward("dec.out_before_head", norm, extract_idx=0)
    hook_forward("dec.logits_raw.gen0", lm_head, extract_idx=0)

    # Mid-generation logits: capture the 9th lm_head invocation. Same
    # naming convention as funasr_nano / canary_qwen C++ side. Lives
    # alongside gen0 on the same lm_head module.
    if lm_head is not None:
        gen8 = CaptureNthHook(target=8, extract_idx=0)
        lm_head.register_forward_hook(gen8)
        hooks["dec.logits_raw.gen8"] = gen8
    return hooks


def register_injection_hook(text_model) -> dict[str, CaptureHook]:
    """Capture the LM input embeddings AFTER audio-feature scatter.

    SALM's forward injects audio frames into inputs_embeds before calling
    the inner Qwen3 transformer. The cleanest capture is a kwargs-aware
    pre-hook on the Qwen3 text model (text_model.forward) that reads
    `inputs_embeds`.
    """
    hooks: dict[str, CaptureHook] = {}
    if text_model is None:
        return hooks

    inj = CaptureHook()

    def pre(module, args, kwargs):
        if inj.value is not None:
            return
        emb = kwargs.get("inputs_embeds")
        if emb is None:
            return
        inj.value = emb

    text_model.register_forward_pre_hook(pre, with_kwargs=True)
    hooks["dec.audio_injected"] = inj
    return hooks


# ---------------------------------------------------------------------------
# Generation driver (SALM)
# ---------------------------------------------------------------------------

def build_prompt(model, audio_path: Path) -> list[list[dict]]:
    """Build SALM's chat-format prompt for a single ASR request.

    Mirrors the model card example:
      [{"role": "user",
        "content": "Transcribe the following: <|audioplaceholder|>",
        "audio": ["<path>"]}]
    """
    locator = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
    return [
        [
            {
                "role": "user",
                "content": f"Transcribe the following: {locator}",
                "audio": [str(audio_path)],
            }
        ]
    ]


def generate_transcript(model, audio_path: Path, max_new_tokens: int) -> tuple[str, list[int]]:
    """Run SALM.generate and return (text, token_ids)."""
    import torch

    prompts = build_prompt(model, audio_path)
    with torch.inference_mode():
        answer_ids = model.generate(prompts=prompts, max_new_tokens=max_new_tokens)

    # answer_ids is a tensor [B, T]; first row.
    if hasattr(answer_ids, "cpu"):
        token_row = answer_ids[0].detach().cpu()
        token_list = [int(x) for x in token_row.tolist()]
    elif isinstance(answer_ids, (list, tuple)):
        token_list = [int(x) for x in (answer_ids[0] if len(answer_ids) else [])]
    else:
        token_list = []

    text = ""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "ids_to_text"):
        try:
            text = tokenizer.ids_to_text(token_list)
        except Exception as exc:
            print(f"  warning: tokenizer.ids_to_text failed: {exc}")
            text = ""
    if not text and tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            text = tokenizer.decode(token_list, skip_special_tokens=True)
        except Exception as exc:
            print(f"  warning: tokenizer.decode failed: {exc}")
    return text.strip(), token_list


# ---------------------------------------------------------------------------
# decode subcommand
# ---------------------------------------------------------------------------

def cmd_decode(args: argparse.Namespace) -> int:
    """Single-pass dump: load SALM, hook perception + LM, run generate,
    write all .f32/.json sidecars + transcript.json."""
    import torch

    configure_torch(args)
    model = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)

    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    # Locate sub-modules.
    perception_parts = find_perception(model)
    enc = perception_parts.get("encoder")
    n_enc_layers = len(enc.layers) if enc is not None and hasattr(enc, "layers") else 0
    enc_blocks = _block_indices(n_enc_layers, args.enc_blocks)
    print(f"perception encoder layers: {n_enc_layers}; dumping enc blocks: {enc_blocks}")

    llm, text_model, embed_tokens, lm_layers, lm_norm, lm_head = find_llm_inner(model)
    n_lm_layers = len(lm_layers) if lm_layers is not None else 0
    lm_blocks = _block_indices(n_lm_layers, args.lm_blocks)
    print(f"LM layers: {n_lm_layers}; dumping LM blocks: {lm_blocks}")

    # Register all hooks.
    enc_hooks = register_perception_hooks(perception_parts, enc_blocks)
    inj_hooks = register_injection_hook(text_model)
    lm_hooks = register_lm_hooks(text_model, embed_tokens, lm_layers, lm_norm, lm_head, lm_blocks)

    # Build provenance.
    locator = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
    locator_id = None
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "text_to_ids"):
        try:
            ids = tokenizer.text_to_ids(locator)
            if ids:
                locator_id = int(ids[0]) if len(ids) == 1 else [int(x) for x in ids]
        except Exception:
            pass

    source = make_source(
        args=args,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype_name(model),
        extra={
            "audio_locator_tag": locator,
            "audio_locator_id": locator_id,
            "max_new_tokens": int(args.max_new_tokens),
        },
    )

    # Run generate (this fires every hook).
    print("\nRunning SALM.generate() ...")
    text, token_ids = generate_transcript(model, audio_path, args.max_new_tokens)
    print(f"  transcript: {text!r}")
    print(f"  token count: {len(token_ids)}")

    # Flush captured tensors.
    def flush(name: str, value, stage: str) -> None:
        if value is None:
            print(f"  WARN: no capture for {name}")
            return
        a = to_np(value)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage=stage, source=source, out_dir=out_dir)

    flush("enc.mel.in", enc_hooks.get("enc.mel.in", CaptureHook()).value, "frontend.mel.norm")
    flush("enc.pre_encode.out", enc_hooks.get("enc.pre_encode.out", CaptureHook()).value, "encoder.pre_encode")
    flush("enc.pos_emb", enc_hooks.get("enc.pos_emb", CaptureHook()).value, "encoder.pos_emb")
    for i in enc_blocks:
        key = f"enc.block.{i}.out"
        flush(key, enc_hooks.get(key, CaptureHook()).value, f"encoder.block{i}.out")
    flush("enc.final", enc_hooks.get("enc.final", CaptureHook()).value, "encoder.final")
    flush("perception.proj.out", enc_hooks.get("perception.proj.out", CaptureHook()).value, "perception.proj.out")

    flush("dec.token_emb", lm_hooks.get("dec.token_emb", CaptureHook()).value, "decoder.token_emb")
    flush("dec.audio_injected", inj_hooks.get("dec.audio_injected", CaptureHook()).value, "decoder.audio_injected")
    for i in lm_blocks:
        key = f"dec.block.{i}.out"
        flush(key, lm_hooks.get(key, CaptureHook()).value, f"decoder.block{i}.out")
    flush("dec.out_before_head", lm_hooks.get("dec.out_before_head", CaptureHook()).value, "decoder.out_before_head")
    flush("dec.logits_raw.gen0", lm_hooks.get("dec.logits_raw.gen0", CaptureHook()).value, "decoder.logits_raw.gen0")
    flush("dec.logits_raw.gen8", lm_hooks.get("dec.logits_raw.gen8", CaptureNthHook(target=8)).value, "decoder.logits_raw.gen8")

    # Write transcript.
    if text or token_ids:
        write_transcript(out_dir, text, source=source, tokens=token_ids)
        print(f"  wrote {out_dir / 'transcript.json'}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="HF model name (e.g. nvidia/canary-qwen-2.5b) or local path",
    )
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )
    # validate.py passes --language; canary-qwen is English-only (no language
    # conditioning in the LM), so this is accepted but not used.
    p.add_argument(
        "--language",
        default=None,
        help="Accepted for validate.py compatibility; canary-qwen is English-only.",
    )
    p.add_argument(
        "--enc-blocks",
        type=int,
        nargs="*",
        default=[],
        help="Encoder block indices to dump (default: 0, mid, last)",
    )
    p.add_argument(
        "--lm-blocks",
        type=int,
        nargs="*",
        default=[],
        help="LM block indices to dump (default: 0, mid, last)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for generate (default: 128, matches the model card example)",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Canary-Qwen SALM reference tensors from NeMo (canonical reference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser(
        "decode",
        help="Single-pass dump: encoder + audio scatter + LM intermediates + transcript",
    )
    _add_common(dp)
    dp.set_defaults(func=cmd_decode)

    # Canary-Qwen runs perception + LM in a single fused forward; an
    # encoder-only pass would still load the 5 GB checkpoint, so `encoder`
    # is an alias for `decode` (qwen3_asr style). validate.py's separate
    # "encoder" / "decoder" stage names map to this single dumper invocation.
    ep = sub.add_parser("encoder", help="Alias for decode (single full-pass dump)")
    _add_common(ep)
    ep.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
