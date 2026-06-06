#!/usr/bin/env python3
"""
dump_reference_voxtral_realtime_transformers.py - generate Voxtral Realtime
(2602) reference tensors and transcripts from the mainline Hugging Face
Transformers implementation (`VoxtralRealtimeForConditionalGeneration`).

This is the STREAMING variant (model_type `voxtral_realtime`), architecturally
DISTINCT from the 2507 `voxtral` models (which have their own dumper). It shares
only the projector shape, the tekken tokenizer, and the family brand. See
`reports/porting/voxtral/forward-map.md` and the intake known_risks for the full
deviation list.

Usage:

    # offline whole-clip oracle (the per-tensor contract Stage 4 implements)
    uv run --project scripts/envs/voxtral_realtime \
      scripts/dump_reference_voxtral_realtime_transformers.py decode \
      --model models/Voxtral-Mini-4B-Realtime-2602 \
      --audio samples/jfk.wav \
      --out build/validate/voxtral_realtime/voxtral-mini-4b-realtime-2602/jfk/decode/ref

    # streaming-replay oracle (transcript + per-step logits for offline<->stream
    # numerical-equivalence verification, intake risk #1)
    uv run --project scripts/envs/voxtral_realtime \
      scripts/dump_reference_voxtral_realtime_transformers.py stream \
      --model models/Voxtral-Mini-4B-Realtime-2602 \
      --audio samples/jfk.wav \
      --out build/validate/voxtral_realtime/voxtral-mini-4b-realtime-2602/jfk/stream/ref

    # encoder subcommand is a no-op (decode dumps every tensor); kept so the
    # uniform validate.py encoder+decode invocation works.

Architecture (audio-llm / ADDITIVE fusion, streaming):
    audio_tower.embedder   causal Conv1d 128->1280 (k3 s1) -> GELU
                           causal Conv1d 1280->1280 (k3 s2) -> GELU  [enc.embedder.out]
    audio_tower.layers x32  RMSNorm pre-norm, RoPE(theta 1e6, hd64), causal +
                            sliding-window(750) attn (q/v/o bias, k no bias),
                            SwiGLU/silu MLP (bias only on down_proj)
    audio_tower.norm        final RMSNorm                              [enc.out]
    get_audio_features: reshape (T,1280)->(T/4,5120)  [4x frame grouping]
    multi_modal_projector: Linear(5120->3072,no bias) -> GELU
                           -> Linear(3072->3072,no bias)               [proj.out]
    forward: inputs_embeds += audio_embeds  (ADDITIVE, 1 audio embed per text
             position; NOT masked_scatter)                             [dec.audio_injected]
    time conditioning: t = sinusoid(num_delay_tokens); each decoder layer applies
             post_attention_layernorm(h) * (1 + ada_rms_norm(t))  (adaptive RMSNorm)
    language_model (26-layer, tied lm_head, RoPE theta 1e6 hd128, GQA 32/8) -> logits

Dump points (offline `decode`):
    enc.mel.in            input log-mel features (fixed global_log_mel_max=1.5)
    enc.embedder.out      conv stem output (causal convs)
    enc.block.<i>.out     selected encoder block outputs (1280-dim)
    enc.out               final encoder output (last_hidden_state)
    proj.out              projector output = audio embeds in text-hidden space
    dec.token_emb         LM token embeddings pre audio-add
    dec.audio_injected    LM input embeddings after additive audio fusion
    dec.block.<i>.out     selected LM decoder block outputs
    dec.out_before_head   LM output after final RMSNorm, pre lm_head
    dec.logits_raw        pre-softmax logits at the first generated position
    dec.logits_raw.gen8   pre-softmax logits 8 positions later

Dump points (streaming `stream`):
    stream.logits_raw     pre-softmax logits at the first generated step
    stream.logits_raw.gen8
    transcript.json       streaming-decoded text (must match offline if equivalent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Shared write helpers keep the on-disk contract (and rms/p99_abs sidecar
# fields) identical across families.
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


def resolve_model(raw: str) -> tuple[str, bool]:
    """Resolve a model argument to (model_id_or_path, local_files_only)."""
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


def model_dtype_name(model) -> str:
    import torch

    dtype = next(model.parameters()).dtype
    return {
        torch.bfloat16: "bf16",
        torch.float16: "f16",
        torch.float32: "f32",
    }.get(dtype, str(dtype).removeprefix("torch."))


def load_reference(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Voxtral Realtime model from {model_id} ({source}, "
        f"transformers {transformers.__version__}, device={args.device})..."
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[args.dtype]
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="eager",  # deterministic hooks
    ).eval()
    model.to(args.device)
    return processor, model


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class CaptureHook:
    """Record the first output tensor a module emits during forward.

    The captured tensor is CLONED: the top-level forward fuses audio with
    `inputs_embeds += audio_embeds` (modeling line 984), an in-place mutation
    of the embed_tokens output. Storing a bare reference would make
    `dec.token_emb` show post-fusion values. Cloning at capture time (the
    embed_tokens hook fires before the `+=`) preserves the true pre-fusion
    tensor and is cheap for a one-time dump.
    """

    def __init__(self) -> None:
        self.value = None

    def __call__(self, module, inputs, output) -> None:
        if self.value is not None:
            return
        if isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        self.value = output.detach().clone()


def _hook_forward(hooks: dict[str, CaptureHook], name: str, module) -> None:
    h = CaptureHook()
    module.register_forward_hook(h)
    hooks[name] = h


def core(model):
    """The submodule holding audio_tower / language_model / projector.

    transformers >=5.7 nests these under a `.model` (VoxtralRealtimeModel)
    wrapper with a separate top-level `lm_head`; earlier checkouts kept them
    at the top level. Tolerate both.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "audio_tower"):
        return inner
    return model


def register_encoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the causal RoPE audio encoder (model.audio_tower).

    VoxtralRealtimeEncoder.forward: embedder (causal conv stem) -> RoPE ->
    sliding-window-causal layers -> final RMSNorm -> last_hidden_state.
    """
    hooks: dict[str, CaptureHook] = {}
    enc = core(model).audio_tower
    _hook_forward(hooks, "enc.embedder.out", enc.embedder)
    for i in blocks:
        if 0 <= i < len(enc.layers):
            _hook_forward(hooks, f"enc.block.{i}.out", enc.layers[i])
    _hook_forward(hooks, "enc.out", enc)
    return hooks


def register_projector_hooks(model) -> dict[str, CaptureHook]:
    """Hook the multimodal projector (audio embeds in text-hidden space).

    The 4x frame-grouping reshape (T,1280)->(T/4,5120) happens in
    model.get_audio_features BEFORE the projector, so the projector output is
    the (T/4, 3072) audio-embedding tensor added into the LM embeddings.
    """
    hooks: dict[str, CaptureHook] = {}
    _hook_forward(hooks, "proj.out", core(model).multi_modal_projector)
    return hooks


def register_decoder_hooks(model, blocks: list[int]) -> dict[str, CaptureHook]:
    """Hook the inner text decoder (VoxtralRealtimeTextModel).

    language_model (VoxtralRealtimeTextForCausalLM):
      model (VoxtralRealtimeTextModel)
        embed_tokens  => dec.token_emb (pre audio-add embedding lookup)
        layers[i]     => dec.block.<i>.out
        norm          => dec.out_before_head

    Additive audio fusion (inputs_embeds += pooler_output) happens in
    VoxtralRealtimeForConditionalGeneration.forward BEFORE delegating to
    language_model; we catch the post-fusion inputs_embeds with a pre-hook
    on the inner text model.
    """
    hooks: dict[str, CaptureHook] = {}
    inner = core(model).language_model  # VoxtralRealtimeTextModel
    _hook_forward(hooks, "dec.token_emb", inner.embed_tokens)
    for i in blocks:
        if 0 <= i < len(inner.layers):
            _hook_forward(hooks, f"dec.block.{i}.out", inner.layers[i])
    _hook_forward(hooks, "dec.out_before_head", inner.norm)

    inj = CaptureHook()

    def pre(module, args, kwargs):
        if inj.value is not None:
            return
        emb = kwargs.get("inputs_embeds")
        if emb is None and args:
            emb = args[0]
        if emb is not None:
            inj.value = emb.detach().clone()

    inner.register_forward_pre_hook(pre, with_kwargs=True)
    hooks["dec.audio_injected"] = inj
    return hooks


def make_source(
    *,
    args: argparse.Namespace,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    model_dtype: str,
    num_delay_tokens: int,
    mode: str,
) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "kind": "voxtral-realtime-transformers-native",
        "transformers_version": transformers.__version__,
        "transformers_file": transformers.__file__,
        "model": model_id,
        "model_dtype": model_dtype,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
        "language": "auto",  # processor hardcodes language=None (auto-detect)
        "num_delay_tokens": int(num_delay_tokens),
        "mode": mode,
        "request": "transcription",
    }


def decode_tokens(processor, token_ids: list[int]) -> str:
    """Decode generated token IDs via the mistral-common tokenizer."""
    tok = processor.tokenizer
    for attempt in (
        lambda: processor.batch_decode([token_ids], skip_special_tokens=True)[0],
        lambda: tok.decode(token_ids, skip_special_tokens=True),
        lambda: tok.decode(token_ids),
    ):
        try:
            return attempt().strip()
        except Exception:  # noqa: BLE001 - fall through to next decode form
            continue
    return ""


def flush(name: str, t, stage: str, *, out_dir: Path, source: dict[str, Any]) -> None:
    if t is None:
        print(f"  WARN: {name}: no value captured")
        return
    a = to_np(t)
    write_tensor(name, a, stage=stage, source={**source, "hook": name}, out_dir=out_dir)
    print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")


def cmd_decode(args: argparse.Namespace) -> int:
    """Offline whole-clip oracle: a single teacher-forced forward pass.

    The encoder, projector and additive fusion do not depend on the emitted
    text tokens, so the whole-clip tensors are well defined. We first run a
    greedy generate() to obtain the emitted token sequence and the reference
    transcript, then run ONE forward(use_cache=False, padding_cache=None) over
    the full mel + full (aligned) token sequence. With use_cache=False the
    encoder runs a single sliding-window-causal pass over every frame -- the
    "encode-once" structure Stage 4 builds in ggml.
    """
    import torch

    configure_torch(args)
    processor, model = load_reference(args)
    model_id, _ = resolve_model(args.model)
    model_dtype = model_dtype_name(model)

    audio_path = Path(args.audio).expanduser().resolve()
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        raise SystemExit(f"Voxtral Realtime expects 16kHz audio; got {sr} Hz in {audio_path}")
    if args.language not in (None, "", "auto", "detect", "en"):
        print(f"  NOTE: --language={args.language!r} ignored; the realtime processor "
              "is auto-detect only (TranscriptionRequest language=None).")

    inputs = processor(audio=pcm, is_streaming=False, return_tensors="pt")
    num_delay_tokens = int(inputs["num_delay_tokens"]) if args.num_delay_tokens is None else args.num_delay_tokens
    model_dt = next(model.parameters()).dtype

    def _cast(v):
        if not hasattr(v, "to"):
            return v
        if torch.is_floating_point(v):
            return v.to(args.device, dtype=model_dt)
        return v.to(args.device)

    inputs = {k: _cast(v) for k, v in inputs.items()}
    mel = inputs["input_features"]
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = make_source(
        args=args, model_id=model_id, audio_path=audio_path, n_samples=pcm.size,
        sample_rate=sr, model_dtype=model_dtype, num_delay_tokens=num_delay_tokens,
        mode="offline",
    )

    # 1) Greedy generate -> transcript + emitted tokens (output length is hard
    #    clamped by the model to ceil(mel_frames / audio_length_per_tok)).
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequences = gen_out.sequences
    new_tokens = sequences[0, prompt_len:].tolist()
    text = decode_tokens(processor, new_tokens)
    print(f"offline transcript: {text!r}")
    print(f"  prompt_len={prompt_len} generated={len(new_tokens)} num_delay_tokens={num_delay_tokens}")

    # 2) Authoritative audio-token count = number of projector audio embeds.
    with torch.inference_mode():
        audio_feats = model.get_audio_features(input_features=mel, use_cache=False, return_dict=True)
    n_audio = audio_feats.pooler_output.shape[1]

    # 3) Align the full token sequence to n_audio for the additive-fusion
    #    broadcast (inputs_embeds[L] += audio_embeds[n_audio] requires L ==
    #    n_audio). Padding only touches positions after the transcript, well
    #    past the gated gen0/gen8 logits, so the dumped tensors are unaffected.
    full_ids = sequences[0].tolist()
    pad_id = 11  # generation_config.pad_token_id
    if len(full_ids) < n_audio:
        full_ids = full_ids + [pad_id] * (n_audio - len(full_ids))
    elif len(full_ids) > n_audio:
        full_ids = full_ids[:n_audio]
    full_ids_t = torch.tensor([full_ids], device=args.device)

    # 4) Single whole-clip teacher-forced forward with all hooks registered.
    n_enc_blocks = len(core(model).audio_tower.layers)
    n_dec_blocks = len(core(model).language_model.layers)
    enc_blocks = sorted({0, n_enc_blocks // 2, n_enc_blocks - 1, *args.enc_blocks})
    dec_blocks = sorted({0, n_dec_blocks // 2, n_dec_blocks - 1, *args.dec_blocks})

    enc_hooks = register_encoder_hooks(model, enc_blocks)
    proj_hooks = register_projector_hooks(model)
    dec_hooks = register_decoder_hooks(model, dec_blocks)
    print(f"hooks: encoder={list(enc_hooks)} projector={list(proj_hooks)} decoder={list(dec_hooks)}")

    write_tensor(
        "enc.mel.in", to_np(mel), stage="encoder",
        source={**source, "hook": "processor.input_features"}, out_dir=out_dir,
    )

    with torch.inference_mode():
        fwd = model.forward(
            input_ids=full_ids_t,
            input_features=mel,
            num_delay_tokens=num_delay_tokens,
            use_cache=False,
            padding_cache=None,
            logits_to_keep=0,
        )

    logits = fwd.logits  # [1, n_audio, vocab]
    g0 = prompt_len - 1                # last prompt position predicts gen0
    if 0 <= g0 < logits.shape[1]:
        flush("dec.logits_raw", logits[:, g0, :], "decoder", out_dir=out_dir, source=source)
    g8 = g0 + 8
    if 0 <= g8 < logits.shape[1]:
        flush("dec.logits_raw.gen8", logits[:, g8, :], "decoder", out_dir=out_dir, source=source)

    for name, hook in enc_hooks.items():
        flush(name, hook.value, "encoder", out_dir=out_dir, source=source)
    for name, hook in proj_hooks.items():
        flush(name, hook.value, "projector", out_dir=out_dir, source=source)
    for name, hook in dec_hooks.items():
        flush(name, hook.value, "decoder", out_dir=out_dir, source=source)

    # Self-consistency: teacher-forced gen0 logits vs incremental generate
    # scores[0]. They should match closely (the offline<->incremental encoder
    # equivalence is what the `stream` pass formally checks).
    if gen_out.scores and 0 <= g0 < logits.shape[1]:
        a = to_np(logits[:, g0, :]); b = to_np(gen_out.scores[0])
        print(f"  self-check gen0 teacher-forced vs generate scores[0]: "
              f"max|d|={np.abs(a - b).max():.3e} argmax_eq={int(a.argmax() == b.argmax())}")

    write_transcript(
        out_dir, text=text,
        source={**source, "hook": "generate.sequences"}, tokens=new_tokens,
    )
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    """Streaming-replay oracle: drive the model with a chunked audio generator.

    Replicates the canonical streaming loop from the HF model docs
    (model_doc/voxtral_realtime.md): the first chunk carries the prompt + is
    centered; subsequent chunks are non-centered and continue the conv padding
    cache + encoder/decoder KV caches. We capture the streaming transcript and
    the per-step logits at the first and ninth generated steps so Stage 4 can
    verify the streaming path is numerically equivalent to the offline path
    (intake risk #1).
    """
    import numpy as _np
    import torch

    configure_torch(args)
    processor, model = load_reference(args)
    model_id, _ = resolve_model(args.model)
    model_dtype = model_dtype_name(model)
    dtype = next(model.parameters()).dtype

    audio_path = Path(args.audio).expanduser().resolve()
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        raise SystemExit(f"Voxtral Realtime expects 16kHz audio; got {sr} Hz in {audio_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hop = processor.feature_extractor.hop_length
    win = processor.feature_extractor.win_length
    # Right-pad raw audio for the model's required right-pad tokens (docs).
    # num_right_pad_tokens is a method in transformers 5.10.2 (a property in
    # older checkouts); tolerate both.
    n_rpt = processor.num_right_pad_tokens
    n_rpt = n_rpt() if callable(n_rpt) else n_rpt
    audio = _np.pad(pcm, (0, n_rpt * processor.raw_audio_length_per_tok))

    first = processor(
        audio=audio[: processor.num_samples_first_audio_chunk],
        is_streaming=True, is_first_audio_chunk=True, return_tensors="pt",
    )
    num_delay_tokens = int(first["num_delay_tokens"]) if args.num_delay_tokens is None else args.num_delay_tokens
    first = {k: (v.to(args.device, dtype=dtype) if (hasattr(v, "to") and torch.is_floating_point(v))
                 else (v.to(args.device) if hasattr(v, "to") else v))
             for k, v in first.items()}

    def feature_gen():
        yield first["input_features"]
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        start_idx = mel_frame_idx * hop - win // 2
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            chunk = processor(
                audio=audio[start_idx:end_idx],
                is_streaming=True, is_first_audio_chunk=False, return_tensors="pt",
            )
            yield chunk["input_features"].to(args.device, dtype=dtype)
            mel_frame_idx += processor.audio_length_per_tok
            start_idx = mel_frame_idx * hop - win // 2

    source = make_source(
        args=args, model_id=model_id, audio_path=audio_path, n_samples=pcm.size,
        sample_rate=sr, model_dtype=model_dtype, num_delay_tokens=num_delay_tokens,
        mode="streaming",
    )

    with torch.inference_mode():
        gen_out = model.generate(
            input_ids=first["input_ids"],
            input_features=feature_gen(),
            num_delay_tokens=first["num_delay_tokens"],
            do_sample=False, num_beams=1,
            return_dict_in_generate=True, output_scores=True,
        )

    prompt_len = first["input_ids"].shape[1]
    new_tokens = gen_out.sequences[0, prompt_len:].tolist()
    text = decode_tokens(processor, new_tokens)
    print(f"streaming transcript: {text!r}")
    print(f"  prompt_len={prompt_len} generated={len(new_tokens)} steps={len(gen_out.scores or [])}")

    if gen_out.scores:
        flush("stream.logits_raw", gen_out.scores[0], "stream", out_dir=out_dir, source=source)
        if len(gen_out.scores) > 8:
            flush("stream.logits_raw.gen8", gen_out.scores[8], "stream", out_dir=out_dir, source=source)

    write_transcript(
        out_dir, text=text,
        source={**source, "hook": "generate.sequences"}, tokens=new_tokens,
    )
    print(f"wrote transcript: {out_dir / 'transcript.json'}")
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    """No-op alias.

    validate.py runs both `encoder` and `decode` subcommands per family dumper.
    Voxtral Realtime captures every tensor (enc.*, proj.*, dec.*) in a single
    decode pass, so the encoder pass is intentionally empty.
    """
    Path(args.out).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--audio", required=True, help="path to audio file (16kHz)")
    p.add_argument("--out", required=True, help="output directory for dumps")
    p.add_argument(
        "--language", default="auto",
        help="accepted for a uniform CLI but ignored: the realtime processor "
             "is auto-detect only (TranscriptionRequest language=None).",
    )
    p.add_argument("--num-delay-tokens", type=int, default=None,
                   help="override delay tokens (default: processor value, config default 6 = 480ms)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    p.add_argument("--torch-threads", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    encoder = sub.add_parser("encoder", help="no-op (decode pass dumps every tensor)")
    add_common_args(encoder)
    encoder.set_defaults(func=cmd_encoder)

    decode = sub.add_parser("decode", help="offline whole-clip encoder+projector+LM dump + transcript")
    add_common_args(decode)
    decode.add_argument("--enc-blocks", type=int, nargs="*", default=[],
                        help="extra encoder block indices (first/mid/last always included)")
    decode.add_argument("--dec-blocks", type=int, nargs="*", default=[],
                        help="extra LM block indices (first/mid/last always included)")
    decode.set_defaults(func=cmd_decode)

    stream = sub.add_parser("stream", help="streaming-replay transcript + per-step logits")
    add_common_args(stream)
    stream.set_defaults(func=cmd_stream)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
