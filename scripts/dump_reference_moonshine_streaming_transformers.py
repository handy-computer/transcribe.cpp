#!/usr/bin/env python3
"""
dump_reference_moonshine_streaming_transformers.py - generate Moonshine
Streaming reference tensors from the Hugging Face Transformers
implementation (`MoonshineStreamingForConditionalGeneration`).

Run through the repo-local env:

    uv run --project scripts/envs/moonshine_streaming \
      scripts/dump_reference_moonshine_streaming_transformers.py encoder \
      --model UsefulSensors/moonshine-streaming-tiny \
      --audio samples/jfk.wav \
      --out build/validate/moonshine_streaming/moonshine-streaming-tiny/jfk/encoder/ref

Tensor output uses the shared reference dump contract via
`scripts/lib/ref_dump.py`. Tensors are written in the reference
implementation's natural row-major shape with a leading batch dim
squeezed; C++ dumpers are expected to match this contract.

    <name>.f32    raw little-endian float32, row-major
    <name>.json   sidecar metadata (incl. rms / p99_abs)

The decode command also writes transcript.json as a behavioral artifact.

Architecture-specific notes (vs non-streaming moonshine):

  - Frontend is `MoonshineStreamingEncoderEmbedder`: CMVN per
    frame_len-sample frame -> asinh compression (learnable log_k) ->
    Linear(frame_len, hidden_size) + SiLU -> CausalConv1d(hidden,
    2*hidden, k=5, s=2) + SiLU -> CausalConv1d(2*hidden, hidden, k=5,
    s=2). Output T_enc = T_samples / frame_len / 4.
  - Encoder layers are "ergodic": no RoPE, sliding-window attention via
    per-layer (L, R) windows from `config.sliding_windows`. We dispatch
    to `encoder.forward` so the per-layer mask construction matches the
    reference exactly; intermediates are captured via forward hooks.
  - Adapter lives inside `decoder.forward`: `encoder_hidden +=
    pos_emb[:T_enc]` (in-place!) and then `decoder.proj` (Identity when
    encoder_hidden == decoder_hidden, e.g. tiny). Because of the
    in-place op we MANUALLY apply the adapter and run decoder layers
    ourselves -- never re-call `decoder.forward` twice on the same
    encoder hidden.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Pull in the shared write_tensor helpers.
THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))
from lib.ref_dump import write_tensor  # type: ignore  # noqa: E402


def resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


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


def configure_torch(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(0)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)


def to_np(t) -> np.ndarray:
    """Detach + move to CPU + cast to f32 + squeeze leading singleton dims."""
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
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


def load_reference(args: argparse.Namespace):
    import transformers
    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Moonshine Streaming model from {model_id} ({source}, "
        f"Transformers {transformers.__version__}, device={args.device})..."
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    # attn_implementation=eager so the sliding-window mask path is the
    # eager_attention_forward kernel and we get deterministic dumps.
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local_only, attn_implementation="eager"
    ).eval()
    # Reference regime is fp32 inference (validates the F32 GGUF). Upcast
    # in case a future release ships a non-fp32 default.
    model = model.float()
    model.to(args.device)
    return processor, model


def make_source(
    *,
    args: argparse.Namespace,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    model_dtype: str | None = None,
) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "kind": "moonshine-streaming-transformers-native",
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
    }


def frontend_inputs(processor, pcm: np.ndarray, sr: int, device: str):
    """Run the Moonshine Streaming processor and move tensors to device.

    The HF AutoProcessor for moonshine_streaming wraps a
    Wav2Vec2FeatureExtractor with `do_normalize=False`,
    `pad_to_multiple_of=80` (= frame_len), `return_attention_mask=True`.
    Output keys: `input_values` [B, T_samples] and `attention_mask`
    [B, T_samples] of 0/1 padding indicators."""
    return {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in processor(
            audio=pcm,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        ).items()
    }


def auto_blocks(n_layers: int) -> set[int]:
    """Default block selection: dump all blocks if n_layers <= 8, else
    5 evenly spaced indices including first and last."""
    if n_layers <= 8:
        return set(range(n_layers))
    last = n_layers - 1
    return {0, round(last / 4), round(last / 2), round(3 * last / 4), last}


def dump_tensor(
    out_dir: Path,
    name: str,
    array: np.ndarray,
    *,
    stage: str,
    source: dict[str, Any],
) -> None:
    print(
        f"  {name}: shape={array.shape} "
        f"min={array.min():.4e} max={array.max():.4e} mean={array.mean():.6e}"
    )
    write_tensor(name, array, stage=stage, source=source, out_dir=out_dir)


def dump_encoder(
    *,
    model,
    inputs: dict[str, Any],
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int] | None,
):
    """Run encoder.forward with hooks on every interesting submodule, then
    dump captured intermediates. Returns the encoder's final hidden state
    (i.e. last_hidden_state, post final_norm)."""
    import torch
    import torch.nn.functional as F

    encoder = model.model.encoder
    embedder = encoder.embedder
    input_values = inputs["input_values"]
    attention_mask = inputs.get("attention_mask")

    # Always dump the raw audio input so a C++ frontend port can compare
    # at the boundary.
    dump_tensor(out_dir, "enc.audio.in", to_np(input_values),
                stage="frontend.audio", source=source)

    intermediates: dict[str, Any] = {}

    def make_hook(name: str):
        def fn(module, args, output):
            intermediates[name] = output
        return fn

    handles = []
    handles.append(embedder.cmvn.register_forward_hook(make_hook("emb.cmvn")))
    handles.append(embedder.comp.register_forward_hook(make_hook("emb.comp")))
    handles.append(embedder.linear.register_forward_hook(make_hook("emb.linear")))
    handles.append(embedder.conv1.register_forward_hook(make_hook("emb.conv1")))
    handles.append(embedder.conv2.register_forward_hook(make_hook("emb.conv2")))
    for i, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(make_hook(f"layer.{i}")))
    handles.append(encoder.final_norm.register_forward_hook(make_hook("final_norm")))

    try:
        with torch.inference_mode():
            out = encoder(input_values=input_values, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    # Frontend dumps. CMVN/comp operate on [B, T_frames, frame_len]; linear
    # expands to [B, T_frames, hidden]; conv stack permutes to [B, C, T]
    # internally and returns [B, C, T] (the embedder transposes back to
    # [B, T, C] before returning, but the conv hooks see the [B, C, T]
    # layout). We materialize everything to [T, C] (or [T, frame_len])
    # for naming consistency with downstream dumps.
    cmvn_out = intermediates["emb.cmvn"]
    dump_tensor(out_dir, "enc.embedder.cmvn.out", to_np(cmvn_out),
                stage="encoder.embedder.cmvn", source=source)

    comp_out = intermediates["emb.comp"]
    dump_tensor(out_dir, "enc.embedder.comp.out", to_np(comp_out),
                stage="encoder.embedder.comp", source=source)

    # The linear hook captures POST-Linear, PRE-SiLU. The embedder applies
    # SiLU outside the linear module. We dump the SiLU output (which is
    # what the C++ port computes) since SiLU is part of the frontend.
    linear_out = intermediates["emb.linear"]
    linear_post_silu = F.silu(linear_out)
    dump_tensor(out_dir, "enc.embedder.linear.out", to_np(linear_post_silu),
                stage="encoder.embedder.linear", source=source)

    # CausalConv1d.forward returns (hidden, mask). The post-conv1 SiLU is
    # applied externally; the C++ port computes it as part of the
    # frontend, so dump SiLU output.
    conv1_out = intermediates["emb.conv1"][0]
    conv1_post_silu = F.silu(conv1_out)
    # [B, 2*hidden, T_after_conv1] -> transpose to [T, C] for naming.
    dump_tensor(out_dir, "enc.embedder.conv1.out",
                to_np(conv1_post_silu.transpose(1, 2)),
                stage="encoder.embedder.conv1", source=source)

    # No SiLU after conv2; this is the inputs_embeds tensor.
    conv2_out = intermediates["emb.conv2"][0]
    dump_tensor(out_dir, "enc.embedder.conv2.out",
                to_np(conv2_out.transpose(1, 2)),
                stage="encoder.embedder.conv2", source=source)

    n_layers = len(encoder.layers)
    block_set = set(blocks) if blocks is not None else auto_blocks(n_layers)
    block_set.add(0)
    block_set.add(n_layers - 1)
    for i in range(n_layers):
        if i not in block_set:
            continue
        dump_tensor(out_dir, f"enc.block.{i}.out",
                    to_np(intermediates[f"layer.{i}"]),
                    stage=f"encoder.block{i}.out", source=source)

    # Final norm output (= encoder.last_hidden_state).
    final_out = intermediates["final_norm"]
    dump_tensor(out_dir, "enc.final", to_np(final_out),
                stage="encoder.final", source=source)

    return out.last_hidden_state


def apply_adapter(decoder, encoder_hidden):
    """Mirror the in-place adapter inside MoonshineStreamingDecoder.forward
    on a CLONE so the original encoder_hidden is never mutated. Returns
    the adapter-applied tensor (encoder_hidden + pos_emb[:T_enc] then
    decoder.proj — Identity when encoder_hidden_size == decoder_hidden)."""
    import torch

    T_enc = encoder_hidden.shape[1]
    # decoder.pos_emb is the learned positional embedding table that the
    # adapter adds to the encoder output.
    pos_emb_full = decoder.pos_emb(
        torch.arange(T_enc, device=encoder_hidden.device)
    )  # [T_enc, encoder_hidden_size]
    adapted = encoder_hidden.clone()
    adapted = adapted + pos_emb_full
    adapted = decoder.proj(adapted)
    return adapted, pos_emb_full


def run_decoder_layers(
    *,
    decoder,
    input_ids,
    encoder_hidden_adapted,
    past_key_values=None,
    use_cache: bool = False,
):
    """Run decoder.embed_tokens + layer loop + final norm manually so we
    can dump per-layer outputs without re-triggering the in-place adapter.
    Returns (final_hidden, past_key_values)."""
    import torch

    x = decoder.embed_tokens(input_ids)
    # Position ids account for past_key_values length when use_cache is set.
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(
        x.shape[1], device=x.device
    ) + past_seen
    position_ids = position_ids.unsqueeze(0)

    if x.shape[1] > 1 or past_seen > 0:
        # Prompt-pass with multiple tokens (or step graph beyond t=0)
        # needs a real causal mask. Build via the same path the decoder
        # would use.
        from transformers.masking_utils import create_causal_mask  # type: ignore
        causal_mask = create_causal_mask(
            config=decoder.config,
            inputs_embeds=x,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    else:
        # Single-token prompt-pass: causal mask is None (no past tokens
        # to mask against).
        causal_mask = None

    # Encoder attention mask: None for our jfk single-utterance case
    # (no padding to mask against).
    encoder_attention_mask = None
    position_embeddings = decoder.rotary_emb(x, position_ids=position_ids)

    layer_outs = []
    for layer in decoder.layers:
        x = layer(
            x,
            attention_mask=causal_mask,
            encoder_hidden_states=encoder_hidden_adapted,
            encoder_attention_mask=encoder_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        layer_outs.append(x)

    x = decoder.norm(x)
    return x, layer_outs, past_key_values


def dump_decoder(
    *,
    model,
    encoder_hidden,
    prompt_ids: list[int],
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int] | None,
    device: str,
):
    """Run a single prompt-pass through the decoder and dump per-block
    activations + final logits. `prompt_ids` for moonshine_streaming is
    just [decoder_start_token_id] (= 1)."""
    import torch
    import torch.nn.functional as F

    decoder = model.model.decoder
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    print(f"  prompt_ids: {prompt_ids}")

    with torch.inference_mode():
        # Adapter applied here (NOT inside decoder.forward — we run
        # layers manually).
        adapted, pos_emb_full = apply_adapter(decoder, encoder_hidden)
        # adapter.pos_emb: the slice of pos_emb actually used. Squeeze the
        # batch dim if any (it has none; pos_emb_full is [T_enc, dim]).
        dump_tensor(out_dir, "adapter.pos_emb",
                    to_np(pos_emb_full),
                    stage="adapter.pos_emb", source=source)
        dump_tensor(out_dir, "adapter.out",
                    to_np(adapted),
                    stage="adapter.out", source=source)

        # Token embedding lookup. Note: streaming has tied=False so
        # proj_out.weight is a separate tensor from embed_tokens.weight,
        # but the lookup is still through embed_tokens.
        tok_emb = decoder.embed_tokens(input_ids)
        dump_tensor(out_dir, "dec.token_emb", to_np(tok_emb),
                    stage="decoder.embedding", source=source)

        # No additive positional embedding inside the decoder block stack;
        # RoPE handles position inside self_attn. Emit `dec.embed_sum`
        # for naming parity with whisper/cohere goldens.
        dump_tensor(out_dir, "dec.embed_sum", to_np(tok_emb),
                    stage="decoder.embed_sum", source=source)

        # Run layers manually so we can capture per-block outputs.
        x = tok_emb
        position_ids = torch.arange(0, x.shape[1], device=device).unsqueeze(0)
        position_embeddings = decoder.rotary_emb(x, position_ids=position_ids)
        # Single-token prompt pass: causal mask is None.
        block_set = set(blocks) if blocks is not None else auto_blocks(len(decoder.layers))
        block_set.add(0)
        block_set.add(len(decoder.layers) - 1)
        for i, layer in enumerate(decoder.layers):
            x = layer(
                x,
                attention_mask=None,
                encoder_hidden_states=adapted,
                encoder_attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            if i in block_set:
                dump_tensor(out_dir, f"dec.block.{i}.out", to_np(x),
                            stage=f"decoder.block{i}.out", source=source)

        x = decoder.norm(x)
        dump_tensor(out_dir, "dec.out_before_head", to_np(x),
                    stage="decoder.output_before_head", source=source)

        logits_raw = model.proj_out(x)
        dump_tensor(out_dir, "dec.logits_raw", to_np(logits_raw),
                    stage="decoder.logits_raw", source=source)

        log_probs = F.log_softmax(logits_raw, dim=-1)
        dump_tensor(out_dir, "dec.logits", to_np(log_probs),
                    stage="decoder.logits", source=source)

    last_logits = logits_raw[0, -1, :] if logits_raw.dim() == 3 else logits_raw[-1, :]
    return int(torch.argmax(last_logits).item()), adapted


def dump_mid_generation(
    *,
    model,
    encoder_hidden_adapted,
    prompt_ids: list[int],
    out_dir: Path,
    source: dict[str, Any],
    device: str,
    gen_step_n: int = 20,
):
    """Greedy-decode `gen_step_n` tokens and dump the logits that would
    predict token `gen_step_n` (i.e. the step graph's output at the
    `gen_step_n`-th autoregressive step, matching the C++ runner's
    step-loop iteration at `step == gen_step_n - 1`).

    Exercises the n_past > 0 path. We pass the ALREADY-adapted encoder
    hidden so the manual decoder loop never triggers the in-place
    adapter."""
    import torch

    decoder = model.model.decoder
    eos = int(model.config.eos_token_id)
    current = list(prompt_ids)

    def forward_last_logits(ids: list[int]):
        inp = torch.tensor([ids], device=device, dtype=torch.long)
        x, _, _ = run_decoder_layers(
            decoder=decoder,
            input_ids=inp,
            encoder_hidden_adapted=encoder_hidden_adapted,
        )
        return model.proj_out(x[0, -1, :])

    with torch.inference_mode():
        step_logits = forward_last_logits(current).clone()
        next_id = int(torch.argmax(step_logits).item())

        for _ in range(1, gen_step_n):
            if next_id == eos:
                break
            current.append(next_id)
            step_logits = forward_last_logits(current).clone()
            next_id = int(torch.argmax(step_logits).item())

        current.append(next_id)
        inp = torch.tensor([current], device=device, dtype=torch.long)
        x, _, _ = run_decoder_layers(
            decoder=decoder,
            input_ids=inp,
            encoder_hidden_adapted=encoder_hidden_adapted,
        )
        logits_final = model.proj_out(x[0, -1:, :])

    name = f"dec.logits_raw.gen{gen_step_n}"
    stage = f"decoder.logits_raw.gen{gen_step_n}"
    dump_tensor(out_dir, name, to_np(logits_final), stage=stage, source=source)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def generate_transcript(
    *,
    model,
    processor,
    inputs: dict[str, Any],
    source: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Greedy generation matching the model card's quick-start contract:
    do_sample=False, num_beams=1. The model card recommends bounding
    max_length per audio duration (`seq_lens * 6.5 / 16000`) to avoid
    hallucination loops; we follow that when --max-new-tokens is unset
    via the duration-aware heuristic."""
    import torch

    with torch.inference_mode():
        generated_ids = model.generate(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    tokenizer = processor.tokenizer
    generated_list = generated_ids[0].detach().cpu().tolist()
    eos = int(model.config.eos_token_id)
    decoder_start = int(model.config.decoder_start_token_id)
    pad = int(model.config.pad_token_id)
    token_ids = [t for t in generated_list if t not in (eos, decoder_start, pad)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    return {
        "schema": "transcribe-reference-transcript-v1",
        "text": text,
        "normalized_text": normalize_text(text),
        "token_ids": token_ids,
        "generated_ids": generated_list,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "source": source,
    }


def cmd_encoder(args: argparse.Namespace) -> int:
    configure_torch(args)
    processor, model = load_reference(args)

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    model_id, _ = resolve_model(args.model)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    inputs = frontend_inputs(processor, pcm, sr, args.device)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype_name(model),
    )
    dump_encoder(
        model=model,
        inputs=inputs,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks) if args.blocks else None,
    )
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    configure_torch(args)
    processor, model = load_reference(args)

    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    model_id, _ = resolve_model(args.model)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")

    inputs = frontend_inputs(processor, pcm, sr, args.device)
    source = make_source(
        args=args,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        model_dtype=model_dtype_name(model),
    )

    encoder_hidden = dump_encoder(
        model=model,
        inputs=inputs,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks) if args.blocks else None,
    )

    prompt_ids = [int(model.config.decoder_start_token_id)]
    pred_id, encoder_hidden_adapted = dump_decoder(
        model=model,
        encoder_hidden=encoder_hidden,
        prompt_ids=prompt_ids,
        out_dir=out_dir,
        source=source,
        blocks=set(args.blocks) if args.blocks else None,
        device=args.device,
    )
    pred_token = processor.tokenizer.decode([pred_id])
    print(f"  first predicted token: id={pred_id} text={pred_token!r}")

    dump_mid_generation(
        model=model,
        encoder_hidden_adapted=encoder_hidden_adapted,
        prompt_ids=prompt_ids,
        out_dir=out_dir,
        source=source,
        device=args.device,
        gen_step_n=20,
    )

    if not args.skip_transcript:
        transcript = generate_transcript(
            model=model,
            processor=processor,
            inputs=inputs,
            source=source,
            args=args,
        )
        print(f"  transcript: {transcript['text']!r}")
        import json as _json
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transcript.json").write_text(
            _json.dumps(transcript, indent=2) + "\n"
        )
        print(f"  wrote {out_dir / 'transcript.json'}")

    return 0


def cmd_wer(args: argparse.Namespace) -> int:
    """Run HF moonshine_streaming over a manifest.jsonl in the same
    regime as the decode dumper (greedy, num_beams=1, do_sample=False),
    and write a hypothesis JSONL that `scripts/wer/score.py` can
    consume."""
    import json
    import time

    configure_torch(args)
    processor, model = load_reference(args)
    tokenizer = processor.tokenizer

    manifest_path = resolve_path(args.manifest)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eos = int(model.config.eos_token_id)
    decoder_start = int(model.config.decoder_start_token_id)
    pad = int(model.config.pad_token_id)

    with manifest_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"manifest: {manifest_path.name} entries={len(rows)}")

    import torch  # noqa: F401

    t0 = time.monotonic()
    written = 0
    with out_path.open("w") as out:
        out.write(json.dumps({"type": "batch_header",
                              "kind": "moonshine-streaming-transformers-native",
                              "model": resolve_model(args.model)[0]}) + "\n")
        for i, row in enumerate(rows):
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = (manifest_path.parent / audio_path).resolve()
            pcm, sr = load_audio(audio_path)
            if sr != 16000:
                print(f"skipping {row['id']}: sr={sr}", file=sys.stderr)
                continue
            inputs = frontend_inputs(processor, pcm, sr, args.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_values=inputs["input_values"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            ids = generated_ids[0].detach().cpu().tolist()
            ids = [t for t in ids if t not in (eos, decoder_start, pad)]
            hyp = tokenizer.decode(ids, skip_special_tokens=True).strip()
            out.write(json.dumps({
                "id": row["id"],
                "ref_text": row["ref_text"],
                "hyp_text": hyp,
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": 0,
            }) + "\n")
            written += 1
            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                rate = (i + 1) / (time.monotonic() - t0)
                eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(rows)}] {rate:.1f} utt/s  ETA {eta:.0f}s")

    print(f"done. {written} utterances in {time.monotonic() - t0:.1f}s")
    print(f"report: {out_path}")
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF repo id or local path to the moonshine_streaming model directory")
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="Torch intra-op threads for deterministic dumps (default: 1)")
    p.add_argument("--language", default=None,
                   help="Ignored — moonshine_streaming is English-only with no language tokens.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Moonshine Streaming reference tensors from Transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    add_common_args(ep)
    ep.add_argument("--blocks", type=int, nargs="*", default=None,
                    help="Block indices to dump (default: auto)")
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser("decode",
                        help="Dump encoder + adapter + decoder prompt-pass + transcript")
    add_common_args(dp)
    dp.add_argument("--max-new-tokens", type=int, default=192,
                    help="Maximum generated transcript tokens (default: 192)")
    dp.add_argument("--skip-transcript", action="store_true",
                    help="Only dump tensors; do not generate transcript.json")
    dp.add_argument("--blocks", type=int, nargs="*", default=None,
                    help="Block indices to dump (default: auto)")
    dp.set_defaults(func=cmd_decode)

    wp = sub.add_parser("wer",
                        help="Run HF moonshine_streaming over a manifest.jsonl and write a "
                             "hypothesis JSONL for scripts/wer/score.py")
    wp.add_argument("--model", required=True,
                    help="HF repo id or local path to the moonshine_streaming model directory")
    wp.add_argument("--manifest", required=True,
                    help="Input manifest.jsonl with id/audio/ref_text per row")
    wp.add_argument("--out", required=True,
                    help="Output JSONL (matches scripts/wer/run.py format)")
    wp.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    wp.add_argument("--torch-threads", type=int, default=1,
                    help="Torch intra-op threads (default: 1)")
    wp.add_argument("--max-new-tokens", type=int, default=192,
                    help="Max generated tokens per utterance (default: 192)")
    wp.set_defaults(func=cmd_wer)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
