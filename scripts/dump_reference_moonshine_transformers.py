#!/usr/bin/env python3
"""
dump_reference_moonshine_transformers.py - generate Moonshine reference
tensors from the Hugging Face Transformers implementation
(`MoonshineForConditionalGeneration`).

Run through the repo-local Moonshine reference environment:

    uv run --project scripts/envs/moonshine \
      scripts/dump_reference_moonshine_transformers.py encoder \
      --model UsefulSensors/moonshine-tiny \
      --audio samples/jfk.wav \
      --out build/validate/moonshine/moonshine-tiny/jfk/encoder/ref

Tensor output uses the shared reference dump contract via
`scripts/lib/ref_dump.py`. Tensors are written in the reference
implementation's natural row-major shape with a leading batch dim
squeezed; C++ dumpers are expected to match this contract, not the
other way around.

    <name>.f32    raw little-endian float32, row-major
    <name>.json   sidecar metadata (incl. rms / p99_abs)

The decode command also writes transcript.json as a behavioral artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Pull in the shared write_tensor / write_transcript helpers. They live
# next to this script under scripts/lib/ref_dump.py; we add scripts/ to
# sys.path so the import is `from lib.ref_dump import ...` regardless of
# uv's working directory.
THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))
from lib.ref_dump import write_tensor, write_transcript  # type: ignore  # noqa: E402


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
    from transformers import AutoProcessor, MoonshineForConditionalGeneration

    model_id, local_only = resolve_model(args.model)
    source = "local path" if local_only else "HuggingFace"
    print(
        f"Loading Moonshine model from {model_id} ({source}, "
        f"Transformers {transformers.__version__}, device={args.device})..."
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local_only
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
        "kind": "moonshine-transformers-native",
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
    """Run the Moonshine processor and move tensors to device.

    The HF AutoProcessor for Moonshine is a Wav2Vec2FeatureExtractor with
    `do_normalize=False` and `feature_size=1` — i.e. it just batches the
    raw 16 kHz waveform. Output keys: `input_values` [B, T_samples] and
    `attention_mask` [B, T_samples]."""
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
    5 evenly spaced indices including first and last. The ≤8 threshold
    covers moonshine-tiny (6) and moonshine-base (8) without leaving
    middle blocks uncovered (Python's banker's rounding made `round(2.5)`
    skip block 3 in the older 5-sample heuristic for n_layers=6)."""
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
    import torch
    import torch.nn.functional as F

    encoder = model.model.encoder
    input_values = inputs["input_values"]

    # Raw-audio frontend input. Layout [n_samples] after squeeze.
    dump_tensor(out_dir, "enc.audio.in", to_np(input_values),
                stage="frontend.audio", source=source)

    with torch.inference_mode():
        # Mirror MoonshineEncoder.forward exactly. unsqueeze(1) gives
        # [B, 1, T_samples], then conv1 (k=127, s=64, no bias), tanh,
        # groupnorm, conv2 (k=7, s=3) + GELU, conv3 (k=3, s=2) + GELU.
        x = input_values.unsqueeze(1)
        x = F.tanh(encoder.conv1(x))
        # Pre-groupnorm dump captures the conv1 output before normalization.
        dump_tensor(out_dir, "enc.conv1.out", to_np(x.transpose(-2, -1)),
                    stage="encoder.conv1", source=source)

        x = encoder.groupnorm(x)
        dump_tensor(out_dir, "enc.groupnorm.out", to_np(x.transpose(-2, -1)),
                    stage="encoder.groupnorm", source=source)

        x = F.gelu(encoder.conv2(x))
        dump_tensor(out_dir, "enc.conv2.out", to_np(x.transpose(-2, -1)),
                    stage="encoder.conv2", source=source)

        x = F.gelu(encoder.conv3(x))
        # After conv3 the channel dim is `hidden_size`. Permute to
        # [B, T_enc, hidden] which is what the encoder layers consume.
        x = x.permute(0, 2, 1)
        dump_tensor(out_dir, "enc.conv3.out", to_np(x),
                    stage="encoder.conv3", source=source)

        # No additive positional embedding — RoPE is applied inside
        # self_attn. C++ uses ggml_rope_ext which computes cos/sin
        # internally; we don't dump them on the reference side because
        # there's no C++ counterpart to compare against (correctness is
        # verified transitively by enc.block.0.out).
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = encoder.rotary_emb(x, position_ids=position_ids)

        block_set = set(blocks) if blocks is not None else auto_blocks(len(encoder.layers))
        block_set.add(0)
        block_set.add(len(encoder.layers) - 1)
        for i, layer in enumerate(encoder.layers):
            x = layer(
                x,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            if i in block_set:
                dump_tensor(out_dir, f"enc.block.{i}.out", to_np(x),
                            stage=f"encoder.block{i}.out", source=source)

        x = encoder.layer_norm(x)
        dump_tensor(out_dir, "enc.final", to_np(x),
                    stage="encoder.final", source=source)

    return x


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
    activations + final logits. `prompt_ids` for moonshine is just
    [decoder_start_token_id] (= 1); there are no language/task tokens."""
    import torch

    decoder = model.model.decoder
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    print(f"  prompt_ids: {prompt_ids}")

    with torch.inference_mode():
        # Token embedding (also tied to the proj_out / lm_head).
        tok_emb = decoder.embed_tokens(input_ids)
        dump_tensor(out_dir, "dec.token_emb", to_np(tok_emb),
                    stage="decoder.embedding", source=source)

        # Moonshine has no separate additive positional embedding for the
        # decoder — RoPE handles position inside self_attn. Still emit
        # `dec.embed_sum` as the input-to-blocks tensor for naming
        # consistency with whisper/cohere goldens.
        x = tok_emb
        dump_tensor(out_dir, "dec.embed_sum", to_np(x),
                    stage="decoder.embed_sum", source=source)

        position_ids = torch.arange(0, x.shape[1], device=device).unsqueeze(0)
        position_embeddings = decoder.rotary_emb(x, position_ids=position_ids)

        # Causal mask is None for a single-token prompt-pass (T=1, no
        # attention positions to mask). The reference's `create_causal_mask`
        # returns None in this regime; layers handle the rest internally.
        block_set = set(blocks) if blocks is not None else auto_blocks(len(decoder.layers))
        block_set.add(0)
        block_set.add(len(decoder.layers) - 1)
        for i, layer in enumerate(decoder.layers):
            x = layer(
                x,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden,
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

        log_probs = torch.log_softmax(logits_raw, dim=-1)
        dump_tensor(out_dir, "dec.logits", to_np(log_probs),
                    stage="decoder.logits", source=source)

    last_logits = logits_raw[0, -1, :] if logits_raw.dim() == 3 else logits_raw[-1, :]
    return int(torch.argmax(last_logits).item())


def dump_mid_generation(
    *,
    model,
    encoder_hidden,
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

    Exercises the n_past > 0 path — the prompt-pass dumps only cover
    n_past == 0, so without this the KV cache update code has zero
    tensor-level coverage. Moonshine's generation_config has no
    suppress_tokens / begin_suppress_tokens, so this is plain greedy
    argmax until eos or `gen_step_n + 1` tokens emitted.
    """
    import torch

    decoder = model.model.decoder
    eos = int(model.config.eos_token_id)
    current = list(prompt_ids)

    def forward_last_logits(ids: list[int]):
        inp = torch.tensor([ids], device=device, dtype=torch.long)
        out = decoder(
            input_ids=inp,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        )
        return model.proj_out(out.last_hidden_state[0, -1, :])

    with torch.inference_mode():
        step_logits = forward_last_logits(current).clone()
        next_id = int(torch.argmax(step_logits).item())

        for _ in range(1, gen_step_n):
            if next_id == eos:
                # Reference EOS hit before reaching the requested gen step.
                # Fall back to last non-EOS state — the dump still exercises
                # n_past>0 because prior steps already grew the prefix.
                break
            current.append(next_id)
            step_logits = forward_last_logits(current).clone()
            next_id = int(torch.argmax(step_logits).item())

        # One more forward with [prompt + generated[0..gen_step_n-1]]
        # so we capture the pre-argmax logits that would pick
        # generated[gen_step_n]. This matches the C++ runner's step graph
        # output at step == gen_step_n - 1 after compute.
        current.append(next_id)
        inp = torch.tensor([current], device=device, dtype=torch.long)
        out = decoder(
            input_ids=inp,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        )
        logits_final = model.proj_out(out.last_hidden_state[0, -1:, :])

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
    `do_sample=False, num_beams=1`, default max_length=194 from the
    generation_config. Moonshine has no language/task tokens, so no
    forced_decoder_ids."""
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
    # Moonshine ships no tokenizer_config.json, so tokenizer.eos_token_id
    # is None even though the model config sets eos_token_id=2. Use the
    # model config for both ends of the filter.
    eos = int(model.config.eos_token_id)
    decoder_start = int(model.config.decoder_start_token_id)
    token_ids = [t for t in generated_list if t not in (eos, decoder_start)]
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
    pred_id = dump_decoder(
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
        encoder_hidden=encoder_hidden,
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
        # Use shared write_transcript helper for the canonical contract
        # but keep the richer payload (normalized_text, token_ids, etc.).
        # write_transcript only persists text + tokens + source; we want
        # the full payload, so write directly to keep symmetry with
        # whisper/cohere golden transcripts.
        import json as _json
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transcript.json").write_text(
            _json.dumps(transcript, indent=2) + "\n"
        )
        print(f"  wrote {out_dir / 'transcript.json'}")

    return 0


def cmd_wer(args: argparse.Namespace) -> int:
    """Run HF moonshine over a manifest.jsonl in the same regime as the
    decode dumper (greedy, num_beams=1, do_sample=False, max_new_tokens
    matching the dumper default), and write a hypothesis JSONL that
    `scripts/wer/score.py` can consume. Output schema mirrors
    `scripts/wer/run.py`'s per-utterance lines (id, ref_text, hyp_text,
    timing fields kept for parity though we leave them at 0)."""
    import json
    import time

    configure_torch(args)
    t_load = time.monotonic()
    processor, model = load_reference(args)
    load_ms = round((time.monotonic() - t_load) * 1000, 1)
    tokenizer = processor.tokenizer

    manifest_path = resolve_path(args.manifest)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eos = int(model.config.eos_token_id)
    decoder_start = int(model.config.decoder_start_token_id)

    with manifest_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if getattr(args, "limit", 0):
        rows = rows[: args.limit]
    print(f"manifest: {manifest_path.name} entries={len(rows)}")
    if args.language:
        print(f"language: {args.language} (recorded in header; no effect on generation)")

    import torch  # noqa: F401  (used inside generate_transcript)

    t0 = time.monotonic()
    written = 0
    header = {
        "type": "batch_header",
        "kind": "moonshine-transformers-native",
        "model": resolve_model(args.model)[0],
        "load_ms": load_ms,
    }
    if args.language:
        header["language"] = args.language
    with out_path.open("w") as out:
        out.write(json.dumps(header) + "\n")
        for i, row in enumerate(rows):
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = (manifest_path.parent / audio_path).resolve()
            pcm, sr = load_audio(audio_path)
            if sr != 16000:
                print(f"skipping {row['id']}: sr={sr}", file=sys.stderr)
                continue
            t_utt = time.monotonic()
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
            ids = [t for t in ids if t not in (eos, decoder_start)]
            hyp = tokenizer.decode(ids, skip_special_tokens=True).strip()
            latency_ms = round((time.monotonic() - t_utt) * 1000, 1)
            out.write(json.dumps({
                "id": row["id"],
                "ref_text": row["ref_text"],
                "hyp_text": hyp,
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": latency_ms,
                "latency_ms": latency_ms,
            }, ensure_ascii=False) + "\n")
            written += 1
            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                rate = (i + 1) / (time.monotonic() - t0)
                eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(rows)}] {rate:.1f} utt/s  ETA {eta:.0f}s",
                      flush=True)

    print(f"done. {written} utterances in {time.monotonic() - t0:.1f}s")
    print(f"report: {out_path}")
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF repo id or local path to the Moonshine model directory")
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )
    # The Moonshine architecture has no language hint tokens, so this
    # flag does not affect generation. It is recorded in the WER report
    # header so scripts/wer/score.py can auto-route the metric
    # (WER vs CER) and the text normalizer for sibling variants like
    # moonshine-tiny-zh / moonshine-base-ko.
    p.add_argument(
        "--language",
        default=None,
        help="BCP-47 code recorded in the report header for downstream "
             "score.py routing. Has no effect on generation (Moonshine "
             "has no language tokens). Required for non-English variants.",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Moonshine reference tensors from Transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    add_common_args(ep)
    ep.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=None,
        help="Block indices to dump (default: auto — all if <=5 layers, "
             "else 5 evenly spaced indices including first and last)",
    )
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser("decode",
                        help="Dump encoder + decoder prompt-pass logits + transcript")
    add_common_args(dp)
    dp.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Maximum generated transcript tokens (default: 192, just under "
             "max_position_embeddings=194)",
    )
    dp.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Only dump tensors; do not generate transcript.json",
    )
    dp.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=None,
        help="Block indices to dump (default: auto — all if <=5 layers, "
             "else 5 evenly spaced indices including first and last)",
    )
    dp.set_defaults(func=cmd_decode)

    wp = sub.add_parser("wer",
                        help="Run HF moonshine over a manifest.jsonl and write a "
                             "hypothesis JSONL for scripts/wer/score.py")
    wp.add_argument("--model", required=True,
                    help="HF repo id or local path to the Moonshine model directory")
    wp.add_argument("--manifest", required=True,
                    help="Input manifest.jsonl with id/audio/ref_text per row")
    wp.add_argument("--out", required=True,
                    help="Output JSONL (matches scripts/wer/run.py format)")
    wp.add_argument("--device", default="cpu",
                    help="Torch device (default: cpu; pass 'mps' on Apple)")
    wp.add_argument("--torch-threads", type=int, default=1,
                    help="Torch intra-op threads (default: 1)")
    wp.add_argument("--max-new-tokens", type=int, default=192,
                    help="Max generated tokens per utterance (default: 192)")
    wp.add_argument("--language", default=None,
                    help="BCP-47 code recorded in the report header for "
                         "score.py auto-routing. No effect on generation.")
    wp.add_argument("--limit", type=int, default=0,
                    help="Process only the first N utterances (0 = all).")
    wp.set_defaults(func=cmd_wer)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
