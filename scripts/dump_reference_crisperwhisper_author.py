#!/usr/bin/env python3
"""
dump_reference_crisperwhisper_author.py - generate CrisperWhisper 2.0
reference tensors from the publisher's own package
(`crisperwhisper`, MIT, https://github.com/nyrahealth/CrisperWhisper),
transformers backend.

Run through the repo-local reference environment:

    uv run --project scripts/envs/crisperwhisper \
      scripts/dump_reference_crisperwhisper_author.py decode \
      --model nyralabs/CrisperWhisper2.0_small \
      --audio samples/jfk.wav \
      --out build/validate/crisperwhisper/crisperwhisper-2.0-small/jfk/ref

Why the author package and not plain transformers
-------------------------------------------------
The graph is stock HF Whisper — the package ships no modeling code. What
it owns is the decode contract, and that is what makes these dumps a
valid oracle:

  * ``TransformersEngine.__init__`` clears ``forced_decoder_ids`` and
    ``begin_suppress_tokens`` on both config and generation_config, and
    loads with ``attn_implementation="eager"``. Letting HF apply its own
    forced ids would diverge on the first decoded token.
  * ``PromptBuilder`` produces the real decoder prefix:
    ``[verbatim_1..5]`` (or ``[intended_1..5]``) BEFORE the Whisper
    ``<|sot|> <|lang|> <|transcribe|> <|notimestamps|>`` prefix, with no
    ``<|startofprev|>`` wrapper.
  * ``generation_config.suppress_tokens`` is resolved to an explicit id
    list per call rather than left to HF defaults.

Reference regime
----------------
Weights ship BF16; this dumper runs the engine at ``compute_type=
"float32"`` (``--compute-dtype``, default ``f32``), i.e. BF16 storage
upcast to F32 compute, and the Stage 4 numerical gate pairs it with an
F32 GGUF built from the same weights (BF16 -> F32 is lossless, so the
two hold identical values).

An earlier version of this note claimed F32 compute "is exactly what
ggml does with a BF16 GGUF". That is false, and Stage 4 measured all
three pairings on ``enc.block.6.out`` (whose reference ``max|x|`` is
~800 — Whisper's outlier channels — against a p99 of 4.3):

    F32 oracle  vs F32 GGUF   1.66e-02   <- the gate
    F32 oracle  vs BF16 GGUF  12.76
    BF16 oracle vs BF16 GGUF  71.13

BF16-on-both-sides is the WORST pairing because the two stacks round in
different places. ggml keeps the whole graph in F32 and rounds only
``mul_mat``'s src1 per dot product (``vec_dot_type = GGML_TYPE_BF16``,
``from_float = ggml_cpu_fp32_to_bf16``); torch at ``dtype=bfloat16``
stores EVERY activation in BF16, including the residual stream and the
embedding table. ggml is the higher-precision of the two, so moving the
oracle to BF16 moves it further away, and it also destroys the exact
0.0 that ``dec.embed_sum`` / ``dec.token_emb`` / ``enc.pos_emb`` must
hold (a BF16 oracle re-rounds the embedding table).

``--compute-dtype bf16`` reproduces that experiment. It is not the
gate.

Rewind features OFF
-------------------
``hallucination_mitigation``, ``early_eot_recovery``, and
``temperature_fallback`` all default to True in the publisher's
``transcribe()`` and all three can rewind and re-decode, which makes
greedy-argmax parity untestable. This dumper never enables them; it
drives ``engine.generate_with_repair(..., hallucination_mitigation=
False)``, which reduces to a plain greedy ``generate``.

Tensor output uses the shared reference dump contract
(``scripts/lib/ref_dump.py``): ``<name>.f32`` raw little-endian float32
row-major, plus a ``<name>.json`` sidecar carrying ``rms`` / ``p99_abs``
for the Stage 2 magnitude-aware tolerance derivation.

The decode command additionally writes ``transcript.json`` (behavioral
artifact consumed by validate.py) and ``word_timestamps.json`` (the
reference Viterbi aligner's output; not a tensor sidecar).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.ref_dump import write_tensor, write_transcript  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE


def resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


def resolve_model(raw: str) -> tuple[str, bool]:
    local = Path(raw).expanduser().resolve()
    if local.is_dir():
        return str(local), True
    return raw, False


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
    """Detach to a contiguous float32 numpy array, squeezing leading batch dims."""
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


def auto_blocks(n_layers: int) -> set[int]:
    """Default block selection: all if <=5 layers, else 5 evenly spaced
    indices including first and last. Keeps tensor coverage constant
    across variant sizes (small=12, medium=24, large=32, turbo enc 32 /
    dec 4) without per-variant --blocks args."""
    if n_layers <= 5:
        return set(range(n_layers))
    last = n_layers - 1
    return {0, round(last / 4), round(last / 2), round(3 * last / 4), last}


def resolve_blocks(requested: set[int] | None, n_layers: int) -> set[int]:
    block_set = set(requested) if requested is not None else auto_blocks(n_layers)
    block_set.add(0)
    block_set.add(n_layers - 1)
    return {b for b in block_set if 0 <= b < n_layers}


class Taps:
    """Forward-hook recorder.

    We deliberately capture through hooks on the real
    ``WhisperEncoder.forward`` / ``WhisperDecoder.forward`` rather than
    re-walking the layer list by hand. Under ``attn_implementation=
    "eager"`` (which this model requires, for cross-attention capture)
    the decoder's causal mask is built inside ``WhisperDecoder.forward``;
    hand-walking layers with ``attention_mask=None`` would silently give
    the prompt pass full bidirectional attention.
    """

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self._handles: list[Any] = []

    def out(self, module, key: str) -> None:
        def fn(_mod, _inp, out):
            self.store[key] = out[0] if isinstance(out, tuple) else out

        self._handles.append(module.register_forward_hook(fn))

    def inp(self, module, key: str) -> None:
        def fn(_mod, inp):
            self.store[key] = inp[0]

        self._handles.append(module.register_forward_pre_hook(fn))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "Taps":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()


def load_engine(args: argparse.Namespace):
    from crisperwhisper.transformers_engine import TransformersEngine
    from crisperwhisper.version import detect_model_version

    model_id, local_only = resolve_model(args.model)
    origin = "local path" if local_only else "HuggingFace"

    version = detect_model_version(model_id)
    if version != 2:
        raise SystemExit(
            f"error: {model_id} detects as CrisperWhisper v{version}. This dumper "
            "targets v2 only (nyralabs/CrisperWhisper2.0_*). v1 ships a remapped "
            "'changed tokenizer' and a different prompt contract; it is explicitly "
            "out of scope for this port."
        )

    compute_type = {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}[args.compute_dtype]
    print(f"Loading CrisperWhisper engine from {model_id} ({origin}, device={args.device}, "
          f"compute_type={compute_type})...")
    engine = TransformersEngine(
        model_id,
        device=args.device,
        compute_type=compute_type,
    )
    return engine, model_id


def make_source(
    *,
    args: argparse.Namespace,
    engine,
    model_id: str,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import crisperwhisper
    import torch
    import transformers

    source: dict[str, Any] = {
        "kind": "crisperwhisper-author-transformers-backend",
        "framework": "author_repo_crisperwhisper",
        "crisperwhisper_version": getattr(crisperwhisper, "__version__", "unknown"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model": model_id,
        "model_dtype": "f32 compute (bf16 storage upcast)",
        "attn_implementation": "eager",
        "device": args.device,
        "torch_threads": args.torch_threads,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
        "language": args.language,
        "n_mels": int(engine.n_mels),
        "rewind_features_disabled": [
            "hallucination_mitigation",
            "early_eot_recovery",
            "temperature_fallback",
        ],
    }
    if extra:
        source.update(extra)
    return source


def build_prompt(engine, language: str, mode: str) -> list[int]:
    from crisperwhisper.prompt import PromptBuilder

    builder = PromptBuilder(engine, language=language)
    if mode == "verbatim":
        return builder.verbatim()
    if mode == "intended":
        return builder.intended()
    raise SystemExit(f"error: unknown --mode {mode!r}; expected 'verbatim' or 'intended'")


def dump_encoder(
    *,
    engine,
    features,
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int] | None,
):
    """Run the real encoder forward with hooks; dump intermediates.

    Returns the encoder's last_hidden_state for the decoder pass.
    """
    import torch
    import torch.nn.functional as F

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage, source, out_dir=out_dir)

    encoder = engine.model.model.encoder
    n_layers = len(encoder.layers)
    block_set = resolve_blocks(blocks, n_layers)

    with Taps() as taps:
        taps.out(encoder.conv1, "conv1")
        taps.out(encoder.conv2, "conv2")
        taps.inp(encoder.layers[0], "embed_out")
        for i in sorted(block_set):
            taps.out(encoder.layers[i], f"block{i}")
        taps.out(encoder.layer_norm, "final")

        with torch.inference_mode():
            enc_out = encoder(features)

    # HF applies gelu outside the conv modules
    # (`inputs_embeds = gelu(self.conv1(input_features))`), so the hooked
    # module output is pre-activation. Apply the same activation here so
    # the dumped tensor is the post-GELU value the C++ graph produces.
    dump("enc.conv1.out", F.gelu(taps.store["conv1"]).transpose(-2, -1), "encoder.conv1")
    dump("enc.conv2.out", F.gelu(taps.store["conv2"]).transpose(-2, -1), "encoder.conv2")
    dump("enc.pos_emb", encoder.embed_positions.weight, "encoder.pos_emb")
    dump("enc.embed.out", taps.store["embed_out"], "encoder.embed")
    for i in sorted(block_set):
        dump(f"enc.block.{i}.out", taps.store[f"block{i}"], f"encoder.block{i}.out")
    dump("enc.final", taps.store["final"], "encoder.final")

    return enc_out.last_hidden_state


def dump_decoder_prompt_pass(
    *,
    engine,
    encoder_hidden,
    prompt_ids: list[int],
    out_dir: Path,
    source: dict[str, Any],
    blocks: set[int] | None,
    prefix: str = "dec",
    full: bool = True,
) -> None:
    """Run the real decoder forward on the prompt with hooks; dump intermediates.

    `full=False` dumps only the raw logits — used for the sibling-mode
    (intended) pass, where the point is that the mode tags change the
    output distribution, not to re-cover the whole block stack.
    """
    import torch

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
        write_tensor(name, a, stage, source, out_dir=out_dir)

    model = engine.model
    decoder = model.model.decoder
    n_layers = len(decoder.layers)
    block_set = resolve_blocks(blocks, n_layers)
    input_ids = torch.tensor([prompt_ids], device=engine.device, dtype=torch.long)

    with Taps() as taps:
        if full:
            taps.out(decoder.embed_tokens, "token_emb")
            taps.out(decoder.embed_positions, "pos_emb")
            taps.inp(decoder.layers[0], "embed_sum")
            for i in sorted(block_set):
                taps.out(decoder.layers[i], f"block{i}")
            taps.out(decoder.layer_norm, "final")

        with torch.inference_mode():
            dec_out = decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden,
                use_cache=False,
            )
            logits_raw = model.proj_out(dec_out.last_hidden_state)

    if full:
        dump(f"{prefix}.token_emb", taps.store["token_emb"], "decoder.embedding")
        dump(f"{prefix}.pos_emb", taps.store["pos_emb"], "decoder.position_embedding")
        dump(f"{prefix}.embed_sum", taps.store["embed_sum"], "decoder.embed_sum")
        for i in sorted(block_set):
            dump(f"{prefix}.block.{i}.out", taps.store[f"block{i}"], f"decoder.block{i}.out")
        dump(f"{prefix}.out_before_head", taps.store["final"], "decoder.output_before_head")

    dump(f"{prefix}.logits_raw", logits_raw, "decoder.logits_raw")
    if full:
        with torch.inference_mode():
            log_probs = torch.log_softmax(logits_raw, dim=-1)
        dump(f"{prefix}.logits", log_probs, "decoder.logits")


def dump_mid_generation(
    *,
    engine,
    encoder_hidden,
    prompt_ids: list[int],
    gen_ids: list[int],
    out_dir: Path,
    source: dict[str, Any],
    gen_step_n: int,
) -> None:
    """Dump the logits row that predicts generated token `gen_step_n`.

    Exercises the n_past > 0 path: the prompt pass alone covers only
    n_past == 0, so without this the C++ KV-cache update has zero
    tensor-level coverage. The forced context is `prompt + gen_ids[:n]`,
    taken from the engine's own greedy trajectory, so this row sits on
    the reference decode path rather than on a re-derived one.
    """
    import torch

    if len(gen_ids) < gen_step_n:
        print(
            f"  note: only {len(gen_ids)} generated tokens; "
            f"clamping gen-step-n from {gen_step_n} to {len(gen_ids)}"
        )
        gen_step_n = len(gen_ids)
    if gen_step_n <= 0:
        print("  warn: no generated tokens; skipping mid-generation dump")
        return

    model = engine.model
    decoder = model.model.decoder
    forced = list(prompt_ids) + list(gen_ids[:gen_step_n])
    input_ids = torch.tensor([forced], device=engine.device, dtype=torch.long)

    with torch.inference_mode():
        dec_out = decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        )
        logits_final = model.proj_out(dec_out.last_hidden_state[:, -1:, :])

    name = f"dec.logits_raw.gen{gen_step_n}"
    a = to_np(logits_final)
    print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} mean={a.mean():.6e}")
    write_tensor(name, a, f"decoder.logits_raw.gen{gen_step_n}", source, out_dir=out_dir)


def write_json_artifact(out_dir: Path, name: str, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {path}")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def prepare(args: argparse.Namespace):
    """Shared setup: engine, audio, features, source metadata."""
    configure_torch(args)
    engine, model_id = load_engine(args)

    audio_path = resolve_path(args.audio)
    pcm, sr = load_audio(audio_path)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"error: audio sample rate is {sr}, expected {SAMPLE_RATE}")
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr} ({pcm.size / sr:.2f}s)")
    if pcm.size > CHUNK_SAMPLES:
        print(
            f"  note: audio exceeds the 30 s window; the engine truncates to the "
            f"first {CHUNK_SAMPLES} samples for single-chunk dumps. Long-form "
            f"<ctx> continuation is a Stage 4 concern, not a tensor-dump one."
        )

    features, mel = engine.extract_features_with_mel(pcm)
    source = make_source(
        args=args,
        engine=engine,
        model_id=model_id,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
    )
    return engine, model_id, audio_path, pcm, features, mel, source


def mel_encoder_layout(mel: np.ndarray) -> np.ndarray:
    """HF gives the log-mel as [n_mels, n_frames]; dump it as [n_frames,
    n_mels].

    That is the layout the C++ encoder's `enc.mel.in` tensor actually holds
    (ggml ne=[n_mels, n_frames] means n_mels is INNERMOST), and it is what
    `scripts/dump_reference_whisper_transformers.py` already writes, so both
    Whisper-graph families compare against one convention. It is also the
    layout `TRANSCRIBE_MEL_FROM_REF` injects verbatim into the encoder, so a
    mel-major dump would silently feed the encoder a transposed spectrogram
    (which decodes as silence, not as an error).
    """
    return np.ascontiguousarray(mel.T)


def cmd_mel(args: argparse.Namespace) -> int:
    engine, _, _, _, _, mel, source = prepare(args)
    out_dir = resolve_path(args.out)
    print(
        f"mel: shape={mel.shape} min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.6f} std={mel.std():.6f}"
    )
    write_tensor("enc.mel.in", mel_encoder_layout(mel), "frontend.mel.norm", source, out_dir=out_dir)
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    engine, _, _, _, features, mel, source = prepare(args)
    out_dir = resolve_path(args.out)
    blocks = set(args.blocks) if args.blocks else None

    write_tensor("enc.mel.in", mel_encoder_layout(mel), "frontend.mel.norm", source, out_dir=out_dir)
    dump_encoder(
        engine=engine,
        features=features,
        out_dir=out_dir,
        source=source,
        blocks=blocks,
    )
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    from crisperwhisper.prompt import strip_prompt_artifacts

    engine, model_id, audio_path, pcm, features, mel, source = prepare(args)
    out_dir = resolve_path(args.out)
    blocks = set(args.blocks) if args.blocks else None
    mode = args.mode
    sibling = "intended" if mode == "verbatim" else "verbatim"

    prompt_ids = build_prompt(engine, args.language, mode)
    sibling_ids = build_prompt(engine, args.language, sibling)
    decoder_prefix = engine.get_decoder_prefix(args.language)
    print(f"  mode={mode} prompt_ids={prompt_ids}")
    print(f"    decoded prompt: {engine.tokenizer.decode(prompt_ids)!r}")
    print(f"    whisper prefix: {decoder_prefix}")
    source = dict(source)
    source["mode"] = mode
    source["prompt_ids"] = list(prompt_ids)
    source["decoder_prefix_ids"] = list(decoder_prefix)

    write_tensor("enc.mel.in", mel_encoder_layout(mel), "frontend.mel.norm", source, out_dir=out_dir)
    encoder_hidden = dump_encoder(
        engine=engine,
        features=features,
        out_dir=out_dir,
        source=source,
        blocks=blocks,
    )

    dump_decoder_prompt_pass(
        engine=engine,
        encoder_hidden=encoder_hidden,
        prompt_ids=prompt_ids,
        out_dir=out_dir,
        source=source,
        blocks=blocks,
        prefix="dec",
        full=True,
    )

    # Sibling-mode prompt pass. The mode tags are the model's control
    # surface and both modes are MUST PASS for this port, so the
    # alternative prompt gets its own logits oracle. Same audio, same
    # encoder output, different decoder prefix.
    sibling_source = dict(source)
    sibling_source["mode"] = sibling
    sibling_source["prompt_ids"] = list(sibling_ids)
    dump_decoder_prompt_pass(
        engine=engine,
        encoder_hidden=encoder_hidden,
        prompt_ids=sibling_ids,
        out_dir=out_dir,
        source=sibling_source,
        blocks=blocks,
        prefix=f"dec.{sibling}",
        full=False,
    )

    # Greedy decode. hallucination_mitigation=False reduces
    # generate_with_repair to a single plain greedy generate; no rewinds,
    # so the trajectory is reproducible token-for-token.
    gen_ids = engine.generate_with_repair(
        features,
        prompt_ids,
        max_length=args.max_new_tokens,
        hallucination_mitigation=False,
    )
    print(f"  generated {len(gen_ids)} tokens")

    dump_mid_generation(
        engine=engine,
        encoder_hidden=encoder_hidden,
        prompt_ids=prompt_ids,
        gen_ids=gen_ids,
        out_dir=out_dir,
        source=source,
        gen_step_n=args.gen_step_n,
    )

    # Cross-attention over the alignment heads: the exact [n_gen, F_enc]
    # matrix the reference Viterbi aligner consumes. This is the oracle
    # for the word-timestamps capability row — without it that row has no
    # tensor-level target at all.
    words_payload: dict[str, Any] | None = None
    if not args.skip_word_timings:
        heads = engine.enable_attention(None)
        attn_ids, attention = engine.generate_with_repair_and_attention(
            features,
            prompt_ids,
            max_length=args.max_new_tokens,
            hallucination_mitigation=False,
        )
        # The attention-capture pass keeps the terminating EOT that the
        # plain greedy pass strips, so a trailing-EOT-only difference is
        # expected and benign. Anything else means the two passes took
        # different trajectories, which would invalidate the attention
        # matrix as an oracle for this transcript.
        shared = min(len(attn_ids), len(gen_ids))
        eot = getattr(engine, "eot_id", None)
        tail = [int(t) for t in list(attn_ids)[shared:] + list(gen_ids)[shared:]]
        if list(attn_ids)[:shared] != list(gen_ids)[:shared] or any(
            t != eot for t in tail
        ):
            print(
                "  warn: attention-capture pass diverged from the greedy pass "
                f"({len(attn_ids)} vs {len(gen_ids)} tokens, and the difference "
                "is not a trailing EOT). The attention matrix may not correspond "
                "to the dumped transcript.",
                file=sys.stderr,
            )
        attn_source = dict(source)
        attn_source["alignment_heads"] = [list(h) for h in heads]
        attn_source["n_generated"] = len(attn_ids)
        write_tensor(
            "dec.xattn.align",
            np.ascontiguousarray(attention, dtype=np.float32),
            "decoder.cross_attention.alignment_heads",
            attn_source,
            out_dir=out_dir,
        )
        print(f"  dec.xattn.align: shape={attention.shape} heads={heads}")

        from crisperwhisper.word_timing import extract_word_timings

        duration_s = min(pcm.size / SAMPLE_RATE, 30.0)
        words = extract_word_timings(
            engine, attn_ids, attention, mel, audio_duration_s=duration_s
        )
        words_payload = {
            "schema": "transcribe-reference-word-timestamps-v1",
            "alignment_heads": [list(h) for h in heads],
            "audio_duration_s": duration_s,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end} for w in words
            ],
            "source": attn_source,
        }
        write_json_artifact(out_dir, "word_timestamps.json", words_payload)

    if not args.skip_transcript:
        text = strip_prompt_artifacts(engine.decode_tokens(gen_ids, skip_special=True))
        print(f"  transcript ({mode}): {text!r}")
        if not text.strip():
            print(
                "error: reference produced an empty transcript. The reference "
                "setup is broken; fix it before any C++ work.",
                file=sys.stderr,
            )
            return 1

        transcript_source = dict(source)
        transcript_source["normalized_text"] = normalize_text(text)
        transcript_source["generation"] = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "hallucination_mitigation": False,
            "early_eot_recovery": False,
            "temperature_fallback": False,
            "suppress_tokens": list(engine.default_suppress_tokens),
        }
        if words_payload is not None:
            transcript_source["n_words_timed"] = len(words_payload["words"])
        write_transcript(out_dir, text, source=transcript_source, tokens=list(gen_ids))
        print(f"  wrote {out_dir / 'transcript.json'}")

    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF repo id or local path to a CrisperWhisper 2.0 checkpoint")
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p.add_argument("--language", default="en", help="Language code (default: en)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="Torch intra-op threads for deterministic dumps (default: 1)")
    p.add_argument("--compute-dtype", default="f32", choices=["bf16", "f16", "f32"],
                   help="Torch compute dtype. Default f32 (BF16 storage upcast to "
                        "F32 compute), which is the Stage 4 gate regime, paired "
                        "with an F32 GGUF. bf16 reproduces the measurement showing "
                        "BF16-on-both-sides is the worst pairing; see the module "
                        "docstring.")
    # Declared on every subcommand, not just `decode`, because validate.py
    # forwards manifest reference.dump_args verbatim to each subcommand it
    # runs. `mel` and `encoder` ignore it (neither depends on the decoder
    # prompt); they accept it so the shared arg list stays uniform.
    p.add_argument("--mode", default="verbatim", choices=["verbatim", "intended"],
                   help="Transcription mode; sets the [verbatim_N]/[intended_N] "
                        "prompt tags (default: verbatim, the model's own default). "
                        "Ignored by the mel and encoder subcommands.")


def add_block_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--blocks", type=int, nargs="*", default=None,
                   help="Block indices to dump (default: auto — all if <=5 layers, "
                        "else 5 evenly spaced indices including first and last)")


def main() -> int:
    # allow_abbrev=False throughout: `--mode verbatim` would otherwise be
    # prefix-matched onto `--model` on any subcommand that lacks --mode,
    # silently turning the mode flag into a bogus repo id.
    p = argparse.ArgumentParser(
        description="Dump CrisperWhisper 2.0 reference tensors from the author package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("mel", help="Dump the log-mel spectrogram only", allow_abbrev=False)
    add_common_args(mp)
    mp.set_defaults(func=cmd_mel)

    ep = sub.add_parser("encoder", help="Dump mel + encoder intermediates", allow_abbrev=False)
    add_common_args(ep)
    add_block_arg(ep)
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser(
        "decode",
        help="Dump mel + encoder + decoder prompt pass + mid-generation logits "
             "+ alignment-head cross-attention + transcript",
        allow_abbrev=False,
    )
    add_common_args(dp)
    add_block_arg(dp)
    dp.add_argument("--max-new-tokens", type=int, default=256,
                    help="Generation budget (default: 256, matching the reference "
                         "transcribe() default — note this is well below "
                         "max_target_positions=448 and the prompt eats into it)")
    dp.add_argument("--gen-step-n", type=int, default=20,
                    help="Dump the logits row predicting generated token N (default: 20)")
    dp.add_argument("--skip-transcript", action="store_true",
                    help="Only dump tensors; do not write transcript.json")
    dp.add_argument("--skip-word-timings", action="store_true",
                    help="Skip the cross-attention capture pass and word_timestamps.json")
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
