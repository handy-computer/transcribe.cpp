#!/usr/bin/env python3
"""Reference dumps for parakeet-ultra from kestrel (the publisher's runtime).

Tensor names match scripts/dump_reference_parakeet_nemo.py wherever the
quantity is the same, so the C++ parakeet observer points compare as-is:

  encoder   enc.mel.in [n_mels, T]          normalized log-mel, frames on the last axis
            enc.pre_encode.out [T', d]      subsampler output (pre rel-pos)
            enc.pos_emb [2T'-1, d]          sinusoid table, positions T'-1 .. -(T'-1)
            enc.block.<i>.out [T', d]       block output (default blocks 0, 12, 23)
            enc.block.<i>.{ff1,attn,conv,ff2}   running residual (default block 0)
            enc.final [T', d]               encoder output before encoder_projector
  decode    dec.enc_out [T', d]             == enc.final
            dec.embed.0 [d_pred]            predictor input at step 0 (blank row, zeros)
            dec.lstm.<l>.{h,c}.0 [d_pred]   LSTM state after the blank step
            dec.joint.0 [V+5]               log_softmax over the whole joint row for
                                            (frame 0, start state); NeMo/C++ convention.
                                            kestrel argmaxes raw logits per slice, which
                                            this shift does not change.
            transcript.json                 kestrel runtime text (+ token ids)
  longform  vad.b<k>.mel [n_mels, Tb]       features of VAD scan block k (120 s blocks,
                                            each normalized on its own)
            vad.b<k>.pre_encode [Tb', d]    subsampler output for the block
            vad.b<k>.proj [Tb', 128]        SiLU(proj)
            vad.b<k>.ctx [Tb', 128]         SiLU(ctx)
            vad.b<k>.prob [Tb']             sigmoid(out), valid frames only
            vad.prob [N]                    all blocks' valid-frame probabilities, in order
            seg.<j>.enc.mel.in [n_mels, T]  features of decoded segment j
            seg.<j>.enc.final [T'v, d]      encoder output for segment j, valid rows only
            longform.segments [n, 2]        (start sample, sample count) per segment, exact
            segments.json                   blocks, speech regions, cut points (sample
                                            indices), per-segment tokens/durations/text
            transcript.json                 kestrel runtime text + word-timestamped segments

Usage:
    uv run --project scripts/envs/parakeet-kestrel \\
        scripts/dump_reference_parakeet_kestrel.py {encoder,decode,longform} \\
        --model moondream/parakeet-ultra --revision <sha> \\
        --audio samples/jfk.wav --out build/validate/parakeet/parakeet-ultra/jfk/ref
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.ref_dump import write_tensor, write_transcript  # noqa: E402

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure_torch(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.manual_seed(0)


def load(args: argparse.Namespace):
    import torch
    from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt

    loaded = load_parakeet_tdt(
        args.model, revision=args.revision, device="cpu", dtype=torch.float32
    )
    if loaded.model.vad_head is None:
        raise SystemExit(f"error: {args.model} carries no vad_head; this dumper is for parakeet-ultra")
    return loaded.model, loaded.tokenizer


def make_runtime(model, tokenizer, args: argparse.Namespace):
    import torch
    from kestrel.models.parakeet_tdt.runtime import ParakeetTdtRuntime

    cfg = types.SimpleNamespace(
        model=args.model,
        device="cpu",
        dtype=torch.float32,
        cpu_threads=max(1, int(args.torch_threads)),
        enable_cuda_graphs=False,
    )
    return ParakeetTdtRuntime(cfg, model=model, tokenizer=tokenizer)


def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"error: {path} is {sr} Hz, expected {SAMPLE_RATE}")
    if pcm.ndim != 1:
        raise SystemExit(f"error: {path} is not mono")
    return pcm


def make_source(args: argparse.Namespace, audio: Path, n_samples: int) -> dict[str, Any]:
    return {
        "framework": "kestrel",
        "kestrel_version": version("kestrel"),
        "model": args.model,
        "revision": args.revision,
        "audio": audio.name,
        "n_samples": n_samples,
        "sample_rate": SAMPLE_RATE,
        "device": "cpu",
        "dtype": "float32",
        "language_hint_ignored": args.language,
    }


def to_np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32)


def features_for(waveform: np.ndarray):
    """The runtime's own feature path for one row (parakeet_cohort_features)."""
    import torch
    from kestrel.models.parakeet_tdt.features import parakeet_cohort_features

    staged = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))[None]
    return parakeet_cohort_features(staged, [int(waveform.size)])


# ---------------------------------------------------------------------------
# Encoder capture (re-executes Encoder.forward_subsampled with taps)
# ---------------------------------------------------------------------------

def run_encoder(model, features, mask, blocks: list[int], sub_blocks: list[int]) -> dict[str, Any]:
    """Same ops, same order as kestrel's Encoder.forward / EncoderBlock.forward;
    asserts bitwise equality with the un-tapped forward at the end."""
    import torch
    from kestrel_kernels import get_runtime

    from kestrel.models.parakeet_tdt.model import _norm_args

    enc = model.encoder
    taps: dict[str, Any] = {}
    with torch.inference_mode():
        hidden, valid = enc.subsampling(features, mask)
        taps["enc.pre_encode.out"] = hidden[0]
        length = hidden.shape[1]
        relative_positions = torch.arange(length - 1, -length, -1, device=hidden.device)
        phase = torch.outer(relative_positions.float(), enc.inverse_frequency)
        positions = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        positions = positions[None].expand(hidden.shape[0], -1, -1).to(hidden.dtype)
        taps["enc.pos_emb"] = positions[0]
        relative = enc.relative_k_proj(positions)
        relative = relative.view(*relative.shape[:2], len(enc.layers), -1)
        relative = relative.permute(2, 0, 1, 3).contiguous()
        pair_mask = valid[:, :, None] & valid[:, None, :]
        for index, (layer, rel_k) in enumerate(zip(enc.layers, relative)):
            if index in sub_blocks:
                conformer = get_runtime(hidden.device).conformer
                normed = conformer.layer_norm(hidden, *_norm_args(layer.norm_feed_forward1))
                h, normed = conformer.add_scaled_layer_norm(
                    hidden, layer.feed_forward1(normed), 0.5, *_norm_args(layer.norm_self_att))
                taps[f"enc.block.{index}.ff1"] = h[0]
                h, normed = conformer.add_scaled_layer_norm(
                    h, layer.self_attn(normed, rel_k, pair_mask), 1.0, *_norm_args(layer.norm_conv))
                taps[f"enc.block.{index}.attn"] = h[0]
                h, normed = conformer.add_scaled_layer_norm(
                    h, layer.conv(normed, valid), 1.0, *_norm_args(layer.norm_feed_forward2))
                taps[f"enc.block.{index}.conv"] = h[0]
                h, out = conformer.add_scaled_layer_norm(
                    h, layer.feed_forward2(normed), 0.5, *_norm_args(layer.norm_out))
                taps[f"enc.block.{index}.ff2"] = h[0]
                hidden = out
            else:
                hidden = layer(hidden, rel_k, pair_mask, valid)
            if index in blocks:
                taps[f"enc.block.{index}.out"] = hidden[0]
        taps["enc.final"] = hidden[0]
        reference, _ = enc(features, mask)
    if not torch.equal(reference[0], hidden[0]):
        diff = (reference[0] - hidden[0]).abs().max().item()
        raise SystemExit(f"error: tapped encoder diverged from kestrel Encoder.forward (max_abs={diff:.3e})")
    taps["_valid"] = valid
    return taps


def default_blocks(model, requested: list[int] | None) -> list[int]:
    n = len(model.encoder.layers)
    if requested:
        return sorted({i for i in requested if 0 <= i < n})
    return sorted({0, n // 2, n - 1})


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_encoder(args: argparse.Namespace) -> int:
    configure_torch(args)
    model, _tok = load(args)
    audio = Path(args.audio).resolve()
    out_dir = Path(args.out).resolve()
    pcm = load_audio(audio)
    source = make_source(args, audio, pcm.size)

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e}")
        write_tensor(name, a, stage, {**source, "hook": name}, out_dir=out_dir)

    features, mask = features_for(pcm)
    blocks = default_blocks(model, args.blocks)
    sub_blocks = sorted(set(args.sub_blocks if args.sub_blocks is not None else [0]))
    taps = run_encoder(model, features, mask, blocks, sub_blocks)

    dump("enc.mel.in", features[0].transpose(0, 1), "frontend.mel.norm")
    dump("enc.pre_encode.out", taps["enc.pre_encode.out"], "encoder.pre_encode")
    dump("enc.pos_emb", taps["enc.pos_emb"], "encoder.pos_emb")
    for i in sorted(set(blocks) | set(sub_blocks)):
        for tag in ("ff1", "attn", "conv", "ff2", "out"):
            key = f"enc.block.{i}.{tag}"
            if key in taps:
                dump(key, taps[key], f"encoder.block{i}.{tag}")
    dump("enc.final", taps["enc.final"], "encoder.final")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    import torch

    configure_torch(args)
    model, tokenizer = load(args)
    audio = Path(args.audio).resolve()
    out_dir = Path(args.out).resolve()
    pcm = load_audio(audio)
    source = make_source(args, audio, pcm.size)
    if pcm.size > 30 * SAMPLE_RATE:
        raise SystemExit("error: decode dumps one whole-clip pass; clips over 30 s go through `longform`")

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e}")
        write_tensor(name, a, stage, {**source, "hook": name}, out_dir=out_dir)

    features, mask = features_for(pcm)
    cfg = model.config
    with torch.inference_mode():
        encoded, valid = model.encoder(features, mask)
        dump("dec.enc_out", encoded[0], "decoder.enc_out")
        token = torch.tensor([[cfg.blank_token_id]])
        dump("dec.embed.0", model.decoder.embedding(token)[0, 0], "decoder.embed")
        decoder_hidden, (h, c) = model.decoder(token, None)
        for layer in range(h.shape[0]):
            dump(f"dec.lstm.{layer}.h.0", h[layer, 0], "decoder.lstm")
            dump(f"dec.lstm.{layer}.c.0", c[layer, 0], "decoder.lstm")
        projected = model.encoder_projector(encoded)
        logits = model.joint(projected[:, 0:1], decoder_hidden)
        dump("dec.joint.0", torch.log_softmax(logits.float(), dim=-1).reshape(-1), "decoder.joint")
        out = model.generate(features, mask)

    seq = out.sequences[0, : int(out.lengths[0])].tolist()
    tokens = [t for t in seq if t not in (tokenizer.blank_token_id, tokenizer.pad_token_id)]

    runtime = make_runtime(model, tokenizer, args)
    result = runtime.forward("transcribe", [{"audio": pcm, "sample_rate": SAMPLE_RATE, "timestamps": "word"}])[0]
    if isinstance(result, Exception):
        raise result
    text = result["text"]
    if text != tokenizer.decode(seq).strip():
        raise SystemExit("error: runtime text differs from model.generate on a single segment")
    print(f"  transcript: {text}")
    write_transcript(out_dir, text, source={**source, "segments": result["segments"]}, tokens=tokens)
    return 0


def cmd_longform(args: argparse.Namespace) -> int:
    """Run kestrel's own segmenter (pause_segments + head_speech) and record
    every VAD block, every cut, and every decoded segment."""
    import torch
    import torch.nn.functional as F
    from kestrel.models.asr.audio import AudioChunks
    from kestrel.models.parakeet_tdt import segment as kseg
    from kestrel.models.parakeet_tdt.vad import head_speech

    configure_torch(args)
    model, tokenizer = load(args)
    audio = Path(args.audio).resolve()
    out_dir = Path(args.out).resolve()
    pcm = load_audio(audio)
    source = make_source(args, audio, pcm.size)

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e}")
        write_tensor(name, a, stage, {**source, "hook": name}, out_dir=out_dir)

    # Tap the head: kestrel's head_speech calls model.speech_probabilities once per
    # scanned block. The tap re-runs the head's three convolutions and asserts the
    # result equals the real call before recording.
    blocks: list[dict[str, Any]] = []
    original = model.speech_probabilities

    def tapped(features, attention_mask):
        probs, valid = original(features, attention_mask)
        with torch.inference_mode():
            hidden, _ = model.encoder.subsampling(features, attention_mask)
            head = model.vad_head
            proj = F.silu(head.proj(hidden.transpose(1, 2)))
            ctx = F.silu(head.ctx(proj))
            mine = torch.sigmoid(head.out(ctx).squeeze(1).float())
        if not torch.equal(mine, probs):
            raise SystemExit("error: tapped vad_head diverged from speech_probabilities")
        n = int(valid[0].sum())
        blocks.append({
            "mel": features[0].transpose(0, 1),
            "pre_encode": hidden[0],
            "proj": proj[0].transpose(0, 1),
            "ctx": ctx[0].transpose(0, 1),
            "prob": probs[0, :n],
            "mel_frames": int(attention_mask[0].sum()),
            "valid_frames": n,
        })
        return probs, valid

    model.speech_probabilities = tapped
    block_regions: list[list[tuple[float, float]]] = []
    block_samples: list[int] = []

    def speech(waveform: np.ndarray, sample_rate: int):
        regions = head_speech(model, waveform, sample_rate)
        block_regions.append([list(r) for r in regions])
        block_samples.append(int(waveform.size))
        return regions

    segments: list[dict[str, Any]] = []
    with AudioChunks(pcm, sample_rate=SAMPLE_RATE, clip_start_seconds=0.0, clip_end_seconds=None,
                     target_sample_rate=SAMPLE_RATE, max_duration_seconds=24 * 60 * 60) as chunks:
        for piece in kseg.pause_segments(chunks, speech):
            start = int(round(piece.clip_start_seconds * SAMPLE_RATE))
            segments.append({
                "index": len(segments),
                "start_sample": start,
                "n_samples": int(piece.waveform.size),
                "start_seconds": piece.clip_start_seconds,
                "duration_seconds": piece.duration_seconds,
                "_waveform": piece.waveform,
            })
    model.speech_probabilities = original

    # Cut points as an exact-compare tensor: [n_segments, 2] = (start sample,
    # sample count). Integer-valued f32 is exact below 2^24 samples (17 min).
    if segments:
        cuts = np.array([[sg["start_sample"], sg["n_samples"]] for sg in segments], dtype=np.float64)
        if cuts.max() >= 2**24:
            raise SystemExit("error: longform.segments needs samples < 2^24 to be exact in f32")
        dump("longform.segments", torch.from_numpy(cuts.astype(np.float32)), "longform.segments")

    # The segmenter is contiguous: every segment is a slice of the file.
    for seg in segments:
        s, n = seg["start_sample"], seg["n_samples"]
        if not np.array_equal(seg["_waveform"], pcm[s:s + n]):
            raise SystemExit(f"error: segment {seg['index']} is not pcm[{s}:{s + n}]")

    for k, block in enumerate(blocks):
        dump(f"vad.b{k}.mel", block["mel"], "vad.mel")
        dump(f"vad.b{k}.pre_encode", block["pre_encode"], "vad.pre_encode")
        dump(f"vad.b{k}.proj", block["proj"], "vad.proj")
        dump(f"vad.b{k}.ctx", block["ctx"], "vad.ctx")
        dump(f"vad.b{k}.prob", block["prob"], "vad.prob")
    if blocks:
        dump("vad.prob", torch.cat([b["prob"] for b in blocks]), "vad.prob")

    # Decode each segment the way the runtime does (one row, cohort features,
    # model.generate) and record tokens/durations and the encoder output.
    for seg in segments:
        wav = seg.pop("_waveform")
        features, mask = features_for(wav)
        with torch.inference_mode():
            encoded, valid = model.encoder(features, mask)
            out = model.generate(features, mask)
        j = seg["index"]
        n_valid = int(valid[0].sum())
        seg["valid_frames"] = n_valid
        # C++ "seg.<j>." names; valid rows only (padded rows are unspecified).
        dump(f"seg.{j}.enc.mel.in", features[0].transpose(0, 1), "segment.mel")
        dump(f"seg.{j}.enc.final", encoded[0, :n_valid], "segment.encoder.final")
        n = int(out.lengths[0])
        seg["token_ids"] = out.sequences[0, :n].tolist()
        seg["durations"] = out.durations[0, :n].tolist()
        seg["text"] = tokenizer.decode(seg["token_ids"]).strip()

    runtime = make_runtime(model, tokenizer, args)
    result = runtime.forward("transcribe", [{"audio": pcm, "sample_rate": SAMPLE_RATE, "timestamps": "word"}])[0]
    if isinstance(result, Exception):
        raise result
    stitched = " ".join(s["text"] for s in segments if s["text"])
    if stitched != result["text"]:
        raise SystemExit("error: per-segment decode does not reproduce the runtime transcript")

    meta = {
        "constants": {
            "segment_seconds": kseg.SEGMENT_SECONDS,
            "min_pause_seconds": kseg.MIN_PAUSE_SECONDS,
            "min_segment_seconds": kseg.MIN_SEGMENT_SECONDS,
            "block_seconds": kseg.BLOCK_SECONDS,
            "encoder_frame_seconds": model.encoder_frame_seconds,
        },
        "n_samples": int(pcm.size),
        "fits_one_segment": pcm.size <= kseg.SEGMENT_SECONDS * SAMPLE_RATE,
        "vad_blocks": [
            {"index": k, "n_samples": block_samples[k], "mel_frames": b["mel_frames"],
             "valid_frames": b["valid_frames"], "speech_regions_seconds": block_regions[k]}
            for k, b in enumerate(blocks)
        ],
        "segments": segments,
        "source": source,
    }
    (out_dir / "segments.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {len(blocks)} VAD blocks, {len(segments)} segments: "
          f"{[(s['start_sample'], s['n_samples']) for s in segments]}")
    print(f"  transcript: {result['text'][:200]}...")
    tokens = [t for s in segments for t in s["token_ids"]
              if t not in (tokenizer.blank_token_id, tokenizer.pad_token_id)]
    write_transcript(out_dir, result["text"], source={**source, "segments": result["segments"]}, tokens=tokens)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id (moondream/parakeet-ultra) or local snapshot dir")
    p.add_argument("--revision", default=None, help="HF revision to pin (kestrel defaults to main)")
    p.add_argument("--audio", required=True, help="16 kHz mono wav")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--torch-threads", type=int, default=1, help="Torch threads (default 1 for deterministic dumps)")
    p.add_argument("--language", default=None,
                   help="Accepted for validate.py compatibility and ignored: kestrel rejects language forcing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ep = sub.add_parser("encoder", help="Whole-clip encoder intermediates (clips <= 30 s)")
    add_common_args(ep)
    ep.add_argument("--blocks", type=int, nargs="*", default=None, help="Block outputs to dump (default 0, mid, last)")
    ep.add_argument("--sub-blocks", type=int, nargs="*", default=None, help="Blocks to dump ff1/attn/conv/ff2 for (default 0)")
    ep.set_defaults(func=cmd_encoder)
    dp = sub.add_parser("decode", help="Decoder first step + joint + runtime transcript (clips <= 30 s)")
    add_common_args(dp)
    dp.set_defaults(func=cmd_decode)
    lp = sub.add_parser("longform", help="VAD-head scan, pause segmentation, per-segment decode, stitched transcript")
    add_common_args(lp)
    lp.set_defaults(func=cmd_longform)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
