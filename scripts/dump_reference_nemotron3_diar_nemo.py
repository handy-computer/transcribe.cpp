#!/usr/bin/env python3
"""
dump_reference_nemotron3_diar_nemo.py - Nemotron-3-Diarization reference
tensors from NVIDIA NeMo Speech (the canonical reference), plus HF
transformers cross-check / push-audio dumps.

Nemotron-3-Diarization is a streaming Sortformer diarizer (arch pattern
`encoder-diarizer`), NOT a transcription model: the output is a T x 8
speaker-activity probability matrix at 10 ms (arrival-order columns) plus
the derived speaker segments.

Forward (one streaming step; cache/FIFO frames are pre_encode embeddings):
    mel [128, T_mel]  (AudioToMelSpectrogramPreprocessor, normalize=NA,
                       BF16-rounded window/fb buffers from the checkpoint)
    -> pre_encode     FeatureStacking x8 + Linear 1024->512 (no bias)
    -> [spkcache | fifo | chunk | rc] concat
    -> embed_norm     LayerNorm
    -> 31 x pre-LN Transformer layer (RoPE, positions restart per step)
    -> final_norm     LayerNorm                         [T_step, 512]
    -> encoder_proj   Linear 512->192                   [T_step, 192]
    -> subpixel_upsample Conv1d 192->1536 k3, reshape   [8*T_step, 192]
    -> ReLU -> first_hidden_to_hidden -> ReLU -> single_hidden_to_spks
       (dropout is inactive at eval)
                                                        [8*T_step, 8] logits
    -> sigmoid; AOSC scoring uses the x8 average-pooled probs.

The model always runs the streaming path (streaming_mode=true in the
checkpoint cfg); "offline" is the very_high_latency preset.

Subcommands:
  encoder     Per-stage activations of the FIRST streaming step (empty
              cache, input = chunk + right context) at --preset (default
              very_high_latency): a single clean forward to bring up the
              graph against. Also the full-clip mel.
  diarize     Streaming m.diarize() at --preset: diar.probs [T_mel, 8] +
              speaker segments, optional AOSC compression internals.
  hf-stream   HF transformers push-audio reference: the processor cuts
              audio chunk by chunk (per-chunk mel, center only on the
              first chunk) in streaming_mode low_latency / very_low_latency
              / ultra_low_latency. Dumps diar.probs.
  hf-offline  HF transformers whole-recording forward (config offline
              preset == very_high_latency). Dumps diar.probs.

Weights: the .nemo stores BF16; NeMo restore_from upcasts parameters to
fp32 (values stay BF16-exact) and this dumper computes in fp32 on CPU.

    uv run --project scripts/envs/nemotron3_diar \
      scripts/dump_reference_nemotron3_diar_nemo.py encoder \
      --model nvidia/Nemotron-3-Diarization \
      --audio samples/nemotron3-diar-8spk-mix.wav \
      --out build/validate/nemotron3_diar/Nemotron-3-Diarization/nemotron3-diar-8spk-mix/encoder/ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ref_dump  # noqa: E402

write_tensor = ref_dump.write_tensor
write_transcript = ref_dump.write_transcript

NEMO_FILENAME = "Nemotron-3-Diarization.nemo"

# Streaming presets, 80 ms encoder frames. Model-card table, plus `small`
# (diagnostic only: forces multi-chunk FIFO + several compressions on a
# short clip). Keep in sync with the C++ preset table.
PRESETS = {
    "very_high_latency": dict(spkcache_len=264, fifo_len=40, chunk_len=340,
                              chunk_right_context=40, spkcache_update_period=300),
    "low_latency": dict(spkcache_len=264, fifo_len=264, chunk_len=9,
                        chunk_right_context=4, spkcache_update_period=222),
    "very_low_latency": dict(spkcache_len=264, fifo_len=264, chunk_len=6,
                             chunk_right_context=2, spkcache_update_period=222),
    "ultra_low_latency": dict(spkcache_len=264, fifo_len=264, chunk_len=3,
                              chunk_right_context=1, spkcache_update_period=222),
    "small": dict(spkcache_len=24, fifo_len=10, chunk_len=20,
                  chunk_right_context=2, spkcache_update_period=20),
}
HF_STREAMING_MODES = ("low_latency", "very_low_latency", "ultra_low_latency")

# Encoder layers dumped from the first step (all 31 would be ~6 MB each on
# the 380-frame step; first/second/middle/last localize drift well enough).
DUMP_LAYERS = (0, 1, 15, 30)


def _resolve_nemo(model: str, revision: str | None) -> str:
    if model.endswith(".nemo") or Path(model).exists():
        return model
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, NEMO_FILENAME, revision=revision)


def _load_nemo(model: str, revision: str | None):
    from nemo.collections.asr.models import SortformerEncLabelModel

    m = SortformerEncLabelModel.restore_from(
        restore_path=_resolve_nemo(model, revision), map_location="cpu", strict=False
    )
    m = m.float().eval()
    return m


def _apply_preset(m, preset: str) -> dict[str, int]:
    cfg = PRESETS[preset]
    sm = m.sortformer_modules
    for k, v in cfg.items():
        setattr(sm, k, v)
    m._check_streaming_parameters()
    return {k: int(getattr(sm, k)) for k in cfg}


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"error: {path} is {sr} Hz; the reference expects 16 kHz")
    return audio.astype(np.float32), int(sr)


def _f32(t: torch.Tensor) -> np.ndarray:
    a = t.detach().to(torch.float32).cpu().numpy()
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a.astype(np.float32))


def _src(model: str, hook: str, **extra: Any) -> dict[str, Any]:
    return {"framework": "nemo", "model": model, "hook": hook, **extra}


def cmd_encoder(args: argparse.Namespace) -> int:
    m = _load_nemo(args.model, args.revision)
    geom = _apply_preset(m, args.preset)
    audio, _ = _read_audio(args.audio)
    out_dir = Path(args.out)
    sig = torch.tensor(audio).unsqueeze(0)
    length = torch.tensor([audio.shape[0]])

    with torch.no_grad():
        mel, mel_len = m.process_signal(audio_signal=sig, audio_signal_length=length)
    n_mel = int(mel_len[0])
    write_tensor("enc.mel.in", _f32(mel[0, :, :n_mel]), "frontend",
                 _src(args.model, "process_signal (trimmed to valid length)",
                      layout="[n_mels, T_mel]"), out_dir=out_dir)

    captured: dict[str, torch.Tensor] = {}

    def grab(name, pick=lambda inp, out: out):
        def hook(_mod, inp, out):
            if name in captured:  # first streaming step only
                return
            t = pick(inp, out)
            t = t[0] if isinstance(t, (tuple, list)) else t
            captured[name] = t.detach().to(torch.float32).cpu()
        return hook

    enc, sm = m.encoder, m.sortformer_modules
    hooks = [
        (enc.pre_encode, grab("enc.pre_encode.out")),
        (enc.embed_norm, grab("enc.embed_norm.out")),
        *[(enc.layers[i], grab(f"enc.layers.{i}.out")) for i in DUMP_LAYERS],
        (enc.final_norm, grab("enc.final_norm.out")),
        (sm.encoder_proj, grab("diar.encoder_proj.out")),
        (sm.subpixel_upsample, grab("diar.subpixel_conv.out")),
        (sm.single_hidden_to_spks, grab("diar.logits")),
    ]
    handles = [mod.register_forward_hook(fn) for mod, fn in hooks]
    # upsample_hidden is a method (Conv1d + [T, 8, H] reshape), not a module.
    orig_upsample = sm.upsample_hidden

    def upsample_hidden(hidden_states):
        out = orig_upsample(hidden_states)
        captured.setdefault("diar.upsample.out", out.detach().to(torch.float32).cpu())
        return out

    sm.upsample_hidden = upsample_hidden
    try:
        with torch.no_grad():
            m.forward(audio_signal=sig, audio_signal_length=length)
    finally:
        for h in handles:
            h.remove()
        sm.upsample_hidden = orig_upsample

    step_frames = min(geom["chunk_len"] + geom["chunk_right_context"], -(-n_mel // 8))
    layouts = {
        "enc.pre_encode.out": "[T_step, 512] feature-stacked projection (cached state lives at this point)",
        "diar.subpixel_conv.out": "[T_step, 1536] conv output before the x8 reshape",
        "diar.upsample.out": "[8*T_step, 192] upsampled hidden after the [T, 8, H] reshape (classifier input, pre-ReLU)",
        "diar.logits": "[8*T_step, 8] pre-sigmoid speaker logits, arrival order",
    }
    for name, t in captured.items():
        a = _f32(t)
        if name == "diar.subpixel_conv.out" and a.shape[0] == 1536:
            a = np.ascontiguousarray(a.T)  # Conv1d emits [C, T]; store time-major
        write_tensor(name, a, "encoder",
                     _src(args.model, f"{name} (first streaming step)", preset=args.preset,
                          streaming_cfg=geom, step_encoder_frames=step_frames,
                          layout=layouts.get(name, "[T_step, D] time-major")),
                     out_dir=out_dir)
    missing = [n for n in ("enc.pre_encode.out", "diar.logits") if n not in captured]
    if missing:
        raise SystemExit(f"error: hooks did not fire for {missing}")
    print(f"wrote {len(captured) + 1} encoder-stage tensors (step0: {step_frames} enc frames) to {out_dir}")
    return 0


def _install_compress_hooks(m) -> list[dict]:
    """Record, per _compress_spkcache call, the selected (sorted) frame indices,
    the is_disabled mask, and the input / gathered preds (batch=1, squeezed)."""
    sm = m.sortformer_modules
    calls: list[dict] = []
    orig_topk = sm._get_topk_indices
    orig_compress = sm._compress_spkcache

    def wrapped_topk(scores):
        topk_indices, is_disabled = orig_topk(scores)
        calls.append({
            "n_frames": int(scores.shape[1]),
            "topk_indices": topk_indices[0].detach().cpu().numpy().astype(np.int64),
            "is_disabled": is_disabled[0].detach().cpu().numpy().astype(bool),
        })
        return topk_indices, is_disabled

    def wrapped_compress(*a, **kw):
        in_preds = kw["preds"] if "preds" in kw else a[1]
        in_preds = in_preds[0].detach().cpu().numpy().astype(np.float32)
        spkcache, spkcache_preds, spk_perm = orig_compress(*a, **kw)
        if calls:
            calls[-1]["input_preds"] = in_preds
            calls[-1]["spkcache_preds"] = spkcache_preds[0].detach().cpu().numpy().astype(np.float32)
        return spkcache, spkcache_preds, spk_perm

    sm._get_topk_indices = wrapped_topk
    sm._compress_spkcache = wrapped_compress
    return calls


def _write_compress_dump(calls: list[dict], dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(calls):
        for key in ("topk_indices", "is_disabled", "spkcache_preds", "input_preds"):
            if key in c:
                np.save(dump_dir / f"compress.{k:03d}.{key}.npy", c[key])
    summary = {"n_calls": len(calls), "n_frames_in": [c["n_frames"] for c in calls]}
    (dump_dir / "compress.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(calls)} _compress_spkcache dumps to {dump_dir}")


def _segments_text(segs) -> str:
    return "\n".join(str(s).strip() for s in segs)


def cmd_diarize(args: argparse.Namespace) -> int:
    m = _load_nemo(args.model, args.revision)
    geom = _apply_preset(m, args.preset)
    audio, sr = _read_audio(args.audio)
    out_dir = Path(args.out)
    print(f"diarize preset={args.preset} {geom}", flush=True)

    calls = _install_compress_hooks(m)
    segs, probs = m.diarize(audio=[audio], batch_size=1, sample_rate=sr, include_tensor_outputs=True)
    p = _f32(torch.as_tensor(np.asarray(probs[0])))
    write_tensor("diar.probs", p, "diarize",
                 _src(args.model, "diarize.probs(streaming)", preset=args.preset, streaming_cfg=geom,
                      layout="[T_mel, 8] sigmoid activity at 10 ms, arrival order",
                      n_compressions=len(calls)),
                 out_dir=out_dir)
    if args.dump_compress:
        _write_compress_dump(calls, Path(args.dump_compress))
    write_transcript(out_dir, _segments_text(segs[0]),
                     source=_src(args.model, "diarize.segments(streaming)", preset=args.preset))
    active = int((p.max(0) > 0.5).sum())
    print(f"wrote diar.probs {list(p.shape)} ({active} active speakers, "
          f"{len(calls)} compressions) + {len(segs[0])} segments to {out_dir}")
    return 0


def _load_hf(model: str, revision: str | None):
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    proc = AutoProcessor.from_pretrained(model, revision=revision)
    hf = AutoModelForAudioFrameClassification.from_pretrained(
        model, revision=revision, dtype=torch.float32).eval()
    return proc, hf


def _hf_src(model: str, hook: str, **extra: Any) -> dict[str, Any]:
    return {"framework": "transformers", "model": model, "hook": hook, **extra}


def _write_hf_probs(args, probs: np.ndarray, hook: str, **extra) -> None:
    out_dir = Path(args.out)
    write_tensor("diar.probs", probs, "diarize",
                 _hf_src(args.model, hook, layout="[T_mel, 8] sigmoid activity at 10 ms", **extra),
                 out_dir=out_dir)
    active = int((probs.max(0) > 0.5).sum())
    print(f"wrote HF diar.probs {list(probs.shape)} ({active} active speakers) to {out_dir}")


def cmd_hf_stream(args: argparse.Namespace) -> int:
    proc, hf = _load_hf(args.model, args.revision)
    proc.set_streaming_mode(args.mode)
    audio, sr = _read_audio(args.audio)

    def chunks():
        yield proc(audio[: proc.num_samples_first_audio_chunk], sampling_rate=sr,
                   is_streaming=True, is_first_audio_chunk=True)
        mel_idx = proc.num_mel_frames_per_step
        start = proc.audio_chunk_start(mel_idx)
        while (end := start + proc.num_samples_per_audio_chunk) <= audio.shape[0]:
            yield proc(audio[start:end], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False)
            mel_idx += proc.num_mel_frames_per_step
            start = proc.audio_chunk_start(mel_idx)
        yield proc(audio[start:], sampling_rate=sr, is_streaming=True,
                   is_first_audio_chunk=False, is_last_audio_chunk=True)

    cache, logits, n_steps = None, [], 0
    with torch.inference_mode():
        for inputs in chunks():
            out = hf(**inputs, speaker_cache=cache)
            logits.append(out.logits)
            cache = out.speaker_cache
            n_steps += 1
    probs = _f32(torch.cat(logits, dim=1).sigmoid())
    _write_hf_probs(args, probs, "processor push-audio streaming (per-chunk mel)",
                    streaming_mode=args.mode, n_steps=n_steps,
                    streaming_cfg=dict(zip(("chunk_len", "chunk_right_context"),
                                           proc.streaming_modes[args.mode])))
    return 0


def cmd_hf_offline(args: argparse.Namespace) -> int:
    proc, hf = _load_hf(args.model, args.revision)
    audio, sr = _read_audio(args.audio)
    inputs = proc(audio, sampling_rate=sr)
    with torch.inference_mode():
        logits = hf(**inputs).logits
    probs = _f32(logits.sigmoid())
    _write_hf_probs(args, probs, "whole-recording forward (config offline preset)",
                    streaming_cfg={"chunk_len": hf.config.chunk_length,
                                   "chunk_right_context": hf.config.chunk_right_context,
                                   "fifo_len": hf.config.fifo_length,
                                   "spkcache_update_period": hf.config.speaker_cache_update_period})
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id or path to the .nemo checkpoint")
    p.add_argument("--revision", default=None, help="HF revision (pinned in the golden manifest)")
    p.add_argument("--audio", required=True, help="Path to a 16 kHz mono WAV")
    p.add_argument("--out", required=True, help="Output ref/ directory")
    # Accepted for the validate.py harness contract; a diarizer takes no language.
    p.add_argument("--torch-threads", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--language", default=None, help=argparse.SUPPRESS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Per-stage activations of the first streaming step")
    add_common_args(ep)
    ep.add_argument("--preset", default="very_high_latency", choices=list(PRESETS))
    ep.set_defaults(func=cmd_encoder)
    dp = sub.add_parser("diarize", help="Streaming T x 8 probs + speaker segments")
    add_common_args(dp)
    dp.add_argument("--preset", default="very_high_latency", choices=list(PRESETS),
                    help="Streaming operating point; the C++ side must run the same preset for parity.")
    dp.add_argument("--dump-compress", default=None,
                    help="Directory for per-_compress_spkcache selected indices + preds.")
    dp.set_defaults(func=cmd_diarize)
    hp = sub.add_parser("hf-stream", help="HF transformers push-audio streaming probs")
    add_common_args(hp)
    hp.add_argument("--mode", default="low_latency", choices=HF_STREAMING_MODES)
    hp.set_defaults(func=cmd_hf_stream)
    op = sub.add_parser("hf-offline", help="HF transformers whole-recording probs")
    add_common_args(op)
    op.set_defaults(func=cmd_hf_offline)
    args = p.parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
