#!/usr/bin/env python3
"""
dump_reference_sortformer_nemo.py - Sortformer streaming diarizer reference
tensors from NVIDIA NeMo (the canonical reference).

Sortformer is a frame-level end-to-end neural diarizer (arch pattern
`encoder-diarizer`), NOT a transcription model. There is no tokenizer and
no text output: the reference "behavioral artifact" is the T x 4 speaker
activity probability matrix plus the derived speaker segments.

Pipeline (offline / full-context forward):
    preprocessor (mel, 128 x T_mel)
    -> encoder            ConformerEncoder / NEST-FastConformer, d=512, x8 subsample
    -> encoder_proj       Linear 512 -> 192  (sortformer_modules.encoder_proj)
    -> transformer_encoder  18L Transformer, d=192
    -> hidden_to_spks     -> sigmoid -> preds [T, 4]

Two reference paths are dumped:
  * `encoder`  runs the offline m.forward() with hooks and dumps the mel and
               the per-stage encoder activations plus the offline preds. This
               is the non-stateful reference the C++ encoder must match first.
  * `diarize`  runs the streaming m.diarize() (the runtime target) and dumps
               the T x 4 probability matrix + the speaker segments.

For short clips (T frames < chunk_len, e.g. the 12s oracle mix = 150 frames
< 188) the streaming path is a single chunk, so offline preds ~= streaming
probs; longer audio exercises the AOSC/FIFO speaker cache and the two will
diverge.

Tensor output uses the shared contract via scripts.lib.ref_dump
(write_tensor records rms / p99_abs for Stage 6 magnitude-aware tolerances).

    uv run --project scripts/envs/sortformer \
      scripts/dump_reference_sortformer_nemo.py encoder \
      --model nvidia/diar_streaming_sortformer_4spk-v2.1 \
      --audio samples/sortformer-2spk-mix.wav \
      --out build/validate/sortformer/diar_streaming_sortformer_4spk-v2.1/sortformer-2spk-mix/encoder/ref

    uv run --project scripts/envs/sortformer \
      scripts/dump_reference_sortformer_nemo.py diarize \
      --model nvidia/diar_streaming_sortformer_4spk-v2.1 \
      --audio samples/sortformer-2spk-mix.wav \
      --out build/validate/sortformer/diar_streaming_sortformer_4spk-v2.1/sortformer-2spk-mix/diarize/ref
"""

from __future__ import annotations

import argparse
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


def _load(model: str):
    from nemo.collections.asr.models import SortformerEncLabelModel

    if model.endswith(".nemo") or Path(model).exists():
        m = SortformerEncLabelModel.restore_from(restore_path=model, map_location="cpu", strict=False)
    else:
        m = SortformerEncLabelModel.from_pretrained(model, map_location="cpu")
    m.eval()
    return m


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _to_2d_time_major(a: np.ndarray) -> np.ndarray:
    """Squeeze batch and return [T, D]. ConformerEncoder emits [B, D, T];
    the transformer path emits [B, T, D]. Detect by which axis is the
    known speaker/model dim is impossible generically, so we squeeze batch
    and, when the first (non-batch) axis looks like a channel dim (512/192)
    and the second is longer, transpose to time-major."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2 and a.shape[0] in (192, 512) and a.shape[1] != a.shape[0]:
        a = a.T  # [D, T] -> [T, D]
    return np.ascontiguousarray(a.astype(np.float32))


def _src(model: str, hook: str) -> dict[str, Any]:
    return {"framework": "nemo", "model": model, "hook": hook}


def cmd_encoder(args: argparse.Namespace) -> int:
    m = _load(args.model)
    audio, sr = _read_audio(args.audio)
    out_dir = Path(args.out)
    sig = torch.tensor(audio).unsqueeze(0)
    length = torch.tensor([audio.shape[0]])

    # Mel (frontend) — dump native NeMo layout [n_mels, T_mel].
    with torch.no_grad():
        mel, mel_len = m.preprocessor(input_signal=sig, length=length)
    write_tensor("enc.mel.in", np.ascontiguousarray(mel[0].numpy().astype(np.float32)),
                 "frontend", _src(args.model, "preprocessor.out"), out_dir=out_dir)

    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = t.detach().to(torch.float32).cpu()
        return hook

    handles = [
        m.encoder.register_forward_hook(grab("fastconformer")),
        m.sortformer_modules.encoder_proj.register_forward_hook(grab("encoder_proj")),
        m.transformer_encoder.register_forward_hook(grab("transformer")),
    ]
    try:
        with torch.no_grad():
            preds = m.forward(audio_signal=sig, audio_signal_length=length)
    finally:
        for h in handles:
            h.remove()

    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    write_tensor("enc.fastconformer.out", _to_2d_time_major(captured["fastconformer"].numpy()),
                 "encoder", _src(args.model, "encoder.out"), out_dir=out_dir)
    write_tensor("enc.encoder_proj.out", _to_2d_time_major(captured["encoder_proj"].numpy()),
                 "encoder", _src(args.model, "encoder_proj.out"), out_dir=out_dir)
    write_tensor("enc.transformer.out", _to_2d_time_major(captured["transformer"].numpy()),
                 "encoder", _src(args.model, "transformer_encoder.out"), out_dir=out_dir)
    write_tensor("diar.preds_offline", _to_2d_time_major(preds.detach().numpy()),
                 "encoder", _src(args.model, "forward.preds(offline,full-context)"), out_dir=out_dir)
    print(f"wrote encoder-stage tensors to {out_dir}")
    return 0


# Streaming presets (frames are 80 ms). Mirror scripts/diar/run_reference_
# sortformer_nemo.py PRESETS plus a `small` preset that forces multi-chunk +
# AOSC compression on the short oracle clip (diar.probs parity). Keep in sync
# with the C++ preset table in src/arch/sortformer/stream.cpp.
_PRESETS = {
    "very_high_latency": dict(chunk_len=340, chunk_right_context=40, fifo_len=40,
                              spkcache_update_period=300, spkcache_len=188),
    "high_latency": dict(chunk_len=124, chunk_right_context=1, fifo_len=124,
                         spkcache_update_period=124, spkcache_len=188),
    "low_latency": dict(chunk_len=6, chunk_right_context=7, fifo_len=188,
                        spkcache_update_period=144, spkcache_len=188),
    "small": dict(chunk_len=20, chunk_right_context=1, fifo_len=10,
                  spkcache_update_period=20, spkcache_len=24),
}


def _apply_preset(m, preset: str) -> None:
    cfg = _PRESETS[preset]
    sm = m.sortformer_modules
    sm.chunk_len = cfg["chunk_len"]
    sm.chunk_right_context = cfg["chunk_right_context"]
    sm.fifo_len = cfg["fifo_len"]
    sm.spkcache_update_period = cfg["spkcache_update_period"]
    sm.spkcache_len = cfg["spkcache_len"]
    if hasattr(sm, "_check_streaming_parameters"):
        sm._check_streaming_parameters()


def _install_compress_hooks(m) -> list[dict]:
    """Wrap sortformer_modules._compress_spkcache / _get_topk_indices to record,
    per compression call, the selected (sorted) frame indices, the is_disabled
    mask, and the gathered spkcache_preds. batch=1 in the CLI path, so squeeze
    the batch dim. Returns the list the hooks append to (one dict per call)."""
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
        # input preds ([1, n_frames, n_spk]) drive the whole selection; capture
        # them (positionally emb_seq, preds, mean_sil_emb, ... or by kw).
        in_preds = kw["preds"] if "preds" in kw else a[1]
        in_preds = in_preds[0].detach().cpu().numpy().astype(np.float32)
        spkcache, spkcache_preds, spk_perm = orig_compress(*a, **kw)
        if calls:  # the just-appended call from wrapped_topk
            calls[-1]["input_preds"] = in_preds
            calls[-1]["spkcache_preds"] = spkcache_preds[0].detach().cpu().numpy().astype(np.float32)
        return spkcache, spkcache_preds, spk_perm

    sm._get_topk_indices = wrapped_topk
    sm._compress_spkcache = wrapped_compress
    return calls


def _write_compress_dump(calls: list[dict], dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(calls):
        np.save(dump_dir / f"compress.{k:03d}.topk_indices.npy", c["topk_indices"])
        np.save(dump_dir / f"compress.{k:03d}.is_disabled.npy", c["is_disabled"])
        if "spkcache_preds" in c:
            np.save(dump_dir / f"compress.{k:03d}.spkcache_preds.npy", c["spkcache_preds"])
        if "input_preds" in c:
            np.save(dump_dir / f"compress.{k:03d}.input_preds.npy", c["input_preds"])
    summary = {"n_calls": len(calls), "n_frames_in": [c["n_frames"] for c in calls]}
    (dump_dir / "compress.summary.json").write_text(_json_dumps(summary))
    print(f"wrote {len(calls)} _compress_spkcache dumps to {dump_dir}")


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2)


def cmd_diarize(args: argparse.Namespace) -> int:
    m = _load(args.model)
    audio, sr = _read_audio(args.audio)
    out_dir = Path(args.out)

    if getattr(args, "preset", None):
        _apply_preset(m, args.preset)
        print(f"diarize preset={args.preset} {_PRESETS[args.preset]}", flush=True)

    compress_calls = _install_compress_hooks(m) if getattr(args, "dump_compress", None) else None

    segs, probs = m.diarize(audio=[audio], batch_size=1, sample_rate=sr,
                            include_tensor_outputs=True)

    if compress_calls is not None:
        _write_compress_dump(compress_calls, Path(args.dump_compress))
    p = np.asarray(probs[0], dtype=np.float32)
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    p = np.ascontiguousarray(p.astype(np.float32))  # [T, 4]
    write_tensor("diar.probs", p, "diarize",
                 {**_src(args.model, "diarize.probs(streaming)"),
                  "streaming_cfg": {
                      "chunk_len": int(m.sortformer_modules.chunk_len),
                      "fifo_len": int(m.sortformer_modules.fifo_len),
                      "spkcache_len": int(m.sortformer_modules.spkcache_len),
                      "spkcache_update_period": int(m.sortformer_modules.spkcache_update_period),
                      "chunk_right_context": int(getattr(m.sortformer_modules, "chunk_right_context", 0)),
                  }},
                 out_dir=out_dir)

    # Behavioral artifact: speaker segments, as RTTM-like text (no transcript
    # exists for a diarizer). "start end speaker" per line, emission order.
    seg_lines = [str(s).strip() for s in segs[0]]
    write_transcript(out_dir, "\n".join(seg_lines),
                     source=_src(args.model, "diarize.segments(streaming)"))
    print(f"wrote diarize probs [{p.shape[0]}x{p.shape[1]}] + {len(seg_lines)} segments to {out_dir}")
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id or path to a .nemo checkpoint")
    p.add_argument("--audio", required=True, help="Path to a 16 kHz mono WAV")
    p.add_argument("--out", required=True, help="Output ref/ directory")
    # Accepted for the validate.py harness contract; a diarizer takes no
    # language, and torch threading does not affect the dumped tensors.
    p.add_argument("--torch-threads", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--language", default=None, help=argparse.SUPPRESS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Dump mel + encoder-stage activations (offline forward)")
    add_common_args(ep)
    ep.set_defaults(func=cmd_encoder)
    dp = sub.add_parser("diarize", help="Dump streaming T x 4 probs + speaker segments")
    add_common_args(dp)
    dp.add_argument("--preset", default=None, choices=list(_PRESETS),
                    help="Streaming operating point (default: checkpoint cfg). Must match the C++ "
                         "TRANSCRIBE_SORTFORMER_STREAM_PRESET for parity.")
    dp.add_argument("--dump-compress", default=None,
                    help="Directory to write per-_compress_spkcache selected frame indices + preds "
                         "(residual-uncertainty #1: compression internal-state parity vs C++).")
    dp.set_defaults(func=cmd_diarize)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
