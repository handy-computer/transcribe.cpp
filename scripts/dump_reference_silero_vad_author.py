#!/usr/bin/env python3
"""
dump_reference_silero_vad_author.py - Silero VAD reference tensors and speech
segments from upstream's own package (silero-vad 6.2.3, TorchScript model).

    uv run --project scripts/envs/silero_vad \
      scripts/dump_reference_silero_vad_author.py encoder \
      --audio samples/jfk.wav \
      --out build/validate/silero_vad/silero-vad-v6.2/jfk/ref

Runs the scoring loop of get_speech_timestamps (reset_states, one
model(chunk, 16000) call per 512-sample frame, the last zero padded) and
writes per-frame tensors stacked over frames, [C, T] activations transposed
to time-major [T, C] as the C++ dumper does (src/arch/silero_vad/model.cpp):

    fe.stft_mag  [n, 4, 129]     enc.0.out [n, 4, 128]   enc.1.out [n, 2, 64]
    enc.2.out    [n, 1, 64]      enc.3.out [n, 1, 128]
    dec.lstm_h   [n, 128]        dec.lstm_c [n, 128]     vad.probs [n]

segments.json holds get_speech_timestamps(audio, model) at default parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.ref_dump import write_tensor  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 512
ENCODER_TENSORS = ("fe.stft_mag", "enc.0.out", "enc.1.out", "enc.2.out", "enc.3.out")


def cmd_encoder(args: argparse.Namespace) -> int:
    import silero_vad
    import soundfile as sf

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    model = silero_vad.load_silero_vad()
    sub = model._model

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1 or int(sr) != SAMPLE_RATE:
        raise SystemExit(f"error: {audio_path} is not {SAMPLE_RATE} Hz mono")
    audio = torch.from_numpy(np.ascontiguousarray(pcm))

    captured: dict[str, list[np.ndarray]] = {k: [] for k in (*ENCODER_TENSORS, "dec.lstm_h", "dec.lstm_c")}

    def tc(t: torch.Tensor) -> np.ndarray:  # [1, C, T] -> [T, C]
        return t.detach()[0].transpose(0, 1).contiguous().numpy().astype(np.float32)

    # TorchScript modules take no forward hooks, so each frame is scored
    # twice: end to end through model(chunk, 16000) exactly as
    # get_speech_timestamps does, and step by step through the scripted
    # submodules in VADRNNJIT.forward order. The two must agree bit for bit,
    # which makes the captured intermediates the reference's own.
    probs = []
    model.reset_states()
    context = torch.zeros(1, sub.context_size_samples)
    state = torch.zeros(2, 1, 128)
    with torch.inference_mode():
        for start in range(0, len(audio), FRAME):
            chunk = audio[start:start + FRAME]
            if len(chunk) < FRAME:
                chunk = torch.nn.functional.pad(chunk, (0, FRAME - len(chunk)))
            p_ref = model(chunk, SAMPLE_RATE)

            x = torch.cat([context, chunk.unsqueeze(0)], dim=1)
            context = x[..., -sub.context_size_samples:]
            y = sub.stft(x)
            captured["fe.stft_mag"].append(tc(y))
            for i in range(4):
                y = getattr(sub.encoder, str(i))(y)
                captured[f"enc.{i}.out"].append(tc(y))
            y, state = sub.decoder(y, state)
            captured["dec.lstm_h"].append(state[0, 0].numpy().astype(np.float32))
            captured["dec.lstm_c"].append(state[1, 0].numpy().astype(np.float32))
            p_step = torch.mean(torch.squeeze(y, 1), [1]).unsqueeze(1)
            if not torch.equal(p_ref, p_step):
                raise SystemExit(f"error: frame {len(probs)}: stepwise {p_step.item()!r} "
                                 f"!= end-to-end {p_ref.item()!r}")
            probs.append(p_ref.item())

    # Cross-check: the replayed loop is exactly get_speech_timestamps.
    model.reset_states()
    segments = silero_vad.get_speech_timestamps(audio, model)
    if segments != silero_vad.get_speech_timestamps_from_probs(probs, audio_length_samples=len(audio)):
        raise SystemExit("error: get_speech_timestamps disagrees with the replayed probabilities")

    source = {
        "kind": "silero-vad-author",
        "framework": "silero-vad",
        "framework_version": silero_vad.__version__,
        "torch_version": torch.__version__,
        "model": "silero_vad.jit (16 kHz sub-model)",
        "device": "cpu",
        "torch_threads": args.torch_threads,
        "model_dtype": "f32",
        "audio": audio_path.name,
        "n_samples": int(len(audio)),
        "n_frames": len(probs),
    }
    for name, values in captured.items():
        stage = "encoder" if name in ENCODER_TENSORS else "decoder"
        write_tensor(name, np.stack(values), stage=stage, source=source, out_dir=out_dir)
    write_tensor("vad.probs", np.asarray(probs, dtype=np.float32), stage="decoder", source=source,
                 out_dir=out_dir)

    (out_dir / "segments.json").write_text(json.dumps(
        {"n_samples": int(len(audio)), "segments": segments, "source": source}, indent=2) + "\n")
    print(f"frames: {len(probs)}  segments: {len(segments)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Silero VAD reference dumper (silero-vad package).")
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Score every frame; dump stage tensors and segments.json")
    ep.add_argument("--model", default="silero-vad", help="Ignored; the model is the pinned package's")
    ep.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    ep.add_argument("--out", required=True, help="Output directory for dumps")
    ep.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads (0 = unchanged)")
    ep.set_defaults(func=cmd_encoder)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
