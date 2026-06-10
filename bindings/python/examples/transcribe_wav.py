#!/usr/bin/env python3
"""Transcribe a 16 kHz mono WAV with the transcribe_cpp bindings.

    uv run examples/transcribe_wav.py <model.gguf> <audio.wav> [--timestamps segment]

WAV decoding here uses only the stdlib ``wave`` module — the binding itself
takes float32 PCM and does not decode containers or resample. Point
TRANSCRIBE_LIBRARY at a libtranscribe shared library, or run from a working tree
with a shared build (cmake -DTRANSCRIBE_BUILD_SHARED=ON).
"""

from __future__ import annotations

import argparse
import array
import sys
import wave

import transcribe_cpp


def load_wav_mono16k(path: str) -> array.array:
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        framerate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if sample_width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got {sample_width * 8}-bit")
    if framerate != 16000:
        raise SystemExit(
            f"{path}: expected 16 kHz, got {framerate} Hz — resample first, e.g. "
            f"ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
        )

    pcm16 = array.array("h")
    pcm16.frombytes(frames)
    if sys.byteorder == "big":
        pcm16.byteswap()

    if n_channels > 1:  # downmix to mono
        mono = array.array("h", [0]) * (len(pcm16) // n_channels)
        for i in range(len(mono)):
            acc = sum(pcm16[i * n_channels + c] for c in range(n_channels))
            mono[i] = int(acc / n_channels)
        pcm16 = mono

    return array.array("f", (s / 32768.0 for s in pcm16))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--timestamps", default="none",
                    choices=["none", "auto", "segment", "word", "token"])
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    print(f"transcribe_cpp {transcribe_cpp.__version__} | native "
          f"{transcribe_cpp.native_version()} ({transcribe_cpp.native_commit()})")
    print(f"library: {transcribe_cpp.library_path()}")

    pcm = load_wav_mono16k(args.audio)

    with transcribe_cpp.Model(args.model, backend=args.backend) as model:
        caps = model.capabilities
        print(f"model: {model.arch}/{model.variant} on {model.backend} | "
              f"max_ts={caps.max_timestamp_kind} translate={caps.supports_translate}")
        with model.session() as session:
            result = session.run(
                pcm, timestamps=args.timestamps, language=args.language
            )

    print(f"\nlanguage: {result.language or '(n/a)'} | "
          f"timestamps: {result.timestamp_kind} | "
          f"encode {result.timings.encode_ms:.0f} ms")
    print(f"\n{result.text.strip()}\n")

    if result.segments and result.timestamp_kind != "none":
        for seg in result.segments:
            print(f"  [{seg.t0_ms / 1000:6.2f} -> {seg.t1_ms / 1000:6.2f}]  "
                  f"{seg.text.strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
