#!/usr/bin/env python3
"""Stream a 16 kHz mono WAV through a streaming model, chunk by chunk.

    uv run examples/stream_wav.py <model.gguf> <audio.wav> [--chunk-ms 250]

Demonstrates the streaming surface: ``Session.stream()`` returns a Stream you
``feed()`` PCM chunks into; ``text()`` exposes the two UI-facing views —
``committed`` (append-only, stable: safe to act on) and ``tentative`` (the
volatile suffix that may still be revised). ``display`` is what a live UI
shows. ``finalize()`` flushes the last hypothesis when the audio ends, and
context-manager exit resets the session back to idle.

The model must advertise streaming (``Model.capabilities.supports_streaming``)
— e.g. moonshine-streaming-tiny; whisper does not stream. ``--realtime``
sleeps each chunk's duration so the schedule (and any family decode
throttles) behaves like live microphone input.
"""

from __future__ import annotations

import argparse
import array
import sys
import time
import wave

import transcribe_cpp


def load_wav_mono16k(path: str) -> array.array:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit(
                f"{path}: expected 16 kHz 16-bit mono — resample first, e.g. "
                f"ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
            )
        pcm16 = array.array("h")
        pcm16.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm16.byteswap()
    return array.array("f", (s / 32768.0 for s in pcm16))


def render(text: "transcribe_cpp.StreamText") -> None:
    # One status line: committed text plain, tentative dimmed. \r-rewrite is
    # enough for a demo; a real UI keys off StreamUpdate.committed_changed /
    # tentative_changed to redraw minimally.
    line = f"{text.committed}\x1b[2m{text.tentative}\x1b[0m"
    print(f"\r\x1b[K{line}", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("--chunk-ms", type=int, default=250,
                    help="feed size in milliseconds (default: 250)")
    ap.add_argument("--realtime", action="store_true",
                    help="sleep each chunk's duration (simulate a live mic)")
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    pcm = load_wav_mono16k(args.audio)
    chunk = max(1, int(16000 * args.chunk_ms / 1000))

    with transcribe_cpp.Model(args.model) as model:
        caps = model.capabilities
        if not caps.supports_streaming:
            raise SystemExit(
                f"{model.arch}/{model.variant} does not support streaming — "
                "use a streaming model (e.g. moonshine-streaming-tiny)"
            )
        print(f"model: {model.arch}/{model.variant} on {model.backend} | "
              f"feeding {args.chunk_ms} ms chunks"
              f"{' at realtime pace' if args.realtime else ''}")

        with model.session() as session:
            with session.stream(language=args.language) as stream:
                for i in range(0, len(pcm), chunk):
                    update = stream.feed(pcm[i:i + chunk])
                    if update.committed_changed or update.tentative_changed:
                        render(stream.text())
                    if args.realtime:
                        time.sleep(args.chunk_ms / 1000)
                update = stream.finalize()
                final = stream.text()
                render(final)
                print()  # land the cursor after the live line
                print(f"\nfinal ({update.audio_committed_ms / 1000:.1f}s audio, "
                      f"revision {stream.revision}):\n{final.committed.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
