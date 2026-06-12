"""The shipped examples must actually run.

``transcribe_wav.py``'s WAV validation happens BEFORE any model load, so the
negative paths (wrong sample rate, with the actionable resample hint) are
model-free; the happy path and the streaming example are model-gated like
every other model test. Each example runs as a subprocess — the way a user
runs it — with TRANSCRIBE_LIBRARY pinned to the library this suite loaded.
"""

from __future__ import annotations

import array
import math
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

import transcribe_cpp as t

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run_example(script: str, *args: str) -> "subprocess.CompletedProcess[str]":
    env = os.environ.copy()
    env["TRANSCRIBE_LIBRARY"] = t.library_path()
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True, text=True, timeout=300, env=env,
    )


def write_wav(path: Path, *, rate: int, channels: int = 1,
              seconds: float = 0.25) -> None:
    n = int(rate * seconds)
    pcm = array.array("h", (
        int(8000 * math.sin(2 * math.pi * 440 * i / rate))
        for i in range(n * channels)
    ))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# --- model-free: validation paths -------------------------------------------


def test_transcribe_wav_rejects_wrong_sample_rate(tmp_path):
    bad = tmp_path / "8k.wav"
    write_wav(bad, rate=8000)
    proc = run_example("transcribe_wav.py", "unused-model.gguf", str(bad))
    assert proc.returncode != 0
    # The error must be actionable: name the rate and how to fix it.
    assert "8000" in proc.stderr
    assert "resample" in proc.stderr.lower()
    assert "ffmpeg" in proc.stderr


def test_transcribe_wav_rejects_wrong_sample_width(tmp_path):
    bad = tmp_path / "8bit.wav"
    n = 4000
    with wave.open(str(bad), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit
        w.setframerate(16000)
        w.writeframes(bytes(n))
    proc = run_example("transcribe_wav.py", "unused-model.gguf", str(bad))
    assert proc.returncode != 0
    assert "16-bit" in proc.stderr


# --- model-gated: end-to-end -------------------------------------------------


def test_transcribe_wav_happy_path(model_path, audio_path):
    proc = run_example("transcribe_wav.py", str(model_path), str(audio_path),
                       "--timestamps", "segment")
    assert proc.returncode == 0, proc.stderr
    assert "country" in proc.stdout.lower(), proc.stdout


def test_transcribe_wav_downmixes_stereo(model_path, tmp_path, audio_pcm):
    # Duplicate the mono canary into 2 interleaved channels: the example's
    # downmix path must produce the same transcript as the mono original.
    stereo = tmp_path / "stereo.wav"
    pcm16 = array.array("h", (max(-32768, min(32767, int(s * 32768.0)))
                              for s in audio_pcm))
    inter = array.array("h", (0 for _ in range(2 * len(pcm16))))
    inter[0::2] = pcm16
    inter[1::2] = pcm16
    with wave.open(str(stereo), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(inter.tobytes())
    proc = run_example("transcribe_wav.py", str(model_path), str(stereo))
    assert proc.returncode == 0, proc.stderr
    assert "country" in proc.stdout.lower(), proc.stdout


# --- stream_wav.py ------------------------------------------------------------


def test_stream_wav_rejects_non_16k(tmp_path):
    bad = tmp_path / "8k.wav"
    write_wav(bad, rate=8000)
    proc = run_example("stream_wav.py", "unused-model.gguf", str(bad))
    assert proc.returncode != 0
    assert "resample" in proc.stderr.lower()


def test_stream_wav_rejects_non_streaming_model(model_path, audio_path):
    # whisper (the default canary) does not stream: the example must say so
    # clearly instead of tracebacking.
    proc = run_example("stream_wav.py", str(model_path), str(audio_path))
    if proc.returncode == 0:
        pytest.skip("default canary unexpectedly supports streaming")
    assert "does not support streaming" in proc.stderr


def test_stream_wav_happy_path(streaming_model_path, audio_path):
    proc = run_example("stream_wav.py", str(streaming_model_path),
                       str(audio_path), "--chunk-ms", "500")
    assert proc.returncode == 0, proc.stderr
    assert "country" in proc.stdout.lower(), proc.stdout
