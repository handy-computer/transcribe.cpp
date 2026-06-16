"""Shared fixtures for the transcribe_cpp pytest suite.

Two tiers of tests:

  - **No-model tests** (test_abi.py) need only an importable package, which
    means a loadable native library. Importing ``transcribe_cpp`` already runs
    the loader, the ABI layout check, and the version gate; these tests pin that
    surface explicitly. They always run in CI's shared-build lane.

  - **Model tests** (test_transcribe.py, test_streaming.py) need a real GGUF and
    audio on disk. They ``skip`` (never fail) when the asset is absent, so the
    suite is honest on a machine without the multi-GB checkpoints.

Asset resolution mirrors the old stdlib smoke runner: a default
whisper-tiny.en + samples/jfk.wav in the repo, each overridable by
``TRANSCRIBE_SMOKE_MODEL`` / ``TRANSCRIBE_SMOKE_AUDIO``.
"""

from __future__ import annotations

import array
import os
import sys
import wave
from pathlib import Path

import pytest

# tests/ -> python/ -> bindings/ -> repo root.
REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "samples"

DEFAULT_MODEL = REPO / "models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf"
DEFAULT_AUDIO = SAMPLES / "jfk.wav"
STREAMING_MODEL = (
    REPO / "models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf"
)
# A streaming model with a language-prompt dictionary: its C++ implementation
# re-reads run_params.language on every feed, which is what the params
# copy-out regression tests exercise.
PROMPTED_STREAMING_MODEL = (
    REPO
    / "models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
)
# Per-family streaming-extension canaries. These exercise the parakeet/voxtral
# stream-extension happy path (materialize -> accept -> begin -> feed). Not in
# the CI fetch-canary set (parakeet is added once the canary repos exist;
# voxtral is local-only — ~2.5 GB Q4_K_M is too heavy for CI), so each gates on
# its own env var / in-repo GGUF and skips cleanly when absent.
PARAKEET_STREAM_MODEL = (
    REPO
    / "models/nemotron-speech-streaming-en-0.6b"
    / "nemotron-speech-streaming-en-0.6b-Q8_0.gguf"
)
PARAKEET_BUFFERED_MODEL = (
    REPO / "models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf"
)
VOXTRAL_MODEL = (
    REPO
    / "models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf"
)


def load_wav(path: Path) -> "array.array":
    """Read a 16 kHz mono 16-bit WAV into a float32 ``array`` in [-1, 1)."""
    with wave.open(str(path), "rb") as w:
        assert (
            w.getsampwidth() == 2
            and w.getframerate() == 16000
            and w.getnchannels() == 1
        ), f"{path} must be 16 kHz 16-bit mono"
        pcm16 = array.array("h")
        pcm16.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm16.byteswap()
    return array.array("f", (s / 32768.0 for s in pcm16))


@pytest.fixture(scope="session")
def transcribe_cpp():
    """The imported package (importing it exercises load + ABI + version gate)."""
    import transcribe_cpp

    return transcribe_cpp


@pytest.fixture(scope="session")
def model_path() -> Path:
    override = os.environ.get("TRANSCRIBE_SMOKE_MODEL")
    path = Path(override) if override else DEFAULT_MODEL
    if not path.is_file():
        pytest.skip(f"model not present: {path} (set TRANSCRIBE_SMOKE_MODEL)")
    return path


@pytest.fixture(scope="session")
def audio_path() -> Path:
    override = os.environ.get("TRANSCRIBE_SMOKE_AUDIO")
    path = Path(override) if override else DEFAULT_AUDIO
    if not path.is_file():
        pytest.skip(f"audio not present: {path} (set TRANSCRIBE_SMOKE_AUDIO)")
    return path


@pytest.fixture(scope="session")
def audio_pcm(audio_path: Path) -> "array.array":
    return load_wav(audio_path)


@pytest.fixture(scope="session")
def streaming_model_path() -> Path:
    override = os.environ.get("TRANSCRIBE_SMOKE_STREAMING_MODEL")
    path = Path(override) if override else STREAMING_MODEL
    if not path.is_file():
        pytest.skip(
            f"streaming model not present: {path} "
            "(set TRANSCRIBE_SMOKE_STREAMING_MODEL)"
        )
    return path


@pytest.fixture(scope="session")
def prompted_streaming_model_path() -> Path:
    override = os.environ.get("TRANSCRIBE_SMOKE_PROMPTED_MODEL")
    path = Path(override) if override else PROMPTED_STREAMING_MODEL
    if not path.is_file():
        pytest.skip(
            f"prompted streaming model not present: {path} "
            "(set TRANSCRIBE_SMOKE_PROMPTED_MODEL)"
        )
    return path


def _family_model(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    path = Path(override) if override else default
    if not path.is_file():
        pytest.skip(f"model not present: {path} (set {env_var})")
    return path


@pytest.fixture(scope="session")
def parakeet_stream_model_path() -> Path:
    """Cache-aware parakeet streaming canary (accepts PARAKEET_STREAM)."""
    return _family_model("TRANSCRIBE_SMOKE_PARAKEET_STREAM_MODEL", PARAKEET_STREAM_MODEL)


@pytest.fixture(scope="session")
def parakeet_buffered_model_path() -> Path:
    """Chunked/buffered parakeet streaming canary (accepts PARAKEET_BUFFERED_STREAM)."""
    return _family_model(
        "TRANSCRIBE_SMOKE_PARAKEET_BUFFERED_MODEL", PARAKEET_BUFFERED_MODEL
    )


@pytest.fixture(scope="session")
def voxtral_model_path() -> Path:
    """Voxtral realtime streaming canary (accepts VOXTRAL_REALTIME_STREAM)."""
    return _family_model("TRANSCRIBE_SMOKE_VOXTRAL_MODEL", VOXTRAL_MODEL)
