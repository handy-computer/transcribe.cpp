"""VAD role on the checked-in toy silero_vad fixture (random weights:
probabilities are structural only): run, and stream parity with run."""

from __future__ import annotations

import array

import pytest

import transcribe_cpp as t
from conftest import REPO

VAD_TOY = REPO / "tests/fixtures/arch_silero_vad.gguf"


def _noise(n: int, seed: int = 7) -> array.array:
    out = array.array("f")
    s = seed | 1
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(((s >> 8) & 0xFFFFFF) / 16777216.0 - 0.5)
    return out


def test_toy_run_and_stream():
    if not VAD_TOY.exists():
        pytest.skip(f"fixture not present: {VAD_TOY}")
    pcm = _noise(16000 + 100)  # 31 full frames + a padded tail
    with t.Model(VAD_TOY, backend="cpu") as model:
        assert model.roles == {t.Role.VAD}
        assert model.vad_info == t.VadInfo(sample_rate=16000, frame_samples=512)
        with model.vad_session(n_threads=1) as vad:
            r = vad.run(pcm, threshold=0.0, min_speech_ms=0)
            assert len(r.probs) == 32
            assert r.segments == (t.VadSegment(0, len(pcm)),)
            probs: list = []
            for pos in range(0, len(pcm), 700):
                probs.extend(vad.feed(pcm[pos:pos + 700]).probs)
            probs.extend(vad.flush().probs)
            assert tuple(probs) == r.probs
