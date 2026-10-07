"""DIARIZE role against real models: roles, diarize info, DiarizeSession.run
and its Sortformer preset extension. The locking / Busy / close / cancel
rules are pinned model-free in test_compute_lock.py."""

from __future__ import annotations

import gc
import math

import pytest

import transcribe_cpp as t
from conftest import SAMPLES, load_wav


@pytest.fixture(scope="module")
def mix_pcm():
    return load_wav(SAMPLES / "sortformer-2spk-mix.wav")


def _turns(rows):
    return [(r.t0_ms, r.t1_ms, r.speaker_id) for r in rows]


def test_sortformer_roles_and_info(sortformer_model_path):
    with t.Model(sortformer_model_path) as model:
        assert model.roles == {t.Role.DIARIZE}
        assert model.diarize_info == t.DiarizeInfo(sample_rate=16000, max_speakers=4)


def test_sortformer_rejects_asr(sortformer_model_path):
    with t.Model(sortformer_model_path) as model:
        with pytest.raises(t.UnsupportedRole):
            model.capabilities
        with pytest.raises(t.UnsupportedRole):
            model.session()
        with model.diarize_session():  # the DIARIZE role still opens
            pass


def test_asr_only_model_rejects_diarize(model_path):
    with t.Model(model_path) as model:
        assert model.roles == {t.Role.ASR}
        with pytest.raises(t.UnsupportedRole):
            model.diarize_info
        with pytest.raises(t.UnsupportedRole):
            model.diarize_session()


def test_diarize_run_returns_turns_per_preset(sortformer_model_path, mix_pcm):
    # Exact segments are pinned in C (tests/sortformer_diarize_unit.cpp); here
    # only that rows come back well-formed and the preset reaches the run.
    with t.Model(sortformer_model_path, backend="cpu") as model:
        with model.diarize_session() as d:
            default = d.run(mix_pcm)
            low = d.run(mix_pcm, family=t.SortformerDiarizeOptions(preset="low_latency"))
            timings = d.timings
    assert default
    for r in default:
        assert 1 <= r.speaker_id <= 4 and r.t0_ms < r.t1_ms and math.isnan(r.p)
    assert _turns(low) != _turns(default)
    assert timings.encode_ms > 0


def test_bad_preset_rejected(sortformer_model_path, mix_pcm):
    class OutOfRange(t.SortformerDiarizeOptions):
        _presets = {**t.SortformerDiarizeOptions._presets, "bogus": 99}

    with t.Model(sortformer_model_path) as model, model.diarize_session() as d:
        with pytest.raises(t.InvalidArgument):
            d.run(mix_pcm, family=OutOfRange(preset="bogus"))  # type: ignore[arg-type]
        with pytest.raises(t.InvalidArgument, match="slot"):
            d.run(mix_pcm, family=t.WhisperRunOptions())
        assert d.run(mix_pcm)  # still usable


def test_diarize_session_keeps_model_alive(sortformer_model_path, mix_pcm):
    d = t.Model(sortformer_model_path).diarize_session()
    gc.collect()
    assert d.run(mix_pcm)
    d.close()
    with pytest.raises(t.TranscribeError, match="closed"):
        d.run(mix_pcm)
