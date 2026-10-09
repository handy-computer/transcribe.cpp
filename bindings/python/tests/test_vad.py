"""VAD role: roles, info, VadSession.run / stream and VadIterator. Runs on the
checked-in toy silero_vad fixture (random weights: probabilities are
structural only). The locking / Busy rules are pinned model-free in
test_compute_lock.py."""

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


@pytest.fixture(scope="module")
def model():
    if not VAD_TOY.exists():
        pytest.skip(f"fixture not present: {VAD_TOY}")
    with t.Model(VAD_TOY, backend="cpu") as m:
        yield m


def test_toy_roles_info(model):
    assert model.roles == {t.Role.VAD}
    assert model.vad_info == t.VadInfo(sample_rate=16000, frame_samples=512)
    with pytest.raises(t.UnsupportedRole):
        model.session()
    with pytest.raises(t.UnsupportedRole):
        model.langid_session()


def test_langid_model_rejects_vad(langid_toy_model_path):
    with t.Model(langid_toy_model_path, backend="cpu") as m:
        with pytest.raises(t.UnsupportedRole):
            m.vad_info
        with pytest.raises(t.UnsupportedRole):
            m.vad_session()


def test_toy_run_contract(model):
    pcm = _noise(16000 + 100)  # 31 full frames + a padded tail
    with model.vad_session(n_threads=1) as vad:
        r = vad.run(pcm)
        assert len(r.probs) == 32 and r.first_frame == 0
        assert all(0.0 <= p <= 1.0 for p in r.probs)
        for seg in r.segments:
            assert 0 <= seg.start_sample < seg.end_sample <= len(pcm)
        # threshold 0 marks every frame as speech: one segment over the clip.
        everything = vad.run(pcm, threshold=0.0, min_speech_ms=0)
        assert everything.segments == (t.VadSegment(0, len(pcm)),)
        assert everything.probs == r.probs
        with pytest.raises(t.InvalidArgument):
            vad.run(pcm, threshold=2.0)
        bad = array.array("f", pcm)
        bad[3] = float("nan")
        with pytest.raises(t.InvalidArgument):
            vad.run(bad)
        assert vad.timings.encode_ms > 0.0


def test_toy_stream_equals_offline(model):
    pcm = _noise(16000 + 100)
    with model.vad_session(n_threads=1) as vad:
        offline = vad.run(pcm).probs
        probs: list = []
        pos = 0
        for size in (100, 0, 700, 512, 3000, 11788):
            r = vad.feed(pcm[pos:pos + size])
            assert r.first_frame == len(probs) and r.segments == ()
            probs.extend(r.probs)
            pos += size
        assert pos == len(pcm) and len(probs) == 31
        tail = vad.flush()
        assert tail.first_frame == 31 and len(tail.probs) == 1
        probs.extend(tail.probs)
        assert tuple(probs) == offline
        # flush ended the stream; reset drops a partial one.
        assert vad.feed(pcm[:600]).first_frame == 0
        vad.reset()
        assert vad.feed(pcm[:512]).first_frame == 0


def test_iterator_events():
    with t.VadIterator(512, threshold=0.5, min_silence_ms=0, speech_pad_ms=0) as it:
        assert it.feed([0.1, 0.9]) == (t.VadEvent("start", 512),)
        assert it.triggered and it.current_sample == 1024
        assert it.feed([]) == ()
        events = it.feed([0.1, 0.1])
        assert events and events[0].type == "end" and not it.triggered
        with pytest.raises(t.InvalidArgument):
            it.feed([1.5])
        it.reset()
        assert it.current_sample == 0
    with pytest.raises(t.InvalidArgument):
        t.VadIterator(0)
    # neg_threshold is passed through: 0.3 is silence by default, speech at 0.2.
    with t.VadIterator(512, neg_threshold=0.2, min_silence_ms=0) as it:
        it.feed([0.9, 0.3, 0.3])
        assert it.triggered
    with pytest.raises(t.InvalidArgument):
        t.VadIterator(512, neg_threshold=0.6)
