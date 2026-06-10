"""Streaming + cancellation tests.

The streaming *gate* runs against the default (non-streaming) whisper model and
needs no streaming checkpoint. The real streaming and cancellation tests need
moonshine-streaming-tiny and ``skip`` when it is absent.
"""

from __future__ import annotations

import pytest

import transcribe_cpp as t


def test_streaming_gate(model_path):
    with t.Model(model_path) as model:
        if model.capabilities.supports_streaming:
            pytest.skip("default model supports streaming; gate not exercised")
        with model.session() as session:
            with pytest.raises(t.NotImplementedByModel):
                session.stream()


def test_streaming_real(streaming_model_path, audio_pcm):
    with t.Model(streaming_model_path) as model:
        assert model.capabilities.supports_streaming
        with model.session() as session, session.stream() as stream:
            for i in range(0, len(audio_pcm), 16000):  # ~1s chunks
                stream.feed(audio_pcm[i : i + 16000])
            update = stream.finalize()
            committed = stream.text().committed
            revision, state = stream.revision, stream.state
    assert update.is_final, update
    assert state == "finished", state
    assert revision >= 1
    assert "country" in committed.lower(), committed


def test_cancellation_aborts_in_flight_feed(streaming_model_path, audio_pcm):
    with t.Model(streaming_model_path) as model, model.session() as session:
        with session.stream() as stream:
            session.cancel()  # request abort, then feed: must abort in-flight
            try:
                stream.feed(audio_pcm)
            except t.TranscribeError:
                assert session.was_aborted, "was_aborted should be True after cancel()"
                return
    raise AssertionError("cancel() did not abort the feed")
