"""Handle-lifetime tests: close ordering, use-after-close, idempotency.

The C contract says a model may only be freed after every derived session
(`transcribe_model_free` docs). The binding enforces that ordering itself:
``Model.close()`` closes live sessions first, and every closed handle turns
later calls into TranscribeError instead of native use-after-free.
"""

from __future__ import annotations

import gc

import pytest

import transcribe_cpp as t


def test_model_close_closes_open_sessions(model_path):
    model = t.Model(model_path)
    session = model.session()
    model.close()  # must close the session first, then free the model
    with pytest.raises(t.TranscribeError, match="closed"):
        session.run([0.0] * 1600)


def test_model_context_exit_with_live_session(model_path, audio_pcm):
    # The session is deliberately NOT closed before the model's context
    # exits — historically a native use-after-free footgun.
    with t.Model(model_path) as model:
        session = model.session()
        result = session.run(audio_pcm)
        assert result.text.strip()
    with pytest.raises(t.TranscribeError, match="closed"):
        session.run(audio_pcm)
    with pytest.raises(t.TranscribeError, match="closed"):
        model.session()


def test_close_is_idempotent(model_path):
    model = t.Model(model_path)
    session = model.session()
    session.close()
    session.close()
    model.close()
    model.close()


def test_session_close_then_model_still_usable(model_path, audio_pcm):
    with t.Model(model_path) as model:
        session = model.session()
        session.close()
        # Closing one session must not affect the model or new sessions.
        with model.session() as fresh:
            assert fresh.run(audio_pcm).text.strip()


def test_session_close_with_active_stream(streaming_model_path, audio_pcm):
    # Closing the session out from under an active stream must surface as a
    # Python error on the next stream call, never a native crash.
    with t.Model(streaming_model_path) as model:
        session = model.session()
        stream = session.stream()
        stream.feed(audio_pcm[:16000])
        session.close()
        with pytest.raises(t.TranscribeError, match="closed"):
            stream.feed(audio_pcm[:16000])


def test_gc_order_session_keeps_model_alive(model_path, audio_pcm):
    session = t.Model(model_path).session()  # model reference dropped at once
    gc.collect()  # the session's strong ref must keep the model alive
    assert session.run(audio_pcm).text.strip()
    session.close()
    gc.collect()
