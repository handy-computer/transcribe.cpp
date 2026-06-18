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
            last_status = stream.last_status
    assert update.is_final, update
    assert state == "finished", state
    assert revision >= 1
    assert last_status is None, last_status
    assert "country" in committed.lower(), committed


def test_streaming_with_language_hint(prompted_streaming_model_path, audio_pcm):
    """Regression: params copy-out contract for streaming.

    Parakeet's prompted streaming re-resolves run_params.language on EVERY
    feed. The caller's params storage (here: ctypes structs local to
    Session.stream()) dies when stream() returns, so the library must have
    copied the string into session-owned storage at begin — a retained
    caller pointer is a use-after-free on the first feed. The ASan lane is
    the real detector; the gc + junk scribbling below makes stale reads
    likelier to also misbehave in a normal build."""
    import gc

    with t.Model(prompted_streaming_model_path) as model:
        caps = model.capabilities
        assert caps.supports_streaming
        assert caps.languages, "prompted model should advertise languages"
        lang = "en" if "en" in caps.languages else caps.languages[0]
        with model.session() as session:
            stream = session.stream(language=lang)
            # The params structs built inside stream() are now dead. Collect
            # and scribble over the freed allocations before the first feed.
            gc.collect()
            junk = [b"\x5a" * 256 for _ in range(4096)]
            with stream:
                for i in range(0, len(audio_pcm), 16000):
                    stream.feed(audio_pcm[i : i + 16000])
                update = stream.finalize()
                committed = stream.text().committed
            del junk
    assert update.is_final
    assert "country" in committed.lower(), committed


def test_streaming_language_hint_second_family(streaming_model_path, audio_pcm):
    """Same contract on moonshine-streaming: it retains a shallow copy of
    run_params for its decode path. No live pointer read today, but the
    retained copy must stay safe to carry — this pins it under ASan."""
    with t.Model(streaming_model_path) as model:
        caps = model.capabilities
        if not caps.languages or "en" not in caps.languages:
            pytest.skip("model does not advertise an 'en' language hint")
        with model.session() as session, session.stream(language="en") as stream:
            for i in range(0, len(audio_pcm), 16000):
                stream.feed(audio_pcm[i : i + 16000])
            stream.finalize()
            committed = stream.text().committed
    assert "country" in committed.lower(), committed


def test_cancellation_aborts_pending_feed(streaming_model_path, audio_pcm):
    # Pre-set the cancel flag, then feed: the abort callback fires at the
    # first poll inside the feed. (True cross-thread mid-run cancellation is
    # covered by TestCancellation in test_transcribe.py.) Must surface as
    # the dedicated Aborted class, not a generic TranscribeError.
    with t.Model(streaming_model_path) as model, model.session() as session:
        with session.stream() as stream:
            session.cancel()
            with pytest.raises(t.Aborted):
                stream.feed(audio_pcm)
            assert session.was_aborted, "was_aborted should be True after cancel()"


# --- stream lifecycle edges ---------------------------------------------------


def test_feed_after_finalize_rejected(streaming_model_path, audio_pcm):
    with t.Model(streaming_model_path) as model, model.session() as session:
        with session.stream() as stream:
            stream.feed(audio_pcm[:16000])
            stream.finalize()
            assert stream.state == "finished"
            with pytest.raises(t.InvalidArgument):
                stream.feed(audio_pcm[:16000])


def test_stream_use_after_reset_rejected(streaming_model_path, audio_pcm):
    with t.Model(streaming_model_path) as model, model.session() as session:
        stream = session.stream()
        stream.feed(audio_pcm[:16000])
        stream.reset()
        with pytest.raises(t.TranscribeError, match="reset"):
            stream.feed(audio_pcm[:16000])
        with pytest.raises(t.TranscribeError, match="reset"):
            stream.text()


def test_stream_reset_idempotent_and_session_reusable(
        streaming_model_path, audio_pcm):
    with t.Model(streaming_model_path) as model, model.session() as session:
        first = session.stream()
        first.feed(audio_pcm[:16000])
        first.reset()
        first.reset()  # idempotent by contract

        # The session returns to idle: a SECOND stream on it must work
        # end-to-end.
        with session.stream() as second:
            for i in range(0, len(audio_pcm), 16000):
                second.feed(audio_pcm[i : i + 16000])
            second.finalize()
            committed = second.text().committed
    assert "country" in committed.lower(), committed


def test_second_stream_while_active_rejected(streaming_model_path, audio_pcm):
    # One stream at a time per session: the C dispatcher rejects a begin
    # while ACTIVE, and the binding surfaces it as InvalidArgument.
    with t.Model(streaming_model_path) as model, model.session() as session:
        with session.stream() as stream:
            stream.feed(audio_pcm[:16000])
            with pytest.raises(t.InvalidArgument):
                session.stream()
