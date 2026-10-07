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
            snapshot = stream.snapshot()
            revision, state = stream.revision, stream.state
            last_status = stream.last_status
    assert update.is_final, update
    assert state == "finished", state
    assert revision >= 1
    assert last_status is None, last_status
    assert "country" in committed.lower(), committed
    assert snapshot.text
    assert isinstance(snapshot.language, str)
    assert snapshot.segments
    assert isinstance(snapshot.words, tuple)
    assert isinstance(snapshot.tokens, tuple)


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
    # the dedicated Aborted class, not a generic TranscribeError. The stream
    # is then FAILED, so it releases the model's stream lease.
    with t.Model(streaming_model_path) as model, model.session() as session:
        with model.session() as sibling, session.stream() as stream:
            session.cancel()
            with pytest.raises(t.Aborted):
                stream.feed(audio_pcm)
            assert session.was_aborted, "was_aborted should be True after cancel()"
            assert stream.state == "failed"
            sibling.run(audio_pcm[:16000])  # not Busy


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
        with pytest.raises(t.TranscribeError, match="reset"):
            stream.snapshot()


def test_superseded_stream_cannot_touch_the_new_one(streaming_model_path, audio_pcm):
    # A finalized Stream stays readable until the session begins another; then
    # it is superseded: feed raises and reset is a no-op on the new stream.
    with t.Model(streaming_model_path) as model, model.session() as session:
        old = session.stream()
        old.feed(audio_pcm[:16000])
        old.finalize()
        assert old.state == "finished"

        new = session.stream()
        new.feed(audio_pcm[:16000])
        revision = new.revision
        with pytest.raises(t.TranscribeError, match="superseded"):
            old.feed(audio_pcm[16000:32000])
        with pytest.raises(t.TranscribeError, match="superseded"):
            old.text()
        old.reset()
        assert new.state == "active"
        assert new.revision == revision
        with model.session() as sibling, pytest.raises(t.Busy):
            sibling.run(audio_pcm[:16000])  # the new stream still holds the lease
        new.finalize()
        new.reset()


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
    # One active stream per MODEL: the stream lease refuses a second begin
    # with Busy, on the same session and on a sibling, before native code.
    with t.Model(streaming_model_path) as model, model.session() as session:
        with model.session() as sibling, session.stream() as stream:
            stream.feed(audio_pcm[:16000])
            with pytest.raises(t.Busy, match="already active"):
                session.stream()
            with pytest.raises(t.Busy, match="already active"):
                sibling.stream()
            assert stream.state == "active"


def test_rejected_feed_keeps_stream_active(streaming_model_path, audio_pcm):
    # Non-finite PCM is rejected BEFORE the family hook: the feed raises
    # InvalidArgument but the native stream stays ACTIVE (only failures
    # inside the hook move it to FAILED). The binding releases the compute
    # lock on that error but KEEPS the stream lease: the stream stays
    # usable and sibling calls stay Busy until it is finalized.
    with t.Model(streaming_model_path) as model, model.session() as session:
        with model.session() as sibling, session.stream() as stream:
            stream.feed(audio_pcm[:16000])
            bad = list(audio_pcm[16000:32000])
            bad[100] = float("nan")
            with pytest.raises(t.InvalidArgument):
                stream.feed(bad)
            assert stream.state == "active"
            assert not model._compute_lock.locked()
            with pytest.raises(t.Busy):
                sibling.run(audio_pcm[:16000])
            for i in range(16000, len(audio_pcm), 16000):
                stream.feed(audio_pcm[i : i + 16000])
            update = stream.finalize()
            committed = stream.text().committed
            ran = sibling.run(audio_pcm).text  # finalize released the lease
    assert update.is_final
    assert "country" in committed.lower(), committed
    assert "country" in ran.lower(), ran


def test_stream_begin_clears_pending_cancel(streaming_model_path, audio_pcm):
    # A cancel() requested before stream() is cleared by the begin: the new
    # stream's feeds are not aborted by a stale flag. (A cancel() AFTER begin
    # is honoured on the next feed — test_cancellation_aborts_pending_feed.)
    with t.Model(streaming_model_path) as model, model.session() as session:
        session.cancel()
        with session.stream() as stream:
            stream.feed(audio_pcm[:16000])
            assert not session.was_aborted
            assert stream.state == "active"
