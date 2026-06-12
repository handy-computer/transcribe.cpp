"""Model-gated transcription tests: run, GIL release, batch, logging.

Each takes the ``model_path`` / ``audio_pcm`` fixtures, which ``skip`` when the
default whisper-tiny.en + jfk.wav assets are absent (override with
``TRANSCRIBE_SMOKE_MODEL`` / ``TRANSCRIBE_SMOKE_AUDIO``). The content assertions
("country") are specific to jfk.wav and hold for any English ASR model.
"""

from __future__ import annotations

import threading
import time

import pytest

import transcribe_cpp as t


def test_transcription_and_gil(model_path, audio_pcm):
    # GIL-release probe: a worker spins in pure Python (GIL-bound). If the native
    # run releases the GIL, the worker advances while transcription is in flight.
    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    time.sleep(0.01)
    try:
        with t.Model(model_path) as model:
            assert model.arch, "model reported an empty arch string"
            with model.session() as session:
                before = counter["n"]
                result = session.run(audio_pcm, timestamps="segment")
                during = counter["n"] - before
    finally:
        stop.set()
        worker.join(timeout=1.0)

    text = result.text.strip().lower()
    assert text, "empty transcription"
    assert "country" in text, f"unexpected transcription: {result.text!r}"
    assert result.segments, "no segments materialized"
    assert during > 100, f"GIL not released during run (worker advanced {during})"


def test_run_batch(model_path, audio_pcm):
    with t.Model(model_path) as model, model.session() as session:
        results = session.run_batch([audio_pcm, audio_pcm], timestamps="segment")
    assert len(results) == 2
    assert all("country" in r.text.lower() for r in results), results
    assert all(r.segments for r in results)


def test_logging_routes_through_public_sink(model_path, audio_pcm):
    received = []
    t.set_log_callback(lambda level, msg: received.append((level, msg)))
    try:
        with t.Model(model_path) as model, model.session() as session:
            session.run(audio_pcm)
            # print_timings publishes through the public log sink at INFO level.
            t._lib.transcribe_print_timings(session._h)
        assert received, "log callback received nothing from the public sink"
    finally:
        t.set_log_callback(None)


# --- TRANSCRIBE_BACKEND default override ------------------------------------


def test_backend_env_invalid_value_rejected(monkeypatch):
    # No model needed: the env value is validated before any file access, and
    # the error must name the env var so a typo is diagnosable.
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "warp9")
    with pytest.raises(t.InvalidArgument, match="TRANSCRIBE_BACKEND"):
        t.Model("nonexistent.gguf")


def test_backend_env_overrides_auto(model_path, audio_pcm, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "cpu")
    with t.Model(model_path) as model:
        assert "cpu" in model.backend.lower(), model.backend
        with model.session() as session:
            text = session.run(audio_pcm).text
    assert "country" in text.lower(), text


def test_backend_explicit_arg_beats_env(model_path, monkeypatch):
    # A deliberately unsatisfiable env value: if the explicit argument didn't
    # win, the load would fail (or land on a non-CPU device).
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "warp9")
    with t.Model(model_path, backend="cpu") as model:
        assert "cpu" in model.backend.lower(), model.backend


# --- transcribe() one-shot helper --------------------------------------------


class TestOneShot:
    def test_path_form(self, model_path, audio_pcm):
        result = t.transcribe(model_path, audio_pcm)
        assert "country" in result.text.lower(), result.text

    def test_model_form_leaves_model_open(self, model_path, audio_pcm):
        with t.Model(model_path) as model:
            result = t.transcribe(model, audio_pcm)
            assert "country" in result.text.lower()
            # The helper must NOT close a caller-owned model: it stays
            # usable for another run afterwards.
            assert model.arch
            again = t.transcribe(model, audio_pcm)
            assert "country" in again.text.lower()

    def test_model_form_ignores_backend_arg(self, model_path, audio_pcm):
        # backend= applies only to the path form; with a Model it is ignored
        # (documented), so an unsatisfiable value must not break the call.
        with t.Model(model_path, backend="cpu") as model:
            result = t.transcribe(model, audio_pcm, backend="cuda")
        assert "country" in result.text.lower()


# --- run_batch edges ----------------------------------------------------------


class TestRunBatch:
    def test_empty_batch_rejected(self, model_path):
        with t.Model(model_path) as model, model.session() as session:
            with pytest.raises(t.InvalidArgument, match="at least one"):
                session.run_batch([])

    def test_mixed_lengths(self, model_path, audio_pcm):
        with t.Model(model_path) as model, model.session() as session:
            results = session.run_batch([audio_pcm, audio_pcm[:16000]])
        assert len(results) == 2
        assert "country" in results[0].text.lower()
        assert isinstance(results[1], t.Result)

    def test_return_exceptions_all_success(self, model_path, audio_pcm):
        with t.Model(model_path) as model, model.session() as session:
            out = session.run_batch([audio_pcm], return_exceptions=True)
        assert len(out) == 1 and isinstance(out[0], t.Result)

    def test_mixed_pcm_forms(self, model_path, audio_pcm):
        import array as _array

        forms = [audio_pcm, audio_pcm.tobytes(), memoryview(audio_pcm)]
        with t.Model(model_path) as model, model.session() as session:
            results = session.run_batch(forms)
        texts = [r.text.lower() for r in results]
        assert all("country" in text for text in texts), texts


# --- session limits -----------------------------------------------------------


def test_session_limits_readable(model_path):
    with t.Model(model_path) as model:
        caps = model.capabilities
        with model.session() as session:
            limits = session.limits
    assert isinstance(limits, t.SessionLimits)
    assert limits.effective_n_ctx >= 0
    assert limits.effective_max_audio_ms >= 0
    assert limits.max_kv_bytes >= 0
    # Coherence with the model-level bound: a session at default n_ctx can
    # never enforce a TIGHTER audio limit than the model advertises.
    if caps.max_audio_ms > 0 and limits.effective_max_audio_ms > 0:
        assert limits.effective_max_audio_ms <= caps.max_audio_ms


# --- spec_k_drafts ------------------------------------------------------------


def test_spec_k_drafts_disabled_matches_default(model_path, audio_pcm):
    # Spec decoding is a lossless performance strategy: disabling it must
    # not change the transcript. (On a model without supports_spec_decode
    # the field is ignored entirely, which trivially also passes.)
    with t.Model(model_path) as model, model.session() as session:
        base = session.run(audio_pcm, spec_k_drafts=-1).text
        nospec = session.run(audio_pcm, spec_k_drafts=0).text
    assert base == nospec


# --- cancellation depth ---------------------------------------------------------


class TestCancellation:
    def test_was_aborted_false_after_clean_run(self, model_path, audio_pcm):
        with t.Model(model_path) as model, model.session() as session:
            session.run(audio_pcm)
            assert session.was_aborted is False

    def test_cross_thread_cancel_in_flight_run(self, model_path, audio_pcm):
        # Long input (tiled jfk) so the run is mid-flight when the timer
        # fires from another thread. The abort must surface as Aborted with
        # the partial transcript attached, and the session must stay usable.
        long_pcm = audio_pcm * 12  # ~2.2 minutes of audio
        with t.Model(model_path) as model, model.session() as session:
            timer = threading.Timer(0.5, session.cancel)
            timer.start()
            try:
                result = session.run(long_pcm)
            except t.Aborted as exc:
                assert session.was_aborted is True
                assert exc.partial_result is not None
                assert isinstance(exc.partial_result, t.Result)
            else:
                # A machine fast enough to finish 2 minutes of audio in
                # 0.5 s makes the race unwinnable; don't fail on speed.
                pytest.skip(f"run finished before cancel fired: {result.text[:40]!r}")
            finally:
                timer.cancel()

            # The cancel flag is cleared at the start of the next run.
            ok = session.run(audio_pcm)
            assert session.was_aborted is False
            assert "country" in ok.text.lower()

    def test_cancel_mid_batch_keeps_per_utterance_view(self, model_path, audio_pcm):
        # Cancellation surfaces at the BATCH level in C; the binding must
        # fold it back into the per-utterance contract (utterance_index +
        # batch_results) instead of discarding completed work.
        long_pcm = audio_pcm * 12
        with t.Model(model_path) as model, model.session() as session:
            timer = threading.Timer(0.5, session.cancel)
            timer.start()
            try:
                session.run_batch([long_pcm, long_pcm])
            except t.Aborted as exc:
                assert session.was_aborted is True
                assert exc.utterance_index is not None
                assert hasattr(exc, "batch_results"), "completed work discarded"
                assert len(exc.batch_results) == 2
                assert all(isinstance(r, (t.Result, t.TranscribeError))
                           for r in exc.batch_results)
            else:
                pytest.skip("batch finished before cancel fired")
            finally:
                timer.cancel()

            # The session stays usable after a cancelled batch.
            ok = session.run(audio_pcm)
            assert session.was_aborted is False
            assert "country" in ok.text.lower()
