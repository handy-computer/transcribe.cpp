"""Model-gated transcription tests: run, GIL release, batch, logging.

Each takes the ``model_path`` / ``audio_pcm`` fixtures, which ``skip`` when the
default whisper-tiny.en + jfk.wav assets are absent (override with
``TRANSCRIBE_SMOKE_MODEL`` / ``TRANSCRIBE_SMOKE_AUDIO``). The content assertions
("country") are specific to jfk.wav and hold for any English ASR model.
"""

from __future__ import annotations

import threading
import time

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
