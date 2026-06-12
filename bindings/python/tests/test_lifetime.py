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


# --- multi-session use of one model ------------------------------------------
#
# SUPPORTED in 0.x: many sessions on one model, run serially (from any
# threads). NOT yet supported: overlapping runs across sessions of one model
# — they share the model's compute backend and some family state, so
# concurrent runs race (observed: corrupted whisper decodes on CPU,
# command-buffer failures on Metal). The xfail below pins the limitation so
# the day a per-session backend architecture lands, the XPASS flags it for
# promotion to a hard test.


def test_serial_sessions_across_threads(model_path, audio_pcm):
    # Two threads, each with its own session, runs SERIALIZED by a lock:
    # the supported session-pool pattern.
    import threading

    lock = threading.Lock()
    results: dict = {}
    errors_seen: list = []

    def worker(tag: str, model: "t.Model") -> None:
        try:
            with model.session() as session:
                with lock:
                    results[tag] = session.run(audio_pcm).text.lower()
        except Exception as exc:  # surface, don't deadlock the join
            errors_seen.append((tag, exc))

    with t.Model(model_path) as model:
        threads = [threading.Thread(target=worker, args=(f"t{i}", model))
                   for i in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=120)
    assert not errors_seen, errors_seen
    assert len(results) == 2
    assert all("country" in text for text in results.values()), results


# Run INSIDE a subprocess: the known failure mode is NATIVE (corrupted
# decodes, Metal command-buffer failures, potentially a crash or a wedged
# thread). xfail can absorb a Python assert but not a segfault of the test
# runner, and a wedged native thread must never reach interpreter teardown
# in the CI process (freeing the session under an in-flight native call is
# a use-after-free). os._exit on the wedge path skips teardown on purpose.
_CONCURRENT_RUNS_SNIPPET = """\
import array, os, sys, threading, wave

import transcribe_cpp as t

model_path, audio_path = sys.argv[1], sys.argv[2]
with wave.open(audio_path, "rb") as w:
    pcm16 = array.array("h")
    pcm16.frombytes(w.readframes(w.getnframes()))
pcm = array.array("f", (s / 32768.0 for s in pcm16))

results, errors = {}, []

def worker(tag, model):
    try:
        with model.session() as session:
            results[tag] = session.run(pcm).text.lower()
    except Exception as exc:
        errors.append((tag, repr(exc)))

with t.Model(model_path) as model:
    threads = [threading.Thread(target=worker, args=(f"t{i}", model), daemon=True)
               for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    if any(th.is_alive() for th in threads):
        print("WEDGED: a run never returned", flush=True)
        os._exit(3)  # never free handles under an in-flight native call

assert not errors, errors
assert len(results) == 2 and all("country" in v for v in results.values()), results
print("CONCURRENT-OK")
"""


@pytest.mark.xfail(
    strict=False,
    reason="known 0.x limitation: overlapping runs on sessions of one model "
    "race on the shared backend/model state (see Model docstring and "
    "notes/bindings-review-fixes.md); per-session backends planned",
)
def test_concurrent_sessions_on_shared_model(model_path, audio_path):
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["TRANSCRIBE_LIBRARY"] = t.library_path()
    proc = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_RUNS_SNIPPET,
         str(model_path), str(audio_path)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert proc.returncode == 0 and "CONCURRENT-OK" in proc.stdout, (
        proc.returncode, proc.stdout[-300:], proc.stderr[-300:])


# --- GC ordering: dropping every reference at once must never crash ----------


def test_gc_drop_model_and_session_together(model_path):
    def scope():
        model = t.Model(model_path)
        session = model.session()
        return None  # both refs die on return, collection order is GC's pick

    scope()
    gc.collect()


def test_gc_drop_with_active_stream(streaming_model_path, audio_pcm):
    def scope():
        model = t.Model(streaming_model_path)
        session = model.session()
        stream = session.stream()
        stream.feed(audio_pcm[:16000])
        return None  # model + session + ACTIVE stream all dropped together

    scope()
    gc.collect()
