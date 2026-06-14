"""Diagnostic: does CPU whisper transcription deadlock through the Python
(ctypes) binding on Windows/MSVC in the wheel posture (GGML_OPENMP=OFF)?

If a run wedges in ggml-cpu's non-OpenMP threadpool barrier, the main thread
blocks inside the ctypes FFI call to transcribe_run. ctypes releases the GIL for
the call, so faulthandler's separate watchdog thread still fires: at the deadline
it dumps every thread's Python stack (main shown parked in `session.run(...)`)
and _exit(1)s. A clean run cancels the watchdog and exits 0.

Env: WHISPER_MODEL (gguf), JFK_WAV (wav), TRANSCRIBE_LIBRARY (the built dll).
Delete with .github/workflows/diag-py-cpu.yml once the wheel posture is decided.
"""

import argparse
import array
import faulthandler
import os
import wave

ap = argparse.ArgumentParser()
ap.add_argument("--timeout", type=int, default=90)
args = ap.parse_args()

faulthandler.dump_traceback_later(args.timeout, exit=True)

import transcribe_cpp as t  # noqa: E402  (after the watchdog is armed)

print("transcribe_cpp loaded; TRANSCRIBE_LIBRARY=", os.environ.get("TRANSCRIBE_LIBRARY"), flush=True)
print("devices:", [(d.kind, d.name) for d in t.backends()], flush=True)

with wave.open(os.environ["JFK_WAV"], "rb") as w:
    pcm16 = array.array("h", w.readframes(w.getnframes()))
pcm = array.array("f", (x / 32768.0 for x in pcm16))

model_path = os.environ["WHISPER_MODEL"]
print("loading model (backend=cpu)...", flush=True)
with t.Model(model_path, backend="cpu") as m:
    print("model backend:", m.backend, flush=True)
    with m.session() as s:
        print("running CPU transcription (watchdog %ds)..." % args.timeout, flush=True)
        text = s.run(pcm).text

faulthandler.cancel_dump_traceback_later()
print("COMPLETED:", text.strip()[:80], flush=True)
assert "country" in text.lower(), text
print("RESULT: PASS (CPU transcription completed, no deadlock)", flush=True)
