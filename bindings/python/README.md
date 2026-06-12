# transcribe-cpp

Python bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
a C/C++ speech-to-text library built on ggml.

> **Status: in development.** The API below works against a locally built native
> library. Prebuilt wheels (bundled native code, GPU provider packages) are not
> published yet — for now you point the binding at a `libtranscribe` shared
> library you built. Watch the repository for the first wheel release.

```python
import transcribe_cpp

with transcribe_cpp.Model("model.gguf") as model:
    with model.session() as session:
        result = session.run(pcm_float32_16k_mono)
        print(result.text)
```

`run()` takes 16 kHz mono float32 PCM (buffer-protocol object or sequence). It
does not decode containers or resample — convert first, e.g.
`ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`. With numpy:

```python
import numpy as np

pcm = np.asarray(audio, dtype=np.float32)   # 1-D, 16 kHz mono in [-1, 1)
# stereo (frames, channels)? downmix first — 2-D input is rejected:
# pcm = audio.mean(axis=1).astype(np.float32)
result = session.run(pcm)
```

Streaming models expose incremental transcription with committed/tentative
text views — see `examples/stream_wav.py`:

```python
with model.session() as session, session.stream() as stream:
    for chunk in pcm_chunks:
        stream.feed(chunk)
        text = stream.text()        # .committed (stable) + .tentative
    stream.finalize()
```

Long transcriptions can be cancelled from another thread with
`session.cancel()` — the run raises `Aborted` with the partial transcript on
`exc.partial_result` (same for `OutputTruncated`).

## Backends and escape hatches

`Model(backend=...)` picks the compute device (`"auto"` → best available);
`transcribe_cpp.backends()` lists what registered and
`backend_available(kind)` probes one kind. Environment overrides:

| Variable | Effect |
|---|---|
| `TRANSCRIBE_BACKEND` | overrides the `backend="auto"` *default* (an explicit `backend=` argument always wins) — the escape hatch for machines whose best-ranked device misbehaves |
| `TRANSCRIBE_NATIVE_PROVIDER` | force a specific installed native provider package (e.g. `cu12`) |
| `TRANSCRIBE_LIBRARY` | load exactly this shared library (developer override) |

Once wheels are published: `pip install transcribe-cpp` ships CPU plus the
platform accelerator (Metal on macOS arm64, Vulkan on Linux/Windows) with
graceful CPU fallback; `pip install "transcribe-cpp[cu12]"` adds the CUDA 12
provider (which also bundles Vulkan, so non-NVIDIA machines keep GPU
acceleration).

## Running from a working tree

The binding loads the native library at import and verifies its ABI layout and
version before use. Build a shared library and the binding finds it
automatically from the repo, or point `TRANSCRIBE_LIBRARY` at one:

```bash
cmake -B build-shared -DTRANSCRIBE_BUILD_SHARED=ON
cmake --build build-shared --target transcribe

cd bindings/python
PYTHONPATH=src uv run --no-project python examples/transcribe_wav.py \
    ../../models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf ../../samples/jfk.wav
```

Run the test suite. No-model tests (import, ABI layout, version gate,
status-code/enum agreement, PCM validation, provider selection) always run;
model tests skip when the default whisper-tiny.en + jfk.wav assets are absent
(override with `TRANSCRIBE_SMOKE_MODEL` / `TRANSCRIBE_SMOKE_AUDIO` /
`TRANSCRIBE_SMOKE_STREAMING_MODEL`):

```bash
cd bindings/python
TRANSCRIBE_LIBRARY=../../build-shared/src/libtranscribe.dylib \
    uv run --extra test pytest
```

## Notes

- One run/stream at a time per `Model` in 0.x: sessions share the model's
  compute backend, so serialize runs across sessions (or load one model per
  worker). See the `Model` docstring.
- Import package: `transcribe_cpp`
- Distribution: `transcribe-cpp`
- License: MIT
