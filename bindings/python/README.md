# transcribe-cpp

Python bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
a C/C++ speech-to-text library built on ggml.

> **Status: in development.** Until wheels are published, use a locally built
> `libtranscribe` through repo auto-discovery or `TRANSCRIBE_LIBRARY`.

```python
import transcribe_cpp

with transcribe_cpp.Model("model.gguf") as model:
    with model.session() as session:
        result = session.run(pcm_float32_16k_mono)
        print(result.text)
```

`run()` takes mono 16 kHz float32 PCM (buffer-protocol object or sequence). It
does not decode containers or resample; convert audio before calling it.

```python
import numpy as np

pcm = np.asarray(audio, dtype=np.float32)   # 1-D, 16 kHz mono
# Downmix stereo first; 2-D input is rejected:
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

## Backends

`Model(backend=...)` picks the compute device (`"auto"` uses the best
available). `transcribe_cpp.backends()` lists registered backends and
`backend_available(kind)` checks one kind.

| Variable | Effect |
|---|---|
| `TRANSCRIBE_BACKEND` | overrides the `"auto"` default; explicit `backend=` still wins |
| `TRANSCRIBE_NATIVE_PROVIDER` | forces an installed native provider package, for example `cu12` |
| `TRANSCRIBE_LIBRARY` | loads exactly this shared library |

Planned wheels will bundle CPU plus platform accelerators;
`transcribe-cpp[cu12]` will add the CUDA 12 provider.

## Running from a working tree

The binding loads the native library at import and verifies its ABI layout and
version before use. Build a shared library, then run from the repo or point
`TRANSCRIBE_LIBRARY` at it:

```bash
cmake -B build-shared -DTRANSCRIBE_BUILD_SHARED=ON
cmake --build build-shared --target transcribe

cd bindings/python
PYTHONPATH=src uv run --no-project python examples/transcribe_wav.py \
    ../../models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf ../../samples/jfk.wav
```

No-model tests always run; model tests skip unless smoke assets are present.
Override paths with `TRANSCRIBE_SMOKE_MODEL`, `TRANSCRIBE_SMOKE_AUDIO`, and
`TRANSCRIBE_SMOKE_STREAMING_MODEL`.

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
