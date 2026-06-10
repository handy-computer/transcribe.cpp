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
`ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`.

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
status-code/enum agreement) always run; model tests skip when the default
whisper-tiny.en + jfk.wav assets are absent (override with
`TRANSCRIBE_SMOKE_MODEL` / `TRANSCRIBE_SMOKE_AUDIO`):

```bash
cd bindings/python
TRANSCRIBE_LIBRARY=../../build-shared/src/libtranscribe.dylib \
    uv run --extra test pytest
```

- Import package: `transcribe_cpp`
- Distribution: `transcribe-cpp`
- License: MIT
