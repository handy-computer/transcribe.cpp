# transcribe-cpp

Python bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
a C/C++ speech-to-text library built on ggml.

> **Status: in development.** Until wheels are published, use a locally built
> `libtranscribe` through repo auto-discovery or `TRANSCRIBE_LIBRARY`.

Upgrading from 0.1? See the
[0.2 migration guide](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/migrating-to-0.2.md),
including the replacement of `gpu_device=` with exact device objects.

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

### Punctuation, capitalization, and text normalization

Generic run controls use `"default"` to preserve each model family's shipped
behavior. Models advertising `model.supports("pnc")` accept `pnc="off"` or
`pnc="on"`; models advertising `model.supports("itn")` accept the equivalent
`itn` values. The options are available on `run()`, `run_batch()`, `stream()`,
and the one-shot `transcribe()` helper.

```python
result = session.run(pcm, pnc="off", itn="on")
```

### Prompting

`vocabulary` (custom terms), `prompt` (context, or the instruction under
`task="instruct"`) and `prefix` (text the model continues from) take effect
where `model.supports()` reports `"vocabulary"`, `"context_prompt"`,
`"instruct"` or `"transcript_prefix"`.

```python
result = session.run(pcm, vocabulary=["Kubernetes", "gRPC"])
```

Streaming models expose incremental transcription with committed/tentative
text views — see `examples/stream_wav.py`:

```python
with model.session() as session, session.stream() as stream:
    for chunk in pcm_chunks:
        stream.feed(chunk)
        text = stream.text()        # .committed (stable) + .tentative
    stream.finalize()
    result = stream.snapshot()      # language, segments, words, tokens, timings
```

Long transcriptions can be cancelled from another thread with
`session.cancel()` — the run raises `Aborted` with the partial transcript on
`exc.partial_result` (same for `OutputTruncated`).

### Diarization

Models whose `model.roles` include `Role.DIARIZE` (Sortformer) answer "who
spoke when" through a diarize session. `run()` returns `SpeakerSegment`
rows (`speaker_id` in `1..model.diarize_info.max_speakers`), grouped by
speaker. Locking, `Busy`, `cancel()` and `close()` work as on `Session`. A
model without the role raises `UnsupportedRole`, as `model.session()` and
`model.capabilities` do on a model without `Role.ASR` (Sortformer serves only
`Role.DIARIZE`).

```python
with model.diarize_session() as diarizer:
    turns = diarizer.run(pcm, family=transcribe_cpp.SortformerDiarizeOptions(
        preset="very_high_latency"))
    for turn in turns:
        print(turn.speaker_id, turn.t0_ms, turn.t1_ms)
```

### Language ID

Models whose `model.roles` include `Role.LANGID` (VoxLingua107 ECAPA-TDNN)
identify the spoken language through a langid session. `run()` returns a
`LangIdResult` with candidates ranked by `p`. Codes are the model's own labels
(`"iw"`, `"jw"`; `model.langid_label_index("he")` resolves aliases), so match
`result.code` against the ASR model's `capabilities.languages` yourself.
`allowed=None` scores every label; `allowed_mass` is the unrestricted
probability the allowed set captured. An empty `allowed` list raises
`InvalidArgument`; clips under `model.langid_info.min_audio_ms` raise
`InputTooShort`, and longer ones than `max_audio_ms` are scored on their first
`max_audio_ms`. Every allowed label comes back as a candidate. Locking,
`Busy`, `cancel()` and `close()` work as on `Session`.

```python
with model.langid_session() as lid:
    result = lid.run(pcm, allowed=["en", "de", "fr"])
    print(result.code, result.candidates[0].p, result.allowed_mass)
```

## Backends

`Model(backend=...)` applies a backend policy (`"auto"` uses the best
available). `transcribe_cpp.backends()` returns process-local device objects;
pass one as `Model(device=device)` for exact selection with no fallback. Persist
a device's `device_id`, not its runtime handle or index. `backend_available(kind)`
checks whether a backend policy can currently be satisfied.

```python
device = next(d for d in transcribe_cpp.backends() if d.device_type == "cpu")
with transcribe_cpp.Model("model.gguf", device=device) as model:
    print(model.device)
```

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

- One compute call at a time per `Model`: the binding serializes calls across
  all sessions of a model with a model-wide lock (load one model per worker
  for parallelism). While a stream is active, other runs and stream begins on
  that model raise `transcribe_cpp.Busy`. See the `Model` docstring.
- Import package: `transcribe_cpp`
- Distribution: `transcribe-cpp`
- License: MIT
