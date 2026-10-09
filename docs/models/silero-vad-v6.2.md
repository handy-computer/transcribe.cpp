# Silero VAD v6.2

Voice activity detector by the Silero Team: one speech probability per
32 ms frame of 16 kHz audio, segmented into speech regions. VAD role
(`include/transcribe/vad.h`, [docs/vad.md](../vad.md)). Family note:
[docs/porting/families/silero_vad.md](../porting/families/silero_vad.md).

- Upstream: [snakers4/silero-vad](https://github.com/snakers4/silero-vad),
  PyPI `silero-vad==6.2.3`, model `silero_vad.jit` (16 kHz sub-model), MIT
- Parameters: 309,633 (F32 GGUF, 1.2 MB)
- Input: 16 kHz mono float32; 512-sample frames, any length, offline or
  streaming

## Downloads

| File | Size |
|---|---:|
| `silero-vad-v6.2-F32.gguf` | 1,241,056 B |

F32 only. Converted locally with
`uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py`
(not yet published to Hugging Face).

### Direct whisper.cpp binary weights

The same family/runtime also loads the official binaries from
[ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad/tree/9ffd54a1e1ee413ddf265af9913beaf518d1639b),
pinned revision `9ffd54a1e1ee413ddf265af9913beaf518d1639b`:

| File | SHA256 |
|---|---|
| `ggml-silero-v5.1.2.bin` | `29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf` |
| `ggml-silero-v6.2.0.bin` | `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987` |

These are upstream publications, not canonical handy-computer GGUF releases.
They use mixed F16/F32 storage; expansion does not restore full-F32 precision.
See [VAD usage](../vad.md#supported-model-files) and the
[validation commands](../porting/families/silero_vad.md#commands).

CPU, Apple M4, nine golden clips, 1/4 threads, against upstream JIT loaded with
**the same stored weights**:

| Binary | Max probability error | Worst clip mean error |
|---|---:|---:|
| v5.1.2 | 3.040e-6 | 1.772e-7 |
| v6.2.0 | 1.071e-5 | 1.498e-7 |

Both have exact segments/live events and bit-identical offline/stream scores.
Storage rounding alone changes probabilities versus original F32 weights by up
to 0.003073 (v5) / 0.010982 (v6); that is separate from native inference error.

## Accuracy

The reference is upstream's own package. Measured as exact speech-segment
parity with `get_speech_timestamps(audio, load_silero_vad())` at default
parameters (`scripts/vad/parity.py`, C++ on CPU, 4 threads):

| Corpus | Files | Audio | Segments | Identical segment lists | Boundary differences |
|---|---:|---:|---:|---:|---:|
| AMI IHM test (meetings) | 16 | 9.06 h | 9,785 | 16 / 16 | 0 |
| LibriSpeech test-clean + FLEURS zh / ja / ar / de | 5,505 | 15.30 h | 12,555 | 5,505 / 5,505 | 0 |

Per-frame probabilities on the nine `validate.py` cases (speech, silence,
noise, Chinese, Japanese, two speakers, music, 197 s): max |C++ - reference|
6.9e-6, mean 4e-8 to 2e-7. An fp64 NumPy re-derivation of the model is
closer to the C++ (max 3.3e-6) than to the reference (max 7.6e-6).

## Speed

Apple M4, CPU, `samples/love-loss.wav` (197 s, 6162 frames), best of
repeated runs, model load excluded. transcribe.cpp built with the default
`GGML_NATIVE=OFF` (`transcribe-cli --repeat`); upstream measured with
`scripts/vad/bench_reference.py` (torch 2.8.0, onnxruntime 1.30.0, one
thread each, as `load_silero_vad` configures them).

| Engine | Threads | Time | x real time |
|---|---:|---:|---:|
| upstream TorchScript (`get_speech_timestamps` loop) | 1 | 405 ms | 486x |
| upstream ONNX (`OnnxWrapper`) | 1 | 490 ms | 402x |
| upstream ONNX sequence (`SileroVADSequence`, 512-frame blocks) | 1 | 267 ms | 738x |
| **transcribe.cpp** | 1 | 113 ms | 1746x |
| **transcribe.cpp** | 4 | 61 ms | 3243x |
| **transcribe.cpp** | 8 | 56 ms | 3552x |

On `samples/jfk.wav` (11 s): 6.2 ms single-threaded. On CPU, streaming
probabilities are bit-identical to an offline run with the same thread count
regardless of chunk size (tested with 1 and 4 threads).

Explicit Metal (`--backend metal`): 35 ms on love-loss, max 2.0e-3 from the
CPU probabilities (ggml-metal stages F32 matmul operands as half). The
default `AUTO` backend is the CPU to preserve reference accuracy.
