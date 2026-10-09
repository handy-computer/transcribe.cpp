# Silero VAD v6.2

Voice activity detector by the Silero Team: one speech probability per
32 ms frame of 16 kHz audio. Not a transcription model; it serves the VAD
role ([docs/vad.md](../vad.md)). Internals:
[docs/porting/families/silero_vad.md](../porting/families/silero_vad.md).

- Upstream: [snakers4/silero-vad](https://github.com/snakers4/silero-vad),
  PyPI `silero-vad==6.2.3`, 16 kHz model, MIT
- Parameters: 309,633

## Downloads

| File | Size |
|---|---:|
| `silero-vad-v6.2-F32.gguf` | 1,241,056 B |

F32 only; quantizing a 1.2 MB model buys nothing. Not yet published; convert
with `uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py`.

whisper.cpp's published binaries also load directly, from
[ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad/tree/9ffd54a1e1ee413ddf265af9913beaf518d1639b):

| File | SHA256 |
|---|---|
| `ggml-silero-v5.1.2.bin` | `29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf` |
| `ggml-silero-v6.2.0.bin` | `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987` |

They store some weights as F16, so their probabilities differ from the F32
original by up to 0.003 (v5) / 0.011 (v6). Against upstream's JIT loaded with
the same stored weights, native inference is within 1.1e-5 with identical
segments.

## Accuracy

Exact speech-segment parity with upstream `get_speech_timestamps` at default
parameters (`scripts/vad/parity.py`, CPU):

| Corpus | Files | Audio | Segments | Identical segment lists |
|---|---:|---:|---:|---:|
| AMI IHM test (meetings) | 16 | 9.06 h | 9,785 | 16 / 16 |
| LibriSpeech test-clean + FLEURS zh / ja / ar / de | 5,505 | 15.30 h | 12,555 | 5,505 / 5,505 |

Per-frame probabilities on the nine `validate.py` clips: max |C++ - reference|
6.9e-6.

## Speed

Apple M4, CPU, `samples/love-loss.wav` (197 s), model load excluded:

| Engine | Threads | Time | x real time |
|---|---:|---:|---:|
| upstream TorchScript (`get_speech_timestamps`) | 1 | 405 ms | 486x |
| upstream ONNX (`OnnxWrapper`) | 1 | 490 ms | 402x |
| **transcribe.cpp** | 1 | 113 ms | 1746x |
| **transcribe.cpp** | 4 | 61 ms | 3243x |

Explicit Metal (`--backend metal`) takes 35 ms but differs from the CPU by up
to 2e-3 in probability (ggml-metal stages F32 matmul operands as half), so
`AUTO` picks the CPU.
