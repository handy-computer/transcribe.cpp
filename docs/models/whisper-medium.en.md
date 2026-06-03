# Whisper medium.en

OpenAI's [`openai/whisper-medium.en`](https://huggingface.co/openai/whisper-medium.en) ported to transcribe.cpp. A 769M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-medium.en) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`2e98eb6`](https://huggingface.co/openai/whisper-medium.en/commit/2e98eb6),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-medium.en-F32.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-F32.gguf) | 2.85 GB | 2.98% |
| F16    | [whisper-medium.en-F16.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-F16.gguf) | 1.44 GB | 2.98% |
| Q8_0   | [whisper-medium.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q8_0.gguf) | 793 MB | 2.97% |
| Q6_K   | [whisper-medium.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q6_K.gguf) | 618 MB | 2.85% |
| Q5_K_M | [whisper-medium.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q5_K_M.gguf) | 556 MB | 2.97% |
| Q4_K_M | [whisper-medium.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q4_K_M.gguf) | 481 MB | 2.96% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the pinned short-form recipe: greedy decode, timestamps off (`<|notimestamps|>`), and language forced to `en` — see [WER methodology](../tools/wer.md#methodology-pinned-recipe). Captured on a single CUDA (L40S) run at batch size 1; quantization, backend, and batching are all generally WER-neutral.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-medium.en/whisper-medium.en-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mel + encode + decode, mean over the recorded
iterations after warmup), with speedup over realtime in parentheses. Units:
`ms` below 1 s, `s` above (2 decimal places). Decode latency dominates as
model size grows; the encoder is only run once per 30-second window.

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  | 249.7 ms (44.0×) | 243.3 ms (45.2×) |
| Metal   | dots (35.3s) | 762.9 ms (46.3×) | 725.9 ms (48.7×) |
| CPU     | jfk (11.0s)  |    4.29 s (2.6×) |    3.37 s (3.3×) |
| CPU     | dots (35.3s) |    9.07 s (3.9×) |    7.23 s (4.9×) |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-medium.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-medium.en-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.74 s (4.0×)  |  2.55 s (4.3×)  |
| Vulkan  | dots (35.3s) |  6.76 s (5.2×)  |  6.44 s (5.5×)  |
| CPU     | jfk (11.0s)  |  11.53 s (1.0×) |  9.36 s (1.2×)  |
| CPU     | dots (35.3s) |  26.63 s (1.3×) |  21.07 s (1.7×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-medium.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-medium.en-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance. Tolerance budget lives at
[`tests/tolerances/whisper-medium.en.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-medium.en.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-medium.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-medium.en.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-medium.en` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `4.679e-06` |   `3.732e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.380e-05` |   `3.135e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `1.717e-05` |   `8.644e-07` | first encoder block |
| `enc.block.23.out`     |  `4.858e-02` |   `5.993e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `2.327e-03` |   `4.378e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `5.245e-06` |   `2.235e-07` | first decoder block, prompt pass |
| `dec.block.23.out`     |  `1.793e-04` |   `5.787e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `2.990e-04` |   `1.923e-05` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `4.959e-05` |   `6.956e-06` | vocab projection (raw logits) |
| `dec.logits`           |  `8.297e-05` |   `2.911e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `3.433e-05` |   `8.558e-06` | step-20 logits (KV-cached path) |

The C++ mel frontend (Slaney filterbank + Hann periodic window +
whisper-style log-mel compression) drives `enc.mel.in` to fp32-vs-fp64
STFT precision drift; downstream tensors stay within budget. KV-cached
decoder runs through F16 self/cross caches by default — flip with
`--kv-type f32` for tighter parity.

## Reproduction

### Convert

The whisper converter loads from a Hugging Face checkpoint and emits a
reference-dtype GGUF.

```bash
uv run --project scripts/envs/whisper \
  scripts/convert-whisper.py openai/whisper-medium.en \
  --revision 2e98eb6
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-medium.en/whisper-medium.en-F32.gguf \
  models/whisper-medium.en/whisper-medium.en-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-medium.en
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-medium.en/whisper-medium.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
