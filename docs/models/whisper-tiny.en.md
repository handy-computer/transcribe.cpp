# Whisper tiny.en

OpenAI's [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en) ported to transcribe.cpp. A 39M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-tiny.en) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`87c7102`](https://huggingface.co/openai/whisper-tiny.en/commit/87c7102),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-tiny.en-F32.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-F32.gguf) | 146 MB | 5.62% |
| F16    | [whisper-tiny.en-F16.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-F16.gguf) | 76 MB | 5.61% |
| Q8_0   | [whisper-tiny.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q8_0.gguf) | 44 MB | 5.59% |
| Q6_K   | [whisper-tiny.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q6_K.gguf) | 43 MB | 5.71% |
| Q5_K_M | [whisper-tiny.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q5_K_M.gguf) | 42 MB | 5.68% |
| Q4_K_M | [whisper-tiny.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q4_K_M.gguf) | 42 MB | 5.79% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the pinned short-form recipe: greedy decode, timestamps off (`<|notimestamps|>`), and language forced to `en` — see [WER methodology](../tools/wer.md#methodology-pinned-recipe). Captured on a single CUDA (L40S) run at batch size 1; quantization, backend, and batching are all generally WER-neutral.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-tiny.en/whisper-tiny.en-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places). Decode latency dominates as model size grows; the encoder
is only run once per 30-second window.

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  39.1 ms (281.2×) |  34.0 ms (323.8×) |
| Metal   | dots (35.3s) | 127.0 ms (278.3×) | 125.8 ms (280.9×) |
| CPU     | jfk (11.0s)  | 165.0 ms (66.7×)  | 161.4 ms (68.1×)  |
| CPU     | dots (35.3s) | 389.4 ms (90.7×)  | 381.8 ms (92.5×)  |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-tiny.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-tiny.en-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  197 ms (56.0×) |  193 ms (56.9×) |
| Vulkan  | dots (35.3s) |  540 ms (65.4×) |  541 ms (65.3×) |
| CPU     | jfk (11.0s)  |  493 ms (22.3×) |  436 ms (25.2×) |
| CPU     | dots (35.3s) |  1.19 s (29.8×) |  1.09 s (32.5×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-tiny.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-tiny.en-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 21 checkpointed tensors fall within per-variant tolerance. Tolerance budget lives at
[`tests/tolerances/whisper-tiny.en.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-tiny.en.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-tiny.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-tiny.en.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-tiny.en` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `7.272e-06` |   `7.293e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.240e-05` |   `2.603e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `2.623e-05` |   `8.075e-07` | first encoder block |
| `enc.block.3.out`      |  `1.709e-02` |   `3.026e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `4.005e-05` |   `2.161e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `2.861e-05` |   `3.417e-07` | first decoder block, prompt pass |
| `dec.block.3.out`      |  `4.196e-05` |   `1.334e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.450e-04` |   `2.361e-05` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `6.807e-05` |   `3.062e-05` | vocab projection (raw logits) |
| `dec.logits`           |  `9.584e-05` |   `3.682e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `7.248e-05` |   `4.521e-05` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-tiny.en \
  --revision 87c7102
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-tiny.en/whisper-tiny.en-F32.gguf \
  models/whisper-tiny.en/whisper-tiny.en-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-tiny.en
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-tiny.en/whisper-tiny.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
