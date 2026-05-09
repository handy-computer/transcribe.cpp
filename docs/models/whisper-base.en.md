# Whisper base.en

OpenAI's [`openai/whisper-base.en`](https://huggingface.co/openai/whisper-base.en) ported to transcribe.cpp. A 74M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-base.en) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`911407f`](https://huggingface.co/openai/whisper-base.en/commit/911407f),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-base.en-F32.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-F32.gguf) | 279 MB | 4.14% |
| F16    | [whisper-base.en-F16.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-F16.gguf) | 144 MB | 4.13% |
| Q8_0   | [whisper-base.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q8_0.gguf) | 81 MB | 4.16% |
| Q6_K   | [whisper-base.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q6_K.gguf) | 65 MB | 4.15% |
| Q5_K_M | [whisper-base.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q5_K_M.gguf) | 61 MB | 4.16% |
| Q4_K_M | [whisper-base.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q4_K_M.gguf) | 56 MB | 4.29% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 4.25%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-base.en/whisper-base.en-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  50.3 ms (218.9×) |  49.5 ms (222.3×) |
| Metal   | dots (35.3s) | 172.4 ms (204.9×) | 161.4 ms (218.9×) |
| CPU     | jfk (11.0s)  | 352.5 ms (31.2×)  | 325.6 ms (33.8×)  |
| CPU     | dots (35.3s) | 780.9 ms (45.2×)  | 733.6 ms (48.2×)  |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-base.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-base.en-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  327 ms (33.6×) |  337 ms (32.7×) |
| Vulkan  | dots (35.3s) |  909 ms (38.9×) |  912 ms (38.7×) |
| CPU     | jfk (11.0s)  |  1.01 s (10.8×) |  836 ms (13.2×) |
| CPU     | dots (35.3s) |  2.41 s (14.7×) |  2.19 s (16.1×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-base.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-base.en-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance. Tolerance budget lives at
[`tests/tolerances/whisper-base.en.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-base.en.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-base.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-base.en.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-base.en` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `5.811e-06` |   `6.342e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.681e-05` |   `2.860e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `2.146e-05` |   `6.671e-07` | first encoder block |
| `enc.block.5.out`      |  `4.608e-02` |   `2.940e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `6.485e-05` |   `1.328e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `6.855e-06` |   `2.809e-07` | first decoder block, prompt pass |
| `dec.block.5.out`      |  `2.861e-05` |   `1.841e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.984e-04` |   `1.569e-05` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `6.676e-05` |   `9.382e-06` | vocab projection (raw logits) |
| `dec.logits`           |  `6.294e-05` |   `2.236e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `7.248e-05` |   `4.264e-05` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-base.en \
  --revision 911407f
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-base.en/whisper-base.en-F32.gguf \
  models/whisper-base.en/whisper-base.en-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-base.en
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-base.en/whisper-base.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
