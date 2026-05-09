# Whisper large-v2

OpenAI's [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2) ported to transcribe.cpp. A 1.55B-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v2) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`ae46427`](https://huggingface.co/openai/whisper-large-v2/commit/ae46427),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-large-v2-F32.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-F32.gguf) | 5.75 GB | 2.68% |
| F16    | [whisper-large-v2-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-F16.gguf) | 2.89 GB | 2.68% |
| Q8_0   | [whisper-large-v2-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q8_0.gguf) | 1.55 GB | 2.65% |
| Q6_K   | [whisper-large-v2-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q6_K.gguf) | 1.21 GB | 2.83% |
| Q5_K_M | [whisper-large-v2-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q5_K_M.gguf) | 1.08 GB | 2.72% |
| Q4_K_M | [whisper-large-v2-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q4_K_M.gguf) | 950 MB | 2.46% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.83%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v2/whisper-large-v2-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 493.2 ms (22.3×) | 499.6 ms (22.0×) |
| Metal   | dots (35.3s) |   1.46 s (24.1×) |   1.40 s (25.2×) |
| CPU     | jfk (11.0s)  |    9.66 s (1.1×) |    7.46 s (1.5×) |
| CPU     | dots (35.3s) |   19.72 s (1.8×) |   15.43 s (2.3×) |

macOS 26.4.1, transcribe.cpp `4d2270e`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v2 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-large-v2-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.27 s (1.8×)  |  6.07 s (1.8×)  |
| Vulkan  | dots (35.3s) |  14.29 s (2.5×) |  13.68 s (2.6×) |
| CPU     | jfk (11.0s)  |  25.73 s (0.4×) |  19.46 s (0.6×) |
| CPU     | dots (35.3s) |  53.75 s (0.7×) |  43.11 s (0.8×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v2 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-large-v2-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v2.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v2.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v2.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v2.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v2` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `3.725e-06` |   `2.701e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.800e-05` |   `4.611e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `5.424e-05` |   `1.111e-06` | first encoder block |
| `enc.block.31.out`     |  `1.822e-02` |   `3.259e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.213e-03` |   `2.945e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `4.530e-06` |   `2.545e-07` | first decoder block, prompt pass |
| `dec.block.31.out`     |  `6.104e-05` |   `1.777e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `4.387e-05` |   `2.792e-06` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `3.910e-05` |   `9.921e-06` | vocab projection (raw logits) |
| `dec.logits`           |  `7.486e-05` |   `1.652e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `9.537e-06` |   `2.033e-06` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-large-v2 \
  --revision ae46427
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v2/whisper-large-v2-F32.gguf \
  models/whisper-large-v2/whisper-large-v2-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v2
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-large-v2/whisper-large-v2-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
