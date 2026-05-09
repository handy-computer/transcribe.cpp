# Whisper large

OpenAI's [`openai/whisper-large`](https://huggingface.co/openai/whisper-large) ported to transcribe.cpp. A 1.55B-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-large) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`4ef9b41`](https://huggingface.co/openai/whisper-large/commit/4ef9b41),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-large-F32.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-F32.gguf) | 5.75 GB | 2.72% |
| F16    | [whisper-large-F16.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-F16.gguf) | 2.89 GB | 2.74% |
| Q8_0   | [whisper-large-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q8_0.gguf) | 1.55 GB | 2.74% |
| Q6_K   | [whisper-large-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q6_K.gguf) | 1.21 GB | 2.62% |
| Q5_K_M | [whisper-large-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q5_K_M.gguf) | 1.08 GB | 2.70% |
| Q4_K_M | [whisper-large-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q4_K_M.gguf) | 950 MB | 2.67% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.73%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large/whisper-large-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 576.9 ms (19.1×) | 502.5 ms (21.9×) |
| Metal   | dots (35.3s) |   1.65 s (21.5×) |   1.55 s (22.8×) |
| CPU     | jfk (11.0s)  |    9.63 s (1.1×) |    7.43 s (1.5×) |
| CPU     | dots (35.3s) |   19.88 s (1.8×) |   15.49 s (2.3×) |

macOS 26.4.1, transcribe.cpp `4d2270e`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-large-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.26 s (1.8×)  |  6.13 s (1.8×)  |
| Vulkan  | dots (35.3s) |  14.41 s (2.5×) |  13.72 s (2.6×) |
| CPU     | jfk (11.0s)  |  26.18 s (0.4×) |  19.83 s (0.6×) |
| CPU     | dots (35.3s) |  55.64 s (0.6×) |  43.98 s (0.8×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-large-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `2.146e-06` |   `3.695e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `2.098e-05` |   `3.419e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `2.718e-05` |   `8.861e-07` | first encoder block |
| `enc.block.31.out`     |  `3.262e-01` |   `7.382e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.916e-03` |   `5.219e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `1.907e-05` |   `3.328e-07` | first decoder block, prompt pass |
| `dec.block.31.out`     |  `2.441e-04` |   `6.015e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `2.522e-04` |   `1.035e-05` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `9.346e-05` |   `1.508e-05` | vocab projection (raw logits) |
| `dec.logits`           |  `8.965e-05` |   `2.757e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `3.052e-05` |   `1.148e-05` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-large \
  --revision 4ef9b41
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large/whisper-large-F32.gguf \
  models/whisper-large/whisper-large-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-large/whisper-large-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
