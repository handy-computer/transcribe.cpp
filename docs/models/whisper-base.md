# Whisper base

OpenAI's [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) ported to transcribe.cpp. A 74M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-base) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`e37978b`](https://huggingface.co/openai/whisper-base/commit/e37978b),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-base-F32.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-F32.gguf) | 279 MB | 5.10% |
| F16    | [whisper-base-F16.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-F16.gguf) | 144 MB | 5.10% |
| Q8_0   | [whisper-base-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q8_0.gguf) | 81 MB | 5.12% |
| Q6_K   | [whisper-base-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q6_K.gguf) | 65 MB | 5.12% |
| Q5_K_M | [whisper-base-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q5_K_M.gguf) | 61 MB | 5.19% |
| Q4_K_M | [whisper-base-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q4_K_M.gguf) | 56 MB | 5.36% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 5.009%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-base/whisper-base-Q8_0.gguf \
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

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  |  52.1 ms (211.0×) |  53.6 ms (205.2×) |
| Metal   | dots (35.3s) | 170.0 ms (207.8×) | 168.3 ms (209.9×) |
| CPU     | jfk (11.0s)  |  374.0 ms (29.4×) |  347.6 ms (31.6×) |
| CPU     | dots (35.3s) |  806.1 ms (43.8×) |  750.3 ms (47.1×) |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-base \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-base-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  351 ms (31.3×) |  356 ms (30.9×) |
| Vulkan  | dots (35.3s) |  922 ms (38.3×) |  946 ms (37.4×) |
| CPU     | jfk (11.0s)  |   1.11 s (9.9×) |  913 ms (12.1×) |
| CPU     | dots (35.3s) |  2.54 s (13.9×) |  2.27 s (15.5×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-base \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-base-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-base.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-base.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-base.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-base.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-base` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `5.960e-06` |   `6.348e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.466e-05` |   `2.544e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `1.794e-05` |   `6.097e-07` | first encoder block |
| `enc.block.5.out`      |  `1.513e-02` |   `5.212e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.699e-03` |   `2.588e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `5.603e-06` |   `2.874e-07` | first decoder block, prompt pass |
| `dec.block.5.out`      |  `1.373e-04` |   `1.643e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.526e-04` |   `9.441e-06` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `6.890e-05` |   `1.462e-05` | vocab projection (raw logits) |
| `dec.logits`           |  `4.768e-05` |   `9.442e-06` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `8.774e-05` |   `5.206e-05` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-base \
  --revision e37978b
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-base/whisper-base-F32.gguf \
  models/whisper-base/whisper-base-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-base
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-base/whisper-base-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
