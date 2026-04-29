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
| F32    | [whisper-medium.en-F32.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-F32.gguf) | 2.85 GB | 2.74% |
| F16    | [whisper-medium.en-F16.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-F16.gguf) | 1.44 GB | 2.73% |
| Q8_0   | [whisper-medium.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q8_0.gguf) | 793 MB | 2.72% |
| Q6_K   | [whisper-medium.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q6_K.gguf) | 618 MB | 2.83% |
| Q5_K_M | [whisper-medium.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q5_K_M.gguf) | 556 MB | 2.74% |
| Q4_K_M | [whisper-medium.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q4_K_M.gguf) | 481 MB | 2.91% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 3.02%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.

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
| Metal   | jfk (11.0s)  | 269.5 ms (40.8×) | 260.7 ms (42.2×) |
| Metal   | dots (35.3s) | 913.8 ms (38.7×) | 854.7 ms (41.3×) |
| CPU     | jfk (11.0s)  |    4.29 s (2.6×) |    3.37 s (3.3×) |
| CPU     | dots (35.3s) |    9.07 s (3.9×) |    7.23 s (4.9×) |

macOS 26.4.1, transcribe.cpp `4d2270e`.

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
| Vulkan  | jfk (11.0s)  |  2.88 s (3.8×)  |  2.87 s (3.8×)  |
| Vulkan  | dots (35.3s) |  7.49 s (4.7×)  |  7.22 s (4.9×)  |
| CPU     | jfk (11.0s)  |  11.53 s (1.0×) |  9.36 s (1.2×)  |
| CPU     | dots (35.3s) |  26.63 s (1.3×) |  21.07 s (1.7×) |

Fedora 43, transcribe.cpp `01127e6`. Vulkan device: `AMD Radeon
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

transcribe.cpp is validated tensor-by-tensor against the transformers
reference (WhisperForConditionalGeneration, fp32 CPU) on the manifest's
case audio. Tolerance budget lives at
[`tests/tolerances/whisper.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper.json).
The reference dtype regime (F32 GGUF + F32 KV cache + production C++ mel
frontend) lights up any frontend regression by design — see the family
note at
[`docs/porting/families/whisper.md`](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/porting/families/whisper.md)
for the full architecture and validation contract.

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-medium.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-medium.en` |

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
