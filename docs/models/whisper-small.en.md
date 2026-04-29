# Whisper small.en

OpenAI's [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en) ported to transcribe.cpp. A 244M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-small.en) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`e872752`](https://huggingface.co/openai/whisper-small.en/commit/e872752),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-small.en-F32.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-F32.gguf) | 924 MB | 3.09% |
| F16    | [whisper-small.en-F16.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-F16.gguf) | 470 MB | 2.97% |
| Q8_0   | [whisper-small.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q8_0.gguf) | 257 MB | 3.09% |
| Q6_K   | [whisper-small.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q6_K.gguf) | 202 MB | 2.97% |
| Q5_K_M | [whisper-small.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q5_K_M.gguf) | 185 MB | 3.12% |
| Q4_K_M | [whisper-small.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q4_K_M.gguf) | 164 MB | 3.08% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 3.05%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-small.en/whisper-small.en-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 119.2 ms (92.3×) | 110.6 ms (99.4×) |
| Metal   | dots (35.3s) | 392.6 ms (90.0×) | 398.5 ms (88.7×) |
| CPU     | jfk (11.0s)  |    1.31 s (8.4×) |    1.13 s (9.8×) |
| CPU     | dots (35.3s) |   2.89 s (12.2×) |   2.52 s (14.0×) |

macOS 26.4.1, transcribe.cpp `4d2270e`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-small.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-small.en-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  1.08 s (10.2×) |  1.04 s (10.6×) |
| Vulkan  | dots (35.3s) |  2.79 s (12.7×) |  2.74 s (12.9×) |
| CPU     | jfk (11.0s)  |  3.68 s (3.0×)  |  2.95 s (3.7×)  |
| CPU     | dots (35.3s) |  8.38 s (4.2×)  |  7.16 s (4.9×)  |

Fedora 43, transcribe.cpp `01127e6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-small.en \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-small.en-publication
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
| Manifest | `tests/golden/whisper/whisper-small.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-small.en` |

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
  scripts/convert-whisper.py openai/whisper-small.en \
  --revision e872752
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-small.en/whisper-small.en-F32.gguf \
  models/whisper-small.en/whisper-small.en-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-small.en
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-small.en/whisper-small.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
