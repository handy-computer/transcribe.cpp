# Whisper medium

OpenAI's [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) ported to transcribe.cpp. A 769M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-medium) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`abdf7c3`](https://huggingface.co/openai/whisper-medium/commit/abdf7c3),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-medium-F32.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-F32.gguf) | 2.85 GB | 2.63% |
| F16    | [whisper-medium-F16.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-F16.gguf) | 1.44 GB | 2.63% |
| Q8_0   | [whisper-medium-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q8_0.gguf) | 793 MB | 2.64% |
| Q6_K   | [whisper-medium-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q6_K.gguf) | 618 MB | 2.59% |
| Q5_K_M | [whisper-medium-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q5_K_M.gguf) | 556 MB | 2.62% |
| Q4_K_M | [whisper-medium-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q4_K_M.gguf) | 481 MB | 2.59% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.90%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-medium/whisper-medium-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 302.3 ms (36.4×)  | 285.6 ms (38.5×)  |
| Metal   | dots (35.3s) |   941.3 ms (37.5×) |   878.1 ms (40.2×) |
| CPU     | jfk (11.0s)  |     4.75 s (2.3×) |     3.90 s (2.8×) |
| CPU     | dots (35.3s) |     9.62 s (3.7×) |     7.93 s (4.5×) |

macOS 26.4.1, transcribe.cpp `e6a8a27`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-medium \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-medium-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |           Q8_0 |        Q4_K_M |
| ------- | ------------ | -------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  3.46 s (3.2×) | 3.49 s (3.1×) |
| Vulkan  | dots (35.3s) |  8.36 s (4.2×) | 8.18 s (4.3×) |
| CPU     | jfk (11.0s)  | 13.14 s (0.8×) | 10.47 s (1.1×) |
| CPU     | dots (35.3s) | 27.87 s (1.3×) | 22.57 s (1.6×) |

Fedora 43, transcribe.cpp `5fccd5d`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-medium \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-medium-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers
reference (WhisperForConditionalGeneration, fp32 CPU) on the manifest's
case audio. Tolerance budget lives at
[`tests/tolerances/whisper-medium.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-medium.json).
The reference dtype regime (F32 GGUF + F32 KV cache + production C++ mel
frontend) lights up any frontend regression by design — see the family
note at
[`docs/porting/families/whisper.md`](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/porting/families/whisper.md)
for the full architecture and validation contract.

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-medium.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-medium.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-medium` |

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
  scripts/convert-whisper.py openai/whisper-medium \
  --revision abdf7c3
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-medium/whisper-medium-F32.gguf \
  models/whisper-medium/whisper-medium-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-medium
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-medium/whisper-medium-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
