# Whisper large-v3

OpenAI's [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) ported to transcribe.cpp. A 1.55B-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding. v3 family adds Cantonese (yue) on top of v2's 99 languages and switches to a 128-bin mel input.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v3) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`06f233f`](https://huggingface.co/openai/whisper-large-v3/commit/06f233f),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F16    | [whisper-large-v3-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-F16.gguf) | 2.88 GB | 1.81% |
| Q8_0   | [whisper-large-v3-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q8_0.gguf) | 1.55 GB | 1.82% |
| Q6_K   | [whisper-large-v3-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q6_K.gguf) | 1.21 GB | 1.83% |
| Q5_K_M | [whisper-large-v3-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q5_K_M.gguf) | 1.08 GB | 1.84% |
| Q4_K_M | [whisper-large-v3-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q4_K_M.gguf) | 951 MB | 1.86% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.01%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
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

### Apple M4

| Backend | Sample      |          F16 |         Q8_0 |        Q4_K_M |
| ------- | ----------- | -----------: | -----------: | ------------: |
| Metal   | jfk (11.0s) |  1.88 s (6×) |  1.68 s (7×) |   1.63 s (7×) |
| CPU     | jfk (11.0s) | 11.30 s (1×) |  7.44 s (1×) |   7.81 s (1×) |

macOS 26.1, transcribe.cpp `11156dd`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3 \
  --quants f16,q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu \
  --iters 5 --warmup 2 \
  --name whisper-large-v3-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.40 s (1.7×)  |  6.46 s (1.7×)  |
| Vulkan  | dots (35.3s) |  15.17 s (2.3×) |  14.94 s (2.4×) |
| CPU     | jfk (11.0s)  |  25.59 s (0.4×) |  19.96 s (0.6×) |
| CPU     | dots (35.3s) |  53.80 s (0.7×) |  43.18 s (0.8×) |

Fedora 43, transcribe.cpp `01127e6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-large-v3-publication
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
| Manifest | `tests/golden/whisper/whisper-large-v3.manifest.json` |
| Tolerance file | `tests/tolerances/whisper.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v3` |

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
  scripts/convert-whisper.py openai/whisper-large-v3 \
  --revision 06f233f
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v3/whisper-large-v3-F16.gguf \
  models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v3
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
