# Whisper large-v3-turbo

OpenAI's [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) ported to transcribe.cpp. A 809M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding. v3 family adds Cantonese (yue) on top of v2's 99 languages and switches to a 128-bin mel input.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v3-turbo) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`41f01f3`](https://huggingface.co/openai/whisper-large-v3-turbo/commit/41f01f3),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F16    | [whisper-large-v3-turbo-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-F16.gguf) | 1.51 GB | 2.01% |
| Q8_0   | [whisper-large-v3-turbo-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q8_0.gguf) | 845 MB | 2.01% |
| Q6_K   | [whisper-large-v3-turbo-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q6_K.gguf) | 660 MB | 2.01% |
| Q5_K_M | [whisper-large-v3-turbo-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q5_K_M.gguf) | 591 MB | 2.03% |
| Q4_K_M | [whisper-large-v3-turbo-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q4_K_M.gguf) | 511 MB | 2.04% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.10%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
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
| Metal   | jfk (11.0s) | 1.06 s (10×) | 1.05 s (10×) |  1.06 s (10×) |
| CPU     | jfk (11.0s) |  8.61 s (1×) |  5.70 s (2×) |   6.07 s (2×) |

macOS 26.1, transcribe.cpp `11156dd`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3-turbo \
  --quants f16,q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu \
  --iters 5 --warmup 2 \
  --name whisper-large-v3-turbo-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |           Q8_0 |        Q4_K_M |
| ------- | ------------ | -------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  4.77 s (2.3×) | 4.92 s (2.2×) |
| Vulkan  | dots (35.3s) | 10.16 s (3.5×) | 10.26 s (3.4×) |
| CPU     | jfk (11.0s)  | 19.85 s (0.6×) | 15.74 s (0.7×) |
| CPU     | dots (35.3s) | 40.18 s (0.9×) | 32.22 s (1.1×) |

Fedora 43, transcribe.cpp `5fccd5d`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3-turbo \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-large-v3-turbo-publication
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
| Manifest | `tests/golden/whisper/whisper-large-v3-turbo.manifest.json` |
| Tolerance file | `tests/tolerances/whisper.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v3-turbo` |

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
  scripts/convert-whisper.py openai/whisper-large-v3-turbo \
  --revision 41f01f3
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v3-turbo/whisper-large-v3-turbo-F16.gguf \
  models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v3-turbo
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
