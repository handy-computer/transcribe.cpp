# Whisper base.en

<!-- catalog:intro -->
Upstream: [`openai/whisper-base.en`](https://huggingface.co/openai/whisper-base.en) at [`911407f`](https://huggingface.co/openai/whisper-base.en/commit/911407f).

OpenAI Whisper base.en — converted to GGUF for transcribe.cpp. English-only; faster than the multilingual model at the same size. Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-base.en) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`911407f`](https://huggingface.co/openai/whisper-base.en/commit/911407f), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-base.en-F32.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-F32.gguf) | 292 MB | 4.30% |
| F16          | [whisper-base.en-F16.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-F16.gguf) | 151 MB | 4.13% |
| Q8_0         | [whisper-base.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q8_0.gguf) |  85 MB | 4.16% |
| Q6_K         | [whisper-base.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q6_K.gguf) |  68 MB | 4.15% |
| Q5_K_M       | [whisper-base.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q5_K_M.gguf) |  64 MB | 4.16% |
| Q4_K_M       | [whisper-base.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-base.en-gguf/resolve/main/whisper-base.en-Q4_K_M.gguf) |  59 MB | 4.29% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 4.25%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 7.60% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |               Q8_0 |             Q4_K_M |
| ------- | ------------ | -----------------: | -----------------: |
| Metal   | jfk (11.0s)  |  51.8 ms (212.45×) |  51.8 ms (212.42×) |
| Metal   | dots (35.3s) | 185.2 ms (190.73×) | 181.2 ms (194.94×) |
| CPU     | jfk (11.0s)  |  173.6 ms (63.35×) |  195.3 ms (56.31×) |
| CPU     | dots (35.3s) |  423.6 ms (83.40×) |  463.9 ms (76.16×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-base.en
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 376 ms (29.22×) | 385 ms (28.56×) |
| Vulkan  | dots (35.3s) | 1.01 s (34.81×) | 1.06 s (33.38×) |
| CPU     | jfk (11.0s)  | 589 ms (18.68×) | 575 ms (19.14×) |
| CPU     | dots (35.3s) | 1.57 s (22.52×) | 1.50 s (23.60×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-base.en
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance. Tolerance budget lives at
[`tests/tolerances/whisper-base.en.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-base.en.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-base.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-base.en.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-base.en` |

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

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-base.en/whisper-base.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
