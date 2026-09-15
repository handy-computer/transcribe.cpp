# Whisper small.en

<!-- catalog:intro -->
Upstream: [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en) at [`e872752`](https://huggingface.co/openai/whisper-small.en/commit/e872752).

OpenAI Whisper small.en — converted to GGUF for transcribe.cpp. English-only; faster than the multilingual model at the same size. Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-small.en) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`e872752`](https://huggingface.co/openai/whisper-small.en/commit/e872752), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-small.en-F32.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-F32.gguf) | 969 MB | 3.11% |
| F16          | [whisper-small.en-F16.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-F16.gguf) | 493 MB | 2.97% |
| Q8_0         | [whisper-small.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q8_0.gguf) | 270 MB | 3.09% |
| Q6_K         | [whisper-small.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q6_K.gguf) | 212 MB | 2.97% |
| Q5_K_M       | [whisper-small.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q5_K_M.gguf) | 194 MB | 3.11% |
| Q4_K_M       | [whisper-small.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q4_K_M.gguf) | 172 MB | 3.09% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 3.05%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 6.14% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 119.2 ms (92.31×) | 114.0 ms (96.47×) |
| Metal   | dots (35.3s) | 384.3 ms (91.93×) | 387.5 ms (91.18×) |
| CPU     | jfk (11.0s)  | 597.0 ms (18.42×) | 953.9 ms (11.53×) |
| CPU     | dots (35.3s) |   1.39 s (25.39×) |   1.60 s (22.10×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-small.en
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 961 ms (11.44×) | 926 ms (11.89×) |
| Vulkan  | dots (35.3s) | 2.62 s (13.51×) | 2.56 s (13.82×) |
| CPU     | jfk (11.0s)  |  1.98 s (5.57×) |  1.90 s (5.79×) |
| CPU     | dots (35.3s) |  5.09 s (6.95×) |  4.96 s (7.12×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-small.en
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance. Tolerance budget lives at
[`tests/tolerances/whisper-small.en.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-small.en.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-small.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-small.en.json` |
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

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-small.en/whisper-small.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
