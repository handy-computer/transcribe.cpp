# Whisper medium.en

<!-- catalog:intro -->
Upstream: [`openai/whisper-medium.en`](https://huggingface.co/openai/whisper-medium.en) at [`2e98eb6`](https://huggingface.co/openai/whisper-medium.en/commit/2e98eb6).

OpenAI Whisper medium.en — converted to GGUF for transcribe.cpp. English-only; faster than the multilingual model at the same size. Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-medium.en) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`2e98eb6`](https://huggingface.co/openai/whisper-medium.en/commit/2e98eb6), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-medium.en-F32.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-F32.gguf) | 3.06 GB | 2.74% |
| F16          | [whisper-medium.en-F16.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-F16.gguf) | 1.54 GB | 2.73% |
| Q8_0         | [whisper-medium.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q8_0.gguf) |  831 MB | 2.72% |
| Q6_K         | [whisper-medium.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q6_K.gguf) |  648 MB | 2.82% |
| Q5_K_M       | [whisper-medium.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q5_K_M.gguf) |  583 MB | 2.75% |
| Q4_K_M       | [whisper-medium.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q4_K_M.gguf) |  504 MB | 2.91% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 3.02%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.88% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 291.6 ms (37.73×) | 270.2 ms (40.72×) |
| Metal   | dots (35.3s) | 912.8 ms (38.71×) |   1.27 s (27.90×) |
| CPU     | jfk (11.0s)  |    1.75 s (6.29×) |    1.89 s (5.83×) |
| CPU     | dots (35.3s) |    3.89 s (9.09×) |    4.15 s (8.51×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-medium.en
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.71 s (4.07×) |  2.65 s (4.16×) |
| Vulkan  | dots (35.3s) |  7.46 s (4.74×) |  7.06 s (5.01×) |
| CPU     | jfk (11.0s)  |  6.20 s (1.77×) |  5.94 s (1.85×) |
| CPU     | dots (35.3s) | 14.99 s (2.36×) | 14.01 s (2.52×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-medium.en
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance. Tolerance budget lives at
[`tests/tolerances/whisper-medium.en.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-medium.en.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-medium.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-medium.en.json` |
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

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-medium.en/whisper-medium.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
