# Whisper large

<!-- catalog:intro -->
Upstream: [`openai/whisper-large`](https://huggingface.co/openai/whisper-large) at [`4ef9b41`](https://huggingface.co/openai/whisper-large/commit/4ef9b41).

OpenAI Whisper large — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-large) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`4ef9b41`](https://huggingface.co/openai/whisper-large/commit/4ef9b41), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-large-F32.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-F32.gguf) | 6.18 GB | 2.72% |
| F16          | [whisper-large-F16.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-F16.gguf) | 3.11 GB | 2.72% |
| Q8_0         | [whisper-large-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q8_0.gguf) | 1.67 GB | 2.71% |
| Q6_K         | [whisper-large-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q6_K.gguf) | 1.30 GB | 2.62% |
| Q5_K_M       | [whisper-large-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q5_K_M.gguf) | 1.16 GB | 2.84% |
| Q4_K_M       | [whisper-large-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-gguf/resolve/main/whisper-large-Q4_K_M.gguf) |  997 MB | 2.67% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 2.73%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  44.92% |
| am       | WER    | 133.86% |
| ar       | WER    |  19.47% |
| as       | WER    | 105.09% |
| az       | WER    |  30.38% |
| be       | WER    |  58.26% |
| bg       | WER    |  19.89% |
| bn       | WER    | 105.08% |
| bs       | WER    |  22.20% |
| ca       | WER    |   7.01% |
| cs       | WER    |  18.77% |
| cy       | WER    |  36.33% |
| da       | WER    |  17.82% |
| de       | WER    |   5.04% |
| el       | WER    |  18.79% |
| en       | WER    |   4.46% |
| es       | WER    |   3.55% |
| et       | WER    |  27.31% |
| fa       | WER    |  37.88% |
| fi       | WER    |  12.73% |
| fil      | WER    |  15.44% |
| fr       | WER    |   6.95% |
| gl       | WER    |  20.10% |
| gu       | WER    | 104.29% |
| ha       | WER    |  90.12% |
| he       | WER    |  30.18% |
| hi       | WER    |  29.10% |
| hr       | WER    |  18.32% |
| hu       | WER    |  21.97% |
| hy       | WER    |  54.73% |
| id       | WER    |   8.93% |
| is       | WER    |  49.76% |
| it       | WER    |   3.67% |
| ja       | CER    |   6.95% |
| jv       | WER    |  86.31% |
| ka       | WER    | 119.65% |
| kk       | WER    |  47.29% |
| km       | CER    | 118.15% |
| kn       | WER    |  74.18% |
| ko       | CER    |   5.25% |
| lb       | WER    |  91.22% |
| ln       | WER    |  81.13% |
| lo       | CER    | 102.35% |
| lt       | WER    |  37.58% |
| lv       | WER    |  30.18% |
| mi       | WER    |  54.38% |
| mk       | WER    |  22.81% |
| ml       | WER    | 101.20% |
| mn       | WER    | 115.92% |
| mr       | WER    |  48.22% |
| ms       | WER    |  11.28% |
| mt       | WER    |  84.11% |
| my       | CER    | 128.00% |
| nb       | WER    |  11.98% |
| ne       | WER    |  55.71% |
| nl       | WER    |   8.48% |
| oc       | WER    |  76.72% |
| pa       | WER    | 102.98% |
| pl       | WER    |   7.44% |
| ps       | WER    |  98.52% |
| pt       | WER    |   4.49% |
| ro       | WER    |  17.52% |
| ru       | WER    |   6.49% |
| sd       | WER    | 176.25% |
| sk       | WER    |  17.17% |
| sl       | WER    |  29.94% |
| sn       | WER    | 142.46% |
| so       | WER    | 105.45% |
| sr       | WER    |  35.57% |
| sv       | WER    |  11.50% |
| sw       | WER    |  52.38% |
| ta       | WER    |  21.70% |
| te       | WER    |  99.27% |
| tg       | WER    |  79.55% |
| th       | CER    |  13.88% |
| tr       | WER    |   8.73% |
| uk       | WER    |   9.87% |
| ur       | WER    |  26.28% |
| uz       | WER    |  96.56% |
| vi       | WER    |  11.51% |
| yo       | WER    | 113.72% |
| zh       | CER    |  19.32% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 596.0 ms (18.46×) | 562.3 ms (19.56×) |
| Metal   | dots (35.3s) |   1.63 s (21.63×) |   1.42 s (24.91×) |
| CPU     | jfk (11.0s)  |    3.70 s (2.97×) |    4.07 s (2.71×) |
| CPU     | dots (35.3s) |    8.05 s (4.39×) |    8.90 s (3.97×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.48 s (1.70×) |  6.51 s (1.69×) |
| Vulkan  | dots (35.3s) | 15.31 s (2.31×) | 15.11 s (2.34×) |
| CPU     | jfk (11.0s)  | 14.00 s (0.79×) | 13.57 s (0.81×) |
| CPU     | dots (35.3s) | 31.33 s (1.13×) | 29.55 s (1.20×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large` |

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

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-large/whisper-large-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
