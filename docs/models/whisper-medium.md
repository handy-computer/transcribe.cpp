# Whisper medium

<!-- catalog:intro -->
Upstream: [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) at [`abdf7c3`](https://huggingface.co/openai/whisper-medium/commit/abdf7c3).

OpenAI Whisper medium — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-medium) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`abdf7c3`](https://huggingface.co/openai/whisper-medium/commit/abdf7c3), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-medium-F32.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-F32.gguf) | 3.06 GB | 2.64% |
| F16          | [whisper-medium-F16.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-F16.gguf) | 1.54 GB | 2.63% |
| Q8_0         | [whisper-medium-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q8_0.gguf) |  832 MB | 2.64% |
| Q6_K         | [whisper-medium-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q6_K.gguf) |  648 MB | 2.59% |
| Q5_K_M       | [whisper-medium-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q5_K_M.gguf) |  583 MB | 2.62% |
| Q4_K_M       | [whisper-medium-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-medium-gguf/resolve/main/whisper-medium-Q4_K_M.gguf) |  504 MB | 2.59% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 2.90%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  47.33% |
| am       | WER    | 116.27% |
| ar       | WER    |  21.90% |
| as       | WER    | 101.63% |
| az       | WER    |  34.57% |
| be       | WER    |  61.87% |
| bg       | WER    |  23.00% |
| bn       | WER    | 102.52% |
| bs       | WER    |  26.69% |
| ca       | WER    |   8.02% |
| cs       | WER    |  22.85% |
| cy       | WER    |  40.08% |
| da       | WER    |  21.17% |
| de       | WER    |   6.23% |
| el       | WER    |  20.06% |
| en       | WER    |   4.64% |
| es       | WER    |   3.80% |
| et       | WER    |  31.52% |
| fa       | WER    |  42.57% |
| fi       | WER    |  14.67% |
| fil      | WER    |  18.36% |
| fr       | WER    |   8.07% |
| gl       | WER    |  22.24% |
| gu       | WER    | 104.11% |
| ha       | WER    |  95.12% |
| he       | WER    |  33.69% |
| hi       | WER    |  26.09% |
| hr       | WER    |  21.19% |
| hu       | WER    |  26.07% |
| hy       | WER    |  58.42% |
| id       | WER    |  10.79% |
| is       | WER    |  51.60% |
| it       | WER    |   4.17% |
| ja       | CER    |   7.35% |
| jv       | WER    |  73.01% |
| ka       | WER    | 128.01% |
| kk       | WER    |  53.09% |
| km       | CER    | 108.92% |
| kn       | WER    |  87.30% |
| ko       | CER    |   5.46% |
| lb       | WER    |  98.33% |
| ln       | WER    |  92.09% |
| lo       | CER    | 101.16% |
| lt       | WER    |  43.34% |
| lv       | WER    |  33.58% |
| mi       | WER    |  95.79% |
| mk       | WER    |  24.75% |
| ml       | WER    | 101.00% |
| mn       | WER    | 110.55% |
| mr       | WER    |  58.43% |
| ms       | WER    |  13.23% |
| mt       | WER    |  85.06% |
| my       | CER    | 117.51% |
| nb       | WER    |  13.66% |
| ne       | WER    |  54.08% |
| nl       | WER    |  10.40% |
| oc       | WER    |  81.55% |
| pa       | WER    | 103.19% |
| pl       | WER    |   8.59% |
| ps       | WER    | 105.77% |
| pt       | WER    |   5.07% |
| ro       | WER    |  24.17% |
| ru       | WER    |   7.30% |
| sd       | WER    | 132.27% |
| sk       | WER    |  18.73% |
| sl       | WER    |  33.98% |
| sn       | WER    | 134.28% |
| so       | WER    | 102.63% |
| sr       | WER    |  55.14% |
| sv       | WER    |  12.47% |
| sw       | WER    |  57.09% |
| ta       | WER    |  23.72% |
| te       | WER    | 102.69% |
| tg       | WER    |  76.32% |
| th       | CER    |  16.08% |
| tr       | WER    |   9.35% |
| uk       | WER    |  11.59% |
| ur       | WER    |  28.67% |
| uz       | WER    | 115.02% |
| vi       | WER    |  13.74% |
| yo       | WER    | 109.00% |
| zh       | CER    |  13.13% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 343.4 ms (32.03×) | 325.9 ms (33.76×) |
| Metal   | dots (35.3s) |   1.02 s (34.62×) | 891.0 ms (39.65×) |
| CPU     | jfk (11.0s)  |    1.88 s (5.84×) |    2.08 s (5.28×) |
| CPU     | dots (35.3s) |    4.08 s (8.66×) |    4.40 s (8.02×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-medium
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  3.04 s (3.62×) |  3.01 s (3.65×) |
| Vulkan  | dots (35.3s) |  7.77 s (4.55×) |  7.40 s (4.77×) |
| CPU     | jfk (11.0s)  |  6.93 s (1.59×) |  6.65 s (1.65×) |
| CPU     | dots (35.3s) | 15.60 s (2.26×) | 14.67 s (2.41×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-medium
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-medium.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-medium.json).

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

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-medium/whisper-medium-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
