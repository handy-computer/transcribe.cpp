# Whisper large-v2

<!-- catalog:intro -->
Upstream: [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2) at [`ae46427`](https://huggingface.co/openai/whisper-large-v2/commit/ae46427).

OpenAI Whisper large-v2 — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v2) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`ae46427`](https://huggingface.co/openai/whisper-large-v2/commit/ae46427), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-large-v2-F32.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-F32.gguf) | 6.18 GB | 2.67% |
| F16          | [whisper-large-v2-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-F16.gguf) | 3.11 GB | 2.68% |
| Q8_0         | [whisper-large-v2-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q8_0.gguf) | 1.67 GB | 2.97% |
| Q6_K         | [whisper-large-v2-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q6_K.gguf) | 1.30 GB | 2.83% |
| Q5_K_M       | [whisper-large-v2-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q5_K_M.gguf) | 1.16 GB | 2.71% |
| Q4_K_M       | [whisper-large-v2-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v2-gguf/resolve/main/whisper-large-v2-Q4_K_M.gguf) |  997 MB | 2.46% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 2.83%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  38.45% |
| am       | WER    | 140.81% |
| ar       | WER    |  17.06% |
| as       | WER    | 104.58% |
| az       | WER    |  24.13% |
| be       | WER    |  46.96% |
| bg       | WER    |  15.81% |
| bn       | WER    | 103.42% |
| bs       | WER    |  17.02% |
| ca       | WER    |   5.56% |
| cs       | WER    |  14.42% |
| cy       | WER    |  30.55% |
| da       | WER    |  14.92% |
| de       | WER    |   4.53% |
| el       | WER    |  13.51% |
| en       | WER    |   4.21% |
| es       | WER    |   3.30% |
| et       | WER    |  23.25% |
| fa       | WER    |  34.25% |
| fi       | WER    |   9.58% |
| fil      | WER    |  13.17% |
| fr       | WER    |   5.81% |
| gl       | WER    |  16.57% |
| gu       | WER    | 103.37% |
| ha       | WER    |  92.22% |
| he       | WER    |  27.78% |
| hi       | WER    |  23.27% |
| hr       | WER    |  14.18% |
| hu       | WER    |  17.84% |
| hy       | WER    |  46.93% |
| id       | WER    |   7.43% |
| is       | WER    |  39.59% |
| it       | WER    |   3.59% |
| ja       | CER    |   5.56% |
| jv       | WER    |  69.69% |
| ka       | WER    | 115.24% |
| kk       | WER    |  40.13% |
| km       | CER    | 150.84% |
| kn       | WER    |  47.64% |
| ko       | CER    |   4.99% |
| lb       | WER    |  92.83% |
| ln       | WER    |  79.40% |
| lo       | CER    | 101.65% |
| lt       | WER    |  30.37% |
| lv       | WER    |  24.49% |
| mi       | WER    |  39.72% |
| mk       | WER    |  18.76% |
| ml       | WER    | 101.85% |
| mn       | WER    | 115.70% |
| mr       | WER    |  39.81% |
| ms       | WER    |   9.38% |
| mt       | WER    |  73.79% |
| my       | CER    | 149.11% |
| nb       | WER    |   9.73% |
| ne       | WER    |  47.74% |
| nl       | WER    |   6.76% |
| oc       | WER    |  75.41% |
| pa       | WER    | 102.11% |
| pl       | WER    |   5.87% |
| ps       | WER    |  94.98% |
| pt       | WER    |   4.40% |
| ro       | WER    |  17.10% |
| ru       | WER    |   5.61% |
| sd       | WER    | 148.34% |
| sk       | WER    |  12.49% |
| sl       | WER    |  24.73% |
| sn       | WER    | 127.84% |
| so       | WER    | 106.80% |
| sr       | WER    |  38.06% |
| sv       | WER    |   9.25% |
| sw       | WER    |  41.85% |
| ta       | WER    |  20.12% |
| te       | WER    | 100.76% |
| tg       | WER    |  89.42% |
| th       | CER    |  12.42% |
| tr       | WER    |   7.63% |
| uk       | WER    |   8.17% |
| ur       | WER    |  23.96% |
| uz       | WER    |  91.97% |
| vi       | WER    |  11.25% |
| yo       | WER    |  96.00% |
| zh       | CER    |  15.39% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v2/whisper-large-v2-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 597.7 ms (18.40×) | 590.0 ms (18.65×) |
| Metal   | dots (35.3s) |   1.60 s (22.08×) |   1.54 s (22.89×) |
| CPU     | jfk (11.0s)  |    3.73 s (2.95×) |    4.15 s (2.65×) |
| CPU     | dots (35.3s) |    8.09 s (4.36×) |    8.64 s (4.09×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v2
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.48 s (1.70×) |  6.51 s (1.69×) |
| Vulkan  | dots (35.3s) | 15.24 s (2.32×) | 15.05 s (2.35×) |
| CPU     | jfk (11.0s)  | 13.96 s (0.79×) | 13.69 s (0.80×) |
| CPU     | dots (35.3s) | 31.19 s (1.13×) | 29.47 s (1.20×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v2
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v2.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v2.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v2.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v2.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v2` |

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
  scripts/convert-whisper.py openai/whisper-large-v2 \
  --revision ae46427
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v2/whisper-large-v2-F32.gguf \
  models/whisper-large-v2/whisper-large-v2-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v2
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-large-v2/whisper-large-v2-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
