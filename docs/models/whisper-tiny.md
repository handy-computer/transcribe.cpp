# Whisper tiny

<!-- catalog:intro -->
Upstream: [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) at [`169d4a4`](https://huggingface.co/openai/whisper-tiny/commit/169d4a4).

OpenAI Whisper tiny — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-tiny) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`169d4a4`](https://huggingface.co/openai/whisper-tiny/commit/169d4a4), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-tiny-F32.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-F32.gguf) | 153 MB | 7.49% |
| F16          | [whisper-tiny-F16.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-F16.gguf) |  80 MB | 7.48% |
| Q8_0         | [whisper-tiny-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q8_0.gguf) |  46 MB | 7.52% |
| Q6_K         | [whisper-tiny-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q6_K.gguf) |  45 MB | 7.54% |
| Q5_K_M       | [whisper-tiny-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q5_K_M.gguf) |  44 MB | 7.82% |
| Q4_K_M       | [whisper-tiny-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q4_K_M.gguf) |  44 MB | 7.78% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 7.54%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  95.30% |
| am       | WER    | 122.14% |
| ar       | WER    |  67.48% |
| as       | WER    | 100.56% |
| az       | WER    |  94.73% |
| be       | WER    |  96.03% |
| bg       | WER    |  84.87% |
| bn       | WER    | 100.37% |
| bs       | WER    |  87.22% |
| ca       | WER    |  46.33% |
| cs       | WER    |  87.17% |
| cy       | WER    | 116.68% |
| da       | WER    |  88.42% |
| de       | WER    |  31.56% |
| el       | WER    |  77.76% |
| en       | WER    |  13.84% |
| es       | WER    |  18.95% |
| et       | WER    | 101.33% |
| fa       | WER    | 100.18% |
| fi       | WER    |  68.58% |
| fil      | WER    |  70.63% |
| fr       | WER    |  44.19% |
| gl       | WER    |  59.44% |
| gu       | WER    | 100.26% |
| ha       | WER    | 105.17% |
| he       | WER    |  77.35% |
| hi       | WER    | 101.52% |
| hr       | WER    |  84.36% |
| hu       | WER    |  90.54% |
| hy       | WER    | 110.50% |
| id       | WER    |  60.49% |
| is       | WER    | 115.88% |
| it       | WER    |  31.24% |
| ja       | CER    |  39.38% |
| jv       | WER    | 106.07% |
| ka       | WER    | 110.34% |
| kk       | WER    | 136.56% |
| km       | CER    | 111.78% |
| kn       | WER    | 100.32% |
| ko       | CER    |  19.07% |
| lb       | WER    |  99.69% |
| ln       | WER    | 103.55% |
| lo       | CER    | 105.48% |
| lt       | WER    | 105.00% |
| lv       | WER    |  95.98% |
| mi       | WER    |  96.27% |
| mk       | WER    |  78.61% |
| ml       | WER    | 100.03% |
| mn       | WER    | 110.21% |
| mr       | WER    | 100.82% |
| ms       | WER    |  62.17% |
| mt       | WER    |  99.85% |
| my       | CER    | 108.53% |
| nb       | WER    |  67.36% |
| ne       | WER    | 101.30% |
| nl       | WER    |  54.86% |
| oc       | WER    |  96.44% |
| pa       | WER    | 100.54% |
| pl       | WER    |  54.22% |
| ps       | WER    | 101.04% |
| pt       | WER    |  24.07% |
| ro       | WER    |  82.43% |
| ru       | WER    |  35.64% |
| sd       | WER    | 105.42% |
| sk       | WER    |  82.20% |
| sl       | WER    |  91.29% |
| sn       | WER    | 118.33% |
| so       | WER    | 105.31% |
| sr       | WER    |  89.20% |
| sv       | WER    |  59.36% |
| sw       | WER    |  99.96% |
| ta       | WER    |  87.46% |
| te       | WER    | 101.18% |
| tg       | WER    | 102.39% |
| th       | CER    |  54.82% |
| tr       | WER    |  47.98% |
| uk       | WER    |  57.49% |
| ur       | WER    |  71.42% |
| uz       | WER    | 103.19% |
| vi       | WER    |  64.49% |
| yo       | WER    | 102.29% |
| zh       | CER    |  41.91% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-tiny/whisper-tiny-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  39.0 ms (282.04×) |  39.5 ms (278.34×) |
| Metal   | dots (35.3s) | 139.4 ms (253.55×) | 141.4 ms (249.80×) |
| CPU     | jfk (11.0s)  |  99.9 ms (110.09×) |  99.7 ms (110.34×) |
| CPU     | dots (35.3s) | 250.5 ms (141.07×) | 253.0 ms (139.65×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-tiny
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 254 ms (43.22×) | 253 ms (43.55×) |
| Vulkan  | dots (35.3s) | 822 ms (42.96×) | 825 ms (42.82×) |
| CPU     | jfk (11.0s)  | 306 ms (35.96×) | 311 ms (35.39×) |
| CPU     | dots (35.3s) | 811 ms (43.55×) | 820 ms (43.08×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-tiny
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 21 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-tiny.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-tiny.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-tiny.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-tiny.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-tiny` |

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
  scripts/convert-whisper.py openai/whisper-tiny \
  --revision 169d4a4
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-tiny/whisper-tiny-F32.gguf \
  models/whisper-tiny/whisper-tiny-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-tiny
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-tiny/whisper-tiny-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
