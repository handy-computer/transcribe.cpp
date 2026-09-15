# Whisper base

<!-- catalog:intro -->
Upstream: [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) at [`e37978b`](https://huggingface.co/openai/whisper-base/commit/e37978b).

OpenAI Whisper base — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-base) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`e37978b`](https://huggingface.co/openai/whisper-base/commit/e37978b), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-base-F32.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-F32.gguf) | 292 MB | 5.11% |
| F16          | [whisper-base-F16.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-F16.gguf) | 151 MB | 5.10% |
| Q8_0         | [whisper-base-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q8_0.gguf) |  85 MB | 5.12% |
| Q6_K         | [whisper-base-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q6_K.gguf) |  68 MB | 5.11% |
| Q5_K_M       | [whisper-base-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q5_K_M.gguf) |  64 MB | 5.19% |
| Q4_K_M       | [whisper-base-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q4_K_M.gguf) |  59 MB | 5.36% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 5.009%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  83.05% |
| am       | WER    | 150.97% |
| ar       | WER    |  52.74% |
| as       | WER    | 100.60% |
| az       | WER    |  81.22% |
| be       | WER    |  92.72% |
| bg       | WER    |  70.53% |
| bn       | WER    | 100.73% |
| bs       | WER    |  71.53% |
| ca       | WER    |  29.48% |
| cs       | WER    |  70.14% |
| cy       | WER    |  98.19% |
| da       | WER    |  63.85% |
| de       | WER    |  19.69% |
| el       | WER    |  59.14% |
| en       | WER    |   9.88% |
| es       | WER    |  11.15% |
| et       | WER    |  81.71% |
| fa       | WER    |  87.72% |
| fi       | WER    |  49.46% |
| fil      | WER    |  49.32% |
| fr       | WER    |  27.91% |
| gl       | WER    |  50.06% |
| gu       | WER    | 100.40% |
| ha       | WER    | 108.15% |
| he       | WER    |  65.56% |
| hi       | WER    | 100.01% |
| hr       | WER    |  64.23% |
| hu       | WER    |  72.26% |
| hy       | WER    | 127.56% |
| id       | WER    |  38.02% |
| is       | WER    |  99.32% |
| it       | WER    |  17.26% |
| ja       | CER    |  25.28% |
| jv       | WER    |  93.07% |
| ka       | WER    | 117.78% |
| kk       | WER    |  99.79% |
| km       | CER    | 134.48% |
| kn       | WER    | 102.88% |
| ko       | CER    |  12.98% |
| lb       | WER    | 107.78% |
| ln       | WER    | 102.73% |
| lo       | CER    | 104.35% |
| lt       | WER    |  91.78% |
| lv       | WER    |  84.60% |
| mi       | WER    |  81.65% |
| mk       | WER    |  63.95% |
| ml       | WER    | 102.84% |
| mn       | WER    | 124.42% |
| mr       | WER    | 100.42% |
| ms       | WER    |  40.87% |
| mt       | WER    | 103.46% |
| my       | CER    | 130.63% |
| nb       | WER    |  49.26% |
| ne       | WER    | 101.15% |
| nl       | WER    |  36.75% |
| oc       | WER    |  88.62% |
| pa       | WER    | 101.13% |
| pl       | WER    |  35.68% |
| ps       | WER    | 101.19% |
| pt       | WER    |  13.91% |
| ro       | WER    |  62.16% |
| ru       | WER    |  22.92% |
| sd       | WER    | 103.23% |
| sk       | WER    |  65.77% |
| sl       | WER    |  77.90% |
| sn       | WER    | 134.76% |
| so       | WER    | 107.06% |
| sr       | WER    |  69.25% |
| sv       | WER    |  42.40% |
| sw       | WER    | 100.69% |
| ta       | WER    |  58.84% |
| te       | WER    | 101.77% |
| tg       | WER    | 108.30% |
| th       | CER    |  38.10% |
| tr       | WER    |  31.09% |
| uk       | WER    |  42.03% |
| ur       | WER    |  55.42% |
| uz       | WER    | 111.42% |
| vi       | WER    |  42.60% |
| yo       | WER    | 103.28% |
| zh       | CER    |  36.21% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-base/whisper-base-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  54.5 ms (201.79×) |  54.6 ms (201.42×) |
| Metal   | dots (35.3s) | 191.6 ms (184.36×) | 187.2 ms (188.77×) |
| CPU     | jfk (11.0s)  |  185.3 ms (59.35×) |  210.8 ms (52.18×) |
| CPU     | dots (35.3s) |  436.6 ms (80.93×) |  482.5 ms (73.23×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-base
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 382 ms (28.80×) | 408 ms (27.00×) |
| Vulkan  | dots (35.3s) | 1.21 s (29.23×) | 1.21 s (29.14×) |
| CPU     | jfk (11.0s)  | 646 ms (17.03×) | 635 ms (17.32×) |
| CPU     | dots (35.3s) | 1.63 s (21.63×) | 1.56 s (22.63×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-base
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-base.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-base.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-base.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-base.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-base` |

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
  scripts/convert-whisper.py openai/whisper-base \
  --revision e37978b
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-base/whisper-base-F32.gguf \
  models/whisper-base/whisper-base-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-base
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-base/whisper-base-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
