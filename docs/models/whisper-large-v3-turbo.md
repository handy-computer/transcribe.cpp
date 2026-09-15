# Whisper large-v3-turbo

<!-- catalog:intro -->
Upstream: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) at [`41f01f3`](https://huggingface.co/openai/whisper-large-v3-turbo/commit/41f01f3).

OpenAI Whisper large-v3-turbo — converted to GGUF for transcribe.cpp. Multilingual transcription and language detection; unlike the full large-v3 model, this turbo variant does not support speech translation. The v3 family adds Cantonese (yue) and uses a 128-bin mel input. Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (100 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding. v3 family adds Cantonese (yue) on top of v2's 99 languages and switches to a 128-bin mel input.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v3-turbo) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`41f01f3`](https://huggingface.co/openai/whisper-large-v3-turbo/commit/41f01f3), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F16          | [whisper-large-v3-turbo-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-F16.gguf) | 1.63 GB | 2.01% |
| Q8_0         | [whisper-large-v3-turbo-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q8_0.gguf) |  886 MB | 2.01% |
| Q6_K         | [whisper-large-v3-turbo-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q6_K.gguf) |  693 MB | 2.01% |
| Q5_K_M       | [whisper-large-v3-turbo-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q5_K_M.gguf) |  620 MB | 2.03% |
| Q4_K_M       | [whisper-large-v3-turbo-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q4_K_M.gguf) |  536 MB | 2.04% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 2.10%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  36.06% |
| am       | WER    | 146.29% |
| ar       | WER    |  15.48% |
| as       | WER    | 101.22% |
| az       | WER    |  23.15% |
| be       | WER    |  50.65% |
| bg       | WER    |  13.58% |
| bn       | WER    |  67.53% |
| bs       | WER    |  14.77% |
| ca       | WER    |   5.42% |
| cs       | WER    |  11.81% |
| cy       | WER    |  36.42% |
| da       | WER    |  13.60% |
| de       | WER    |   4.54% |
| el       | WER    |  13.26% |
| en       | WER    |   4.38% |
| es       | WER    |   3.12% |
| et       | WER    |  18.44% |
| fa       | WER    |  30.56% |
| fi       | WER    |   8.29% |
| fil      | WER    |  12.08% |
| fr       | WER    |   5.51% |
| gl       | WER    |  12.76% |
| gu       | WER    |  78.95% |
| ha       | WER    |  97.24% |
| he       | WER    |  29.71% |
| hi       | WER    |  18.85% |
| hr       | WER    |  12.54% |
| hu       | WER    |  15.07% |
| hy       | WER    |  45.62% |
| id       | WER    |   7.20% |
| is       | WER    |  21.39% |
| it       | WER    |   2.77% |
| ja       | CER    |   4.82% |
| jv       | WER    |  53.80% |
| ka       | WER    | 109.21% |
| kk       | WER    |  21.27% |
| km       | CER    |  95.20% |
| kn       | WER    |  32.57% |
| ko       | CER    |   5.24% |
| lb       | WER    |  87.21% |
| ln       | WER    |  75.39% |
| lo       | CER    | 115.41% |
| lt       | WER    |  25.11% |
| lv       | WER    |  19.53% |
| mi       | WER    |  48.91% |
| mk       | WER    |  17.85% |
| ml       | WER    |  98.75% |
| mn       | WER    | 101.49% |
| mr       | WER    |  36.12% |
| ms       | WER    |   8.64% |
| mt       | WER    |  70.92% |
| my       | CER    | 121.67% |
| nb       | WER    |   9.10% |
| ne       | WER    |  43.15% |
| nl       | WER    |   5.98% |
| oc       | WER    |  70.94% |
| pa       | WER    |  99.53% |
| pl       | WER    |   5.81% |
| ps       | WER    |  91.81% |
| pt       | WER    |   4.17% |
| ro       | WER    |  10.90% |
| ru       | WER    |   5.93% |
| sd       | WER    | 122.10% |
| sk       | WER    |  10.21% |
| sl       | WER    |  20.56% |
| sn       | WER    | 110.94% |
| so       | WER    | 101.29% |
| sr       | WER    |  32.36% |
| sv       | WER    |   8.72% |
| sw       | WER    |  33.96% |
| ta       | WER    |  27.41% |
| te       | WER    |  63.03% |
| tg       | WER    | 106.06% |
| th       | CER    |  13.15% |
| tr       | WER    |   6.97% |
| uk       | WER    |   7.31% |
| ur       | WER    |  23.19% |
| uz       | WER    | 102.52% |
| vi       | WER    |   9.48% |
| yo       | WER    |  99.38% |
| yue      | CER    |  34.62% |
| zh       | CER    |   8.50% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 303.8 ms (36.21×) | 291.1 ms (37.79×) |
| Metal   | dots (35.3s) | 691.2 ms (51.12×) | 666.7 ms (53.00×) |
| CPU     | jfk (11.0s)  |    2.85 s (3.85×) |    3.11 s (3.54×) |
| CPU     | dots (35.3s) |    5.80 s (6.09×) |    6.30 s (5.61×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v3-turbo
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  4.40 s (2.50×) |  4.45 s (2.47×) |
| Vulkan  | dots (35.3s) |  9.59 s (3.69×) |  9.67 s (3.65×) |
| CPU     | jfk (11.0s)  |  9.85 s (1.12×) |  9.58 s (1.15×) |
| CPU     | dots (35.3s) | 20.36 s (1.74×) | 19.72 s (1.79×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v3-turbo
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 22 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v3-turbo.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v3-turbo.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v3-turbo.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v3-turbo.json` |
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

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
