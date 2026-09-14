# Whisper small

<!-- catalog:intro -->
Upstream: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) at [`973afd2`](https://huggingface.co/openai/whisper-small/commit/973afd2).

OpenAI Whisper small — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-small) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`973afd2`](https://huggingface.co/openai/whisper-small/commit/973afd2), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1) on 2026-04-26.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [whisper-small-F32.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-F32.gguf) | 969 MB | 3.34% |
| F16          | [whisper-small-F16.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-F16.gguf) | 493 MB | 3.33% |
| Q8_0         | [whisper-small-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q8_0.gguf) | 270 MB | 3.33% |
| Q6_K         | [whisper-small-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q6_K.gguf) | 212 MB | 3.33% |
| Q5_K_M       | [whisper-small-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q5_K_M.gguf) | 194 MB | 3.37% |
| Q4_K_M       | [whisper-small-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q4_K_M.gguf) | 172 MB | 3.40% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 3.432%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  62.20% |
| am       | WER    | 128.30% |
| ar       | WER    |  32.15% |
| as       | WER    | 104.15% |
| az       | WER    |  51.64% |
| be       | WER    |  76.85% |
| bg       | WER    |  40.39% |
| bn       | WER    | 103.04% |
| bs       | WER    |  42.35% |
| ca       | WER    |  14.19% |
| cs       | WER    |  40.57% |
| cy       | WER    |  64.47% |
| da       | WER    |  35.52% |
| de       | WER    |   9.86% |
| el       | WER    |  33.98% |
| en       | WER    |   6.51% |
| es       | WER    |   5.92% |
| et       | WER    |  54.79% |
| fa       | WER    |  58.44% |
| fi       | WER    |  26.48% |
| fil      | WER    |  28.52% |
| fr       | WER    |  13.30% |
| gl       | WER    |  32.72% |
| gu       | WER    | 104.02% |
| ha       | WER    |  94.21% |
| he       | WER    |  46.06% |
| hi       | WER    |  42.05% |
| hr       | WER    |  36.05% |
| hu       | WER    |  42.39% |
| hy       | WER    |  87.84% |
| id       | WER    |  18.02% |
| is       | WER    |  74.54% |
| it       | WER    |   7.97% |
| ja       | CER    |  12.81% |
| ka       | WER    | 130.64% |
| kk       | WER    |  73.54% |
| km       | CER    | 116.96% |
| kn       | WER    |  99.65% |
| ko       | CER    |   7.70% |
| lb       | WER    | 110.02% |
| ln       | WER    |  98.65% |
| lo       | CER    | 101.52% |
| lt       | WER    |  70.21% |
| lv       | WER    |  57.42% |
| mi       | WER    |  62.82% |
| mk       | WER    |  41.53% |
| ml       | WER    | 100.35% |
| mn       | WER    | 142.37% |
| mr       | WER    |  63.66% |
| ms       | WER    |  21.48% |
| mt       | WER    |  97.32% |
| my       | CER    | 132.91% |
| nb       | WER    |  25.53% |
| ne       | WER    |  70.48% |
| nl       | WER    |  18.48% |
| oc       | WER    |  90.43% |
| pa       | WER    | 101.31% |
| pl       | WER    |  16.82% |
| ps       | WER    |  93.61% |
| pt       | WER    |   7.65% |
| ro       | WER    |  33.88% |
| ru       | WER    |  11.90% |
| sd       | WER    | 112.24% |
| sk       | WER    |  36.05% |
| sl       | WER    |  52.70% |
| sn       | WER    | 132.90% |
| so       | WER    | 103.39% |
| sr       | WER    |  44.91% |
| sv       | WER    |  23.10% |
| sw       | WER    |  76.22% |
| ta       | WER    |  35.35% |
| te       | WER    | 102.23% |
| tg       | WER    |  86.78% |
| th       | CER    |  22.55% |
| tr       | WER    |  15.95% |
| uk       | WER    |  20.42% |
| ur       | WER    |  39.75% |
| uz       | WER    | 114.92% |
| vi       | WER    |  22.47% |
| yo       | WER    | 118.50% |
| zh       | CER    |  23.06% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-small/whisper-small-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |               Q8_0 |             Q4_K_M |
| ------- | ------------ | -----------------: | -----------------: |
| Metal   | jfk (11.0s)  |  113.1 ms (97.2×)† |  113.5 ms (96.9×)† |
| Metal   | dots (35.3s) | 349.3 ms (101.2×)† | 340.0 ms (103.9×)† |
| CPU     | jfk (11.0s)  |     1.43 s (7.71×) |     1.30 s (8.44×) |
| CPU     | dots (35.3s) |    3.01 s (11.75×) |    2.74 s (12.91×) |

Apple M4 Max: transcribe.cpp `4d2270e` on 2026-04-28. † published before provenance was recorded; not yet re-measured.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-small
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 1.02 s (10.82×) | 1.02 s (10.76×) |
| Vulkan  | dots (35.3s) | 2.81 s (12.57×) | 2.74 s (12.91×) |
| CPU     | jfk (11.0s)  |  2.17 s (5.07×) |  2.11 s (5.22×) |
| CPU     | dots (35.3s) |  5.31 s (6.65×) |  5.16 s (6.84×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-small
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-small.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-small.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-small.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-small.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-small` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `5.454e-06` |   `4.815e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `2.146e-05` |   `2.704e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `2.134e-05` |   `6.858e-07` | first encoder block |
| `enc.block.11.out`     |  `1.669e-02` |   `6.139e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `2.100e-03` |   `3.882e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `4.768e-06` |   `2.997e-07` | first decoder block, prompt pass |
| `dec.block.11.out`     |  `3.662e-04` |   `2.471e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.755e-04` |   `7.318e-06` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `3.719e-05` |   `7.083e-06` | vocab projection (raw logits) |
| `dec.logits`           |  `8.965e-05` |   `1.926e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `2.289e-05` |   `8.815e-06` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-small \
  --revision 973afd2
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-small/whisper-small-F32.gguf \
  models/whisper-small/whisper-small-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-small
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-small/whisper-small-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
