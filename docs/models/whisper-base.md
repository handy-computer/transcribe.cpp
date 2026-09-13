# Whisper base

<!-- catalog:intro -->
Upstream: [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) at [`e37978b`](https://huggingface.co/openai/whisper-base/commit/e37978b).

OpenAI Whisper base — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-base) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`e37978b`](https://huggingface.co/openai/whisper-base/commit/e37978b),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

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

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 5.009%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.
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

Cells are compute latency (mel + encode + decode, mean over the recorded
iterations after warmup), with speedup over realtime in parentheses. Units:
`ms` below 1 s, `s` above (2 decimal places). Decode latency dominates as
model size grows; the encoder is only run once per 30-second window.

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  |    52.1 ms (211×) |  53.6 ms (205.2×) |
| Metal   | dots (35.3s) | 170.0 ms (207.8×) | 168.3 ms (209.9×) |
| CPU     | jfk (11.0s)  | 373.9 ms (29.42×) | 347.6 ms (31.65×) |
| CPU     | dots (35.3s) | 806.1 ms (43.83×) | 750.3 ms (47.09×) |
<!-- /catalog -->

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-base \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-base-publication
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  351 ms (31.3×) |  356 ms (30.9×) |
| Vulkan  | dots (35.3s) |  922 ms (38.3×) |  946 ms (37.4×) |
| CPU     | jfk (11.0s)  |  1.11 s (9.95×) | 913 ms (12.05×) |
| CPU     | dots (35.3s) | 2.54 s (13.92×) | 2.27 s (15.53×) |
<!-- /catalog -->

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-base \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-base-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-base.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-base.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-base.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-base.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-base` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `5.960e-06` |   `6.348e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.466e-05` |   `2.544e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `1.794e-05` |   `6.097e-07` | first encoder block |
| `enc.block.5.out`      |  `1.513e-02` |   `5.212e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.699e-03` |   `2.588e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `5.603e-06` |   `2.874e-07` | first decoder block, prompt pass |
| `dec.block.5.out`      |  `1.373e-04` |   `1.643e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.526e-04` |   `9.441e-06` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `6.890e-05` |   `1.462e-05` | vocab projection (raw logits) |
| `dec.logits`           |  `4.768e-05` |   `9.442e-06` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `8.774e-05` |   `5.206e-05` | step-20 logits (KV-cached path) |

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
