# Whisper large-v2

<!-- catalog:intro -->
Upstream: [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2) at [`ae46427`](https://huggingface.co/openai/whisper-large-v2/commit/ae46427).

OpenAI Whisper large-v2 — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v2) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`ae46427`](https://huggingface.co/openai/whisper-large-v2/commit/ae46427),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

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

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.83%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.
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
Compute latency (mel + encode + decode), speedup over realtime in parentheses; mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 493.1 ms (22.31×) | 499.6 ms (22.02×) |
| Metal   | dots (35.3s) |   1.37 s (25.7×)† |   1.40 s (25.22×) |
| CPU     | jfk (11.0s)  |    9.66 s (1.14×) |    7.46 s (1.48×) |
| CPU     | dots (35.3s) |   19.72 s (1.79×) |   15.43 s (2.29×) |

Apple M4 Max: transcribe.cpp `4d2270e` on 2026-04-28. † published before provenance was recorded; not yet re-measured.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v2
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.27 s (1.75×) |  6.35 s (1.73×) |
| Vulkan  | dots (35.3s) | 14.29 s (2.5×)† | 13.68 s (2.6×)† |
| CPU     | jfk (11.0s)  | 25.73 s (0.4×)† | 19.46 s (0.6×)† |
| CPU     | dots (35.3s) | 53.75 s (0.7×)† | 43.11 s (0.82×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `01127e6` on 2026-04-28. † published before provenance was recorded; not yet re-measured.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v2
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v2.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v2.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v2.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v2.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v2` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `3.725e-06` |   `2.701e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `1.800e-05` |   `4.611e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `5.424e-05` |   `1.111e-06` | first encoder block |
| `enc.block.31.out`     |  `1.822e-02` |   `3.259e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.213e-03` |   `2.945e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `4.530e-06` |   `2.545e-07` | first decoder block, prompt pass |
| `dec.block.31.out`     |  `6.104e-05` |   `1.777e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `4.387e-05` |   `2.792e-06` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `3.910e-05` |   `9.921e-06` | vocab projection (raw logits) |
| `dec.logits`           |  `7.486e-05` |   `1.652e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `9.537e-06` |   `2.033e-06` | step-20 logits (KV-cached path) |

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
