# Granite Speech 4.1-2b

<!-- catalog:intro -->
Upstream: [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) at [`8f4bb5f`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/commit/8f4bb5f).

Offline multilingual speech-to-text. IBM Granite Speech 4.1-2b is an
audio-LLM with the same architecture as 4.0-1b (Conformer encoder with
block-local Shaw attention, BLIP-2 Q-Former projector, Granite-4.0-1b-base
autoregressive LLM decoder) and improved punctuation and casing over 4.0-1b.
Takes a 16 kHz mono WAV and produces a transcript. Transcribes English,
French, German, Spanish, Portuguese, and Japanese. Translates between
English and each of those five other languages in either direction
(en ↔ fr, en ↔ de, en ↔ es, en ↔ pt, en ↔ ja) — always via English, no
direct fr↔de etc.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text covering English plus French, German,
Spanish, Portuguese, and Japanese. Takes a 16 kHz mono WAV and produces a
transcript.

Translation pairs: English ↔ French, English ↔ German, English ↔ Spanish,
English ↔ Portuguese, English ↔ Japanese, plus English-to-Italian and
English-to-Mandarin. Always via English — there is no direct fr↔de, fr↔es,
etc. Pass the target language as a BCP-47 code via `--translate
--target-language <code>`; the source language is inferred from the audio.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`8f4bb5f`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/commit/8f4bb5f), pinned 2026-05-17. Validated against the Transformers reference at transcribe.cpp commit [`275332d`](https://github.com/handy-computer/transcribe.cpp/tree/275332d) on 2026-05-17.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [granite-speech-4.1-2b-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-BF16.gguf) | 4.63 GB | 1.31% |
| F16          | [granite-speech-4.1-2b-F16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-F16.gguf) | 4.63 GB | 1.32% |
| Q8_0         | [granite-speech-4.1-2b-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q8_0.gguf) | 2.56 GB | 1.32% |
| Q6_K         | [granite-speech-4.1-2b-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q6_K.gguf) | 2.02 GB | 1.29% |
| Q5_K_M       | [granite-speech-4.1-2b-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q5_K_M.gguf) | 1.83 GB | 1.33% |
| Q4_K_M       | [granite-speech-4.1-2b-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q4_K_M.gguf) | 1.60 GB | 1.37% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding. BF16 reference baseline (re-run locally with the model card's exact
prompt): 1.31% — 0.02pp below upstream's published 1.33%, likely a minor
normalization difference on the publisher side and well within bootstrap CI overlap.
Text normalizer: Whisper `EnglishTextNormalizer`, the same normalizer Open ASR
Leaderboard uses.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 6.25% |
| en       | WER    | 4.14% |
| es       | WER    | 5.48% |
| fr       | WER    | 7.61% |
| ja       | CER    | 6.30% |
| pt       | WER    | 9.80% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

Translation:

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-Q8_0.gguf \
  --translate --target-language de \
  samples/jfk.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 145 ms (75.92×) | 139 ms (79.11×) |
| Metal   | dots (35.3s) | 445 ms (79.32×) | 458 ms (77.22×) |
| CPU     | jfk (11.0s)  |  1.33 s (8.30×) |  1.37 s (8.05×) |
| CPU     | dots (35.3s) |  4.13 s (8.56×) |  4.26 s (8.30×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.56 s (4.29×) |  2.55 s (4.31×) |
| Vulkan  | dots (35.3s) |  7.15 s (4.94×) |  6.80 s (5.20×) |
| CPU     | jfk (11.0s)  |  3.86 s (2.85×) |  3.89 s (2.83×) |
| CPU     | dots (35.3s) | 13.06 s (2.70×) | 12.59 s (2.81×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt/ja) | Yes    |
| Translate (en↔ASR, en→it/zh) | Yes (`--translate --target-language <bcp47>`) |
| Word-level timestamps       | No (use the `-plus` variant) |
| Keyword biasing             | No (upstream supports via prompt; not exposed in v1 of transcribe.cpp) |

## Numerical Validation

Tensor-level parity with the transformers reference on `samples/jfk.wav`.
Per-tensor `max_abs` / `mean_abs` budgets in
[`tests/tolerances/granite.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/granite.json).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite \
  scripts/convert-granite.py ibm-granite/granite-speech-4.1-2b \
  --repo-id ibm-granite/granite-speech-4.1-2b
```

### Quantize

```bash
for PRESET in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/granite-speech-4.1-2b/granite-speech-4.1-2b-BF16.gguf \
    models/granite-speech-4.1-2b/granite-speech-4.1-2b-${PRESET}.gguf \
    --quant ${PRESET}
done
```

### Validate

```bash
uv run scripts/validate.py all --family granite --variant granite-speech-4.1-2b
```

### Reproduce WER

```bash
uv run scripts/wer/run.py \
  --model models/granite-speech-4.1-2b/granite-speech-4.1-2b-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/granite-speech-4.1-2b-BF16.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/granite-speech-4.1-2b-BF16.test-clean.jsonl
```
