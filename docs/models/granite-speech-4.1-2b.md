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

Licensed Apache-2.0. Ported from upstream commit
[`8f4bb5f`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/commit/8f4bb5f31ae98971bd218169f00065a041d20058),
pinned 2026-05-17.

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

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding. BF16 reference baseline (re-run locally with the
model card's exact prompt): 1.31% — 0.02pp below upstream's published
1.33%, likely a minor normalization difference on the publisher side and
well within bootstrap CI overlap. Text normalizer: Whisper
`EnglishTextNormalizer`, the same normalizer Open ASR Leaderboard uses.
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
Compute latency (mel + encode + decode), speedup over realtime in parentheses; mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  127 ms (86.64×) |  130 ms (84.90×) |
| Metal   | dots (35.3s) | 343 ms (103.09×) | 349 ms (101.11×) |
| CPU     | jfk (11.0s)  |   1.58 s (6.97×) |   1.45 s (7.58×) |
| CPU     | dots (35.3s) |   4.89 s (7.22×) |   4.44 s (7.95×) |

Apple M4 Max: transcribe.cpp `de05c43` on 2026-05-21.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.41 s (4.56×) |  2.44 s (4.51×) |
| Vulkan  | dots (35.3s) |  6.52 s (5.42×) |  6.61 s (5.34×) |
| CPU     | jfk (11.0s)  |  5.55 s (1.98×) |  4.55 s (2.42×) |
| CPU     | dots (35.3s) | 17.56 s (2.01×) | 14.51 s (2.43×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `dbe5814` on 2026-05-18.
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
