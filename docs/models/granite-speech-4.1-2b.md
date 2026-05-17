# Granite Speech 4.1-2b

IBM's [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
ported to transcribe.cpp. An audio-LLM with the same architecture as
4.0-1b (Conformer encoder with block-local Shaw attention, BLIP-2 Q-Former
projector, Granite-4.0-1b-base autoregressive LLM decoder) and improved
punctuation/casing over 4.0-1b.

## What it's for

Offline multilingual speech-to-text covering English plus French, German,
Spanish, Portuguese, and Japanese. Takes a 16 kHz mono WAV and produces a
transcript.

Translation pairs: English ↔ French, English ↔ German, English ↔ Spanish,
English ↔ Portuguese, English ↔ Japanese. Always via English — there is no
direct fr↔de, fr↔es, etc. Pass the target language as a BCP-47 code via
`--translate --target-language <code>`; the source language is inferred
from the audio.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`8f4bb5f`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/commit/8f4bb5f31ae98971bd218169f00065a041d20058),
pinned 2026-05-17.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-speech-4.1-2b-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-BF16.gguf)     | 4.63 GB | 1.31% |
| F16    | [granite-speech-4.1-2b-F16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-F16.gguf)       | 4.63 GB | 1.32% |
| Q8_0   | [granite-speech-4.1-2b-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q8_0.gguf)     | 2.56 GB | 1.32% |
| Q6_K   | [granite-speech-4.1-2b-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q6_K.gguf)     | 2.02 GB | 1.29% |
| Q5_K_M | [granite-speech-4.1-2b-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q5_K_M.gguf) | 1.83 GB | 1.33% |
| Q4_K_M | [granite-speech-4.1-2b-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q4_K_M.gguf) | 1.60 GB | 1.37% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with
greedy decoding and the model-card prompt `transcribe the speech with proper
punctuation and capitalization.`. BF16 reference baseline (transformers,
re-run locally with that prompt): 1.31% — 0.02pp below upstream's published
1.33%, within bootstrap CI overlap. Text normalizer: Whisper
`EnglishTextNormalizer`, the same normalizer Open ASR Leaderboard uses.

## Quick start

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

Cells are wall-clock latency on the 11.0 s `samples/jfk.wav` (mean over 5
iterations after 2 warmups), with speedup over realtime in parentheses. Q8_0.

### Apple M4

| Backend | Sample      |       Q8_0        |
| ------- | ----------- | ----------------: |
| Metal   | jfk (11.0s) |    954 ms (12×)   |
| CPU     | jfk (11.0s) |   2.45 s (4×)     |

macOS 26.1, transcribe.cpp `275332d`.

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt/ja) | Yes    |
| Translate (X→En, En→X)      | Yes (`--translate --target-language <bcp47>`) |
| Word-level timestamps       | No (use the `-plus` variant) |
| Keyword biasing             | No (upstream supports via prompt; not exposed in v1 of transcribe.cpp) |

## Numerical validation

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
