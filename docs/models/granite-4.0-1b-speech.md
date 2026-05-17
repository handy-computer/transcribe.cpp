# Granite Speech 4.0-1b

IBM's [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)
ported to transcribe.cpp. An audio-LLM: a Conformer encoder with block-local
Shaw attention, a BLIP-2 Q-Former projector, and the Granite-4.0-1b-base LLM
as an autoregressive decoder.

## What it's for

Offline multilingual speech-to-text covering English plus French, German,
Spanish, Portuguese, and Japanese. The model takes a 16 kHz mono WAV and
produces a transcript.

Translation pairs: English ↔ French, English ↔ German, English ↔ Spanish,
English ↔ Portuguese, English ↔ Japanese. Always via English — there is no
direct fr↔de, fr↔es, etc. Pass the target language as a BCP-47 code via
`--translate --target-language <code>`; the source language is inferred
from the audio.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`bd87ab8`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech/commit/bd87ab862416353633ea431fe49b1614003623c5),
pinned 2026-05-17.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-4.0-1b-speech-BF16.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-BF16.gguf)     | 4.63 GB | 1.42% |
| F16    | [granite-4.0-1b-speech-F16.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-F16.gguf)       | 4.63 GB | 1.42% |
| Q8_0   | [granite-4.0-1b-speech-Q8_0.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q8_0.gguf)     | 2.56 GB | 1.44% |
| Q6_K   | [granite-4.0-1b-speech-Q6_K.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q6_K.gguf)     | 2.02 GB | 1.41% |
| Q5_K_M | [granite-4.0-1b-speech-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q5_K_M.gguf) | 1.83 GB | 1.42% |
| Q4_K_M | [granite-4.0-1b-speech-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q4_K_M.gguf) | 1.60 GB | 1.48% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with
greedy decoding. The BF16 reference baseline (transformers, re-run locally
with the model-card prompt `USER: <|audio|>can you transcribe the speech
into a written format?\n ASSISTANT:`) is 1.42%, matching IBM's published
Open ASR Leaderboard number exactly. Text normalizer: Whisper
`EnglishTextNormalizer`, the same normalizer Open ASR Leaderboard uses.

## Quick start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

Translation (granite uses one chat template, parameterized by target language):

```bash
build/bin/transcribe-cli \
  -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-Q8_0.gguf \
  --translate --target-language de \
  samples/jfk.wav
```

## Performance

Cells are wall-clock latency on the 11.0 s `samples/jfk.wav` (mean over 5
iterations after 2 warmups), with speedup over realtime in parentheses. Q8_0.

### Apple M4

| Backend | Sample      |       Q8_0        |
| ------- | ----------- | ----------------: |
| Metal   | jfk (11.0s) |    959 ms (11×)   |
| CPU     | jfk (11.0s) |   2.44 s (5×)     |

macOS 26.1, transcribe.cpp `275332d`.

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt/ja) | Yes    |
| Translate (X→En, En→X)      | Yes (`--translate --target-language <bcp47>`) |
| Word-level timestamps       | No (use the `-plus` variant) |
| Speaker diarization         | No (upstream supports via prompt; not exposed in v1 of transcribe.cpp) |

## Numerical validation

Tensor-level parity with the transformers reference on `samples/jfk.wav`.
Per-tensor `max_abs` / `mean_abs` budgets in
[`tests/tolerances/granite.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/granite.json).
Drift is dominated by BF16 reduction-order noise in the 40-layer LLM stack
plus a localized band at the last Shaw block-local attention window
boundary in `enc.block.15.out`. No structural deltas vs the reference.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite \
  scripts/convert-granite.py ibm-granite/granite-4.0-1b-speech \
  --repo-id ibm-granite/granite-4.0-1b-speech
```

### Quantize

```bash
for PRESET in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/granite-4.0-1b-speech/granite-4.0-1b-speech-BF16.gguf \
    models/granite-4.0-1b-speech/granite-4.0-1b-speech-${PRESET}.gguf \
    --quant ${PRESET}
done
```

### Validate

```bash
uv run scripts/validate.py all --family granite --variant granite-4.0-1b-speech
```

### Reproduce WER

```bash
uv run scripts/wer/run.py \
  --model models/granite-4.0-1b-speech/granite-4.0-1b-speech-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/granite-4.0-1b-speech-BF16.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/granite-4.0-1b-speech-BF16.test-clean.jsonl
```
