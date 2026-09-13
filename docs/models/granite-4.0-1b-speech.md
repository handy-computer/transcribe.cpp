# Granite Speech 4.0-1b

<!-- catalog:intro -->
Upstream: [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) at [`bd87ab8`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech/commit/bd87ab8).

Offline multilingual speech-to-text. IBM Granite Speech 4.0-1b is an
audio-LLM: a Conformer encoder with block-local Shaw attention, a BLIP-2
Q-Former projector, and the Granite-4.0-1b-base LLM as an autoregressive
decoder. Takes a 16 kHz mono WAV and produces a transcript; the LLM half is
what writes the text. Transcribes English, French, German, Spanish,
Portuguese, and Japanese. Translates between English and each of those
five other languages in either direction (en ↔ fr, en ↔ de, en ↔ es,
en ↔ pt, en ↔ ja) — always via English, no direct fr↔de etc.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text covering English plus French, German,
Spanish, Portuguese, and Japanese. The model takes a 16 kHz mono WAV and
produces a transcript.

Translation pairs: English ↔ French, English ↔ German, English ↔ Spanish,
English ↔ Portuguese, English ↔ Japanese, plus English-to-Italian and
English-to-Mandarin. Always via English — there is no direct fr↔de, fr↔es,
etc. Pass the target language as a BCP-47 code via `--translate
--target-language <code>`; the source language is inferred from the audio.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`bd87ab8`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech/commit/bd87ab862416353633ea431fe49b1614003623c5),
pinned 2026-05-17.

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [granite-4.0-1b-speech-BF16.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-BF16.gguf) | 4.63 GB | 1.42% |
| F16          | [granite-4.0-1b-speech-F16.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-F16.gguf) | 4.63 GB | 1.42% |
| Q8_0         | [granite-4.0-1b-speech-Q8_0.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q8_0.gguf) | 2.56 GB | 1.44% |
| Q6_K         | [granite-4.0-1b-speech-Q6_K.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q6_K.gguf) | 2.02 GB | 1.41% |
| Q5_K_M       | [granite-4.0-1b-speech-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q5_K_M.gguf) | 1.83 GB | 1.42% |
| Q4_K_M       | [granite-4.0-1b-speech-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-4.0-1b-speech-gguf/resolve/main/granite-4.0-1b-speech-Q4_K_M.gguf) | 1.60 GB | 1.48% |
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding. BF16 reference baseline (re-run locally with the
model card's exact prompt): 1.42% — matches the upstream Open ASR
Leaderboard number exactly. Text normalizer: Whisper
`EnglishTextNormalizer`, the same normalizer Open ASR Leaderboard uses.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 7.29% |
| en       | WER    | 4.66% |
| es       | WER    | 5.91% |
| fr       | WER    | 8.44% |
| ja       | CER    | 6.44% |
| pt       | WER    | 9.60% |
<!-- /catalog -->

## Quick Start

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

Cells are compute latency (mel + encode + decode), with speedup over realtime in parentheses.

### Apple M4 Max

Mean over 3 iterations after 1 warmup.

<!-- catalog:perf machine=m4-max -->
| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  126 ms (86.92×) |  129 ms (85.41×) |
| Metal   | dots (35.3s) | 341 ms (103.69×) | 347 ms (101.77×) |
| CPU     | jfk (11.0s)  |   1.55 s (7.08×) |   1.30 s (8.47×) |
| CPU     | dots (35.3s) |   4.81 s (7.34×) |   4.25 s (8.31×) |
<!-- /catalog -->

macOS 26.4, transcribe.cpp `de05c43`.

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

Mean over 3 iterations after 1 warmup.

<!-- catalog:perf machine=ryzen-4750u -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.43 s (4.52×) |  2.45 s (4.49×) |
| Vulkan  | dots (35.3s) |  6.50 s (5.43×) |  6.61 s (5.35×) |
| CPU     | jfk (11.0s)  |  5.27 s (2.09×) |  4.30 s (2.56×) |
| CPU     | dots (35.3s) | 17.39 s (2.03×) | 13.72 s (2.58×) |
<!-- /catalog -->

Linux 6.18 (Fedora 43), transcribe.cpp `dbe5814`.

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt/ja) | Yes    |
| Translate (en↔ASR, en→it/zh) | Yes (`--translate --target-language <bcp47>`) |
| Word-level timestamps       | No (use the `-plus` variant) |
| Speaker diarization         | No (upstream supports via prompt; not exposed in v1 of transcribe.cpp) |

## Numerical Validation

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
