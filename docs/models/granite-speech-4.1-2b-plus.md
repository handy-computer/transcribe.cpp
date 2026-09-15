# Granite Speech 4.1-2b-plus

<!-- catalog:intro -->
Upstream: [`ibm-granite/granite-speech-4.1-2b-plus`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) at [`edd3bf5`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/commit/edd3bf5).

Offline multilingual speech-to-text with word-level timestamps. IBM Granite
Speech 4.1-2b-plus is the timestamp-and-diarization variant of the
Granite-Speech family. Same architecture as the base 4.1-2b (Conformer
encoder, BLIP-2 Q-Former projector, Granite-4.0-1b autoregressive LLM
decoder) with two changes: the encoder concatenates mid-layer (idx 3) and
final-layer hidden states (`cat_hidden_layers=[3]`, doubling the projector
K/V input from 1024 to 2048), and the LM token embeddings are tied with
the lm_head. Takes a 16 kHz mono WAV and produces a transcript, with
`--timestamps word` returning structured per-word timestamps (parsed from the
model's `[T:N]` centisecond markers), or `--diarize` returning structured
speaker-attributed turns from its separate SAA prompt. Those two prompt tasks
cannot be combined. Transcribes English,
French, German, Spanish, and Portuguese (no Japanese on this variant).
This variant is transcription-only: unlike the base granite-speech-4.1-2b,
it does not perform speech translation.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text with word-level timestamps. Covers
English plus French, German, Spanish, and Portuguese (no Japanese on this
variant). Takes a 16 kHz mono WAV and produces a transcript; with
`--timestamps word` it returns per-word start/end times. Internally the model
emits `[T:N]` end-of-word centisecond markers; the runtime parses them into
structured word timestamps and returns a clean transcript.

This variant is transcription-only. Unlike the base
[`granite-speech-4.1-2b`](granite-speech-4.1-2b.md), it does not perform
speech translation.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`edd3bf5`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/commit/edd3bf5), pinned 2026-05-17. Validated against the Transformers reference at transcribe.cpp commit [`275332d`](https://github.com/handy-computer/transcribe.cpp/tree/275332d) on 2026-05-17.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [granite-speech-4.1-2b-plus-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-BF16.gguf) | 4.23 GB | 1.49% |
| F16          | [granite-speech-4.1-2b-plus-F16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-F16.gguf) | 4.23 GB | 1.48% |
| Q8_0         | [granite-speech-4.1-2b-plus-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q8_0.gguf) | 2.35 GB | 1.50% |
| Q6_K         | [granite-speech-4.1-2b-plus-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q6_K.gguf) | 1.86 GB | 1.46% |
| Q5_K_M       | [granite-speech-4.1-2b-plus-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q5_K_M.gguf) | 1.69 GB | 1.48% |
| Q4_K_M       | [granite-speech-4.1-2b-plus-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q4_K_M.gguf) | 1.49 GB | 1.56% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding with the model-card chat template (system prompt + leading-space
user instruction + `add_generation_prompt=True`). BF16 reference baseline (re-run
locally with that exact prompt): 1.48%; 0.04pp above upstream's published 1.44%,
within bootstrap CI overlap and likely a chat-template / normalization difference on
the publisher side. Text normalizer: Whisper `EnglishTextNormalizer`, the same
normalizer Open ASR Leaderboard uses. The `add_generation_prompt=True` is
load-bearing — without it the model emits 25-27 empty hypotheses on short test-clean
clips and WER blows up to ~26%. The transcribe.cpp runtime hard-codes the prompt
correctly; this note only matters if you reproduce the reference.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| de       | WER    |  8.06% |
| en       | WER    |  4.46% |
| es       | WER    |  6.53% |
| fr       | WER    |  8.82% |
| pt       | WER    | 10.61% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

Word-level timestamps:

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-Q8_0.gguf \
  --timestamps word \
  samples/jfk.wav
```

The runtime returns a clean transcript plus structured per-word start/end
times. (Internally the model emits `[T:N]` markers giving each word's end time
in centiseconds, modulo 1000 with a 10 s rollover that the runtime unwraps; the
markers are stripped from the returned text.)

```
text: and so my fellow americans ask not what your country can do for you ask what you can do for your country
words: 22
  [   0.30 ->    0.57] and
  [   0.57 ->    0.95] so
  [   0.95 ->    1.25] my
  [   1.25 ->    1.60] fellow
  [   1.60 ->    2.13] americans
  ...
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |          Q4_K_M |
| ------- | ------------ | ---------------: | --------------: |
| Metal   | jfk (11.0s)  |  134 ms (82.33×) | 137 ms (80.36×) |
| Metal   | dots (35.3s) | 350 ms (100.92×) | 353 ms (99.98×) |
| CPU     | jfk (11.0s)  |   1.47 s (7.46×) |  1.50 s (7.33×) |
| CPU     | dots (35.3s) |   3.85 s (9.18×) |  4.14 s (8.54×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.58 s (4.27×) |  2.58 s (4.26×) |
| Vulkan  | dots (35.3s) |  7.04 s (5.02×) |  7.53 s (4.69×) |
| CPU     | jfk (11.0s)  |  4.32 s (2.55×) |  4.34 s (2.53×) |
| CPU     | dots (35.3s) | 13.35 s (2.65×) | 13.10 s (2.70×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt)    | Yes    |
| Translate                   | No (ASR-only variant; use the base [granite-speech-4.1-2b](granite-speech-4.1-2b.md) for translation) |
| Word-level timestamps       | Yes (`--timestamps word`, structured per-word t0/t1 parsed from the model's `[T:N]` markers) |
| Speaker diarization         | Yes (`--diarize`; structured per-turn `speaker_id`, no timing) |

Speaker attribution is off by default. `--diarize` selects Granite's distinct
speaker-attribution prompt and parses `[Speaker N]:` markers into segment and
speaker-turn rows. This task carries no timestamps: `--timestamps none` and
`auto` are accepted, while explicit `segment`, `word`, or `token` requests are
rejected instead of silently downgraded. The speaker-attribution and word-
timestamp prompts cannot be combined.

## Numerical Validation

Tensor-level parity with the transformers reference on `samples/jfk.wav`.
Per-tensor `max_abs` / `mean_abs` budgets in
[`tests/tolerances/granite.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/granite.json).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite \
  scripts/convert-granite.py ibm-granite/granite-speech-4.1-2b-plus \
  --repo-id ibm-granite/granite-speech-4.1-2b-plus
```

### Quantize

```bash
for PRESET in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-BF16.gguf \
    models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-${PRESET}.gguf \
    --quant ${PRESET}
done
```

### Validate

```bash
uv run scripts/validate.py all --family granite --variant granite-speech-4.1-2b-plus
```

### Reproduce WER

```bash
uv run scripts/wer/run.py \
  --model models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/granite-speech-4.1-2b-plus-BF16.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/granite-speech-4.1-2b-plus-BF16.test-clean.jsonl
```
