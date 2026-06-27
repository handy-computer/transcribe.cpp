# Granite Speech 4.1-2b-plus

IBM's [`ibm-granite/granite-speech-4.1-2b-plus`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
ported to transcribe.cpp. The timestamp-and-diarization variant of the
Granite-Speech family. Same architecture as the base 4.1-2b (Conformer
encoder, BLIP-2 Q-Former projector, Granite-4.0-1b autoregressive LLM
decoder) with two changes: the encoder concatenates mid-layer (idx 3) and
final-layer hidden states (doubling the projector K/V input from 1024 to
2048), and the LM token embeddings are tied with the lm_head.

## What it's for

Offline multilingual speech-to-text with word-level timestamps. Covers
English plus French, German, Spanish, and Portuguese (no Japanese on this
variant). Takes a 16 kHz mono WAV and produces a transcript, optionally
interleaved with `[SS:N]` word-timestamp markers (centiseconds since
segment start).

Translation pairs: English ↔ French, English ↔ German, English ↔ Spanish,
English ↔ Portuguese. Always via English — there is no direct fr↔de,
fr↔es, etc. Pass the target language as a BCP-47 code via
`--translate --target-language <code>`; the source language is inferred
from the audio.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`edd3bf5`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/commit/edd3bf54fbb06d8e263aa0c1939321d67b073f86),
pinned 2026-05-17.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-speech-4.1-2b-plus-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-BF16.gguf)     | 4.23 GB | 1.49% |
| F16    | [granite-speech-4.1-2b-plus-F16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-F16.gguf)       | 4.23 GB | 1.48% |
| Q8_0   | [granite-speech-4.1-2b-plus-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q8_0.gguf)     | 2.35 GB | 1.50% |
| Q6_K   | [granite-speech-4.1-2b-plus-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q6_K.gguf)     | 1.86 GB | 1.46% |
| Q5_K_M | [granite-speech-4.1-2b-plus-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q5_K_M.gguf) | 1.69 GB | 1.48% |
| Q4_K_M | [granite-speech-4.1-2b-plus-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-plus-gguf/resolve/main/granite-speech-4.1-2b-plus-Q4_K_M.gguf) | 1.49 GB | 1.56% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with
greedy decoding and the model-card chat template (system prompt + leading-
space user instruction + `add_generation_prompt=True`). BF16 reference
baseline (transformers, re-run locally with that exact prompt): 1.48%;
0.04pp above upstream's published 1.44%, within bootstrap CI overlap and
likely a chat-template / normalization difference on the publisher side.
Text normalizer: Whisper `EnglishTextNormalizer`. The transcribe.cpp runtime
hard-codes the correct chat template; the WER quoted here is what the C++
runtime actually scores.

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

Output includes `[SS:N]` markers (centiseconds since segment start)
interleaved with words:

```
[SS:23] And [SS:52] so [SS:92] my [SS:121] fellow [SS:154] Americans ...
```

Translation:

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-Q8_0.gguf \
  --translate --target-language de \
  samples/jfk.wav
```

## Performance

Cells are wall-clock latency, with speedup over realtime in parentheses.

### Apple M4 Max

Mean over 3 iterations after 1 warmup.

**Metal**

| Sample       |     Q4_K_M       |       Q8_0       |
| ------------ | ---------------: | ---------------: |
| jfk (11.0s)  |    280 ms (39×)  |    308 ms (36×)  |
| dots (35.3s) |   1.02 s (34×)   |   1.18 s (30×)   |

**CPU**

| Sample       |     Q4_K_M       |       Q8_0       |
| ------------ | ---------------: | ---------------: |
| jfk (11.0s)  |   1.87 s (5.9×)  |   2.04 s (5.4×)  |
| dots (35.3s) |   5.71 s (6.2×)  |   6.91 s (5.1×)  |

macOS 26.4, transcribe.cpp `de05c43`.

### Apple M4

Mean over 5 iterations after 2 warmups. Q8_0.

| Backend | Sample      |       Q8_0        |
| ------- | ----------- | ----------------: |
| Metal   | jfk (11.0s) |   1.00 s (11×)    |
| CPU     | jfk (11.0s) |   2.44 s (5×)     |

macOS 26.1, transcribe.cpp `275332d`.

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

Mean over 3 iterations after 1 warmup.

**Vulkan (RADV)**

| Sample       |     Q4_K_M       |       Q8_0       |
| ------------ | ---------------: | ---------------: |
| jfk (11.0s)  |   3.63 s (3.0×)  |   3.85 s (2.9×)  |
| dots (35.3s) |  12.29 s (2.9×)  |  13.42 s (2.6×)  |

**CPU**

| Sample       |     Q4_K_M       |       Q8_0       |
| ------------ | ---------------: | ---------------: |
| jfk (11.0s)  |   6.08 s (1.8×)  |   7.80 s (1.4×)  |
| dots (35.3s) |  20.45 s (1.7×)  |  26.23 s (1.3×)  |

Linux 6.18 (Fedora 43), transcribe.cpp `dbe5814`.

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt)    | Yes    |
| Translate (En→X)            | Yes (`--translate --target-language <bcp47>`) |
| Word-level timestamps       | Yes (`--timestamps word`, `[SS:N]` markers) |
| Speaker diarization         | No (upstream supports via prompt; not exposed in v1 of transcribe.cpp) |

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
