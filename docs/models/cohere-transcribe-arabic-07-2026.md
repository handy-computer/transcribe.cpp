# Cohere Transcribe Arabic 07-2026

<!-- catalog:intro -->
Upstream: [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) at [`0a8193c`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026/commit/0a8193c).

Offline Arabic speech-to-text, including dialectal Arabic and
Arabic-English code-switching, with English as a secondary language. An
Arabic-focused adaptation of the Cohere Transcribe 03-2026 architecture:
a Conformer encoder with a Transformer encoder-decoder head
(cross-attention, tied token embedding). Takes a 16 kHz mono WAV and a
language flag (`-l ar` or `-l en`) and produces a transcript. Decoding
is autoregressive.
<!-- /catalog -->

## What it's for

Offline Arabic speech-to-text, including dialectal Arabic and
Arabic-English code-switching, with English as a secondary language. The
model takes a 16 kHz mono WAV and produces a transcript; pass the language
(`-l ar` or `-l en`). Decoding is autoregressive.

See Cohere's [model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`0a8193c`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026/commit/0a8193c), pinned 2026-07-07. Validated against the Transformers reference at transcribe.cpp commit [`d89ecb7`](https://github.com/handy-computer/transcribe.cpp/tree/d89ecb7) on 2026-07-07.
<!-- /catalog -->

## Input limits

Accepts up to about **6.7 minutes (400 s)** of 16 kHz mono audio per call — the
encoder's positional table is the binding limit. Longer audio is rejected up
front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated; split
it into shorter segments. See the [input-length contract](../input-limits.md).

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (FLEURS ar) |
| --- | --- | ---: | ---: |
| BF16         | [cohere-transcribe-arabic-07-2026-BF16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-BF16.gguf) | 4.11 GB | 11.02% |
| F16          | [cohere-transcribe-arabic-07-2026-F16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-F16.gguf) | 4.11 GB | 11.00% |
| Q8_0         | [cohere-transcribe-arabic-07-2026-Q8_0.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q8_0.gguf) | 2.41 GB | 11.06% |
| Q6_K         | [cohere-transcribe-arabic-07-2026-Q6_K.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q6_K.gguf) | 1.97 GB | 11.07% |
| Q5_K_M       | [cohere-transcribe-arabic-07-2026-Q5_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q5_K_M.gguf) | 1.77 GB | 10.95% |
| Q4_K_M       | [cohere-transcribe-arabic-07-2026-Q4_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q4_K_M.gguf) | 1.56 GB | 11.18% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full FLEURS ar split (428 utterances), batch size 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM, scored with the Whisper BasicTextNormalizer; the
FLEURS Arabic split is `ar_eg`, Egyptian-dialect speech. BF16 reference baseline,
measured with native Transformers on the same manifest: 11.00%; the BF16 port scores
11.02%, and every quant falls inside the reference's 95% confidence interval. FLEURS
Arabic is Egyptian-dialect speech; upstream numbers published on other Arabic test
sets are not directly comparable.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 | Q5_K_M |
| --- | --- | ---: | ---: |
| en       | WER    | 4.88% |  4.88% |

**LibriSpeech test-clean**

| Language | Metric |  BF16 |   F16 |  Q8_0 |  Q6_K | Q5_K_M | Q4_K_M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en       | WER    | 1.33% | 1.33% | 1.34% | 1.34% |  1.34% |  1.34% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/cohere-transcribe-arabic-07-2026/cohere-transcribe-arabic-07-2026-Q8_0.gguf \
  -l ar \
  input.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

The tables below were measured on
[Cohere Transcribe 03-2026](cohere-transcribe-03-2026.md). This variant is
the same architecture with identical tensor shapes and quantization layout
(only the weight values differ), so per-quant throughput carries over.

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 142 ms (77.53×) | 143 ms (77.15×) |
| Metal   | dots (35.3s) | 469 ms (75.36×) | 460 ms (76.89×) |
| CPU     | jfk (11.0s)  | 912 ms (12.07×) |  1.31 s (8.37×) |
| CPU     | dots (35.3s) | 3.36 s (10.53×) | 3.43 s (10.29×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |           Q8_0 |         Q4_K_M |
| ------- | ------------ | -------------: | -------------: |
| Vulkan  | jfk (11.0s)  | 1.35 s (8.17×) | 1.31 s (8.40×) |
| Vulkan  | dots (35.3s) | 4.19 s (8.44×) | 4.00 s (8.83×) |
| CPU     | jfk (11.0s)  | 2.40 s (4.58×) | 2.46 s (4.48×) |
| CPU     | dots (35.3s) | 8.75 s (4.04×) | 8.94 s (3.95×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction (substitute this variant's slug):

```bash
uv run scripts/bench/run.py \
  --models cohere-transcribe-arabic-07-2026 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name cohere-transcribe-arabic-07-2026-publication
```

## Numerical Validation

The cohere family implementation is validated tensor-by-tensor against the
Transformers reference on the base
[Cohere Transcribe 03-2026](cohere-transcribe-03-2026.md#numerical-validation)
checkpoint (all 22 checkpointed tensors within family tolerance, transcript
verbatim). This variant shares that implementation unchanged — same
architecture, tensor shapes, and blob-identical SentencePiece tokenizer —
and is validated end-to-end: the C++ BF16 port scores 11.02% WER on the
full FLEURS Arabic test split against 11.00% for the native Transformers
reference on the same manifest (within +0.02pp), with per-utterance
hypotheses matching the reference on the upstream sample audio. A
per-variant golden tensor manifest has not been generated.

| Field | Value |
| --- | --- |
| Reference | Transformers, `CohereLabs/cohere-transcribe-arabic-07-2026` |
| Reference WER runner | `scripts/wer/run_reference_cohere_transformers.py` |
| Family tensor manifest | `tests/golden/cohere/cohere-transcribe-03-2026.manifest.json` (base variant) |
| WER reports | `reports/wer/cohere-transcribe-arabic-07-2026-*.fleurs-ar.b8.jsonl` |

## Reproduction

### Convert

Downloads the upstream HF repo via `huggingface-cli` (or an existing local
clone) and converts with the family-specific script. Output path is derived
from the repo id. The upstream repo is gated; accept the license on
Hugging Face first.

```bash
uv run --project scripts/envs/cohere \
  scripts/convert-cohere.py CohereLabs/cohere-transcribe-arabic-07-2026
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/cohere-transcribe-arabic-07-2026/cohere-transcribe-arabic-07-2026-BF16.gguf \
  models/cohere-transcribe-arabic-07-2026/cohere-transcribe-arabic-07-2026-F16.gguf \
  --quant F16
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_COHERE_GGUF=models/cohere-transcribe-arabic-07-2026/cohere-transcribe-arabic-07-2026-BF16.gguf \
  ctest --test-dir build --output-on-failure -R 'cohere'
```

### WER

```bash
uv run scripts/wer/ingest.py fleurs --lang ar

uv run scripts/wer/run.py \
  --model models/cohere-transcribe-arabic-07-2026/cohere-transcribe-arabic-07-2026-BF16.gguf \
  --manifest samples/wer/fleurs-ar.manifest.jsonl \
  --language ar \
  --out reports/wer/cohere-transcribe-arabic-07-2026-BF16.fleurs-ar.jsonl

uv run scripts/wer/score.py \
  reports/wer/cohere-transcribe-arabic-07-2026-BF16.fleurs-ar.jsonl --language ar
```
