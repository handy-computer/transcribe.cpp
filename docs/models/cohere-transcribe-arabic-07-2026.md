# Cohere Transcribe Arabic 07-2026

Cohere's [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
ported to transcribe.cpp. An Arabic-focused adaptation of the
[Cohere Transcribe 03-2026](cohere-transcribe-03-2026.md) architecture: a
Conformer encoder with a Transformer encoder-decoder head (cross-attention,
tied token embedding), retrained for Arabic.

## What it's for

Offline Arabic speech-to-text, including dialectal Arabic and
Arabic-English code-switching, with English as a secondary language. The
model takes a 16 kHz mono WAV and produces a transcript; pass the language
(`-l ar` or `-l en`). Decoding is autoregressive.

See Cohere's [model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`0a8193c`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026/commit/0a8193caa4f3f92131471ab08824e488141cb392),
pinned 2026-07-07.

## Input limits

Accepts up to about **6.7 minutes (400 s)** of 16 kHz mono audio per call — the
encoder's positional table is the binding limit. Longer audio is rejected up
front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated; split
it into shorter segments. See the [input-length contract](../input-limits.md).

## Download

| Quantization | Download | Size | WER (FLEURS Arabic test) |
| --- | --- | ---: | ---: |
| BF16   | [cohere-transcribe-arabic-07-2026-BF16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-BF16.gguf)     | 4.10 GB | 11.02% |
| F16    | [cohere-transcribe-arabic-07-2026-F16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-F16.gguf)       | 4.11 GB | 11.00% |
| Q8_0   | [cohere-transcribe-arabic-07-2026-Q8_0.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q8_0.gguf)     | 2.41 GB | 11.06% |
| Q6_K   | [cohere-transcribe-arabic-07-2026-Q6_K.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q6_K.gguf)     | 1.97 GB | 11.07% |
| Q5_K_M | [cohere-transcribe-arabic-07-2026-Q5_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q5_K_M.gguf) | 1.77 GB | 10.95% |
| Q4_K_M | [cohere-transcribe-arabic-07-2026-Q4_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q4_K_M.gguf) | 1.56 GB | 11.18% |

WER is measured on the full FLEURS Arabic (`ar_eg`) test split (428
utterances) with greedy decoding and no external LM, scored with the Whisper
`BasicTextNormalizer` (the Arabic routing in `scripts/wer/score.py`).
BF16 reference baseline, measured with native Transformers on the same
manifest: 11.00%; our BF16 port scores 11.02%, and every quant falls inside
the reference's 95% confidence interval. Note that FLEURS Arabic is
Egyptian-dialect speech; upstream numbers published on other Arabic test
sets are not directly comparable.

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  150 ms (74×)  |  154 ms (71×)  |
| Metal   | dots (35.3s) |  491 ms (72×)  |  465 ms (76×)  |
| CPU     | jfk (11.0s)  | 1.21 s (9×)    | 1.05 s (11×)   |
| CPU     | dots (35.3s) | 4.13 s (9×)    | 3.49 s (10×)   |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  1.43 s (8×)   |  1.33 s (8×)   |
| Vulkan  | dots (35.3s) |  4.25 s (8×)   |  4.25 s (8×)   |
| CPU     | jfk (11.0s)  |  3.57 s (3×)   |  2.90 s (4×)   |
| CPU     | dots (35.3s) | 12.40 s (3×)   | 10.08 s (4×)   |

Fedora 43, transcribe.cpp `2ab01b8`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

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
