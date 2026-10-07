# Cohere Transcribe 03-2026

<!-- catalog:intro -->
Upstream: [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) at [`76b8b23`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/commit/76b8b23).

Offline multilingual speech-to-text covering 14 languages (English, French,
German, Spanish, Italian, Portuguese, Dutch, Polish, Greek, Arabic, Japanese,
Chinese, Vietnamese, Korean). A Conformer encoder with a Transformer
encoder-decoder head (cross-attention, tied token embedding). Takes a 16 kHz
mono WAV and produces a transcript. Decoding is autoregressive.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text covering 14 languages: English, French,
German, Spanish, Italian, Portuguese, Dutch, Polish, Greek, Arabic, Japanese,
Chinese, Vietnamese, Korean. The model takes a 16 kHz mono WAV and produces a
transcript. Cohere uses an encoder-decoder with cross-attention, decoding is
autoregressive.

See Cohere's [model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`76b8b23`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/commit/76b8b23), pinned 2026-04-16. Validated against the Transformers reference at transcribe.cpp commit [`bf0d0b7`](https://github.com/handy-computer/transcribe.cpp/tree/bf0d0b7) on 2026-04-18.
<!-- /catalog -->

## Input limits

Accepts up to about **6.7 minutes (400 s)** of 16 kHz mono audio per call — the
encoder's positional table is the binding limit. Longer audio is rejected up
front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated; split
it into shorter segments. See the [input-length contract](../input-limits.md).

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [cohere-transcribe-03-2026-BF16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-BF16.gguf) | 4.11 GB | 1.26% |
| F16          | [cohere-transcribe-03-2026-F16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-F16.gguf) | 4.11 GB | 1.26% |
| Q8_0         | [cohere-transcribe-03-2026-Q8_0.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q8_0.gguf) | 2.41 GB | 1.27% |
| Q6_K         | [cohere-transcribe-03-2026-Q6_K.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q6_K.gguf) | 1.97 GB | 1.27% |
| Q5_K_M       | [cohere-transcribe-03-2026-Q5_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q5_K_M.gguf) | 1.77 GB | 1.25% |
| Q4_K_M       | [cohere-transcribe-03-2026-Q4_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q4_K_M.gguf) | 1.56 GB | 1.24% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch sizes 1 and 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM. BF16 reference baseline: 1.26%. Cohere's
self-reported number on the same split is 1.25% (Open ASR Leaderboard, as of
2026-03-26). Both ours and Cohere's numbers use the Whisper EnglishTextNormalizer,
so the comparison is apples-to-apples and our port matches the upstream reference
within rounding.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| ar       | WER    | 13.60% |
| de       | WER    |  5.06% |
| el       | WER    |  8.96% |
| en       | WER    |  5.08% |
| es       | WER    |  3.97% |
| fr       | WER    |  5.23% |
| it       | WER    |  3.24% |
| ja       | CER    |  5.13% |
| ko       | CER    |  6.57% |
| nl       | WER    |  7.16% |
| pl       | WER    |  6.15% |
| pt       | WER    |  5.18% |
| vi       | WER    |  7.39% |
| zh       | CER    | 11.18% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-F16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 144 ms (76.51×) | 144 ms (76.32×) |
| Metal   | dots (35.3s) | 470 ms (75.14×) | 492 ms (71.76×) |
| CPU     | jfk (11.0s)  | 926 ms (11.87×) | 1.00 s (10.95×) |
| CPU     | dots (35.3s) | 3.25 s (10.86×) |  3.54 s (9.99×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |           Q8_0 |         Q4_K_M |
| ------- | ------------ | -------------: | -------------: |
| Vulkan  | jfk (11.0s)  | 1.43 s (7.67×) | 1.41 s (7.81×) |
| Vulkan  | dots (35.3s) | 4.15 s (8.52×) | 4.00 s (8.83×) |
| CPU     | jfk (11.0s)  | 2.40 s (4.59×) | 2.47 s (4.46×) |
| CPU     | dots (35.3s) | 8.76 s (4.03×) | 8.93 s (3.96×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models cohere-transcribe-03-2026
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the Transformers
reference implementation on `samples/jfk.wav`. All 22 checkpointed tensors
fall within family tolerance, and the final transcript matches the reference
verbatim.

| Field | Value |
| --- | --- |
| Reference | Transformers, `CohereLabs/cohere-transcribe-03-2026` |
| Dump script | `scripts/dump_reference_cohere_transformers.py` |
| Manifest | `tests/golden/cohere/cohere-transcribe-03-2026.manifest.json` |
| Command | `uv run scripts/validate.py compare --family cohere` |

The expected divergence is in the frontend: C++ runs the STFT in fp64 where
the reference runs fp32. The gap enters at the mel spectrogram, propagates
through the encoder, and attenuates to a few tenths by the final encoder
block. Decoder numerics track the reference within fp32 round-off on the
first autoregressive step.

## Reproduction

### Convert

Downloads the upstream HF repo via `huggingface-cli` (or an existing local
clone) and converts with the family-specific script. Output path is derived
from the repo id.

```bash
uv run --project scripts/envs/cohere \
  scripts/convert-cohere.py CohereLabs/cohere-transcribe-03-2026
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf \
  models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family cohere
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_COHERE_GGUF=models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf \
  ctest --test-dir build --output-on-failure -R 'cohere'
```
