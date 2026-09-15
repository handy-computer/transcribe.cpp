# Qwen3-ASR 0.6B

<!-- catalog:intro -->
Upstream: [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) at [`5eb1441`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/commit/5eb1441).

Offline multilingual speech-to-text. An 18-layer bidirectional audio encoder
feeds a 28-layer Qwen3 causal LM with audio-token injection (fused
audio+text sequence, no cross-attention). Auto-detects the audio's language
across 30 languages and emits the transcript in that language. Takes a
16 kHz mono WAV; explicit language hints are not supported at this time.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text. The model auto-detects the audio's
language and emits a transcript in that language; 30 languages are
covered including English, Chinese, Japanese, Korean, German, French,
Spanish, Arabic, Russian, Hindi, and Vietnamese. `transcribe-cli` reads
a 16 kHz mono WAV and returns the transcript text.

See the
[Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`5eb1441`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/commit/5eb1441), pinned 2026-04-19. Validated against the qwen_asr 0.0.6 reference at transcribe.cpp commit [`3f61df7`](https://github.com/handy-computer/transcribe.cpp/tree/3f61df7) on 2026-04-20.
<!-- /catalog -->

The author's `qwen_asr` package is likewise Apache-2.0.

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [Qwen3-ASR-0.6B-BF16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-BF16.gguf) | 1.57 GB | 2.12% |
| F16          | [Qwen3-ASR-0.6B-F16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-F16.gguf) | 1.58 GB | 2.12% |
| Q8_0         | [Qwen3-ASR-0.6B-Q8_0.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf) |  850 MB | 2.11% |
| Q6_K         | [Qwen3-ASR-0.6B-Q6_K.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q6_K.gguf) |  690 MB | 2.11% |
| Q5_K_M       | [Qwen3-ASR-0.6B-Q5_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q5_K_M.gguf) |  645 MB | 2.21% |
| Q4_K_M       | [Qwen3-ASR-0.6B-Q4_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q4_K_M.gguf) |  590 MB | 2.26% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Scored with the Whisper-style English text normalizer and jiwer 3.x on an Apple M4.
Qwen3-ASR is a multilingual model — this number characterizes the English case only.
BF16 / F16 / Q8_0 / Q6_K are all within bootstrap CI of each other; Q5_K_M and
Q4_K_M show a small but real regression driven by the tied token-embedding / head.
Reproduce with `scripts/wer/run.py` + `scripts/wer/score.py`.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| ar       | WER    | 24.51% |
| cs       | WER    | 44.50% |
| da       | WER    | 36.07% |
| de       | WER    |  6.80% |
| el       | WER    | 49.12% |
| en       | WER    |  4.23% |
| es       | WER    |  4.88% |
| fa       | WER    | 50.30% |
| fi       | WER    | 46.49% |
| fil      | WER    | 35.43% |
| fr       | WER    |  7.76% |
| hi       | WER    | 12.68% |
| hu       | WER    | 56.24% |
| id       | WER    |  8.49% |
| it       | WER    |  5.19% |
| ja       | CER    |  8.61% |
| ko       | CER    |  5.82% |
| mk       | WER    | 35.09% |
| ms       | WER    | 17.18% |
| nl       | WER    | 13.90% |
| pl       | WER    | 25.06% |
| pt       | WER    |  6.57% |
| ro       | WER    | 40.65% |
| ru       | WER    | 10.30% |
| sv       | WER    | 35.72% |
| th       | CER    |  8.81% |
| tr       | WER    | 16.74% |
| vi       | WER    |  9.32% |
| yue      | CER    |  7.91% |
| zh       | CER    |  7.57% |
<!-- /catalog -->

**FLEURS-zh** (945 utterances) CER: 7.6% on the upstream `qwen_asr`
reference, 7.64% on the Q8_0 port (95% CI [6.74%, 8.51%]); within
bootstrap noise. Reproduce with
`uv run scripts/wer/run.py --model … --dataset fleurs:zh`; reference run
via `uv run --project scripts/envs/qwen3_asr scripts/wer/run_reference_qwen3_asr_author.py`.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Public API caveat — language hints

This port accepts `params.language == NULL` (auto-detect) and
**rejects any explicit language hint** with
`TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`. The `capabilities.languages`
list documents the 30 languages the model can auto-detect, not a set
of caller-settable hints. Rendering caller-supplied hints into the
chat template is tracked as follow-up work; see the family note at
`docs/porting/families/qwen3_asr.md` for details.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 147 ms (75.02×) | 134 ms (81.78×) |
| Metal   | dots (35.3s) | 556 ms (63.50×) | 511 ms (69.19×) |
| CPU     | jfk (11.0s)  | 545 ms (20.20×) | 547 ms (20.10×) |
| CPU     | dots (35.3s) | 2.06 s (17.13×) | 1.97 s (17.95×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 1.03 s (10.73×) | 908 ms (12.12×) |
| Vulkan  | dots (35.3s) |  4.07 s (8.68×) |  3.55 s (9.96×) |
| CPU     | jfk (11.0s)  |  1.86 s (5.91×) |  1.67 s (6.58×) |
| CPU     | dots (35.3s) |  7.55 s (4.68×) |  6.75 s (5.23×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models qwen3-asr-0.6b
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the author reference
implementation (`qwen_asr` 0.0.6 / transformers 4.57.6) on
`samples/jfk.wav`. All 13 checkpointed tensors fall within family tolerance
on CPU / Metal / Vulkan, and the transcript matches the reference verbatim.

| Field | Value |
| --- | --- |
| Reference | `qwen_asr` 0.0.6 (Alibaba author package) |
| Dump script | `scripts/dump_reference_qwen3_asr_author.py` |
| Manifest | `tests/golden/qwen3_asr/qwen3-asr-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-0.6b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-0.6B \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
  models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-Q4_K_M.gguf \
  --quant Q4_K_M
```

### Validate

```bash
uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-0.6b
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_QWEN3_ASR_0_6B_GGUF=$PWD/models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
TRANSCRIBE_QWEN3_ASR_GGUF=$PWD/models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
  ctest --test-dir build --output-on-failure -R 'qwen3_asr'
```
