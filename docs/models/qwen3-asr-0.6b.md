# Qwen3-ASR 0.6B

Alibaba's [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
ported to transcribe.cpp. An 18-layer bidirectional audio encoder feeds a
28-layer Qwen3 causal LM with audio-token injection (no cross-attention —
the LM processes a fused audio+text sequence through a chat template).

## What it's for

Offline multilingual speech-to-text. The model auto-detects the audio's
language and emits a transcript in that language; 30 languages are
covered including English, Chinese, Japanese, Korean, German, French,
Spanish, Arabic, Russian, Hindi, and Vietnamese. `transcribe-cli` reads
a 16 kHz mono WAV and returns the transcript text.

See the
[Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0 (weights) / Apache-2.0 (author `qwen_asr` package).
Ported from upstream commit
[`5eb1441`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/commit/5eb144179a02acc5e5ba31e748d22b0cf3e303b0).

## Download

| Quantization | Size | Notes |
| --- | ---: | --- |
| BF16   | ~1.5 GB | accuracy reference (converter output) |
| Q8_0   | ~790 MB | weight-only quant for embed + linear |
| Q4_K_M | ~470 MB | smallest shipped preset; Q6_K for the tied embed |

Published GGUF URLs will land alongside the Cohere checkpoints in a
follow-up release. Until then, convert locally:

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-0.6B \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0
```

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

Cells are wall-clock latency (single run on the JFK 11s sample),
with speedup over realtime in parentheses.

### Apple M4 Max

| Backend | Mel    | Encode  | Decode  | Realtime |
| ------- | -----: | ------: | ------: | -------: |
| Metal   |  66 ms |   36 ms |  202 ms |     33×  |
| Vulkan  |  43 ms |  150 ms |  370 ms |     20×  |
| CPU     |  63 ms | 3576 ms | 2256 ms |      2×  |

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the author
reference implementation (`qwen_asr` 0.0.6 / transformers 4.57.6) on
`samples/jfk.wav`. All 13 checkpointed tensors fall within family
tolerance on CPU / Metal / Vulkan, and the post-prefix transcript
text matches the reference exactly.

| Field | Value |
| --- | --- |
| Reference | `qwen_asr` 0.0.6 (Alibaba author package) |
| Dump script | `scripts/dump_reference_qwen3_asr_author.py` |
| Manifest | `tests/golden/qwen3_asr/qwen3-asr-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-0.6b` |

Selected tensors (observed on CPU; see tolerance file for budgets):

| Tensor | Max abs diff | Mean abs diff |
| --- | ---: | ---: |
| `enc.mel.in`         | `3.906e-03` | `6.608e-04` |
| `enc.block.0.out`    | `5.758e-01` | `1.378e-02` |
| `enc.block.17.out`   | `1.634e+01` | `1.294e-01` |
| `enc.proj.out`       | `1.161e-01` | `2.577e-03` |
| `dec.audio_injected` | `1.161e-01` | `2.332e-03` |
| `dec.block.0.out`    | `1.967e+00` | `2.970e-02` |
| `dec.block.27.out`   | `8.556e+01` | `1.256e+00` |
| `dec.logits_raw`     | `1.677e+00` | `3.364e-01` |

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
