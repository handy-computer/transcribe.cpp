# Qwen3-ASR 1.7B

Alibaba's [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
ported to transcribe.cpp. Architecture is the same audio-LLM pattern as
the 0.6B variant (24-layer bidirectional audio encoder + Qwen3 causal LM
with audio-token injection); the 1.7B is wider: encoder `d_model=1024`
(16 heads), LM `hidden_size=2048`, `intermediate_size=6144`.

## What it's for

Same contract as the 0.6B: offline multilingual STT, 30-language
auto-detect, 16 kHz mono WAV in → transcript text out. Targets the same
use cases as Qwen3-ASR-0.6B with more parameters for accuracy headroom.

See the
[Qwen3-ASR-1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
for training data and upstream evaluation.

Licensed Apache-2.0 (weights) / Apache-2.0 (author `qwen_asr` package).
Ported from upstream commit
[`7278e1e`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/commit/7278e1e70fe206f11671096ffdd38061171dd6e5).

## Download

| Quantization | Size | Notes |
| --- | ---: | --- |
| BF16   | ~3.9 GB | accuracy reference (converter output) |
| Q8_0   | ~2.0 GB | weight-only quant for embed + linear |
| Q4_K_M | ~1.2 GB | smallest shipped preset; Q6_K for the tied embed |

Published GGUF URLs will land alongside the Cohere checkpoints in a
follow-up release. Until then, convert locally:

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
```

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Public API caveat — language hints

Same contract as the 0.6B: `params.language == NULL` runs auto-detect;
any explicit hint returns `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`. See the
0.6B doc and the family note at `docs/porting/families/qwen3_asr.md` for
the rationale and the planned follow-up.

## Performance

Detailed per-backend numbers land after the 1.7B benchmark matrix
publishes. The runtime shape matches the 0.6B (same mel frontend, same
encoder/decoder step shape, same KV cache policy); scale the 0.6B
numbers by roughly the parameter ratio as a first approximation.

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the author
reference implementation (`qwen_asr` 0.0.6 / transformers 4.57.6) on
`samples/jfk.wav`. Tolerances are set in
`tests/tolerances/qwen3_asr-1.7b.json`.

| Field | Value |
| --- | --- |
| Reference | `qwen_asr` 0.0.6 (Alibaba author package) |
| Dump script | `scripts/dump_reference_qwen3_asr_author.py` |
| Manifest | `tests/golden/qwen3_asr/qwen3-asr-1.7b.manifest.json` |
| Command | `uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-1.7b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
  models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-Q4_K_M.gguf \
  --quant Q4_K_M
```

### Validate

```bash
uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-1.7b
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_QWEN3_ASR_1_7B_GGUF=$PWD/models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
TRANSCRIBE_QWEN3_ASR_GGUF=$PWD/models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
  ctest --test-dir build --output-on-failure -R 'qwen3_asr'
```
