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

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [Qwen3-ASR-0.6B-BF16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-BF16.gguf)     | 1499 MB | 2.11% |
| F16    | [Qwen3-ASR-0.6B-F16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-F16.gguf)       | 1507 MB | 2.12% |
| Q8_0   | [Qwen3-ASR-0.6B-Q8_0.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf)     |  811 MB | 2.11% |
| Q6_K   | [Qwen3-ASR-0.6B-Q6_K.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q6_K.gguf)     |  763 MB | 2.10% |
| Q5_K_M | [Qwen3-ASR-0.6B-Q5_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q5_K_M.gguf) |  699 MB | 2.20% |
| Q4_K_M | [Qwen3-ASR-0.6B-Q4_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q4_K_M.gguf) |  639 MB | 2.32% |

WER measured on LibriSpeech `test-clean` (2620 utterances), Whisper-style
English text normalizer, jiwer 3.x, metal backend on Apple M4. Reproduce
with `scripts/wer/run.py` + `scripts/wer/score.py`.

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  155 ms (71×)  |  142 ms (77×)  |
| Metal   | dots (35.3s) |  597 ms (59×)  |  527 ms (67×)  |
| CPU     | jfk (11.0s)  |  660 ms (17×)  |  588 ms (19×)  |
| CPU     | dots (35.3s) | 2.26 s (16×)   | 2.10 s (17×)   |

macOS 26.3.1, transcribe.cpp `0c88a71`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  | 1.27 s (8.7×)  | 1.08 s (10.1×) |
| Vulkan  | dots (35.3s) | 4.87 s (7.3×)  | 3.99 s (8.9×)  |
| CPU     | jfk (11.0s)  | 2.37 s (4.6×)  | 1.92 s (5.7×)  |
| CPU     | dots (35.3s) | 8.61 s (4.1×)  | 7.34 s (4.8×)  |

Fedora 43, transcribe.cpp `3d16f74`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Qwen3-ASR-0.6B \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name qwen3-asr-0.6b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the author
reference implementation (`qwen_asr` 0.0.6 / transformers 4.57.6) on
`samples/jfk.wav`. All 13 checkpointed tensors fall within family
tolerance on CPU / Metal / Vulkan, and the transcript matches the
reference verbatim. Last validated at commit
[`3f61df7`](https://github.com/handy-computer/transcribe.cpp/tree/3f61df7).

| Field | Value |
| --- | --- |
| Reference | `qwen_asr` 0.0.6 (Alibaba author package) |
| Dump script | `scripts/dump_reference_qwen3_asr_author.py` |
| Manifest | `tests/golden/qwen3_asr/qwen3-asr-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-0.6b` |

Selected tensors (observed on CPU; see tolerance file for budgets):

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`         | `3.906e-03` | `6.608e-04` | fp64 vs fp32 STFT precision gap |
| `enc.subsample.out`  | `1.814e-02` | `1.891e-03` | After the 4× conv subsampler |
| `enc.block.0.out`    | `5.645e-02` | `2.768e-03` | Early encoder |
| `enc.block.17.out`   | `8.221e-01` | `9.720e-03` | Final encoder block |
| `enc.proj.out`       | `1.484e-02` | `2.324e-04` | Audio→LM width projection |
| `dec.audio_injected` | `1.484e-02` | `2.103e-04` | Fused audio+text sequence |
| `dec.token_emb`      | `0.000e+00` | `0.000e+00` | Exact match |
| `dec.block.0.out`    | `1.719e-01` | `3.127e-03` | Early LM |
| `dec.block.27.out`   | `2.912e+01` | `3.824e-01` | Final LM block (accumulated) |
| `dec.out_before_head`| `2.844e+01` | `1.786e-01` | Pre-head hidden state |
| `dec.logits_raw`     | `1.054e+00` | `1.749e-01` | Raw logits |

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
