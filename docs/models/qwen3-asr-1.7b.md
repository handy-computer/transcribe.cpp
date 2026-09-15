# Qwen3-ASR 1.7B

<!-- catalog:intro -->
Upstream: [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) at [`7278e1e`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/commit/7278e1e).

Offline multilingual speech-to-text. Same audio-LLM architecture as the
0.6B variant (bidirectional audio encoder feeding a Qwen3 causal LM with
audio-token injection), wider: encoder `d_model=1024` (16 heads), LM
`hidden_size=2048`, `intermediate_size=6144`. Auto-detects the audio's
language across 30 languages and emits the transcript in that language.
Takes a 16 kHz mono WAV; explicit language hints are not supported at
this time.
<!-- /catalog -->

## What it's for

Same contract as the 0.6B: offline multilingual STT, 30-language
auto-detect, 16 kHz mono WAV in → transcript text out. Targets the same
use cases as Qwen3-ASR-0.6B with more parameters for accuracy headroom.

See the
[Qwen3-ASR-1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
for training data and upstream evaluation.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`7278e1e`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/commit/7278e1e), pinned 2026-04-19. Validated against the qwen_asr 0.0.6 reference at transcribe.cpp commit [`3f61df7`](https://github.com/handy-computer/transcribe.cpp/tree/3f61df7) on 2026-04-20.
<!-- /catalog -->

The author's `qwen_asr` package is likewise Apache-2.0.

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [Qwen3-ASR-1.7B-BF16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-BF16.gguf) | 4.08 GB | 1.62% |
| F16          | [Qwen3-ASR-1.7B-F16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-F16.gguf) | 4.09 GB | 1.62% |
| Q8_0         | [Qwen3-ASR-1.7B-Q8_0.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q8_0.gguf) | 2.19 GB | 1.62% |
| Q6_K         | [Qwen3-ASR-1.7B-Q6_K.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q6_K.gguf) | 1.69 GB | 1.65% |
| Q5_K_M       | [Qwen3-ASR-1.7B-Q5_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q5_K_M.gguf) | 1.52 GB | 1.65% |
| Q4_K_M       | [Qwen3-ASR-1.7B-Q4_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q4_K_M.gguf) | 1.32 GB | 1.81% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Scored with the Whisper-style English text normalizer and jiwer 3.x on an Apple M4.
Qwen3-ASR is a multilingual model — this number characterizes the English case only.
The larger decoder gives 1.7B more quantization headroom than the 0.6B; BF16 / F16 /
Q8_0 / Q6_K / Q5_K_M are all within bootstrap CI of each other, and Q4_K_M regresses
only ~0.2 WER points. Reproduce with `scripts/wer/run.py` + `scripts/wer/score.py`.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| ar       | WER    | 14.91% |
| cs       | WER    | 22.97% |
| da       | WER    | 21.23% |
| de       | WER    |  4.25% |
| el       | WER    | 29.22% |
| en       | WER    |  3.23% |
| es       | WER    |  3.31% |
| fa       | WER    | 28.29% |
| fi       | WER    | 25.48% |
| fil      | WER    | 24.29% |
| fr       | WER    |  4.52% |
| hi       | WER    |  7.84% |
| hu       | WER    | 32.84% |
| id       | WER    |  5.37% |
| it       | WER    |  2.68% |
| ja       | CER    |  5.29% |
| ko       | CER    |  4.60% |
| mk       | WER    | 18.22% |
| ms       | WER    | 10.42% |
| nl       | WER    |  7.43% |
| pl       | WER    | 12.50% |
| pt       | WER    |  4.37% |
| ro       | WER    | 20.46% |
| ru       | WER    |  6.25% |
| sv       | WER    | 19.68% |
| th       | CER    |  6.89% |
| tr       | WER    |  9.46% |
| vi       | WER    |  6.15% |
| yue      | CER    |  6.13% |
| zh       | CER    |  7.14% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 243 ms (45.33×) | 209 ms (52.65×) |
| Metal   | dots (35.3s) | 959 ms (36.83×) | 804 ms (43.95×) |
| CPU     | jfk (11.0s)  | 1.06 s (10.35×) |  1.23 s (8.96×) |
| CPU     | dots (35.3s) |  4.00 s (8.83×) |  3.80 s (9.31×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.47 s (4.45×) |  2.02 s (5.45×) |
| Vulkan  | dots (35.3s) |  9.49 s (3.72×) |  7.89 s (4.48×) |
| CPU     | jfk (11.0s)  |  4.10 s (2.68×) |  3.51 s (3.14×) |
| CPU     | dots (35.3s) | 15.63 s (2.26×) | 13.07 s (2.70×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models qwen3-asr-1.7b
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the author
reference implementation (`qwen_asr` 0.0.6 / transformers 4.57.6) on
`samples/jfk.wav`. All 13 checkpointed tensors fall within family
tolerance, and the transcript matches the reference verbatim.
Tolerances are set in `tests/tolerances/qwen3_asr-1.7b.json`. Last
validated at commit
[`3f61df7`](https://github.com/handy-computer/transcribe.cpp/tree/3f61df7).

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
