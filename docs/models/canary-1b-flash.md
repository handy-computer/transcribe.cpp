# Canary 1B Flash

<!-- catalog:intro -->
Upstream: [`nvidia/canary-1b-flash`](https://huggingface.co/nvidia/canary-1b-flash) at [`a9a55e0`](https://huggingface.co/nvidia/canary-1b-flash/commit/a9a55e0).

Offline multilingual speech-to-text and translation. A multitask AED
with a 32-layer FastConformer encoder and a 4-layer Transformer decoder.
Supports automatic speech recognition in English, German, Spanish, and
French, and bidirectional EN↔{DE, ES, FR} translation. Takes a 16 kHz
mono WAV and produces a transcript. Not a streaming model; word/segment
timestamps are upstream-experimental and not exposed in the v1 port.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and translation. The model takes a
16 kHz mono WAV and produces a transcript. Supports:

- **ASR** in English, German, Spanish, and French (with explicit language
  hint).
- **Translation** between English and German, Spanish, or French (both
  directions).

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-1b-flash)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`a9a55e0`](https://huggingface.co/nvidia/canary-1b-flash/commit/a9a55e0), pinned 2026-05-08. Validated against the NeMo reference at transcribe.cpp commit [`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda) on 2026-05-08.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [canary-1b-flash-F32.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-F32.gguf) | 3.56 GB | 1.62% |
| F16          | [canary-1b-flash-F16.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-F16.gguf) | 1.79 GB | 1.62% |
| Q8_0         | [canary-1b-flash-Q8_0.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q8_0.gguf) | 1.05 GB | 1.62% |
| Q6_K         | [canary-1b-flash-Q6_K.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q6_K.gguf) |  858 MB | 1.65% |
| Q5_K_M       | [canary-1b-flash-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q5_K_M.gguf) |  770 MB | 1.64% |
| Q4_K_M       | [canary-1b-flash-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q4_K_M.gguf) |  677 MB | 1.59% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM. F32 reference baseline: 1.62%. NVIDIA's
self-reported number on the upstream model card is 1.48%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 6.13% |
| en       | WER    | 4.75% |
| es       | WER    | 6.73% |
| fr       | WER    | 7.22% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

# ASR (English)
build/bin/transcribe-cli \
  -m models/canary-1b-flash/canary-1b-flash-Q8_0.gguf \
  -l en \
  samples/jfk.wav

# Translation (English audio → German text)
build/bin/transcribe-cli \
  -m models/canary-1b-flash/canary-1b-flash-Q8_0.gguf \
  --task translate \
  -l en --target-language de \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags specific to canary:

- `--pnc` / `--no-pnc` — punctuation & capitalization (default on).
- `-l <code>` — source language code (`en`, `de`, `es`, `fr`).
- `--task translate` + `--target-language <code>` — switch to translation
  mode.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |               Q8_0 |             Q4_K_M |
| ------- | ------------ | -----------------: | -----------------: |
| Metal   | jfk (11.0s)  |  93.5 ms (117.69×) |  91.8 ms (119.77×) |
| Metal   | dots (35.3s) | 333.9 ms (105.82×) | 315.5 ms (112.00×) |
| CPU     | jfk (11.0s)  |  416.7 ms (26.40×) |  429.5 ms (25.61×) |
| CPU     | dots (35.3s) |    1.44 s (24.47×) |    1.49 s (23.78×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Vulkan  | jfk (11.0s)  | 691.8 ms (15.90×) | 693.5 ms (15.86×) |
| Vulkan  | dots (35.3s) |   2.35 s (15.06×) |   2.24 s (15.78×) |
| CPU     | jfk (11.0s)  |   1.09 s (10.07×) |    1.10 s (9.96×) |
| CPU     | dots (35.3s) |    4.63 s (7.63×) |    4.54 s (7.79×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models canary-1b-flash
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav`. All checkpointed tensors fall within family
tolerance and the F32 transcript matches the NeMo reference. Last
validated at commit
[`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/canary-1b-flash` |
| Dump script | `scripts/dump_reference_canary_nemo.py` |
| Manifest | `tests/golden/canary/canary-1b-flash.manifest.json` |
| Tolerances | `tests/tolerances/canary.json` |
| Command | `uv run scripts/validate.py all --family canary --variant canary-1b-flash` |

For the full porting writeup, see
[`docs/porting/families/canary.md`](../porting/families/canary.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/canary \
  scripts/convert-canary.py nvidia/canary-1b-flash --repo-id nvidia/canary-1b-flash
```

### Quantize

```bash
uv run scripts/quantize-all.py models/canary-1b-flash/canary-1b-flash-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family canary --variant canary-1b-flash
```
