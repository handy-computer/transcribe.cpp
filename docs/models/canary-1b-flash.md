# Canary 1B Flash

NVIDIA's [`nvidia/canary-1b-flash`](https://huggingface.co/nvidia/canary-1b-flash)
ported to transcribe.cpp. An 883M-parameter multitask AED with a 32-layer
FastConformer encoder and a 4-layer Transformer decoder.

## What it's for

Offline multilingual speech-to-text and translation. The model takes a
16 kHz mono WAV and produces a transcript. Supports:

- **ASR** in English, German, Spanish, and French (with explicit language
  hint).
- **Translation** from English to German, Spanish, or French.

Higher accuracy than canary-180m-flash at ~5× the model size; both share
the same prompt format and the same shallow 4-layer decoder, so
per-token decode latency is similar — the cost is in the encoder.

Not a streaming model. Word and segment timestamps are upstream-experimental
and not exposed in the v1 port.

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-1b-flash)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`a9a55e0`](https://huggingface.co/nvidia/canary-1b-flash/commit/a9a55e0295e7dd50d0c8c2a19491900a0daf24f3),
pinned 2026-05-08.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [canary-1b-flash-F32.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-F32.gguf)       | 3.3 GB | 1.62% |
| F16    | [canary-1b-flash-F16.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-F16.gguf)       | 1.7 GB | 1.62% |
| Q8_0   | [canary-1b-flash-Q8_0.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q8_0.gguf)     | 1.0 GB | 1.62% |
| Q6_K   | [canary-1b-flash-Q6_K.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q6_K.gguf)     | 818 MB | 1.65% |
| Q5_K_M | [canary-1b-flash-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q5_K_M.gguf) | 734 MB | 1.64% |
| Q4_K_M | [canary-1b-flash-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-flash-gguf/resolve/main/canary-1b-flash-Q4_K_M.gguf) | 646 MB | 1.59% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding and no external LM. F32 reference baseline: 1.62%.
NVIDIA's self-reported number on the upstream model card is 1.48%; the
small gap is well inside the Stage 7 ref-dtype gate (|Δ| ≤ 1pp). Quants
are all within ~0.05pp of the F32 baseline.

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places).

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  | 118.9 ms (92.5×) | 113.6 ms (96.9×) |
| Metal   | dots (35.3s) | 414.8 ms (85.2×) | 390.2 ms (90.5×) |
| CPU     | jfk (11.0s)  | 518.3 ms (21.2×) | 430.1 ms (25.6×) |
| CPU     | dots (35.3s) |   1.79 s (19.7×) |   1.51 s (23.4×) |

macOS 26.4.1, transcribe.cpp `0f42b37`.

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Vulkan  | jfk (11.0s)  | 763.5 ms (14.4×) | 704.2 ms (15.6×) |
| Vulkan  | dots (35.3s) |   2.46 s (14.4×) |   2.31 s (15.3×) |
| CPU     | jfk (11.0s)  |    1.49 s (7.4×) |    1.13 s (9.7×) |
| CPU     | dots (35.3s) |    5.48 s (6.4×) |    4.51 s (7.8×) |

Fedora Linux 43, transcribe.cpp `4d44530`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models canary-1b-flash \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name canary-1b-flash-publication
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
