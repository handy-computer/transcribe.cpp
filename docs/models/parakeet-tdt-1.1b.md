# Parakeet TDT 1.1B

NVIDIA's [`nvidia/parakeet-tdt-1.1b`](https://huggingface.co/nvidia/parakeet-tdt-1.1b)
ported to transcribe.cpp. A 1.1B-parameter FastConformer-XL encoder with a
TDT/RNN-T transducer decoder (predictor + joint with duration head).

## What it's for

Offline English speech-to-text with greedy TDT decoding. Output is
**lowercase, no punctuation** (per the upstream model card). Token-,
word- and segment-level timestamps are available. Not a streaming model;
does not translate.

The 1.1B-class TDT — same family as the published `parakeet-tdt-0.6b-v2`
but with a larger encoder. TDT decoding (with its duration logits)
generally runs faster than plain RNN-T at comparable accuracy because
each step can advance more than one frame.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt-1.1b)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`53276c6`](https://huggingface.co/nvidia/parakeet-tdt-1.1b/commit/53276c64),
pinned 2026-05-09.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-tdt-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-F32.gguf) | 4.28 GB |                        1.39% |
| F16    | [parakeet-tdt-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-F16.gguf) | 2.15 GB |                        1.39% |
| Q8_0   | [parakeet-tdt-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q8_0.gguf) | 1.27 GB |                        1.38% |
| Q6_K   | [parakeet-tdt-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q6_K.gguf) | 1.04 GB |                        1.40% |
| Q5_K_M | [parakeet-tdt-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q5_K_M.gguf) |  936 MB |                        1.39% |
| Q4_K_M | [parakeet-tdt-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q4_K_M.gguf) |  825 MB |                        1.42% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy TDT decoding and no external LM. F32 reference baseline: 1.39%. NVIDIA's self-reported number on the same split is 1.39% (from the [HF model card](https://huggingface.co/nvidia/parakeet-tdt-1.1b)).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt-1.1b/parakeet-tdt-1.1b-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places). Cells gated on `Tctl < 55°C` per backend.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |   97 ms (113×) |  100 ms (110×) |
| Metal   | dots (35.3s) |  259 ms (137×) |  258 ms (137×) |
| CPU     | jfk (11.0s)  |  652 ms (17×)  |  528 ms (21×)  |
| CPU     | dots (35.3s) | 2.26 s (16×)   | 1.87 s (19×)   |

macOS 26.4.1, transcribe.cpp `a6c097e`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  1.01 s (11×) |  1.03 s (11×) |
| Vulkan  | dots (35.3s) |  3.11 s (11×) |  3.12 s (11×) |
| CPU     | jfk (11.0s)  |  1.91 s (6×)  |  1.57 s (7×)  |
| CPU     | dots (35.3s) |  6.87 s (5×)  |  5.92 s (6×)  |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-tdt-1.1b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-tdt-1.1b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`
via `scripts/validate.py`, sharing the parakeet family tolerance file. The
family-level forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources (fp64 STFT, mel amplification,
attenuation through the encoder).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt-1.1b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt-1.1b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-tdt-1.1b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-1.1b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt-1.1b/parakeet-tdt-1.1b-F32.gguf \
  models/parakeet-tdt-1.1b/parakeet-tdt-1.1b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-tdt-1.1b
```
