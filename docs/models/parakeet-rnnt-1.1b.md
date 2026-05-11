# Parakeet RNN-T 1.1B

NVIDIA's [`nvidia/parakeet-rnnt-1.1b`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
ported to transcribe.cpp. A 1.1B-parameter FastConformer-XL encoder with a
classic RNN-T transducer decoder (predictor + joint, no duration head).

## What it's for

Offline English speech-to-text with greedy RNN-T decoding. Output is
**lowercase, no punctuation** (per the upstream model card). Token- and
word-level timestamps are available. Not a streaming model; does not
translate.

The largest pure RNN-T variant in the family. NVIDIA reports the lowest
LibriSpeech test-clean WER of any parakeet variant on the same split.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`a07b19e`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/commit/a07b19e9),
pinned 2026-05-10.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-rnnt-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-F32.gguf) | 4.28 GB |                        1.45% |
| F16    | [parakeet-rnnt-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-F16.gguf) | 2.15 GB |                        1.45% |
| Q8_0   | [parakeet-rnnt-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q8_0.gguf) | 1.27 GB |                        1.46% |
| Q6_K   | [parakeet-rnnt-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q6_K.gguf) | 1.04 GB |                        1.43% |
| Q5_K_M | [parakeet-rnnt-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q5_K_M.gguf) |  936 MB |                        1.43% |
| Q4_K_M | [parakeet-rnnt-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q4_K_M.gguf) |  825 MB |                        1.41% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy RNN-T decoding and no external LM. F32 reference baseline: 1.45%. NVIDIA's self-reported number on the same split is 1.46% (from the [HF model card](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-rnnt-1.1b/parakeet-rnnt-1.1b-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  96 ms (114×) |  97 ms (114×) |
| Metal   | dots (35.3s) | 258 ms (137×) | 265 ms (133×) |
| CPU     | jfk (11.0s)  |  606 ms (18×) |  506 ms (22×) |
| CPU     | dots (35.3s) |  2.05 s (17×) |  1.72 s (20×) |

macOS 26.4.1, transcribe.cpp `12f1076`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  1.69 s (7×)  |  1.68 s (7×)  |
| Vulkan  | dots (35.3s) |  5.45 s (6×)  |  5.49 s (6×)  |
| CPU     | jfk (11.0s)  |  2.52 s (4×)  |  2.23 s (5×)  |
| CPU     | dots (35.3s) |  9.16 s (4×)  |  8.16 s (4×)  |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-rnnt-1.1b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-rnnt-1.1b-publication
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
| Reference | NeMo, `nvidia/parakeet-rnnt-1.1b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-rnnt-1.1b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-rnnt-1.1b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-rnnt-1.1b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-rnnt-1.1b/parakeet-rnnt-1.1b-F32.gguf \
  models/parakeet-rnnt-1.1b/parakeet-rnnt-1.1b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-rnnt-1.1b
```
