# Parakeet TDT-CTC 1.1B

NVIDIA's [`nvidia/parakeet-tdt_ctc-1.1b`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b)
ported to transcribe.cpp. A hybrid 1.1B-parameter FastConformer-XL encoder
with both TDT and CTC heads sharing the same encoder; transcribe.cpp uses
the TDT head by default.

## What it's for

Offline English speech-to-text. Outputs cased, punctuated transcripts via
greedy TDT decoding. Token-, word- and segment-level timestamps are
available. Not a streaming model; does not translate.

The hybrid head architecture means a single GGUF carries both decoders;
transcribe.cpp dispatches to TDT, which is faster on this codebase due
to duration-aware frame skipping.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`675e786`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b/commit/675e7868),
pinned 2026-05-09.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-tdt_ctc-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-F32.gguf) | 4.28 GB |                        1.87% |
| F16    | [parakeet-tdt_ctc-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-F16.gguf) | 2.15 GB |                        1.87% |
| Q8_0   | [parakeet-tdt_ctc-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q8_0.gguf) | 1.27 GB |                        1.87% |
| Q6_K   | [parakeet-tdt_ctc-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q6_K.gguf) | 1.04 GB |                        1.87% |
| Q5_K_M | [parakeet-tdt_ctc-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q5_K_M.gguf) |  936 MB |                        1.87% |
| Q4_K_M | [parakeet-tdt_ctc-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q4_K_M.gguf) |  825 MB |                        1.91% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy TDT decoding and no external LM. F32 reference baseline: 1.87%. NVIDIA's self-reported number on the same split is 1.82% (from the [HF model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b)).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt_ctc-1.1b/parakeet-tdt_ctc-1.1b-Q8_0.gguf \
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

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  1.17 s (9×)  |  1.18 s (9×)  |
| Vulkan  | dots (35.3s) |  3.47 s (10×) |  3.52 s (10×) |
| CPU     | jfk (11.0s)  |  2.05 s (5×)  |  1.73 s (6×)  |
| CPU     | dots (35.3s) |  6.90 s (5×)  |  6.00 s (6×)  |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-tdt_ctc-1.1b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-tdt_ctc-1.1b-publication
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
| Reference | NeMo, `nvidia/parakeet-tdt_ctc-1.1b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt_ctc-1.1b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-tdt_ctc-1.1b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt_ctc-1.1b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt_ctc-1.1b/parakeet-tdt_ctc-1.1b-F32.gguf \
  models/parakeet-tdt_ctc-1.1b/parakeet-tdt_ctc-1.1b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-tdt_ctc-1.1b
```
