# Parakeet CTC 1.1B

NVIDIA's [`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b)
ported to transcribe.cpp. A 1.1B-parameter FastConformer-XL encoder with a
linear CTC head.

## What it's for

Offline English speech-to-text with greedy CTC decoding. Output is
**lowercase, no punctuation** (the upstream model card explicitly notes
"lower case English alphabet"). Token- and word-level timestamps are
available. Not a streaming model; does not translate.

This is the largest pure-CTC variant in the parakeet family and trades
size for accuracy. NVIDIA reports a 0.04 percentage-point improvement on
LibriSpeech test-clean over the 0.6B sibling.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-ctc-1.1b)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`a707e81`](https://huggingface.co/nvidia/parakeet-ctc-1.1b/commit/a707e818195cb97c8f7da2fc36b221a29f69a5db),
pinned 2026-05-09.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | parakeet-ctc-1.1b-F32.gguf    | 4.25 GB | — |
| Q8_0   | parakeet-ctc-1.1b-Q8_0.gguf   | 1.26 GB | — |
| Q4_K_M | parakeet-ctc-1.1b-Q4_K_M.gguf | 818 MB  | — |

WER will be measured on the full LibriSpeech test-clean split (2620
utterances) with greedy CTC decoding and no external LM. NVIDIA reports
1.83% WER on the same split (from the
[HF model card](https://huggingface.co/nvidia/parakeet-ctc-1.1b)).
Per-quant numbers will be filled in once the reference-machine WER sweep
completes.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-ctc-1.1b/parakeet-ctc-1.1b-Q8_0.gguf \
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
| Vulkan  | jfk (11.0s)  |  825 ms (13×) |  822 ms (13×) |
| Vulkan  | dots (35.3s) |  2.34 s (15×) |  2.33 s (15×) |
| CPU     | jfk (11.0s)  |  1.75 s (6×)  |  1.38 s (8×)  |
| CPU     | dots (35.3s) |  6.08 s (6×)  |  5.12 s (7×)  |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-ctc-1.1b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-ctc-1.1b-publication
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
| Reference | NeMo, `nvidia/parakeet-ctc-1.1b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-ctc-1.1b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-ctc-1.1b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-ctc-1.1b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-ctc-1.1b/parakeet-ctc-1.1b-F32.gguf \
  models/parakeet-ctc-1.1b/parakeet-ctc-1.1b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-ctc-1.1b
```
