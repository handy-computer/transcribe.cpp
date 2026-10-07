# Parakeet TDT 1.1B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt-1.1b`](https://huggingface.co/nvidia/parakeet-tdt-1.1b) at [`53276c6`](https://huggingface.co/nvidia/parakeet-tdt-1.1b/commit/53276c6).

Offline English speech-to-text. A FastConformer-XL encoder with a TDT/RNNT transducer decoder. Takes a 16 kHz mono WAV and produces a transcript with optional token-level timestamps. Not a streaming model and does not translate.
<!-- /catalog -->

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

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`53276c6`](https://huggingface.co/nvidia/parakeet-tdt-1.1b/commit/53276c6), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-tdt-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-F32.gguf) | 4.28 GB | 1.39% |
| F16          | [parakeet-tdt-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-F16.gguf) | 2.15 GB | 1.39% |
| Q8_0         | [parakeet-tdt-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q8_0.gguf) | 1.27 GB | 1.38% |
| Q6_K         | [parakeet-tdt-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q6_K.gguf) | 1.04 GB | 1.40% |
| Q5_K_M       | [parakeet-tdt-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q5_K_M.gguf) |  936 MB | 1.39% |
| Q4_K_M       | [parakeet-tdt-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-1.1b-gguf/resolve/main/parakeet-tdt-1.1b-Q4_K_M.gguf) |  825 MB | 1.42% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy TDT/RNN-T transducer decoding, no external LM. F32 reference baseline: 1.39%.
NVIDIA's self-reported number on the same split is 1.39%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.24% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  81 ms (135.38×) |  84 ms (130.98×) |
| Metal   | dots (35.3s) | 196 ms (180.48×) | 201 ms (175.57×) |
| CPU     | jfk (11.0s)  |  675 ms (16.31×) |  518 ms (21.25×) |
| CPU     | dots (35.3s) |  1.76 s (20.06×) |  1.76 s (20.06×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 694 ms (15.86×) | 706 ms (15.58×) |
| Vulkan  | dots (35.3s) | 2.07 s (17.03×) | 2.12 s (16.65×) |
| CPU     | jfk (11.0s)  |  1.12 s (9.82×) |  1.21 s (9.08×) |
| CPU     | dots (35.3s) |  4.54 s (7.79×) |  4.66 s (7.57×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-tdt-1.1b
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
