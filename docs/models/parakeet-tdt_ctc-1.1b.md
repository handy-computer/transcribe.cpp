# Parakeet TDT-CTC 1.1B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt_ctc-1.1b`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b) at [`675e786`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b/commit/675e786).

Offline English speech-to-text with punctuation and capitalization. A FastConformer-XL encoder with a TDT/RNNT transducer decoder (the auxiliary CTC head from the upstream hybrid checkpoint is dropped at convert time). Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. Outputs cased, punctuated transcripts via
greedy TDT decoding. Token-, word- and segment-level timestamps are
available. Not a streaming model; does not translate.

The hybrid head architecture means a single GGUF carries both decoders;
transcribe.cpp dispatches to TDT, which is faster on this codebase due
to duration-aware frame skipping.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`675e786`](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b/commit/675e786), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-tdt_ctc-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-F32.gguf) | 4.28 GB | 1.87% |
| F16          | [parakeet-tdt_ctc-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-F16.gguf) | 2.15 GB | 1.87% |
| Q8_0         | [parakeet-tdt_ctc-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q8_0.gguf) | 1.27 GB | 1.87% |
| Q6_K         | [parakeet-tdt_ctc-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q6_K.gguf) | 1.04 GB | 1.87% |
| Q5_K_M       | [parakeet-tdt_ctc-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q5_K_M.gguf) |  936 MB | 1.87% |
| Q4_K_M       | [parakeet-tdt_ctc-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-1.1b-gguf/resolve/main/parakeet-tdt_ctc-1.1b-Q4_K_M.gguf) |  825 MB | 1.91% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy TDT/RNN-T transducer decoding, no external LM. F32 reference baseline: 1.87%.
NVIDIA's self-reported number on the same split is 1.82%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.68% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  72 ms (153.68×) |  73 ms (150.60×) |
| Metal   | dots (35.3s) | 184 ms (191.86×) | 188 ms (188.19×) |
| CPU     | jfk (11.0s)  |  458 ms (24.01×) |  509 ms (21.62×) |
| CPU     | dots (35.3s) |  1.44 s (24.54×) |  1.57 s (22.56×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 698 ms (15.76×) | 706 ms (15.59×) |
| Vulkan  | dots (35.3s) | 1.93 s (18.32×) | 1.96 s (18.06×) |
| CPU     | jfk (11.0s)  | 1.10 s (10.03×) |  1.20 s (9.18×) |
| CPU     | dots (35.3s) |  4.11 s (8.60×) |  4.23 s (8.35×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-tdt_ctc-1.1b
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
