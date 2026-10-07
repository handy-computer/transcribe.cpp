# Parakeet CTC 0.6B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-ctc-0.6b`](https://huggingface.co/nvidia/parakeet-ctc-0.6b) at [`ad09ba1`](https://huggingface.co/nvidia/parakeet-ctc-0.6b/commit/ad09ba1).

Offline English speech-to-text with greedy CTC decoding. A FastConformer-Large encoder with a linear CTC head — the simplest and fastest decoder in the parakeet family. Output is lowercase, no punctuation. Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline English speech-to-text with greedy CTC decoding. Output is
**lowercase, no punctuation** (the upstream model card explicitly notes
"lower case English alphabet"). Token- and word-level timestamps are
available. Not a streaming model; does not translate.

The encoder is identical in shape to `parakeet-tdt-0.6b-v2` (24 layers,
1024-d), so this variant is the fastest 0.6B-class option in the family
on this codebase.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`ad09ba1`](https://huggingface.co/nvidia/parakeet-ctc-0.6b/commit/ad09ba1), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-ctc-0.6b-F32.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-F32.gguf) | 2.44 GB | 1.87% |
| F16          | [parakeet-ctc-0.6b-F16.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-F16.gguf) | 1.22 GB | 1.87% |
| Q8_0         | [parakeet-ctc-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q8_0.gguf) |  722 MB | 1.87% |
| Q6_K         | [parakeet-ctc-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q6_K.gguf) |  594 MB | 1.84% |
| Q5_K_M       | [parakeet-ctc-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q5_K_M.gguf) |  533 MB | 1.87% |
| Q4_K_M       | [parakeet-ctc-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q4_K_M.gguf) |  469 MB | 1.90% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy CTC decoding, no external LM. F32 reference baseline: 1.87%. NVIDIA's
self-reported number on the same split is 1.87%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 5.53% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  48 ms (227.64×) |  48 ms (226.86×) |
| Metal   | dots (35.3s) | 111 ms (317.21×) | 113 ms (311.61×) |
| CPU     | jfk (11.0s)  |  270 ms (40.66×) |  294 ms (37.48×) |
| CPU     | dots (35.3s) |  921 ms (38.35×) |  987 ms (35.80×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 418 ms (26.29×) | 427 ms (25.77×) |
| Vulkan  | dots (35.3s) | 1.14 s (30.91×) | 1.16 s (30.57×) |
| CPU     | jfk (11.0s)  | 664 ms (16.58×) | 723 ms (15.21×) |
| CPU     | dots (35.3s) | 2.57 s (13.74×) | 2.64 s (13.36×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-ctc-0.6b
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`
via `scripts/validate.py`, sharing the parakeet family tolerance file. The
encoder shape is identical to `parakeet-tdt-0.6b-v2` and uses the same
FastConformer code path; the family-level forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources (fp64 STFT, mel amplification,
attenuation through the encoder).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-ctc-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-ctc-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-ctc-0.6b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-ctc-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-F32.gguf \
  models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-ctc-0.6b
```
