# Parakeet RNN-T 0.6B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-rnnt-0.6b`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) at [`c0c1f09`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b/commit/c0c1f09).

Offline English speech-to-text with greedy RNN-T decoding. A FastConformer-Large encoder with an RNN-T transducer decoder. Output is lowercase, no punctuation. Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline English speech-to-text with greedy RNN-T decoding. Output is
**lowercase, no punctuation** (per the upstream model card). Token- and
word-level timestamps are available. Not a streaming model; does not
translate.

The encoder shape matches `parakeet-tdt-0.6b-v2` (24 layers, 1024-d) but
the head is plain RNN-T rather than TDT — the joint network emits exactly
`vocab + 1` logits with no duration extras, and decoding has no
frame-skip choice. Per-frame iterative decode means RNN-T runs slower
than the CTC variant at the same encoder size.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-rnnt-0.6b)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`c0c1f09`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b/commit/c0c1f09), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-rnnt-0.6b-F32.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-F32.gguf) | 2.47 GB | 1.62% |
| F16          | [parakeet-rnnt-0.6b-F16.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-F16.gguf) | 1.24 GB | 1.62% |
| Q8_0         | [parakeet-rnnt-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q8_0.gguf) |  730 MB | 1.62% |
| Q6_K         | [parakeet-rnnt-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q6_K.gguf) |  601 MB | 1.62% |
| Q5_K_M       | [parakeet-rnnt-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q5_K_M.gguf) |  540 MB | 1.62% |
| Q4_K_M       | [parakeet-rnnt-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q4_K_M.gguf) |  476 MB | 1.66% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy RNN-T decoding, no external LM. F32 reference baseline: 1.62%. NVIDIA's
self-reported number on the same split is 1.63%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.57% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  55 ms (201.01×) |  56 ms (198.17×) |
| Metal   | dots (35.3s) | 138 ms (255.29×) | 142 ms (248.43×) |
| CPU     | jfk (11.0s)  |  281 ms (39.19×) |  301 ms (36.52×) |
| CPU     | dots (35.3s) |  977 ms (36.16×) |  1.04 s (34.02×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 425 ms (25.87×) | 431 ms (25.51×) |
| Vulkan  | dots (35.3s) | 1.25 s (28.33×) | 1.27 s (27.86×) |
| CPU     | jfk (11.0s)  | 675 ms (16.30×) | 720 ms (15.29×) |
| CPU     | dots (35.3s) | 2.64 s (13.39×) | 2.71 s (13.05×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-rnnt-0.6b
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
| Reference | NeMo, `nvidia/parakeet-rnnt-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-rnnt-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-rnnt-0.6b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-rnnt-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-F32.gguf \
  models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-rnnt-0.6b
```
