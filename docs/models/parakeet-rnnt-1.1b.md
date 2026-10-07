# Parakeet RNN-T 1.1B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-rnnt-1.1b`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) at [`a07b19e`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/commit/a07b19e).

Offline English speech-to-text with greedy RNN-T decoding. A FastConformer-XL encoder with an RNN-T transducer decoder. Output is lowercase, no punctuation. Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline English speech-to-text with greedy RNN-T decoding. Output is
**lowercase, no punctuation** (per the upstream model card). Token- and
word-level timestamps are available. Not a streaming model; does not
translate.

The largest pure RNN-T variant in the family, and among the most accurate.
On our LibriSpeech test-clean runs `parakeet-tdt-1.1b` edges it out (1.38% vs
1.46% Q8_0); RNN-T trades a little accuracy for the simpler transducer head.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`a07b19e`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/commit/a07b19e), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-rnnt-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-F32.gguf) | 4.28 GB | 1.45% |
| F16          | [parakeet-rnnt-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-F16.gguf) | 2.15 GB | 1.45% |
| Q8_0         | [parakeet-rnnt-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q8_0.gguf) | 1.27 GB | 1.46% |
| Q6_K         | [parakeet-rnnt-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q6_K.gguf) | 1.04 GB | 1.43% |
| Q5_K_M       | [parakeet-rnnt-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q5_K_M.gguf) |  936 MB | 1.43% |
| Q4_K_M       | [parakeet-rnnt-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-1.1b-gguf/resolve/main/parakeet-rnnt-1.1b-Q4_K_M.gguf) |  825 MB | 1.41% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy RNN-T decoding, no external LM. F32 reference baseline: 1.45%. NVIDIA's
self-reported number on the same split is 1.46%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.45% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  84 ms (131.63×) |  86 ms (128.14×) |
| Metal   | dots (35.3s) | 201 ms (175.51×) | 207 ms (170.62×) |
| CPU     | jfk (11.0s)  |  492 ms (22.37×) |  516 ms (21.34×) |
| CPU     | dots (35.3s) |  1.64 s (21.53×) |  1.86 s (19.04×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 701 ms (15.70×) | 710 ms (15.50×) |
| Vulkan  | dots (35.3s) | 2.15 s (16.47×) | 2.18 s (16.20×) |
| CPU     | jfk (11.0s)  |  1.12 s (9.80×) |  1.21 s (9.08×) |
| CPU     | dots (35.3s) |  4.79 s (7.38×) |  4.73 s (7.48×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-rnnt-1.1b
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
