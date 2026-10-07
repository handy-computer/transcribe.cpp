# Parakeet CTC 1.1B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b) at [`a707e81`](https://huggingface.co/nvidia/parakeet-ctc-1.1b/commit/a707e81).

Offline English speech-to-text with greedy CTC decoding. A FastConformer-XL encoder with a linear CTC head. Output is lowercase, no punctuation. Not a streaming model and does not translate.
<!-- /catalog -->

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

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`a707e81`](https://huggingface.co/nvidia/parakeet-ctc-1.1b/commit/a707e81), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-ctc-1.1b-F32.gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf/resolve/main/parakeet-ctc-1.1b-F32.gguf) | 4.25 GB | 1.85% |
| F16          | [parakeet-ctc-1.1b-F16.gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf/resolve/main/parakeet-ctc-1.1b-F16.gguf) | 2.13 GB | 1.85% |
| Q8_0         | [parakeet-ctc-1.1b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf/resolve/main/parakeet-ctc-1.1b-Q8_0.gguf) | 1.26 GB | 1.85% |
| Q6_K         | [parakeet-ctc-1.1b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf/resolve/main/parakeet-ctc-1.1b-Q6_K.gguf) | 1.04 GB | 1.85% |
| Q5_K_M       | [parakeet-ctc-1.1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf/resolve/main/parakeet-ctc-1.1b-Q5_K_M.gguf) |  929 MB | 1.84% |
| Q4_K_M       | [parakeet-ctc-1.1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ctc-1.1b-gguf/resolve/main/parakeet-ctc-1.1b-Q4_K_M.gguf) |  818 MB | 1.90% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy CTC decoding, no external LM. F32 reference baseline: 1.85%. NVIDIA's
self-reported number on the same split is 1.83%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 5.61% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  79 ms (140.04×) |  80 ms (136.77×) |
| Metal   | dots (35.3s) | 177 ms (200.12×) | 180 ms (195.92×) |
| CPU     | jfk (11.0s)  |  458 ms (24.00×) |  502 ms (21.92×) |
| CPU     | dots (35.3s) |  1.58 s (22.32×) |  1.70 s (20.80×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 691 ms (15.92×) | 711 ms (15.47×) |
| Vulkan  | dots (35.3s) | 2.00 s (17.65×) | 2.02 s (17.49×) |
| CPU     | jfk (11.0s)  |  1.10 s (9.97×) |  1.21 s (9.06×) |
| CPU     | dots (35.3s) |  4.48 s (7.88×) |  4.63 s (7.63×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-ctc-1.1b
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
