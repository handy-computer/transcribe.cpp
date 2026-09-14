# Parakeet CTC 1.1B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-ctc-1.1b`](https://huggingface.co/nvidia/parakeet-ctc-1.1b) at [`a707e81`](https://huggingface.co/nvidia/parakeet-ctc-1.1b/commit/a707e81).

Offline English speech-to-text with greedy CTC decoding. A 1.1B-parameter FastConformer-XL encoder with a linear CTC head. Output is lowercase, no punctuation. Not a streaming model and does not translate.
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

Licensed CC-BY-4.0. Ported from upstream commit
[`a707e81`](https://huggingface.co/nvidia/parakeet-ctc-1.1b/commit/a707e818195cb97c8f7da2fc36b221a29f69a5db),
pinned 2026-05-10.

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
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  88 ms (125.48×) |  90 ms (121.85×) |
| Metal   | dots (35.3s) | 221 ms (160.02×) | 220 ms (160.38×) |
| CPU     | jfk (11.0s)  |  601 ms (18.30×) |  500 ms (22.01×) |
| CPU     | dots (35.3s) |  2.04 s (17.30×) |  1.70 s (20.83×) |

Apple M4 Max: transcribe.cpp `a6c097e` on 2026-05-10.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 826 ms (13.32×) | 823 ms (13.37×) |
| Vulkan  | dots (35.3s) | 2.34 s (15.13×) | 2.33 s (15.19×) |
| CPU     | jfk (11.0s)  |  1.75 s (6.27×) |  1.38 s (7.99×) |
| CPU     | dots (35.3s) |  6.08 s (5.81×) |  5.12 s (6.90×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `57997dc` on 2026-05-10.
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
