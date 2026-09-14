# Parakeet RNN-T 1.1B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-rnnt-1.1b`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) at [`a07b19e`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/commit/a07b19e).

Offline English speech-to-text with greedy RNN-T decoding. A 1.1B-parameter FastConformer-XL encoder with an RNN-T transducer decoder. Output is lowercase, no punctuation. Not a streaming model and does not translate.
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

Licensed CC-BY-4.0. Ported from upstream commit
[`a07b19e`](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/commit/a07b19e9),
pinned 2026-05-10.

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
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |           Q8_0 |          Q4_K_M |
| ------- | ------------ | -------------: | --------------: |
| Metal   | jfk (11.0s)  |  96 ms (114×)† |   97 ms (114×)† |
| Metal   | dots (35.3s) | 258 ms (137×)† |  265 ms (133×)† |
| CPU     | jfk (11.0s)  |  606 ms (18×)† |   506 ms (22×)† |
| CPU     | dots (35.3s) |  2.05 s (17×)† | 1.86 s (19.01×) |

Apple M4 Max: transcribe.cpp `a6c097e` on 2026-05-10. † published before provenance was recorded; not yet re-measured.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 1.01 s (10.85×) | 1.04 s (10.60×) |
| Vulkan  | dots (35.3s) | 3.34 s (10.58×) | 3.30 s (10.70×) |
| CPU     | jfk (11.0s)  |  1.93 s (5.70×) |  1.58 s (6.98×) |
| CPU     | dots (35.3s) |  7.12 s (4.96×) |  6.18 s (5.72×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `12f1076` on 2026-05-11.
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
