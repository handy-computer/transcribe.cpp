# Moonshine Streaming Small

<!-- catalog:intro -->
Upstream: [`UsefulSensors/moonshine-streaming-small`](https://huggingface.co/UsefulSensors/moonshine-streaming-small) at [`2c03650`](https://huggingface.co/UsefulSensors/moonshine-streaming-small/commit/2c03650).

English speech-to-text in both one-shot and streaming modes. An encoder-decoder
ASR model designed for streaming use (ergodic encoder + sliding-window
attention, 50 Hz time-domain frontend). Same family as
moonshine-streaming-tiny; deeper encoder/decoder (10 / 10 layers) and wider
hidden dims (encoder 620 / decoder 512). Takes a 16 kHz mono WAV and produces a
transcript. No translation, no multilingual capability, no timestamps.
<!-- /catalog -->

## What it's for

English speech-to-text in both one-shot and streaming modes. The model takes a
16 kHz mono WAV and produces a transcript. It does not translate, has no
multilingual capability, and does not emit timestamps.

See Useful Sensors' [model card](https://huggingface.co/UsefulSensors/moonshine-streaming-small)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`2c03650`](https://huggingface.co/UsefulSensors/moonshine-streaming-small/commit/2c03650), pinned 2026-05-06. Validated against the HF Transformers v5.7.0 reference at transcribe.cpp commit [`0d312ce`](https://github.com/handy-computer/transcribe.cpp/tree/0d312ce) on 2026-05-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [moonshine-streaming-small-F32.gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf/resolve/main/moonshine-streaming-small-F32.gguf) | 562 MB | 2.53% |
| F16          | [moonshine-streaming-small-F16.gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf/resolve/main/moonshine-streaming-small-F16.gguf) | 282 MB | 2.53% |
| Q8_0         | [moonshine-streaming-small-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf/resolve/main/moonshine-streaming-small-Q8_0.gguf) | 199 MB | 2.54% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances). Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding (`num_beams=1`, `do_sample=False`). F32 reference baseline: 2.53%.
Useful Sensors' self-reported number on this split is 2.49% from the Open ASR
Leaderboard table; the +0.04pp residual matches the same scoring /
text-normalization difference seen on the tiny variant where we cross-checked
against the HF Transformers reference (4.52% on the same manifest, 99.6% identical
hypotheses to our F32) and confirmed it is not a numerical drift in the port. Q6_K /
Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 8.55% |
<!-- /catalog -->

Q6_K / Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-streaming-small/moonshine-streaming-small-Q8_0.gguf \
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

| Backend | Sample       |            Q8_0 |
| ------- | ------------ | --------------: |
| Metal   | jfk (11.0s)  | 85 ms (129.36×) |
| Metal   | dots (35.3s) | 603 ms (58.59×) |
| CPU     | jfk (11.0s)  | 172 ms (63.75×) |
| CPU     | dots (35.3s) | 722 ms (48.91×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |
| ------- | ------------ | --------------: |
| Vulkan  | jfk (11.0s)  | 374 ms (29.38×) |
| Vulkan  | dots (35.3s) | 2.50 s (14.16×) |
| CPU     | jfk (11.0s)  | 620 ms (17.75×) |
| CPU     | dots (35.3s) |  3.79 s (9.32×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models moonshine-streaming-small
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the HF Transformers
reference (`MoonshineStreamingForConditionalGeneration`, fp32 inference,
`attn_implementation="eager"`) on `samples/jfk.wav`. All contract tensors
fall within family tolerance, and the final transcript matches the
reference.

| Field | Value |
| --- | --- |
| Reference | HF Transformers v5.7.0, `UsefulSensors/moonshine-streaming-small` |
| Dump script | `scripts/dump_reference_moonshine_streaming_transformers.py` |
| Manifest | `tests/golden/moonshine_streaming/moonshine-streaming-small.manifest.json` |
| Command | `uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-small` |

Tolerances are recorded at family scope in
`tests/tolerances/moonshine_streaming.json`. The dominant drift source is
BLAS reduction-order differences between PyTorch's matmul kernels and
ggml's `mul_mat` (Accelerate / Metal / ggml-cpu). Drift accumulates
roughly linearly with depth across the 10-layer encoder, the adapter, and
the 10-layer decoder; the final logit budget stays well below 1e-3
absolute / 1e-4 mean.

Small additionally exercises the **non-square encoder attention** path —
encoder residual dim 620, attention dim 512 — that the original
moonshine-streaming-tiny port did not separate (320 = 8 × 40 there). Q/K/V
project residual_dim → attn_dim and O projects attn_dim → residual_dim;
the C++ port carries both shapes through the encoder block.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/convert-moonshine_streaming.py UsefulSensors/moonshine-streaming-small
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/moonshine-streaming-small/moonshine-streaming-small-F32.gguf \
  models/moonshine-streaming-small/moonshine-streaming-small-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-small
```

### WER sweep

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-streaming-small/moonshine-streaming-small-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-small-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-small-F32.test-clean.jsonl
```
