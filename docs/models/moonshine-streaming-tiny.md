# Moonshine Streaming Tiny

<!-- catalog:intro -->
Upstream: [`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) at [`f8e9dfd`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny/commit/f8e9dfd).

English speech-to-text in both one-shot and streaming modes. An
encoder-decoder ASR model designed for streaming use (ergodic encoder +
sliding-window attention, 50 Hz time-domain frontend). Takes a 16 kHz mono WAV
and produces a transcript. No translation, no multilingual capability, no
timestamps.
<!-- /catalog -->

## What it's for

English speech-to-text in both one-shot and streaming modes. The model takes a
16 kHz mono WAV and produces a transcript. It does not translate, has no
multilingual capability, and does not emit timestamps.

See Useful Sensors' [model card](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`f8e9dfd`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny/commit/f8e9dfd), pinned 2026-05-06. Validated against the HF Transformers v5.7.0 reference at transcribe.cpp commit [`0d312ce`](https://github.com/handy-computer/transcribe.cpp/tree/0d312ce) on 2026-05-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [moonshine-streaming-tiny-F32.gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-F32.gguf) | 178 MB | 4.53% |
| F16          | [moonshine-streaming-tiny-F16.gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-F16.gguf) |  90 MB | 4.53% |
| Q8_0         | [moonshine-streaming-tiny-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-Q8_0.gguf) |  50 MB | 4.52% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances). Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding (`num_beams=1`, `do_sample=False`). F32 reference baseline: 4.53%.
The HF Transformers reference scored on the same manifest in the same regime lands
at 4.52% with 99.6% byte-identical hypotheses to our F32, so the port is at exact
parity with the reference. Useful Sensors' self-reported number on this split is
4.49% from the Open ASR Leaderboard table; the +0.04pp residual is a scoring /
text-normalization difference vs that methodology, not a numerical drift in the
port. Q6_K / Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| en       | WER    | 18.18% |
<!-- /catalog -->

Q6_K / Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.

### Streaming vs offline parity (Q8_0)

The streaming session API (`transcribe_stream_begin/feed/finalize`)
produces a final transcript at parity with the offline one-shot path
on Q8_0:

| Mode | WER | Sub / Del / Ins | CLI errors |
| --- | ---: | ---: | ---: |
| Offline (one-shot)                | **4.52%** | 1764 / 250 / 383 | 0 |
| Streaming `--stream-chunk-ms 500` | **4.54%** | 1772 / 252 / 381 | 0 |

The +0.02 pp delta is 8 extra word errors across ~52 000 reference
words, sitting comfortably inside the 95% confidence interval overlap
([4.22%, 4.88%] vs [4.22%, 4.90%]) and well below any reasonable
regression threshold. The residual comes from PCM trimming during
streaming producing tiny float-precision differences in the encoder
output at the left-context boundary, which occasionally flip a single
argmax late in the AR decode. Reports:
`reports/wer/moonshine-streaming-tiny-Q8_0.test-clean.{offline,stream-500ms}.score.json`.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf \
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

| Backend | Sample       |             Q8_0 |
| ------- | ------------ | ---------------: |
| Metal   | jfk (11.0s)  |  46 ms (239.48×) |
| Metal   | dots (35.3s) | 316 ms (111.87×) |
| CPU     | jfk (11.0s)  |  41 ms (270.63×) |
| CPU     | dots (35.3s) | 210 ms (168.59×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |
| ------- | ------------ | --------------: |
| Vulkan  | jfk (11.0s)  | 138 ms (80.00×) |
| Vulkan  | dots (35.3s) | 922 ms (38.30×) |
| CPU     | jfk (11.0s)  | 146 ms (75.29×) |
| CPU     | dots (35.3s) | 948 ms (37.25×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models moonshine-streaming-tiny
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the HF Transformers
reference (`MoonshineStreamingForConditionalGeneration`, fp32 inference,
`attn_implementation="eager"`) on `samples/jfk.wav`. All 37 contract
tensors fall within family tolerance, and the final transcript matches the
reference.

| Field | Value |
| --- | --- |
| Reference | HF Transformers v5.7.0, `UsefulSensors/moonshine-streaming-tiny` |
| Dump script | `scripts/dump_reference_moonshine_streaming_transformers.py` |
| Manifest | `tests/golden/moonshine_streaming/moonshine-streaming-tiny.manifest.json` |
| Command | `uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-tiny` |

The dominant drift source is BLAS reduction-order differences between
PyTorch's matmul kernels and ggml's `mul_mat` (Accelerate / Metal /
ggml-cpu). Drift accumulates roughly linearly with depth across encoder +
adapter + decoder; the final logit budget stays well below 1e-3 absolute /
1e-4 mean.

## Reproduction

### Convert

Loads directly from Useful Sensors' Hugging Face repo via
`AutoProcessor` + `MoonshineStreamingForConditionalGeneration`. Output path
is derived from the repo id.

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/convert-moonshine_streaming.py UsefulSensors/moonshine-streaming-tiny
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`:

```bash
build/bin/transcribe-quantize \
  models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf \
  models/moonshine-streaming-tiny/moonshine-streaming-tiny-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-tiny
```

### WER sweep

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-tiny-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-tiny-F32.test-clean.jsonl
```
