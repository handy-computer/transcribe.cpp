# Moonshine tiny

<!-- catalog:intro -->
Upstream: [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny) at [`390624e`](https://huggingface.co/UsefulSensors/moonshine-tiny/commit/390624e).

Useful Sensors Moonshine tiny — an encoder-decoder transformer for English
speech recognition. Consumes raw 16 kHz PCM directly via a three-layer Conv1d
stem (no STFT, no mel) and emits transcript-only output. English-only; no
translation, no language detection, no timestamps.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns
a transcript. Architecturally distinct from whisper: the frontend is a learned
conv stem on raw waveform rather than a log-mel spectrogram, and the decoder
emits transcript tokens only — no language tokens, no `<|translate|>`, no
timestamp tokens. English-only; no translation, no language detection, no
timestamps.

See the [upstream model card](https://huggingface.co/UsefulSensors/moonshine-tiny)
for training data, intended use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`390624e`](https://huggingface.co/UsefulSensors/moonshine-tiny/commit/390624e), pinned 2026-05-05. Validated against the transformers reference at transcribe.cpp commit [`07a8a84`](https://github.com/handy-computer/transcribe.cpp/tree/07a8a84) on 2026-05-05.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [moonshine-tiny-F32.gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-F32.gguf) | 110 MB | 4.58% |
| F16          | [moonshine-tiny-F16.gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-F16.gguf) |  59 MB | 4.58% |
| Q8_0         | [moonshine-tiny-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-Q8_0.gguf) |  35 MB | 4.60% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Decoded with the transcribe.cpp defaults (greedy, num_beams=1, max_length=194,
matching the upstream generation_config). Useful Sensors' self-reported number on
the same split is 4.55% (model card). Our F32 reference baseline lands at 4.58%,
within rounding of upstream and well within the ±1.00 pp Stage 7 acceptance gate.
Q8_0 drift is +0.02 pp vs F32 — within bootstrap CI noise.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| en       | WER    | 14.13% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-tiny/moonshine-tiny-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 56 ms (197.19×) |
| Metal   | dots (35.3s) | 422 ms (83.81×) |
| CPU     | jfk (11.0s)  | 54 ms (201.68×) |
| CPU     | dots (35.3s) | 373 ms (94.78×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |
| ------- | ------------ | --------------: |
| Vulkan  | jfk (11.0s)  | 132 ms (83.21×) |
| Vulkan  | dots (35.3s) | 938 ms (37.65×) |
| CPU     | jfk (11.0s)  | 175 ms (62.93×) |
| CPU     | dots (35.3s) | 1.79 s (19.71×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models moonshine-tiny
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference
(`MoonshineForConditionalGeneration`, fp32 CPU) on the manifest's case
(`samples/jfk.wav`). All checkpointed tensors fall within per-variant
tolerance. Tolerances are shared across moonshine-tiny and moonshine-base in
[`tests/tolerances/moonshine.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/moonshine.json)
(same correctness regime: F32 GGUF, F32 KV, raw PCM passthrough, CPU
single-thread).

| Field | Value |
| --- | --- |
| Reference | transformers 5.7.0 (`MoonshineForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/moonshine/moonshine-tiny.manifest.json` |
| Tolerance file | `tests/tolerances/moonshine.json` |
| Command | `uv run scripts/validate.py all --family moonshine --variant moonshine-tiny` |

The conv stem (kernel-127 stride-64 → tanh → GroupNorm → kernel-7 stride-3 →
GELU → kernel-3 stride-2 → GELU) drives `enc.conv1.out` and downstream
encoder gates to fp32 reduction-order noise (1e-6 to 1e-4); the partial-RoPE
self-attn and SwiGLU decoder MLP land in the same regime. KV cache runs F32
to match the F32 weights — flip with `--kv-type f16` if you want a tighter
memory footprint.

## Reproduction

### Convert

The Moonshine converter loads from a Hugging Face checkpoint and emits a
reference-dtype (F32) GGUF.

```bash
uv run --project scripts/envs/moonshine \
  scripts/convert-moonshine.py UsefulSensors/moonshine-tiny \
  --revision 390624ed33d594443aa4aa221f5b9f283b545b5a
```

### Quantize

`scripts/quantize-all.py` reads the per-architecture preset matrix from
`scripts/lib/quant_policy.py`. For moonshine that is `("F16", "Q8_0")` —
the K-tier presets are skipped at this size (see Download note).

```bash
uv run scripts/quantize-all.py models/moonshine-tiny/moonshine-tiny-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine --variant moonshine-tiny
```

### Score WER

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-tiny/moonshine-tiny-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-tiny-F32.librispeech-test-clean.jsonl

uv run scripts/wer/score.py \
  reports/wer/moonshine-tiny-F32.librispeech-test-clean.jsonl
```
