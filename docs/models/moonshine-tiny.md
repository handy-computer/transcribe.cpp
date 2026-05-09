# Moonshine tiny

Useful Sensors' [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny)
ported to transcribe.cpp. A 27M-parameter encoder-decoder transformer that
consumes raw 16 kHz PCM directly (no STFT, no mel filterbank) via a three-layer
Conv1d stem.

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns
a transcript. Architecturally distinct from whisper: the frontend is a learned
conv stem on raw waveform rather than a log-mel spectrogram, and the decoder
emits transcript tokens only — no language tokens, no `<|translate|>`, no
timestamp tokens. English-only; no translation, no language detection, no
timestamps.

See the [upstream model card](https://huggingface.co/UsefulSensors/moonshine-tiny)
for training data, intended use, and the original evaluation methodology.

Licensed MIT. Ported from upstream commit
[`390624e`](https://huggingface.co/UsefulSensors/moonshine-tiny/commit/390624ed33d594443aa4aa221f5b9f283b545b5a),
pinned 2026-05-05. Validated against the transformers reference at
transcribe.cpp commit
[`07a8a84`](https://github.com/handy-computer/transcribe.cpp/tree/07a8a84)
on 2026-05-05.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32  | [moonshine-tiny-F32.gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-F32.gguf)   | 105 MB | 4.58% |
| F16  | [moonshine-tiny-F16.gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-F16.gguf)   |  57 MB | 4.58% |
| Q8_0 | [moonshine-tiny-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-Q8_0.gguf) |  34 MB | 4.60% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with
the transcribe.cpp default decode (greedy, `num_beams=1`, `max_length=194` —
matching the upstream `generation_config`). Useful Sensors' self-reported
number on the same split is 4.55% (model card). Our F32 reference baseline
lands at 4.58%, within rounding of upstream and well within the ±1.00 pp
Stage 7 acceptance gate. Q8_0 drift is +0.02 pp vs F32 — within bootstrap CI
noise. Only F16 and Q8_0 are shipped as derived presets: at moonshine-tiny's
shapes (hidden 288, intermediate 1152, vocab 32768) none of the dimensions
divide the k-quant super-block size of 256, so Q6_K / Q5_K_M / Q4_K_M would
all fall back to Q8_0 storage and be near-duplicates.

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

Cells are wall-clock latency (mean over 5 iterations after 2 warmups),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above
(2 decimal places).

### Apple M4 Max

| Backend | Sample       |         Q8_0 |
| ------- | ------------ | -----------: |
| Metal   | jfk (11.0s)  |  61 ms (180×) |
| Metal   | dots (35.3s) | 478 ms (74×)  |
| CPU     | jfk (11.0s)  |  52 ms (210×) |
| CPU     | dots (35.3s) | 366 ms (97×)  |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |
| ------- | ------------ | ------------: |
| Vulkan  | jfk (11.0s)  |  143 ms (77×) |
| Vulkan  | dots (35.3s) | 1.02 s (35×)  |
| CPU     | jfk (11.0s)  |  163 ms (68×) |
| CPU     | dots (35.3s) | 1.53 s (23×)  |

Fedora 43, transcribe.cpp `f68eb15`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models moonshine-tiny \
  --quants q8_0 \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 5 --warmup 2 \
  --name moonshine-publication
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
