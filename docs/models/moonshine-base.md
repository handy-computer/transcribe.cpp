# Moonshine base

<!-- catalog:intro -->
Upstream: [`UsefulSensors/moonshine-base`](https://huggingface.co/UsefulSensors/moonshine-base) at [`7a73d8d`](https://huggingface.co/UsefulSensors/moonshine-base/commit/7a73d8d).

Useful Sensors Moonshine base — an encoder-decoder transformer for English
speech recognition. Consumes raw 16 kHz PCM directly via a three-layer Conv1d
stem (no STFT, no mel) and emits transcript-only output. Wider and deeper
than moonshine-tiny (8 encoder / 8 decoder layers, hidden size 416, partial
RoPE 0.62). English-only; no translation, no language detection, no
timestamps.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns
a transcript. Same architecture family as moonshine-tiny — raw-waveform conv
stem frontend, partial-RoPE attention, SwiGLU decoder MLP — scaled up. Decoder
emits transcript tokens only: no language tokens, no `<|translate|>`, no
timestamp tokens. English-only; no translation, no language detection, no
timestamps.

See the [upstream model card](https://huggingface.co/UsefulSensors/moonshine-base)
for training data, intended use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`7a73d8d`](https://huggingface.co/UsefulSensors/moonshine-base/commit/7a73d8d), pinned 2026-05-05. Validated against the transformers reference at transcribe.cpp commit [`07a8a84`](https://github.com/handy-computer/transcribe.cpp/tree/07a8a84) on 2026-05-05.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [moonshine-base-F32.gguf](https://huggingface.co/handy-computer/moonshine-base-gguf/resolve/main/moonshine-base-F32.gguf) | 248 MB | 3.28% |
| F16          | [moonshine-base-F16.gguf](https://huggingface.co/handy-computer/moonshine-base-gguf/resolve/main/moonshine-base-F16.gguf) | 132 MB | 3.28% |
| Q8_0         | [moonshine-base-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-base-gguf/resolve/main/moonshine-base-Q8_0.gguf) |  77 MB | 3.26% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Decoded with the transcribe.cpp defaults (greedy, num_beams=1, max_length=194,
matching the upstream generation_config). Upstream reports 3.27% on the same split
(Moonshine paper, Table 2; also Open ASR Leaderboard). Our F32 reference baseline
lands at 3.28%, identical to upstream within rounding and well within the ±1.00 pp
Stage 7 acceptance gate. Q8_0 lands at 3.26%, slightly under F32 — that delta sits
inside the 95% bootstrap CI and is noise, not a real improvement.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| en       | WER    | 12.25% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-base/moonshine-base-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  89 ms (123.33×) |
| Metal   | dots (35.3s) |  740 ms (47.76×) |
| CPU     | jfk (11.0s)  | 100 ms (109.83×) |
| CPU     | dots (35.3s) |  690 ms (51.17×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |
| ------- | ------------ | --------------: |
| Vulkan  | jfk (11.0s)  | 216 ms (50.90×) |
| Vulkan  | dots (35.3s) | 1.74 s (20.28×) |
| CPU     | jfk (11.0s)  | 306 ms (35.95×) |
| CPU     | dots (35.3s) | 3.20 s (11.03×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models moonshine-base
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference
(`MoonshineForConditionalGeneration`, fp32 CPU) on the manifest's case
(`samples/jfk.wav`). All checkpointed tensors fall within per-variant
tolerance. Tolerances are shared across moonshine-tiny and moonshine-base in
[`tests/tolerances/moonshine.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/moonshine.json)
— both variants run in the same correctness regime, only depth differs (base
adds gate entries for `enc/dec.block.{6,7}`).

| Field | Value |
| --- | --- |
| Reference | transformers 5.7.0 (`MoonshineForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/moonshine/moonshine-base.manifest.json` |
| Tolerance file | `tests/tolerances/moonshine.json` |
| Command | `uv run scripts/validate.py all --family moonshine --variant moonshine-base` |

The conv stem (kernel-127 stride-64 → tanh → GroupNorm → kernel-7 stride-3 →
GELU → kernel-3 stride-2 → GELU) drives `enc.conv1.out` and downstream
encoder gates to fp32 reduction-order noise (1e-6 to 1e-4); the partial-RoPE
self-attn (factor 0.62 — narrower than tiny's 0.9) and SwiGLU decoder MLP
land in the same regime. KV cache runs F32 to match the F32 weights — flip
with `--kv-type f16` if you want a tighter memory footprint.

## Reproduction

### Convert

The Moonshine converter loads from a Hugging Face checkpoint and emits a
reference-dtype (F32) GGUF.

```bash
uv run --project scripts/envs/moonshine \
  scripts/convert-moonshine.py UsefulSensors/moonshine-base \
  --revision 7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf
```

### Quantize

`scripts/quantize-all.py` reads the per-architecture preset matrix from
`scripts/lib/quant_policy.py`. For moonshine that is `("F16", "Q8_0")` —
the K-tier presets are skipped at this size (see Download note).

```bash
uv run scripts/quantize-all.py models/moonshine-base/moonshine-base-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine --variant moonshine-base
```

### Score WER

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-base/moonshine-base-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-base-F32.librispeech-test-clean.jsonl

uv run scripts/wer/score.py \
  reports/wer/moonshine-base-F32.librispeech-test-clean.jsonl
```
