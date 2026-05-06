# Moonshine base

Useful Sensors' [`UsefulSensors/moonshine-base`](https://huggingface.co/UsefulSensors/moonshine-base)
ported to transcribe.cpp. A 62M-parameter encoder-decoder transformer that
consumes raw 16 kHz PCM directly (no STFT, no mel filterbank) via a three-layer
Conv1d stem. Wider and deeper than moonshine-tiny (8 encoder / 8 decoder
layers, hidden size 416, intermediate 1664, partial RoPE 0.62).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns
a transcript. Same architecture family as moonshine-tiny — raw-waveform conv
stem frontend, partial-RoPE attention, SwiGLU decoder MLP — scaled up. Decoder
emits transcript tokens only: no language tokens, no `<|translate|>`, no
timestamp tokens. English-only; no translation, no language detection, no
timestamps.

See the [upstream model card](https://huggingface.co/UsefulSensors/moonshine-base)
for training data, intended use, and the original evaluation methodology.

Licensed MIT. Ported from upstream commit
[`7a73d8d`](https://huggingface.co/UsefulSensors/moonshine-base/commit/7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf),
pinned 2026-05-05. Validated against the transformers reference at
transcribe.cpp commit
[`07a8a84`](https://github.com/handy-computer/transcribe.cpp/tree/07a8a84)
on 2026-05-05.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32  | [moonshine-base-F32.gguf](https://huggingface.co/handy-computer/moonshine-base-gguf/resolve/main/moonshine-base-F32.gguf)   | 236 MB | 3.28% |
| F16  | [moonshine-base-F16.gguf](https://huggingface.co/handy-computer/moonshine-base-gguf/resolve/main/moonshine-base-F16.gguf)   | 126 MB | 3.28% |
| Q8_0 | [moonshine-base-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-base-gguf/resolve/main/moonshine-base-Q8_0.gguf) |  74 MB | 3.26% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with
the transcribe.cpp default decode (greedy, `num_beams=1`, `max_length=194` —
matching the upstream `generation_config`). Upstream reports 3.27% on the same
split (Moonshine paper, Table 2; also Open ASR Leaderboard). Our F32 reference
baseline lands at 3.28%, identical to upstream within rounding and well within
the ±1.00 pp Stage 7 acceptance gate. Q8_0 lands at 3.26%, slightly under F32
— that delta sits inside the 95% bootstrap CI and is noise, not a real
improvement. Only F16 and Q8_0 are shipped as derived presets: at
moonshine-base's shapes (hidden 416, intermediate 1664, vocab 32768) none of
the dimensions divide the k-quant super-block size of 256, so Q6_K / Q5_K_M /
Q4_K_M would all fall back to Q8_0 storage and be near-duplicates.

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

Per-quant summary at
[`reports/wer/moonshine-base.librispeech-test-clean.summary.md`](https://github.com/handy-computer/transcribe.cpp/blob/main/reports/wer/moonshine-base.librispeech-test-clean.summary.md).
