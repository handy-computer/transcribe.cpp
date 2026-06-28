# Cohere Transcribe 03-2026

Cohere's [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
ported to transcribe.cpp. A Conformer encoder with a Transformer encoder-decoder
head (cross-attention, tied token embedding).

## What it's for

Offline multilingual speech-to-text covering 14 languages: English, French,
German, Spanish, Italian, Portuguese, Dutch, Polish, Greek, Arabic, Japanese,
Chinese, Vietnamese, Korean. The model takes a 16 kHz mono WAV and produces a
transcript. Cohere uses an encoder-decoder with cross-attention, decoding is
autoregressive.

See Cohere's [model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`76b8b23`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/commit/76b8b23e8607f35f0265a23d481b338fb0e26aea),
pinned 2026-04-16.

## Input limits

Accepts up to about **6.7 minutes (400 s)** of 16 kHz mono audio per call — the
encoder's positional table is the binding limit. Longer audio is rejected up
front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated; split
it into shorter segments. See the [input-length contract](../input-limits.md).

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [cohere-transcribe-03-2026-BF16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-BF16.gguf)     | 4.10 GB | 1.26% |
| F16    | [cohere-transcribe-03-2026-F16.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-F16.gguf)       | 4.11 GB | 1.26% |
| Q8_0   | [cohere-transcribe-03-2026-Q8_0.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q8_0.gguf)     | 2.41 GB | 1.27% |
| Q6_K   | [cohere-transcribe-03-2026-Q6_K.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q6_K.gguf)     | 1.97 GB | 1.27% |
| Q5_K_M | [cohere-transcribe-03-2026-Q5_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q5_K_M.gguf) | 1.76 GB | 1.25% |
| Q4_K_M | [cohere-transcribe-03-2026-Q4_K_M.gguf](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q4_K_M.gguf) | 1.55 GB | 1.25% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding and no external LM. BF16 reference baseline: 1.26%.
Cohere's self-reported number on the same split is 1.25% (Open ASR Leaderboard,
as of 2026-03-26). Text normalizer: Whisper `EnglishTextNormalizer` — the same
normalizer the Open ASR Leaderboard uses, so the comparison is apples-to-apples.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-F16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  150 ms (74×)  |  154 ms (71×)  |
| Metal   | dots (35.3s) |  491 ms (72×)  |  465 ms (76×)  |
| CPU     | jfk (11.0s)  | 1.21 s (9×)    | 1.05 s (11×)   |
| CPU     | dots (35.3s) | 4.13 s (9×)    | 3.49 s (10×)   |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  1.43 s (8×)   |  1.33 s (8×)   |
| Vulkan  | dots (35.3s) |  4.25 s (8×)   |  4.25 s (8×)   |
| CPU     | jfk (11.0s)  |  3.57 s (3×)   |  2.90 s (4×)   |
| CPU     | dots (35.3s) | 12.40 s (3×)   | 10.08 s (4×)   |

Fedora 43, transcribe.cpp `2ab01b8`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models cohere-transcribe-03-2026 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name cohere-transcribe-03-2026-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the Transformers
reference implementation on `samples/jfk.wav`. All 22 checkpointed tensors
fall within family tolerance, and the final transcript matches the reference
verbatim. Last validated at commit
[`bf0d0b7`](https://github.com/handy-computer/transcribe.cpp/tree/bf0d0b7).

| Field | Value |
| --- | --- |
| Reference | Transformers, `CohereLabs/cohere-transcribe-03-2026` |
| Dump script | `scripts/dump_reference_cohere_transformers.py` |
| Manifest | `tests/golden/cohere/cohere-transcribe-03-2026.manifest.json` |
| Command | `uv run scripts/validate.py compare --family cohere` |

Selected tensors:

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`          | `2.678e-01` | `4.238e-03` | fp64 vs fp32 STFT precision gap |
| `enc.pre_encode.out`  | `8.794e+00` | `3.049e-02` | Mel gap propagated through pre-encoder |
| `enc.block.0.out`     | `4.542e+00` | `1.939e-02` | Early encoder |
| `enc.block.23.out`    | `3.173e+00` | `3.118e-02` | Mid-encoder |
| `enc.block.47.out`    | `2.039e-01` | `4.615e-03` | Final encoder block |
| `enc.final`           | `2.039e-01` | `4.615e-03` | Encoder output |
| `enc_dec_proj.out`    | `3.693e-01` | `1.074e-02` | Encoder→decoder projection |
| `dec.token_emb`       | `2.980e-08` | `2.910e-12` | Exact within fp32 round-off |
| `dec.pos_emb`         | `0.000e+00` | `0.000e+00` | Exact |
| `dec.embed_norm`      | `1.241e-01` | `1.327e-03` | LayerNorm output |
| `dec.block.0.out`     | `3.409e+00` | `2.491e-02` | Early decoder |
| `dec.block.7.out`     | `3.177e+01` | `1.416e-01` | Final decoder block (accumulated) |
| `dec.out_before_head` | `2.465e-01` | `1.339e-02` | Pre-head projection |
| `dec.logits_raw`      | `6.675e-01` | `3.328e-02` | Raw logits |
| `dec.logits`          | `nan`       | `nan`       | Softmax: `-inf` entries produce `nan`; first diff index 114692 is masked |

The expected divergence is in the frontend: C++ runs the STFT in fp64 where
the reference runs fp32. The gap enters at the mel spectrogram, propagates
through the encoder, and attenuates to a few tenths by the final encoder
block. Decoder numerics track the reference within fp32 round-off on the
first autoregressive step.

## Reproduction

### Convert

Downloads the upstream HF repo via `huggingface-cli` (or an existing local
clone) and converts with the family-specific script. Output path is derived
from the repo id.

```bash
uv run --project scripts/envs/cohere \
  scripts/convert-cohere.py CohereLabs/cohere-transcribe-03-2026
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf \
  models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family cohere
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_COHERE_GGUF=models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf \
  ctest --test-dir build --output-on-failure -R 'cohere'
```
