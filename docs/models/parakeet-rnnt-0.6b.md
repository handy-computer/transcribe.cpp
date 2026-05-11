# Parakeet RNN-T 0.6B

NVIDIA's [`nvidia/parakeet-rnnt-0.6b`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b)
ported to transcribe.cpp. A 0.6B-parameter FastConformer-Large encoder with a
classic RNN-T transducer decoder (predictor + joint, no duration head).

## What it's for

Offline English speech-to-text with greedy RNN-T decoding. Output is
**lowercase, no punctuation** (per the upstream model card). Token- and
word-level timestamps are available. Not a streaming model; does not
translate.

The encoder shape matches `parakeet-tdt-0.6b-v2` (24 layers, 1024-d) but
the head is plain RNN-T rather than TDT — the joint network emits exactly
`vocab + 1` logits with no duration extras, and decoding has no
frame-skip choice. Per-frame iterative decode means RNN-T runs slower
than the CTC variant at the same encoder size.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-rnnt-0.6b)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`c0c1f09`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b/commit/c0c1f09fdc3f18b0b2ddbeafd5d6684f1b38078f),
pinned 2026-05-10.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-rnnt-0.6b-F32.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-F32.gguf) | 2.47 GB |                        1.62% |
| F16    | [parakeet-rnnt-0.6b-F16.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-F16.gguf) | 1.24 GB |                        1.62% |
| Q8_0   | [parakeet-rnnt-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q8_0.gguf) |  730 MB |                        1.62% |
| Q6_K   | [parakeet-rnnt-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q6_K.gguf) |  601 MB |                        1.62% |
| Q5_K_M | [parakeet-rnnt-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q5_K_M.gguf) |  540 MB |                        1.62% |
| Q4_K_M | [parakeet-rnnt-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-rnnt-0.6b-gguf/resolve/main/parakeet-rnnt-0.6b-Q4_K_M.gguf) |  476 MB |                        1.59% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy RNN-T decoding and no external LM. F32 reference baseline: 1.62%. NVIDIA's self-reported number on the same split is 1.63% (from the [HF model card](https://huggingface.co/nvidia/parakeet-rnnt-0.6b)).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places). Cells gated on `Tctl < 55°C` per backend.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |   85 ms (130×) |   86 ms (128×) |
| Metal   | dots (35.3s) |  258 ms (137×) |  264 ms (134×) |
| CPU     | jfk (11.0s)  |  382 ms (29×)  |  325 ms (34×)  |
| CPU     | dots (35.3s) | 1.30 s (27×)   | 1.11 s (32×)   |

macOS 26.4.1, transcribe.cpp `a6c097e`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  742 ms (15×) |  754 ms (15×) |
| Vulkan  | dots (35.3s) |  2.55 s (14×) |  2.59 s (14×) |
| CPU     | jfk (11.0s)  |  1.24 s (9×)  |  1.07 s (10×) |
| CPU     | dots (35.3s) |  4.71 s (7×)  |  4.14 s (9×)  |

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-rnnt-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-rnnt-0.6b-publication
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
| Reference | NeMo, `nvidia/parakeet-rnnt-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-rnnt-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-rnnt-0.6b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-rnnt-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-F32.gguf \
  models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-rnnt-0.6b
```
