# Parakeet CTC 0.6B

NVIDIA's [`nvidia/parakeet-ctc-0.6b`](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
ported to transcribe.cpp. A 0.6B-parameter FastConformer-Large encoder with a
linear CTC head — the simplest decoder in the parakeet family and therefore
the fastest.

## What it's for

Offline English speech-to-text with greedy CTC decoding. Output is
**lowercase, no punctuation** (the upstream model card explicitly notes
"lower case English alphabet"). Token- and word-level timestamps are
available. Not a streaming model; does not translate.

The encoder is identical in shape to `parakeet-tdt-0.6b-v2` (24 layers,
1024-d), so this variant is the fastest 0.6B-class option in the family
on this codebase.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`ad09ba1`](https://huggingface.co/nvidia/parakeet-ctc-0.6b/commit/ad09ba1cc62743fbc9814de5d2016fca9096485a),
pinned 2026-05-09.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-ctc-0.6b-F32.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-F32.gguf) | 2.44 GB |                        1.87% |
| F16    | [parakeet-ctc-0.6b-F16.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-F16.gguf) | 1.22 GB |                        1.87% |
| Q8_0   | [parakeet-ctc-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q8_0.gguf) |  722 MB |                        1.87% |
| Q6_K   | [parakeet-ctc-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q6_K.gguf) |  594 MB |                        1.84% |
| Q5_K_M | [parakeet-ctc-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q5_K_M.gguf) |  533 MB |                        1.87% |
| Q4_K_M | [parakeet-ctc-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ctc-0.6b-gguf/resolve/main/parakeet-ctc-0.6b-Q4_K_M.gguf) |  469 MB |                        1.90% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy CTC decoding and no external LM. F32 reference baseline: 1.87%. NVIDIA's self-reported number on the same split is 1.87% (from the [HF model card](https://huggingface.co/nvidia/parakeet-ctc-0.6b)).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-Q8_0.gguf \
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

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  520 ms (21×) |  537 ms (20×) |
| Vulkan  | dots (35.3s) |  1.50 s (24×) |  1.50 s (24×) |
| CPU     | jfk (11.0s)  |  1.07 s (10×) |  863 ms (13×) |
| CPU     | dots (35.3s) |  3.67 s (10×) |  3.14 s (11×) |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-ctc-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-ctc-0.6b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`
via `scripts/validate.py`, sharing the parakeet family tolerance file. The
encoder shape is identical to `parakeet-tdt-0.6b-v2` and uses the same
FastConformer code path; the family-level forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources (fp64 STFT, mel amplification,
attenuation through the encoder).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-ctc-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-ctc-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-ctc-0.6b` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-ctc-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-F32.gguf \
  models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-ctc-0.6b
```
