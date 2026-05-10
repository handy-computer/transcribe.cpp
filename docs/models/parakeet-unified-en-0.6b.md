# Parakeet Unified EN 0.6B

NVIDIA's [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
ported to transcribe.cpp. A 0.6B-parameter FastConformer encoder with an
RNN-T transducer decoder, trained as a "unified" streaming/offline model.

## What it's for

English speech-to-text with greedy RNN-T decoding. Outputs cased,
punctuated transcripts. Token- and word-level timestamps are available.
This port runs the model in **offline** mode only — the streaming
attention contexts in the upstream encoder are present in the GGUF but
transcribe.cpp does not yet expose a streaming entry point.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
for training data, intended use, streaming methodology, and upstream
evaluation results.

Licensed CC-BY-4.0. Ported from upstream commit
[`d4ac992`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b/commit/d4ac9928),
pinned 2026-05-09.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean, offline) |
| --- | --- | ---: | ---: |
| F32    | parakeet-unified-en-0.6b-F32.gguf    | 2.47 GB | — |
| Q8_0   | parakeet-unified-en-0.6b-Q8_0.gguf   | 731 MB  | — |
| Q4_K_M | parakeet-unified-en-0.6b-Q4_K_M.gguf | 477 MB  | — |

WER will be measured on the full LibriSpeech test-clean split (2620
utterances) with greedy RNN-T decoding and no external LM, in offline
mode. NVIDIA reports 1.63% offline WER on the same split (from the
[HF model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)).
Per-quant numbers will be filled in once the reference-machine WER sweep
completes.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf \
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
| Vulkan  | jfk (11.0s)  |  1.47 s (7×)  |  1.49 s (7×)  |
| Vulkan  | dots (35.3s) |  5.15 s (7×)  |  5.14 s (7×)  |
| CPU     | jfk (11.0s)  |  1.99 s (6×)  |  1.83 s (6×)  |
| CPU     | dots (35.3s) |  7.24 s (5×)  |  6.69 s (5×)  |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-unified-en-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-unified-en-0.6b-publication
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
| Reference | NeMo, `nvidia/parakeet-unified-en-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-unified-en-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-unified-en-0.6b` |

## Known limitations

- Streaming decoding is not exposed by transcribe.cpp. The model is
  loaded with full-utterance attention and is benchmarked / measured in
  offline mode only.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-unified-en-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf \
  models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-unified-en-0.6b
```
