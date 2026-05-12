# Nemotron Speech Streaming EN 0.6B

NVIDIA's [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
ported to transcribe.cpp. A 0.6B-parameter cache-aware streaming
FastConformer encoder with an RNN-T transducer decoder.

## What it's for

English speech-to-text with greedy RNN-T decoding. Outputs cased,
punctuated transcripts. Token- and word-level timestamps are available.

This port runs the model in **offline** mode. The encoder preserves the
upstream `att_context_size=[70, 13]` (1.12s) cache-aware attention mask
end-to-end. Currently transcribe.cpp has no streaming support.

See NVIDIA's [model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
for training data, intended use, streaming methodology, and the full
latency-vs-accuracy table.

Licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Ported from upstream commit
[`ef3bf40`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/commit/ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4),
pinned 2026-05-11.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean, offline) |
| --- | --- | ---: | ---: |
| F32    | [nemotron-speech-streaming-en-0.6b-F32.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-F32.gguf) | 2.30 GB | 2.31% |
| F16    | [nemotron-speech-streaming-en-0.6b-F16.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-F16.gguf) | 1.16 GB | 2.31% |
| Q8_0   | [nemotron-speech-streaming-en-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q8_0.gguf) |  696 MB | 2.31% |
| Q6_K   | [nemotron-speech-streaming-en-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q6_K.gguf) |  573 MB | 2.29% |
| Q5_K_M | [nemotron-speech-streaming-en-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q5_K_M.gguf) |  514 MB | 2.34% |
| Q4_K_M | [nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf) |  453 MB | 2.38% |

WER is measured on the full LibriSpeech test-clean split (2620
utterances) with greedy RNN-T decoding. F32 reference baseline: 2.31%.
NVIDIA's self-reported number on the same split at
`att_context_size=[70, 13]` (1.12s chunk, w/o PnC) is 2.32% (from the
[HF model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  73 ms (151×) |  73 ms (151×) |
| Metal   | dots (35.3s) | 224 ms (158×) | 221 ms (160×) |
| CPU     | jfk (11.0s)  |  329 ms (33×) |  330 ms (33×) |
| CPU     | dots (35.3s) |  1.12 s (31×) |  1.12 s (31×) |

macOS 26.4.1, transcribe.cpp `12f1076`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  812 ms (14×) |  813 ms (14×) |
| Vulkan  | dots (35.3s) |  2.93 s (12×) |  2.98 s (12×) |
| CPU     | jfk (11.0s)  |  1.39 s (8×)  |  1.22 s (9×)  |
| CPU     | dots (35.3s) |  5.21 s (7×)  |  4.76 s (7×)  |

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models nemotron-speech-streaming-en-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name nemotron-speech-streaming-en-0.6b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav` via `scripts/validate.py`. Per-tensor tolerances live
in a per-variant file
([`tests/tolerances/nemotron-speech-streaming-en-0.6b.json`](../../tests/tolerances/nemotron-speech-streaming-en-0.6b.json))
rather than the family-shared one because the unnormalised log-mel
(NeMo's `normalize="NA"` no-op) lands on a different magnitude scale
than the per-feature-normalised siblings. The family-level forward map
at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources, with a Variant Notes row
covering this model's chunked-attention mask, causal `CausalConv2D`
pre-encode, and LayerNorm conv module.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/nemotron-speech-streaming-en-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/nemotron-speech-streaming-en-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant nemotron-speech-streaming-en-0.6b` |

## Known limitations

- Streaming decoding is not exposed by transcribe.cpp. The model runs
  in offline mode only; the chunked-attention mask
  (`att_context_size=[70, 13]`) is preserved at inference so the
  transcript matches NVIDIA's published offline behavior, but the
  incremental-chunk session API and the lower-latency presets
  (`[70,6]`/`[70,1]`/`[70,0]`) are not yet implemented.
- Multilingual transcription is not supported. The model is
  English-only by training.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/nemotron-speech-streaming-en-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-F32.gguf \
  models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant nemotron-speech-streaming-en-0.6b
```
