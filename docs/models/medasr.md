# MedASR

Google's [`google/medasr`](https://huggingface.co/google/medasr) ported to transcribe.cpp. 105M-parameter encoder-CTC for medical-dictation English ASR. 17-layer Conformer encoder with RoPE attention (rope_theta=10000), macaron FFNs (residual scalars [1.5, 0.5]), BatchNorm conv module (kernel=32, residual scalars [2.0, 1.0]), and a Linear 512→512 CTC head over a SentencePiece BPE vocabulary.

## What it's for

Offline English speech-to-text optimized for medical dictation (radiology, internal medicine, family medicine). Decoder is greedy CTC; no language model, no beam search. This matches what HuggingFace transformers' `AutoModelForCTC` does by default.

Trained on ~5,000 hours of de-identified physician dictations on top of a LibriHeavy 50k-hour pretrain. The upstream model card flags lower accuracy on non-native accents and a male-skewed speaker distribution.

Licensed under the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms). The upstream repo is gated; you must accept the HF terms before download.

Ported from upstream commit [`ae1e484`](https://huggingface.co/google/medasr/commit/ae1e4845b4b07479735d93e1e591e566435b7104), pinned 2026-06-04.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [medasr-F32.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-F32.gguf)       | 417 MB | 17.88% |
| F16    | [medasr-F16.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-F16.gguf)       | 202 MB | 17.88% |
| Q8_0   | [medasr-Q8_0.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q8_0.gguf)     | 122 MB | 17.86% |
| Q6_K   | [medasr-Q6_K.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q6_K.gguf)     | 101 MB | 17.93% |
| Q5_K_M | [medasr-Q5_K_M.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q5_K_M.gguf) |  90 MB | 17.91% |
| Q4_K_M | [medasr-Q4_K_M.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q4_K_M.gguf) |  79 MB | 18.14% |

**Recommended default: Q8_0.** Smallest preset with no statistically detectable WER degradation versus F32 (122 MB; +0.00 pp within bootstrap CI). Q4_K_M shows a real +0.26 pp degradation on LibriSpeech and is shipped for completeness but **not recommended** — prefer Q5_K_M if you need smaller than Q8_0.

WER measured on the full LibriSpeech test-clean split (2,620 utterances) with greedy CTC decoding and no external LM. F32 reference baseline (HuggingFace transformers, Mac MPS): **17.88%**; transcribe.cpp F32 matches exactly. Absolute WER is higher than general-purpose ASR (e.g. Whisper-base ≈ 5%) because the model is fine-tuned for medical dictation — on the publisher's internal RAD-DICT / GENERAL-DICT / FM-DICT datasets the model scores 6.6%–9.3%, but those datasets are not publicly reproducible. See [`reports/wer/medasr.test-clean.summary.md`](../../reports/wer/medasr.test-clean.summary.md) for the full sweep.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/medasr/medasr-Q8_0.gguf \
  --language en \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2 decimal places).

### Apple M4

| Backend | Sample        |          Q8_0 |        Q4_K_M |
| ------- | ------------- | ------------: | ------------: |
| Metal   | jfk (11.0 s)  |  74 ms (148×) |  71 ms (155×) |
| Metal   | dots (35.3 s) | 168 ms (211×) | 168 ms (210×) |

macOS 26.1, transcribe.cpp `782abfd`. Metal device: `Apple M4`. Mel pipeline uses the shared `MelFrontend` (Accelerate vDSP fp64 FFT + cblas_sgemm); encoder is the conformer + RoPE + BatchNorm-conv graph in `src/arch/medasr/encoder.cpp`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models medasr --quants q8_0,q4_k_m --samples jfk,dots \
  --backends metal --iters 3 --warmup 1 --name medasr-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the upstream HuggingFace Transformers reference on `samples/jfk.wav` via `scripts/validate.py`. The family-level forward map at [`reports/porting/medasr/forward-map.md`](../../reports/porting/medasr/forward-map.md) documents the per-stage divergence sources (fp64 vDSP STFT, BatchNorm fusion, CUDA fp16-accumulator workarounds in the macaron + conv residual stack).

| Field | Value |
| --- | --- |
| Reference | `transformers @ 65dc2615` (dev commit; v5.0.0 not yet released), `AutoModelForCTC.from_pretrained("google/medasr")` device=mps fp32 |
| Dump script | `scripts/dump_reference_medasr_transformers.py` |
| Manifest | `tests/golden/medasr/medasr.manifest.json` |
| Command | `uv run scripts/validate.py all --family medasr --variant medasr` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/medasr \
  scripts/convert-medasr.py google/medasr
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/medasr/medasr-F32.gguf \
  models/medasr/medasr-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family medasr --variant medasr
```
