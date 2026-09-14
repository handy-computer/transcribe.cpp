# MedASR

<!-- catalog:intro -->
Upstream: [`google/medasr`](https://huggingface.co/google/medasr) at [`ae1e484`](https://huggingface.co/google/medasr/commit/ae1e484).

Offline English speech-to-text optimized for medical dictation (radiology, internal medicine, family medicine). 17-layer Conformer encoder with RoPE attention, macaron FFNs, and a 512-token SentencePiece CTC head. Greedy CTC decode; no language model, no beam search.
<!-- /catalog -->

## What it's for

Offline English speech-to-text optimized for medical dictation (radiology, internal medicine, family medicine). Decoder is greedy CTC; no language model, no beam search. This matches what HuggingFace transformers' `AutoModelForCTC` does by default.

Trained on ~5,000 hours of de-identified physician dictations on top of a LibriHeavy 50k-hour pretrain. The upstream model card flags lower accuracy on non-native accents and a male-skewed speaker distribution.

Licensed under the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms). The upstream repo is gated; you must accept the HF terms before download.

Ported from upstream commit [`ae1e484`](https://huggingface.co/google/medasr/commit/ae1e4845b4b07479735d93e1e591e566435b7104), pinned 2026-06-04.

## Input limits

MedASR is trained for audio up to about **400 seconds (~6.7 min)** — the
encoder's rotary-position window. Longer audio is accepted, but the library logs
a `WARN` and accuracy may degrade past that window; it is not rejected. Segment
long recordings for best results. See the
[input-length contract](../input-limits.md).

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [medasr-F32.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-F32.gguf) | 421 MB | 17.88% |
| F16          | [medasr-F16.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-F16.gguf) | 211 MB | 17.88% |
| Q8_0         | [medasr-Q8_0.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q8_0.gguf) | 128 MB | 17.86% |
| Q6_K         | [medasr-Q6_K.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q6_K.gguf) | 106 MB | 17.93% |
| Q5_K_M       | [medasr-Q5_K_M.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q5_K_M.gguf) |  94 MB | 17.91% |
| Q4_K_M       | [medasr-Q4_K_M.gguf](https://huggingface.co/handy-computer/medasr-gguf/resolve/main/medasr-Q4_K_M.gguf) |  83 MB | 18.14% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy CTC decoding, no external LM. F32 reference baseline (HuggingFace
transformers, Mac MPS): 17.88%; transcribe.cpp F32 matches exactly. Absolute WER is
higher than general-purpose ASR (e.g. Whisper-base ~5%) because the model is
fine-tuned for medical dictation — on the publisher's internal RAD-DICT /
GENERAL-DICT / FM-DICT datasets the model scores 6.6%–9.3%, but those datasets are
not publicly reproducible. Q8_0 is the recommended default (smallest preset with no
statistically detectable WER degradation); Q4_K_M shows a real +0.26 pp degradation
and is shipped for completeness but not recommended — prefer Q5_K_M if you need
smaller than Q8_0.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| en       | WER    | 37.48% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  | 38 ms (290×)† | 44 ms (248×)† |
| Metal   | dots (35.3s) | 84 ms (419×)† | 90 ms (394×)† |
| CPU     | jfk (11.0s)  | 161 ms (68×)† | 180 ms (61×)† |
| CPU     | dots (35.3s) | 558 ms (63×)† | 623 ms (57×)† |

Apple M4 Max. † published before provenance was recorded; not yet re-measured.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 161 ms (68.17×) | 173 ms (63.63×) |
| Vulkan  | dots (35.3s) | 479 ms (73.76×) | 493 ms (71.61×) |
| CPU     | jfk (11.0s)  | 542 ms (20.30×) | 488 ms (22.56×) |
| CPU     | dots (35.3s) | 1.84 s (19.21×) | 1.63 s (21.73×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `79d139a` on 2026-06-04.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models medasr
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
