# Parakeet TDT-CTC 110M

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) at [`431a349`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/commit/431a349).

Offline English speech-to-text with punctuation and capitalization. A 110M-parameter FastConformer encoder with a TDT/RNNT transducer decoder (the auxiliary CTC head from the upstream hybrid checkpoint is dropped at convert time). Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline English speech-to-text, optimised for fast inference at the cost of
some accuracy versus the larger Parakeet variants. Outputs cased,
punctuated transcripts. Token-, word- and segment-level timestamps are
available. The model takes a 16 kHz mono WAV and produces a transcript.
It is not a streaming model and does not translate.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`431a349`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/commit/431a349f3051ab85c22b9b7a2741b5fe77065665),
pinned 2026-05-10.

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-tdt_ctc-110m-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-F32.gguf) | 457 MB | 2.43% |
| F16          | [parakeet-tdt_ctc-110m-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-F16.gguf) | 229 MB | 2.43% |
| Q8_0         | [parakeet-tdt_ctc-110m-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q8_0.gguf) | 135 MB | 2.43% |
| Q6_K         | [parakeet-tdt_ctc-110m-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q6_K.gguf) | 112 MB | 2.44% |
| Q5_K_M       | [parakeet-tdt_ctc-110m-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q5_K_M.gguf) | 101 MB | 2.47% |
| Q4_K_M       | [parakeet-tdt_ctc-110m-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf) |  90 MB | 2.53% |
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances) with greedy TDT/RNN-T transducer decoding and no external LM. F32 reference baseline: 2.43%. NVIDIA's self-reported number on the same split is 2.40%.
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are compute latency (mel + encode + decode; mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places). Cells gated on `Tctl < 55°C` per backend.

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  34 ms (320.03×) |     35 ms (315×) |
| Metal   | dots (35.3s) |  99 ms (358.29×) |  98 ms (360.02×) |
| CPU     | jfk (11.0s)  |  94 ms (117.37×) |  88 ms (124.33×) |
| CPU     | dots (35.3s) | 325 ms (108.62×) | 311 ms (113.49×) |
<!-- /catalog -->

macOS 26.4.1, transcribe.cpp `12f1076`.

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 315 ms (34.94×) | 322 ms (34.22×) |
| Vulkan  | dots (35.3s) | 1.18 s (30.00×) | 1.19 s (29.59×) |
| CPU     | jfk (11.0s)  | 420 ms (26.16×) | 394 ms (27.89×) |
| CPU     | dots (35.3s) | 1.70 s (20.76×) | 1.64 s (21.53×) |
<!-- /catalog -->

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-tdt_ctc-110m \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-tdt_ctc-110m-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`
via `scripts/validate.py`, sharing the parakeet family tolerance file. The
shared FastConformer encoder code path is the same as the published
`parakeet-tdt-0.6b-v2` variant; the family-level forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources (fp64 STFT, mel amplification,
attenuation through the encoder).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt_ctc-110m` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt_ctc-110m.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-tdt_ctc-110m` |

## Reproduction

### Convert

Loads directly from NVIDIA's NeMo checkpoint via `ASRModel.from_pretrained`.
Output path is derived from the repo id.

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt_ctc-110m
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0; repeat with
`Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-F32.gguf \
  models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-tdt_ctc-110m
```
