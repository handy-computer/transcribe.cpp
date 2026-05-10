# Parakeet TDT-CTC 110M

NVIDIA's [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m)
ported to transcribe.cpp. A hybrid 110M-parameter FastConformer encoder with
both TDT and CTC heads sharing the same encoder; transcribe.cpp uses the TDT
head by default.

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
pinned 2026-05-09.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-tdt_ctc-110m-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-F32.gguf) | 457 MB |                        2.43% |
| F16    | [parakeet-tdt_ctc-110m-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-F16.gguf) | 229 MB |                        2.43% |
| Q8_0   | [parakeet-tdt_ctc-110m-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q8_0.gguf) | 135 MB |                        2.43% |
| Q6_K   | [parakeet-tdt_ctc-110m-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q6_K.gguf) | 112 MB |                        2.44% |
| Q5_K_M | [parakeet-tdt_ctc-110m-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q5_K_M.gguf) | 101 MB |                        2.47% |
| Q4_K_M | [parakeet-tdt_ctc-110m-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf) |  90 MB |                        2.53% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy TDT decoding and no external LM. F32 reference baseline: 2.43%. NVIDIA's self-reported number on the same split is 2.40% (from the [HF model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m)).

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places). Cells gated on `Tctl < 55°C` per backend.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |   36 ms (310×) |   44 ms (251×) |
| Metal   | dots (35.3s) |  100 ms (353×) |  100 ms (354×) |
| CPU     | jfk (11.0s)  |   94 ms (117×) |   89 ms (124×) |
| CPU     | dots (35.3s) |  326 ms (108×) |  312 ms (113×) |

macOS 26.4.1, transcribe.cpp `a6c097e`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  403 ms (27×) |  406 ms (27×) |
| Vulkan  | dots (35.3s) |  1.36 s (26×) |  1.39 s (25×) |
| CPU     | jfk (11.0s)  |  508 ms (22×) |  485 ms (23×) |
| CPU     | dots (35.3s) |  1.87 s (19×) |  1.84 s (19×) |

Fedora 43, transcribe.cpp `57997dc`. Vulkan device: `AMD Radeon
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
