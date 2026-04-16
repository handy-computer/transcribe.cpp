# Parakeet TDT 0.6B v3

NVIDIA's [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
ported to transcribe.cpp. A 0.6B-parameter Conformer encoder with a TDT/RNNT
transducer decoder.

## What it's for

Offline multilingual speech-to-text. The model takes a 16 kHz mono WAV and
produces a transcript with optional token-level timestamps. It is not a
streaming model and does not translate. v3 extends v2's English coverage to
<TBD: language list — see NVIDIA model card; e.g. "25 European languages">.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`6d590f7`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/commit/6d590f77001d318fb17a0b5bf7ee329a91b52598),
pinned 2026-04-16.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-tdt-0.6b-v3-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-F32.gguf)       | 2.51 GB | 1.95% |
| F16    | [parakeet-tdt-0.6b-v3-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-F16.gguf)       | 1.26 GB | 1.95% |
| Q8_0   | [parakeet-tdt-0.6b-v3-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf)     | 740 MB  | 1.94% |
| Q6_K   | [parakeet-tdt-0.6b-v3-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q6_K.gguf)     | 627 MB  | 1.93% |
| Q5_K_M | [parakeet-tdt-0.6b-v3-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q5_K_M.gguf) | 565 MB  | 1.93% |
| Q4_K_M | [parakeet-tdt-0.6b-v3-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q4_K_M.gguf) | 502 MB  | 1.99% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy transducer decoding and no external LM. F32 reference baseline:
1.95%. NVIDIA's self-reported number on the same split is 1.93% (from the
[HF model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3))

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are latency in ms (wall-clock mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses.

### Apple M4 Max

| Backend | Sample       |        Q8_0 |      Q4_K_M |
| ------- | ------------ | ----------: | ----------: |
| Metal   | jfk (11.0s)  |  75 (147×)  |  77 (144×)  |
| Metal   | dots (35.3s) | 226 (156×)  | 226 (156×)  |
| CPU     | jfk (11.0s)  | 386 (28×)   | 345 (32×)   |
| CPU     | dots (35.3s) | 1311 (27×)  | 1176 (30×)  |

macOS 26.3.1, transcribe.cpp `<TBD: git short sha at publication>`. Raw data:
`reports/perf/apple-m4-max/parakeet-v3-publication_parakeet_{metal,cpu}.json`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       | Q8_0 | Q4_K_M |
| ------- | ------------ | ---: | -----: |
| Vulkan  | jfk (11.0s)  | TBD  | TBD    |
| Vulkan  | dots (35.3s) | TBD  | TBD    |
| CPU     | jfk (11.0s)  | TBD  | TBD    |
| CPU     | dots (35.3s) | TBD  | TBD    |

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --family parakeet \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-tdt-0.6b-v3-publication
```

## Numerical Validation

> **Pending:** `tests/golden/parakeet/parakeet-tdt-0.6b-v3.manifest.json`
> does not yet exist. v3 shares v2's architecture (0.6B Conformer + TDT),
> so the dump script and methodology transfer, but a v3-specific manifest
> and reference dump set are needed before this table can be filled in.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt-0.6b-v3` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt-0.6b-v3.manifest.json` *(to be added)* |
| Command | `uv run scripts/validate.py compare --family parakeet --variant parakeet-tdt-0.6b-v3` |

Selected tensors: *TBD — run `uv run scripts/validate.py all --family parakeet --variant parakeet-tdt-0.6b-v3` after the manifest is in place.*

## Reproduction

### Convert

Loads directly from NVIDIA's NeMo checkpoint via `ASRModel.from_pretrained`.
Output path is derived from the repo id.

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v3
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F32.gguf \
  models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-tdt-0.6b-v3
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_PARAKEET_GGUF=models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```
