# Canary 1B

NVIDIA's [`nvidia/canary-1b`](https://huggingface.co/nvidia/canary-1b)
ported to transcribe.cpp. A 1B-parameter multitask AED with a 24-layer
FastConformer encoder and a 24-layer Transformer decoder — the original
canary release.

> **License: CC-BY-NC-4.0 (non-commercial only).** This is the only
> canary variant under a non-commercial license. Every shipped GGUF
> carries `general.license: CC-BY-NC-4.0` in its KV metadata so
> downstream tooling can detect this without re-reading the model card.

## What it's for

Offline multilingual speech-to-text and translation. The model takes a
16 kHz mono WAV and produces a transcript. Supports:

- **ASR** in English, German, Spanish, and French.
- **Translation** between supported pairs.

Architecturally distinct from the rest of the canary family:

- **24+24** layers (deepest decoder in the family). Per-token decode
  latency is ~3–4× the flash variants on the same backend; the
  flash/180m-flash variants have a shallow 4-layer decoder.
- Uses the **canary-1 prompt format** (4 slots: source-lang,
  target-lang, taskname, pnc). The flash variants and canary-1b-v2 use
  the 5-slot **canary2** format that adds a `<toggle_timestamps>` slot.
- No timestamps (the upstream model does not advertise word/segment
  timestamps; the flash and v2 variants do, via a side aligner).

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-1b)
for training data, intended use, and upstream evaluation methodology.

Ported from upstream commit
[`1698acf`](https://huggingface.co/nvidia/canary-1b/commit/1698acf1700ed316ffce1cb42d79437c7e360cfa),
pinned 2026-05-08.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [canary-1b-F32.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-F32.gguf)       | 3.8 GB | 1.55% |
| F16    | [canary-1b-F16.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-F16.gguf)       | 1.9 GB | 1.55% |
| Q8_0   | [canary-1b-Q8_0.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q8_0.gguf)     | 1.1 GB | 1.55% |
| Q6_K   | [canary-1b-Q6_K.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q6_K.gguf)     | 891 MB | 1.57% |
| Q5_K_M | [canary-1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q5_K_M.gguf) | 799 MB | 1.57% |
| Q4_K_M | [canary-1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q4_K_M.gguf) | 696 MB | 1.55% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding and no external LM. F32 reference baseline: 1.55%.
NVIDIA's self-reported number on the upstream model card is 1.48%; the
small gap is well inside the Stage 7 ref-dtype gate (|Δ| ≤ 1pp). Quants
are exceptionally flat — every preset is within ~0.02pp of F32, the
tightest cluster of any canary variant.

## Quick Start

```bash
cmake -B build
cmake --build build

# ASR (English)
build/bin/transcribe-cli \
  -m models/canary-1b/canary-1b-Q8_0.gguf \
  -l en \
  samples/jfk.wav

# Translation (English audio → German text)
build/bin/transcribe-cli \
  -m models/canary-1b/canary-1b-Q8_0.gguf \
  --task translate \
  -l en --target-language de \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags specific to canary:

- `--pnc` / `--no-pnc` — punctuation & capitalization (default on).
- `-l <code>` — source language code (`en`, `de`, `es`, `fr`).
- `--task translate` + `--target-language <code>` — switch to translation
  mode. canary-1 uses an explicit `<|translate|>` task token (the
  canary2 variants infer translate from src ≠ tgt instead).

## Performance

Bench numbers will be added after the public release. Note that the
24-layer decoder makes this the slowest canary variant for decode-bound
workloads — for the same audio, expect ~3–4× the wall time of
canary-1b-flash on the same backend.

```bash
uv run scripts/bench/run.py \
  --models canary-1b \
  --quants q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name canary-1b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav`. All checkpointed tensors fall within family
tolerance and the F32 transcript matches the NeMo reference. Last
validated at commit
[`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/canary-1b` |
| Dump script | `scripts/dump_reference_canary_nemo.py` |
| Manifest | `tests/golden/canary/canary-1b.manifest.json` |
| Tolerances | `tests/tolerances/canary.json` |
| Command | `uv run scripts/validate.py all --family canary --variant canary-1b` |

For the full porting writeup, see
[`docs/porting/families/canary.md`](../porting/families/canary.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/canary \
  scripts/convert-canary.py nvidia/canary-1b --repo-id nvidia/canary-1b
```

### Quantize

```bash
uv run scripts/quantize-all.py models/canary-1b/canary-1b-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family canary --variant canary-1b
```
