# Canary 180M Flash

NVIDIA's [`nvidia/canary-180m-flash`](https://huggingface.co/nvidia/canary-180m-flash)
ported to transcribe.cpp. A 182M-parameter multitask AED with a 17-layer
FastConformer encoder and a 4-layer Transformer decoder.

## What it's for

Offline multilingual speech-to-text and translation. The model takes a
16 kHz mono WAV and produces a transcript. Supports:

- **ASR** in English, German, Spanish, and French (with explicit language
  hint).
- **Translation** from English to German, Spanish, or French.

Not a streaming model. Word and segment timestamps are upstream-experimental
and not exposed in the v1 port (deferred — would require porting the
`_timestamps_asr_model` CTC aligner from the `.nemo` archive).

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-180m-flash)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`b12ab41`](https://huggingface.co/nvidia/canary-180m-flash/commit/b12ab418510d093e83890178fd0e8b0d0f7918a6),
pinned 2026-05-08.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [canary-180m-flash-F32.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-F32.gguf)       | 721 MB | 1.94% |
| F16    | [canary-180m-flash-F16.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-F16.gguf)       | 364 MB | 1.94% |
| Q8_0   | [canary-180m-flash-Q8_0.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q8_0.gguf)     | 208 MB | 1.93% |
| Q6_K   | [canary-180m-flash-Q6_K.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q6_K.gguf)     | 168 MB | 1.93% |
| Q5_K_M | [canary-180m-flash-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q5_K_M.gguf) | 151 MB | 1.90% |
| Q4_K_M | [canary-180m-flash-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q4_K_M.gguf) | 133 MB | 1.93% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding and no external LM. F32 reference baseline: 1.94%.
On the same wavs, NeMo's reference run produces 1.93% — one substitution
difference out of ~27k reference words — so the F32 port matches the
reference framework at the noise floor. NVIDIA's self-reported number on
the upstream model card is 1.87%; the small gap is consistent with
LibriSpeech version / setup differences and is well inside the Stage 7
ref-dtype gate (|Δ| ≤ 1pp).

Quants are all within ~0.04pp of the F32 baseline; the WER hierarchy is
flat for this model. Q8_0 and Q4_K_M are both safe choices.

## Quick Start

```bash
cmake -B build
cmake --build build

# ASR (English)
build/bin/transcribe-cli \
  -m models/canary-180m-flash/canary-180m-flash-Q8_0.gguf \
  -l en \
  samples/jfk.wav

# Translation (English audio → German text)
build/bin/transcribe-cli \
  -m models/canary-180m-flash/canary-180m-flash-Q8_0.gguf \
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
  mode.

## Performance

Bench numbers will be added after the public release; the harness lives
at `scripts/bench/run.py`. Reproduction:

```bash
uv run scripts/bench/run.py \
  --models canary-180m-flash \
  --quants q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name canary-180m-flash-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav`. All 17 checkpointed tensors fall within family
tolerance, and the F32 transcript matches the NeMo reference at the
noise floor (one substitution out of ~27k reference words across full
test-clean). Last validated at commit
[`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/canary-180m-flash` |
| Dump script | `scripts/dump_reference_canary_nemo.py` |
| Manifest | `tests/golden/canary/canary-180m-flash.manifest.json` |
| Tolerances | `tests/tolerances/canary.json` |
| Command | `uv run scripts/validate.py all --family canary --variant canary-180m-flash` |

For the full porting writeup, see
[`docs/porting/families/canary.md`](../porting/families/canary.md).

## Reproduction

### Convert

Loads from NVIDIA's NeMo checkpoint via `EncDecMultiTaskModel.from_pretrained`.
Output path is derived from the repo id.

```bash
uv run --project scripts/envs/canary \
  scripts/convert-canary.py nvidia/canary-180m-flash --repo-id nvidia/canary-180m-flash
```

### Quantize

Run `transcribe-quantize` once per target preset, or use the helper
that produces all five derived presets in one call:

```bash
uv run scripts/quantize-all.py models/canary-180m-flash/canary-180m-flash-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family canary --variant canary-180m-flash
```
