# Canary 1B v2

NVIDIA's [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2)
ported to transcribe.cpp. A 978M-parameter multitask AED with a 32-layer
FastConformer encoder and an 8-layer Transformer decoder, covering 25
European languages.

## What it's for

Offline multilingual speech-to-text and translation across **25 European
languages**: Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian,
Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian,
Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish,
Swedish, Russian, and Ukrainian.

The model takes a 16 kHz mono WAV and produces a transcript. Supports:

- **ASR** for any of the 25 supported languages (with explicit language
  hint).
- **Translation** between supported language pairs (per the upstream
  model card).

This is the broadest-coverage canary variant. The other multilingual
variants (180m-flash, 1b-flash) cover only English/German/Spanish/French.

Not a streaming model. Word and segment timestamps from the upstream
model are not exposed in the v1 port.

Architecturally distinct from the other canary variants in that v2 uses
a single SentencePiece tokenizer (`lang_codes = ["all"]`) instead of an
aggregate of per-language SP models — relevant if you ever need to
inspect or extend the tokenizer.

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-1b-v2)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`87bc526`](https://huggingface.co/nvidia/canary-1b-v2/commit/87bc52657add533cd0156b3fc1aef027280754bf),
pinned 2026-05-08.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [canary-1b-v2-F32.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-F32.gguf)       | 3.7 GB | 1.92% |
| F16    | [canary-1b-v2-F16.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-F16.gguf)       | 1.8 GB | 1.92% |
| Q8_0   | [canary-1b-v2-Q8_0.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q8_0.gguf)     | 1.1 GB | 1.91% |
| Q6_K   | [canary-1b-v2-Q6_K.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q6_K.gguf)     | 889 MB | 1.94% |
| Q5_K_M | [canary-1b-v2-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q5_K_M.gguf) | 798 MB | 1.93% |
| Q4_K_M | [canary-1b-v2-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q4_K_M.gguf) | 701 MB | 1.91% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding and no external LM. F32 reference baseline: 1.92%.
NVIDIA's self-reported number on the upstream model card is 2.18%; our
F32 port comes in slightly under upstream (Δ −0.26pp) and is well inside
the Stage 7 ref-dtype gate. Quants are all within ~0.03pp of the F32
baseline. WER on the other 24 languages has not been measured by this
port; the LibriSpeech number is English-only and is not a comprehensive
multilingual benchmark.

## Quick Start

```bash
cmake -B build
cmake --build build

# ASR (any supported language)
build/bin/transcribe-cli \
  -m models/canary-1b-v2/canary-1b-v2-Q8_0.gguf \
  -l en \
  samples/jfk.wav

# ASR (German)
build/bin/transcribe-cli \
  -m models/canary-1b-v2/canary-1b-v2-Q8_0.gguf \
  -l de \
  samples/german.wav

# Translation (English audio → German text)
build/bin/transcribe-cli \
  -m models/canary-1b-v2/canary-1b-v2-Q8_0.gguf \
  --task translate \
  -l en --target-language de \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags specific to canary:

- `--pnc` / `--no-pnc` — punctuation & capitalization. Note: canary-1b-v2
  ignores `--no-pnc` and emits PNC-on output regardless. This matches
  upstream NeMo behavior on this checkpoint.
- `-l <code>` — source language code (one of the 25 supported BCP-47
  codes).
- `--task translate` + `--target-language <code>` — switch to translation
  mode.

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places).

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  | 133.5 ms (82.4×) | 129.3 ms (85.1×) |
| Metal   | dots (35.3s) | 466.8 ms (75.7×) | 457.0 ms (77.3×) |
| CPU     | jfk (11.0s)  | 555.0 ms (19.8×) | 453.5 ms (24.3×) |
| CPU     | dots (35.3s) |   1.96 s (18.0×) |   1.66 s (21.3×) |

macOS 26.4.1, transcribe.cpp `0f42b37`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models canary-1b-v2 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name canary-1b-v2-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav`. All checkpointed tensors fall within family
tolerance and the F32 transcript matches the NeMo reference. Last
validated at commit
[`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/canary-1b-v2` |
| Dump script | `scripts/dump_reference_canary_nemo.py` |
| Manifest | `tests/golden/canary/canary-1b-v2.manifest.json` |
| Tolerances | `tests/tolerances/canary.json` |
| Command | `uv run scripts/validate.py all --family canary --variant canary-1b-v2` |

For the full porting writeup, see
[`docs/porting/families/canary.md`](../porting/families/canary.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/canary \
  scripts/convert-canary.py nvidia/canary-1b-v2 --repo-id nvidia/canary-1b-v2
```

### Quantize

```bash
uv run scripts/quantize-all.py models/canary-1b-v2/canary-1b-v2-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family canary --variant canary-1b-v2
```
