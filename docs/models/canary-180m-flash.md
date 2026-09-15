# Canary 180M Flash

<!-- catalog:intro -->
Upstream: [`nvidia/canary-180m-flash`](https://huggingface.co/nvidia/canary-180m-flash) at [`b12ab41`](https://huggingface.co/nvidia/canary-180m-flash/commit/b12ab41).

Offline multilingual speech-to-text and translation. A multitask AED
with a 17-layer FastConformer encoder and a 4-layer Transformer decoder.
Supports automatic speech recognition in English, German, Spanish, and
French, and bidirectional EN↔{DE, ES, FR} translation. Takes a 16 kHz
mono WAV and produces a transcript. Not a streaming model; word/segment
timestamps are upstream-experimental and not exposed in the v1 port.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and translation. The model takes a
16 kHz mono WAV and produces a transcript. Supports:

- **ASR** in English, German, Spanish, and French (with explicit language
  hint).
- **Translation** between English and German, Spanish, or French (both
  directions).

Not a streaming model. Word and segment timestamps are upstream-experimental
and not exposed in the v1 port (deferred — would require porting the
`_timestamps_asr_model` CTC aligner from the `.nemo` archive).

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-180m-flash)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`b12ab41`](https://huggingface.co/nvidia/canary-180m-flash/commit/b12ab41), pinned 2026-05-08. Validated against the NeMo reference at transcribe.cpp commit [`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda) on 2026-05-08.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [canary-180m-flash-F32.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-F32.gguf) | 756 MB | 1.94% |
| F16          | [canary-180m-flash-F16.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-F16.gguf) | 382 MB | 1.94% |
| Q8_0         | [canary-180m-flash-Q8_0.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q8_0.gguf) | 218 MB | 1.93% |
| Q6_K         | [canary-180m-flash-Q6_K.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q6_K.gguf) | 176 MB | 1.93% |
| Q5_K_M       | [canary-180m-flash-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q5_K_M.gguf) | 159 MB | 1.90% |
| Q4_K_M       | [canary-180m-flash-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q4_K_M.gguf) | 139 MB | 1.93% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM. F32 reference baseline: 1.94%. On the same wavs,
NeMo's reference run produces 1.93% (one substitution difference out of ~27k
reference words), so the F32 port matches the reference framework at the noise
floor. NVIDIA's self-reported number on the upstream model card is 1.87%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 7.33% |
| en       | WER    | 5.98% |
| es       | WER    | 6.54% |
| fr       | WER    | 8.53% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |               Q8_0 |             Q4_K_M |
| ------- | ------------ | -----------------: | -----------------: |
| Metal   | jfk (11.0s)  |  64.0 ms (171.97×) |  59.2 ms (185.69×) |
| Metal   | dots (35.3s) | 261.8 ms (134.95×) | 240.4 ms (146.98×) |
| CPU     | jfk (11.0s)  |  127.8 ms (86.10×) |  129.5 ms (84.94×) |
| CPU     | dots (35.3s) |  492.7 ms (71.72×) |  490.1 ms (72.09×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Vulkan  | jfk (11.0s)  | 307.1 ms (35.82×) | 283.6 ms (38.78×) |
| Vulkan  | dots (35.3s) |   1.16 s (30.35×) |   1.07 s (33.14×) |
| CPU     | jfk (11.0s)  | 437.8 ms (25.13×) | 420.9 ms (26.13×) |
| CPU     | dots (35.3s) |   1.93 s (18.33×) |   1.79 s (19.69×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models canary-180m-flash
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav`. All 17 checkpointed tensors fall within family
tolerance, and the F32 transcript matches the NeMo reference at the noise
floor (one substitution out of ~27k reference words across full
test-clean).

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
