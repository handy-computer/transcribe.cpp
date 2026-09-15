# Canary 1B v2

<!-- catalog:intro -->
Upstream: [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) at [`87bc526`](https://huggingface.co/nvidia/canary-1b-v2/commit/87bc526).

Offline multilingual speech-to-text and translation across 25 European
languages. A multitask AED with a 32-layer FastConformer encoder and an
8-layer Transformer decoder. Supports automatic speech recognition for
any of the 25 supported languages, plus translation between supported
language pairs (per the upstream model card). Takes a 16 kHz mono WAV and
produces a transcript. Not a streaming model; word and segment timestamps
from the upstream model are not exposed in the v1 port.
<!-- /catalog -->

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

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-1b-v2)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`87bc526`](https://huggingface.co/nvidia/canary-1b-v2/commit/87bc526), pinned 2026-05-08. Validated against the NeMo reference at transcribe.cpp commit [`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda) on 2026-05-08.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [canary-1b-v2-F32.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-F32.gguf) | 3.92 GB | 1.92% |
| F16          | [canary-1b-v2-F16.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-F16.gguf) | 1.97 GB | 1.92% |
| Q8_0         | [canary-1b-v2-Q8_0.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q8_0.gguf) | 1.14 GB | 1.91% |
| Q6_K         | [canary-1b-v2-Q6_K.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q6_K.gguf) |  932 MB | 1.94% |
| Q5_K_M       | [canary-1b-v2-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q5_K_M.gguf) |  837 MB | 1.93% |
| Q4_K_M       | [canary-1b-v2-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q4_K_M.gguf) |  735 MB | 1.91% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM. F32 reference baseline: 1.92%. NVIDIA's
self-reported number on the upstream model card is 2.18%; our F32 port comes in
slightly under the upstream-reported number (Δ −0.26pp) and is likely down to
scoring differences.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| bg       | WER    |  9.22% |
| cs       | WER    |  8.56% |
| da       | WER    | 11.34% |
| de       | WER    |  4.46% |
| el       | WER    | 26.02% |
| en       | WER    |  4.47% |
| es       | WER    |  3.10% |
| et       | WER    | 12.72% |
| fi       | WER    |  8.86% |
| fr       | WER    |  5.09% |
| hr       | WER    |  8.40% |
| hu       | WER    | 13.06% |
| it       | WER    |  3.10% |
| lt       | WER    | 13.45% |
| lv       | WER    | 10.41% |
| mt       | WER    | 19.75% |
| nl       | WER    |  6.28% |
| pl       | WER    |  6.88% |
| pt       | WER    |  4.50% |
| ro       | WER    |  6.87% |
| ru       | WER    |  7.83% |
| sk       | WER    |  6.84% |
| sl       | WER    | 12.76% |
| sv       | WER    |  9.74% |
| uk       | WER    | 10.58% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |               Q8_0 |             Q4_K_M |
| ------- | ------------ | -----------------: | -----------------: |
| Metal   | jfk (11.0s)  | 105.1 ms (104.66×) | 104.1 ms (105.66×) |
| Metal   | dots (35.3s) |  384.7 ms (91.84×) |  362.1 ms (97.57×) |
| CPU     | jfk (11.0s)  |  415.1 ms (26.50×) |  442.4 ms (24.86×) |
| CPU     | dots (35.3s) |    1.49 s (23.66×) |    1.56 s (22.71×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Vulkan  | jfk (11.0s)  | 758.9 ms (14.49×) | 732.3 ms (15.02×) |
| Vulkan  | dots (35.3s) |   2.57 s (13.77×) |   2.41 s (14.64×) |
| CPU     | jfk (11.0s)  |    1.15 s (9.54×) |    1.15 s (9.54×) |
| CPU     | dots (35.3s) |    4.97 s (7.11×) |    4.77 s (7.40×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models canary-1b-v2
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
