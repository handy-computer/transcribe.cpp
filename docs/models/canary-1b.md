# Canary 1B

<!-- catalog:intro -->
Upstream: [`nvidia/canary-1b`](https://huggingface.co/nvidia/canary-1b) at [`1698acf`](https://huggingface.co/nvidia/canary-1b/commit/1698acf).

Offline multilingual speech-to-text and translation. A multitask AED
with a 24-layer FastConformer encoder and a 24-layer Transformer
decoder — the original canary release. Supports automatic speech
recognition in English, German, Spanish, and French, and translation
between supported pairs. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. **License: CC-BY-NC-4.0
(non-commercial only)** — the only canary variant under a
non-commercial license.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and translation. The model takes a
16 kHz mono WAV and produces a transcript. Supports:

- **ASR** in English, German, Spanish, and French.
- **Translation** between supported pairs.

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-1b)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-NC-4.0. Ported from upstream commit [`1698acf`](https://huggingface.co/nvidia/canary-1b/commit/1698acf), pinned 2026-05-08. Validated against the NeMo reference at transcribe.cpp commit [`db53eda`](https://github.com/handy-computer/transcribe.cpp/tree/db53eda) on 2026-05-08.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [canary-1b-F32.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-F32.gguf) | 4.09 GB | 1.55% |
| F16          | [canary-1b-F16.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-F16.gguf) | 2.05 GB | 1.55% |
| Q8_0         | [canary-1b-Q8_0.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q8_0.gguf) | 1.16 GB | 1.55% |
| Q6_K         | [canary-1b-Q6_K.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q6_K.gguf) |  934 MB | 1.57% |
| Q5_K_M       | [canary-1b-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q5_K_M.gguf) |  838 MB | 1.57% |
| Q4_K_M       | [canary-1b-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-1b-gguf/resolve/main/canary-1b-Q4_K_M.gguf) |  730 MB | 1.55% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM. F32 reference baseline: 1.55%. NVIDIA's
self-reported number on the upstream model card is 1.48%; likely this is due to
differences in how we score WER, based on the results we have from
canary-180m-flash.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 6.45% |
| en       | WER    | 4.44% |
| es       | WER    | 6.06% |
| fr       | WER    | 7.44% |
<!-- /catalog -->

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

The 24-layer decoder makes this the slowest canary variant for decode-bound
workloads — roughly 1.5× the wall time of canary-1b-flash on the same
backend, and the GPU win over CPU is smaller here than on the *flash
variants because the autoregressive decoder pass dominates and a
batch-1 / single-token forward is too small to amortize Vulkan dispatch
overhead.

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 207.3 ms (53.07×) | 187.5 ms (58.67×) |
| Metal   | dots (35.3s) |   1.01 s (34.96×) | 930.9 ms (37.95×) |
| CPU     | jfk (11.0s)  | 425.7 ms (25.84×) | 430.2 ms (25.57×) |
| CPU     | dots (35.3s) |   1.79 s (19.74×) |   1.73 s (20.41×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Vulkan  | jfk (11.0s)  | 962.2 ms (11.43×) | 879.4 ms (12.51×) |
| Vulkan  | dots (35.3s) |    4.36 s (8.10×) |    3.81 s (9.28×) |
| CPU     | jfk (11.0s)  |    1.40 s (7.88×) |    1.26 s (8.73×) |
| CPU     | dots (35.3s) |    6.96 s (5.08×) |    6.26 s (5.65×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models canary-1b
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
