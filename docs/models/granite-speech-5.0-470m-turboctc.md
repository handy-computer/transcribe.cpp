# Granite Speech 5.0 470M TurboCTC

<!-- catalog:intro -->
Upstream: [`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) at [`18ca3c1`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/commit/18ca3c1).

Offline English speech-to-text. A Granite Conformer encoder with a self-conditioned CTC
head. Takes a 16 kHz mono WAV and produces a transcript. Not a streaming model. English
only.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. English only, and it does not translate.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`18ca3c1`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/commit/18ca3c1), pinned 2026-09-12. Validated against the transformers reference at transcribe.cpp commit [`b9427cf`](https://github.com/handy-computer/transcribe.cpp/tree/b9427cf) on 2026-09-12.
<!-- /catalog -->

There is also a non-commercial sibling,
[`granite-speech-5.0-470m-turboctc-nc`](granite-speech-5.0-470m-turboctc-nc.md),
trained on more data and slightly more accurate. It is CC-BY-NC-SA-4.0, so use
this one for anything commercial.

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [granite-speech-5.0-470m-turboctc-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-BF16.gguf) | 948 MB | 1.34% |
| F16          | [granite-speech-5.0-470m-turboctc-F16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-F16.gguf) | 948 MB | 1.33% |
| Q8_0         | [granite-speech-5.0-470m-turboctc-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q8_0.gguf) | 506 MB | 1.33% |
| Q6_K         | [granite-speech-5.0-470m-turboctc-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q6_K.gguf) | 392 MB | 1.33% |
| Q5_K_M       | [granite-speech-5.0-470m-turboctc-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q5_K_M.gguf) | 336 MB | 1.34% |
| Q4_K_M       | [granite-speech-5.0-470m-turboctc-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q4_K_M.gguf) | 279 MB | 1.35% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none, language hint `en`, decoded on cuda. Measured at transcribe.cpp `9daf396`.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy CTC decoding, no external LM. Measured reference baseline (transformers
5.17.0, F32, CPU): 1.33%, 95% CI [1.20, 1.47].
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.61% |
<!-- /catalog -->

## Quick Start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-Q8_0.gguf \
  samples/jfk.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 36 ms (301.30×) | 39 ms (284.51×) |
| Metal   | dots (35.3s) | 86 ms (410.20×) | 95 ms (371.09×) |
| CPU     | jfk (11.0s)  | 243 ms (45.19×) | 254 ms (43.29×) |
| CPU     | dots (35.3s) | 752 ms (46.95×) | 769 ms (45.93×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 588 ms (18.69×) | 588 ms (18.72×) |
| Vulkan  | dots (35.3s) | 1.31 s (27.04×) | 1.31 s (26.89×) |
| CPU     | jfk (11.0s)  | 615 ms (17.90×) | 652 ms (16.87×) |
| CPU     | dots (35.3s) | 2.13 s (16.58×) | 2.14 s (16.51×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Q8_0 is usually a little faster than Q4_K_M despite being 1.8× the size, so
pick Q4_K_M for footprint rather than speed. Cost is linear in audio length:
realtime factor is flat from 11 s out to 5 minutes. On CPU, avoid BF16 — that
matmul kernel does not thread; use any quantized tier instead. Batching is
supported and gives identical output, but buys almost nothing here.

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against Hugging Face transformers
on `samples/jfk.wav` and `samples/dots.wav`. All 35 checkpointed tensors fall
within family tolerance, and the final transcript matches the reference
verbatim.

| Field | Value |
| --- | --- |
| Reference | transformers 5.17.0, `ibm-granite/granite-speech-5.0-470m-turboctc` |
| Dump script | `scripts/dump_reference_granite5_ctc_transformers.py` |
| Manifest | `tests/golden/granite5_ctc/granite-speech-5.0-470m-turboctc.manifest.json` |
| Tolerances | `tests/tolerances/granite5_ctc.json` |
| Command | `uv run scripts/validate.py all --family granite5_ctc --variant granite-speech-5.0-470m-turboctc` |

Drift is BF16 matmul accumulation compounding over 16 blocks, and stays small
relative to tensor magnitude. It does not reach the output: zero argmax
differences against the reference on either sample, and 2619 of 2620
LibriSpeech test-clean hypotheses byte-identical to the reference run.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite5_ctc \
  scripts/convert-granite5_ctc.py ibm-granite/granite-speech-5.0-470m-turboctc \
  --repo-id ibm-granite/granite-speech-5.0-470m-turboctc
```

### Quantize

```bash
uv run scripts/quantize-all.py \
  models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family granite5_ctc \
  --variant granite-speech-5.0-470m-turboctc
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_GRANITE5_CTC_GGUF=models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf \
  ctest --test-dir build --output-on-failure -R granite5_ctc
```
