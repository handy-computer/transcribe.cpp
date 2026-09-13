# Granite Speech 5.0 470M TurboCTC

IBM's [`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
ported to transcribe.cpp. A 470M-parameter Granite Conformer encoder with a
self-conditioned CTC head.

## What it's for

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. English only, and it does not translate.

Licensed Apache-2.0. Ported from upstream commit
[`18ca3c1`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/commit/18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0),
pinned 2026-09-12.

There is also a non-commercial sibling,
[`granite-speech-5.0-470m-turboctc-nc`](granite-speech-5.0-470m-turboctc-nc.md),
trained on more data and slightly more accurate. It is CC-BY-NC-SA-4.0, so use
this one for anything commercial.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-speech-5.0-470m-turboctc-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-BF16.gguf)     | 948 MB | 1.33% |
| F16    | [granite-speech-5.0-470m-turboctc-F16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-F16.gguf)       | 948 MB | 1.33% |
| Q8_0   | [granite-speech-5.0-470m-turboctc-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q8_0.gguf)     | 506 MB | 1.34% |
| Q6_K   | [granite-speech-5.0-470m-turboctc-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q6_K.gguf)     | 392 MB | 1.33% |
| Q5_K_M | [granite-speech-5.0-470m-turboctc-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q5_K_M.gguf) | 336 MB | 1.34% |
| Q4_K_M | [granite-speech-5.0-470m-turboctc-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q4_K_M.gguf) | 279 MB | 1.34% |

Measured on the full LibriSpeech test-clean split (2620 utterances), greedy CTC
decoding, no external LM. Reference baseline (transformers 5.17.0, F32, CPU):
**1.33%**, 95% CI [1.20%, 1.47%]. Every tier falls inside that CI, so the
ordering between them is not meaningful and quantizing down to Q4_K_M costs
very little. Clean read speech only, not checked on noisy or accented audio.

## Quick Start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-Q8_0.gguf \
  samples/jfk.wav
```

## Performance

Wall-clock latency (mean over 3 iterations after 1 warmup), with speedup over
realtime in parentheses.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  | 37.5 ms (293×) | 38.9 ms (283×) |
| Metal   | dots (35.3s) | 85.6 ms (413×) | 87.9 ms (402×) |
| CPU     | jfk (11.0s)  |  232 ms (48×)  |  233 ms (47×)  |
| CPU     | dots (35.3s) |  703 ms (50×)  |  689 ms (51×)  |

macOS 26.6.2, transcribe.cpp `54b241e`.

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  652 ms (17×) |  647 ms (17×) |
| Vulkan  | dots (35.3s) | 1.48 s (24×)  | 1.51 s (23×)  |
| CPU     | jfk (11.0s)  |  696 ms (16×) |  663 ms (17×) |
| CPU     | dots (35.3s) | 2.28 s (16×)  | 2.25 s (16×)  |

Fedora 43, transcribe.cpp `3a5ed01`. Vulkan device: `AMD Radeon Graphics (RADV RENOIR)`.

### Apple M4

| Backend | Sample            |          Q8_0 |        Q4_K_M |
| ------- | ----------------- | ------------: | ------------: |
| Metal   | jfk (11.0s)       |  100 ms (110×) |  103 ms (107×) |
| Metal   | dots (35.3s)      |  272 ms (130×) |  279 ms (127×) |
| Metal   | dots-full (305.9s)|  2.33 s (131×) |  2.38 s (128×) |
| CPU     | jfk (11.0s)       |  388 ms (28×)  |  423 ms (26×)  |
| CPU     | dots (35.3s)      |  1.19 s (30×)  |  1.27 s (28×)  |
| CPU     | dots-full (305.9s)|  8.30 s (37×)  |  9.73 s (31×)  |

macOS 26.5.1, transcribe.cpp `f2d5e31`.

Q8_0 is usually a little faster than Q4_K_M despite being 1.8× the size, so
pick Q4_K_M for footprint rather than speed. Cost is linear in audio length:
realtime factor is flat from 11 s out to 5 minutes. On CPU, avoid BF16 — that
matmul kernel does not thread; use any quantized tier instead. Batching is
supported and gives identical output, but buys almost nothing here.

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against Hugging Face transformers
on `samples/jfk.wav` and `samples/dots.wav`. All 35 checkpointed tensors fall
within family tolerance, and the final transcript matches the reference
verbatim. Last validated at commit
[`b9427cf`](https://github.com/handy-computer/transcribe.cpp/tree/b9427cf).

| Field | Value |
| --- | --- |
| Reference | transformers 5.17.0, `ibm-granite/granite-speech-5.0-470m-turboctc` |
| Dump script | `scripts/dump_reference_granite5_ctc_transformers.py` |
| Manifest | `tests/golden/granite5_ctc/granite-speech-5.0-470m-turboctc.manifest.json` |
| Tolerances | `tests/tolerances/granite5_ctc.json` |
| Command | `uv run scripts/validate.py all --family granite5_ctc --variant granite-speech-5.0-470m-turboctc` |

Selected tensors (the `dots` case, 441 output frames):

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `mel.in`               | `8.821e-06` | `1.007e-07` | Production C++ frontend, no injection |
| `enc.input_linear.out` | `1.372e-02` | `1.097e-03` | 0.18% of RMS |
| `enc.block.7.out`      | `1.316e-02` | `1.011e-03` | Before the self-conditioning injection |
| `enc.ctc.mid_logits`   | `6.251e-02` | `3.951e-03` | Mid-layer CTC head |
| `enc.block.15.out`     | `2.190e-02` | `1.370e-03` | Final encoder output |
| `enc.ctc_logits`       | `1.513e-01` | `5.442e-03` | 0.06% of RMS |

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
