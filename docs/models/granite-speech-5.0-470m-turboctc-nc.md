# Granite Speech 5.0 470M TurboCTC NC

<!-- catalog:intro -->
Upstream: [`ibm-granite/granite-speech-5.0-470m-turboctc-nc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc) at [`0eb7b4f`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc/commit/0eb7b4f).

Offline English speech-to-text, research and non-commercial use only. A Granite Conformer
encoder with a self-conditioned CTC head. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. English only.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. English only, and it does not translate.

Same architecture as the Apache-2.0 sibling, trained on more data (~75,000 h vs
~60,000 h). IBM reports 4.85% aggregate WER across the 8 Open ASR leaderboard
test sets for this model, against 5.00% for the sibling.

<!-- catalog:pin -->
Licensed CC-BY-NC-SA-4.0. Ported from upstream commit [`0eb7b4f`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc/commit/0eb7b4f), pinned 2026-09-12. Validated against the transformers reference at transcribe.cpp commit [`f1d0e10`](https://github.com/handy-computer/transcribe.cpp/tree/f1d0e10) on 2026-09-12.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [granite-speech-5.0-470m-turboctc-nc-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-BF16.gguf) | 948 MB | 1.29% |
| F16          | [granite-speech-5.0-470m-turboctc-nc-F16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-F16.gguf) | 949 MB | 1.28% |
| Q8_0         | [granite-speech-5.0-470m-turboctc-nc-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q8_0.gguf) | 506 MB | 1.29% |
| Q6_K         | [granite-speech-5.0-470m-turboctc-nc-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q6_K.gguf) | 392 MB | 1.29% |
| Q5_K_M       | [granite-speech-5.0-470m-turboctc-nc-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q5_K_M.gguf) | 336 MB | 1.29% |
| Q4_K_M       | [granite-speech-5.0-470m-turboctc-nc-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q4_K_M.gguf) | 279 MB | 1.34% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none, language hint `en`, decoded on cuda. Measured at transcribe.cpp `9daf396`.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy CTC decoding, no external LM. Measured reference baseline (transformers
5.17.0, F32, CPU): 1.29%, 95% CI [1.15, 1.42].
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.30% |
<!-- /catalog -->

## Quick Start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc-nc/granite-speech-5.0-470m-turboctc-nc-Q8_0.gguf \
  samples/jfk.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 35 ms (312.54×) | 37 ms (295.13×) |
| Metal   | dots (35.3s) | 83 ms (424.15×) | 87 ms (407.09×) |
| CPU     | jfk (11.0s)  | 244 ms (45.18×) | 255 ms (43.17×) |
| CPU     | dots (35.3s) | 746 ms (47.39×) | 764 ms (46.26×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 583 ms (18.88×) | 605 ms (18.18×) |
| Vulkan  | dots (35.3s) | 1.32 s (26.77×) | 1.34 s (26.45×) |
| CPU     | jfk (11.0s)  | 614 ms (17.91×) | 653 ms (16.84×) |
| CPU     | dots (35.3s) | 2.13 s (16.59×) | 2.12 s (16.63×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Q8_0 is usually a little faster than Q4_K_M despite being 1.8× the size, so
pick Q4_K_M for footprint rather than speed. Cost is linear in audio length, so
realtime factor holds up on long files. Batching is supported and gives
identical output, but buys almost nothing here.

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against Hugging Face transformers
on `samples/jfk.wav` and `samples/dots.wav`. All 35 checkpointed tensors fall
within variant tolerance, and the final transcript matches the reference
verbatim.

| Field | Value |
| --- | --- |
| Reference | transformers 5.17.0, `ibm-granite/granite-speech-5.0-470m-turboctc-nc` |
| Dump script | `scripts/dump_reference_granite5_ctc_transformers.py` |
| Manifest | `tests/golden/granite5_ctc/granite-speech-5.0-470m-turboctc-nc.manifest.json` |
| Tolerances | `tests/tolerances/granite5_ctc-nc.json` |
| Command | `uv run scripts/validate.py all --family granite5_ctc --variant granite-speech-5.0-470m-turboctc-nc` |

Drift is BF16 matmul accumulation, amplified by a massive-activation channel in
encoder blocks 10-13 that sits ~20x above the rest of its tensor. It is
concentrated on single frames rather than spread out, and it does not reach the
output: zero argmax differences against the reference on either sample, and
2618 of 2620 LibriSpeech test-clean hypotheses byte-identical to the reference
run. Running the reference itself in BF16 puts PyTorch further from its own F32
than transcribe.cpp is.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite5_ctc \
  scripts/convert-granite5_ctc.py ibm-granite/granite-speech-5.0-470m-turboctc-nc \
  --repo-id ibm-granite/granite-speech-5.0-470m-turboctc-nc \
  --revision 0eb7b4fe726a294815dc45d342860465b5af68ef
```

### Quantize

```bash
uv run scripts/quantize-all.py \
  models/granite-speech-5.0-470m-turboctc-nc/granite-speech-5.0-470m-turboctc-nc-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family granite5_ctc \
  --variant granite-speech-5.0-470m-turboctc-nc
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_GRANITE5_CTC_GGUF=models/granite-speech-5.0-470m-turboctc-nc/granite-speech-5.0-470m-turboctc-nc-BF16.gguf \
  ctest --test-dir build --output-on-failure -R granite5_ctc
```
