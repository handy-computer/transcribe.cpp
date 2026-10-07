# Parakeet TDT 0.6B v3

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) at [`6d590f7`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/commit/6d590f7).

Offline multilingual speech-to-text covering 25 European languages. A
Conformer encoder with a TDT/RNNT transducer decoder. Takes a 16 kHz mono
WAV and produces a transcript with optional token-level timestamps. Not a
streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text. The model takes a 16 kHz mono WAV and
produces a transcript with optional token-level timestamps. It is not a
streaming model and does not translate. v3 extends v2's English coverage to
25 European languages: Bulgarian, Croatian, Czech, Danish, Dutch, English,
Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian,
Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian,
Spanish, Swedish, Ukrainian.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`6d590f7`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/commit/6d590f7), pinned 2026-04-16. Validated against the NeMo reference at transcribe.cpp commit [`bf0d0b7`](https://github.com/handy-computer/transcribe.cpp/tree/bf0d0b7) on 2026-04-18.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-tdt-0.6b-v3-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-F32.gguf) | 2.51 GB | 1.95% |
| F16          | [parakeet-tdt-0.6b-v3-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-F16.gguf) | 1.26 GB | 1.95% |
| Q8_0         | [parakeet-tdt-0.6b-v3-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf) |  740 MB | 1.94% |
| Q6_K         | [parakeet-tdt-0.6b-v3-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q6_K.gguf) |  610 MB | 1.93% |
| Q5_K_M       | [parakeet-tdt-0.6b-v3-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q5_K_M.gguf) |  549 MB | 1.92% |
| Q4_K_M       | [parakeet-tdt-0.6b-v3-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q4_K_M.gguf) |  485 MB | 1.98% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy transducer decoding, no external LM. F32 reference baseline: 1.95%. NVIDIA's
self-reported number on the same split is 1.93%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| bg       | WER    | 12.81% |
| cs       | WER    | 12.31% |
| da       | WER    | 18.64% |
| de       | WER    |  5.24% |
| el       | WER    | 35.33% |
| en       | WER    |  4.83% |
| es       | WER    |  3.65% |
| et       | WER    | 17.96% |
| fi       | WER    | 13.30% |
| fr       | WER    |  5.30% |
| hr       | WER    | 12.59% |
| hu       | WER    | 16.06% |
| it       | WER    |  3.02% |
| lt       | WER    | 22.20% |
| lv       | WER    | 23.77% |
| mt       | WER    | 20.63% |
| nl       | WER    |  7.66% |
| pl       | WER    |  7.37% |
| pt       | WER    |  4.96% |
| ro       | WER    | 12.62% |
| ru       | WER    |  6.54% |
| sk       | WER    | 10.19% |
| sl       | WER    | 24.30% |
| sv       | WER    | 15.25% |
| uk       | WER    |  6.84% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  60 ms (181.74×) |  62 ms (178.45×) |
| Metal   | dots (35.3s) | 164 ms (215.45×) | 167 ms (212.12×) |
| CPU     | jfk (11.0s)  |  286 ms (38.48×) |  310 ms (35.53×) |
| CPU     | dots (35.3s) |  1.00 s (35.16×) |  1.08 s (32.71×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 453 ms (24.29×) | 458 ms (24.01×) |
| Vulkan  | dots (35.3s) | 1.37 s (25.84×) | 1.39 s (25.50×) |
| CPU     | jfk (11.0s)  | 729 ms (15.09×) | 794 ms (13.86×) |
| CPU     | dots (35.3s) | 2.89 s (12.22×) | 2.97 s (11.89×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-tdt-0.6b-v3
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`.
All 18 checkpointed tensors fall within family tolerance, and the final
transcript matches the NeMo reference verbatim.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt-0.6b-v3` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt-0.6b-v3.manifest.json` |
| Command | `uv run scripts/validate.py compare --family parakeet --variant parakeet-tdt-0.6b-v3` |

Same divergence profile as v2: C++ runs the STFT in fp64 where NeMo runs fp32.
The gap enters at the mel spectrogram, is amplified through the pre-encoder
and early Conformer blocks, and attenuates to near-zero by the final encoder
output. v3's pre-encode weights amplify the gap ~3.5× more than v2's, but the
24-layer conformer still drives the final encoder output to ~0.03.

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

TRANSCRIBE_PARAKEET_GGUF=models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```
