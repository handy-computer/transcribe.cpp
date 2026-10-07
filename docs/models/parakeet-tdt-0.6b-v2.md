# Parakeet TDT 0.6B v2

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) at [`1b149a3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/commit/1b149a3).

Offline English speech-to-text. A Conformer encoder with a TDT/RNNT transducer
decoder. Takes a 16 kHz mono WAV and produces a transcript with optional
token-level timestamps. Not a streaming model; no multilingual capability (see
v3 for that).
<!-- /catalog -->

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and produces
a transcript with optional token-level timestamps. It is not a streaming model,
does not translate, and v2 has no multilingual capability. For multilingual
see v3.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`1b149a3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/commit/1b149a3), pinned 2026-04-15. Validated against the NeMo reference at transcribe.cpp commit [`bf0d0b7`](https://github.com/handy-computer/transcribe.cpp/tree/bf0d0b7) on 2026-04-18.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-tdt-0.6b-v2-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-F32.gguf) | 2.47 GB | 1.68% |
| F16          | [parakeet-tdt-0.6b-v2-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-F16.gguf) | 1.24 GB | 1.68% |
| Q8_0         | [parakeet-tdt-0.6b-v2-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q8_0.gguf) |  730 MB | 1.69% |
| Q6_K         | [parakeet-tdt-0.6b-v2-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q6_K.gguf) |  600 MB | 1.70% |
| Q5_K_M       | [parakeet-tdt-0.6b-v2-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q5_K_M.gguf) |  539 MB | 1.70% |
| Q4_K_M       | [parakeet-tdt-0.6b-v2-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf) |  475 MB | 1.72% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy transducer decoding, no external LM. F32 reference baseline: 1.68%. NVIDIA's
self-reported number on the same split is 1.69%, so the F32 and Q8_0 ports match the
upstream reference within rounding.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 4.11% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  55 ms (199.95×) |  56 ms (196.11×) |
| Metal   | dots (35.3s) | 142 ms (248.28×) | 145 ms (243.28×) |
| CPU     | jfk (11.0s)  |  278 ms (39.64×) |  318 ms (34.64×) |
| CPU     | dots (35.3s) |  987 ms (35.80×) |  1.09 s (32.53×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 416 ms (26.45×) | 418 ms (26.34×) |
| Vulkan  | dots (35.3s) | 1.24 s (28.42×) | 1.26 s (28.09×) |
| CPU     | jfk (11.0s)  | 696 ms (15.81×) | 749 ms (14.69×) |
| CPU     | dots (35.3s) | 2.77 s (12.73×) | 2.83 s (12.48×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-tdt-0.6b-v2
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`.
All 18 checkpointed tensors fall within family tolerance, and the final
transcript matches the NeMo reference verbatim.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt-0.6b-v2` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt-0.6b-v2.manifest.json` |
| Command | `uv run scripts/validate.py compare --family parakeet` |

The expected divergence is in the frontend: C++ runs the STFT in fp64 where
NeMo runs fp32. The gap enters at the mel spectrogram, is amplified through
the pre-encoder and early Conformer blocks, and attenuates to near-zero by the
final encoder output. Decoder LSTM state is encoder-independent on the first
step and matches at fp32 round-off.

## Reproduction

### Convert

Loads directly from NVIDIA's NeMo checkpoint via `ASRModel.from_pretrained`.
Output path is derived from the repo id.

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v2
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_PARAKEET_GGUF=models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```
