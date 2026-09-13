# Parakeet TDT 0.6B v3

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) at [`6d590f7`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/commit/6d590f7).

Offline multilingual speech-to-text covering 25 European languages. A
0.6B-parameter Conformer encoder with a TDT/RNNT transducer decoder. Takes
a 16 kHz mono WAV and produces a transcript with optional token-level
timestamps. Not a streaming model and does not translate.
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

Licensed CC-BY-4.0. Ported from upstream commit
[`6d590f7`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/commit/6d590f77001d318fb17a0b5bf7ee329a91b52598),
pinned 2026-04-16.

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

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances) with
greedy transducer decoding and no external LM. F32 reference baseline: 1.95%.
NVIDIA's self-reported number on the same split is 1.93%.
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

Cells are compute latency (mel + encode + decode; mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  74 ms (149.59×) |  75 ms (146.35×) |
| Metal   | dots (35.3s) | 224 ms (157.78×) | 224 ms (157.68×) |
| CPU     | jfk (11.0s)  |  386 ms (28.53×) |     323 ms (34×) |
| CPU     | dots (35.3s) |  1.31 s (26.98×) |     1.11 s (32×) |
<!-- /catalog -->

macOS 26.4.1, transcribe.cpp `12f1076`.

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 854 ms (12.88×) | 864 ms (12.72×) |
| Vulkan  | dots (35.3s) | 3.06 s (11.54×) | 3.10 s (11.42×) |
| CPU     | jfk (11.0s)  |  1.41 s (7.80×) |  1.22 s (9.01×) |
| CPU     | dots (35.3s) |     5.34 s (7×) |     4.78 s (7×) |
<!-- /catalog -->

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-tdt-0.6b-v3 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-tdt-0.6b-v3-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`.
All 18 checkpointed tensors fall within family tolerance, and the final
transcript matches the NeMo reference verbatim. Last validated at commit
[`bf0d0b7`](https://github.com/handy-computer/transcribe.cpp/tree/bf0d0b7).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt-0.6b-v3` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt-0.6b-v3.manifest.json` |
| Command | `uv run scripts/validate.py compare --family parakeet --variant parakeet-tdt-0.6b-v3` |

Selected tensors:

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`          | `5.189e+00` | `1.639e-03` | fp64 vs fp32 STFT precision gap |
| `enc.pre_encode.out`  | `6.940e+03` | `2.438e+02` | Mel gap amplified through pre-encoder (v3 amplifies more aggressively than v2) |
| `enc.block.0.out`     | `1.296e+03` | `1.468e+01` | Early encoder, still amplified |
| `enc.block.12.out`    | `1.285e+03` | `1.433e+01` | Mid-encoder |
| `enc.block.23.out`    | `3.055e-02` | `3.040e-04` | Converged by final block |
| `enc.final`           | `3.055e-02` | `3.040e-04` | Final encoder output |
| `dec.enc_out`         | `3.055e-02` | `3.040e-04` | Decoder input from encoder |
| `dec.embed.0`         | `0.000e+00` | `0.000e+00` | Exact match |
| `dec.lstm.*`          | `<= 1.192e-07` | near zero | fp32 round-off on first step |
| `dec.joint.0`         | `1.190e+01` | `1.098e+01` | Joint projection over encoder drift |

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
