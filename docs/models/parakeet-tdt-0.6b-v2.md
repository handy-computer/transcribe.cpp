# Parakeet TDT 0.6B v2

NVIDIA's [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
ported to transcribe.cpp. A 0.6B-parameter Conformer encoder with a TDT/RNNT
transducer decoder.

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and produces
a transcript with optional token-level timestamps. It is not a streaming model,
does not translate, and v2 has no multilingual capability. For multilingual
see v3.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
for training data, intended use, and upstream evaluation methodology.

Licensed CC-BY-4.0. Ported from upstream commit
[`1b149a3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/commit/1b149a3589351c96ddb101709fe7dd9c7069572f),
pinned 2026-04-15.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-tdt-0.6b-v2-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-F32.gguf)       | 2.47 GB | 1.68% |
| F16    | [parakeet-tdt-0.6b-v2-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-F16.gguf)       | 1.24 GB | 1.68% |
| Q8_0   | [parakeet-tdt-0.6b-v2-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q8_0.gguf)     | 730 MB  | 1.69% |
| Q6_K   | [parakeet-tdt-0.6b-v2-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q6_K.gguf)     | 608 MB  | 1.70% |
| Q5_K_M | [parakeet-tdt-0.6b-v2-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q5_K_M.gguf) | 547 MB  | 1.69% |
| Q4_K_M | [parakeet-tdt-0.6b-v2-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf) | 483 MB  | 1.72% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy transducer decoding and no external LM. F32 reference baseline:
1.68%. NVIDIA's self-reported number on the same split is 1.69% (from the
[HF model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)), so our
F32 and Q8_0 ports match the upstream reference within rounding.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet/parakeet-tdt-0.6b-v2.f16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |   68 ms (161×) |   68 ms (161×) |
| Metal   | dots (35.3s) |  190 ms (186×) |  191 ms (185×) |
| CPU     | jfk (11.0s)  |  373 ms (29×)  |  315 ms (35×)  |
| CPU     | dots (35.3s) | 1.27 s (28×)   | 1.10 s (32×)   |

macOS 26.3.1, transcribe.cpp `3912397`. Raw data:
`reports/perf/apple-m4-max/post-unification_parakeet_{metal,cpu}.json`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  546 ms (20×)  |  586 ms (19×)  |
| Vulkan  | dots (35.3s) | 1.75 s (20×)   | 1.76 s (20×)   |
| CPU     | jfk (11.0s)  | 1.19 s (9×)    | 1.04 s (11×)   |
| CPU     | dots (35.3s) | 4.22 s (8×)    | 3.66 s (10×)   |

Fedora 43, transcribe.cpp `140ed3a`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`. Raw data:
`reports/perf/amd-ryzen-7-4750u-pro/parakeet-tdt-0-6b-v2-publication_parakeet_{cpu,vulkan}.json`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --family parakeet \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-tdt-0.6b-v2-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`.
All 18 checkpointed tensors fall within family tolerance, and the final
transcript matches the NeMo reference verbatim. Last validated at commit
[`3912397`](https://github.com/handy-computer/transcribe.cpp/tree/3912397).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt-0.6b-v2` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt-0.6b-v2.manifest.json` |
| Command | `uv run scripts/validate.py compare --family parakeet` |
| Artifact | `reports/numerics/parakeet-tdt-0.6b-v2.json` |

Selected tensors:

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`          | `5.189e+00` | `1.639e-03` | fp64 vs fp32 STFT precision gap |
| `enc.pre_encode.out`  | `2.011e+03` | `6.590e+01` | Mel gap amplified through pre-encoder |
| `enc.block.0.out`     | `1.016e+03` | `1.396e+01` | Early encoder, still amplified |
| `enc.block.12.out`    | `1.040e+03` | `1.330e+01` | Mid-encoder |
| `enc.block.23.out`    | `4.392e-02` | `1.169e-03` | Converged by final block |
| `enc.final`           | `4.392e-02` | `1.169e-03` | Final encoder output |
| `dec.enc_out`         | `4.392e-02` | `1.169e-03` | Decoder input from encoder |
| `dec.embed.0`         | `0.000e+00` | `0.000e+00` | Exact match |
| `dec.lstm.*`          | `<= 1.788e-07` | near zero | fp32 round-off on first step |
| `dec.joint.0`         | `7.033e+01` | `6.956e+01` | Joint projection over encoder drift |

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

TRANSCRIBE_REAL_PARAKEET_GGUF=models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```
