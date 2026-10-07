# Parakeet TDT-CTC 110M

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) at [`431a349`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/commit/431a349).

Offline English speech-to-text with punctuation and capitalization. A FastConformer encoder with a TDT/RNNT transducer decoder (the auxiliary CTC head from the upstream hybrid checkpoint is dropped at convert time). Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline English speech-to-text, optimised for fast inference at the cost of
some accuracy versus the larger Parakeet variants. Outputs cased,
punctuated transcripts. Token-, word- and segment-level timestamps are
available. The model takes a 16 kHz mono WAV and produces a transcript.
It is not a streaming model and does not translate.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`431a349`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/commit/431a349), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-tdt_ctc-110m-F32.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-F32.gguf) | 457 MB | 2.43% |
| F16          | [parakeet-tdt_ctc-110m-F16.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-F16.gguf) | 229 MB | 2.43% |
| Q8_0         | [parakeet-tdt_ctc-110m-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q8_0.gguf) | 135 MB | 2.43% |
| Q6_K         | [parakeet-tdt_ctc-110m-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q6_K.gguf) | 112 MB | 2.44% |
| Q5_K_M       | [parakeet-tdt_ctc-110m-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q5_K_M.gguf) | 101 MB | 2.47% |
| Q4_K_M       | [parakeet-tdt_ctc-110m-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf) |  90 MB | 2.53% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy TDT/RNN-T transducer decoding, no external LM. F32 reference baseline: 2.43%.
NVIDIA's self-reported number on the same split is 2.40%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 6.11% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  27 ms (405.32×) |  27 ms (401.40×) |
| Metal   | dots (35.3s) |  69 ms (509.69×) |  69 ms (510.03×) |
| CPU     | jfk (11.0s)  |  70 ms (156.94×) |  78 ms (140.61×) |
| CPU     | dots (35.3s) | 252 ms (140.19×) | 279 ms (126.50×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 142 ms (77.29×) | 144 ms (76.32×) |
| Vulkan  | dots (35.3s) | 467 ms (75.62×) | 468 ms (75.48×) |
| CPU     | jfk (11.0s)  | 199 ms (55.36×) | 231 ms (47.53×) |
| CPU     | dots (35.3s) | 790 ms (44.70×) | 870 ms (40.62×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-tdt_ctc-110m
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`
via `scripts/validate.py`, sharing the parakeet family tolerance file. The
shared FastConformer encoder code path is the same as the published
`parakeet-tdt-0.6b-v2` variant; the family-level forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources (fp64 STFT, mel amplification,
attenuation through the encoder).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-tdt_ctc-110m` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-tdt_ctc-110m.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-tdt_ctc-110m` |

## Reproduction

### Convert

Loads directly from NVIDIA's NeMo checkpoint via `ASRModel.from_pretrained`.
Output path is derived from the repo id.

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt_ctc-110m
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0; repeat with
`Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-F32.gguf \
  models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-tdt_ctc-110m
```
