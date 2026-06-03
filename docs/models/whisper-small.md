# Whisper small

OpenAI's [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) ported to transcribe.cpp. A 244M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-small) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`973afd2`](https://huggingface.co/openai/whisper-small/commit/973afd2),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-small-F32.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-F32.gguf) | 924 MB | 3.39% |
| F16    | [whisper-small-F16.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-F16.gguf) | 470 MB | 3.48% |
| Q8_0   | [whisper-small-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q8_0.gguf) | 257 MB | 3.38% |
| Q6_K   | [whisper-small-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q6_K.gguf) | 202 MB | 3.40% |
| Q5_K_M | [whisper-small-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q5_K_M.gguf) | 185 MB | 3.34% |
| Q4_K_M | [whisper-small-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q4_K_M.gguf) | 164 MB | 3.44% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the pinned short-form recipe: greedy decode, timestamps off (`<|notimestamps|>`), and language forced to `en` — see [WER methodology](../tools/wer.md#methodology-pinned-recipe). Captured on a single CUDA (L40S) run at batch size 1; quantization, backend, and batching are all generally WER-neutral.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-small/whisper-small-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mel + encode + decode, mean over the recorded
iterations after warmup), with speedup over realtime in parentheses. Units:
`ms` below 1 s, `s` above (2 decimal places). Decode latency dominates as
model size grows; the encoder is only run once per 30-second window.

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  | 113.1 ms (97.2×) | 113.5 ms (96.9×) |
| Metal   | dots (35.3s) | 349.3 ms (101.2×) | 340.0 ms (103.9×) |
| CPU     | jfk (11.0s)  |    1.43 s (7.7×) |    1.30 s (8.4×) |
| CPU     | dots (35.3s) |   3.01 s (11.8×) |   2.74 s (12.9×) |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-small \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-small-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  1.03 s (10.6×) |  0.96 s (11.4×) |
| Vulkan  | dots (35.3s) |  2.57 s (13.7×) |  2.47 s (14.3×) |
| CPU     | jfk (11.0s)  |  3.95 s (2.8×)  |  3.27 s (3.4×)  |
| CPU     | dots (35.3s) |  8.91 s (4.0×)  |  7.47 s (4.7×)  |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-small \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-small-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-small.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-small.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-small.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-small.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-small` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `3.381e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `5.454e-06` |   `4.815e-08` | fp32 conv stem |
| `enc.conv2.out`        |  `2.146e-05` |   `2.704e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `2.134e-05` |   `6.858e-07` | first encoder block |
| `enc.block.11.out`     |  `1.669e-02` |   `6.139e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `2.100e-03` |   `3.882e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `4.768e-06` |   `2.997e-07` | first decoder block, prompt pass |
| `dec.block.11.out`     |  `3.662e-04` |   `2.471e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.755e-04` |   `7.318e-06` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `3.719e-05` |   `7.083e-06` | vocab projection (raw logits) |
| `dec.logits`           |  `8.965e-05` |   `1.926e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `2.289e-05` |   `8.815e-06` | step-20 logits (KV-cached path) |

The C++ mel frontend (Slaney filterbank + Hann periodic window +
whisper-style log-mel compression) drives `enc.mel.in` to fp32-vs-fp64
STFT precision drift; downstream tensors stay within budget. KV-cached
decoder runs through F16 self/cross caches by default — flip with
`--kv-type f32` for tighter parity.

## Reproduction

### Convert

The whisper converter loads from a Hugging Face checkpoint and emits a
reference-dtype GGUF.

```bash
uv run --project scripts/envs/whisper \
  scripts/convert-whisper.py openai/whisper-small \
  --revision 973afd2
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-small/whisper-small-F32.gguf \
  models/whisper-small/whisper-small-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-small
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-small/whisper-small-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
