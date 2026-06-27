# Whisper large-v3

OpenAI's [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) ported to transcribe.cpp. A 1.55B-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (100 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding. v3 family adds Cantonese (yue) on top of v2's 99 languages and switches to a 128-bin mel input.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v3) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`06f233f`](https://huggingface.co/openai/whisper-large-v3/commit/06f233f),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F16    | [whisper-large-v3-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-F16.gguf) | 2.88 GB | 1.81% |
| Q8_0   | [whisper-large-v3-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q8_0.gguf) | 1.55 GB | 1.82% |
| Q6_K   | [whisper-large-v3-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q6_K.gguf) | 1.21 GB | 1.83% |
| Q5_K_M | [whisper-large-v3-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q5_K_M.gguf) | 1.08 GB | 1.84% |
| Q4_K_M | [whisper-large-v3-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q4_K_M.gguf) | 951 MB | 1.86% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with transcribe.cpp's default greedy decode and segment timestamps enabled — the same runs summarized in the [Whisper family table](whisper.md#all-variants). Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions add ~0.1pp of run-to-run variance on the noise floor, and quantization is otherwise generally WER-neutral. See the [WER methodology](../tools/wer.md) for the harness.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  | 508.6 ms (21.6×) | 517.0 ms (21.3×) |
| Metal   | dots (35.3s) |   1.38 s (25.7×) |   1.35 s (26.1×) |
| CPU     | jfk (11.0s)  |    9.68 s (1.1×) |    7.48 s (1.5×) |
| CPU     | dots (35.3s) |   19.86 s (1.8×) |   15.45 s (2.3×) |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-large-v3-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.30 s (1.7×)  |  6.07 s (1.8×)  |
| Vulkan  | dots (35.3s) |  14.42 s (2.5×) |  13.75 s (2.6×) |
| CPU     | jfk (11.0s)  |  25.59 s (0.4×) |  19.96 s (0.6×) |
| CPU     | dots (35.3s) |  53.80 s (0.7×) |  43.18 s (0.8×) |

Fedora 43, transcribe.cpp `e0fa0f6`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-large-v3-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v3.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v3.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v3.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v3.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v3` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `4.055e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `1.574e-03` |   `4.796e-05` | fp32 conv stem |
| `enc.conv2.out`        |  `4.495e-03` |   `6.128e-05` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `1.093e-02` |   `1.163e-03` | first encoder block |
| `enc.block.31.out`     |  `1.439e+00` |   `1.996e-03` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.332e+00` |   `2.135e-03` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `1.465e-02` |   `6.079e-04` | first decoder block, prompt pass |
| `dec.block.31.out`     |  `4.006e-01` |   `1.022e-02` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `1.770e-01` |   `1.271e-02` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `1.371e-01` |   `2.566e-02` | vocab projection (raw logits) |
| `dec.logits`           |  `2.347e-01` |   `1.833e-02` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `2.925e-02` |   `5.238e-03` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-large-v3 \
  --revision 06f233f
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v3/whisper-large-v3-F16.gguf \
  models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v3
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
