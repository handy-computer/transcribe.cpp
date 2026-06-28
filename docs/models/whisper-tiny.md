# Whisper tiny

OpenAI's [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) ported to transcribe.cpp. A 39M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (99 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-tiny) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`169d4a4`](https://huggingface.co/openai/whisper-tiny/commit/169d4a4),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-tiny-F32.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-F32.gguf) | 146 MB | 7.54% |
| F16    | [whisper-tiny-F16.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-F16.gguf) | 76 MB | 7.49% |
| Q8_0   | [whisper-tiny-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q8_0.gguf) | 44 MB | 7.53% |
| Q6_K   | [whisper-tiny-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q6_K.gguf) | 43 MB | 7.63% |
| Q5_K_M | [whisper-tiny-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q5_K_M.gguf) | 42 MB | 7.63% |
| Q4_K_M | [whisper-tiny-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q4_K_M.gguf) | 42 MB | 7.76% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with transcribe.cpp's default greedy decode and segment timestamps enabled — the same runs summarized in the [Whisper family table](whisper.md#all-variants). Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions add ~0.1pp of run-to-run variance on the noise floor, and quantization is otherwise generally WER-neutral. See the [WER methodology](../tools/wer.md) for the harness.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-tiny/whisper-tiny-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places). Decode latency dominates as model size grows; the encoder
is only run once per 30-second window.

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  36.4 ms (302.5×) |  37.9 ms (290.1×) |
| Metal   | dots (35.3s) | 117.1 ms (301.8×) | 117.3 ms (301.3×) |
| CPU     | jfk (11.0s)  | 174.7 ms (63.0×)  | 169.8 ms (64.8×)  |
| CPU     | dots (35.3s) | 396.3 ms (89.2×)  | 390.4 ms (90.5×)  |

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-tiny \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-tiny-publication
```

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  200 ms (55.1×) |  209 ms (52.6×) |
| Vulkan  | dots (35.3s) |  528 ms (66.9×) |  529 ms (66.8×) |
| CPU     | jfk (11.0s)  |  531 ms (20.7×) |  467 ms (23.6×) |
| CPU     | dots (35.3s) |  1.23 s (28.8×) |  1.14 s (31.0×) |

Fedora 43, transcribe.cpp `2ab01b8`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-tiny \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-tiny-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's cases (`samples/jfk.wav` and `samples/german.wav`). All 21 checkpointed tensors fall within per-variant tolerance, and the transcripts match the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-tiny.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-tiny.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-tiny.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-tiny.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-tiny` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `3.946e-05` |   `1.396e-07` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `1.365e-05` |   `1.140e-07` | fp32 conv stem |
| `enc.conv2.out`        |  `1.550e-05` |   `2.354e-07` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `1.407e-05` |   `5.994e-07` | first encoder block |
| `enc.block.3.out`      |  `8.335e-03` |   `4.857e-06` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `2.285e-03` |   `3.926e-06` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `6.676e-06` |   `2.617e-07` | first decoder block, prompt pass |
| `dec.block.3.out`      |  `1.144e-04` |   `1.228e-06` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `3.204e-04` |   `1.665e-05` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `6.294e-05` |   `1.858e-05` | vocab projection (raw logits) |
| `dec.logits`           |  `7.248e-05` |   `2.055e-05` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `4.959e-05` |   `1.305e-05` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-tiny \
  --revision 169d4a4
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-tiny/whisper-tiny-F32.gguf \
  models/whisper-tiny/whisper-tiny-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-tiny
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-tiny/whisper-tiny-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
