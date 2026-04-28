# Whisper tiny.en

OpenAI's [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en) ported to transcribe.cpp. A 39M-parameter
encoder-decoder transformer (audio encoder + autoregressive text decoder with
cross-attention).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and returns a transcript. English-only checkpoints are typically faster and slightly more accurate than the multilingual model at the same parameter count, but they cannot transcribe other languages and cannot translate. Long audio is handled via 30-second chunked decoding.

See the [upstream model card](https://huggingface.co/openai/whisper-tiny.en) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`87c7102`](https://huggingface.co/openai/whisper-tiny.en/commit/87c7102),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [whisper-tiny.en-F32.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-F32.gguf) | 146 MB | 5.77% |
| F16    | [whisper-tiny.en-F16.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-F16.gguf) | 76 MB | 5.78% |
| Q8_0   | [whisper-tiny.en-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q8_0.gguf) | 44 MB | 5.72% |
| Q6_K   | [whisper-tiny.en-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q6_K.gguf) | 43 MB | 5.83% |
| Q5_K_M | [whisper-tiny.en-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q5_K_M.gguf) | 42 MB | 5.91% |
| Q4_K_M | [whisper-tiny.en-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/main/whisper-tiny.en-Q4_K_M.gguf) | 42 MB | 5.96% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 5.66%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-tiny.en/whisper-tiny.en-Q8_0.gguf \
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

### Apple M4

| Backend | Sample      |          F16 |         Q8_0 |        Q4_K_M |
| ------- | ----------- | -----------: | -----------: | ------------: |
| Metal   | jfk (11.0s) | 73 ms (150×) | 69 ms (159×) |  52 ms (213×) |
| CPU     | jfk (11.0s) | 225 ms (49×) | 180 ms (61×) |  164 ms (67×) |

macOS 26.1, transcribe.cpp `11156dd`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-tiny.en \
  --quants f16,q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu \
  --iters 5 --warmup 2 \
  --name whisper-tiny.en-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers
reference (WhisperForConditionalGeneration, fp32 CPU) on the manifest's
case audio. Tolerance budget lives at
[`tests/tolerances/whisper.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper.json).
The reference dtype regime (F32 GGUF + F32 KV cache + production C++ mel
frontend) lights up any frontend regression by design — see the family
note at
[`docs/porting/families/whisper.md`](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/porting/families/whisper.md)
for the full architecture and validation contract.

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-tiny.en.manifest.json` |
| Tolerance file | `tests/tolerances/whisper.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-tiny.en` |

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
  scripts/convert-whisper.py openai/whisper-tiny.en \
  --revision 87c7102
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-tiny.en/whisper-tiny.en-F32.gguf \
  models/whisper-tiny.en/whisper-tiny.en-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-tiny.en
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_WHISPER_GGUF=$PWD/models/whisper-tiny.en/whisper-tiny.en-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
