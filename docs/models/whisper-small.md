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
| F32    | [whisper-small-F32.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-F32.gguf) | 924 MB | 3.34% |
| F16    | [whisper-small-F16.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-F16.gguf) | 470 MB | 3.33% |
| Q8_0   | [whisper-small-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q8_0.gguf) | 257 MB | 3.33% |
| Q6_K   | [whisper-small-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q6_K.gguf) | 202 MB | 3.33% |
| Q5_K_M | [whisper-small-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q5_K_M.gguf) | 185 MB | 3.37% |
| Q4_K_M | [whisper-small-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-small-gguf/resolve/main/whisper-small-Q4_K_M.gguf) | 164 MB | 3.40% |

WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 3.432%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled.

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

### Apple M4

| Backend | Sample      |          F16 |         Q8_0 |        Q4_K_M |
| ------- | ----------- | -----------: | -----------: | ------------: |
| Metal   | jfk (11.0s) | 359 ms (31×) | 331 ms (33×) |  324 ms (34×) |
| CPU     | jfk (11.0s) |  1.79 s (6×) |  1.26 s (9×) |   1.34 s (8×) |

macOS 26.1, transcribe.cpp `11156dd`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-small \
  --quants f16,q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu \
  --iters 5 --warmup 2 \
  --name whisper-small-publication
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
| Manifest | `tests/golden/whisper/whisper-small.manifest.json` |
| Tolerance file | `tests/tolerances/whisper.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-small` |

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
