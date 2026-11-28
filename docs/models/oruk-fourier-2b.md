# Oruk Fourier 2B

Upstream: [`oruk/oruk-fourier-2b`](https://huggingface.co/oruk/oruk-fourier-2b) at [`8c2a887`](https://huggingface.co/oruk/oruk-fourier-2b/commit/8c2a8871f1304aa2f634da4fb95e24438c4945b9).

English-specialized post-training of Qwen3-ASR 1.7B (2.04B parameters) by Oruk AI. Uses the identical `qwen3_asr` GGML architecture as `qwen3-asr-1.7b` (24-layer bidirectional audio encoder `d_model=1024`, 28-layer Qwen3 causal LM `hidden_size=2048`, tied word embeddings) and runs out of the box with `transcribe.cpp`'s existing `qwen3_asr` loader — 16 kHz mono WAV in, English transcript text out.

Licensed **CC-BY-NC-4.0** (non-commercial open weights; see the [Oruk Fourier 2B model card](https://huggingface.co/oruk/oruk-fourier-2b#license) for license terms).

## Download

A pre-built single-file Q8_0 GGUF validated on `transcribe.cpp v0.4.2` is published directly at the upstream Hugging Face repository:

| Quantization | Download | Size | SHA-256 | LibriSpeech test-clean (WER) |
| --- | --- | ---: | --- | ---: |
| Q8_0 | [Fourier-2B-Q8_0.gguf](https://huggingface.co/oruk/oruk-fourier-2b/resolve/main/Fourier-2B-Q8_0.gguf) | 2.19 GB | `6a7c419b…22981` | **1.40%** |

## Quick start

```bash
curl -L -o models/oruk-fourier-2b/Fourier-2B-Q8_0.gguf \
  https://huggingface.co/oruk/oruk-fourier-2b/resolve/main/Fourier-2B-Q8_0.gguf

build/bin/transcribe-cli \
  -m models/oruk-fourier-2b/Fourier-2B-Q8_0.gguf \
  --language en \
  samples/jfk.wav
```

## Accuracy (Q8_0, `transcribe.cpp v0.4.2`)

Measured with `transcribe.cpp v0.4.2` (`275cd50d`) on 1× NVIDIA L40S (`cuda`, `batch_size=8`, `language=en`, `sort_by_length=true`, `timestamps=none`) using Whisper-style English normalization (`evaluations/q8-english/results.json` on Hugging Face):

| Dataset | Split | Utterances | `oruk-fourier-2b` Q8_0 | `qwen3-asr-1.7b` Q8_0 | Errors (`S / D / I`) |
| --- | --- | ---: | ---: | ---: | --- |
| LibriSpeech | `test-clean` | 2,620 | **1.40%** (95% CI 1.26–1.53%) | 1.62% (95% CI 1.47–1.76%) | `563 / 111 / 64` (0 empty) |
| FLEURS (`en_us`) | `test` | 647 | 3.58% (95% CI 3.18–4.00%) | 3.23% (95% CI 2.85–3.65%) | `327 / 90 / 101` (0 empty) |

Both models share the exact same 2.19 GB (`2,185,030,848` bytes) Q8_0 footprint.

## Speed (Apple M5 Max, `transcribe-bench`)

Measured with `uv run scripts/bench/run.py --models oruk-fourier-2b --samples jfk,dots --iters 3 --warmup 1 --backends metal,cpu` at commit `a0729fa` on Apple M5 Max (128 GB unified memory, macOS 26.4.1):

| Backend | Sample | Duration | Load | Encode | Decode | Wall | xRT (wall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal (`MTL0`) | `jfk` | 11.0 s | 150.8 ms | 25.0 ms | 189.6 ms | 217.7 ms | **50.52×** |
| Metal (`MTL0`) | `dots` | 35.3 s | 133.0 ms | 54.5 ms | 757.3 ms | 818.6 ms | **43.16×** |
| CPU | `jfk` | 11.0 s | 107.9 ms | 384.4 ms | 934.0 ms | 1,325.3 ms | **8.30×** |
| CPU | `dots` | 35.3 s | 129.9 ms | 1,274.0 ms | 3,625.5 ms | 4,911.6 ms | **7.19×** |

*(Not yet in `catalog/` pending maintainer reference-fleet speed measurements on `m4`, `m4-max`, and `ryzen-4750u` for `asr-publication-v2`.)*

## Capabilities and limits

Same `qwen3_asr` runtime contract and ~87-minute single-call context cap (`65,536` decoder tokens) as `qwen3-asr-1.7b`, scoped to **English** (`--language en`). Supports `--vocabulary` and `--prompt`. Does not support translation, streaming, diarization, or timestamps.
