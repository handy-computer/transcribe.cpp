# Nemotron 3.5 ASR Streaming 0.6B

NVIDIA's [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
ported to transcribe.cpp. A 0.6B-parameter cache-aware streaming
FastConformer encoder with an RNN-T transducer decoder — the multilingual
successor to
[`nemotron-speech-streaming-en-0.6b`](nemotron-speech-streaming-en-0.6b.md).

## What it's for

Multilingual speech-to-text across **32 supported language-locales** (19
transcription-ready + 13 broad-coverage; the tokenizer also recognizes 8
adaptation-ready locales that require fine-tuning — see the upstream model
card for the full list) with greedy RNN-T decoding.
Outputs cased, punctuated transcripts (native PnC). Token- and word-level
timestamps are available.

The target language is selected per call (`--language en-US`,
`fr-FR`, `de-DE`, …). The model also supports `auto` language
detection, in which case it emits a `<lang-XX>` tag in the transcript.
A language **must** be provided — there is no implicit default; passing
an unsupported tag returns "unsupported language".

The encoder is cache-aware and trained with four runtime latency
settings (`att_context_size` ∈ `[56, 0]` / `[56, 3]` / `[56, 6]` /
`[56, 13]` = 0 / 240 / 480 / 1040 ms lookahead). Both paths ship: the
**offline** path (`transcribe_run`) defaults to `[56, 13]` (1.12 s) for
the headline accuracy, and **chunked streaming** is selectable at runtime
via `--stream-chunk-ms 1120 --stream-att-right {0,3,6,13}`. Streaming at
R=13 is byte-equal to the offline transcript; lower-R settings trade
lookahead for latency. Accuracy and latency numbers below are for the
offline `[56, 13]` path.

See NVIDIA's [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
for training data, the full language list, intended use, and the
latency-vs-accuracy table.

Licensed under [OpenMDW-1.1](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).
Ported from upstream commit
[`24b151a`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/24b151a851dd15909e1fc611b11bb2da52b9fc81),
pinned 2026-06-08.

## Input limits

No practical per-call length limit (`transcribe_capabilities.max_audio_ms == 0`):
the FastConformer encoder's positional encoding is recomputed per call and the
RNN-T transducer has no decoder context window, so audio of any length is
processed in a single pass — pass arbitrarily long recordings. `n_ctx` is a
no-op for this model — there is no context/KV ceiling to lower. The cache-aware
streaming path carries constant-memory caches rather than a growing KV, so it
stays unbounded for the same reason. See the
[input-length contract](../input-limits.md).

## Download

| Quantization | Download | Size |
| --- | --- | ---: |
| F32    | [nemotron-3.5-asr-streaming-0.6b-F32.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-F32.gguf) | 2.38 GB |
| F16    | [nemotron-3.5-asr-streaming-0.6b-F16.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-F16.gguf) | 1.19 GB |
| Q8_0   | [nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf) | 716 MB |
| Q6_K   | [nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf) | 593 MB |
| Q5_K_M | [nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf) | 534 MB |
| Q4_K_M | [nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf) | 473 MB |

**Accuracy.** Word error rate at the offline `att_context_size=[56,13]`
(1.12 s) setting, `--language en-US`, greedy RNN-T. C++ hypotheses were
generated on an L4 GPU and scored with the whisper-normalizer; the
reference column is NVIDIA NeMo measured on the same manifests. For
context, NVIDIA's self-reported FLEURS en-US WER is **7.91%** (and an
**8.84%** 19-locale macro-average) per the
[HF model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

| Preset | FLEURS test en (n=647) | LibriSpeech test-clean (n=2620) |
| --- | ---: | ---: |
| Reference (NeMo) | 7.99 | 3.03 |
| F32    | 7.97 | 3.04 |
| F16    | 7.97 | 3.03 |
| Q8_0   | 7.88 | 3.06 |
| Q6_K   | 8.02 | 3.07 |
| Q5_K_M | 8.15 | 3.10 |
| Q4_K_M | 8.49 | 3.28 |

The F32 reference dtype meets the measured-Oracle gate on both datasets.
F16/Q8_0/Q6_K/Q5_K_M land inside the reference 95% CI; Q4_K_M carries the
largest quantization loss (+0.50 on FLEURS, +0.25 on LibriSpeech) but is
accepted for shipping.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
  --language en-US \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places). Cells gated on `Tctl < 55°C` per backend.

The decoder runs through a reused ggml graph for the joint output
projection (the 13k-vocab RNN-T joint that dominates this variant's
decode) and a thread-parallel predictor; both are the default, so these
are out-of-the-box numbers with no tuning.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  113 ms (98×) |  113 ms (98×) |
| Metal   | dots (35.3s) |  361 ms (98×) |  368 ms (96×) |
| CPU     | jfk (11.0s)  |  367 ms (30×) |  362 ms (30×) |
| CPU     | dots (35.3s) |  1.28 s (28×) |  1.25 s (28×) |

macOS 26.5 (Darwin 25.5.0), transcribe.cpp `d9708f1`. Metal device:
Apple M4 Max (`MTLGPUFamilyApple9`).

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  773 ms (14×) |  783 ms (14×) |
| Vulkan  | dots (35.3s) |  2.37 s (15×) |  2.37 s (15×) |
| CPU     | jfk (11.0s)  |  1.37 s (8×)  |  1.09 s (10×) |
| CPU     | dots (35.3s) |  4.76 s (7×)  |  4.17 s (8×)  |

Fedora 43, transcribe.cpp `ef35659`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models nemotron-3.5-asr-streaming-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name nemotron-3.5-asr-streaming-0.6b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav` via `scripts/validate.py`. Per-tensor tolerances live
in a per-variant file
([`tests/tolerances/nemotron-3.5-asr-streaming-0.6b.json`](../../tests/tolerances/nemotron-3.5-asr-streaming-0.6b.json))
rather than the family-shared one because the unnormalised log-mel
(NeMo's `normalize="NA"` no-op) lands on a different magnitude scale than
the per-feature-normalised siblings.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/nemotron-3.5-asr-streaming-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant nemotron-3.5-asr-streaming-0.6b` |

Validation uses the F32 reference GGUF; the shipped quants are accepted
on WER (Stage 7), not tensor tolerances.

## Capabilities

- **Languages:** 32 supported language-locales (e.g. `en-US`, `en-GB`, `es-ES`,
  `fr-FR`, `de-DE`, `it-IT`, `pt-BR`, `nl-NL`, `ru-RU`, `zh-CN`, `ja-JP`,
  `ko-KR`, `hi-IN`, `ar-AR`, …), selected via `--language <locale>`. (The
  tokenizer recognizes 40; the 8 adaptation-ready locales need fine-tuning.)
- **Language detection:** `auto` mode emits `<ll-RR>` locale tags (e.g.
  `<en-US>`, `<zh-CN>`) in the raw token stream. A tag can appear anywhere
  in the sequence, not only at the end. They are stripped from the returned
  text by default (offline and streaming); pass `keep_special_tags` / CLI
  `--raw-tokens` to keep them.
- **Punctuation & capitalization:** native (PnC).
- **Timestamps:** token- and word-level.
- **Streaming:** cache-aware chunked streaming, selectable via
  `--stream-chunk-ms 1120 --stream-att-right {0,3,6,13}` (the four trained
  latency settings). R=13 is byte-equal to the offline transcript;
  lower-R settings commit earlier for lower latency.
- **Translation / diarization / VAD:** not supported.

## Known limitations

- The auxiliary CTC head present in the upstream checkpoint is dropped at
  conversion (the RNN-T head is the inference path); CTC-argmax timestamps
  are not available.
- WER is gated on English only (FLEURS test en + LibriSpeech test-clean
  against the NeMo Oracle). The other 39 locales are exercised
  functionally but not WER-scored here. Published latency numbers cover
  the offline `[56, 13]` path; the sub-1.12 s streaming settings are
  functionally validated (byte-equal at R=13) but not separately
  benchmarked.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/nemotron-3.5-asr-streaming-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-F32.gguf \
  models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant nemotron-3.5-asr-streaming-0.6b
```
