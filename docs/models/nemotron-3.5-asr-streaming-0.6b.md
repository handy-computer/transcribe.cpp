# Nemotron 3.5 ASR Streaming 0.6B

NVIDIA's [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
ported to transcribe.cpp. A 0.6B-parameter cache-aware streaming
FastConformer encoder with an RNN-T transducer decoder — the multilingual
successor to
[`nemotron-speech-streaming-en-0.6b`](nemotron-speech-streaming-en-0.6b.md).

## What it's for

Multilingual speech-to-text across **40 language-locales** (19
transcription-ready, plus broad-coverage and adaptation tiers — see the
upstream model card for the full list) with greedy RNN-T decoding.
Outputs cased, punctuated transcripts (native PnC). Token- and word-level
timestamps are available.

The target language is selected per call (`--language en-US`,
`fr-FR`, `de-DE`, …). The model also supports `auto` language
detection, in which case it emits a `<lang-XX>` tag in the transcript.
A language **must** be provided — there is no implicit default; passing
an unsupported tag returns "unsupported language".

The encoder is cache-aware and trained with four runtime latency
settings (`att_context_size` ∈ `[56, 0]` / `[56, 3]` / `[56, 6]` /
`[56, 13]` = 0 / 240 / 480 / 1040 ms lookahead). This port currently
ships and benchmarks the **offline** path (`transcribe_run`) at the
`[56, 13]` (1.12 s) setting that yields the headline accuracy; streaming
bring-up follows the predecessor's cache-aware design.

See NVIDIA's [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
for training data, the full language list, intended use, and the
latency-vs-accuracy table.

Licensed under [OpenMDW-1.1](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).
Ported from upstream commit
[`24b151a`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/24b151a851dd15909e1fc611b11bb2da52b9fc81).

## Download

| Quantization | Download | Size |
| --- | --- | ---: |
| F32    | [nemotron-3.5-asr-streaming-0.6b-F32.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-F32.gguf) | 2.38 GB |
| F16    | [nemotron-3.5-asr-streaming-0.6b-F16.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-F16.gguf) | 1.19 GB |
| Q8_0   | [nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf) | 716 MB |
| Q6_K   | [nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf) | 593 MB |
| Q5_K_M | [nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf) | 534 MB |
| Q4_K_M | [nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf) | 473 MB |

**Accuracy:** the authoritative per-quant WER sweep (Stage 7) is pending.
For context, NVIDIA's self-reported FLEURS WER at `att_context_size=[56,13]`
(LangID mode) is **7.91%** for en-US and a **8.84%** macro-average across
the 19 transcription-ready locales (from the
[HF model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)).

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

- **Languages:** 40 language-locales (e.g. `en-US`, `en-GB`, `es-ES`,
  `fr-FR`, `de-DE`, `it-IT`, `pt-BR`, `nl-NL`, `ru-RU`, `zh-CN`, `ja-JP`,
  `ko-KR`, `hi-IN`, `ar-AR`, …), selected via `--language <locale>`.
- **Language detection:** `auto` mode emits a `<lang-XX>` tag in the
  transcript.
- **Punctuation & capitalization:** native (PnC).
- **Timestamps:** token- and word-level.
- **Streaming:** cache-aware streaming is supported by the architecture
  (four trained latency settings); detailed streaming parity for this
  variant is pending validation.
- **Translation / diarization / VAD:** not supported.

## Known limitations

- The language tag (`<lang-XX>`) is currently surfaced in the transcript
  text; tag stripping (`strip_lang_tags`) is not yet wired into the CLI.
- The auxiliary CTC head present in the upstream checkpoint is dropped at
  conversion (the RNN-T head is the inference path); CTC-argmax timestamps
  are not available.
- The Stage 7 multilingual WER sweep and streaming-parity validation are
  pending; only the offline `[56, 13]` path has been benchmarked.

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
