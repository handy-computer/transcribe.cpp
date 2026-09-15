# Nemotron 3.5 ASR Streaming 0.6B

<!-- catalog:intro -->
Upstream: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) at [`24b151a`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/24b151a).

Multilingual speech-to-text across 32 supported language-locales (the model's tokenizer recognizes 40, but 8 are adaptation-ready and need fine-tuning) with punctuation and capitalization. A cache-aware streaming FastConformer encoder with a prompt-conditioned RNN-T transducer decoder; the target language is selected per call (--language en-US, fr-FR, de-DE, ...) and an auto mode emits a <lang-XX> tag. Ships both the offline path (att_context_size=[56, 13], 1.12s, headline accuracy) and runtime-selectable chunked streaming (--stream-chunk-ms 1120 --stream-att-right {0,3,6,13}).
<!-- /catalog -->

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

<!-- catalog:pin -->
Licensed [OpenMDW-1.1](https://openmdw.ai/license/1-1/). Ported from upstream commit [`24b151a`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/commit/24b151a), pinned 2026-06-08. Validated against the NeMo reference at transcribe.cpp commit [`909e94e`](https://github.com/handy-computer/transcribe.cpp/tree/909e94e) on 2026-06-08.
<!-- /catalog -->

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

<!-- catalog:downloads metric=false -->
| Quantization | Download |    Size |
| --- | --- | ---: |
| F32          | [nemotron-3.5-asr-streaming-0.6b-F32.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-F32.gguf) | 2.55 GB |
| F16          | [nemotron-3.5-asr-streaming-0.6b-F16.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-F16.gguf) | 1.28 GB |
| Q8_0         | [nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf) |  751 MB |
| Q6_K         | [nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q6_K.gguf) |  621 MB |
| Q5_K_M       | [nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q5_K_M.gguf) |  560 MB |
| Q4_K_M       | [nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf) |  496 MB |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full FLEURS en split (647 utterances), batch sizes 1 and 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy RNN-T decoding with whisper-normalizer scoring; the per-quant column is
FLEURS en. NeMo reference baseline on the same manifest: 7.99% (NVIDIA self-reports
7.91% en-US). On LibriSpeech test-clean (2620 utterances) the same presets score F32
3.04 / F16 3.03 / Q8_0 3.06 / Q6_K 3.07 / Q5_K_M 3.10 / Q4_K_M 3.28, against a 3.03%
NeMo reference.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| ar       | WER    | 15.93% |
| bg       | WER    | 22.02% |
| cs       | WER    | 23.00% |
| da       | WER    | 28.51% |
| de       | WER    | 10.33% |
| es       | WER    |  6.30% |
| et       | WER    | 31.84% |
| fi       | WER    | 21.91% |
| fr       | WER    | 10.78% |
| hi       | WER    |  8.61% |
| hr       | WER    | 26.21% |
| hu       | WER    | 32.12% |
| it       | WER    |  5.78% |
| ja       | CER    | 13.52% |
| ko       | CER    |  8.89% |
| nb       | WER    | 19.24% |
| nl       | WER    | 13.61% |
| pl       | WER    | 17.54% |
| pt       | WER    |  8.52% |
| ro       | WER    | 28.28% |
| ru       | WER    | 12.61% |
| sk       | WER    | 23.25% |
| sv       | WER    | 24.32% |
| tr       | WER    | 15.40% |
| uk       | WER    | 14.88% |
| vi       | WER    | 13.96% |
| zh       | CER    | 18.87% |

**LibriSpeech test-clean**

| Language | Metric |   F32 |   F16 |  Q8_0 |  Q6_K | Q5_K_M | Q4_K_M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en       | WER    | 3.04% | 3.04% | 3.05% | 3.08% |  3.10% |  3.30% |
<!-- /catalog -->

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

The decoder runs through a reused ggml graph for the joint output
projection (the 13k-vocab RNN-T joint that dominates this variant's
decode) and a thread-parallel predictor; both are the default, so these
are out-of-the-box numbers with no tuning.

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  76 ms (143.94×) |  77 ms (143.14×) |
| Metal   | dots (35.3s) | 256 ms (138.26×) | 256 ms (137.88×) |
| CPU     | jfk (11.0s)  |  358 ms (30.76×) |  355 ms (31.01×) |
| CPU     | dots (35.3s) |  1.19 s (29.73×) |  1.21 s (29.23×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 640 ms (17.18×) | 644 ms (17.09×) |
| Vulkan  | dots (35.3s) | 2.07 s (17.09×) | 2.09 s (16.88×) |
| CPU     | jfk (11.0s)  | 951 ms (11.56×) | 993 ms (11.07×) |
| CPU     | dots (35.3s) |  3.67 s (9.62×) |  3.74 s (9.45×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models nemotron-3.5-asr-streaming-0.6b
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
- The measured-Oracle release gate uses English (FLEURS test en +
  LibriSpeech test-clean). The publication catalog additionally carries a
  Q8_0 FLEURS result for every supported language. Published latency numbers cover
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
