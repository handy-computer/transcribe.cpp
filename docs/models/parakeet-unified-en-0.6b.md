# Parakeet Unified EN 0.6B

<!-- catalog:intro -->
Upstream: [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) at [`d4ac992`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b/commit/d4ac992).

English speech-to-text with punctuation and capitalization. A FastConformer encoder with an RNN-T transducer decoder, trained as a 'unified' streaming/offline model. This port runs the model in both offline and buffered streaming modes.
<!-- /catalog -->

## What it's for

English speech-to-text with greedy RNN-T decoding. Outputs cased,
punctuated transcripts. Token- and word-level timestamps are available.

This port runs the model in both **offline** and **buffered streaming**
modes from the same GGUF weights:

- Offline (`transcribe_run`): the full audio is consumed in one pass
  with full attention.
- Streaming (`transcribe_stream_{begin,feed,finalize}` /
  `transcribe-cli --stream-chunk-ms N --stream-buf-{left,chunk,right}-ms N`):
  per-chunk encoder forward passes over a sliding `[left | chunk |
  right]` PCM window with the `chunked_limited_with_rc` 3-tuple
  attention mask, plus an RNN-T-state-carrying greedy decoder across
  chunks. The runtime `(L, C, R)` tuple is selected at `stream_begin`
  from the model's training menu (`L ∈ {70}`, `C ∈ {1, 2, 7, 13}`,
  `R ∈ {0, 1, 2, 3, 4, 7, 13}` encoder frames at the 80ms
  post-subsample rate). All six published `(L, C, R)` configurations
  are supported; see the [Streaming](#streaming) section below for
  the WER and per-chunk latency of each. Default is `(70, 13, 13)` =
  `5.6s / 1.04s / 1.04s` (2.08s lookahead) — the highest-accuracy
  tuple and the row NVIDIA reports the model card's streaming WER for.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
for training data, intended use, streaming methodology, and upstream
evaluation results.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`d4ac992`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b/commit/d4ac992), pinned 2026-05-10. Validated against the NeMo reference at transcribe.cpp commit [`42528dd`](https://github.com/handy-computer/transcribe.cpp/tree/42528dd) on 2026-05-10.
<!-- /catalog -->

## Download

<!-- catalog:downloads label="LibriSpeech test-clean, offline" -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean, offline) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-unified-en-0.6b-F32.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-F32.gguf) | 2.47 GB | 1.59% |
| F16          | [parakeet-unified-en-0.6b-F16.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-F16.gguf) | 1.24 GB | 1.59% |
| Q8_0         | [parakeet-unified-en-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q8_0.gguf) |  731 MB | 1.60% |
| Q6_K         | [parakeet-unified-en-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q6_K.gguf) |  602 MB | 1.61% |
| Q5_K_M       | [parakeet-unified-en-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q5_K_M.gguf) |  541 MB | 1.58% |
| Q4_K_M       | [parakeet-unified-en-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q4_K_M.gguf) |  477 MB | 1.62% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy RNN-T decoding, no external LM. F32 reference baseline: 1.59%. NVIDIA's
self-reported number on the same split is 1.63%.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 3.99% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  59 ms (187.72×) |  60 ms (183.34×) |
| Metal   | dots (35.3s) | 155 ms (228.39×) | 161 ms (219.30×) |
| CPU     | jfk (11.0s)  |  285 ms (38.55×) |  307 ms (35.86×) |
| CPU     | dots (35.3s) |  979 ms (36.09×) |  1.05 s (33.77×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 4750U Pro

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 448 ms (24.55×) | 455 ms (24.15×) |
| Vulkan  | dots (35.3s) | 1.36 s (26.01×) | 1.37 s (25.74×) |
| CPU     | jfk (11.0s)  | 712 ms (15.45×) | 788 ms (13.96×) |
| CPU     | dots (35.3s) | 2.84 s (12.46×) | 2.95 s (11.97×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-unified-en-0.6b
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on `samples/jfk.wav`
via `scripts/validate.py`, sharing the parakeet family tolerance file. The
family-level forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources (fp64 STFT, mel amplification,
attenuation through the encoder).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/parakeet-unified-en-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/parakeet-unified-en-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-unified-en-0.6b` |

## Streaming

Buffered streaming is supported for all six published `(L, C, R)`
configurations from the model's training menu. The cpp driver
reproduces NeMo's reference inference loop
(`speech_to_text_streaming_infer_rnnt.py`) per-chunk: identical
chunk geometry, bit-exact `audio_in` windows, and the same
variable-stride algorithm — step 0 consumes `chunk + right` audio,
steady-state consumes `chunk`, and the final step folds the trailing
right slot plus the ragged tail into one `is_last` emit.

### End of stream

Finalize flushes retained right context even when the input ends exactly
on a chunk boundary. It also appends `R` frames of silence so final speech
frames keep their trained lookahead and delayed RNN-T tokens are emitted.
Synthetic samples are not included in stream audio accounting, and token
timestamps are clamped to the real input duration. This deliberately
differs from NeMo on the final `R` real frames only. Decoding the masked
conv-overhang frame instead was rejected because it hallucinated trailing
words on naturally ending clips.

### Supported configurations

The runtime `(L, C, R)` is set via `--stream-buf-left-ms /
--stream-buf-chunk-ms / --stream-buf-right-ms` (multiples of the
80ms post-subsample frame). Per-chunk lookahead latency is
`chunk + right` — the audio you must buffer before the chunk's
tokens can be emitted.

| `(L, C, R)` frames | `(left, chunk, right)` | Lookahead latency |       Use case |
| ------------------ | ---------------------- | ----------------: | -------------- |
| `(70, 1, 0)`       | `5.60 / 0.08 / 0.00 s` |              80ms | ultra-low latency (not recommended — accuracy collapses, see below) |
| `(70, 1, 1)`       | `5.60 / 0.08 / 0.08 s` |             160ms | low latency |
| `(70, 2, 2)`       | `5.60 / 0.16 / 0.16 s` |             320ms | low latency |
| `(70, 2, 4)`       | `5.60 / 0.16 / 0.32 s` |             480ms | low latency, asymmetric |
| `(70, 7, 7)`       | `5.60 / 0.56 / 0.56 s` |            1.12 s | balanced |
| `(70, 13, 13)`     | `5.60 / 1.04 / 1.04 s` |            2.08 s | best accuracy (default) |

### Transcript WER (LibriSpeech test-clean-512)

Scored on the 512-utterance test-clean subset for all six configs,
F32 cpp vs the same NeMo buffered-streaming reference loop. The
`Δ` column is cpp minus REF in absolute percentage points.

| `(L, C, R)` | Latency | REF WER% (CI)         | cpp F32 WER% (CI)     |     Δpp |
| ----------- | ------: | --------------------- | --------------------- | ------: |
| `(70, 1, 0)` |   80ms | 5.57% [4.86, 6.31]    | 5.76% [5.03, 6.59]    | **+0.19** |
| `(70, 1, 1)` |  160ms | 1.90% [1.61, 2.26]    | 1.90% [1.61, 2.26]    |   +0.00 |
| `(70, 2, 2)` |  320ms | 1.64% [1.33, 1.95]    | 1.64% [1.33, 1.95]    |   +0.00 |
| `(70, 2, 4)` |  480ms | 1.54% [1.26, 1.88]    | 1.55% [1.27, 1.89]    |   +0.01 |
| `(70, 7, 7)` |  1.12s | 1.42% [1.15, 1.74]    | 1.40% [1.13, 1.72]    |   -0.02 |
| `(70, 13, 13)` | 2.08s | 1.44% [1.16, 1.79]   | 1.44% [1.16, 1.78]    |   +0.00 |

The cpp column includes the end-of-stream handling described above.
Five of six configurations land within 0.02pp of the NeMo reference,
well inside the parakeet family's 0.5% gate. The zero-lookahead row is
an outlier in both implementations; use `(70, 1, 1)` for the lowest
practical latency.

### Streaming parity reproduction

```bash
uv run --project scripts/envs/parakeet \
  scripts/validate_buffered_streaming.py \
    --nemo /path/to/parakeet-unified-en-0.6b.nemo \
    --gguf models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf \
    --audio samples/jfk.wav \
    --out build/validate_buffered_streaming/parakeet-unified/jfk/default \
    --backend cpu --threads 1
```

WER reproduction for a given `(L, C, R)` config (e.g. `(70, 7, 7)`,
1.12s lookahead — pass `chunk_ms` and `right_ms` derived from `C` and
`R` at the 80ms-per-frame rate):

```bash
uv run --project scripts/envs/parakeet \
  scripts/wer/run_reference_parakeet_buffered_streaming_nemo.py \
    --model /path/to/parakeet-unified-en-0.6b.nemo \
    --manifest samples/wer/test-clean.512.manifest.jsonl \
    --left-secs 5.60 --chunk-secs 0.56 --right-secs 0.56 \
    --out reports/wer/parakeet-unified-buffered-REF-L70-C7-R7.test-clean-512.jsonl

uv run scripts/wer/run.py \
  --manifest samples/wer/test-clean.512.manifest.jsonl \
  --model models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf \
  --out reports/wer/parakeet-unified-buffered-F32-L70-C7-R7.test-clean-512.jsonl \
  --stream-chunk-ms 560 \
  --stream-buf-left-ms 5600 --stream-buf-chunk-ms 560 --stream-buf-right-ms 560 \
  --language en
```

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-unified-en-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf \
  models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-unified-en-0.6b
```
