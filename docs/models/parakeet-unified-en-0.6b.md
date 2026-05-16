# Parakeet Unified EN 0.6B

NVIDIA's [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
ported to transcribe.cpp. A 0.6B-parameter FastConformer encoder with an
RNN-T transducer decoder, trained as a "unified" streaming/offline model.

## What it's for

English speech-to-text with greedy RNN-T decoding. Outputs cased,
punctuated transcripts. Token- and word-level timestamps are available.

This port runs the model in both **offline** and **buffered streaming**
modes from the same GGUF weights:

- Offline (`transcribe_run`): the full audio is consumed in one pass
  with full attention.
- Streaming (`transcribe_stream_{begin,feed,finalize}` /
  `transcribe-cli --stream-chunk-ms N`): per-chunk encoder forward
  passes over a sliding `[left | chunk | right]` PCM window with the
  `chunked_limited_with_rc` 3-tuple attention mask, plus an
  RNN-T-state-carrying greedy decoder across chunks. The runtime
  `(L, C, R)` tuple is selected at `stream_begin` from the model's
  training menu (`L ∈ {70}`, `C ∈ {1, 2, 7, 13}`, `R ∈ {0, 1, 2, 3,
  4, 7, 13}` encoder frames at the 80ms post-subsample rate). Default
  is the highest-accuracy tuple `(70, 13, 13)` = `5.6s / 1.04s /
  1.04s` (2.08s latency) — the row NVIDIA publishes the model card's
  streaming WER for.

See NVIDIA's [model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
for training data, intended use, streaming methodology, and upstream
evaluation results.

Licensed CC-BY-4.0. Ported from upstream commit
[`d4ac992`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b/commit/d4ac9928),
pinned 2026-05-10.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean, offline) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-unified-en-0.6b-F32.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-F32.gguf) | 2.47 GB |                                 1.59% |
| F16    | [parakeet-unified-en-0.6b-F16.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-F16.gguf) | 1.24 GB |                                 1.59% |
| Q8_0   | [parakeet-unified-en-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q8_0.gguf) |  731 MB |                                 1.60% |
| Q6_K   | [parakeet-unified-en-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q6_K.gguf) |  602 MB |                                 1.61% |
| Q5_K_M | [parakeet-unified-en-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q5_K_M.gguf) |  541 MB |                                 1.58% |
| Q4_K_M | [parakeet-unified-en-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q4_K_M.gguf) |  477 MB |                                 1.62% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with greedy RNN-T decoding and no external LM. F32 reference baseline: 1.59%. NVIDIA's self-reported number on the same split is 1.63% (from the [HF model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)).

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places). Cells gated on `Tctl < 55°C` per backend.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  69 ms (158×) |  71 ms (155×) |
| Metal   | dots (35.3s) | 210 ms (168×) | 209 ms (169×) |
| CPU     | jfk (11.0s)  |  375 ms (29×) |  318 ms (35×) |
| CPU     | dots (35.3s) |  1.27 s (28×) |  1.09 s (32×) |

macOS 26.4.1, transcribe.cpp `12f1076`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  839 ms (13×) |  848 ms (13×) |
| Vulkan  | dots (35.3s) |  3.03 s (12×) |  3.05 s (12×) |
| CPU     | jfk (11.0s)  |  1.35 s (8×)  |  1.18 s (9×)  |
| CPU     | dots (35.3s) |  5.22 s (7×)  |  4.66 s (8×)  |

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models parakeet-unified-en-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name parakeet-unified-en-0.6b-publication
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

## Streaming parity

Buffered streaming on `samples/jfk.wav` at the default `(L=70, C=13,
R=13)` frames tuple (= 5.6 / 1.04 / 1.04 s) reproduces NeMo's
reference inference loop
(`speech_to_text_streaming_infer_rnnt.py`) per-chunk: identical
chunk geometry, bit-exact `audio_in` windows, and per-tensor
`enc_out` drift inside the parakeet family's accepted noise envelope
(`tests/tolerances/parakeet.json` allows `max_abs=3.7` at `enc.final`;
observed max_abs ≤ 1.0 on CPU 1-thread). The cpp driver uses the
same variable-stride algorithm — step 0 consumes `chunk + right`
audio, steady-state consumes `chunk`, and the final step folds the
trailing right slot plus the ragged tail into one is_last emit —
so chunk boundaries are byte-identical to the reference.

Final-transcript byte match holds on long-form audio (e.g.
`samples/dots.wav`, 35.3 s, 33 chunks). On short-form audio
(`samples/jfk.wav`, 11 s) greedy RNN-T can tip a single token at
the last chunk on fp32 encoder noise — the cpp transcript adds a
trailing `.` that the ref doesn't emit. This is a single-token edit
inside the same per-chunk geometry; corpus-level WER on
LibriSpeech test-clean is the gate of record.

### Transcript WER (LibriSpeech test-clean)

Scored against the same reference framework as the offline WER table,
at the default `(70, 13, 13)` tuple:

| Variant              | N    | WER%  |  95% CI         | Sub | Del | Ins |
| -------------------- | ---: | ----: | --------------- | --: | --: | --: |
| REF (NeMo buffered)  | 512  |  1.44 | [1.16, 1.79]    | 113 |  26 |  18 |
| cpp-F32 (variable-stride) | 512 |  1.46 | [1.17, 1.82]    | 114 |  26 |  19 |

`|Δ| = 0.02%` absolute between cpp and ref — comfortably inside the
family's 0.5% gate, and substitution / deletion / insertion counts
match to within 1 each. Variable-stride brings cpp into near-perfect
alignment with the reference algorithm; the residual 1-sub / 1-ins
gap is fp32 encoder noise tipping a small number of greedy
decisions at chunk boundaries.

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

WER reproduction:

```bash
uv run --project scripts/envs/parakeet \
  scripts/wer/run_reference_parakeet_buffered_streaming_nemo.py \
    --model /path/to/parakeet-unified-en-0.6b.nemo \
    --manifest samples/wer/test-clean.512.manifest.jsonl \
    --out reports/wer/parakeet-unified-buffered-REF-default.test-clean-512.jsonl

uv run scripts/wer/run.py \
  --manifest samples/wer/test-clean.512.manifest.jsonl \
  --model models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf \
  --out reports/wer/parakeet-unified-buffered-F32-default.test-clean-512.jsonl \
  --stream-chunk-ms 500 --language en
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
