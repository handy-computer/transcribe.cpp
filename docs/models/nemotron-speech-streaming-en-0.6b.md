# Nemotron Speech Streaming EN 0.6B

NVIDIA's [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
ported to transcribe.cpp. A 0.6B-parameter cache-aware streaming
FastConformer encoder with an RNN-T transducer decoder.

## What it's for

English speech-to-text with greedy RNN-T decoding. Outputs cased,
punctuated transcripts. Token- and word-level timestamps are available.

This port runs the model in both **offline** and **cache-aware streaming**
modes:

- Offline (`transcribe_run`): the full audio is consumed in one pass.
  The encoder preserves the upstream `att_context_size=[70, 13]` (1.12s)
  cache-aware attention mask end-to-end.
- Streaming (`transcribe_stream_{begin,feed,finalize}` /
  `transcribe-cli --stream-chunk-ms N`): incremental PCM feeds drive
  per-chunk encoder forward passes with `cache_last_channel` /
  `cache_last_time` carried across calls, and an LSTM-state-carrying
  RNN-T greedy decoder. All four latency settings from the upstream
  multi-lookahead training menu are selectable via
  `--stream-att-right R ∈ {0, 1, 6, 13}` (lookahead 0 / 80 / 480 /
  1040 ms respectively). Default = `13` (max accuracy).

See NVIDIA's [model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
for training data, intended use, streaming methodology, and the full
latency-vs-accuracy table.

Licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Ported from upstream commit
[`ef3bf40`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/commit/ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4),
pinned 2026-05-11.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean, offline) |
| --- | --- | ---: | ---: |
| F32    | [nemotron-speech-streaming-en-0.6b-F32.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-F32.gguf) | 2.30 GB | 2.31% |
| F16    | [nemotron-speech-streaming-en-0.6b-F16.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-F16.gguf) | 1.16 GB | 2.31% |
| Q8_0   | [nemotron-speech-streaming-en-0.6b-Q8_0.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q8_0.gguf) |  696 MB | 2.31% |
| Q6_K   | [nemotron-speech-streaming-en-0.6b-Q6_K.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q6_K.gguf) |  573 MB | 2.29% |
| Q5_K_M | [nemotron-speech-streaming-en-0.6b-Q5_K_M.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q5_K_M.gguf) |  514 MB | 2.34% |
| Q4_K_M | [nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf](https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf) |  453 MB | 2.38% |

WER is measured on the full LibriSpeech test-clean split (2620
utterances) with greedy RNN-T decoding. F32 reference baseline: 2.31%.
NVIDIA's self-reported number on the same split at
`att_context_size=[70, 13]` (1.12s chunk, w/o PnC) is 2.32% (from the
[HF model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)).

## Streaming WER

In cache-aware streaming mode the encoder runs incrementally over fixed
chunks while carrying constant-memory caches. WER on the full LibriSpeech
test-clean split (2620 utterances) at the default `att_context_right=13`
setting.

| Quantization | WER (streaming, R=13) |
| --- | ---: |
| F16  | 2.29% |
| Q8_0 | 2.31% |

Streaming matches offline (2.31%) and NVIDIA's NeMo cache-aware streaming
reference on the same split (2.31%)

Reproduction:

```bash
uv run scripts/wer/run.py \
  --cli build/bin/transcribe-cli \
  --model models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --manifest <librispeech test-clean manifest> --out hyps.jsonl \
  --stream-chunk-ms 1040 --stream-att-right 13
uv run scripts/wer/score.py hyps.jsonl
```

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  73 ms (151×) |  73 ms (151×) |
| Metal   | dots (35.3s) | 224 ms (158×) | 221 ms (160×) |
| CPU     | jfk (11.0s)  |  329 ms (33×) |  330 ms (33×) |
| CPU     | dots (35.3s) |  1.12 s (31×) |  1.12 s (31×) |

macOS 26.4.1, transcribe.cpp `12f1076`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  812 ms (14×) |  813 ms (14×) |
| Vulkan  | dots (35.3s) |  2.93 s (12×) |  2.98 s (12×) |
| CPU     | jfk (11.0s)  |  1.39 s (8×)  |  1.22 s (9×)  |
| CPU     | dots (35.3s) |  5.21 s (7×)  |  4.76 s (7×)  |

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models nemotron-speech-streaming-en-0.6b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name nemotron-speech-streaming-en-0.6b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav` via `scripts/validate.py`. Per-tensor tolerances live
in a per-variant file
([`tests/tolerances/nemotron-speech-streaming-en-0.6b.json`](../../tests/tolerances/nemotron-speech-streaming-en-0.6b.json))
rather than the family-shared one because the unnormalised log-mel
(NeMo's `normalize="NA"` no-op) lands on a different magnitude scale
than the per-feature-normalised siblings. The family-level forward map
at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources, with a Variant Notes row
covering this model's chunked-attention mask, causal `CausalConv2D`
pre-encode, and LayerNorm conv module.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/nemotron-speech-streaming-en-0.6b` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/nemotron-speech-streaming-en-0.6b.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant nemotron-speech-streaming-en-0.6b` |

## Streaming parity

Validated against NeMo's `transcribe_simulate_cache_aware_streaming`
reference via `scripts/validate_streaming.py`:

| `--stream-att-right` | Lookahead | First-chunk delay | jfk.wav vs NeMo | whole-earth.wav (84s) vs NeMo |
| --- | --- | --- | --- | --- |
| 13 (default) | 1040 ms | 1.12 s | byte-equal | byte-equal |
| 6            |  480 ms | 0.56 s | byte-equal | (not measured) |
| 1            |   80 ms | 0.16 s | 1 char (extra trailing period) | (not measured) |
| 0            |    0 ms | 0.08 s | 1 char (extra trailing period) | (not measured) |

The R∈{1, 0} difference is because our `stream_finalize` emits a final
partial-chunk that produces the closing punctuation; NeMo's last call
at those settings has so few mel frames the closing token doesn't
score above blank. Both outputs are valid.

### Transcript WER (LibriSpeech test-clean 512)

End-to-end WER measured against the NeMo reference on the same
512-utterance test-clean subset the offline parakeet family uses for
its WER gate. The cpp side runs on Metal (F32 and Q8_0) — the common
macOS deployment path. Gate: `|cpp_WER − ref_WER| ≤ 0.5%` absolute per
R, the same threshold every other shipped parakeet variant clears.

| Variant | R | Lookahead | WER% | Δ vs ref | 95% CI | lat_p50 (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | --- | ---: | :---: |
| REF      | 13 | 1040 ms | 1.67 | —     | [1.38, 2.02] |  718 | — |
| cpp-F32  | 13 | 1040 ms | 1.68 | +0.01 | [1.38, 2.03] |  231 | PASS |
| cpp-Q8_0 | 13 | 1040 ms | 1.66 | −0.01 | [1.37, 2.01] |  206 | PASS |
| REF      |  6 |  480 ms | 1.68 | —     | [1.36, 2.07] | 1477 | — |
| cpp-F32  |  6 |  480 ms | 1.70 | +0.02 | [1.37, 2.09] |  436 | PASS |
| cpp-Q8_0 |  6 |  480 ms | 1.68 | +0.00 | [1.37, 2.05] |  406 | PASS |
| REF      |  1 |   80 ms | 1.90 | —     | [1.57, 2.29] | 4507 | — |
| cpp-F32  |  1 |   80 ms | 1.83 | −0.07 | [1.51, 2.21] | 1437 | PASS |
| cpp-Q8_0 |  1 |   80 ms | 1.83 | −0.07 | [1.50, 2.22] | 1428 | PASS |
| REF      |  0 |    0 ms | 1.96 | —     | [1.62, 2.35] | 5107 | — |
| cpp-F32  |  0 |    0 ms | 1.85 | −0.11 | [1.54, 2.23] | 2830 | PASS |
| cpp-Q8_0 |  0 |    0 ms | 1.85 | −0.11 | [1.53, 2.24] | 2809 | PASS |

All 8 cpp/ref pairs PASS with max |Δ| = 0.11% — comfortably inside the
95% CI on 512 utterances (±0.5% at typical 1–3% WER). Q8_0 is
statistically indistinguishable from F32 at every R (Δ between Q8 and
F32 ≤ 0.02%): the streaming path does not introduce additional quant
degradation beyond what the offline parakeet WER gate already covers.

Reference is NeMo's `conformer_stream_step` per chunk on CPU/fp32
(`scripts/wer/run_reference_parakeet_streaming_nemo.py`); cpp side is
`build/bin/transcribe-cli --stream-chunk-ms 500 --stream-att-right R`
on Metal. Test plan: [`docs/plans/nemotron-streaming-wer-plan.md`](../plans/nemotron-streaming-wer-plan.md).
Reproduce / refresh the table from score sidecars:

```bash
uv run scripts/wer/streaming_table.py \
    --reports-dir reports/wer --gate 0.5 --markdown
```

### Per-tensor parity (numerical validation)

`tests/tolerances/nemotron-speech-streaming-en-0.6b.streaming.json`
pins per-kind tolerances calibrated to CPU 1-thread observed drift via
the `/porting-2-oracle` Stage 4 recipe: `max_abs = max(1.5 × observed,
max(1e-4 × p99_abs, 1e-6))`. With these tolerances and
`--backend cpu --threads 1`, the harness reports
**3554/3554 streaming-tensor pairs in tolerance across R ∈ {0, 1, 6, 13}**
on jfk.wav.

Drift profile (observed, CPU 1-thread, R=13):

| Tensor kind | p99_abs | rms | observed max_abs | observed rel | budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mel_in` | 16.6 | 11.7 | 0.81 | 4.9% | 1.21 |
| `cache_lc_in/out` | 0.18 | 0.066 | 7.1e-3 | 3.9% | 0.011 |
| `cache_lt_in/out` | 18 | 3.81 | 0.22 | 1.2% | 0.33 |
| `enc_out` | 0.24 | 0.090 | 6.5e-3 | 2.7% | 9.7e-3 |
| `channel_len` | 70 | 70 | 0 | 0% | 7e-3 |

Apples-to-apples vs the **offline** path on the same model:

| Extraction point | Offline drift (max_abs / rel) | Streaming chunk-0 drift |
| --- | --- | --- |
| post-block-0 (`enc.block.0.out`) | 0.41 / 1.40% | n/a (no equiv. dump) |
| post-block-12 (`enc.block.12.out`) | 0.42 / 1.23% | n/a |
| post-block-23 (`enc.block.23.out`) | 7.2e-3 / 3.32% | n/a |
| post-encoder (`dec.enc_out`) | 7.2e-3 / 3.32% | `enc_out` mid-stream max 1.9e-3 / 0.79% |

Streaming drift is **tighter than offline** at the post-encoder
extraction point (chunked_limited attention sums over fewer keys per
query than full attention, so per-matmul reduction-order noise
accumulates less). Stage 2 provisional tolerances
(`max(1e-4 × p99_abs, 1e-6)`) — intentionally aggressive to catch
algorithm bugs, not be passable for any nontrivial encoder — are
exceeded on 107/150 streaming tensors at CPU 1-thread, vs 8/14 on the
offline path: the same family-wide float-noise regime. The
[regular parakeet family tolerance file](../../tests/tolerances/parakeet.json)
allows 50× max_abs at `enc.block.0.out` and 3.7 at `enc.final` — our
streaming sits inside that envelope by orders of magnitude.

Metal contributes an additional ~18 tensor failures at Stage 2
provisional thresholds (125 vs 107 over budget on R=13), but
transcripts stay byte-equal end-to-end. Metal-specific tolerances
would be a separate widening.

Reproduce (matches the calibration regime of
`tests/tolerances/nemotron-speech-streaming-en-0.6b.streaming.json` —
CPU 1-thread, F32, committed per-tensor tolerances — and exits non-zero
on any tensor or transcript regression across the full R menu):

```bash
uv run --project scripts/envs/parakeet scripts/validate_streaming.py \
    --hf-model nvidia/nemotron-speech-streaming-en-0.6b \
    --gguf models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-F32.gguf \
    --audio samples/jfk.wav \
    --out build/validate_streaming/nemotron/jfk \
    --right 13 6 1 0 \
    --backend cpu --threads 1 \
    --tolerances tests/tolerances/nemotron-speech-streaming-en-0.6b.streaming.json
```

Omitting `--backend cpu --threads 1 --tolerances …` falls back to Metal
and loose 1e-3 defaults, which surfaces the ~18 Metal-specific failures
noted above and does not exercise the committed gate.

### Note on mel right-edge timing

To match NeMo's mel-on-full-audio reference behavior, the C++ side
delays each chunk emission by a small right-edge margin
(`ceil((n_fft/2) / hop_length)` mel frames — 2 frames = 20 ms on the
nemotron preprocessor's `n_fft=512` / `hop=160`). Without this, our
per-feed mel recomputation produces reflect-padded values for the last
few mel frames of each chunk, which never matches the reference's
full-audio mel. The margin is "true" lookahead in the cache-aware
streaming sense and is independent of the `att_context_right`-driven
encoder lookahead.

## Known limitations

- Multilingual transcription is not supported. The model is
  English-only by training.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/nemotron-speech-streaming-en-0.6b
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-F32.gguf \
  models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant nemotron-speech-streaming-en-0.6b
```
