# Granite Speech 5.0 470M TurboCTC

IBM's [`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
ported to transcribe.cpp. A 470M-parameter Granite Conformer encoder with a
self-conditioned CTC head and no autoregressive decoder.

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and produces
a transcript. It is not a streaming model, does not translate, and is
English-only — the 16384-entry vocabulary carries no language tokens, so
`--language` reaches nothing.

Three properties make it unusual among the ported encoders:

- **Decoding is a single argmax pass.** There is no beam search and no
  autoregressive loop. Across the benchmarks below, encoder work dominates
  wall time.
- **Cost is linear in audio length.** Attention is block-local (128-frame
  blocks with Shaw relative positions) rather than global, so there is no
  quadratic term. Measured on an M4, realtime factor is flat from 11 seconds
  out to 5 minutes of audio; see Performance below for the numbers.
- **The benchmark transcripts are lowercase and unpunctuated.** Both `jfk` and
  `dots` have no restored casing or punctuation; `dots` does render the number
  in `10 years` as digits.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`18ca3c1`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/commit/18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0),
pinned 2026-09-12.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-speech-5.0-470m-turboctc-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-BF16.gguf)     | 948 MB | 1.33% |
| F16    | [granite-speech-5.0-470m-turboctc-F16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-F16.gguf)       | 948 MB | 1.33% |
| Q8_0   | [granite-speech-5.0-470m-turboctc-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q8_0.gguf)     | 506 MB | 1.34% |
| Q6_K   | [granite-speech-5.0-470m-turboctc-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q6_K.gguf)     | 392 MB | 1.33% |
| Q5_K_M | [granite-speech-5.0-470m-turboctc-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q5_K_M.gguf) | 336 MB | 1.34% |
| Q4_K_M | [granite-speech-5.0-470m-turboctc-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-Q4_K_M.gguf) | 279 MB | 1.34% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances) with
greedy CTC decoding and no external LM. Measured reference baseline
(transformers, F32, CPU): **1.33%**. IBM publishes no LibriSpeech test-clean
figure for this model — the model card ships bar charts only, and the single
public number is 5.00% aggregate WER across the 8 Open ASR leaderboard test
sets — so the baseline above is our own reference run rather than an upstream
score.

On this split the whole quant matrix spans 7 word errors out of roughly 53,000
words, and every tier falls inside the reference 95% CI of [1.20%, 1.47%], so
the ordering between tiers is not statistically meaningful. Quantization down
to Q4_K_M costs very little here; that result is from clean read speech and has
not been checked on noisy or accented audio.

## Quick Start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-Q8_0.gguf \
  samples/jfk.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above
(2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  | 37.5 ms (293×) | 38.9 ms (283×) |
| Metal   | dots (35.3s) | 85.6 ms (413×) | 87.9 ms (402×) |
| CPU     | jfk (11.0s)  |  232 ms (48×)  |  233 ms (47×)  |
| CPU     | dots (35.3s) |  703 ms (50×)  |  689 ms (51×)  |

macOS 26.6.2, transcribe.cpp `54b241e`.

### Apple M4

| Backend | Sample            |          Q8_0 |        Q4_K_M |
| ------- | ----------------- | ------------: | ------------: |
| Metal   | jfk (11.0s)       |  100 ms (110×) |  103 ms (107×) |
| Metal   | dots (35.3s)      |  272 ms (130×) |  279 ms (127×) |
| Metal   | dots-full (305.9s)|  2.33 s (131×) |  2.38 s (128×) |
| CPU     | jfk (11.0s)       |  388 ms (28×)  |  423 ms (26×)  |
| CPU     | dots (35.3s)      |  1.19 s (30×)  |  1.27 s (28×)  |
| CPU     | dots-full (305.9s)|  8.30 s (37×)  |  9.73 s (31×)  |

macOS 26.5.1, transcribe.cpp `f2d5e31`.

Two things to read off this table:

- **Realtime factor holds flat as audio grows.** Metal goes 130× at 35 s to
  131× at 306 s, and CPU actually improves, 30× to 37×. That is the block-local
  attention: cost is `O(T × context_size)`, so there is no quadratic term to
  erode throughput on long files.
- **Q8_0 is faster than Q4_K_M** on both backends — 3% on Metal, 7-9% on CPU —
  despite being 1.8× the file size. K-quant unpacking costs more than the memory
  bandwidth it saves at this model size. Q4_K_M earns its place on footprint,
  not speed.

BF16 is the slowest option on CPU by a wide margin: the BF16 matmul kernel does
not thread (encode is 8.9 s at 1 thread and 9.4 s at 8 for 11 s of audio, where
the same graph on F32 scales 6.9 s to 2.4 s). Metal is unaffected. On CPU,
prefer any quantized tier.

Batching is implemented and byte-identical to single-stream output, but buys
little on GPU: per-utterance latency on Metal is 97.3 ms at batch 1, a best of
94.2 ms at batch 2-4, and 97.7 ms at batch 32. A single 11 s clip already
saturates the device.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models granite-speech-5.0-470m-turboctc \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name granite-speech-5.0-470m-turboctc-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against Hugging Face transformers
on `samples/jfk.wav` and `samples/dots.wav`. All 35 checkpointed tensors fall
within family tolerance, and the final transcript matches the reference
verbatim. Last validated at commit
[`b9427cf`](https://github.com/handy-computer/transcribe.cpp/tree/b9427cf).

| Field | Value |
| --- | --- |
| Reference | transformers 5.17.0, `ibm-granite/granite-speech-5.0-470m-turboctc` |
| Dump script | `scripts/dump_reference_granite5_ctc_transformers.py` |
| Manifest | `tests/golden/granite5_ctc/granite-speech-5.0-470m-turboctc.manifest.json` |
| Command | `uv run scripts/validate.py all --family granite5_ctc` |

Selected tensors (the `dots` case, 441 output frames):

| Tensor | Max abs diff | Mean abs diff |
| --- | ---: | ---: |
| `mel.in`                | `8.821e-06` | `1.007e-07` |
| `enc.input_linear.out`  | `1.372e-02` | `1.097e-03` |
| `enc.block.7.out`       | `1.316e-02` | `1.011e-03` |
| `enc.ctc.mid_logits`    | `6.251e-02` | `3.951e-03` |
| `enc.block.15.out`      | `2.190e-02` | `1.370e-03` |
| `enc.ctc_logits`        | `1.513e-01` | `5.442e-03` |

The drift comes from BF16 matmul accumulation compounding over 16 blocks, and
stays small relative to tensor magnitude — on `enc.ctc_logits` the mean drift
is 0.06% of RMS. It does not reach the output: there are zero argmax
differences against the reference on either sample, and 2619 of 2620
hypotheses on LibriSpeech test-clean are byte-identical to the reference run.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite5_ctc \
  scripts/convert-granite5_ctc.py ibm-granite/granite-speech-5.0-470m-turboctc \
  --repo-id ibm-granite/granite-speech-5.0-470m-turboctc
```

### Quantize

```bash
uv run scripts/quantize-all.py \
  models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family granite5_ctc \
  --variant granite-speech-5.0-470m-turboctc
```

### Score WER

```bash
uv run scripts/wer/run.py \
  --model models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/granite-speech-5.0-470m-turboctc-BF16.librispeech-test-clean.jsonl

uv run scripts/wer/score.py \
  reports/wer/granite-speech-5.0-470m-turboctc-BF16.librispeech-test-clean.jsonl
```
