# GigaAM-v3 e2e-CTC

<!-- catalog:intro -->
Upstream: [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3) at [`cec030b`](https://huggingface.co/ai-sage/GigaAM-v3/commit/cec030b).

Offline Russian speech-to-text with greedy CTC decoding. 16-layer Conformer encoder with a 1×1 Conv1d CTC head. Output is cased Russian with punctuation, decoded from a 256-piece SentencePiece tokenizer.
<!-- /catalog -->

## What it's for

Offline Russian speech-to-text with greedy CTC decoding. Output includes punctuation and Cyrillic casing inline from a 256-piece SentencePiece tokenizer.

Decoder is greedy; no language model, no beam search. Short-form only
(≤ 25 s per utterance; upstream `transcribe_longform` chunking via
PyAnnote VAD is intentionally not ported). Token-level timestamps are
emitted at the encoder frame rate (40 ms granularity); word- and
segment-level timestamps are out of scope.

The encoder is shared across all four ported GigaAM-v3 variants but
weights are per-variant fine-tuned (the encoder hidden state differs
across heads). Variants in this family:

- [`gigaam-v3-e2e-rnnt`](./gigaam-v3-e2e-rnnt.md): RNN-T, cased+punctuated
- [`gigaam-v3-e2e-ctc`](./gigaam-v3-e2e-ctc.md): CTC, cased+punctuated
- [`gigaam-v3-rnnt`](./gigaam-v3-rnnt.md): RNN-T, lowercased no-punct
- [`gigaam-v3-ctc`](./gigaam-v3-ctc.md): CTC, lowercased no-punct

See ai-sage's [model card](https://huggingface.co/ai-sage/GigaAM-v3)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`cec030b`](https://huggingface.co/ai-sage/GigaAM-v3/commit/cec030b), pinned 2026-05-12. Validated against the gigaam author package reference at transcribe.cpp commit [`42b96d9`](https://github.com/handy-computer/transcribe.cpp/tree/42b96d9) on 2026-05-12.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (FLEURS ru) |
| --- | --- | ---: | ---: |
| F32          | [gigaam-v3-e2e-ctc-F32.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-F32.gguf) | 884 MB | 5.50% |
| F16          | [gigaam-v3-e2e-ctc-F16.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-F16.gguf) | 449 MB | 5.50% |
| Q8_0         | [gigaam-v3-e2e-ctc-Q8_0.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf) | 272 MB | 5.53% |
| Q6_K         | [gigaam-v3-e2e-ctc-Q6_K.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q6_K.gguf) | 226 MB | 5.56% |
| Q5_K_M       | [gigaam-v3-e2e-ctc-Q5_K_M.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q5_K_M.gguf) | 205 MB | 5.58% |
| Q4_K_M       | [gigaam-v3-e2e-ctc-Q4_K_M.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q4_K_M.gguf) | 182 MB | 5.57% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full FLEURS ru split (775 utterances), batch sizes 1 and 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding, no external LM. F32 reference baseline: 5.50%. Upstream `gigaam`
author package measured on the same manifest: 6.93%; the 1.4 pp gap is upstream
rejecting 5 long (>25 s) utterances with `Too long wav file, use
'transcribe_longform' method.` (counted as 100% deletion errors). On the 770-utt
subset both sides decode, transcribe.cpp matches upstream exactly. ai-sage does not
publish a FLEURS ru WER; this number is measured here.
<!-- /catalog -->

Upstream (`gigaam` author package at `6e4b027c`) measured on the same
manifest: **6.93%**. The 1.4 pp gap is the upstream package
rejecting 5 long (>25 s) FLEURS utterances with
`Too long wav file, use 'transcribe_longform' method.`, counted as
100% deletion errors against upstream. On the 770-utterance subset both
sides decode, C++ matches upstream **exactly** (`reports/wer/gigaam.fleurs-ru.summary.md`).

ai-sage does not publish a FLEURS ru WER; this number is measured here
for transparency.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/gigaam-v3-e2e-ctc/gigaam-v3-e2e-ctc-Q8_0.gguf \
  --language ru \
  samples/ru.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample           |            Q8_0 |          Q4_K_M |
| ------- | ---------------- | --------------: | --------------: |
| Metal   | ru-short (11.0s) | 27 ms (405.69×) | 27 ms (411.99×) |
| Metal   | ru-long (33.8s)  | 64 ms (525.57×) | 66 ms (512.90×) |
| CPU     | ru-short (11.0s) | 368 ms (29.86×) | 361 ms (30.39×) |
| CPU     | ru-long (33.8s)  | 1.16 s (29.17×) | 1.24 s (27.21×) |

Apple M4 Max: transcribe.cpp `94f1f45` on 2026-09-15.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample           |            Q8_0 |          Q4_K_M |
| ------- | ---------------- | --------------: | --------------: |
| Vulkan  | ru-short (11.0s) | 318 ms (34.53×) | 326 ms (33.69×) |
| Vulkan  | ru-long (33.8s)  | 934 ms (36.22×) | 954 ms (35.48×) |
| CPU     | ru-short (11.0s) | 824 ms (13.33×) | 928 ms (11.83×) |
| CPU     | ru-long (33.8s)  |  3.53 s (9.58×) |  3.74 s (9.06×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `522ccd68` on 2026-09-15.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models gigaam-v3-e2e-ctc
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the upstream
`gigaam` package on `samples/ru.wav` via `scripts/validate.py`,
sharing the gigaam family tolerance file. The family-level forward map
at [`reports/porting/gigaam/forward-map.md`](../../reports/porting/gigaam/forward-map.md)
documents the per-stage divergence sources (fp32 STFT, depthwise conv1d
reduction order, attention accumulation through 16 Conformer blocks).

| Field | Value |
| --- | --- |
| Reference | `gigaam` package @ `6e4b027c`, `gigaam.load_model('v3_e2e_ctc', fp16_encoder=False, device='cpu')` |
| Dump script | `scripts/dump_reference_gigaam_author.py` |
| Manifest | `tests/golden/gigaam/gigaam-v3-e2e-ctc.manifest.json` |
| Command | `uv run scripts/validate.py all --family gigaam --variant gigaam-v3-e2e-ctc` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/gigaam \
  scripts/convert-gigaam.py ai-sage/GigaAM-v3 \
  --repo-id gigaam-v3-e2e-ctc --variant-key v3_e2e_ctc
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/gigaam-v3-e2e-ctc/gigaam-v3-e2e-ctc-F32.gguf \
  models/gigaam-v3-e2e-ctc/gigaam-v3-e2e-ctc-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family gigaam --variant gigaam-v3-e2e-ctc
```
