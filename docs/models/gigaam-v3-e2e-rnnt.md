# GigaAM-v3 e2e-RNN-T

ai-sage's [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
(main (= v3_e2e_rnnt) branch) ported to transcribe.cpp. A 16-layer Conformer encoder (768-d, 16 heads, rotary positional embeddings, conv1d ×4 subsampling) feeding an RNN-T transducer head (single LSTM-320 predictor + joint network). Vocabulary is 1024 SentencePiece pieces + blank — covers cased Cyrillic plus `.,?!` directly.

## What it's for

Offline Russian speech-to-text with greedy RNN-T decoding. Output includes punctuation and Cyrillic casing inline from a 1024-piece SentencePiece tokenizer.

Decoder is greedy; no language model, no beam search. Short-form only
(≤ 25 s per utterance; upstream `transcribe_longform` chunking via
PyAnnote VAD is intentionally not ported). Token-level timestamps are
emitted at the encoder frame rate (40 ms granularity); word- and
segment-level timestamps are out of scope.

The encoder is shared across all four ported GigaAM-v3 variants but
weights are per-variant fine-tuned (the encoder hidden state differs
across heads). Variants in this family:

- [`gigaam-v3-e2e-rnnt`](./gigaam-v3-e2e-rnnt.md) — RNN-T, cased+punctuated
- [`gigaam-v3-e2e-ctc`](./gigaam-v3-e2e-ctc.md) — CTC, cased+punctuated
- [`gigaam-v3-rnnt`](./gigaam-v3-rnnt.md) — RNN-T, lowercased no-punct
- [`gigaam-v3-ctc`](./gigaam-v3-ctc.md) — CTC, lowercased no-punct

See ai-sage's [model card](https://huggingface.co/ai-sage/GigaAM-v3)
for training data, intended use, and upstream evaluation methodology.

Licensed MIT. Ported from upstream commit
[`ec1dc1f`](https://huggingface.co/ai-sage/GigaAM-v3/commit/ec1dc1f01d0d627ab2c0d3acc1e235702300d95e),
pinned 2026-05-12.

## Download

| Quantization | Download | Size | WER (FLEURS ru) |
| --- | --- | ---: | ---: |
| F32    | [gigaam-v3-e2e-rnnt-F32.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/gigaam-v3-e2e-rnnt-F32.gguf) |  849 MB |   5.35% |
| F16    | [gigaam-v3-e2e-rnnt-F16.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/gigaam-v3-e2e-rnnt-F16.gguf) |  431 MB |   5.35% |
| Q8_0   | [gigaam-v3-e2e-rnnt-Q8_0.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/gigaam-v3-e2e-rnnt-Q8_0.gguf) |  261 MB |   5.36% |
| Q6_K   | [gigaam-v3-e2e-rnnt-Q6_K.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/gigaam-v3-e2e-rnnt-Q6_K.gguf) |  217 MB |   5.37% |
| Q5_K_M | [gigaam-v3-e2e-rnnt-Q5_K_M.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/gigaam-v3-e2e-rnnt-Q5_K_M.gguf) |  197 MB |   5.42% |
| Q4_K_M | [gigaam-v3-e2e-rnnt-Q4_K_M.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf/resolve/main/gigaam-v3-e2e-rnnt-Q4_K_M.gguf) |  175 MB |   5.36% |

WER is measured on the full FLEURS ru test split (775 utterances) with
greedy decoding and no external LM. F32 reference baseline: **5.35%**.

Upstream (`gigaam` author package at `6e4b027c`) measured on the same
manifest: **6.78%**. The 1.4 pp gap is the upstream package
rejecting 5 long (>25 s) FLEURS utterances with
`Too long wav file, use 'transcribe_longform' method.` — counted as
100% deletion errors against upstream. On the 770-utterance subset both
sides decode, C++ matches upstream **exactly** (`reports/wer/gigaam.fleurs-ru.summary.md`).

ai-sage does not publish a FLEURS ru WER; this number is measured here
for transparency.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-Q8_0.gguf \
  --language ru \
  samples/ru.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample     |         Q8_0 |       Q4_K_M |
| ------- | ---------- | -----------: | -----------: |
| Metal   | ru (4.5s)  |  51 ms (88×) |  51 ms (89×) |
| CPU     | ru (4.5s)  | 177 ms (25×) | 172 ms (26×) |

macOS 26.4.1, transcribe.cpp `ef55b52`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models gigaam-v3-ctc,gigaam-v3-rnnt,gigaam-v3-e2e-ctc,gigaam-v3-e2e-rnnt \
  --quants q8_0,q4_k_m \
  --samples ru \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name gigaam-publication
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
| Reference | `gigaam` package @ `6e4b027c`, `gigaam.load_model('v3_e2e_rnnt', fp16_encoder=False, device='cpu')` |
| Dump script | `scripts/dump_reference_gigaam_author.py` |
| Manifest | `tests/golden/gigaam/gigaam-v3-e2e-rnnt.manifest.json` |
| Command | `uv run scripts/validate.py all --family gigaam --variant gigaam-v3-e2e-rnnt` |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/gigaam \
  scripts/convert-gigaam.py ai-sage/GigaAM-v3 \
  --repo-id gigaam-v3-e2e-rnnt --variant-key v3_e2e_rnnt
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-F32.gguf \
  models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family gigaam --variant gigaam-v3-e2e-rnnt
```
