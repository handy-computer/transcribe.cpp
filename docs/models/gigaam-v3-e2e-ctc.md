# GigaAM-v3 e2e-CTC

ai-sage's [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
(e2e_ctc branch) ported to transcribe.cpp. Same 16-layer Conformer encoder as `gigaam-v3-e2e-rnnt`, paired with a 1×1 Conv1d CTC head. 256-piece SentencePiece vocabulary keeps the head compact while preserving punctuation and Cyrillic casing in output. Faster than RNN-T at comparable accuracy on short utterances.

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

Licensed MIT. Ported from upstream commit
[`cec030b`](https://huggingface.co/ai-sage/GigaAM-v3/commit/cec030b4c4f35d928e4a9044a3bdb29ebd499fac),
pinned 2026-05-12.

## Download

| Quantization | Download | Size | WER (FLEURS ru) |
| --- | --- | ---: | ---: |
| F32    | [gigaam-v3-e2e-ctc-F32.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-F32.gguf) |  843 MB |   5.50% |
| F16    | [gigaam-v3-e2e-ctc-F16.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-F16.gguf) |  428 MB |   5.50% |
| Q8_0   | [gigaam-v3-e2e-ctc-Q8_0.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf) |  260 MB |   5.50% |
| Q6_K   | [gigaam-v3-e2e-ctc-Q6_K.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q6_K.gguf) |  216 MB |   5.56% |
| Q5_K_M | [gigaam-v3-e2e-ctc-Q5_K_M.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q5_K_M.gguf) |  195 MB |   5.58% |
| Q4_K_M | [gigaam-v3-e2e-ctc-Q4_K_M.gguf](https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q4_K_M.gguf) |  174 MB |   5.57% |

WER is measured on the full FLEURS ru test split (775 utterances) with
greedy decoding and no external LM. F32 reference baseline: **5.50%**.

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample     |          Q8_0 |        Q4_K_M |
| ------- | ---------- | ------------: | ------------: |
| Metal   | ru (4.5s)  |  40 ms (112×) |  40 ms (111×) |
| CPU     | ru (4.5s)  | 164 ms (27×)  | 161 ms (28×)  |

macOS 26.4.1, transcribe.cpp `ef55b52`.

### AMD Ryzen 7 PRO 4750U

| Backend | Sample     |         Q8_0 |       Q4_K_M |
| ------- | ---------- | -----------: | -----------: |
| Vulkan  | ru (4.5s)  | 152 ms (30×) | 155 ms (29×) |
| CPU     | ru (4.5s)  | 494 ms (9×)  | 397 ms (11×) |

Fedora Linux 43, transcribe.cpp `ef55b52`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

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
