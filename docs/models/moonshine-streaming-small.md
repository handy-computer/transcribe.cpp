# Moonshine Streaming Small

Useful Sensors' [`UsefulSensors/moonshine-streaming-small`](https://huggingface.co/UsefulSensors/moonshine-streaming-small)
ported to transcribe.cpp. A 123M-parameter encoder-decoder English ASR model
designed for streaming use (ergodic encoder + sliding-window attention,
50 Hz time-domain frontend). Same family as the tiny and medium variants;
deeper encoder/decoder (10 / 10 layers vs 6 / 6 for tiny) and wider hidden
dims (encoder 620 / decoder 512).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and produces
a transcript. It does not translate, has no multilingual capability, and does
not emit timestamps.

See Useful Sensors' [model card](https://huggingface.co/UsefulSensors/moonshine-streaming-small)
for training data, intended use, and upstream evaluation methodology.

Licensed MIT. Ported from upstream commit
[`2c03650`](https://huggingface.co/UsefulSensors/moonshine-streaming-small/commit/2c036506f23a09c18df5a50057599ba6d9280999),
pinned 2026-05-06.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32  | [moonshine-streaming-small-F32.gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf/resolve/main/moonshine-streaming-small-F32.gguf)   | 536 MB | 2.53% |
| F16  | [moonshine-streaming-small-F16.gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf/resolve/main/moonshine-streaming-small-F16.gguf)   | 269 MB | 2.53% |
| Q8_0 | [moonshine-streaming-small-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-streaming-small-gguf/resolve/main/moonshine-streaming-small-Q8_0.gguf) | 189 MB | 2.54% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding (`num_beams=1`, `do_sample=False`). F32 reference
baseline: 2.53%. Useful Sensors' self-reported number on this split is
2.49% from the Open ASR Leaderboard table; the +0.04pp residual matches
the same scoring / text-normalization difference seen on the tiny variant
where we cross-checked against the HF Transformers reference (4.52% on the
same manifest, 99.6% identical hypotheses to our F32) and confirmed it is
not a numerical drift in the port.

Q6_K / Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-streaming-small/moonshine-streaming-small-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 5 iterations after 2 warmups),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above
(2 decimal places).

### Apple M4 Max

| Backend | Sample       |         Q8_0 |
| ------- | ------------ | -----------: |
| Metal   | jfk (11.0s)  |  82 ms (134×) |
| Metal   | dots (35.3s) | 612 ms (58×)  |
| CPU     | jfk (11.0s)  | 174 ms (63×)  |
| CPU     | dots (35.3s) | 699 ms (51×)  |

macOS 26.4.1, transcribe.cpp `0d312ce`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models moonshine-streaming-small \
  --quants q8_0 \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 5 --warmup 2 \
  --name moonshine-streaming-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the HF Transformers
reference (`MoonshineStreamingForConditionalGeneration`, fp32 inference,
`attn_implementation="eager"`) on `samples/jfk.wav`. All contract tensors
fall within family tolerance, and the final transcript matches the
reference. Last validated at commit
[`0d312ce`](https://github.com/handy-computer/transcribe.cpp/tree/0d312ce).

| Field | Value |
| --- | --- |
| Reference | HF Transformers v5.7.0, `UsefulSensors/moonshine-streaming-small` |
| Dump script | `scripts/dump_reference_moonshine_streaming_transformers.py` |
| Manifest | `tests/golden/moonshine_streaming/moonshine-streaming-small.manifest.json` |
| Command | `uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-small` |

Tolerances are recorded at family scope in
`tests/tolerances/moonshine_streaming.json`. The dominant drift source is
BLAS reduction-order differences between PyTorch's matmul kernels and
ggml's `mul_mat` (Accelerate / Metal / ggml-cpu). Drift accumulates
roughly linearly with depth across the 10-layer encoder, the adapter, and
the 10-layer decoder; the final logit budget stays well below 1e-3
absolute / 1e-4 mean.

Small additionally exercises the **non-square encoder attention** path —
encoder residual dim 620, attention dim 512 — that the original
moonshine-streaming-tiny port did not separate (320 = 8 × 40 there). Q/K/V
project residual_dim → attn_dim and O projects attn_dim → residual_dim;
the C++ port carries both shapes through the encoder block.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/convert-moonshine_streaming.py UsefulSensors/moonshine-streaming-small
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/moonshine-streaming-small/moonshine-streaming-small-F32.gguf \
  models/moonshine-streaming-small/moonshine-streaming-small-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-small
```

### WER sweep

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-streaming-small/moonshine-streaming-small-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-small-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-small-F32.test-clean.jsonl
```
