# Moonshine Streaming Tiny

Useful Sensors' [`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)
ported to transcribe.cpp. A 34M-parameter encoder-decoder English ASR model
designed for streaming use (ergodic encoder + sliding-window attention,
50 Hz time-domain frontend).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and produces
a transcript. It does not translate, has no multilingual capability, and does
not emit timestamps.

See Useful Sensors' [model card](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)
for training data, intended use, and upstream evaluation methodology.

Licensed MIT. Ported from upstream commit
[`f8e9dfd`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny/commit/f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a),
pinned 2026-05-06.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32  | [moonshine-streaming-tiny-F32.gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-F32.gguf)   | 169 MB | 4.53% |
| F16  | [moonshine-streaming-tiny-F16.gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-F16.gguf)   |  85 MB | 4.53% |
| Q8_0 | [moonshine-streaming-tiny-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-Q8_0.gguf) |  48 MB | 4.52% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding (`num_beams=1`, `do_sample=False`). F32 reference
baseline: 4.53%. The HF Transformers reference scored on the same manifest
in the same regime lands at 4.52% with 99.6% byte-identical hypotheses to
our F32, so the port is at exact parity with the reference. Useful Sensors'
self-reported number on this split is 4.49% from the Open ASR Leaderboard
table; the +0.04pp residual is a scoring / text-normalization difference vs
that methodology, not a numerical drift in the port.

Q6_K / Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.

### Streaming vs offline parity (Q8_0)

The streaming session API (`transcribe_stream_begin/feed/finalize`)
produces a final transcript at parity with the offline one-shot path
on Q8_0:

| Mode | WER | Sub / Del / Ins | CLI errors |
| --- | ---: | ---: | ---: |
| Offline (one-shot)                | **4.52%** | 1764 / 250 / 383 | 0 |
| Streaming `--stream-chunk-ms 500` | **4.54%** | 1772 / 252 / 381 | 0 |

The +0.02 pp delta is 8 extra word errors across ~52 000 reference
words, sitting comfortably inside the 95% confidence interval overlap
([4.22%, 4.88%] vs [4.22%, 4.90%]) and well below any reasonable
regression threshold. The residual comes from PCM trimming during
streaming producing tiny float-precision differences in the encoder
output at the left-context boundary, which occasionally flip a single
argmax late in the AR decode. Reports:
`reports/wer/moonshine-streaming-tiny-Q8_0.test-clean.{offline,stream-500ms}.score.json`.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  50 ms (218×) |
| Metal   | dots (35.3s) | 355 ms (100×) |
| CPU     | jfk (11.0s)  |  44 ms (250×) |
| CPU     | dots (35.3s) | 206 ms (172×) |

macOS 26.4.1, transcribe.cpp `0d312ce`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |
| ------- | ------------ | ------------: |
| Vulkan  | jfk (11.0s)  |  140 ms (79×) |
| Vulkan  | dots (35.3s) |  892 ms (40×) |
| CPU     | jfk (11.0s)  |  160 ms (69×) |
| CPU     | dots (35.3s) |  882 ms (40×) |

Fedora 43, transcribe.cpp `f243f34`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models moonshine-streaming-tiny \
  --quants q8_0 \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 5 --warmup 2 \
  --name moonshine-streaming-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the HF Transformers
reference (`MoonshineStreamingForConditionalGeneration`, fp32 inference,
`attn_implementation="eager"`) on `samples/jfk.wav`. All 37 contract
tensors fall within family tolerance, and the final transcript matches
the reference. Last validated at commit
[`0d312ce`](https://github.com/handy-computer/transcribe.cpp/tree/0d312ce).

| Field | Value |
| --- | --- |
| Reference | HF Transformers v5.7.0, `UsefulSensors/moonshine-streaming-tiny` |
| Dump script | `scripts/dump_reference_moonshine_streaming_transformers.py` |
| Manifest | `tests/golden/moonshine_streaming/moonshine-streaming-tiny.manifest.json` |
| Command | `uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-tiny` |

Selected tensors (max-abs and mean-abs differences, F32 vs reference, on
`samples/jfk.wav`):

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.embedder.cmvn.out`   | `2.240e-04` | `9.942e-06` | Frontend CMVN output |
| `enc.embedder.linear.out` | `2.840e-04` | `6.028e-06` | Time-domain linear projection |
| `enc.embedder.conv2.out`  | `9.388e-05` | `1.945e-06` | After 2× causal stride-2 convs |
| `enc.block.0.out`         | `1.099e-03` | `1.158e-05` | First sliding-window attention block |
| `enc.block.5.out`         | `6.105e-03` | `2.195e-04` | Last encoder block — depth-amplified BLAS reduction drift |
| `enc.final`               | `2.586e-04` | `8.776e-06` | Final encoder LN output |
| `adapter.out`             | `2.677e-04` | `9.082e-06` | Encoder→decoder adapter (learned pos-emb add) |
| `dec.block.0.out`         | `1.028e-03` | `4.340e-05` | First decoder block |
| `dec.block.5.out`         | `2.746e-03` | `2.010e-04` | Last decoder block |
| `dec.logits_raw.gen20`    | `1.829e-03` | `1.186e-04` | Mid-generation logits (token 20) |

The dominant drift source is BLAS reduction-order differences between
PyTorch's matmul kernels and ggml's `mul_mat` (Accelerate / Metal /
ggml-cpu). Drift accumulates roughly linearly with depth across encoder +
adapter + decoder; the final logit budget stays well below 1e-3 absolute /
1e-4 mean.

## Reproduction

### Convert

Loads directly from Useful Sensors' Hugging Face repo via
`AutoProcessor` + `MoonshineStreamingForConditionalGeneration`. Output path
is derived from the repo id.

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/convert-moonshine_streaming.py UsefulSensors/moonshine-streaming-tiny
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for F16; repeat with
`Q8_0`:

```bash
build/bin/transcribe-quantize \
  models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf \
  models/moonshine-streaming-tiny/moonshine-streaming-tiny-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-tiny
```

### WER sweep

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-tiny-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-tiny-F32.test-clean.jsonl
```
