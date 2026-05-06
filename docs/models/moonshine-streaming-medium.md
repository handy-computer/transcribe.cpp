# Moonshine Streaming Medium

Useful Sensors' [`UsefulSensors/moonshine-streaming-medium`](https://huggingface.co/UsefulSensors/moonshine-streaming-medium)
ported to transcribe.cpp. A 245M-parameter encoder-decoder English ASR model
designed for streaming use (ergodic encoder + sliding-window attention,
50 Hz time-domain frontend). Same family as the tiny and small variants;
deepest of the three (14 / 14 layers) and widest hidden dims (encoder 768 /
decoder 640).

## What it's for

Offline English speech-to-text. The model takes a 16 kHz mono WAV and produces
a transcript. It does not translate, has no multilingual capability, and does
not emit timestamps. 

See Useful Sensors' [model card](https://huggingface.co/UsefulSensors/moonshine-streaming-medium)
for training data, intended use, and upstream evaluation methodology.

Licensed MIT. Ported from upstream commit
[`57b8436`](https://huggingface.co/UsefulSensors/moonshine-streaming-medium/commit/57b843633a8c183cadf6699ffa761377a933a866),
pinned 2026-05-06.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32  | [moonshine-streaming-medium-F32.gguf](https://huggingface.co/handy-computer/moonshine-streaming-medium-gguf/resolve/main/moonshine-streaming-medium-F32.gguf)   | 1015 MB | 2.16% |
| F16  | [moonshine-streaming-medium-F16.gguf](https://huggingface.co/handy-computer/moonshine-streaming-medium-gguf/resolve/main/moonshine-streaming-medium-F16.gguf)   |  509 MB | 2.16% |
| Q8_0 | [moonshine-streaming-medium-Q8_0.gguf](https://huggingface.co/handy-computer/moonshine-streaming-medium-gguf/resolve/main/moonshine-streaming-medium-Q8_0.gguf) |  282 MB | 2.16% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy decoding (`num_beams=1`, `do_sample=False`). F32 reference
baseline: 2.16%. Quants are numerically indistinguishable from F32 on this
manifest. Useful Sensors' self-reported number on this split is 2.08% from
the Open ASR Leaderboard table; the +0.08pp residual matches the same
scoring / text-normalization difference seen across the tiny and small
variants (where the tiny cross-check against the HF Transformers reference
on the same manifest landed within 0.01pp of our port), and is not a
numerical drift in the port.

Q6_K / Q5_K_M / Q4_K_M GGUFs are not currently shipped for this variant.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-streaming-medium/moonshine-streaming-medium-Q8_0.gguf \
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
| Metal   | jfk (11.0s)  |  124 ms (89×) |
| Metal   | dots (35.3s) |  952 ms (37×) |
| CPU     | jfk (11.0s)  |  281 ms (39×) |
| CPU     | dots (35.3s) | 1.11 s (32×)  |

macOS 26.4.1, transcribe.cpp `0d312ce`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |
| ------- | ------------ | ------------: |
| Vulkan  | jfk (11.0s)  |  570 ms (19×) |
| Vulkan  | dots (35.3s) | 4.01 s (9×)   |
| CPU     | jfk (11.0s)  | 1.07 s (10×)  |
| CPU     | dots (35.3s) | 6.50 s (5×)   |

Fedora 43, transcribe.cpp `f243f34`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models moonshine-streaming-medium \
  --quants q8_0 \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
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
| Reference | HF Transformers v5.7.0, `UsefulSensors/moonshine-streaming-medium` |
| Dump script | `scripts/dump_reference_moonshine_streaming_transformers.py` |
| Manifest | `tests/golden/moonshine_streaming/moonshine-streaming-medium.manifest.json` |
| Command | `uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-medium` |

Tolerances are recorded at family scope in
`tests/tolerances/moonshine_streaming.json`. The dominant drift source is
BLAS reduction-order differences between PyTorch's matmul kernels and
ggml's `mul_mat`. Drift accumulates roughly linearly with depth across the
14-layer encoder, the adapter, and the 14-layer decoder; the family
tolerances were widened from the tiny baseline using
`max(1.5 × observed, prior, 1e-6)` to absorb the deeper-stack accumulation
without inflating the budget where it isn't needed. The final logit budget
remains well below 1e-3 absolute / 1e-4 mean.

Like the small variant, medium uses **non-square encoder attention**
(encoder residual dim 768, attention dim 640). The C++ port carries both
shapes through the encoder block.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/convert-moonshine_streaming.py UsefulSensors/moonshine-streaming-medium
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/moonshine-streaming-medium/moonshine-streaming-medium-F32.gguf \
  models/moonshine-streaming-medium/moonshine-streaming-medium-F16.gguf \
  --quant F16
```

### Validate

```bash
uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-medium
```

### WER sweep

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-streaming-medium/moonshine-streaming-medium-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-medium-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-medium-F32.test-clean.jsonl
```
