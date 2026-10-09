# Silero VAD v6.2

<!-- catalog:intro -->
Upstream: [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad) at [`5cd7945`](https://github.com/snakers4/silero-vad/commit/5cd7945).
Model: 309,633 parameters. Input: 16,000 Hz mono; 512 samples/frame (32 ms).

Voice activity detection by the Silero Team. Not a transcription model:
the VAD role produces speech probabilities, offline speech segments,
and streaming probabilities for live START/END events. CPU is the default
backend for reference fidelity.
<!-- /catalog -->

API and examples: [VAD usage](../vad.md). Architecture and validation:
[family note](../porting/families/silero_vad.md).

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`5cd7945`](https://github.com/snakers4/silero-vad/commit/5cd7945), pinned 2026-10-09. Validated against the silero-vad 6.2.3 (TorchScript, CPU, F32) reference at transcribe.cpp commit [`24fda783`](https://github.com/handy-computer/transcribe.cpp/tree/24fda783) on 2026-10-09.
Source artifact: `silero-vad==6.2.3`, [`silero_vad/data/silero_vad.jit`](https://pypi.org/project/silero-vad/6.2.3/). SHA256: `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`.
<!-- /catalog -->

## Downloads

<!-- catalog:downloads metric=false -->
| Quantization | Download | Size |
| --- | --- | ---: |
| F32          | `silero-vad-v6.2-F32.gguf` | 1 MB |

Canonical publication pending; filenames above are local artifacts, not download links.
<!-- /catalog -->

F32 only: this small detector does not need quantization. Until the canonical
GGUF is uploaded, convert it locally:

```bash
uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py
```

### Compatibility

<!-- catalog:prose field=compatibility -->
The loader also accepts whisper.cpp's `ggml-silero-v5.1.2.bin` and
`ggml-silero-v6.2.0.bin`, available from
[ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad).
These are third-party compatibility inputs, not this project's canonical
GGUF downloads. Their mixed F16/F32 weights are expanded to F32 at load.
<!-- /catalog -->

## Streaming latency

The headline is single-frame feed-call latency. Larger frame-aligned chunks
show amortized processing cost, not a reduction in audio collection time.
These timings do not measure START/END detection delay, which also depends
on the iterator's silence and padding policy.

<!-- catalog:stream-perf machine=m4 -->
Streaming feed-call wall latency; the 32 ms chunk is the headline. 128 ms and 512 ms chunks show feed-size amortization, not independent per-frame latency.
Includes the Python ctypes/native API wall call; excludes model load and audio capture. One pass over the audio with preserved stream state. Warmup calls are excluded from the measured call statistics. Chunk duration is audio per feed, not the full clip duration.
Median and p95 are measured per feed call. Amortized median/frame is median/feed divided by frames/feed, not a separately measured single-frame latency. Mean feed-call latency remains in `total_ms` metadata.
Samples/frame: 512. Threads: 1. Warmup calls: 32. Source samples: `love-loss-32ms`, `love-loss-128ms`, `love-loss-512ms`.

| Backend | GGUF | Chunk  | Frames/feed | Median/feed |    p95/feed | Amortized median/frame | Calls |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu     | F32  | 32 ms  |           1 | 0.059125 ms | 0.064042 ms | 0.059125 ms | 6,161 |
| cpu     | F32  | 128 ms |           4 | 0.073916 ms |    0.077 ms | 0.018479 ms | 1,540 |
| cpu     | F32  | 512 ms |          16 | 0.265292 ms |   0.2775 ms | 0.0165807 ms |   385 |

Machine: `m4`.
Measured at transcribe.cpp `24fda783` on 2026-10-09, profile `vad-publication-v1`.
<!-- /catalog -->

`AUTO` selects CPU for reference fidelity. Explicit GPU selection is available,
but ggml-metal's half-precision staging of F32 operands can change probabilities;
GPU timings are not part of this CPU publication profile.
