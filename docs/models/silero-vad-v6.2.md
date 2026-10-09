# Silero VAD v6.2

<!-- catalog:intro -->
Upstream: [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad) at [`5cd7945`](https://github.com/snakers4/silero-vad/commit/5cd7945).

Voice activity detection by the Silero Team. Not a transcription model:
the VAD role produces per-frame speech probabilities (offline or
streaming) and offline speech segments. CPU is the default backend for
reference fidelity. transcribe.cpp also loads whisper.cpp's
`ggml-silero-v5.1.2.bin` / `v6.2.0.bin`.
<!-- /catalog -->

API and examples: [VAD usage](../vad.md). Architecture and validation:
[family note](../porting/families/silero_vad.md).

<!-- catalog:pin -->
Licensed MIT. Ported from upstream commit [`5cd7945`](https://github.com/snakers4/silero-vad/commit/5cd7945), pinned 2026-10-09. Validated against the silero-vad 6.2.3 (TorchScript, CPU, F32) reference at transcribe.cpp commit [`24fda783`](https://github.com/handy-computer/transcribe.cpp/tree/24fda783) on 2026-10-09.
<!-- /catalog -->

Source weights: `silero_vad/data/silero_vad.jit` from `silero-vad==6.2.3` on
PyPI (SHA256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`).

## Downloads

<!-- catalog:downloads metric=false -->
| Quantization | Download | Size |
| --- | --- | ---: |
| F32          | [silero-vad-v6.2-F32.gguf](https://huggingface.co/handy-computer/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf) | 1 MB |
<!-- /catalog -->

F32 only: the model has 309K parameters and does not need quantization. To
convert it yourself:

```bash
uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py
```

The loader also accepts whisper.cpp's `ggml-silero-v5.1.2.bin` and
`ggml-silero-v6.2.0.bin` from
[ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad), so
whisper.cpp users can keep their files. Their mixed F16/F32 weights are
expanded to F32 at load.

## Performance

One CPU thread, `scripts/vad/bench.py`. The `-32ms` / `-128ms` / `-512ms`
rows are live streaming: the time of one `transcribe_vad_stream_feed` call
with that much audio, averaged over a pass of `love-loss.wav`. 32 ms is one
frame, so it is the per-frame latency. The other rows are
`transcribe_vad_run` over the whole clip.

<!-- catalog:perf machine=m4 dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `vad-publication-v1`: mean over 10 iterations after 3 warmup.

| Backend | Sample             |                 F32 |
| ------- | ------------------ | ------------------: |
| CPU     | love-loss-32ms     |  0.059 ms (540.27×) |
| CPU     | love-loss-128ms    | 0.070 ms (1816.82×) |
| CPU     | love-loss-512ms    | 0.262 ms (1953.56×) |
| CPU     | jfk (11.0s)        |   5.8 ms (1886.05×) |
| CPU     | love-loss (197.2s) | 100.7 ms (1957.63×) |

Apple M4: transcribe.cpp `11b76d35` on 2026-10-09.
<!-- /catalog -->

`AUTO` selects CPU for reference fidelity. Explicit GPU selection is available,
but ggml-metal's half-precision staging of F32 operands can change
probabilities.
