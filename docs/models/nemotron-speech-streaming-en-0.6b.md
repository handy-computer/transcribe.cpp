# Nemotron Speech Streaming EN 0.6B

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  73 ms (151×) |  73 ms (151×) |
| Metal   | dots (35.3s) | 224 ms (158×) | 221 ms (160×) |
| CPU     | jfk (11.0s)  |  329 ms (33×) |  330 ms (33×) |
| CPU     | dots (35.3s) |  1.12 s (31×) |  1.12 s (31×) |

macOS 26.4.1, transcribe.cpp `12f1076`.

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
