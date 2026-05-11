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

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  812 ms (14×) |  813 ms (14×) |
| Vulkan  | dots (35.3s) |  2.93 s (12×) |  2.98 s (12×) |
| CPU     | jfk (11.0s)  |  1.39 s (8×)  |  1.22 s (9×)  |
| CPU     | dots (35.3s) |  5.21 s (7×)  |  4.76 s (7×)  |

Fedora 43, transcribe.cpp `12f1076`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

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
