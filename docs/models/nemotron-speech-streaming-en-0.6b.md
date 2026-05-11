# Nemotron Speech Streaming EN 0.6B

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |    94 ms (117×) |    95 ms (116×) |
| Metal   | dots (35.3s) |   290 ms (122×) |   295 ms (120×) |
| CPU     | jfk (11.0s)  |   347 ms (32×)  |   346 ms (32×)  |
| CPU     | dots (35.3s) |  1.18 s (30×)   |  1.16 s (30×)   |

macOS 26.4.1, transcribe.cpp `c2e7bf9`.

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
