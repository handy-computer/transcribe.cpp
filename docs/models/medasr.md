# MedASR

Google's [`google/medasr`](https://huggingface.co/google/medasr) ported to
transcribe.cpp. A Conformer encoder with a CTC head, 105M params,
English-only, fine-tuned on physician dictations.

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  163 ms (68×) |  174 ms (63×) |
| Vulkan  | dots (35.3s) |  481 ms (74×) |  495 ms (71×) |
| CPU     | jfk (11.0s)  |  543 ms (20×) |  488 ms (23×) |
| CPU     | dots (35.3s) | 1.84 s (19×)  | 1.63 s (22×)  |

Fedora 43, transcribe.cpp `79d139a`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models medasr \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name medasr-publication
```
