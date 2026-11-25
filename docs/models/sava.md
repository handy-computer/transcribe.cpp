# SAVA

<!-- catalog:intro -->
Upstream: [`inworld/sava`](https://huggingface.co/inworld/sava) at [`1beae7c`](https://huggingface.co/inworld/sava/commit/1beae7c).

Streaming end-of-turn and voice activity detection by Inworld: every 40 ms
it gives the probability that the speaker's turn has ended and, 120 ms
later, the probability that the frame held speech. Not a transcription
model: it serves the TURN role (live stream or whole clip) and the VAD
role. Any language. CPU is the default backend.
<!-- /catalog -->

API and examples: [turn detection](../turn.md), [voice activity detection](../vad.md).
Architecture and validation: [family note](../porting/families/sava.md).

Source weights: `sava.safetensors` (bf16) from
[`inworld/sava`](https://huggingface.co/inworld/sava) at
`1beae7c1dd0df6b0f92030644fc905eb087dc6a0` (SHA256
`aa610ee9b5d3d1be96f70581a8a3affe7cf20d06816b4ada1efdd8ab6274be34`).
Upstream code: [`inworld-ai/sava`](https://github.com/inworld-ai/sava)
(`inworld-sava` 1.0.0).

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`1beae7c`](https://huggingface.co/inworld/sava/commit/1beae7c), pinned 2026-11-21. Validated against the inworld-sava 1.0.0 (PyTorch, fp32) reference at transcribe.cpp commit [`3465deb3`](https://github.com/handy-computer/transcribe.cpp/tree/3465deb3) on 2026-11-25.
<!-- /catalog -->

## How to use it

Live audio: feed a TURN stream chunks of any size; each feed returns the turn
score of every 640-sample frame it completed and the speech score of every
frame the model judged (3 frames, 120 ms, behind). When a turn ends is the
caller's choice: [turn detection](../turn.md#stream-sava) gives Inworld's
rule, and `examples/turn_live` (`transcribe-turn-live --turn
sava-Q8_0.gguf in.wav`) runs it.

A whole recording: `transcribe_turn_run` scores the clip as one stream and
returns `p` (the last frame's turn score), `ended` (`p >= threshold`,
default 0.9) and every frame's turn and speech scores.

```bash
build/bin/transcribe-cli -m sava-Q8_0.gguf samples/sava-speech.wav
# turn: p=0.998900 ended=1 threshold=0.90
```

Speech only: the VAD role (`transcribe_vad_*`) returns the same speech
scores; `transcribe_vad_run` cuts them into segments with Silero's rule on
40 ms frames.

## Downloads

<!-- catalog:downloads -->
| Quantization | Download |   Size | Accuracy (Pipecat Smart Turn v3.2 test) |
| --- | --- | ---: | ---: |
| F32          | [sava-F32.gguf](https://huggingface.co/handy-computer/sava-gguf/resolve/main/sava-F32.gguf) | 103 MB | 93.36% |
| F16          | [sava-F16.gguf](https://huggingface.co/handy-computer/sava-gguf/resolve/main/sava-F16.gguf) |  52 MB | 93.36% |
| Q8_0         | [sava-Q8_0.gguf](https://huggingface.co/handy-computer/sava-gguf/resolve/main/sava-Q8_0.gguf) |  28 MB | 93.36% |
<!-- /catalog -->

The release weights are bf16, and upstream runs them in fp32:

- **F32** holds them exactly.
- **F16** holds them exactly except 1,065 of 25.5M tiny values (each off by
  at most 3e-8). It computes like F32, because ggml keeps the activations of
  an F16 matmul in fp32. Its scores match F32 to 4e-6.
- **Q8_0** is the smallest file, and the fastest for a streamed frame.

To convert and quantize it yourself:

```bash
uv run --project scripts/envs/sava scripts/convert-sava.py
build/bin/transcribe-quantize models/sava/sava-F32.gguf models/sava/sava-F16.gguf --quant F16
build/bin/transcribe-quantize models/sava/sava-F32.gguf models/sava/sava-Q8_0.gguf --quant Q8_0
```

## Accuracy

<!-- catalog:recipe -->
Accuracy on the full Pipecat Smart Turn v3.2 test split (31,527 clips), a clip called complete when p > 0.5, scored on cpu. Measured at transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
A clip's score is the turn score of its last 40 ms frame
(`transcribe_turn_run`'s `p`, the clip scored as one stream); the clip
is called complete when `p > 0.5`, Inworld's benchmark rule. The API's
default cut-off is 0.9, Inworld's live rule. At 0.9, F32 and F16 make
the reference's decision on every clip; Q8_0 on 31,401. False positives
(incomplete clips called complete) and false negatives (the reverse) are
shares of all clips, as Pipecat publishes them. Inworld publishes 93.35% and an
AUC of 0.97946 (fp32, A100) but no per-clip scores; the reference is
upstream's PyTorch package (inworld-sava 1.0.0, CPU, fp32) on the same
clips: 93.36%, AUC 0.97946.
<!-- /catalog -->

<!-- catalog:turn reference="the PyTorch reference (inworld-sava 1.0.0, fp32)" -->
| GGUF | Accuracy (95% CI) | False positives | False negatives |     AUC | Same decision as the PyTorch reference (inworld-sava 1.0.0, fp32) | Max abs p difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32  | 93.36% (93.08-93.64) | 3.58% | 3.06% | 0.97946 | 31,527 / 31,527 | 1.1e-05 |
| F16  | 93.36% (93.08-93.64) | 3.58% | 3.06% | 0.97946 | 31,527 / 31,527 | 1.2e-05 |
| Q8_0 | 93.36% (93.08-93.63) | 3.60% | 3.03% | 0.97935 | 31,451 / 31,527 | 0.17 |

Measured at transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

At the API's cut-off (0.9) and at the best threshold:

| GGUF | Accuracy (0.9) | Best accuracy (threshold) |
| --- | ---: | ---: |
| F32  | 88.29% | 93.51% (0.45) |
| F16  | 88.29% | 93.51% (0.45) |
| Q8_0 | 88.23% | 93.48% (0.45) |

Inworld publishes, on the same set, a best of 93.51% (0.45) for SAVA, and
93.71% (AUC 0.98250, best 93.81% at 0.36) for Smart Turn v3.2.

- **F32 vs Inworld's published SAVA:** the AUC matches to 1e-7; real
  recordings 95.84% (Inworld: 95.86%), synthetic ones 92.79% (92.78%).
- **At 0.9:** F32 and F16 make the PyTorch reference's decision on every
  clip; Q8_0 on 31,401.
- **F16 vs F32:** the same decision on every clip at 0.5 and 0.9; p within
  4.2e-6.
- **Q8_0 vs F32:** 76 flips at 0.5 and 126 at 0.9 (99.76% / 99.60% agree);
  p differs by 2.8e-3 on average (p99 3.5e-2).
- **Rows:** `reports/turn/sava-{F32,F16,Q8_0}.smart-turn-data-v3.2-test.jsonl`
  (`scripts/turn/eval_cpp.py`), scored by `scripts/turn/score.py
  --ref-jsonl` against `reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl`
  (`scripts/turn/run_reference_sava_torch.py`). Their `p_fp32` / `p_int8`
  columns are Smart Turn's.

On the 8 golden recordings (`tests/golden/sava/`), F32 is within 4e-5 of the
reference and F16 within 1.5e-6 of F32 (it passes the F32 tolerance gate),
both with the same decisions. Q8_0 differs by up to 0.065 per frame (mean
1e-3 to 4e-3 on speech). Inworld's turn rule on its scores finds the same
turns on 7 of the 8; on the 5 min `dots-full` 6 of 59 change (2 merges, 3
moved ends, 1 start moved by a frame), each at a score within 0.004 of 0.9
or a speech score within 0.03 of 0.5.

## Performance

CPU on two machines: an Apple M1 (Mac mini), standing in for the
unreachable Apple M4 reference machine, and an AMD Ryzen 7 PRO 4750U.

- `jfk-8s`: `transcribe_turn_run` on the first 8 s of `samples/jfk.wav`
  (200 frames).
- `jfk-40ms`: one streamed frame, the mean of 640-sample
  `transcribe_turn_stream_feed` calls over the clip.
- `jfk-40ms-x8`: one `transcribe_turn_stream_feed_batch` tick for 8 users
  (a 640-sample frame each, 0.32 s of audio).
- `-1t`: one thread; otherwise the library's default (the usable CPUs,
  capped at 8: 8 on both machines).

### Apple M1 (proxy for the M4)

<!-- catalog:perf machine=m1 dp_ms=1 -->
Compute latency (front end + model), speedup over realtime in parentheses; profile `turn-publication-v1`: mean over 20 iterations after 3 warmup.

| Backend | Sample           |               F32 |               F16 |              Q8_0 |
| ------- | ---------------- | ----------------: | ----------------: | ----------------: |
| CPU     | jfk-40ms         |   3.2 ms (12.43×) |   2.8 ms (14.06×) |   2.1 ms (19.13×) |
| CPU     | jfk-40ms-1t      |    4.4 ms (9.09×) |    7.1 ms (5.63×) |   3.1 ms (12.97×) |
| CPU     | jfk-40ms-x8      |   7.4 ms (43.36×) |   8.0 ms (40.15×) |   6.5 ms (49.26×) |
| CPU     | jfk-40ms-x8-1t   |  10.0 ms (32.10×) |  16.5 ms (19.36×) |   8.2 ms (39.18×) |
| CPU     | jfk-8s (8.0s)    | 73.9 ms (108.21×) | 108.5 ms (73.74×) | 78.6 ms (101.83×) |
| CPU     | jfk-8s-1t (8.0s) | 177.8 ms (44.99×) | 354.7 ms (22.55×) | 139.6 ms (57.30×) |

Apple M1 (proxy for the Apple M4 reference machine, which was unreachable): transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u dp_ms=1 -->
Compute latency (front end + model), speedup over realtime in parentheses; profile `turn-publication-v1`: mean over 20 iterations after 3 warmup.

| Backend | Sample           |               F32 |               F16 |              Q8_0 |
| ------- | ---------------- | ----------------: | ----------------: | ----------------: |
| CPU     | jfk-40ms         |    5.9 ms (6.73×) |    4.4 ms (9.03×) |   3.7 ms (10.80×) |
| CPU     | jfk-40ms-1t      |    9.8 ms (4.08×) |    8.4 ms (4.75×) |    6.9 ms (5.77×) |
| CPU     | jfk-40ms-x8      |  13.6 ms (23.46×) |  12.1 ms (26.38×) |  11.8 ms (27.21×) |
| CPU     | jfk-40ms-x8-1t   |  20.1 ms (15.94×) |  19.1 ms (16.75×) |  18.0 ms (17.78×) |
| CPU     | jfk-8s (8.0s)    | 107.5 ms (74.41×) | 106.2 ms (75.36×) | 109.1 ms (73.34×) |
| CPU     | jfk-8s-1t (8.0s) | 221.6 ms (36.10×) | 245.7 ms (32.57×) | 248.0 ms (32.26×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `8415e8e2` on 2026-11-24.
<!-- /catalog -->

The batched feed reads the weights once for all users: fed one by one, 8
users take about 37 ms per tick (F32, M1, 1 thread). Each user's scores are
bit-identical to feeding that session alone. On a shared machine prefer 1
thread per compute: with other jobs running, multi-threaded steps can be far
slower.

**F16 and Q8_0 are not faster than F32 everywhere.** On ARM, F16 runs
1.6-2.0x slower than F32 on one thread (ggml has no tiled kernel for F16
weights x fp32 activations there). On the Ryzen, F16 and Q8_0 are about 10%
slower for a whole clip on one thread, while F16 is faster for a streamed
frame. Q8_0 is the fastest streamed frame on both: the pick for live
streams; F16 is half of F32's size with F32's results.

A streamed frame runs a 4-frame graph, which makes a stream bit-identical to
a whole-clip run for any chunking, batched or not. Upstream's PyTorch (1
thread, M1) takes 0.37 ms per clip frame and 10.5 ms per streamed frame.

`AUTO` selects the CPU, the validated path.

A BF16 file (`convert-sava.py --outtype bf16`) is not published: its storage
is lossless, but ggml rounds a BF16 matmul's activations to bf16 (5 of the
59 `dots-full` turns move; 1 of 3,000 Pipecat clips flips at 0.5, 3 at
0.9), and without native BF16 it is 15-22x slower than F32.
