# SAVA

<!-- catalog:intro -->
Upstream: [`inworld/sava`](https://github.com/inworld-ai/sava) at [`1beae7c`](https://github.com/inworld-ai/sava/commit/1beae7c).

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
Licensed Apache-2.0. Ported from upstream commit [`1beae7c`](https://github.com/inworld-ai/sava/commit/1beae7c), pinned 2026-11-21. Validated against the inworld-sava 1.0.0 (torch 2.14.1, CPU, fp32) on inworld/sava@1beae7c reference at transcribe.cpp commit [`3465deb3`](https://github.com/handy-computer/transcribe.cpp/tree/3465deb3) on 2026-11-25.
<!-- /catalog -->

## How to use it

**Live audio.** Feed 16 kHz audio to one TURN stream in chunks of any size.
Each call returns:

- the turn score of every 640-sample frame it completed;
- the speech score of every frame the model has judged. A frame is judged
  3 frames (120 ms) after its audio.

The caller decides when a turn ends. Inworld's reference rule:

1. A turn opens on the first frame with speech `>= 0.5`, 800 ms earlier.
2. While it is open, every silent frame checks the turn score of the step
   that judged it (`>= 0.9` closes the turn).
3. 1.6 s of silence closes it anyway.

`examples/turn_live` (`transcribe-turn-live --turn sava-Q8_0.gguf in.wav`)
runs that rule in 20 ms chunks.

**A whole recording.** `transcribe_turn_run` scores the clip as one stream
and returns:

- `p` and `ended`: the last frame's turn score, and `p >= 0.9`;
- every frame's turn and speech scores.

```bash
build/bin/transcribe-cli -m sava-Q8_0.gguf samples/sava-speech.wav
# turn: p=0.998900 ended=1 threshold=0.90
```

**Speech only.** The VAD role (`transcribe_vad_*`) returns the same speech
scores (lookahead 3 frames, threshold 0.5). `transcribe_vad_run` cuts them
into segments with Silero's rule on 40 ms frames.

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
(`transcribe_turn_run`'s `p` on the whole clip, scored as one stream);
the clip is called complete when `p > 0.5`, Inworld's benchmark rule.
The API's default cut-off is 0.9, Inworld's live rule. False positives
(incomplete clips called complete) and false negatives (the reverse) are
shares of all clips, as Pipecat publishes them. Inworld publishes 93.35%
and an AUC of 0.97946 for SAVA (fp32, A100); it publishes no per-clip
scores, so there is no per-clip agreement column.
<!-- /catalog -->

<!-- catalog:turn -->
| GGUF | Accuracy (95% CI) | False positives | False negatives |     AUC |
| --- | ---: | ---: | ---: | ---: |
| F32  | 93.36% (93.08-93.64) | 3.58% | 3.06% | 0.97946 |
| F16  | 93.36% (93.08-93.64) | 3.58% | 3.06% | 0.97946 |
| Q8_0 | 93.36% (93.08-93.63) | 3.60% | 3.03% | 0.97935 |

Measured at transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

At the API's cut-off (0.9) and at the best threshold:

| GGUF | Accuracy (0.9) | Best accuracy (threshold) |
| --- | ---: | ---: |
| F32  | 88.29% | 93.51% (0.45) |
| F16  | 88.29% | 93.51% (0.45) |
| Q8_0 | 88.23% | 93.48% (0.45) |

Inworld's published figures on the same set: SAVA (fp32, A100) 93.35% at
0.5, AUC 0.97946, best 93.51% (0.45); Smart Turn v3.2 93.71%, AUC 0.98250,
best 93.81% (0.36).

Details:

- **F32 vs Inworld's published SAVA:** the AUC matches to 1e-7. On real
  recordings accuracy is 95.84% (Inworld: 95.86%); on synthetic ones 92.79%
  (Inworld: 92.78%).
- **F16 vs F32:** the same decision on every clip at 0.5 and at 0.9. The
  clip score differs by at most 4.2e-6.
- **Q8_0 vs F32:** the same decision on 99.76% of clips at 0.5 (76 flips)
  and 99.60% at 0.9 (126 flips). The clip score differs by 2.8e-3 on average
  (p99 3.5e-2).
- **Rows:** `reports/turn/sava-{F32,F16,Q8_0}.smart-turn-data-v3.2-test.jsonl`
  from `scripts/turn/eval_cpp.py`, scored by `scripts/turn/score.py`. The
  runs' `p_fp32` / `p_int8` columns are Smart Turn's.

On the 8 golden recordings (`tests/golden/sava/`):

- **F32 vs the reference:** within 4e-5 in probability, with the same
  decisions.
- **F16 vs F32:** within 1.5e-6 per frame, with the same decisions on all 8.
  F16 also passes the F32 tolerance gate (`validate.py` with
  `--gguf sava-F16.gguf`).
- **Q8_0 vs F32:** per-frame scores differ by up to 0.065 (mean 1e-3 to
  4e-3 on speech clips). Inworld's turn rule applied to the scores finds the
  same turns on 7 of the 8. On the 5 min
  `dots-full` 6 of 59 turns change: Q8_0 finds 57, with turns merged at
  59.4 s and 104.1 s, ends moved at 101.8 s, 145.7 s and 174.4 s, and one
  start moved by a frame. Each change is a score within 0.004 of the 0.9
  cut-off, or a speech score within 0.03 of 0.5.

## Performance

The catalog's cells, CPU, on two machines: an Apple M1 (Mac mini), standing
in for the Apple M4 reference machine, which was unreachable, and an AMD
Ryzen 7 PRO 4750U:

- `jfk-8s`: `transcribe_turn_run` on the first 8 s of `samples/jfk.wav`
  (200 frames).
- `jfk-40ms`: one streamed frame, the mean time of a 640-sample
  `transcribe_turn_stream_feed` over the whole clip.
- `jfk-40ms-x8`: one `transcribe_turn_stream_feed_batch` tick for 8 users
  (a 640-sample frame each, so 0.32 s of audio per tick).
- `-1t` runs on one thread; without it, the library's default thread count
  (all 8 cores of the M1, 16 hardware threads of the Ryzen).

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

### Per frame, 4 threads (M1)

The M1, idle machine, with 4 threads and per frame. Milliseconds of
compute per 40 ms frame:

| File | Whole clip (`jfk.wav`), 1 thread | Whole clip, 4 threads | Stream (640-sample feeds), 1 thread | Stream, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0.90 | 0.38 | 4.6 | 3.3 |
| F16 | 1.81 | 0.59 | 7.2 | 2.9 |
| Q8_0 | 0.70 | 0.28 | 3.0 | 1.5 |

Several live users on one model: `transcribe_turn_stream_feed_batch` steps
one frame of every user in one compute, reading the weights once. Times are
per tick, one 640-sample frame for each of 8 users:

| File | 8 users, batched, 1 thread | 8 users, batched, 4 threads | 8 users fed one by one, 1 thread |
| --- | ---: | ---: | ---: |
| F32 | 10.1 ms (1.27 per user) | 6.3 ms (0.79 per user) | about 37 ms |
| F16 | 17.2 ms (2.15 per user) | 6.9 ms (0.86 per user) | about 58 ms |
| Q8_0 | 8.1 ms (1.01 per user) | 4.9 ms (0.61 per user) | about 24 ms |

At 1 thread, one core serves about 30 users in real time with F32 and about
40 with Q8_0. Each user's scores are bit-identical to feeding that session
alone.

On a shared machine, prefer 1 thread per compute. With other jobs running,
multi-threaded steps can be far slower than single-threaded ones.

**F16 and Q8_0 are not faster than F32 everywhere.** On the M1 (ARM), F16
runs 1.6-2.0x slower than F32 on one thread: ggml multiplies F16 weights by
fp32 activations without its fast tiled kernel on ARM. On the Ryzen (x86),
F16 and Q8_0 are both about 10% slower than F32 for a whole clip on one
thread, while for a streamed frame F16 is faster than F32 and Q8_0 the
fastest. Q8_0 is the fastest streamed frame on both machines. Choose a file
by need:

- **Q8_0:** the smallest, and the fastest for live streams.
- **F16:** half the size of F32 with F32's results.
- **F32:** the reference.

A whole clip runs in blocks of up to 64 frames. A streamed frame runs a
4-frame graph: the padding makes a stream bit-identical to a whole-clip run
for any chunking, at the cost of computing 4 columns for 1. This holds for
F32, F16 and Q8_0, batched or not. For comparison, upstream's PyTorch (1
thread) takes 0.37 ms per clip frame and 10.5 ms per streamed frame on the
same machine.

`AUTO` selects the CPU, the validated path.

A BF16 file (`convert-sava.py --outtype bf16`) is not published:

- Its storage is lossless, but ggml rounds a BF16 matmul's activations to
  bf16. It moves 5 of the 59 `dots-full` turns and flips 1 of 3,000 Pipecat
  clips at 0.5 (3 at 0.9).
- Without native BF16 it is 15-22x slower than F32: 20.7 / 5.7 ms per clip
  frame and 82 / 22 ms per streamed frame (1 / 4 threads, M1).
